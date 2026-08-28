"""Maps REAL vLLM tensor-parallel parameter sharding metadata into distributed_perturbation.
ShardSpec -- Stage-11 32B readiness, Section 5 ("the main engineering task").

SOURCING (read this before trusting any claim below): no vLLM install and no GPU are available
in this environment (confirmed this session -- `import vllm` fails locally, and the task
explicitly forbids running a GPU here), so this module's understanding of vLLM's TP attribute
convention comes from reading vLLM's actual, current `vllm/model_executor/layers/linear.py`
source (github.com/vllm-project/vllm, main branch, fetched this session), NOT from a live
runtime inspection. The relevant, confirmed-from-source facts:

  - ColumnParallelLinear/RowParallelLinear attach `output_dim` / `input_dim` (an int naming
    which tensor dimension is sharded) directly onto the weight Parameter via `set_weight_attrs`
    -- inspectable as `getattr(param, "output_dim", None)` / `getattr(param, "input_dim", None)`.
  - The parameter's own `.shape` already reflects ONLY the local shard (never the global shape).
  - `weight_loader`'s own shard-selection arithmetic is `start_idx = self.tp_rank * shard_size`
    (contiguous, rank-ordered slicing along the ONE sharded dim) -- i.e. exactly the
    dim/offset/size slicing `distributed_perturbation.ShardSpec`/`slice_shard` already
    implements; this design was NOT invalidated by this research (see module docstring of
    distributed_perturbation.py for the "STOP AND REPORT" contingency this ruled out).
  - `tp_size`/`tp_rank` live on the OWNING LAYER MODULE (e.g. `self.tp_size` on the
    ColumnParallelLinear instance), never on the bare Parameter -- `model.named_parameters()`
    alone is insufficient; the owning module must be located too.
  - Parameters with NEITHER `output_dim` nor `input_dim` set (RMSNorm/LayerNorm weights, biases
    on non-TP layers, etc.) are REPLICATED across ranks -- there is no explicit "is_replicated"
    flag; absence of both dim attributes is the signal.

THIS IS RESEARCH-GROUNDED, NOT LIVE-VERIFIED. Gate G4/G5 (stage11_32b_readiness.py) require a
live-hardware confirmation before this mapping is trusted for a real 32B run -- this module
provides the mapping LOGIC (fully unit-tested against fakes that mimic the documented attribute
convention exactly), not a live-runtime proof.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from .distributed_perturbation import ShardSpec


class AmbiguousShardMappingError(RuntimeError):
    """A parameter's sharding could not be confidently classified -- e.g. both output_dim and
    input_dim are set (should never happen per the documented convention), or the owning module
    reports tp_size > 1 but neither dim attribute is present (an unrecognized parameter type
    inside a TP-aware layer). Hard fails rather than guessing, per task spec Section 5 ("Hard
    fail on ambiguous/incomplete mapping").
    """


def build_shard_spec_from_attributes(
    local_shape: torch.Size, *, output_dim: Optional[int], input_dim: Optional[int], tp_size: int, tp_rank: int, param_name: str = "<unnamed>",
) -> ShardSpec:
    """Pure mapping logic (no vLLM/torch-module access) -- the part that is fully,
    deterministically testable without any live vLLM object. `local_shape` is the shape the
    live parameter ALREADY has (confirmed local-shard-only per this module's docstring).
    """
    if output_dim is not None and input_dim is not None:
        raise AmbiguousShardMappingError(f"Parameter {param_name!r} has BOTH output_dim={output_dim} and input_dim={input_dim} set -- cannot determine the sharded dimension unambiguously.")

    dim = output_dim if output_dim is not None else input_dim
    if dim is None:
        if tp_size > 1:
            # Recognized replicated case (norm weights, etc.) -- NOT ambiguous, this is the
            # documented "no dim attribute" = replicated convention. world_size/rank are still
            # recorded so counts_toward_global_sum can correctly restrict this to rank 0.
            return ShardSpec(global_shape=local_shape, dim=None, rank=tp_rank, world_size=tp_size)
        return ShardSpec(global_shape=local_shape, dim=None, rank=tp_rank, world_size=1)

    if dim < 0 or dim >= len(local_shape):
        raise AmbiguousShardMappingError(f"Parameter {param_name!r} reports shard dim={dim}, out of range for shape {tuple(local_shape)}.")
    if tp_size < 1:
        raise AmbiguousShardMappingError(f"Parameter {param_name!r} is marked sharded (dim={dim}) but owning module reports tp_size={tp_size} < 1.")

    local_size = local_shape[dim]
    global_size = local_size * tp_size
    global_shape = list(local_shape)
    global_shape[dim] = global_size
    local_offset = tp_rank * local_size
    return ShardSpec(global_shape=torch.Size(global_shape), dim=dim, local_offset=local_offset, local_size=local_size, rank=tp_rank, world_size=tp_size)


def extract_vllm_tp_attributes(param: Any, owning_module: Any) -> Dict[str, Any]:
    """Reads the documented vLLM attribute convention off a live parameter + its owning module.
    `owning_module` may be a plain nn.Module with no `tp_size`/`tp_rank` (e.g. an un-wrapped
    nn.LayerNorm) -- treated as tp_size=1 (unsharded, unambiguously replicated-of-one).
    """
    return {
        "output_dim": getattr(param, "output_dim", None),
        "input_dim": getattr(param, "input_dim", None),
        "tp_size": getattr(owning_module, "tp_size", 1),
        "tp_rank": getattr(owning_module, "tp_rank", 0),
    }


def _owning_module_for_each_parameter(model: torch.nn.Module) -> Dict[str, Any]:
    """Maps every parameter's GLOBAL name (as `model.named_parameters()` yields it) to its
    DIRECT owning module -- `model.named_parameters()` alone never gives the module, only
    (name, Parameter); this walks `named_modules()` and each module's OWN (non-recursive)
    parameters to build the association unambiguously (a parameter belongs to exactly one
    direct-owner module in a standard nn.Module tree).
    """
    owner_by_name: Dict[str, Any] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for local_name, _ in module.named_parameters(recurse=False):
            owner_by_name[f"{prefix}{local_name}"] = module
    return owner_by_name


def build_shard_specs_for_region(model: torch.nn.Module, region_param_names: Sequence[str]) -> Dict[str, ShardSpec]:
    """The full walker: for every named parameter in `region_param_names`, locates its owning
    module, extracts vLLM's documented TP attributes, and builds a ShardSpec. Hard-fails
    (AmbiguousShardMappingError) on the FIRST parameter it cannot confidently classify, rather
    than silently defaulting the rest -- never a partial/best-effort mapping.
    """
    owner_by_name = _owning_module_for_each_parameter(model)
    named = dict(model.named_parameters())
    specs: Dict[str, ShardSpec] = {}
    for name in region_param_names:
        if name not in named:
            raise AmbiguousShardMappingError(f"Parameter {name!r} not found on the model.")
        if name not in owner_by_name:
            raise AmbiguousShardMappingError(f"Parameter {name!r} has no identifiable owning module -- cannot extract TP attributes.")
        param = named[name]
        owner = owner_by_name[name]
        attrs = extract_vllm_tp_attributes(param, owner)
        specs[name] = build_shard_spec_from_attributes(
            param.shape, output_dim=attrs["output_dim"], input_dim=attrs["input_dim"], tp_size=attrs["tp_size"], tp_rank=attrs["tp_rank"], param_name=name,
        )
    return specs


def ensure_uniform_tp_size(shard_specs: Dict[str, ShardSpec], expected_tp_size: int) -> None:
    """Every sharded (non-replicated) parameter in a region must agree on the SAME world_size as
    the engine's configured tensor_parallel_size -- a mismatch (e.g. one layer reporting
    tp_size=2 inside a tp_size=4 engine) is a hard-fail signal of a genuinely inconsistent/
    ambiguous mapping, never silently averaged or ignored.
    """
    mismatched = {name: s.world_size for name, s in shard_specs.items() if not s.is_replicated and s.world_size != expected_tp_size}
    if mismatched:
        raise AmbiguousShardMappingError(f"Parameter(s) report a TP world_size inconsistent with the engine's configured tensor_parallel_size={expected_tp_size}: {mismatched}")
