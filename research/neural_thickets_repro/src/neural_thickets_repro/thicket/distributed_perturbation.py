"""Tensor-parallel (TP)-aware extension of thicket.perturbation.apply_anatomical_relative_l2 --
NEW infrastructure for Stage-11 32B readiness, never a modification of the existing function.

CONFIRMED STRUCTURAL FACT (grounds this whole module -- not an assumption): every RPC dispatch
path in this project today is explicitly TP=1-only. run_global_visual_thicket_pilot.py's own
`_validate_collective_rpc_results` hard-fails with "this pilot is TP=1-only ... expects exactly
1" whenever `collective_rpc` returns more than one per-worker result, and
thicket.perturbation.apply_anatomical_relative_l2 computes theta/noise L2 norms via a purely
LOCAL for-loop over `model.named_parameters()` -- there is no cross-rank reduction anywhere in
the current codebase. TP>1 support is therefore genuinely NEW engineering, not a bug fix, and
3B/7B (which only ever call the ORIGINAL, untouched `apply_anatomical_relative_l2` through the
existing, unmodified `scoped_anatomical_perturbation.py` call sites) are structurally unaffected
by anything in this module: every function here is new, additively imported, never wired into
an existing 3B/7B call site.

THE SCIENTIFIC INVARIANT THIS MODULE PROTECTS (task spec Section 6): for anatomical region a,
    ||epsilon_a||_2 / ||theta_a||_2 = r
must hold over the FULL region, even when region a's parameter tensors are SHARDED across TP
ranks. Computing this ratio from any ONE rank's local shard alone would silently apply radius r
independently per-shard -- a different (and uncontrolled) global perturbation magnitude than the
frozen experiment design specifies. The fix: every rank computes its LOCAL contribution to
theta_sq_sum/noise_sq_sum (reusing the existing, unmodified, memory-bounded
`chunked_squared_l2_sum`), an ALL-REDUCE (SUM) combines these into the identical GLOBAL sum on
every rank, and only then is the rescale scalar `scale = r * global_theta_l2_norm /
global_noise_l2_norm` computed -- identical on every rank, applied to each rank's own local
noise/shard.

RNG / DIRECTION SEMANTICS (task spec Section 7): rather than inventing a new per-shard seed
derivation scheme (whose independence/collision properties would be unverified without live
distributed hardware), every rank generates the noise for the PARAMETER'S FULL (GLOBAL, unsharded)
shape using the EXACT SAME seed-to-tensor procedure `_generate_noise` already uses today
(`_generate_full_shape_noise` below is a direct, provably-equivalent generalization -- see its
own docstring and the accompanying equivalence test), then slices out only its own shard. This
guarantees the drawn Gaussian direction is, by construction, IDENTICAL to what a single TP=1
process would draw for that parameter and seed -- "one globally normalized Gaussian direction,
each shard receiving its part" (task spec Section 6), not a new independent per-shard stream.
The tradeoff (documented, deliberate, matching this project's established "correctness over
efficiency, never silently change direction/Gaussian semantics" posture -- see
scoped_anatomical_perturbation.py's own docstring on the un-chunked `_generate_noise` call):
every rank momentarily materializes the FULL parameter's noise tensor before discarding all but
its own slice. No new seed-derivation function was needed; documenting why the EXISTING
per-parameter seed derivation already satisfies distributed determinism (a pure function of
parameter name + direction seed, never of worker/rank identity) IS the Section-7 deliverable.

INTEGRATION STATUS (read before assuming this is wired into a live runner): `ShardSpec` and
`process_group` are accepted as EXPLICIT, INJECTED parameters, never auto-detected from a live
vLLM worker/parameter object. This project has no live GPU/vLLM TP session to verify against in
this milestone (task spec: "DO NOT RUN GPU LOCALLY"), and this module's own docstring pattern
(see `_generate_noise`'s neighboring "unverified without live GPU hardware" note in
scoped_anatomical_perturbation.py) is to say so explicitly rather than fabricate a vLLM API
surface. Wiring real vLLM TP shard metadata into `ShardSpec` construction is the next, live-GPU
-gated integration step (see stage11_32b_readiness.py's G4 gate), not something asserted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .memory_bounded_ops import DEFAULT_CHUNK_ELEMENTS, chunked_squared_l2_diff_sum, chunked_squared_l2_sum
from .perturbation import AnatomicalRelativeL2Record, DegenerateRegionError
from ..perturb_cpu import _generate_noise


# =================================================================================================
# Shard description + collective all-reduce (injectable, identity by default at world_size=1)
# =================================================================================================


@dataclass(frozen=True)
class ShardSpec:
    """Describes how ONE parameter tensor is distributed across TP ranks.

    `dim=None` means the parameter is REPLICATED (every rank holds the full tensor, e.g. norm
    weights under some TP schemes) -- its squared-L2 contribution must be counted by exactly ONE
    rank (by convention, rank 0) when summed into the global total, or it would be double
    (world_size-times-over) counted. `dim` set means the tensor is SHARDED along that dimension;
    `local_offset`/`local_size` give this rank's slice bounds along `dim` (both required together
    when `dim` is not None).
    """
    global_shape: torch.Size
    dim: Optional[int] = None
    local_offset: int = 0
    local_size: Optional[int] = None
    rank: int = 0
    world_size: int = 1

    @property
    def is_replicated(self) -> bool:
        return self.dim is None

    @property
    def counts_toward_global_sum(self) -> bool:
        """Replicated tensors count only on rank 0 (avoids world_size-times double-counting);
        sharded tensors count on every rank (each rank owns a disjoint slice).
        """
        return (not self.is_replicated) or self.rank == 0


def slice_shard(full_tensor: torch.Tensor, shard: ShardSpec) -> torch.Tensor:
    """Extracts this rank's local slice of `full_tensor` (already on the shard's target device)
    per `shard`'s dim/offset/size. Returns `full_tensor` unchanged when replicated.
    """
    if shard.is_replicated:
        return full_tensor
    if shard.local_size is None:
        raise ValueError("ShardSpec.local_size is required when dim is set.")
    index = [slice(None)] * full_tensor.dim()
    index[shard.dim] = slice(shard.local_offset, shard.local_offset + shard.local_size)
    return full_tensor[tuple(index)]


def identity_all_reduce_sum(value: float, process_group: Any = None) -> float:
    """The TP=1 / no-process-group default: returns `value` unchanged. Passing this (or leaving
    `all_reduce_sum` at its default) makes every function in this module byte-behaviorally
    identical, at world_size=1, to summing purely locally -- i.e. identical to the existing,
    untouched `apply_anatomical_relative_l2` (proven by test, not merely asserted).
    """
    return value


def torch_distributed_all_reduce_sum(value: float, process_group: Any) -> float:
    """Real collective reduction via `torch.distributed.all_reduce`, for callers that DO have a
    live process group. Not exercised by this project's CPU test suite (no distributed runtime
    available here) -- provided as the real implementation a live-GPU integration step wires in;
    `identity_all_reduce_sum` (or a fake, in tests) stands in for it everywhere this module is
    unit-tested.
    """
    import torch.distributed as dist

    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
    return float(tensor.item())


AllReduceSumFn = Callable[[float, Any], float]


# =================================================================================================
# Distributed noise generation -- provably equivalent to _generate_noise at world_size=1
# =================================================================================================


def _generate_full_shape_noise(global_shape: torch.Size, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    """Direct generalization of perturb_cpu._generate_noise: identical generator-seeding and
    `torch.randn` call, with shape/dtype/device passed explicitly instead of read off an
    already-local (possibly already-sharded) parameter tensor. At `global_shape == param.shape`
    (the unsharded/TP=1 case) this is PROVABLY bit-identical to `_generate_noise(param, seed)`
    (see tests/test_thicket_distributed_perturbation.py) -- same Generator, same seed, same randn
    call, only the shape's source differs.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randn(global_shape, dtype=dtype, device=device, generator=gen)


# =================================================================================================
# Global (distributed) relative-L2 perturbation
# =================================================================================================


@torch.no_grad()
def apply_anatomical_relative_l2_distributed(
    model: torch.nn.Module, region: str, region_param_names: Sequence[str], seed: int, r: float,
    shard_specs: Dict[str, ShardSpec], *, base_state: Optional[Dict[str, torch.Tensor]] = None,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS, all_reduce_sum: AllReduceSumFn = identity_all_reduce_sum,
    process_group: Any = None,
) -> AnatomicalRelativeL2Record:
    """TP-aware analog of thicket.perturbation.apply_anatomical_relative_l2. Structurally mirrors
    that function's two-pass design exactly (Pass 1: local norm accumulation; Pass 2: regenerate
    noise, apply, measure realized displacement) with two additions:
      1. `theta_sq_sum`/`noise_sq_sum` are ALL-REDUCED (summed across ranks) before deriving the
         rescale `scale` -- every rank ends up with the IDENTICAL global sums and therefore the
         IDENTICAL scale, satisfying ||epsilon_a||_2/||theta_a||_2 = r over the FULL region.
      2. noise for each parameter is drawn at its GLOBAL shape (`_generate_full_shape_noise`,
         every rank draws the SAME full-shape tensor for a given name+seed) and then sliced to
         this rank's local shard (`slice_shard`) before use -- never a new per-shard RNG stream.
      3. Replicated parameters (`shard.is_replicated`) contribute to the norm sums on rank 0 ONLY
         (`shard.counts_toward_global_sum`), preventing world_size-times double counting; every
         rank still receives (and applies) the correct slice of noise for its own parameters.

    At world_size=1 with `all_reduce_sum=identity_all_reduce_sum` (the default) and every
    ShardSpec's `global_shape == local parameter shape` (i.e. nothing is actually sharded), this
    function is numerically IDENTICAL to `apply_anatomical_relative_l2` (proven by test).
    """
    region_param_names = tuple(sorted(set(region_param_names)))
    if not region_param_names:
        raise DegenerateRegionError(f"Region {region!r} has zero parameters -- refusing to perturb an empty region.")
    named = dict(model.named_parameters())
    missing = [n for n in region_param_names if n not in named]
    if missing:
        raise DegenerateRegionError(f"Region {region!r} references parameter name(s) not found on the model: {missing[:10]}")
    missing_shards = [n for n in region_param_names if n not in shard_specs]
    if missing_shards:
        raise ValueError(f"Missing ShardSpec for region {region!r} parameter(s): {missing_shards[:10]}")

    # Pass 1: LOCAL theta/noise squared-L2 sums, then a global all-reduce.
    local_theta_sq_sum = 0.0
    local_noise_sq_sum = 0.0
    for name in region_param_names:
        shard = shard_specs[name]
        if not shard.counts_toward_global_sum:
            continue
        p = named[name]
        local_theta_sq_sum += chunked_squared_l2_sum(p.detach(), chunk_elements=chunk_elements)
        full_noise = _generate_full_shape_noise(shard.global_shape, p.dtype, p.device, _param_noise_seed(seed, name))
        local_shard_noise = slice_shard(full_noise, shard)
        local_noise_sq_sum += chunked_squared_l2_sum(local_shard_noise, chunk_elements=chunk_elements)
        del full_noise, local_shard_noise

    global_theta_sq_sum = all_reduce_sum(local_theta_sq_sum, process_group)
    global_noise_sq_sum = all_reduce_sum(local_noise_sq_sum, process_group)

    theta_l2_norm = global_theta_sq_sum ** 0.5
    raw_noise_l2_norm = global_noise_sq_sum ** 0.5
    if raw_noise_l2_norm == 0.0:
        raise DegenerateRegionError(f"Sampled noise for region {region!r} has zero global norm -- cannot rescale to a nonzero target ratio.")
    scale = (r * theta_l2_norm) / raw_noise_l2_norm

    # Pass 2: regenerate the SAME per-parameter noise (bit-identical -- same seed policy as
    # Pass 1), slice to this rank's shard, apply, measure LOCAL realized displacement, then
    # all-reduce the realized/designed sums too (needed for an honest global diagnostic record).
    local_designed_sq_sum = 0.0
    local_realized_sq_sum = 0.0
    for name in region_param_names:
        shard = shard_specs[name]
        p = named[name]
        theta_before = base_state[name] if base_state is not None else p.detach().clone()
        full_noise = _generate_full_shape_noise(shard.global_shape, p.dtype, p.device, _param_noise_seed(seed, name))
        local_shard_noise = slice_shard(full_noise, shard)
        delta = scale * local_shard_noise
        if shard.counts_toward_global_sum:
            local_designed_sq_sum += chunked_squared_l2_sum(delta.detach(), chunk_elements=chunk_elements)
        p.add_(delta.to(dtype=p.dtype))
        if shard.counts_toward_global_sum:
            local_realized_sq_sum += chunked_squared_l2_diff_sum(p.detach(), theta_before, chunk_elements=chunk_elements)
        del full_noise, local_shard_noise, delta

    global_designed_sq_sum = all_reduce_sum(local_designed_sq_sum, process_group)
    global_realized_sq_sum = all_reduce_sum(local_realized_sq_sum, process_group)

    return AnatomicalRelativeL2Record(
        region=region, seed=seed, requested_r=r, theta_l2_norm=theta_l2_norm, raw_noise_l2_norm=raw_noise_l2_norm,
        scale=scale, designed_epsilon_l2_norm=global_designed_sq_sum ** 0.5, realized_epsilon_l2_norm=global_realized_sq_sum ** 0.5,
        region_param_names=region_param_names,
    )


def _param_noise_seed(seed: int, name: str) -> int:
    """IDENTITY function today (returns `seed` unchanged), named and factored out explicitly so
    a future per-parameter seed-namespacing need (if one is ever discovered) has exactly one call
    site to change -- not because one is needed now. `_generate_noise`'s existing behavior is
    already a pure function of `(shape, dtype, device, seed)` regardless of parameter name; using
    the SAME `seed` for every parameter in a region (exactly as `apply_anatomical_relative_l2`
    already does today) is what makes a "direction" a single coherent Gaussian draw over the
    region, not per-parameter-independent noise -- changing this would be a scientific semantics
    change, never done silently.
    """
    return seed


# =================================================================================================
# Distributed restoration-verification aggregation (task spec Section 8)
# =================================================================================================


def _validate_collective_rpc_results_multi_worker(results: Any, *, label: str, expected_world_size: int) -> List:
    """Multi-worker analog of run_global_visual_thicket_pilot._validate_collective_rpc_results --
    that function hard-asserts `len(results) == 1` (TP=1-only, unchanged, still used by every
    existing 3B/7B call site). This one requires `len(results) == expected_world_size` instead,
    for the NEW TP>1 dispatch path -- a distinct function, never a relaxation of the existing one.
    """
    if not isinstance(results, list):
        raise TypeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected vLLM's own list-of-per-worker-results contract. Got: {results!r}")
    if len(results) != expected_world_size:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; expected exactly {expected_world_size} (the configured tensor_parallel_size).")
    return results


def collective_rpc_all_workers(engine: Any, method: Any, args: tuple = (), *, label: str, expected_world_size: int, ray_get: Optional[Any] = None) -> List:
    """TP>1 collective dispatch -- returns ALL per-worker results (never unwraps to a single
    value, unlike the TP=1-only `_collective_rpc_single_worker`). Same `ray_get`-injection
    pattern already established project-wide for CPU testability.
    """
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results_multi_worker(results, label=label, expected_world_size=expected_world_size)


# =================================================================================================
# Live G4/G5 gate-check (worker-side RPC callable) -- Section 14: "live TP global-relative-L2" /
# "live TP RNG/direction semantics" verification, run against REAL sharded parameters on a REAL
# TP>1 engine before any real candidate is evaluated. NOT executed this session (no GPU) -- this
# is the function a pod-side pre-flight step would call via collective_rpc_all_workers.
# =================================================================================================


def g4_g5_live_relative_l2_check_rpc(
    worker_self, region_param_names: Sequence[str], seed: int, r: float, process_group: Any = None, *, all_reduce_sum: Optional[AllReduceSumFn] = None,
) -> Dict[str, Any]:
    """Runs ONE real apply_anatomical_relative_l2_distributed call against `worker_self`'s own
    live (possibly sharded) parameters, using vllm_shard_mapping to build real ShardSpecs rather
    than injected fakes, and reports whether the realized global relative-L2 lands within
    tolerance of `r` -- the live evidence stage11_32b_readiness.g4_distributed_relative_l2_
    semantics()/g5_distributed_rng_semantics() need to move off NOT_YET_VERIFIED. Every rank
    calls this identically; the caller (collective_rpc_all_workers) collects one dict per rank.

    `all_reduce_sum` defaults to `identity_all_reduce_sum` at tensor_parallel_size<=1 (nothing to
    reduce across -- and no real `torch.distributed` process group is required in that case) and
    to the real `torch_distributed_all_reduce_sum` at tensor_parallel_size>1 (a genuine live
    multi-rank run always has one). Pass it explicitly to substitute a test double.
    """
    from .vllm_shard_mapping import build_shard_specs_for_region, ensure_uniform_tp_size

    tp_size = getattr(worker_self, "tensor_parallel_size", 1)
    if all_reduce_sum is None:
        all_reduce_sum = identity_all_reduce_sum if tp_size <= 1 else torch_distributed_all_reduce_sum
    shard_specs = build_shard_specs_for_region(worker_self.model_runner.model, region_param_names)
    ensure_uniform_tp_size(shard_specs, tp_size if tp_size > 1 else next((s.world_size for s in shard_specs.values() if not s.is_replicated), 1))

    record = apply_anatomical_relative_l2_distributed(
        worker_self.model_runner.model, "g4_g5_live_probe", region_param_names, seed=seed, r=r,
        shard_specs=shard_specs, all_reduce_sum=all_reduce_sum, process_group=process_group,
    )
    realized_relative_l2 = record.realized_epsilon_l2_norm / record.theta_l2_norm if record.theta_l2_norm else float("inf")
    return {
        "requested_r": r, "realized_relative_l2": realized_relative_l2, "theta_l2_norm": record.theta_l2_norm,
        "raw_noise_l2_norm": record.raw_noise_l2_norm, "scale": record.scale, "tp_rank": getattr(worker_self, "rank", 0),
    }


def classify_g4_g5_live_check(per_rank_results: List[Dict[str, Any]], *, tolerance: float = 1e-6) -> bool:
    """Every rank must report the SAME theta_l2_norm/raw_noise_l2_norm/scale (proving the
    all-reduce genuinely synchronized them) AND the realized relative-L2 must land within
    `tolerance` of the requested r -- both conditions are required for a PASS.
    """
    if not per_rank_results:
        return False
    first = per_rank_results[0]
    same_across_ranks = all(
        abs(r["theta_l2_norm"] - first["theta_l2_norm"]) < 1e-9 and abs(r["raw_noise_l2_norm"] - first["raw_noise_l2_norm"]) < 1e-9 and abs(r["scale"] - first["scale"]) < 1e-9
        for r in per_rank_results
    )
    within_tolerance = all(abs(r["realized_relative_l2"] - r["requested_r"]) <= tolerance for r in per_rank_results)
    return same_across_ranks and within_tolerance


def aggregate_distributed_restoration_verification(local_results: List[Dict]) -> Dict:
    """Combines each rank's LOCAL restoration-verification dict (each already computed by the
    existing, memory-bounded per-rank check -- e.g. thicket.worker_rpc.
    verify_exact_fixed_base_restoration_rpc or cpu_base_snapshot.
    verify_exact_fixed_base_restoration_cpu_rpc, called once per rank/shard) into the GLOBAL
    verdict the task spec requires: global n_differing == 0, global max_abs_drift == 0.
    Pure aggregation -- no tensors touched here, no new GPU memory of any kind.
    """
    if not local_results:
        raise ValueError("aggregate_distributed_restoration_verification requires at least one per-rank result.")
    global_max_abs_drift = max(r["max_abs_drift"] for r in local_results)
    # Ranks report a fraction, not a raw count -- fraction==0.0 is the exact-equality signal per
    # rank (any nonzero fraction means at least one differing element on that rank).
    any_rank_has_differing_elements = any(r.get("fraction_elements_differing", 0.0) != 0.0 for r in local_results)
    global_ok = all(r["ok"] for r in local_results) and global_max_abs_drift == 0.0 and not any_rank_has_differing_elements
    return {
        "ok": global_ok, "global_max_abs_drift": global_max_abs_drift,
        "n_ranks": len(local_results), "any_rank_has_differing_elements": any_rank_has_differing_elements,
        "per_rank": local_results,
    }
