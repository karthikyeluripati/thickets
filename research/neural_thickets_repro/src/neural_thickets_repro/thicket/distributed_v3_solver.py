"""Distributed (TP-aware) counterpart of scoped_anatomical_perturbation.
scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 -- Stage-11 32B readiness,
closing the G4 blocker documented in stage11_32b_readiness.V3_SOLVER_DISTRIBUTED_EXTENSION_NOTE.

EXACT LEGACY V3 ALGORITHM (read from scoped_anatomical_perturbation.py's real source this
session, not simplified -- every branch below is a deliberate, literal mirror):

  1. Requires worker_self._base_weights (GPU-resident legacy snapshot) -- HERE, requires
     worker_self._base_weights_cpu (cpu_base_snapshot's CPU-resident snapshot) instead.
  2. `_evaluate(trial_r)`: reset_to_base_weights() -> apply_anatomical_relative_l2(model, region,
     names, seed, trial_r, base_state=...) -> verify out-of-region drift == 0 -> return
     {realized_relative_l2, designed_relative_l2}. The SAME `seed` is passed on EVERY call (never
     resampled) -- `_generate_noise` is a pure function of (shape, dtype, device, seed), so the
     raw Gaussian direction is bit-identical every trial; only `trial_r` (which linearly
     determines `scale = trial_r * theta_l2_norm / raw_noise_l2_norm`) changes.
  3. `solve_bf16_radius(_evaluate, r, ...)`: phase-1 proportional correction before any bracket
     exists, phase-2 deterministic bisection once a two-sided bracket forms, plateau detection on
     an exact-repeat or floating-point-indistinguishable bracket midpoint. Up to
     MAX_RADIUS_SOLVER_ITERATIONS(20) combined iterations.
  4. Converged (abs error <= 1e-6): accept immediately, radius_acceptance_mode="strict" -- the
     model's CURRENT weights already equal the accepted trial (no re-apply).
  5. Not converged, no bracket ever formed: expand_bracket_and_resolve_bf16_radius --
     deterministic geometric expansion (2x, 4x, ..., 2^24x the original displacement) of the SAME
     evaluate_fn/seed/direction until a crossing sample is found (then resumes bisection via the
     SAME _bf16_radius_core_loop) or MAX_BRACKET_EXPANSION_STEPS(24) exhausts -- still hard-fails
     (RadiusCorrectionFailedError) if no bracket ever forms.
  6. A PROVEN plateau (from either step 3 or 5): select_quantization_limited_acceptance picks the
     bracket endpoint closer to `r`, requires its RELATIVE error <= 1e-3 (0.1%) or hard-fails
     (QuantizationToleranceExceededError) -- then an EXPLICIT reset -> reapply(SAME seed,
     candidate_scalar) -> remeasure -> verify EXACT bit-for-bit reproduction of the previously
     observed realized value -> verify out-of-region invariance, before accepting
     (radius_acceptance_mode="quantization_limited"). Never trusts whatever state bisection's
     search happened to leave loaded.

WHAT THIS MODULE REUSES, UNCHANGED, BY IMPORT (never re-derived): solve_bf16_radius,
_bf16_radius_core_loop (via expand_bracket_and_resolve_bf16_radius), expand_bracket_and_resolve_
bf16_radius, select_quantization_limited_acceptance, _build_quantization_aware_result,
RadiusCorrectionFailedError, CorrectionOutOfRegionDriftError, QuantizationToleranceExceededError,
QuantizationPlateauError, and every frozen constant (RADIUS_REALIZATION_TOLERANCE,
QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE, MAX_RADIUS_SOLVER_ITERATIONS,
MAX_BRACKET_EXPANSION_STEPS). The bracket/bisection/plateau DECISION LOGIC is pure control flow
over whatever `evaluate_fn` returns -- feeding it GLOBALLY-REDUCED (rather than per-rank-local)
realized/designed relative-L2 values, from an `evaluate_fn` that is itself distributed, makes
every rank's solver trajectory identical BY CONSTRUCTION (the same pure function, fed the same
already-synchronized inputs, on every rank) without reimplementing a single bracket/bisection
line. `verify_solver_rank_consensus` (below) is the explicit safety-net check the task spec
Section 8 asks for on top of that structural guarantee.

WHAT IS NEW: `_evaluate` becomes distributed -- reset uses cpu_base_snapshot.reset_to_base_
weights_cpu_rpc (this rank's local shard, chunk-bounded), the apply step computes GLOBAL
theta/noise/designed/realized norms via all-reduce (mirroring distributed_perturbation.
apply_anatomical_relative_l2_distributed's Pass-1/Pass-2 structure, but using a CROSS-DEVICE
chunked diff for the realized-epsilon measurement so the CPU-resident base snapshot is NEVER
re-cloned onto GPU in full -- see `_distributed_trial_apply`'s own docstring for why reusing
apply_anatomical_relative_l2_distributed's `base_state=None` fallback naively would silently
reintroduce the exact GPU-memory-doubling problem cpu_base_weights exists to eliminate), and the
out-of-region drift check is aggregated across ranks (MAX of max_abs_drift, SUM of n_differing).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .cpu_base_snapshot import _chunked_cross_device_drift, reset_to_base_weights_cpu_rpc
from .distributed_perturbation import (
    AllReduceSumFn,
    ShardSpec,
    _generate_full_shape_noise,
    _param_noise_seed,
    identity_all_reduce_sum,
    slice_shard,
    torch_distributed_all_reduce_sum,
)
from .memory_bounded_ops import DEFAULT_CHUNK_ELEMENTS, chunked_squared_l2_sum
from .perturbation import AnatomicalRelativeL2Record, DegenerateRegionError
from ..scoped_anatomical_perturbation import (
    MAX_BRACKET_EXPANSION_STEPS,
    MAX_RADIUS_SOLVER_ITERATIONS,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    RADIUS_REALIZATION_TOLERANCE,
    CorrectionOutOfRegionDriftError,
    QuantizationToleranceExceededError,
    RadiusCorrectionFailedError,
    _build_quantization_aware_result,
    expand_bracket_and_resolve_bf16_radius,
    select_quantization_limited_acceptance,
    solve_bf16_radius,
)

QUANTIZATION_AWARE_METHOD_V3_DISTRIBUTED = "fixed_direction_bf16_quantization_aware_v3_distributed"


def _chunked_cross_device_squared_l2_diff_sum(gpu_tensor: torch.Tensor, cpu_tensor: torch.Tensor, chunk_elements: int) -> float:
    """sum_j (gpu_j - cpu_j)^2, moving only ONE bounded CHUNK of `cpu_tensor` to `gpu_tensor`'s
    device at a time -- never a full second GPU-resident copy of the region (which is exactly
    the memory-doubling problem cpu_base_weights exists to eliminate; naively passing the CPU
    snapshot as `base_state` to apply_anatomical_relative_l2_distributed would silently
    reintroduce it, since that function's Pass-2 diff assumes same-device tensors). Same
    mathematical quantity thicket.memory_bounded_ops.chunked_squared_l2_diff_sum computes,
    cross-device.
    """
    flat_gpu, flat_cpu = gpu_tensor.reshape(-1), cpu_tensor.reshape(-1)
    n = flat_gpu.numel()
    total = 0.0
    for start in range(0, n, chunk_elements):
        end = start + chunk_elements
        gpu_chunk = flat_gpu[start:end]
        if gpu_chunk.numel() == 0:
            continue
        cpu_chunk = flat_cpu[start:end].to(device=gpu_chunk.device)
        diff = (gpu_chunk.double() - cpu_chunk.double())
        total += float((diff * diff).sum().item())
    return total


@torch.no_grad()
def _distributed_trial_apply(
    model: torch.nn.Module, region_param_names: Sequence[str], seed: int, trial_r: float, shard_specs: Dict[str, ShardSpec],
    base_weights_cpu: Dict[str, torch.Tensor], *, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
    all_reduce_sum: AllReduceSumFn = identity_all_reduce_sum, process_group: Any = None,
) -> AnatomicalRelativeL2Record:
    """ONE trial of the distributed solver's evaluate step: assumes the caller has ALREADY reset
    every rank's shard to base (via reset_to_base_weights_cpu_rpc) immediately before this call --
    Pass 1 (global theta/noise norm) therefore reads `p.detach()` directly (equals theta_0 right
    now, no clone needed); Pass 2 applies the perturbation and measures the GLOBAL realized
    displacement via `_chunked_cross_device_squared_l2_diff_sum` against `base_weights_cpu`
    (never a GPU-resident re-clone of the region). Structurally mirrors distributed_perturbation.
    apply_anatomical_relative_l2_distributed's two-pass shape exactly, differing only in how the
    Pass-2 realized-diff reference is obtained.
    """
    region_param_names = tuple(sorted(set(region_param_names)))
    if not region_param_names:
        raise DegenerateRegionError("Region has zero parameters -- refusing to perturb an empty region.")
    named = dict(model.named_parameters())

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
        raise DegenerateRegionError("Sampled noise has zero global norm -- cannot rescale to a nonzero target ratio.")
    scale = (trial_r * theta_l2_norm) / raw_noise_l2_norm

    local_designed_sq_sum = 0.0
    local_realized_sq_sum = 0.0
    for name in region_param_names:
        shard = shard_specs[name]
        p = named[name]
        full_noise = _generate_full_shape_noise(shard.global_shape, p.dtype, p.device, _param_noise_seed(seed, name))
        local_shard_noise = slice_shard(full_noise, shard)
        delta = scale * local_shard_noise
        if shard.counts_toward_global_sum:
            local_designed_sq_sum += chunked_squared_l2_sum(delta.detach(), chunk_elements=chunk_elements)
        p.add_(delta.to(dtype=p.dtype))
        if shard.counts_toward_global_sum:
            local_realized_sq_sum += _chunked_cross_device_squared_l2_diff_sum(p.detach(), base_weights_cpu[name], chunk_elements)
        del full_noise, local_shard_noise, delta

    global_designed_sq_sum = all_reduce_sum(local_designed_sq_sum, process_group)
    global_realized_sq_sum = all_reduce_sum(local_realized_sq_sum, process_group)

    return AnatomicalRelativeL2Record(
        region="", seed=seed, requested_r=trial_r, theta_l2_norm=theta_l2_norm, raw_noise_l2_norm=raw_noise_l2_norm,
        scale=scale, designed_epsilon_l2_norm=global_designed_sq_sum ** 0.5, realized_epsilon_l2_norm=global_realized_sq_sum ** 0.5,
        region_param_names=region_param_names,
    )


def _distributed_out_of_region_drift(
    model: torch.nn.Module, base_weights_cpu: Dict[str, torch.Tensor], region_names_set: set, *, chunk_elements: int,
) -> Dict[str, Any]:
    """LOCAL (this rank only) out-of-region check -- the caller,
    `aggregate_distributed_out_of_region_drift`, combines every rank's local result (MAX of
    max_abs_drift, SUM of n_differing). For the S1 whole_model track (32B smoke's actual target),
    region_names_set covers EVERY trainable parameter by construction, so this is vacuously zero
    (nothing left outside the region) -- kept as a real, generic, always-executed check rather
    than special-cased away, so a future anatomy-track reuse (a real, non-empty complement) is
    correct without modification.
    """
    local_max_abs = 0.0
    local_n_differing = 0
    for name, p in model.named_parameters():
        if name in region_names_set:
            continue
        cpu_ref = base_weights_cpu.get(name)
        if cpu_ref is None:
            continue
        flat_p, flat_ref = p.detach().reshape(-1), cpu_ref.reshape(-1)
        n = flat_p.numel()
        for start in range(0, n, chunk_elements):
            end = start + chunk_elements
            p_chunk = flat_p[start:end]
            if p_chunk.numel() == 0:
                continue
            ref_chunk = flat_ref[start:end].to(device=p_chunk.device)
            diff = (p_chunk.double() - ref_chunk.double()).abs()
            local_max_abs = max(local_max_abs, float(diff.max().item()) if diff.numel() else 0.0)
            local_n_differing += int((p_chunk != ref_chunk).sum().item())
    # MAX cannot be expressed through a SUM all-reduce -- the caller (aggregate_distributed_
    # out_of_region_drift) combines local_max_abs_drift with a dedicated MAX-reduction instead.
    return {"local_max_abs_drift": local_max_abs, "local_n_differing": local_n_differing}


def all_reduce_max(value: float, process_group: Any) -> float:
    """MAX-reduction analog of distributed_perturbation.torch_distributed_all_reduce_sum --
    max_abs_drift must be combined with MAX (the worst rank's drift), never SUM.
    """
    import torch.distributed as dist

    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=process_group)
    return float(tensor.item())


def identity_all_reduce_max(value: float, process_group: Any = None) -> float:
    return value


def aggregate_distributed_out_of_region_drift(
    model: torch.nn.Module, base_weights_cpu: Dict[str, torch.Tensor], region_names_set: set,
    *, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS, all_reduce_sum: AllReduceSumFn = identity_all_reduce_sum,
    all_reduce_max: AllReduceSumFn = identity_all_reduce_max, process_group: Any = None,
) -> Dict[str, Any]:
    local = _distributed_out_of_region_drift(model, base_weights_cpu, region_names_set, chunk_elements=chunk_elements)
    global_max_abs = all_reduce_max(local["local_max_abs_drift"], process_group)
    global_n_differing = all_reduce_sum(local["local_n_differing"], process_group)
    return {"max_abs_drift": global_max_abs, "n_differing": global_n_differing}


class SolverRankConsensusError(RuntimeError):
    """Raised by verify_solver_rank_consensus -- one or more ranks' distributed v3 solver
    trajectory (accepted_scalar / radius_acceptance_mode / realized_relative_l2 /
    solver_iterations / quantization_plateau) disagrees with the others. Since every rank runs
    the identical pure bracket/bisection logic fed by the SAME globally-reduced values, this
    should never trigger under correct all-reduce wiring -- it exists as an explicit safety net
    (task spec Section 8), not as the primary correctness mechanism.
    """


_CONSENSUS_FIELDS: Sequence[str] = ("accepted_scalar", "radius_acceptance_mode", "realized_relative_l2", "solver_iterations", "quantization_plateau")


def verify_solver_rank_consensus(per_rank_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_rank_results:
        raise ValueError("verify_solver_rank_consensus requires at least one per-rank result.")
    first = per_rank_results[0]
    mismatched: Dict[str, List[Any]] = {}
    for field_name in _CONSENSUS_FIELDS:
        values = [r.get(field_name) for r in per_rank_results]
        if any(v != values[0] for v in values):
            mismatched[field_name] = values
    if mismatched:
        raise SolverRankConsensusError(f"Distributed v3 solver ranks disagree on: {mismatched}")
    return {"ok": True, "n_ranks": len(per_rank_results), "consensus_fields": {f: first.get(f) for f in _CONSENSUS_FIELDS}}


@torch.no_grad()
def scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed(
    worker_self, seed: int, r: float, region_name: str, region_param_names: Sequence[str], shard_specs: Dict[str, ShardSpec],
    *, all_reduce_sum: AllReduceSumFn = torch_distributed_all_reduce_sum, all_reduce_max: AllReduceSumFn = all_reduce_max, process_group: Any = None,
    max_iterations: int = MAX_RADIUS_SOLVER_ITERATIONS, strict_tolerance: float = RADIUS_REALIZATION_TOLERANCE,
    quantization_plateau_relative_tolerance: float = QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    max_bracket_expansion_steps: int = MAX_BRACKET_EXPANSION_STEPS, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> Dict:
    """Distributed counterpart of scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3
    -- see module docstring for the exact algorithm this mirrors branch-for-branch. Requires
    worker_self._base_weights_cpu (cpu_base_snapshot.store_base_weights_cpu_rpc must already have
    been called). `all_reduce_sum`/`all_reduce_max` default to the REAL torch.distributed
    collectives (a genuine multi-rank run always has a process group); pass
    distributed_perturbation.identity_all_reduce_sum / identity_all_reduce_max for world_size=1
    or CPU-only equivalence testing.
    """
    if not hasattr(worker_self, "_base_weights_cpu"):
        raise RuntimeError(
            "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed requires "
            "cpu_base_snapshot.store_base_weights_cpu_rpc() to have already been called on this worker."
        )
    model = worker_self.model_runner.model
    base_weights_cpu = worker_self._base_weights_cpu
    region_names_set = set(region_param_names)
    last_record: Dict[str, Any] = {"value": None}

    def _evaluate(trial_r: float) -> Dict[str, float]:
        reset_to_base_weights_cpu_rpc(worker_self)
        record = _distributed_trial_apply(
            model, region_param_names, seed, trial_r, shard_specs, base_weights_cpu,
            chunk_elements=chunk_elements, all_reduce_sum=all_reduce_sum, process_group=process_group,
        )
        out_of_region_drift = aggregate_distributed_out_of_region_drift(
            model, base_weights_cpu, region_names_set, chunk_elements=chunk_elements,
            all_reduce_sum=all_reduce_sum, all_reduce_max=all_reduce_max, process_group=process_group,
        )
        if out_of_region_drift["max_abs_drift"] != 0.0:
            raise CorrectionOutOfRegionDriftError(
                f"Distributed BF16 quantization-aware solver trial for region {region_name!r} (seed={seed}) "
                f"changed parameters outside the selected region: max_abs_drift={out_of_region_drift['max_abs_drift']}, "
                f"n_differing={out_of_region_drift['n_differing']}."
            )
        last_record["value"] = record
        return {"realized_relative_l2": record.realized_relative_l2, "designed_relative_l2": record.designed_relative_l2}

    solver_result = solve_bf16_radius(_evaluate, r, max_iterations=max_iterations, tolerance=strict_tolerance)

    if solver_result["converged"]:
        record = last_record["value"]
        return _finalize(region_name, seed, r, "strict", False, solver_result["accepted_scalar"], record, solver_result, strict_tolerance, quantization_plateau_relative_tolerance)

    if not solver_result["quantization_plateau"]:
        expansion_result = expand_bracket_and_resolve_bf16_radius(
            _evaluate, r, solver_result, tolerance=strict_tolerance,
            max_expansion_steps=max_bracket_expansion_steps, max_bisection_iterations=max_iterations,
        )
        if expansion_result["converged"]:
            record = last_record["value"]
            return _finalize(region_name, seed, r, "strict", False, expansion_result["accepted_scalar"], record, expansion_result, strict_tolerance, quantization_plateau_relative_tolerance)
        if not expansion_result["quantization_plateau"]:
            raise RadiusCorrectionFailedError(
                f"Distributed BF16 quantization-aware solver did not converge within tolerance={strict_tolerance} "
                f"after {len(solver_result['attempts'])} original attempts, and deterministic bracket expansion "
                f"({expansion_result.get('expansion_steps_taken', 0)} steps) still found no quantization plateau "
                f"for region {region_name!r} (seed={seed}, requested r={r})."
            )
        solver_result = expansion_result

    nearest_below = solver_result["nearest_realized_below"]
    nearest_above = solver_result["nearest_realized_above"]
    final_attempt = solver_result["attempts"][-1]
    if nearest_below is None or nearest_above is None or final_attempt["bracket_low_scale"] is None or final_attempt["bracket_high_scale"] is None:
        raise RadiusCorrectionFailedError(f"Quantization plateau reported for region {region_name!r} (seed={seed}) but the solver's own bracket is incomplete.")

    selection = select_quantization_limited_acceptance(nearest_below, nearest_above, r, relative_tolerance=quantization_plateau_relative_tolerance)
    nearest_realized = selection["nearest_realized"]
    candidate_scalar = final_attempt["bracket_low_scale"] if selection["which"] == "below" else final_attempt["bracket_high_scale"]

    if not selection["accepted"]:
        raise QuantizationToleranceExceededError(
            f"Quantization plateau proven for region {region_name!r} (seed={seed}, requested r={r}), but the "
            f"nearest attainable relative error {selection['relative_error']} exceeds {quantization_plateau_relative_tolerance}."
        )

    reset_to_base_weights_cpu_rpc(worker_self)
    reproduction_record = _distributed_trial_apply(
        model, region_param_names, seed, candidate_scalar, shard_specs, base_weights_cpu,
        chunk_elements=chunk_elements, all_reduce_sum=all_reduce_sum, process_group=process_group,
    )
    reproduction_drift = aggregate_distributed_out_of_region_drift(
        model, base_weights_cpu, region_names_set, chunk_elements=chunk_elements,
        all_reduce_sum=all_reduce_sum, all_reduce_max=all_reduce_max, process_group=process_group,
    )
    if reproduction_drift["max_abs_drift"] != 0.0:
        raise CorrectionOutOfRegionDriftError(f"Reapplying the selected quantization-limited scalar for region {region_name!r} (seed={seed}) changed out-of-region parameters.")
    if reproduction_record.realized_relative_l2 != nearest_realized:
        raise RadiusCorrectionFailedError(
            f"Reapplying the selected quantization-limited scalar for region {region_name!r} (seed={seed}) did not "
            f"exactly reproduce the previously observed attainable state: expected {nearest_realized}, got {reproduction_record.realized_relative_l2}."
        )

    return _finalize(
        region_name, seed, r, "quantization_limited", True, candidate_scalar, reproduction_record, solver_result,
        strict_tolerance, quantization_plateau_relative_tolerance, nearest_below, nearest_above,
    )


def _finalize(region_name, seed, r, mode, quantization_limited, accepted_scalar, record, solver_result, strict_tolerance, quantization_plateau_relative_tolerance, nearest_below=None, nearest_above=None) -> Dict:
    result = _build_quantization_aware_result(
        region_name=region_name, seed=seed, r=r, radius_acceptance_mode=mode, quantization_limited=quantization_limited,
        accepted_scalar=accepted_scalar, record=record, solver_result=solver_result, strict_tolerance=strict_tolerance,
        quantization_plateau_relative_tolerance=quantization_plateau_relative_tolerance,
        nearest_realized_below=nearest_below, nearest_realized_above=nearest_above,
    )
    result["radius_realization_method"] = QUANTIZATION_AWARE_METHOD_V3_DISTRIBUTED
    return result
