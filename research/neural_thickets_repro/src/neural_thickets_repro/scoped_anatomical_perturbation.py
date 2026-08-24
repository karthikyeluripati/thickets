"""GPU worker-side dispatch wrapper around `thicket.perturbation.apply_anatomical_relative_l2`
-- the Stage 7B anatomical-calibration analog of `scoped_perturbation.scoped_apply_perturbation`
(which wraps the older, expectation-only `scopes.compute_relative_l2_sigma` scale mode instead).

Deliberately a SEPARATE module, not an extension of scoped_perturbation.py: apply_anatomical_
relative_l2 is a distinct, exact-rescale scientific choice (VISUAL_THICKET_EXPERIMENT_SPEC.md
section C2/C3), and the region vocabulary here is thicket.anatomy's L1 atlas regions ("vision",
"multimodal_connector_or_merger", "language"), not scopes.py's PERTURBATION_SCOPES registry
("vision_encoder", "vision_merger", "full_lm") -- the two name the same underlying parameter
sets (see tests/test_scoped_anatomical_perturbation.py), but this module speaks the anatomy
vocabulary directly, since that is what Stage 7's calibration plan and calibration runner use.

Dispatched via vLLM's collective_rpc(Callable, ...), the same mechanism scoped_perturbation.py
already established -- never touches external/RandOpt, never calls perturb_self_weights.

=================================================================================================
BF16 REALIZED-RADIUS CORRECTION (this repair pass -- ROOT CAUSE, proven by instrumentation, not
assumed): a real RunPod Stage-7B smoke candidate (region=vision, requested r=0.035698828543799424)
hard-failed its requested-radius invariant: realized r=0.03569534313727009, abs error 3.485e-06,
against a 1e-6 tolerance. `apply_anatomical_relative_l2`'s ORIGINAL `realized_epsilon_l2_norm`
field was computed from `delta = scale * noise` -- i.e. the norm of the additive tensor BEFORE
`p.add_()` -- never from the actual post-addition parameter change. On bf16 weights, the in-place
`p.add_(delta)` itself rounds AGAIN (bf16's ~8-bit mantissa cannot represent every sum of a
bf16 base value and a bf16 delta exactly), so `p_after - p_before` (fp32-measured) is measurably
different from `delta` -- reproduced directly (tests/test_scoped_anatomical_perturbation.py) with
real bf16 tensors: for a 500,000-element region, one-shot "designed" abs error ~4.1e-5 vs
"realized" (true, post-add) abs error also nonzero and NOT equal to the designed value; for a
tiny 2,000-element region the realized error alone reaches ~5.9e-5. `thicket.perturbation.
apply_anatomical_relative_l2` (this repair pass) now measures and returns BOTH the designed
value (pre-add) and the TRUE realized value (post-add, via `base_state`/a fresh clone) --
`scoped_apply_anatomical_perturbation` below still reports the (now correctly measured) one-shot
result; the NEW `scoped_apply_anatomical_perturbation_bf16_corrected` iteratively corrects the
SAME fixed seeded Gaussian direction's scalar magnitude (never resampling, never changing which
noise is drawn) until the TRUE realized radius is within tolerance, confirmed to converge in 2-3
iterations for realistic (500k+ element) region sizes (see the same test file).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .diagnostics.perturb_restore_drift import measure_drift
from .thicket.perturbation import apply_anatomical_relative_l2

# The ONLY realization method this module implements -- persisted into every checkpoint/run
# manifest as an identity field (run_stage7b_anatomical_calibration.py) so a pre-existing
# checkpoint written under a different (e.g. the original, uncorrected one-shot) method is never
# silently resumed under this one.
BF16_RADIUS_REALIZATION_METHOD = "fixed_direction_bf16_corrected_v1"
MAX_RADIUS_CORRECTION_ITERATIONS = 5
RADIUS_REALIZATION_TOLERANCE = 1e-6  # never loosened -- see module docstring


class RadiusCorrectionFailedError(RuntimeError):
    """The TRUE realized relative-L2 could not be brought within RADIUS_REALIZATION_TOLERANCE
    of the requested radius after MAX_RADIUS_CORRECTION_ITERATIONS attempts (a genuine
    numerical plateau -- e.g. too few elements in the region for bf16 quantization noise to
    average out) -- hard-fails with the full attempt history, never silently accepted with a
    looser tolerance.
    """


class CorrectionOutOfRegionDriftError(RuntimeError):
    """A correction attempt changed a parameter outside the selected anatomical region --
    hard-fails immediately; the correction procedure must never leak outside its declared
    region on any attempt, not just the accepted one.
    """


def scoped_apply_anatomical_perturbation(
    worker_self, seed: int, r: float, region_name: str, region_param_names: Sequence[str],
) -> Dict:
    """One-shot (uncorrected) apply -- kept for direct comparison/testing against the corrected
    version, and for any caller that only needs the DESIGNED (pre-BF16-add) estimate. Stage 7B's
    own candidate lifecycle (run_stage7b_anatomical_calibration.py) no longer dispatches this
    directly -- it dispatches scoped_apply_anatomical_perturbation_bf16_corrected below, which
    checks the TRUE post-addition realized radius, not the designed one.

    Dispatched via collective_rpc(scoped_apply_anatomical_perturbation, args=(seed, r,
    region_name, region_param_names)). Defensive restore-then-perturb (same discipline as
    scoped_perturbation.scoped_apply_perturbation): always applies from the exact stored base,
    never from whatever the current (possibly already-perturbed) state happens to be.
    """
    worker_self.reset_to_base_weights()
    model = worker_self.model_runner.model
    base_state = getattr(worker_self, "_base_weights", None)
    record = apply_anatomical_relative_l2(model, region_name, region_param_names, seed, r, base_state=base_state)
    return {
        "region": record.region,
        "seed": record.seed,
        "requested_relative_l2": record.requested_r,
        "designed_relative_l2": record.designed_relative_l2,
        "realized_relative_l2": record.realized_relative_l2,
        "theta_l2_norm": record.theta_l2_norm,
        "raw_noise_l2_norm": record.raw_noise_l2_norm,
        "scale": record.scale,
        "designed_epsilon_l2_norm": record.designed_epsilon_l2_norm,
        "realized_epsilon_l2_norm": record.realized_epsilon_l2_norm,
        "region_param_count": record.param_count,
    }


def scoped_apply_anatomical_perturbation_bf16_corrected(
    worker_self, seed: int, r: float, region_name: str, region_param_names: Sequence[str],
    *, max_iterations: int = MAX_RADIUS_CORRECTION_ITERATIONS, tolerance: float = RADIUS_REALIZATION_TOLERANCE,
) -> Dict:
    """Iteratively corrects the SAME fixed seeded Gaussian direction's scalar magnitude until
    the TRUE (post-BF16-addition) realized relative-L2 is within `tolerance` of `r`, or hard
    -fails after `max_iterations` attempts. Never resamples: `_generate_noise(p, seed)` is
    deterministic in (tensor shape/dtype/device, seed), so calling `apply_anatomical_relative_l2`
    again with the SAME `seed` regenerates the BIT-IDENTICAL raw noise every attempt -- only the
    `r` argument (which linearly determines `scale = r * theta_l2_norm / raw_noise_l2_norm`,
    with `theta_l2_norm`/`raw_noise_l2_norm` themselves fixed across attempts since base+noise
    are unchanged) is adjusted, which is exactly the requested `corrected_scale = current_scale
    * requested_radius / r_actual` update expressed through the `r` "knob" apply_anatomical_
    relative_l2 already exposes -- no change to that function's scale-derivation formula.

    Requires `worker_self._base_weights` (from `store_base_weights()`) -- used both as
    `apply_anatomical_relative_l2`'s `base_state` (avoiding an extra per-attempt clone of a
    possibly huge region) and as the reset target / out-of-region drift baseline for EVERY
    attempt, not just the first.
    """
    if not hasattr(worker_self, "_base_weights"):
        raise RuntimeError(
            "scoped_apply_anatomical_perturbation_bf16_corrected requires store_base_weights() "
            "to have already been called on this worker (no base snapshot to reset/measure against)."
        )
    model = worker_self.model_runner.model
    base_state = worker_self._base_weights
    region_names_set = set(region_param_names)

    attempts: List[Dict[str, Any]] = []
    current_r = r
    record = None
    for iteration in range(1, max_iterations + 1):
        worker_self.reset_to_base_weights()
        record = apply_anatomical_relative_l2(model, region_name, region_param_names, seed, current_r, base_state=base_state)

        out_of_region_drift = measure_drift(model, base_state, param_filter=lambda n: n not in region_names_set)
        if out_of_region_drift["max_abs_drift"] != 0.0:
            raise CorrectionOutOfRegionDriftError(
                f"Correction attempt {iteration} for region {region_name!r} (seed={seed}) changed "
                f"parameters outside the selected region: max_abs_drift={out_of_region_drift['max_abs_drift']}, "
                f"fraction_elements_differing={out_of_region_drift['fraction_elements_differing']}."
            )

        realized_r = record.realized_relative_l2
        designed_r = record.designed_relative_l2
        absolute_error = abs(realized_r - r)
        attempts.append({
            "iteration": iteration,
            "r_input": current_r,
            "designed_relative_l2": designed_r,
            "designed_abs_error": abs(designed_r - r),
            "realized_relative_l2": realized_r,
            "realized_abs_error": absolute_error,
            "scale": record.scale,
        })
        if absolute_error <= tolerance:
            break
        current_r = current_r * r / realized_r

    final = attempts[-1]
    converged = final["realized_abs_error"] <= tolerance
    if not converged:
        raise RadiusCorrectionFailedError(
            f"BF16 radius correction did not converge within tolerance={tolerance} after "
            f"{max_iterations} attempts for region {region_name!r} (seed={seed}, requested "
            f"r={r}): final realized={final['realized_relative_l2']}, "
            f"abs_error={final['realized_abs_error']}. Attempts: {attempts}"
        )

    return {
        "region": region_name,
        "seed": seed,
        "direction_seed": seed,
        "requested_relative_l2": r,
        "designed_relative_l2": final["designed_relative_l2"],
        "designed_abs_error": final["designed_abs_error"],
        "realized_relative_l2": final["realized_relative_l2"],
        "realized_abs_error": final["realized_abs_error"],
        "initial_realized_relative_l2": attempts[0]["realized_relative_l2"],
        "final_realized_relative_l2": final["realized_relative_l2"],
        "final_absolute_radius_error": final["realized_abs_error"],
        "final_scale": final["scale"],
        "correction_iterations": len(attempts),
        "radius_realization_method": BF16_RADIUS_REALIZATION_METHOD,
        "theta_l2_norm": record.theta_l2_norm,
        "raw_noise_l2_norm": record.raw_noise_l2_norm,
        "realized_epsilon_l2_norm": record.realized_epsilon_l2_norm,
        "region_param_count": record.param_count,
        "attempts": attempts,
    }


def diag_snapshot_base(worker_self) -> str:
    """Diagnostic-only in-worker base snapshot for drift measurement -- same shape as
    diagnostics/scope_isolation_gpu_check.py's _diag_snapshot_base, duplicated rather than
    cross-imported (consistent with this project's per-diagnostic-module convention).
    """
    model = worker_self.model_runner.model
    worker_self._anatomical_diag_base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return f"snapshotted {len(worker_self._anatomical_diag_base_state)} tensors"


def diag_region_drift(worker_self, region_param_names: Sequence[str]) -> Dict:
    """Splits drift into in-region (the perturbed region's own parameters) vs out-of-region
    (the literal complement across the entire runtime model) -- reuses measure_drift's
    param_filter, same pattern as scope_isolation_gpu_check.py's _diag_scope_drift. Must be
    called after diag_snapshot_base and before reset_to_base_weights.
    """
    if not hasattr(worker_self, "_anatomical_diag_base_state"):
        raise RuntimeError("diag_snapshot_base was never called on this worker before diag_region_drift")
    model = worker_self.model_runner.model
    base_state = worker_self._anatomical_diag_base_state
    selected_names = set(region_param_names)

    in_region = measure_drift(model, base_state, param_filter=lambda n: n in selected_names)
    out_of_region = measure_drift(model, base_state, param_filter=lambda n: n not in selected_names)
    total_params = sum(1 for _ in model.named_parameters())
    return {
        "in_region": in_region,
        "out_of_region": out_of_region,
        "region_param_count": len(selected_names),
        "out_of_region_param_count": total_params - len(selected_names),
    }


def diag_full_model_drift(worker_self) -> Dict:
    if not hasattr(worker_self, "_anatomical_diag_base_state"):
        raise RuntimeError("diag_snapshot_base was never called on this worker before diag_full_model_drift")
    model = worker_self.model_runner.model
    return measure_drift(model, worker_self._anatomical_diag_base_state)
