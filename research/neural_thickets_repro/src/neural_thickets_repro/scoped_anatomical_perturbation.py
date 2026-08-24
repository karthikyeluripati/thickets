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

=================================================================================================
BF16 BRACKETED SOLVER v2 (this repair pass -- the v1 proportional-only corrector, above, was
found on a real RunPod full-calibration run to OSCILLATE near the target without converging: at
the smallest frozen radius (region=vision, r=0.0035698828543799426) its 5 attempts alternated
overshoot/undershoot -- attempt 4 (overshoot, abs error 1.37e-6) was the closest, attempt 5
(undershoot, abs error 2.98e-6) then moved AWAY again, and the fixed 5-attempt budget ran out.
DIAGNOSIS (not assumed): the map from scalar magnitude `r` to the TRUE bf16-realized relative-L2
is not smooth near this radius/region combination -- it is a piecewise-constant "staircase"
caused by bf16's coarse rounding of each per-element `p.add_()` (at a very small requested
radius, the per-element delta is tiny relative to many weight magnitudes' own bf16 ULP, so many
elements' contributions round away or snap between only a few representable outcomes). A linear
extrapolation (`r_next = r * requested/realized`, valid for a smooth/monotonic map) can legally
overshoot the true root and oscillate forever on a staircase, exactly as observed. `solve_bf16_
radius` below replaces blind linear extrapolation, once a sign change is observed, with
deterministic BISECTION inside a maintained [low, high] scalar bracket (undershoot/overshoot
respectively) -- provably non-oscillating (the bracket width is halved, or a plateau is
explicitly detected, every bisection step) -- falling back to proportional correction only
BEFORE a bracket exists (mirrors the task's own step D: "use proportional correction only to
quickly approach target"). Never resamples the seeded direction; only the scalar magnitude
argument to `apply_anatomical_relative_l2` changes between trials, exactly as in v1.
=================================================================================================
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .diagnostics.perturb_restore_drift import measure_drift
from .thicket.perturbation import apply_anatomical_relative_l2

# The realization methods this module implements -- persisted into every checkpoint/run
# manifest as an identity field (run_stage7b_anatomical_calibration.py) so a pre-existing
# checkpoint written under a different method is never silently resumed under another.
BF16_RADIUS_REALIZATION_METHOD = "fixed_direction_bf16_corrected_v1"  # superseded by v2 below; kept for direct comparison/audit
BF16_RADIUS_REALIZATION_METHOD_V2 = "fixed_direction_bf16_bracketed_v2"
MAX_RADIUS_CORRECTION_ITERATIONS = 5  # v1's fixed budget -- proven insufficient for the oscillating case above
MAX_RADIUS_SOLVER_ITERATIONS = 20  # v2's budget
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


class QuantizationPlateauError(RuntimeError):
    """Not raised directly -- RadiusCorrectionFailedError is always what propagates out of
    scoped_apply_anatomical_perturbation_bf16_bracketed; this marker exists only so
    `solve_bf16_radius`'s own returned `quantization_plateau` flag has a matching, documented
    exception name to point readers at when they see it in an error message.
    """


def solve_bf16_radius(
    evaluate_fn: Callable[[float], Dict[str, float]], requested_r: float, *,
    max_iterations: int = MAX_RADIUS_SOLVER_ITERATIONS, tolerance: float = RADIUS_REALIZATION_TOLERANCE,
) -> Dict[str, Any]:
    """Pure control-flow solver -- `evaluate_fn` is the ONLY side-effecting call (real callers
    pass a closure that resets to base, applies the SAME fixed seeded direction at the trial
    scalar, verifies outside-region invariance, and measures the TRUE post-BF16-add realized
    radius -- see scoped_apply_anatomical_perturbation_bf16_bracketed below; test callers pass a
    pure function over a scripted sequence). No GPU/model access happens in this function
    itself, which is what makes the bracket/bisection/plateau LOGIC directly unit-testable
    against hand-crafted sequences (including the exact observed live oscillating sequence),
    independent of real floating-point noise.

    `evaluate_fn(trial_r) -> {"realized_relative_l2": float, "designed_relative_l2": float}`.

    Phase 1 (no bracket yet): proportional correction, `next_r = current_r * requested_r /
    realized_r` -- identical to v1's only strategy, used here ONLY to quickly approach the
    target (step D of the algorithm) before a bracket exists.

    Phase 2 (bracket exists, i.e. at least one observed realized value <= requested_r AND at
    least one >= requested_r): deterministic BISECTION -- `next_r = (bracket_low_scale +
    bracket_high_scale) / 2`. The bracket is tightened (never widened) every time a new
    observation lands on either side, so its scalar width never increases.

    Plateau detection (quantization_plateau=True): triggers when (a) a bisection trial repeats
    an EXACT realized value already seen during the bisection phase (the achievable value set is
    discrete and we are revisiting the same discrete step), or (b) the bracket's own scalar
    midpoint is no longer distinguishable in floating point from either endpoint (the bracket
    cannot be subdivided any further at all). Either way this is treated as a genuine numerical
    floor, never silently accepted as "close enough".
    """
    attempts: List[Dict[str, Any]] = []
    bracket_low: Optional[Tuple[float, float]] = None  # (scale, realized) with realized <= requested_r, tightest (largest scale) so far
    bracket_high: Optional[Tuple[float, float]] = None  # (scale, realized) with realized >= requested_r, tightest (smallest scale) so far
    seen_bisection_realized: set = set()

    current_r = requested_r
    converged = False
    plateau = False
    best_index: Optional[int] = None

    for iteration in range(1, max_iterations + 1):
        measurement = evaluate_fn(current_r)
        realized_r = measurement["realized_relative_l2"]
        designed_r = measurement.get("designed_relative_l2", realized_r)
        absolute_error = abs(realized_r - requested_r)

        if realized_r <= requested_r and (bracket_low is None or current_r > bracket_low[0]):
            bracket_low = (current_r, realized_r)
        if realized_r >= requested_r and (bracket_high is None or current_r < bracket_high[0]):
            bracket_high = (current_r, realized_r)

        attempts.append({
            "iteration": iteration, "scalar": current_r, "designed_relative_l2": designed_r,
            "realized_relative_l2": realized_r, "absolute_error": absolute_error,
            "bracket_low_scale": bracket_low[0] if bracket_low else None,
            "bracket_high_scale": bracket_high[0] if bracket_high else None,
            "bracket_low_realized": bracket_low[1] if bracket_low else None,
            "bracket_high_realized": bracket_high[1] if bracket_high else None,
        })

        if best_index is None or absolute_error < attempts[best_index]["absolute_error"]:
            best_index = iteration - 1

        if absolute_error <= tolerance:
            converged = True
            break

        if bracket_low is not None and bracket_high is not None:
            if realized_r in seen_bisection_realized:
                plateau = True
                break
            seen_bisection_realized.add(realized_r)
            next_r = (bracket_low[0] + bracket_high[0]) / 2.0
            if next_r == bracket_low[0] or next_r == bracket_high[0]:
                plateau = True
                break
            current_r = next_r
        else:
            if realized_r == 0:
                plateau = True
                break
            current_r = current_r * requested_r / realized_r
    else:
        if bracket_low is not None and bracket_high is not None:
            plateau = True

    best = attempts[best_index]
    return {
        "converged": converged,
        "quantization_plateau": bool(plateau and not converged),
        "attempts": attempts,
        "best_iteration": best["iteration"],
        "best_scalar": best["scalar"],
        "best_designed_relative_l2": best["designed_relative_l2"],
        "best_realized_relative_l2": best["realized_relative_l2"],
        "best_absolute_error": best["absolute_error"],
        "accepted_scalar": (best["scalar"] if converged else None),
        "nearest_realized_below": bracket_low[1] if bracket_low else None,
        "nearest_realized_above": bracket_high[1] if bracket_high else None,
    }


def scoped_apply_anatomical_perturbation_bf16_bracketed(
    worker_self, seed: int, r: float, region_name: str, region_param_names: Sequence[str],
    *, max_iterations: int = MAX_RADIUS_SOLVER_ITERATIONS, tolerance: float = RADIUS_REALIZATION_TOLERANCE,
) -> Dict:
    """v2: replaces v1's proportional-only correction with solve_bf16_radius's bracketed search
    (see module docstring for why v1 could oscillate without converging). Every trial resets to
    the frozen theta_0, applies the SAME fixed seeded direction at the trial scalar, verifies
    outside-region invariance, and measures the TRUE realized radius -- exactly the v1
    discipline, just with a smarter next-scalar choice. `worker_self._base_weights` (from
    `store_base_weights()`) is required, same as v1.

    PRESERVES ACTUAL INFERENCE WEIGHTS: `solve_bf16_radius` returns (accepts) immediately on the
    FIRST trial within tolerance, with no further trial or reset afterward -- so the model's
    current parameter state when this function returns IS exactly the accepted trial's weights,
    never reconstructed through a second numerical path.

    Raises RadiusCorrectionFailedError (never silently accepts a looser tolerance) if the
    solver does not converge within `max_iterations`, including full attempt evidence and,
    when detected, `quantization_plateau=True` with the nearest achievable realized values on
    either side of the target.
    """
    if not hasattr(worker_self, "_base_weights"):
        raise RuntimeError(
            "scoped_apply_anatomical_perturbation_bf16_bracketed requires store_base_weights() "
            "to have already been called on this worker (no base snapshot to reset/measure against)."
        )
    model = worker_self.model_runner.model
    base_state = worker_self._base_weights
    region_names_set = set(region_param_names)
    last_record: Dict[str, Any] = {"value": None}

    def _evaluate(trial_r: float) -> Dict[str, float]:
        worker_self.reset_to_base_weights()
        record = apply_anatomical_relative_l2(model, region_name, region_param_names, seed, trial_r, base_state=base_state)

        out_of_region_drift = measure_drift(model, base_state, param_filter=lambda n: n not in region_names_set)
        if out_of_region_drift["max_abs_drift"] != 0.0:
            raise CorrectionOutOfRegionDriftError(
                f"BF16 bracketed solver trial for region {region_name!r} (seed={seed}) changed "
                f"parameters outside the selected region: max_abs_drift={out_of_region_drift['max_abs_drift']}, "
                f"fraction_elements_differing={out_of_region_drift['fraction_elements_differing']}."
            )

        last_record["value"] = record
        return {"realized_relative_l2": record.realized_relative_l2, "designed_relative_l2": record.designed_relative_l2}

    solver_result = solve_bf16_radius(_evaluate, r, max_iterations=max_iterations, tolerance=tolerance)

    if not solver_result["converged"]:
        raise RadiusCorrectionFailedError(
            f"BF16 bracketed radius solver did not converge within tolerance={tolerance} after "
            f"{len(solver_result['attempts'])} attempts for region {region_name!r} (seed={seed}, "
            f"requested r={r}): quantization_plateau={solver_result['quantization_plateau']}, "
            f"best_realized={solver_result['best_realized_relative_l2']}, "
            f"best_absolute_error={solver_result['best_absolute_error']}, "
            f"nearest_realized_below={solver_result['nearest_realized_below']}, "
            f"nearest_realized_above={solver_result['nearest_realized_above']}. "
            f"Attempts: {solver_result['attempts']}"
        )

    record = last_record["value"]
    return {
        "region": region_name,
        "seed": seed,
        "direction_seed": seed,
        "requested_relative_l2": r,
        "designed_relative_l2": solver_result["best_designed_relative_l2"],
        "designed_abs_error": abs(solver_result["best_designed_relative_l2"] - r),
        "realized_relative_l2": solver_result["best_realized_relative_l2"],
        "realized_abs_error": solver_result["best_absolute_error"],
        "initial_realized_relative_l2": solver_result["attempts"][0]["realized_relative_l2"],
        "final_realized_relative_l2": solver_result["best_realized_relative_l2"],
        "final_absolute_radius_error": solver_result["best_absolute_error"],
        "final_scale": record.scale,
        "correction_iterations": len(solver_result["attempts"]),
        "solver_iterations": len(solver_result["attempts"]),
        "quantization_plateau": solver_result["quantization_plateau"],
        "nearest_realized_below": solver_result["nearest_realized_below"],
        "nearest_realized_above": solver_result["nearest_realized_above"],
        "radius_realization_method": BF16_RADIUS_REALIZATION_METHOD_V2,
        "theta_l2_norm": record.theta_l2_norm,
        "raw_noise_l2_norm": record.raw_noise_l2_norm,
        "realized_epsilon_l2_norm": record.realized_epsilon_l2_norm,
        "region_param_count": record.param_count,
        "attempts": solver_result["attempts"],
    }


"""
=================================================================================================
QUANTIZATION-AWARE ACCEPTANCE -- v3 (this repair pass): the v2 bracketed solver was then run for
real on the Stage-7B three-region smallest-radius numerical smoke and produced DECISIVE evidence
that strict 1e-6 is physically unattainable for one specific (region, radius) cell: vision and
language both converged strictly (abs errors 9.91e-7 and 3.46e-7), but the connector region
proved a genuine quantization_plateau -- the bracketed solver observed an EXACT repeated realized
value during bisection (two distinct trial scalars producing bit-identical bf16-realized
outcomes), with the target provably bracketed between two attainable BF16 states
(0.0035686268537125777 below, 0.0035711468798464217 above) whose nearest relative error is
1.256e-6 absolute / requested ~= 3.52e-4 relative (0.0352%) -- comfortably inside a 0.1% bound.

v3 does NOT change the solver (solve_bf16_radius, above, is reused unchanged -- same bisection,
same plateau detection) -- it adds a strictly narrower ACCEPTANCE layer on top:
  - if solve_bf16_radius converges strictly (abs error <= RADIUS_REALIZATION_TOLERANCE): accept
    exactly as v2 did (radius_acceptance_mode="strict").
  - if it does NOT converge and quantization_plateau is NOT proven: hard-fail exactly as v2 did
    (RadiusCorrectionFailedError) -- there is no fallback for a merely-exhausted, non-plateaued
    search; only a PROVEN quantization floor may trigger the fallback below.
  - if quantization_plateau IS proven: select whichever of the two bracket endpoints
    (nearest_realized_below/above) is closer to the requested radius, compute its RELATIVE error
    (a fraction of the requested radius, not an absolute number -- deliberately scale-appropriate
    across the six-decade-wide frozen radius grid), and accept it (radius_acceptance_mode=
    "quantization_limited") ONLY if that relative error is <= QUANTIZATION_PLATEAU_RELATIVE_
    TOLERANCE (1e-3, i.e. 0.1%) -- otherwise hard-fail with QuantizationToleranceExceededError.
    This 1e-3 bound is a NUMERICAL ADMISSIBILITY bound, never an experimental hyperparameter and
    never chosen from any capability/task performance signal -- it exists purely to state how
    close a bf16-representable weight state must be to the requested radius, on a request-
    relative scale, before the pipeline accepts "this is what 'radius r' physically means on
    this hardware" in place of the exact nominal value. The six frozen scientific radii (spanning
    ~0.0036 to ~0.36, two orders of magnitude) are entirely unaffected: this bound only ever
    activates for the specific (region, radius, seed) cells where the bracketed solver has
    already PROVEN (never assumed) that no closer bf16 state exists, and the ACTUAL realized
    radius -- never the nominal requested one -- is what gets persisted and used downstream.

Accepting the selected fallback scalar is NOT simply "reuse whatever the solver happened to
leave loaded" -- the solver's own trial history may have moved past that scalar while searching
(bisection can revisit/pass through states after they were first observed). Section 3 of the
task requires an explicit, separate reset -> reapply(SAME seed, SAME direction, the selected
scalar) -> remeasure -> verify-exact-reproduction -> verify-outside-region-invariance sequence
before any capability evaluation is allowed to see the fallback state -- implemented literally
below, never skipped for the fallback path.
=================================================================================================
"""

QUANTIZATION_AWARE_METHOD_V3 = "fixed_direction_bf16_quantization_aware_v3"
QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE = 1e-3  # 0.1% -- numerical admissibility bound, frozen, never task-performance-derived


class QuantizationToleranceExceededError(RadiusCorrectionFailedError):
    """A quantization_plateau WAS proven, but even the nearest attainable bf16 state's relative
    error exceeds QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE -- the fallback is refused (never
    silently accepted at a looser bound) and this propagates as a RadiusCorrectionFailedError
    to any caller only catching that base class.
    """


def select_quantization_limited_acceptance(
    nearest_realized_below: float, nearest_realized_above: float, requested_r: float,
    *, relative_tolerance: float = QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
) -> Dict[str, Any]:
    """Pure decision logic (no GPU/model access) -- given a PROVEN plateau's two bracket
    endpoints, picks whichever is closer to the requested radius and checks its RELATIVE error
    against `relative_tolerance`. Separated from scoped_apply_anatomical_perturbation_bf16_
    quantization_aware_v3 so the exact acceptance arithmetic (including the real connector
    numbers from the Stage-7B smoke) is directly unit-testable without constructing real bf16
    tensors that happen to plateau at those exact values.

    Returns {"which": "below"|"above", "nearest_realized": float, "absolute_error": float,
    "relative_error": float, "accepted": bool}. Never raises -- callers decide what to do with
    accepted=False (this function only computes the arithmetic and the admissibility check).
    """
    below_error = abs(nearest_realized_below - requested_r)
    above_error = abs(nearest_realized_above - requested_r)
    if below_error <= above_error:
        which, nearest_realized, absolute_error = "below", nearest_realized_below, below_error
    else:
        which, nearest_realized, absolute_error = "above", nearest_realized_above, above_error
    relative_error = absolute_error / requested_r if requested_r > 0 else 0.0
    return {
        "which": which, "nearest_realized": nearest_realized, "absolute_error": absolute_error,
        "relative_error": relative_error, "accepted": relative_error <= relative_tolerance,
    }


def scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(
    worker_self, seed: int, r: float, region_name: str, region_param_names: Sequence[str],
    *, max_iterations: int = MAX_RADIUS_SOLVER_ITERATIONS, strict_tolerance: float = RADIUS_REALIZATION_TOLERANCE,
    quantization_plateau_relative_tolerance: float = QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
) -> Dict:
    """v3: reuses solve_bf16_radius (v2's solver, UNCHANGED) and adds the two-tier acceptance
    rule described in the module docstring above. Requires worker_self._base_weights, same as
    v1/v2.
    """
    if not hasattr(worker_self, "_base_weights"):
        raise RuntimeError(
            "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 requires "
            "store_base_weights() to have already been called on this worker (no base snapshot "
            "to reset/measure against)."
        )
    model = worker_self.model_runner.model
    base_state = worker_self._base_weights
    region_names_set = set(region_param_names)
    last_record: Dict[str, Any] = {"value": None}

    def _evaluate(trial_r: float) -> Dict[str, float]:
        worker_self.reset_to_base_weights()
        record = apply_anatomical_relative_l2(model, region_name, region_param_names, seed, trial_r, base_state=base_state)

        out_of_region_drift = measure_drift(model, base_state, param_filter=lambda n: n not in region_names_set)
        if out_of_region_drift["max_abs_drift"] != 0.0:
            raise CorrectionOutOfRegionDriftError(
                f"BF16 quantization-aware solver trial for region {region_name!r} (seed={seed}) "
                f"changed parameters outside the selected region: max_abs_drift="
                f"{out_of_region_drift['max_abs_drift']}, fraction_elements_differing="
                f"{out_of_region_drift['fraction_elements_differing']}."
            )

        last_record["value"] = record
        return {"realized_relative_l2": record.realized_relative_l2, "designed_relative_l2": record.designed_relative_l2}

    solver_result = solve_bf16_radius(_evaluate, r, max_iterations=max_iterations, tolerance=strict_tolerance)

    if solver_result["converged"]:
        record = last_record["value"]
        return _build_quantization_aware_result(
            region_name=region_name, seed=seed, r=r, radius_acceptance_mode="strict", quantization_limited=False,
            accepted_scalar=solver_result["accepted_scalar"], record=record, solver_result=solver_result,
            strict_tolerance=strict_tolerance, quantization_plateau_relative_tolerance=quantization_plateau_relative_tolerance,
        )

    if not solver_result["quantization_plateau"]:
        raise RadiusCorrectionFailedError(
            f"BF16 quantization-aware solver did not converge within tolerance={strict_tolerance} "
            f"after {len(solver_result['attempts'])} attempts for region {region_name!r} "
            f"(seed={seed}, requested r={r}), and no quantization plateau was proven -- refusing "
            f"the quantization-limited fallback, which requires a PROVEN plateau. "
            f"best_realized={solver_result['best_realized_relative_l2']}, "
            f"best_absolute_error={solver_result['best_absolute_error']}. Attempts: {solver_result['attempts']}"
        )

    nearest_below = solver_result["nearest_realized_below"]
    nearest_above = solver_result["nearest_realized_above"]
    final_attempt = solver_result["attempts"][-1]
    if nearest_below is None or nearest_above is None or final_attempt["bracket_low_scale"] is None or final_attempt["bracket_high_scale"] is None:
        raise RadiusCorrectionFailedError(
            f"Quantization plateau reported for region {region_name!r} (seed={seed}) but the "
            f"solver's own bracket is incomplete -- refusing the fallback without a proven "
            f"bracket on BOTH sides of the target. Attempts: {solver_result['attempts']}"
        )

    selection = select_quantization_limited_acceptance(
        nearest_below, nearest_above, r, relative_tolerance=quantization_plateau_relative_tolerance,
    )
    nearest_realized = selection["nearest_realized"]
    candidate_scalar = final_attempt["bracket_low_scale"] if selection["which"] == "below" else final_attempt["bracket_high_scale"]

    if not selection["accepted"]:
        raise QuantizationToleranceExceededError(
            f"Quantization plateau proven for region {region_name!r} (seed={seed}, requested "
            f"r={r}), but the nearest attainable bf16 state's relative error "
            f"{selection['relative_error']} exceeds the {quantization_plateau_relative_tolerance} "
            f"(0.1%) admissibility bound -- nearest_realized_below={nearest_below}, "
            f"nearest_realized_above={nearest_above}, best_absolute_error="
            f"{solver_result['best_absolute_error']}. Refusing to accept."
        )

    # Section 3: explicit reset -> reapply(SAME seed/direction, selected scalar) -> remeasure ->
    # verify EXACT reproduction -> verify outside-region invariance, before returning -- never
    # simply trusts whatever the solver's search happened to leave loaded.
    worker_self.reset_to_base_weights()
    reproduction_record = apply_anatomical_relative_l2(model, region_name, region_param_names, seed, candidate_scalar, base_state=base_state)
    reproduction_drift = measure_drift(model, base_state, param_filter=lambda n: n not in region_names_set)
    if reproduction_drift["max_abs_drift"] != 0.0:
        raise CorrectionOutOfRegionDriftError(
            f"Reapplying the selected quantization-limited scalar for region {region_name!r} "
            f"(seed={seed}) changed parameters outside the selected region: max_abs_drift="
            f"{reproduction_drift['max_abs_drift']}."
        )
    if reproduction_record.realized_relative_l2 != nearest_realized:
        raise RadiusCorrectionFailedError(
            f"Reapplying the selected quantization-limited scalar for region {region_name!r} "
            f"(seed={seed}) did not exactly reproduce the previously observed attainable state: "
            f"expected {nearest_realized}, got {reproduction_record.realized_relative_l2}. Noise "
            f"generation must be deterministic -- this indicates a real bug, not a tolerance issue."
        )

    return _build_quantization_aware_result(
        region_name=region_name, seed=seed, r=r, radius_acceptance_mode="quantization_limited", quantization_limited=True,
        accepted_scalar=candidate_scalar, record=reproduction_record, solver_result=solver_result,
        strict_tolerance=strict_tolerance, quantization_plateau_relative_tolerance=quantization_plateau_relative_tolerance,
        nearest_realized_below=nearest_below, nearest_realized_above=nearest_above,
    )


def _build_quantization_aware_result(
    *, region_name: str, seed: int, r: float, radius_acceptance_mode: str, quantization_limited: bool,
    accepted_scalar: float, record: Any, solver_result: Dict[str, Any], strict_tolerance: float,
    quantization_plateau_relative_tolerance: float, nearest_realized_below: Optional[float] = None,
    nearest_realized_above: Optional[float] = None,
) -> Dict[str, Any]:
    realized_r = record.realized_relative_l2
    designed_r = record.designed_relative_l2
    absolute_error = abs(realized_r - r)
    relative_error = absolute_error / r if r > 0 else 0.0
    nb = nearest_realized_below if nearest_realized_below is not None else solver_result["nearest_realized_below"]
    na = nearest_realized_above if nearest_realized_above is not None else solver_result["nearest_realized_above"]
    attainable_gap = (na - nb) if (nb is not None and na is not None) else None

    return {
        "region": region_name,
        "seed": seed,
        "direction_seed": seed,
        "requested_relative_l2": r,
        "designed_relative_l2": designed_r,
        "designed_abs_error": abs(designed_r - r),
        "realized_relative_l2": realized_r,  # the ACTUAL realized value -- never the nominal requested value
        "absolute_radius_error": absolute_error,
        "relative_radius_error": relative_error,
        "realized_abs_error": absolute_error,  # backward-compat key name (matches v1/v2's own field)
        "radius_acceptance_mode": radius_acceptance_mode,
        "quantization_limited": quantization_limited,
        "nearest_realized_below": nb,
        "nearest_realized_above": na,
        "attainable_gap": attainable_gap,
        "accepted_scalar": accepted_scalar,
        "final_scale": record.scale,
        "solver_iterations": len(solver_result["attempts"]),
        "correction_iterations": len(solver_result["attempts"]),  # backward-compat key name
        "quantization_plateau": solver_result["quantization_plateau"],
        "strict_tolerance": strict_tolerance,
        "quantization_plateau_relative_tolerance": quantization_plateau_relative_tolerance,
        "radius_realization_method": QUANTIZATION_AWARE_METHOD_V3,
        "theta_l2_norm": record.theta_l2_norm,
        "raw_noise_l2_norm": record.raw_noise_l2_norm,
        "realized_epsilon_l2_norm": record.realized_epsilon_l2_norm,
        "region_param_count": record.param_count,
        "initial_realized_relative_l2": solver_result["attempts"][0]["realized_relative_l2"],
        "final_realized_relative_l2": realized_r,
        "final_absolute_radius_error": absolute_error,
        "attempts": solver_result["attempts"],
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
