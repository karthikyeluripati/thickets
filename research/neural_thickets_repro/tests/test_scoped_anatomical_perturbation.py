"""Tests for scoped_anatomical_perturbation.py's pure-logic dispatch helpers, exercised against
a fake worker (SimpleNamespace + a real synthetic model) -- same no-GPU-needed pattern as
tests/test_scope_isolation_gpu_check.py. The collective_rpc plumbing itself needs the pod.
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.scoped_anatomical_perturbation import (
    BF16_RADIUS_REALIZATION_METHOD,
    BF16_RADIUS_REALIZATION_METHOD_V2,
    MAX_BRACKET_EXPANSION_STEPS,
    QUANTIZATION_AWARE_METHOD_V3,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    CorrectionOutOfRegionDriftError,
    MAX_RADIUS_CORRECTION_ITERATIONS,
    MAX_RADIUS_SOLVER_ITERATIONS,
    QuantizationToleranceExceededError,
    RADIUS_REALIZATION_TOLERANCE,
    RadiusCorrectionFailedError,
    diag_full_model_drift,
    diag_region_drift,
    diag_snapshot_base,
    expand_bracket_and_resolve_bf16_radius,
    scoped_apply_anatomical_perturbation,
    scoped_apply_anatomical_perturbation_bf16_bracketed,
    scoped_apply_anatomical_perturbation_bf16_corrected,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
    select_quantization_limited_acceptance,
    solve_bf16_radius,
)
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas


def _fake_worker(model):
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    worker.reset_to_base_weights = lambda: None  # no-op: this test never perturbs before calling
    return worker


def test_scoped_apply_anatomical_perturbation_hits_requested_radius(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = dict(model.named_parameters())
    atlas = build_anatomy_atlas(list(named))
    region_names = atlas.region("language").param_names
    worker = _fake_worker(model)

    result = scoped_apply_anatomical_perturbation(worker, seed=42, r=0.05, region_name="language", region_param_names=region_names)

    assert result["realized_relative_l2"] == pytest.approx(0.05, abs=1e-6)
    assert result["requested_relative_l2"] == 0.05
    assert result["region"] == "language"
    assert result["region_param_count"] == len(region_names)


def test_scoped_apply_anatomical_perturbation_calls_reset_to_base_first(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = dict(model.named_parameters())
    atlas = build_anatomy_atlas(list(named))
    region_names = atlas.region("vision").param_names

    calls = []
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    worker.reset_to_base_weights = lambda: calls.append("reset")

    scoped_apply_anatomical_perturbation(worker, seed=1, r=0.01, region_name="vision", region_param_names=region_names)
    assert calls == ["reset"]


def test_scoped_apply_anatomical_perturbation_leaves_outside_region_untouched(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named_before = {k: v.detach().clone() for k, v in model.named_parameters()}
    atlas = build_anatomy_atlas(list(named_before))
    region_names = set(atlas.region("multimodal_connector_or_merger").param_names)
    worker = _fake_worker(model)

    scoped_apply_anatomical_perturbation(worker, seed=7, r=0.02, region_name="multimodal_connector_or_merger", region_param_names=list(region_names))

    for name, p in model.named_parameters():
        if name not in region_names:
            assert torch.equal(p.detach(), named_before[name]), f"{name} changed outside its region"


def test_diag_snapshot_base_and_full_model_drift_zero_when_unchanged(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    msg = diag_snapshot_base(worker)
    assert "snapshotted" in msg
    drift = diag_full_model_drift(worker)
    assert drift["max_abs_drift"] == 0.0


def test_diag_full_model_drift_requires_snapshot_first(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    with pytest.raises(RuntimeError, match="diag_snapshot_base was never called"):
        diag_full_model_drift(worker)


def test_diag_region_drift_detects_in_region_change_and_out_of_region_unchanged(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_names = atlas.region("vision").param_names

    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    diag_snapshot_base(worker)

    with torch.no_grad():
        next(model.visual.patch_embed.parameters()).add_(1.0)

    drift = diag_region_drift(worker, region_names)
    # measure_drift's chunked float64 accumulation (this repair pass -- see thicket/
    # memory_bounded_ops.py) is STRICTLY MORE precise than the float32 diff it replaces, so it
    # can reveal a genuine ~2^-24 (~5.96e-8) bf16-rounding discrepancy that float32 subtraction
    # happened to round away to exactly 1.0 via round-half-to-even -- pytest.approx with a
    # tolerance far above that discrepancy (and far below any real drift signal) is the correct
    # check, not exact equality against a value float32 only reached by coincidental rounding.
    assert drift["in_region"]["max_abs_drift"] == pytest.approx(1.0, abs=1e-6)
    assert drift["out_of_region"]["max_abs_drift"] == 0.0
    assert drift["region_param_count"] == len(region_names)


def test_diag_region_drift_requires_snapshot_first(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    with pytest.raises(RuntimeError, match="diag_snapshot_base was never called"):
        diag_region_drift(worker, ["some.name"])


# --- Bridges anatomy.py's L1 region vocabulary to scopes.py's PERTURBATION_SCOPES vocabulary,
# proving the two independently-named registries select IDENTICAL parameter sets -------------


@pytest.mark.parametrize(
    "anatomy_region,scope_name",
    [("vision", "vision_encoder"), ("multimodal_connector_or_merger", "vision_merger"), ("language", "full_lm")],
)
def test_anatomy_region_matches_equivalent_scopes_registry_scope(anatomy_region, scope_name, runtime_wrapped_vlm_32vision_factory):
    from neural_thickets_repro.scopes import build_scope_manifest

    model = runtime_wrapped_vlm_32vision_factory()
    named_parameters = list(model.named_parameters())
    atlas = build_anatomy_atlas([n for n, _ in named_parameters])

    anatomy_names = set(atlas.region(anatomy_region).param_names)
    scope_names = set(build_scope_manifest(scope_name, named_parameters).selected_param_names)
    assert anatomy_names == scope_names


# =================================================================================================
# BF16 realized-radius correction (RunPod live-failure regression): a real smoke candidate
# (region=vision, r=0.035698828543799424) hard-failed with realized r=0.03569534313727009 (abs
# error 3.485e-06) because the ORIGINAL apply_anatomical_relative_l2 measured its own
# "realized_epsilon_l2_norm" from the additive delta tensor BEFORE the in-place bf16 `p.add_()`,
# not from the true post-addition weight change -- proven directly below with real bf16 tensors,
# not merely asserted.
# =================================================================================================

REQUESTED_R = 0.035698828543799424  # the exact live-failure radius (FULL_CALIBRATION_RADII[2])
DIRECTION_SEED = 12345


class _TwoTensorBF16Model(torch.nn.Module):
    """`region_layer.weight` is the perturbed region; `outside_layer.weight` stands in for
    every parameter outside it -- real bf16 CPU tensors (torch supports bf16 arithmetic on
    CPU), so this exercises the ACTUAL rounding behavior the RunPod failure came from, not a
    simulation of it.
    """

    def __init__(self, region_elements: int, outside_elements: int = 100):
        super().__init__()
        self.region_layer = torch.nn.Linear(region_elements, 1, bias=False)
        self.outside_layer = torch.nn.Linear(outside_elements, 1, bias=False)


def _bf16_worker(region_elements: int, outside_elements: int = 100, init_seed: int = 0):
    """A region large enough (500,000 elements) for bf16 quantization noise to average out
    across the L2 norm converges within MAX_RADIUS_CORRECTION_ITERATIONS (confirmed directly
    below); a small region (2,000 elements) does NOT, at REQUESTED_R/DIRECTION_SEED -- both
    confirmed empirically, not assumed, and used as the "converges" vs "genuine plateau, hard
    -fails" test cases respectively.
    """
    torch.manual_seed(init_seed)
    model = _TwoTensorBF16Model(region_elements, outside_elements).to(torch.bfloat16)
    base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    reset_calls = {"count": 0}

    def _reset():
        reset_calls["count"] += 1
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(base_weights[name])

    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    worker.reset_to_base_weights = _reset
    worker._base_weights = base_weights
    return worker, model, base_weights, reset_calls


def test_bf16_correction_constants_are_frozen():
    assert MAX_RADIUS_CORRECTION_ITERATIONS == 5
    assert RADIUS_REALIZATION_TOLERANCE == 1e-6
    assert BF16_RADIUS_REALIZATION_METHOD == "fixed_direction_bf16_corrected_v1"


def test_bf16_correction_one_shot_misses_tolerance_but_correction_converges():
    """Reproduces the live failure mode directly: the FIRST attempt (one-shot, uncorrected)
    misses the 1e-6 tolerance, and the correction loop brings it within tolerance in more than
    one attempt (never claiming a false one-shot success).
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert len(result["attempts"]) > 1, "expected the one-shot attempt to miss tolerance, requiring correction"
    assert result["attempts"][0]["realized_abs_error"] > RADIUS_REALIZATION_TOLERANCE
    assert result["realized_abs_error"] <= RADIUS_REALIZATION_TOLERANCE
    assert result["radius_realization_method"] == BF16_RADIUS_REALIZATION_METHOD
    assert result["correction_iterations"] == len(result["attempts"])
    assert result["correction_iterations"] <= MAX_RADIUS_CORRECTION_ITERATIONS


def test_bf16_correction_preserves_fixed_direction_only_scale_changes():
    """The underlying seeded noise is regenerated fresh every attempt but is bit-identical
    (same seed/shape/dtype/device) -- proven directly, not assumed -- and every attempt's SCALE
    differs (that is the only thing the correction loop is allowed to change).
    """
    from neural_thickets_repro.perturb_cpu import _generate_noise

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert len(result["attempts"]) > 1

    noise_a = _generate_noise(base_weights["region_layer.weight"], DIRECTION_SEED)
    noise_b = _generate_noise(base_weights["region_layer.weight"], DIRECTION_SEED)
    assert torch.equal(noise_a, noise_b), "the seeded direction must be bit-identical across regenerations"

    scales = [a["scale"] for a in result["attempts"]]
    assert len(set(scales)) == len(scales), "every correction attempt must use a distinct scalar magnitude"


def test_bf16_correction_final_weights_correspond_to_the_accepted_radius():
    """The ACTUAL final model weights (not merely the reported numbers) correspond to the
    accepted radius -- re-measured independently from model state after the call returns.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    theta_before = base_weights["region_layer.weight"].float()
    theta_after = model.region_layer.weight.detach().float()
    realized_delta_l2 = (theta_after - theta_before).pow(2).sum().sqrt().item()
    # Uses the SAME reported theta_l2_norm (not an independently-recomputed one, which can
    # differ from the internal computation at the ~1e-9 level from summation-order alone) --
    # isolates the check to "does the actual weight delta match what was reported", which is
    # the property this test exists to prove.
    independently_measured_ratio = realized_delta_l2 / result["theta_l2_norm"]

    assert independently_measured_ratio == pytest.approx(result["realized_relative_l2"], abs=1e-9)
    assert abs(independently_measured_ratio - REQUESTED_R) <= RADIUS_REALIZATION_TOLERANCE


def test_bf16_correction_outside_region_bitwise_unchanged():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert torch.equal(model.outside_layer.weight.detach(), base_weights["outside_layer.weight"])


def test_bf16_correction_resets_base_before_every_attempt():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert reset_calls["count"] == result["correction_iterations"]


def test_bf16_correction_never_evaluates_capabilities():
    import inspect

    source = inspect.getsource(scoped_apply_anatomical_perturbation_bf16_corrected)
    for forbidden in ("run_benchmark", "CapabilityContext", "aggregate_metrics", "SamplingParams"):
        assert forbidden not in source


def test_bf16_correction_fails_with_evidence_on_a_genuine_plateau():
    """A small enough region (2,000 elements) does not converge within MAX_RADIUS_CORRECTION_
    ITERATIONS at REQUESTED_R/DIRECTION_SEED (confirmed directly, not assumed) -- must hard-fail
    with the full attempt history as evidence, never silently accept a looser tolerance.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)
    region_names = ["region_layer.weight"]

    with pytest.raises(RadiusCorrectionFailedError) as exc_info:
        scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert "Attempts:" in str(exc_info.value)
    assert str(MAX_RADIUS_CORRECTION_ITERATIONS) in str(exc_info.value)


def test_bf16_correction_failed_convergence_still_resets_outside_region_and_reset_count_bounded():
    """Even on a hard-fail, every attempt made (up to the max) still respected the
    out-of-region invariant and the reset-before-every-attempt discipline.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)
    region_names = ["region_layer.weight"]

    with pytest.raises(RadiusCorrectionFailedError):
        scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert reset_calls["count"] == MAX_RADIUS_CORRECTION_ITERATIONS
    assert torch.equal(model.outside_layer.weight.detach(), base_weights["outside_layer.weight"])


def test_bf16_correction_same_seed_produces_the_same_corrected_result():
    worker_1, model_1, base_weights_1, _ = _bf16_worker(region_elements=500_000)
    worker_2, model_2, base_weights_2, _ = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result_1 = scoped_apply_anatomical_perturbation_bf16_corrected(worker_1, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    result_2 = scoped_apply_anatomical_perturbation_bf16_corrected(worker_2, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert result_1["correction_iterations"] == result_2["correction_iterations"]
    assert result_1["final_scale"] == result_2["final_scale"]
    assert result_1["final_realized_relative_l2"] == result_2["final_realized_relative_l2"]
    assert torch.equal(model_1.region_layer.weight.detach(), model_2.region_layer.weight.detach())


def test_bf16_correction_requires_base_weights_stored():
    torch.manual_seed(0)
    model = _TwoTensorBF16Model(1000).to(torch.bfloat16)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))  # no _base_weights
    with pytest.raises(RuntimeError, match="store_base_weights"):
        scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", ["region_layer.weight"])


def test_bf16_correction_hard_fails_on_out_of_region_drift(monkeypatch):
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    def _broken_measure_drift(model_arg, base_state, param_filter=None):
        return {"max_abs_drift": 0.5, "mean_abs_drift": 0.1, "relative_norm_drift": 0.1, "fraction_elements_differing": 0.01}

    monkeypatch.setattr(module, "measure_drift", _broken_measure_drift)

    with pytest.raises(CorrectionOutOfRegionDriftError):
        scoped_apply_anatomical_perturbation_bf16_corrected(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)


# =================================================================================================
# solve_bf16_radius: pure control-flow solver (no GPU/model access) -- the bracketed/bisection
# replacement for v1's proportional-only correction, which was observed to oscillate on a real
# RunPod full-calibration run without converging (see module docstring for the live sequence).
# =================================================================================================


def test_solve_bf16_radius_constants_are_frozen():
    assert MAX_RADIUS_SOLVER_ITERATIONS == 20
    assert BF16_RADIUS_REALIZATION_METHOD_V2 == "fixed_direction_bf16_bracketed_v2"


def _scripted_evaluate_fn(sequence):
    it = iter(sequence)

    def _evaluate(trial_r):
        v = next(it)
        return {"realized_relative_l2": v, "designed_relative_l2": v}

    return _evaluate


# The EXACT live-failure sequence reported from the real RunPod full-calibration attempt
# (region=vision, requested r=0.0035698828543799426): 5 proportional-only attempts oscillating
# overshoot/undershoot around the target, closest (attempt 4) still 1.37e-6 > the 1e-6 tolerance.
LIVE_OSCILLATING_SEQUENCE = [
    0.003713521124129136,
    0.003573362306920378,
    0.0035728117233154925,
    0.0035712537875961575,
    0.003566903799826982,
]
LIVE_REQUESTED_R = 0.0035698828543799426


def test_solve_bf16_radius_v1_style_proportional_alone_would_have_failed_these_5():
    """Sanity-checks the fixture itself: none of the 5 real observed values is within 1e-6 of
    the target -- this is genuinely the failure v1 hit, not a fabricated scenario.
    """
    for v in LIVE_OSCILLATING_SEQUENCE:
        assert abs(v - LIVE_REQUESTED_R) > RADIUS_REALIZATION_TOLERANCE


def test_solve_bf16_radius_observed_sequence_converges_once_an_attainable_point_exists():
    """Replays the exact 5 live observations, then injects a 6th trial landing exactly on the
    target -- proves the solver (unlike v1) keeps searching via bisection once a bracket forms
    (after attempt 5's undershoot) instead of giving up, and accepts as soon as an attainable
    point is found.
    """
    sequence = LIVE_OSCILLATING_SEQUENCE + [LIVE_REQUESTED_R]
    result = solve_bf16_radius(_scripted_evaluate_fn(sequence), LIVE_REQUESTED_R, max_iterations=20, tolerance=1e-6)

    assert result["converged"] is True
    assert result["quantization_plateau"] is False
    assert len(result["attempts"]) == 6
    assert result["best_realized_relative_l2"] == LIVE_REQUESTED_R
    assert result["best_absolute_error"] == 0.0


def test_solve_bf16_radius_does_not_simply_oscillate_forever():
    """After the 5 live observations (last one an undershoot, forming a bracket against attempt
    4's overshoot), the 6th trial's scalar must be a BISECTION midpoint of the bracket -- NOT
    the proportional formula v1 used (which is exactly what caused the observed oscillation).
    """
    sequence = LIVE_OSCILLATING_SEQUENCE + [LIVE_REQUESTED_R]
    result = solve_bf16_radius(_scripted_evaluate_fn(sequence), LIVE_REQUESTED_R, max_iterations=20, tolerance=1e-6)

    attempt_5 = result["attempts"][4]
    attempt_6 = result["attempts"][5]
    assert attempt_5["bracket_low_scale"] is not None and attempt_5["bracket_high_scale"] is not None
    expected_bisection_midpoint = (attempt_5["bracket_low_scale"] + attempt_5["bracket_high_scale"]) / 2.0
    assert attempt_6["scalar"] == expected_bisection_midpoint
    proportional_next = attempt_5["scalar"] * LIVE_REQUESTED_R / attempt_5["realized_relative_l2"]
    assert attempt_6["scalar"] != proportional_next


def test_solve_bf16_radius_bracket_has_realizations_on_opposite_sides_of_target():
    sequence = LIVE_OSCILLATING_SEQUENCE + [LIVE_REQUESTED_R]
    result = solve_bf16_radius(_scripted_evaluate_fn(sequence), LIVE_REQUESTED_R, max_iterations=20, tolerance=1e-6)

    last_with_bracket = result["attempts"][4]  # attempt 5, the first one with both sides set
    assert last_with_bracket["bracket_low_realized"] <= LIVE_REQUESTED_R
    assert last_with_bracket["bracket_high_realized"] >= LIVE_REQUESTED_R


def test_solve_bf16_radius_bracket_shrinks_deterministically():
    """A smooth, purely synthetic monotonic evaluate_fn (realized = r, i.e. a perfect
    zero-noise oracle) -- proves the bracket's scalar width strictly halves once bisection
    begins, converging to within tolerance in a bounded, predictable number of steps.
    """
    requested = 0.1

    def _evaluate(trial_r):
        # deliberately offset so the first two trials straddle the target before bisection begins
        return {"realized_relative_l2": trial_r, "designed_relative_l2": trial_r}

    # seed two straddling observations manually via a wrapper sequence, then let pure bisection run
    sequence = [0.2, 0.05]  # trial1: overshoot (proportional next), trial2: undershoot -> bracket forms
    scripted = _scripted_evaluate_fn(sequence)

    def _combined(trial_r):
        try:
            return scripted(trial_r)
        except StopIteration:
            return _evaluate(trial_r)

    # tolerance chosen so pure bisection (halving a 0.15-wide bracket) provably converges well
    # within max_iterations=20: 0.15 / 2**12 ~= 3.7e-5 < 1e-4, i.e. ~12 halvings needed.
    result = solve_bf16_radius(_combined, requested, max_iterations=20, tolerance=1e-4)

    widths = []
    prev_low = prev_high = None
    for a in result["attempts"]:
        if a["bracket_low_scale"] is not None and a["bracket_high_scale"] is not None:
            widths.append(a["bracket_high_scale"] - a["bracket_low_scale"])
    # widths must be non-increasing once the bracket exists (never widens)
    assert all(widths[i] >= widths[i + 1] - 1e-15 for i in range(len(widths) - 1))
    assert result["converged"] is True


def test_solve_bf16_radius_detects_plateau_when_no_attainable_point_within_tolerance():
    requested = 0.01
    x_high, x_low = 0.0100050, 0.0099950  # both 5e-6 away from target -- neither within 1e-6
    sequence = [x_high, x_low, x_low]  # 3rd trial repeats the 2nd's exact realized value -> plateau

    result = solve_bf16_radius(_scripted_evaluate_fn(sequence), requested, max_iterations=20, tolerance=1e-6)

    assert result["converged"] is False
    assert result["quantization_plateau"] is True
    assert result["nearest_realized_below"] == x_low
    assert result["nearest_realized_above"] == x_high
    assert result["best_absolute_error"] == pytest.approx(5e-6)
    assert len(result["attempts"]) == 3  # stops promptly, does not burn all 20 iterations


def test_solve_bf16_radius_tracks_best_so_far_correctly_even_when_last_attempt_is_worse():
    """Mirrors the live sequence's own shape: attempt 4 (overshoot, error 1.37e-6) is closer to
    target than attempt 5 (undershoot, error 2.98e-6) -- best-so-far must report attempt 4, not
    simply the most recent attempt.
    """
    result = solve_bf16_radius(_scripted_evaluate_fn(LIVE_OSCILLATING_SEQUENCE), LIVE_REQUESTED_R, max_iterations=5, tolerance=1e-6)

    assert result["converged"] is False  # v1's exact failure -- 5 attempts, none within tolerance
    assert result["best_iteration"] == 4
    assert result["best_realized_relative_l2"] == LIVE_OSCILLATING_SEQUENCE[3]
    assert result["best_absolute_error"] == pytest.approx(abs(LIVE_OSCILLATING_SEQUENCE[3] - LIVE_REQUESTED_R))
    assert result["best_absolute_error"] == min(abs(v - LIVE_REQUESTED_R) for v in LIVE_OSCILLATING_SEQUENCE)


def test_solve_bf16_radius_converges_immediately_when_first_trial_within_tolerance():
    sequence = [LIVE_REQUESTED_R]
    result = solve_bf16_radius(_scripted_evaluate_fn(sequence), LIVE_REQUESTED_R, max_iterations=20, tolerance=1e-6)
    assert result["converged"] is True
    assert len(result["attempts"]) == 1
    assert result["accepted_scalar"] == LIVE_REQUESTED_R


# =================================================================================================
# scoped_apply_anatomical_perturbation_bf16_bracketed: the GPU-facing v2 wrapper, real bf16
# tensors -- same _bf16_worker fixture already established for v1's tests above.
# =================================================================================================


def test_bracketed_v2_one_shot_misses_but_solver_converges():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert result["realized_abs_error"] <= RADIUS_REALIZATION_TOLERANCE
    assert result["radius_realization_method"] == BF16_RADIUS_REALIZATION_METHOD_V2
    assert result["quantization_plateau"] is False
    assert result["solver_iterations"] <= MAX_RADIUS_SOLVER_ITERATIONS


def test_bracketed_v2_preserves_fixed_direction_only_scale_changes():
    from neural_thickets_repro.perturb_cpu import _generate_noise

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    noise_a = _generate_noise(base_weights["region_layer.weight"], DIRECTION_SEED)
    noise_b = _generate_noise(base_weights["region_layer.weight"], DIRECTION_SEED)
    assert torch.equal(noise_a, noise_b)

    if len(result["attempts"]) > 1:
        scalars = [a["scalar"] for a in result["attempts"]]
        assert len(set(scalars)) == len(scalars)


def test_bracketed_v2_final_weights_correspond_to_the_accepted_radius():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    theta_before = base_weights["region_layer.weight"].float()
    theta_after = model.region_layer.weight.detach().float()
    realized_delta_l2 = (theta_after - theta_before).pow(2).sum().sqrt().item()
    independently_measured_ratio = realized_delta_l2 / result["theta_l2_norm"]

    assert independently_measured_ratio == pytest.approx(result["realized_relative_l2"], abs=1e-9)
    assert abs(independently_measured_ratio - REQUESTED_R) <= RADIUS_REALIZATION_TOLERANCE


def test_bracketed_v2_outside_region_bitwise_unchanged():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert torch.equal(model.outside_layer.weight.detach(), base_weights["outside_layer.weight"])


def test_bracketed_v2_resets_before_every_trial():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert reset_calls["count"] == len(result["attempts"])


def test_bracketed_v2_never_evaluates_capabilities():
    import inspect

    source = inspect.getsource(scoped_apply_anatomical_perturbation_bf16_bracketed)
    for forbidden in ("run_benchmark", "CapabilityContext", "aggregate_metrics", "SamplingParams"):
        assert forbidden not in source
    source = inspect.getsource(solve_bf16_radius)
    for forbidden in ("run_benchmark", "CapabilityContext", "aggregate_metrics", "SamplingParams"):
        assert forbidden not in source


def test_bracketed_v2_fails_with_evidence_on_a_genuine_plateau():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)
    region_names = ["region_layer.weight"]

    with pytest.raises(RadiusCorrectionFailedError) as exc_info:
        scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    message = str(exc_info.value)
    assert "quantization_plateau" in message
    assert "nearest_realized_below" in message
    assert "nearest_realized_above" in message


def test_bracketed_v2_same_seed_produces_the_same_result():
    worker_1, model_1, base_weights_1, _ = _bf16_worker(region_elements=500_000)
    worker_2, model_2, base_weights_2, _ = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result_1 = scoped_apply_anatomical_perturbation_bf16_bracketed(worker_1, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    result_2 = scoped_apply_anatomical_perturbation_bf16_bracketed(worker_2, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert result_1["solver_iterations"] == result_2["solver_iterations"]
    assert result_1["final_realized_relative_l2"] == result_2["final_realized_relative_l2"]
    assert torch.equal(model_1.region_layer.weight.detach(), model_2.region_layer.weight.detach())


def test_bracketed_v2_requires_base_weights_stored():
    torch.manual_seed(0)
    model = _TwoTensorBF16Model(1000).to(torch.bfloat16)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))  # no _base_weights
    with pytest.raises(RuntimeError, match="store_base_weights"):
        scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", ["region_layer.weight"])


def test_bracketed_v2_hard_fails_on_out_of_region_drift(monkeypatch):
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    def _broken_measure_drift(model_arg, base_state, param_filter=None):
        return {"max_abs_drift": 0.5, "mean_abs_drift": 0.1, "relative_norm_drift": 0.1, "fraction_elements_differing": 0.01}

    monkeypatch.setattr(module, "measure_drift", _broken_measure_drift)

    with pytest.raises(CorrectionOutOfRegionDriftError):
        scoped_apply_anatomical_perturbation_bf16_bracketed(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)


# --- v1 vs v2 method distinction (checkpoint/run isolation lives in run_stage7b_anatomical_
# calibration.py's own tests; this confirms the two constants this module exposes are distinct) -


def test_v1_and_v2_realization_methods_are_distinct_constants():
    assert BF16_RADIUS_REALIZATION_METHOD == "fixed_direction_bf16_corrected_v1"
    assert BF16_RADIUS_REALIZATION_METHOD_V2 == "fixed_direction_bf16_bracketed_v2"
    assert BF16_RADIUS_REALIZATION_METHOD != BF16_RADIUS_REALIZATION_METHOD_V2


# =================================================================================================
# select_quantization_limited_acceptance: pure decision arithmetic -- exercised directly against
# the REAL connector numbers from the Stage-7B three-region smallest-radius numerical smoke.
# =================================================================================================

# The exact live evidence from the v2 smoke run (region=connector, smallest frozen radius).
CONNECTOR_REQUESTED_R = 0.0035698828543799426
CONNECTOR_NEAREST_BELOW = 0.0035686268537125777
CONNECTOR_NEAREST_ABOVE = 0.0035711468798464217


def test_quantization_aware_constants_are_frozen():
    assert QUANTIZATION_AWARE_METHOD_V3 == "fixed_direction_bf16_quantization_aware_v3"
    assert QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE == 1e-3


def test_select_quantization_limited_acceptance_reproduces_the_exact_connector_evidence():
    """The exact numbers reported from the live Stage-7B connector-region smoke: nearest-below
    is closer (abs error 1.256e-6) than nearest-above (abs error 1.264e-6), relative error
    ~=3.52e-4 (0.0352%) -- comfortably inside the 0.1% bound, so accepted.
    """
    result = select_quantization_limited_acceptance(CONNECTOR_NEAREST_BELOW, CONNECTOR_NEAREST_ABOVE, CONNECTOR_REQUESTED_R)

    assert result["which"] == "below"
    assert result["nearest_realized"] == CONNECTOR_NEAREST_BELOW
    assert result["absolute_error"] == pytest.approx(1.2560006673648615e-06)
    assert result["relative_error"] == pytest.approx(0.00035183246022312903)
    assert result["accepted"] is True


def test_select_quantization_limited_acceptance_picks_the_closer_side():
    # above closer than below
    result = select_quantization_limited_acceptance(0.0090, 0.0101, 0.0100)
    assert result["which"] == "above"
    assert result["nearest_realized"] == 0.0101


def test_select_quantization_limited_acceptance_rejects_when_relative_error_exceeds_tolerance():
    # both 5e-3 away from a requested r=1.0 -> relative error 0.5%, well above 0.1%
    result = select_quantization_limited_acceptance(0.995, 1.005, 1.0)
    assert result["accepted"] is False
    assert result["relative_error"] == pytest.approx(0.005)


def test_select_quantization_limited_acceptance_accepts_at_exactly_the_boundary():
    # Uses the SAME floating-point computation for the tolerance as for the relative error
    # itself (rather than a separately-typed "1e-3" literal, which can differ from the actual
    # computed value by a representation-level ULP) -- proves the boundary is inclusive ("<=")
    # without floating-point-equality flakiness.
    below, above, requested = 0.999, 1.002, 1.0
    relative_error_at_boundary = abs(below - requested) / requested
    result = select_quantization_limited_acceptance(below, above, requested, relative_tolerance=relative_error_at_boundary)
    assert result["which"] == "below"
    assert result["relative_error"] == relative_error_at_boundary
    assert result["accepted"] is True


# =================================================================================================
# scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3: the GPU-facing v3 wrapper,
# real bf16 tensors -- reuses the _bf16_worker fixture already established for v1/v2's tests.
# =================================================================================================


def test_v3_strict_path_accepted_normally_when_it_converges():
    """A region large enough to converge strictly (same 500,000-element / DIRECTION_SEED case
    v2's own tests already confirmed converges) -- v3 must accept it as "strict", not fall
    through to the quantization-limited path at all.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert result["radius_acceptance_mode"] == "strict"
    assert result["quantization_limited"] is False
    assert result["realized_abs_error"] <= RADIUS_REALIZATION_TOLERANCE
    assert result["radius_realization_method"] == QUANTIZATION_AWARE_METHOD_V3
    assert result["quantization_plateau"] is False


def test_v3_quantization_limited_path_accepts_nearest_attainable_within_tolerance():
    """A region (2,000 elements) confirmed directly to plateau with relative error ~=1.05e-4
    (well within 0.1%) at REQUESTED_R/DIRECTION_SEED -- v2 would have hard-failed this outright;
    v3 must accept it via the quantization-limited fallback, and the REPORTED realized value
    must be the actual nearest attainable state, never the nominal requested value.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert result["radius_acceptance_mode"] == "quantization_limited"
    assert result["quantization_limited"] is True
    assert result["quantization_plateau"] is True
    assert result["relative_radius_error"] <= QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE
    assert result["realized_relative_l2"] != REQUESTED_R  # the ACTUAL realized value, never the nominal requested one
    assert result["realized_relative_l2"] in (result["nearest_realized_below"], result["nearest_realized_above"])
    assert result["nearest_realized_below"] is not None
    assert result["nearest_realized_above"] is not None
    assert result["attainable_gap"] == pytest.approx(result["nearest_realized_above"] - result["nearest_realized_below"])


def test_v3_hard_fails_when_relative_error_exceeds_tolerance_even_with_proven_plateau():
    """A region (100 elements, a specific seed) confirmed directly to plateau with relative
    error ~=0.28% -- exceeds the 0.1% admissibility bound, so v3 must refuse the fallback
    (QuantizationToleranceExceededError), never silently accept a looser bound.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=100)

    with pytest.raises(QuantizationToleranceExceededError):
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, 1, REQUESTED_R, "region", ["region_layer.weight"])


def test_v3_quantization_tolerance_exceeded_is_a_radius_correction_failed_error():
    """QuantizationToleranceExceededError subclasses RadiusCorrectionFailedError so any caller
    only catching the base class still sees it.
    """
    assert issubclass(QuantizationToleranceExceededError, RadiusCorrectionFailedError)


def test_v3_hard_fails_when_no_plateau_is_proven_and_strict_never_converges(monkeypatch):
    """If solve_bf16_radius neither converges NOR proves a plateau (e.g. genuinely exhausted
    max_iterations without ever forming a bracket) -- refuses any fallback, since the
    quantization-limited path requires a PROVEN plateau, not merely "gave up".
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    def _fake_solve(evaluate_fn, r, max_iterations, tolerance):
        evaluate_fn(r)  # still perturbs once, so a real record exists (mirrors a genuine trial)
        return {
            "converged": False, "quantization_plateau": False, "attempts": [{"iteration": 1, "scalar": r, "realized_relative_l2": r * 2, "absolute_error": r}],
            "best_iteration": 1, "best_scalar": r, "best_designed_relative_l2": r * 2, "best_realized_relative_l2": r * 2,
            "best_absolute_error": r, "accepted_scalar": None, "nearest_realized_below": None, "nearest_realized_above": None,
        }

    monkeypatch.setattr(module, "solve_bf16_radius", _fake_solve)

    with pytest.raises(RadiusCorrectionFailedError) as exc_info:
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert not isinstance(exc_info.value, QuantizationToleranceExceededError)


def test_v3_hard_fails_when_plateau_claimed_but_bracket_incomplete(monkeypatch):
    """A (hypothetical, monkeypatched) plateau report missing one side of the bracket must
    refuse the fallback rather than guessing -- the real solve_bf16_radius never actually
    produces this combination, but the acceptance layer must not trust it blindly regardless.
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    def _fake_solve(evaluate_fn, r, max_iterations, tolerance):
        evaluate_fn(r)
        return {
            "converged": False, "quantization_plateau": True,
            "attempts": [{"iteration": 1, "scalar": r, "realized_relative_l2": r * 1.5, "absolute_error": r * 0.5, "bracket_low_scale": None, "bracket_high_scale": r}],
            "best_iteration": 1, "best_scalar": r, "best_designed_relative_l2": r * 1.5, "best_realized_relative_l2": r * 1.5,
            "best_absolute_error": r * 0.5, "accepted_scalar": None, "nearest_realized_below": None, "nearest_realized_above": r * 1.5,
        }

    monkeypatch.setattr(module, "solve_bf16_radius", _fake_solve)

    with pytest.raises(RadiusCorrectionFailedError):
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)


def test_v3_preserves_fixed_direction_only_scale_changes():
    from neural_thickets_repro.perturb_cpu import _generate_noise

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)  # exercises the fallback path
    region_names = ["region_layer.weight"]

    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    noise_a = _generate_noise(base_weights["region_layer.weight"], DIRECTION_SEED)
    noise_b = _generate_noise(base_weights["region_layer.weight"], DIRECTION_SEED)
    assert torch.equal(noise_a, noise_b)


def test_v3_reconstructed_accepted_state_matches_selected_plateau_state():
    """For the quantization-limited fallback specifically: the ACTUAL final model weights
    (re-measured independently after the call returns) correspond exactly to the selected
    nearest-attainable state, not merely to whatever the solver's search happened to leave
    loaded -- proves the explicit reset/reapply/remeasure sequence (section 3) actually ran.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert result["radius_acceptance_mode"] == "quantization_limited"

    theta_before = base_weights["region_layer.weight"].float()
    theta_after = model.region_layer.weight.detach().float()
    realized_delta_l2 = (theta_after - theta_before).pow(2).sum().sqrt().item()
    independently_measured_ratio = realized_delta_l2 / result["theta_l2_norm"]

    assert independently_measured_ratio == pytest.approx(result["realized_relative_l2"], abs=1e-9)


def test_v3_outside_region_bitwise_unchanged_strict_path():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert torch.equal(model.outside_layer.weight.detach(), base_weights["outside_layer.weight"])


def test_v3_outside_region_bitwise_unchanged_quantization_limited_path():
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)
    region_names = ["region_layer.weight"]
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert torch.equal(model.outside_layer.weight.detach(), base_weights["outside_layer.weight"])


def test_v3_never_evaluates_capabilities():
    import inspect

    source = inspect.getsource(scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3)
    for forbidden in ("run_benchmark", "CapabilityContext", "aggregate_metrics", "SamplingParams"):
        assert forbidden not in source


def test_v3_same_seed_produces_the_same_result_strict_and_fallback():
    for region_elements in (500_000, 2_000):
        worker_1, model_1, base_weights_1, _ = _bf16_worker(region_elements=region_elements)
        worker_2, model_2, base_weights_2, _ = _bf16_worker(region_elements=region_elements)
        region_names = ["region_layer.weight"]

        result_1 = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_1, DIRECTION_SEED, REQUESTED_R, "region", region_names)
        result_2 = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_2, DIRECTION_SEED, REQUESTED_R, "region", region_names)

        assert result_1["radius_acceptance_mode"] == result_2["radius_acceptance_mode"]
        assert result_1["realized_relative_l2"] == result_2["realized_relative_l2"]
        assert torch.equal(model_1.region_layer.weight.detach(), model_2.region_layer.weight.detach())


def test_v3_requires_base_weights_stored():
    torch.manual_seed(0)
    model = _TwoTensorBF16Model(1000).to(torch.bfloat16)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    with pytest.raises(RuntimeError, match="store_base_weights"):
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", ["region_layer.weight"])


def test_v3_hard_fails_on_out_of_region_drift_during_trials(monkeypatch):
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    def _broken_measure_drift(model_arg, base_state, param_filter=None):
        return {"max_abs_drift": 0.5, "mean_abs_drift": 0.1, "relative_norm_drift": 0.1, "fraction_elements_differing": 0.01}

    monkeypatch.setattr(module, "measure_drift", _broken_measure_drift)

    with pytest.raises(CorrectionOutOfRegionDriftError):
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)


def test_v3_hard_fails_on_out_of_region_drift_on_the_fallback_path_too(monkeypatch):
    """Section 4's out-of-region invariant applies to the FALLBACK path (a region that would
    otherwise take the quantization-limited route) exactly as it does to the strict path --
    forcing measure_drift to always report a leak must still raise, whichever path is taken.
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)  # would otherwise take the fallback path
    region_names = ["region_layer.weight"]

    def _always_broken(model_arg, base_state, param_filter=None):
        return {"max_abs_drift": 0.5, "mean_abs_drift": 0.1, "relative_norm_drift": 0.1, "fraction_elements_differing": 0.01}

    monkeypatch.setattr(module, "measure_drift", _always_broken)

    with pytest.raises(CorrectionOutOfRegionDriftError):
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)


# =================================================================================================
# POST-V3-FAILURE DETERMINISTIC BRACKET EXPANSION (this repair pass): a real Stage-9 full run
# (1129/1152 perturbations already checkpointed) hard-failed with RadiusCorrectionFailedError on
# region=language_late, seed=980336641146292533, requested r=0.07139765708759885 -- ALL 20
# original attempts realized the IDENTICAL value 0.07139927430659475 regardless of trial scalar
# (a bf16 quantization staircase that never stepped below the target within the original
# solver's tiny proportional-correction neighborhood), so no bracket ever formed and v3 refused
# any fallback. expand_bracket_and_resolve_bf16_radius (see module docstring) activates ONLY in
# that exact "no bracket ever formed" branch, deterministically searching farther via geometric
# expansion before giving up, then handing off to the SAME bisection/plateau logic
# solve_bf16_radius itself uses (never a re-derived acceptance rule).
# =================================================================================================

LANGUAGE_LATE_REQUESTED_R = 0.07139765708759885  # the exact live Stage-9 full-run failure radius
LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU = 0.07139927430659475  # the exact live-observed constant realized value


def test_bracket_expansion_constants_are_frozen():
    assert MAX_BRACKET_EXPANSION_STEPS == 24


def test_bracket_expansion_regression_reproduces_the_exact_live_language_late_failure_shape():
    """Sanity-checks the fixture itself against the literal reported numbers: all 20 attempts
    realize the SAME value, none within tolerance, and the reported absolute error matches.
    """
    assert abs(LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU - LANGUAGE_LATE_REQUESTED_R) == pytest.approx(1.6172189959001715e-06)
    assert abs(LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU - LANGUAGE_LATE_REQUESTED_R) > RADIUS_REALIZATION_TOLERANCE


def test_bracket_expansion_original_20_attempts_never_form_a_bracket_on_the_live_failure_shape():
    """Confirms the ORIGINAL 20-attempt solve_bf16_radius call (byte-for-byte unchanged) really
    does exhaust all 20 attempts with bracket_low never found and quantization_plateau False --
    exactly the live failure, not a fabricated scenario -- before any expansion logic runs.
    """

    def _always_high(trial_r):
        return {"realized_relative_l2": LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU, "designed_relative_l2": LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU}

    result = solve_bf16_radius(_always_high, LANGUAGE_LATE_REQUESTED_R, max_iterations=20, tolerance=1e-6)

    assert result["converged"] is False
    assert result["quantization_plateau"] is False
    assert len(result["attempts"]) == 20
    assert result["bracket_low_scale"] is None
    assert result["bracket_high_scale"] is not None
    for a in result["attempts"]:
        assert a["realized_relative_l2"] == LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU
        assert a["bracket_low_scale"] is None


def test_bracket_expansion_regression_finds_a_lower_plateau_and_accepts_quantization_limited():
    """Item 6's exact regression fixture: the original 20 attempts are stuck on the SAME
    (real, live) high value; a lower attainable BF16 state exists but only becomes visible below
    a crossover scalar reachable solely via bracket expansion (never within the original 20 tiny
    proportional steps, confirmed above). Must: (1) run the original 20 unchanged, (2) enter
    expansion, (3) discover the lower attainable value, (4) prove the bracket via the EXISTING
    plateau rule, (5) select the nearer endpoint, (6) accept only because relative error <=1e-3.
    """

    def _always_high(trial_r):
        return {"realized_relative_l2": LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU, "designed_relative_l2": LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU}

    original_result = solve_bf16_radius(_always_high, LANGUAGE_LATE_REQUESTED_R, max_iterations=20, tolerance=1e-6)
    last_original_scalar = original_result["attempts"][-1]["scalar"]
    crossover = last_original_scalar / 2.0  # unreachable within the original 20 attempts by construction
    # Closer to target than the high plateau's own 1.6172189959001715e-06 gap, so "below" wins.
    low_plateau = LANGUAGE_LATE_REQUESTED_R - 1.2e-6

    def _staircase(trial_r):
        v = LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU if trial_r >= crossover else low_plateau
        return {"realized_relative_l2": v, "designed_relative_l2": v}

    expansion_result = expand_bracket_and_resolve_bf16_radius(
        _staircase, LANGUAGE_LATE_REQUESTED_R, original_result, tolerance=1e-6, max_expansion_steps=24, max_bisection_iterations=20,
    )

    assert expansion_result["expansion_used"] is True
    assert expansion_result["converged"] is False
    assert expansion_result["quantization_plateau"] is True
    assert expansion_result["nearest_realized_below"] == pytest.approx(low_plateau)
    assert expansion_result["nearest_realized_above"] == pytest.approx(LANGUAGE_LATE_OBSERVED_HIGH_PLATEAU)
    # original 20 attempts are present, unmodified, as the prefix of the combined history
    assert expansion_result["attempts"][:20] == original_result["attempts"]

    selection = select_quantization_limited_acceptance(
        expansion_result["nearest_realized_below"], expansion_result["nearest_realized_above"], LANGUAGE_LATE_REQUESTED_R,
    )
    assert selection["accepted"] is True
    assert selection["which"] == "below"
    assert selection["relative_error"] <= QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE


def test_bracket_expansion_real_bf16_end_to_end_resolves_a_confirmed_real_no_bracket_case():
    """Confirmed directly (not assumed): region_elements=50, seed=4, DIRECTION_SEED-independent
    of DIRECTION_SEED (uses its own literal seed=4) at LANGUAGE_LATE_REQUESTED_R produces a REAL
    bf16 no-bracket original-20-attempt failure (bracket_low never found). The full v3 wrapper,
    with real bf16 tensors end-to-end (no scripted/mocked evaluate_fn), must now resolve it via
    bracket expansion rather than hard-failing.
    """
    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=50)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, 4, LANGUAGE_LATE_REQUESTED_R, "region", region_names)

    assert result["bracket_expansion_used"] is True
    assert result["bracket_expansion_steps_taken"] >= 1
    assert result["radius_acceptance_mode"] in ("strict", "quantization_limited")
    if result["radius_acceptance_mode"] == "quantization_limited":
        assert result["relative_radius_error"] <= QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE
    else:
        assert result["realized_abs_error"] <= RADIUS_REALIZATION_TOLERANCE

    theta_before = base_weights["region_layer.weight"].float()
    theta_after = model.region_layer.weight.detach().float()
    realized_delta_l2 = (theta_after - theta_before).pow(2).sum().sqrt().item()
    independently_measured_ratio = realized_delta_l2 / result["theta_l2_norm"]
    # A small (50-element) region -- summation-order float noise is proportionally larger than
    # in the file's other 500,000-element checks, which use abs=1e-9; still far tighter than the
    # 1e-3 quantization-limited admissibility bound this result is judged against.
    assert independently_measured_ratio == pytest.approx(result["realized_relative_l2"], abs=1e-7)
    assert torch.equal(model.outside_layer.weight.detach(), base_weights["outside_layer.weight"])


def test_bracket_expansion_never_invoked_when_original_20_attempts_converge_strictly(monkeypatch):
    """Negative/preservation test: for a candidate that converges strictly within the original 20
    attempts (the same 500,000-element case already confirmed to converge), bracket expansion
    must NEVER be called at all -- monkeypatched to raise if it is.
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("expand_bracket_and_resolve_bf16_radius must not be called when the original solver converges")

    monkeypatch.setattr(module, "expand_bracket_and_resolve_bf16_radius", _must_not_be_called)

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert result["radius_acceptance_mode"] == "strict"
    assert result["bracket_expansion_used"] is False


def test_bracket_expansion_never_invoked_when_original_20_attempts_already_prove_a_plateau(monkeypatch):
    """Negative/preservation test: for a candidate that already proves a two-sided plateau within
    the original 20 attempts (the same 2,000-element case already confirmed to plateau),
    bracket expansion must NEVER be called -- monkeypatched to raise if it is.
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("expand_bracket_and_resolve_bf16_radius must not be called when the original solver already proved a plateau")

    monkeypatch.setattr(module, "expand_bracket_and_resolve_bf16_radius", _must_not_be_called)

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=2_000)
    region_names = ["region_layer.weight"]

    result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert result["radius_acceptance_mode"] == "quantization_limited"
    assert result["bracket_expansion_used"] is False


def test_bracket_expansion_negative_A_solver_succeeding_within_20_attempts_is_untouched(monkeypatch):
    """Negative test A: if the original solver succeeds at attempt <=20, the patched wrapper's
    output must be identical to what it always was -- proven by literally not invoking expansion
    (see test above) and reproducing the exact same accepted realized value/scale as a fresh,
    independently-run worker.
    """
    worker_1, model_1, base_weights_1, _ = _bf16_worker(region_elements=500_000)
    worker_2, model_2, base_weights_2, _ = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    result_1 = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_1, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    result_2 = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_2, DIRECTION_SEED, REQUESTED_R, "region", region_names)

    assert result_1["radius_acceptance_mode"] == result_2["radius_acceptance_mode"] == "strict"
    assert result_1["realized_relative_l2"] == result_2["realized_relative_l2"]
    assert result_1["bracket_expansion_used"] is False and result_2["bracket_expansion_used"] is False


def test_bracket_expansion_pure_never_crosses_still_reports_no_bracket_after_max_steps():
    """Negative test B (pure logic): a one-sided original result whose evaluate_fn ALWAYS
    returns the same overshoot value, regardless of scalar, never crosses to the missing side --
    expand_bracket_and_resolve_bf16_radius must exhaust max_expansion_steps and report
    converged=False, quantization_plateau=False (never inventing a bracket), with an explicit
    exhausted_reason -- never silently accepted. Uses a hand-built one-sided original_result
    (rather than a real solve_bf16_radius run, whose own aggressive proportional shrink would
    otherwise search the ENTIRE positive scalar domain within 1-2 doublings) so the expansion
    genuinely consumes several steps before giving up.
    """
    requested_r = REQUESTED_R
    high_value = requested_r * 1.001
    last_original_scalar = requested_r * 0.999  # tiny displacement already explored -- realistic shape

    original_result = {
        "converged": False, "quantization_plateau": False,
        "attempts": [{
            "iteration": 20, "scalar": last_original_scalar, "designed_relative_l2": high_value,
            "realized_relative_l2": high_value, "absolute_error": abs(high_value - requested_r),
            "bracket_low_scale": None, "bracket_high_scale": last_original_scalar,
            "bracket_low_realized": None, "bracket_high_realized": high_value,
        }],
        "best_iteration": 20, "best_scalar": last_original_scalar, "best_designed_relative_l2": high_value,
        "best_realized_relative_l2": high_value, "best_absolute_error": abs(high_value - requested_r),
        "accepted_scalar": None, "nearest_realized_below": None, "nearest_realized_above": high_value,
        "bracket_low_scale": None, "bracket_high_scale": last_original_scalar,
    }

    def _always_high(trial_r):
        return {"realized_relative_l2": high_value, "designed_relative_l2": high_value}

    expansion_result = expand_bracket_and_resolve_bf16_radius(
        _always_high, requested_r, original_result, tolerance=1e-6, max_expansion_steps=8, max_bisection_iterations=20,
    )

    assert expansion_result["converged"] is False
    assert expansion_result["quantization_plateau"] is False
    assert expansion_result["bracket_expansion_exhausted_reason"] == "max_expansion_steps_exhausted_without_crossing"
    assert expansion_result["expansion_steps_taken"] == 8


def test_bracket_expansion_negative_B_wrapper_hard_fails_when_expansion_never_crosses(monkeypatch):
    """Negative test B (wrapper level): when bracket expansion itself reports it never found a
    crossing, the wrapper must still hard-fail with RadiusCorrectionFailedError (never
    QuantizationToleranceExceededError, since no plateau was ever proven), exactly like the
    pre-existing "no plateau proven" hard-fail this branch has always had.
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    def _fake_solve(evaluate_fn, r, max_iterations, tolerance):
        evaluate_fn(r)
        return {
            "converged": False, "quantization_plateau": False,
            "attempts": [{"iteration": 1, "scalar": r, "designed_relative_l2": r * 2, "realized_relative_l2": r * 2, "absolute_error": r,
                          "bracket_low_scale": None, "bracket_high_scale": r, "bracket_low_realized": None, "bracket_high_realized": r * 2}],
            "best_iteration": 1, "best_scalar": r, "best_designed_relative_l2": r * 2, "best_realized_relative_l2": r * 2,
            "best_absolute_error": r, "accepted_scalar": None, "nearest_realized_below": None, "nearest_realized_above": r * 2,
            "bracket_low_scale": None, "bracket_high_scale": r,
        }

    def _fake_expand(evaluate_fn, r, original_solver_result, **kwargs):
        return {
            "converged": False, "quantization_plateau": False, "attempts": original_solver_result["attempts"],
            "best_iteration": 1, "best_scalar": r, "best_designed_relative_l2": r * 2, "best_realized_relative_l2": r * 2,
            "best_absolute_error": r, "accepted_scalar": None, "nearest_realized_below": None, "nearest_realized_above": r * 2,
            "bracket_low_scale": None, "bracket_high_scale": r,
            "expansion_used": True, "expansion_steps_taken": 24,
            "bracket_expansion_exhausted_reason": "max_expansion_steps_exhausted_without_crossing",
        }

    monkeypatch.setattr(module, "solve_bf16_radius", _fake_solve)
    monkeypatch.setattr(module, "expand_bracket_and_resolve_bf16_radius", _fake_expand)

    with pytest.raises(RadiusCorrectionFailedError) as exc_info:
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)
    assert not isinstance(exc_info.value, QuantizationToleranceExceededError)


def test_bracket_expansion_negative_C_bracket_found_but_relative_error_too_large_still_hard_fails():
    """Negative test C: given a PROVEN plateau (a genuine two-sided bracket) whose nearest
    attainable relative error exceeds the 1e-3 admissibility bound, the acceptance layer must
    still refuse the fallback -- proven directly on the pure decision function, exactly as the
    existing v2/v3 plateau-but-too-far case already does (select_quantization_limited_acceptance
    is completely unchanged by this repair pass).
    """
    # Both endpoints 5e-3 away from a requested r=1.0 -> relative error 0.5%, well above 0.1%.
    result = select_quantization_limited_acceptance(0.995, 1.005, 1.0)
    assert result["accepted"] is False
    assert result["relative_error"] == pytest.approx(0.005)


def test_bracket_expansion_wrapper_hard_fails_when_plateau_found_but_relative_error_too_large(monkeypatch):
    """Negative test C (wrapper level): expansion reports a genuinely proven plateau, but the
    nearest attainable state's relative error exceeds tolerance -- the wrapper must refuse via
    QuantizationToleranceExceededError (never silently accept a looser bound), exactly like the
    pre-existing plateau-too-far case.
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as module

    worker, model, base_weights, reset_calls = _bf16_worker(region_elements=500_000)
    region_names = ["region_layer.weight"]

    def _fake_solve(evaluate_fn, r, max_iterations, tolerance):
        evaluate_fn(r)
        return {
            "converged": False, "quantization_plateau": False,
            "attempts": [{"iteration": 1, "scalar": r, "designed_relative_l2": r * 1.02, "realized_relative_l2": r * 1.02, "absolute_error": r * 0.02,
                          "bracket_low_scale": None, "bracket_high_scale": r, "bracket_low_realized": None, "bracket_high_realized": r * 1.02}],
            "best_iteration": 1, "best_scalar": r, "best_designed_relative_l2": r * 1.02, "best_realized_relative_l2": r * 1.02,
            "best_absolute_error": r * 0.02, "accepted_scalar": None, "nearest_realized_below": None, "nearest_realized_above": r * 1.02,
            "bracket_low_scale": None, "bracket_high_scale": r,
        }

    def _fake_expand(evaluate_fn, r, original_solver_result, **kwargs):
        below, above = r * 0.99, r * 1.02  # both ~1-2% away -- well past the 0.1% admissibility bound
        attempts = list(original_solver_result["attempts"]) + [{
            "iteration": 21, "scalar": r * 0.99, "designed_relative_l2": below, "realized_relative_l2": below,
            "absolute_error": abs(below - r), "bracket_low_scale": r * 0.99, "bracket_high_scale": r,
            "bracket_low_realized": below, "bracket_high_realized": above,
        }]
        return {
            "converged": False, "quantization_plateau": True, "attempts": attempts,
            "best_iteration": 21, "best_scalar": r * 0.99, "best_designed_relative_l2": below, "best_realized_relative_l2": below,
            "best_absolute_error": abs(below - r), "accepted_scalar": None,
            "nearest_realized_below": below, "nearest_realized_above": above,
            "bracket_low_scale": r * 0.99, "bracket_high_scale": r,
            "expansion_used": True, "expansion_steps_taken": 1, "bracket_expansion_exhausted_reason": None,
        }

    monkeypatch.setattr(module, "solve_bf16_radius", _fake_solve)
    monkeypatch.setattr(module, "expand_bracket_and_resolve_bf16_radius", _fake_expand)

    with pytest.raises(QuantizationToleranceExceededError):
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, DIRECTION_SEED, REQUESTED_R, "region", region_names)


def test_bracket_expansion_negative_D_strict_hit_during_expansion_accepted_as_strict_not_quantization_limited():
    """Negative/positive test D: if expansion finds an exact <=1e-6 solution during the crossing
    search itself (before ever reaching bisection), it must be accepted as STRICT
    (radius_acceptance_mode="strict", quantization_limited=False), never mischaracterized as
    quantization_limited.
    """
    requested_r = REQUESTED_R
    high_value = requested_r * 1.001

    def _overshoot_only(trial_r):
        return {"realized_relative_l2": high_value, "designed_relative_l2": high_value}

    original_result = solve_bf16_radius(_overshoot_only, requested_r, max_iterations=20, tolerance=1e-6)
    assert original_result["converged"] is False
    assert original_result["bracket_low_scale"] is None

    # Derive the EXACT scalar the geometric expansion will try on step 3 (using the identical
    # formula the implementation uses), then script that specific step to hit the target exactly.
    last_scalar = original_result["attempts"][-1]["scalar"]
    base_displacement = abs(requested_r - last_scalar)
    exact_hit_scalar = requested_r - base_displacement * (2 ** 3)

    def _exact_hit_on_step_3(trial_r):
        if trial_r == exact_hit_scalar:
            return {"realized_relative_l2": requested_r, "designed_relative_l2": requested_r}
        return {"realized_relative_l2": high_value, "designed_relative_l2": high_value}

    expansion_result = expand_bracket_and_resolve_bf16_radius(
        _exact_hit_on_step_3, requested_r, original_result, tolerance=1e-6, max_expansion_steps=24, max_bisection_iterations=20,
    )
    assert expansion_result["converged"] is True
    assert expansion_result["quantization_plateau"] is False
    assert expansion_result["best_absolute_error"] == 0.0
    assert expansion_result["accepted_scalar"] == exact_hit_scalar
    assert expansion_result["expansion_steps_taken"] == 3


def test_bracket_expansion_negative_E_no_unbracketed_candidate_is_ever_accepted():
    """Negative test E: whenever expand_bracket_and_resolve_bf16_radius reports
    quantization_plateau=False (no proven bracket, whether from the original solver or after
    expansion), both nearest_realized_below and nearest_realized_above are never BOTH non-None
    at the same time -- a "bracket" is never reported as proven without genuine two-sided
    evidence.
    """
    requested_r = REQUESTED_R

    def _always_high(trial_r):
        return {"realized_relative_l2": requested_r * 2, "designed_relative_l2": requested_r * 2}

    original_result = solve_bf16_radius(_always_high, requested_r, max_iterations=20, tolerance=1e-6)
    expansion_result = expand_bracket_and_resolve_bf16_radius(
        _always_high, requested_r, original_result, tolerance=1e-6, max_expansion_steps=8, max_bisection_iterations=20,
    )
    assert expansion_result["quantization_plateau"] is False
    assert not (expansion_result["nearest_realized_below"] is not None and expansion_result["nearest_realized_above"] is not None)


def test_bracket_expansion_negative_F_no_tolerance_values_changed():
    """Negative test F: the frozen strict tolerance and quantization-limited relative-error bound
    are untouched by this repair pass.
    """
    assert RADIUS_REALIZATION_TOLERANCE == 1e-6
    assert QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE == 1e-3
    assert MAX_RADIUS_SOLVER_ITERATIONS == 20


def test_bracket_expansion_raises_on_an_already_fully_bracketed_original_result():
    """expand_bracket_and_resolve_bf16_radius must refuse to run at all if handed a
    solver_result that already has a full bracket (a caller bug, not a numerical scenario --
    that result should have proven a quantization_plateau on its own).
    """
    x_high, x_low = 0.0100050, 0.0099950
    sequence = iter([x_high, x_low, x_low])  # 3rd repeats -> plateau, matching the existing solve_bf16_radius test fixture

    def _evaluate(trial_r):
        return {"realized_relative_l2": next(sequence), "designed_relative_l2": 0.0}

    original_result = solve_bf16_radius(_evaluate, 0.01, max_iterations=20, tolerance=1e-6)
    assert original_result["bracket_low_scale"] is not None and original_result["bracket_high_scale"] is not None
    assert original_result["quantization_plateau"] is True

    with pytest.raises(ValueError):
        expand_bracket_and_resolve_bf16_radius(_evaluate, 0.01, original_result)
