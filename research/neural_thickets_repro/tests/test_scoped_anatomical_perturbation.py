"""Tests for scoped_anatomical_perturbation.py's pure-logic dispatch helpers, exercised against
a fake worker (SimpleNamespace + a real synthetic model) -- same no-GPU-needed pattern as
tests/test_scope_isolation_gpu_check.py. The collective_rpc plumbing itself needs the pod.
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.scoped_anatomical_perturbation import (
    BF16_RADIUS_REALIZATION_METHOD,
    CorrectionOutOfRegionDriftError,
    MAX_RADIUS_CORRECTION_ITERATIONS,
    RADIUS_REALIZATION_TOLERANCE,
    RadiusCorrectionFailedError,
    diag_full_model_drift,
    diag_region_drift,
    diag_snapshot_base,
    scoped_apply_anatomical_perturbation,
    scoped_apply_anatomical_perturbation_bf16_corrected,
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
    assert drift["in_region"]["max_abs_drift"] == 1.0
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
