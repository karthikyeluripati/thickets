"""Tests for diagnostics/stage7b_radius_realization_smoke.py -- CPU-only, same fake-engine
philosophy as tests/test_run_stage7b_anatomical_calibration.py. The real GPU/Ray/vLLM engine is
never launched.
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.diagnostics.stage7b_radius_realization_smoke import (
    SMALLEST_CALIBRATION_RADIUS,
    derive_smallest_radius_seed,
    report_post_solve_outside_region_drift,
    run_one_region_smallest_radius_smoke,
)
from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.run_stage7b_anatomical_calibration import (
    FULL_CALIBRATION_RADII,
    FULL_CALIBRATION_REGIONS,
    STAGE7B_BASE_SEED,
    PERTURBATION_MODE,
)
from neural_thickets_repro.scoped_anatomical_perturbation import RadiusCorrectionFailedError
from neural_thickets_repro.thicket.perturbation import generate_perturbation_population


def _identity_ray_get(x):
    return x


def test_smallest_calibration_radius_is_the_first_frozen_radius():
    assert SMALLEST_CALIBRATION_RADIUS == FULL_CALIBRATION_RADII[0] == 0.0035698828543799426


def test_three_regions_under_test():
    assert len(FULL_CALIBRATION_REGIONS) == 3
    assert FULL_CALIBRATION_REGIONS == ("vision", "multimodal_connector_or_merger", "language")


# --- derive_smallest_radius_seed: must match the REAL Stage-7B population's own seed ------------


@pytest.mark.parametrize("region", FULL_CALIBRATION_REGIONS)
def test_derive_smallest_radius_seed_matches_the_real_stage7b_population(region):
    seed = derive_smallest_radius_seed(region, model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1")

    real_population = generate_perturbation_population(
        mode=PERTURBATION_MODE, n=1, base_seed=STAGE7B_BASE_SEED, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="whatever_mask_hash_the_real_run_actually_uses",
        anatomy_region=region, radius=SMALLEST_CALIBRATION_RADIUS, sigma=None,
    )
    assert seed == real_population[0].seed


def test_derive_smallest_radius_seed_differs_across_regions():
    seeds = {
        region: derive_smallest_radius_seed(region, model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1")
        for region in FULL_CALIBRATION_REGIONS
    }
    assert len(set(seeds.values())) == 3


def test_derive_smallest_radius_seed_is_deterministic():
    a = derive_smallest_radius_seed("vision", model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1")
    b = derive_smallest_radius_seed("vision", model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1")
    assert a == b


# --- report_post_solve_outside_region_drift ------------------------------------------------------


class _TwoTensorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.region_layer = torch.nn.Linear(50, 1, bias=False)
        self.outside_layer = torch.nn.Linear(50, 1, bias=False)


def test_report_post_solve_outside_region_drift_zero_when_unchanged():
    torch.manual_seed(0)
    model = _TwoTensorModel()
    base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model), _base_weights=base_weights)

    report = report_post_solve_outside_region_drift(worker, ["region_layer.weight"])

    assert report["outside_region_max_abs_drift"] == 0.0
    assert report["outside_region_changed_tensor_count"] == 0
    assert report["outside_region_total_tensor_count"] == 1


def test_report_post_solve_outside_region_drift_detects_a_real_leak():
    torch.manual_seed(0)
    model = _TwoTensorModel()
    base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model), _base_weights=base_weights)

    with torch.no_grad():
        model.outside_layer.weight.add_(1.0)

    report = report_post_solve_outside_region_drift(worker, ["region_layer.weight"])
    # measure_drift's chunked float64 accumulation (Stage-11 7B whole_model OOM fix -- see
    # thicket/memory_bounded_ops.py) is STRICTLY MORE precise than the float32 diff it replaces,
    # so it can reveal a genuine ~2^-24 (~5.96e-8) bf16-rounding discrepancy that float32
    # subtraction happened to round away to exactly 1.0 via round-half-to-even.
    assert report["outside_region_max_abs_drift"] == pytest.approx(1.0, abs=1e-6)
    assert report["outside_region_changed_tensor_count"] == 1


def test_report_post_solve_outside_region_drift_requires_base_weights():
    torch.manual_seed(0)
    model = _TwoTensorModel()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    with pytest.raises(RuntimeError, match="store_base_weights"):
        report_post_solve_outside_region_drift(worker, ["region_layer.weight"])


# --- run_one_region_smallest_radius_smoke: fake-engine end-to-end lifecycle ----------------------


class _FakeSmokeEngine:
    """Persistent-worker-shaped fake, same philosophy as _FakeCalibrationEngine in
    test_run_stage7b_anatomical_calibration.py.
    """

    def __init__(self, model):
        self._model = model
        self._base_weights = None
        self.calls = []
        self._worker_self = SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            reset_to_base_weights=self._reset_to_base_weights,
            _should_perturb=should_perturb,
        )
        self.collective_rpc = SimpleNamespace(remote=self._collective_rpc)

    def store_base_weights(self):
        self._base_weights = {name: p.detach().clone() for name, p in self._model.named_parameters()}
        self._worker_self._base_weights = self._base_weights

    def _reset_to_base_weights(self):
        if self._base_weights is None:
            raise RuntimeError("store_base_weights not called")
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                p.copy_(self._base_weights[name])

    def _collective_rpc(self, method, args=()):
        label = method if isinstance(method, str) else getattr(method, "__name__", "callable")
        self.calls.append(label)
        if method == "reset_to_base_weights":
            self._reset_to_base_weights()
            return [True]
        if callable(method):
            return [method(self._worker_self, *args)]
        raise ValueError(f"unsupported method {method!r}")


def test_run_one_region_smallest_radius_smoke_succeeds_and_restores(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas

    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names = atlas.region("language").param_names

    engine = _FakeSmokeEngine(model)
    engine.store_base_weights()
    base_snapshot = {k: v.clone() for k, v in engine._base_weights.items()}

    result = run_one_region_smallest_radius_smoke(engine, "language", region_param_names, seed=42, ray_get=_identity_ray_get)

    assert result["region"] == "language"
    assert result["requested_radius"] == SMALLEST_CALIBRATION_RADIUS
    assert result["solved"] is True
    assert result["absolute_error"] <= 1e-6
    assert result["outside_region_changed_tensor_count"] == 0
    assert result["outside_region_max_abs_drift"] == 0.0
    assert result["restoration_exact"] is True
    assert result["error"] is None

    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_snapshot[name])


def test_run_one_region_smallest_radius_smoke_never_evaluates_capabilities():
    import inspect

    from neural_thickets_repro.diagnostics import stage7b_radius_realization_smoke as module

    source = inspect.getsource(module)
    for forbidden in ("run_benchmark", "CapabilityContext", "aggregate_metrics", "SamplingParams", "AutoTokenizer"):
        assert forbidden not in source


def test_run_one_region_smallest_radius_smoke_reports_plateau_failure_and_still_restores(monkeypatch):
    torch.manual_seed(0)
    model = _TwoTensorModel()
    engine = _FakeSmokeEngine(model)
    engine.store_base_weights()
    base_snapshot = {k: v.clone() for k, v in engine._base_weights.items()}

    import neural_thickets_repro.diagnostics.stage7b_radius_realization_smoke as module
    from neural_thickets_repro.scoped_anatomical_perturbation import QuantizationToleranceExceededError

    def _broken_apply(worker_self, seed, r, region_name, region_param_names):
        # QuantizationToleranceExceededError specifically -- a PROVEN plateau whose nearest
        # attainable state's relative error still exceeds 0.1% -- distinct from a plain
        # RadiusCorrectionFailedError (no plateau proven at all), which the smoke module
        # correctly reports as quantization_plateau=False.
        raise QuantizationToleranceExceededError("simulated plateau: quantization_plateau=True nearest_realized_below=x nearest_realized_above=y")

    monkeypatch.setattr(module, "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3", _broken_apply)

    result = run_one_region_smallest_radius_smoke(engine, "region", ["region_layer.weight"], seed=1, ray_get=_identity_ray_get)

    assert result["solved"] is False
    assert result["quantization_plateau"] is True
    assert result["error"] is not None
    assert result["restoration_exact"] is True
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_snapshot[name])


def test_run_one_region_smallest_radius_smoke_reports_non_plateau_failure_and_still_restores(monkeypatch):
    """A plain RadiusCorrectionFailedError (no plateau proven at all -- the solver simply ran
    out of attempts) must be reported as quantization_plateau=False, distinct from the
    QuantizationToleranceExceededError plateau case above.
    """
    torch.manual_seed(0)
    model = _TwoTensorModel()
    engine = _FakeSmokeEngine(model)
    engine.store_base_weights()
    base_snapshot = {k: v.clone() for k, v in engine._base_weights.items()}

    import neural_thickets_repro.diagnostics.stage7b_radius_realization_smoke as module

    def _broken_apply(worker_self, seed, r, region_name, region_param_names):
        raise RadiusCorrectionFailedError("simulated: did not converge, no plateau proven")

    monkeypatch.setattr(module, "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3", _broken_apply)

    result = run_one_region_smallest_radius_smoke(engine, "region", ["region_layer.weight"], seed=1, ray_get=_identity_ray_get)

    assert result["solved"] is False
    assert result["quantization_plateau"] is False
    assert result["error"] is not None
    assert result["restoration_exact"] is True
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_snapshot[name])


def test_output_root_is_disjoint_from_stage7b_calibration_output():
    from neural_thickets_repro.diagnostics.stage7b_radius_realization_smoke import REPO_ROOT

    default_out = REPO_ROOT / "results" / "stage7b_radius_realization_smoke" / "report.json"
    calibration_root = REPO_ROOT / "results" / "stage7b_anatomical_calibration"
    assert calibration_root not in default_out.parents
    assert "stage7b_radius_realization_smoke" in str(default_out)
