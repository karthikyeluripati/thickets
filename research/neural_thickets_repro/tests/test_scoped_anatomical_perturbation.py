"""Tests for scoped_anatomical_perturbation.py's pure-logic dispatch helpers, exercised against
a fake worker (SimpleNamespace + a real synthetic model) -- same no-GPU-needed pattern as
tests/test_scope_isolation_gpu_check.py. The collective_rpc plumbing itself needs the pod.
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.scoped_anatomical_perturbation import (
    diag_full_model_drift,
    diag_region_drift,
    diag_snapshot_base,
    scoped_apply_anatomical_perturbation,
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
