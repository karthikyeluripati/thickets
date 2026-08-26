"""Tests for scaling_common.py -- CPU-only. Covers the scale-generic infrastructure shared by
every child of the unified Stage-11 scaling experiment: ScalingModelSpec registry, the
RUNNABLE_SCALES execution gate, model-revision resolution, the whole_model region-label
translation, live anatomy-audit generalization (proven against a REAL small torch model, same
philosophy as test_run_stage11_coarse_anatomical_atlas_7b.py), the independent per-scale
direction-seed namespace, and the (network-free, factual-only) model-family comparability report.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import neural_thickets_repro.scaling_common as sc
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas


# =================================================================================================
# ScalingModelSpec registry + RUNNABLE_SCALES execution gate
# =================================================================================================


def test_registry_has_all_four_scales():
    assert set(sc.SCALING_MODEL_REGISTRY.keys()) == {"3B", "7B", "32B", "72B"}
    for scale, spec in sc.SCALING_MODEL_REGISTRY.items():
        assert spec.scale_label == scale
        assert "Qwen2.5-VL" in spec.model_name
        assert scale in spec.model_name


def test_runnable_scales_are_exactly_3b_and_7b():
    assert sc.RUNNABLE_SCALES == ("3B", "7B")


def test_ensure_scale_runnable_passes_for_3b_and_7b():
    sc.ensure_scale_runnable("3B")
    sc.ensure_scale_runnable("7B")  # must not raise


def test_ensure_scale_runnable_blocks_32b_and_72b():
    with pytest.raises(sc.ScaleNotYetEnabledError):
        sc.ensure_scale_runnable("32B")
    with pytest.raises(sc.ScaleNotYetEnabledError):
        sc.ensure_scale_runnable("72B")


def test_get_scaling_model_spec_unknown_scale_raises():
    with pytest.raises(KeyError):
        sc.get_scaling_model_spec("999B")


# =================================================================================================
# Section 1: model-revision resolution -- moved here, unchanged behavior
# =================================================================================================


def test_resolve_immutable_model_revision_passes_through_an_already_pinned_sha():
    sha = "a" * 40
    result = sc.resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", sha)
    assert result == {"model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "requested_ref": sha, "resolved_revision": sha, "resolution_method": "already_pinned"}


def test_resolve_immutable_model_revision_resolves_a_mutable_ref_via_hf_api():
    fake_info = SimpleNamespace(sha="b" * 40)
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.return_value = fake_info
        result = sc.resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")
    assert result["resolved_revision"] == "b" * 40
    assert result["resolution_method"] == "resolved_via_hf_api"


def test_resolve_immutable_model_revision_hard_fails_on_hub_exception():
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = RuntimeError("network down")
        with pytest.raises(sc.ModelRevisionResolutionError):
            sc.resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")


def test_resolve_immutable_model_revision_hard_fails_on_malformed_sha():
    fake_info = SimpleNamespace(sha="not-a-real-sha")
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.return_value = fake_info
        with pytest.raises(sc.ModelRevisionResolutionError):
            sc.resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")


def test_resolve_immutable_model_revision_hard_fails_on_none_sha():
    fake_info = SimpleNamespace(sha=None)
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.model_info.return_value = fake_info
        with pytest.raises(sc.ModelRevisionResolutionError):
            sc.resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")


# =================================================================================================
# Section 1 (Track S1): "whole_model" region-label translation
# =================================================================================================


def test_atlas_key_for_label_translates_whole_model_to_full_model():
    assert sc._atlas_key_for_label("whole_model") == "full_model"
    assert sc._atlas_key_for_label("vision") == "vision"  # non-whole_model labels pass through unchanged


# =================================================================================================
# Scale/track-generic live anatomy audit -- proven against a REAL small torch model
# =================================================================================================


def test_report_scaling_anatomy_audit_whole_model_covers_100_percent(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    audit = sc.report_scaling_anatomy_audit(worker, (sc.WHOLE_MODEL_REGION_LABEL,))
    info = audit["regions"][sc.WHOLE_MODEL_REGION_LABEL]
    assert info["n_elements"] == audit["total_model_elements"]
    assert info["percentage_of_total_elements"] == pytest.approx(100.0)
    assert audit["uncovered_by_full_model"] == []
    sc.ensure_whole_model_covers_100_percent(audit)  # must not raise


def test_report_scaling_anatomy_audit_anatomy_track_matches_existing_stage11_regions(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    regions = ("vision", "multimodal_connector_or_merger", "language")
    audit = sc.report_scaling_anatomy_audit(worker, regions)
    assert set(audit["regions"].keys()) == set(regions)
    assert audit["union_equals_full_model"] is True
    assert audit["pairwise_disjoint"] is True
    total_from_regions = sum(info["n_elements"] for info in audit["regions"].values())
    assert total_from_regions == audit["total_model_elements"]
    sc.ensure_scaling_anatomy_audit_passes(audit, regions)  # must not raise


def test_whole_model_and_anatomy_track_agree_on_total_model_elements(runtime_wrapped_vlm_32vision_factory):
    """whole_model's own audit and the 3-region anatomy audit's SUM must report the identical
    total_model_elements -- both are read from the same live model, just partitioned differently.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    whole = sc.report_scaling_anatomy_audit(worker, (sc.WHOLE_MODEL_REGION_LABEL,))
    anatomy = sc.report_scaling_anatomy_audit(worker, ("vision", "multimodal_connector_or_merger", "language"))
    assert whole["total_model_elements"] == anatomy["total_model_elements"]
    assert whole["regions"][sc.WHOLE_MODEL_REGION_LABEL]["n_elements"] == sum(info["n_elements"] for info in anatomy["regions"].values())


def test_ensure_scaling_anatomy_audit_passes_rejects_missing_region():
    fake_audit = {"regions": {"vision": {"n_tensors": 1}}, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": []}
    with pytest.raises(RuntimeError):
        sc.ensure_scaling_anatomy_audit_passes(fake_audit, ("vision", "language"))


def test_ensure_whole_model_covers_100_percent_rejects_a_partial_audit():
    fake_audit = {"regions": {"whole_model": {"n_elements": 50, "percentage_of_total_elements": 50.0}}, "total_model_elements": 100}
    with pytest.raises(RuntimeError):
        sc.ensure_whole_model_covers_100_percent(fake_audit)


def test_report_region_param_names_for_scaling_whole_model(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    result = sc.report_region_param_names_for_scaling(worker, (sc.WHOLE_MODEL_REGION_LABEL,))
    all_names = [n for n, _ in model.named_parameters()]
    assert sorted(result[sc.WHOLE_MODEL_REGION_LABEL]["param_names"]) == sorted(all_names)


def test_compute_anatomy_audit_hash_is_deterministic_and_label_generic(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    audit1 = sc.report_scaling_anatomy_audit(worker, (sc.WHOLE_MODEL_REGION_LABEL,))
    audit2 = sc.report_scaling_anatomy_audit(worker, (sc.WHOLE_MODEL_REGION_LABEL,))
    assert sc.compute_anatomy_audit_hash(audit1) == sc.compute_anatomy_audit_hash(audit2)


# =================================================================================================
# Section 7: independent per-scale direction-family seed namespace
# =================================================================================================


def test_build_scaling_direction_seed_bank_deterministic():
    bank1 = sc.build_scaling_direction_seed_bank(42, "7B", ("whole_model",), 8)
    bank2 = sc.build_scaling_direction_seed_bank(42, "7B", ("whole_model",), 8)
    assert bank1 == bank2
    assert len(bank1["whole_model"]) == 8
    assert len(set(bank1["whole_model"])) == 8


def test_build_scaling_direction_seed_bank_independent_across_scales_even_with_same_base_seed_and_region():
    bank_3b = sc.build_scaling_direction_seed_bank(42, "3B", ("whole_model",), 8)
    bank_7b = sc.build_scaling_direction_seed_bank(42, "7B", ("whole_model",), 8)
    assert bank_3b != bank_7b
    assert set(bank_3b["whole_model"]).isdisjoint(set(bank_7b["whole_model"]))


def test_build_scaling_direction_seed_bank_independent_of_the_pre_scaling_stage11_namespace():
    """The pre-existing run_stage11_coarse_anatomical_atlas_7b.build_stage11_direction_seed_bank
    uses a 3-argument namespace ("stage11_direction_family", region, i) with no scale_label --
    this scaling-generic bank ADDS scale_label as an explicit namespace component, so even for
    the one scale (7B) where the old namespace remains valid (it is only ever used for
    scale="7B"/track="anatomy"), the two banks must never coincide.
    """
    from neural_thickets_repro.run_stage11_coarse_anatomical_atlas_7b import build_stage11_direction_seed_bank as old_bank_fn

    old_bank = old_bank_fn(42, ("language",), 8)
    new_bank = sc.build_scaling_direction_seed_bank(42, "7B", ("language",), 8)
    assert old_bank != new_bank
    assert set(old_bank["language"]).isdisjoint(set(new_bank["language"]))


def test_compute_direction_seed_bank_hash_is_order_independent_over_regions():
    bank = sc.build_scaling_direction_seed_bank(1, "7B", ("vision", "language"), 4)
    reordered = {"language": bank["language"], "vision": bank["vision"]}
    assert sc.compute_direction_seed_bank_hash(bank) == sc.compute_direction_seed_bank_hash(reordered)


# =================================================================================================
# Section 5: model-family comparability report -- factual metadata only, no scientific claim
# =================================================================================================


def test_build_model_family_comparability_report_without_live_fetch_reports_fetched_false():
    report = sc.build_model_family_comparability_report(list(sc.SCALING_MODEL_REGISTRY.values()))
    assert set(report["scales"].keys()) == {"3B", "7B", "32B", "72B"}
    for entry in report["scales"].values():
        assert entry["fetched"] is False
        assert entry["resolved_commit_sha"] is None
        assert len(entry["caveats"]) >= 1


def test_build_model_family_comparability_report_with_fake_live_fetch():
    def fake_fetch(model_name):
        return {"sha": "c" * 40, "config_architecture": "Qwen2_5_VLForConditionalGeneration", "parameter_count": 7_000_000_000}

    report = sc.build_model_family_comparability_report([sc.SCALING_MODEL_REGISTRY["7B"]], hf_model_info_fn=fake_fetch)
    entry = report["scales"]["7B"]
    assert entry["fetched"] is True
    assert entry["resolved_commit_sha"] == "c" * 40
    assert entry["parameter_count"] == 7_000_000_000


def test_build_model_family_comparability_report_fetch_failure_degrades_gracefully():
    def failing_fetch(model_name):
        raise RuntimeError("no network")

    report = sc.build_model_family_comparability_report([sc.SCALING_MODEL_REGISTRY["32B"]], hf_model_info_fn=failing_fetch)
    entry = report["scales"]["32B"]
    assert entry["fetched"] is False
    assert "fetch_error" in entry


def test_comparability_report_never_makes_a_scientific_claim():
    report = sc.build_model_family_comparability_report(list(sc.SCALING_MODEL_REGISTRY.values()))
    assert "no scientific claim" in report["note"].lower()
