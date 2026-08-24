import math

import pytest

from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.anatomy_inventory import (
    build_full_anatomy_inventory,
    build_region_report,
    compute_tensor_norm_stats,
)


def _named_params(model):
    return dict(model.named_parameters())


def test_compute_tensor_norm_stats_matches_manual_computation(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))
    region = atlas.region("vision_early")

    stats = compute_tensor_norm_stats(region, named)

    manual_sq_sum = sum(named[n].detach().float().pow(2).sum().item() for n in region.param_names)
    manual_elems = sum(named[n].numel() for n in region.param_names)
    assert stats["total_element_count"] == manual_elems
    assert stats["l2_norm"] == pytest.approx(math.sqrt(manual_sq_sum))
    assert stats["rms_magnitude"] == pytest.approx(math.sqrt(manual_sq_sum) / math.sqrt(manual_elems))


def test_compute_tensor_norm_stats_raises_on_missing_name(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))
    region = atlas.region("vision")
    incomplete = {k: v for k, v in named.items() if k != region.param_names[0]}
    with pytest.raises(KeyError):
        compute_tensor_norm_stats(region, incomplete)


def test_build_region_report_has_all_required_fields(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))
    total = sum(p.numel() for p in named.values())

    report = build_region_report(atlas, "language", named, total_model_param_count=total)

    for key in (
        "region", "level", "parent", "tensor_count", "parameter_count",
        "percent_of_total_model_parameters", "l2_norm", "rms_magnitude",
        "first_10_parameter_names", "last_10_parameter_names", "layer_indices", "mask_hash",
    ):
        assert key in report, f"missing {key}"
    assert report["tensor_count"] == len(atlas.region("language").param_names)
    assert 0.0 <= report["percent_of_total_model_parameters"] <= 100.0
    assert len(report["first_10_parameter_names"]) <= 10
    assert len(report["last_10_parameter_names"]) <= 10


def test_build_region_report_layer_indices_populated_for_depth_bands(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))
    total = sum(p.numel() for p in named.values())

    vision_late = build_region_report(atlas, "vision_late", named, total_model_param_count=total)
    language_early = build_region_report(atlas, "language_early", named, total_model_param_count=total)
    vision = build_region_report(atlas, "vision", named, total_model_param_count=total)

    assert vision_late["layer_indices"] == list(range(22, 32))
    assert language_early["layer_indices"] == list(range(0, 4))
    assert vision["layer_indices"] == []  # no single depth axis at L1


def test_build_full_anatomy_inventory_covers_every_atlas_region(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))

    inventory = build_full_anatomy_inventory(atlas, named, model_family="qwen2_5_vl", model_revision="deadbeef")

    assert set(inventory["regions"]) == set(atlas.regions)
    assert inventory["model_revision"] == "deadbeef"
    assert inventory["total_model_parameter_count"] == sum(p.numel() for p in named.values())
    assert inventory["validation"]["ok"] is True
    assert inventory["validation"]["empty_regions"] == []
    assert inventory["validation"]["sibling_overlaps"] == {}
    # language's embed_tokens/norm/lm_head sit outside any numbered layer -> reported, not an error.
    assert "language" in inventory["validation"]["uncovered_by_parent"]


def test_build_full_anatomy_inventory_l1_regions_sum_to_total(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))
    inventory = build_full_anatomy_inventory(atlas, named, model_family="qwen2_5_vl", model_revision="x")

    l1_sum = (
        inventory["regions"]["vision"]["parameter_count"]
        + inventory["regions"]["multimodal_connector_or_merger"]["parameter_count"]
        + inventory["regions"]["language"]["parameter_count"]
    )
    assert l1_sum == inventory["total_model_parameter_count"]


def test_build_full_anatomy_inventory_raises_on_empty_region():
    from neural_thickets_repro.thicket.anatomy import AnatomyAtlas, AnatomyRegion, compute_mask_hash
    from neural_thickets_repro.thicket.anatomy import AnatomyValidationError

    empty = AnatomyRegion(name="vision", level=1, parent="full_model", param_names=(), mask_hash=compute_mask_hash(()))
    atlas = AnatomyAtlas(model_family="x", regions={"vision": empty}, lm_namespace_convention=None, lm_layer_indices=(), vision_block_indices=())
    with pytest.raises(AnatomyValidationError):
        build_full_anatomy_inventory(atlas, {}, model_family="x", model_revision="y")


def test_build_full_anatomy_inventory_deterministic(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))
    a = build_full_anatomy_inventory(atlas, named, model_family="qwen2_5_vl", model_revision="x")
    b = build_full_anatomy_inventory(atlas, named, model_family="qwen2_5_vl", model_revision="x")
    assert a == b
