import pytest

from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.upstream_scope import compute_upstream_scope_inventory


def _named_params(model):
    return dict(model.named_parameters())


def test_upstream_scope_excludes_vision_and_connector_entirely(runtime_wrapped_vlm_32vision_factory):
    """The Stage-6 upstream rule excludes anything prefixed visual./model.visual. --
    multimodal_connector_or_merger (visual.merger.*) is a SUBSET of that prefix, so it must be
    100% excluded too, not merely "mostly" excluded. Measured against the live fixture's real
    parameter names, never assumed.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))

    report = compute_upstream_scope_inventory(atlas, named)

    assert report["per_region"]["vision"]["tensors_perturbed"] == 0
    assert report["per_region"]["vision"]["percent_perturbed"] == 0.0
    assert report["per_region"]["multimodal_connector_or_merger"]["tensors_perturbed"] == 0
    assert report["per_region"]["multimodal_connector_or_merger"]["percent_perturbed"] == 0.0


def test_upstream_scope_perturbs_language_entirely(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))

    report = compute_upstream_scope_inventory(atlas, named)

    language_region = atlas.region("language")
    assert report["per_region"]["language"]["tensors_perturbed"] == language_region.param_count
    assert report["per_region"]["language"]["tensors_excluded"] == 0
    assert report["per_region"]["language"]["percent_perturbed"] == pytest.approx(100.0)


def test_upstream_scope_equals_language_region_exactly(runtime_wrapped_vlm_32vision_factory):
    """Measured (not assumed) confirmation that the upstream `_should_perturb`-derived scope
    is exactly the `language` L1 region for this fixture's naming convention -- the property
    the real Qwen2.5-VL-3B-Instruct GPU run must also confirm before Stage 6 is described as
    "language-only" in any paper wording.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))

    report = compute_upstream_scope_inventory(atlas, named)

    comparison = report["upstream_scope_vs_language_region"]
    assert comparison["equals_language_region"] is True
    assert comparison["in_upstream_not_in_language_count"] == 0
    assert comparison["in_language_not_in_upstream_count"] == 0


def test_upstream_scope_full_model_percent_perturbed_matches_should_perturb(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))

    report = compute_upstream_scope_inventory(atlas, named)

    all_names = list(named)
    perturbed_count = sum(named[n].numel() for n in all_names if should_perturb(n))
    total_count = sum(p.numel() for p in named.values())
    expected_pct = 100.0 * perturbed_count / total_count
    assert report["per_region"]["full_model"]["percent_perturbed"] == pytest.approx(expected_pct)
    assert report["per_region"]["full_model"]["parameters_perturbed"] == perturbed_count


def test_upstream_perturbed_scope_stats_match_manual_computation(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))

    report = compute_upstream_scope_inventory(atlas, named)
    scope = report["upstream_perturbed_scope"]

    perturbed_names = [n for n in named if should_perturb(n)]
    manual_param_count = sum(named[n].numel() for n in perturbed_names)
    manual_sq_sum = sum(named[n].detach().float().pow(2).sum().item() for n in perturbed_names)

    assert scope["tensor_count"] == len(perturbed_names)
    assert scope["parameter_count"] == manual_param_count
    assert scope["l2_norm"] == pytest.approx(manual_sq_sum ** 0.5)
    assert scope["rms_magnitude"] == pytest.approx((manual_sq_sum ** 0.5) / (manual_param_count ** 0.5))


def test_compute_upstream_scope_inventory_deterministic(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named = _named_params(model)
    atlas = build_anatomy_atlas(list(named))
    a = compute_upstream_scope_inventory(atlas, named)
    b = compute_upstream_scope_inventory(atlas, named)
    assert a == b
