import torch

from neural_thickets_repro.thicket.perturbation import (
    DegenerateRegionError,
    PerturbationManifest,
    apply_anatomical_relative_l2,
    apply_global_gaussian_upstream,
    compute_perturbation_id,
    generate_perturbation_population,
    undo_anatomical_relative_l2,
    undo_global_gaussian_upstream,
)


def _clone_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _states_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


# --- C1: global_gaussian_upstream is a thin, unmodified wrapper ------------------------------


def test_global_gaussian_upstream_matches_perturb_cpu_directly(dummy_vlm_factory):
    from neural_thickets_repro.perturb_cpu import perturb

    m1, m2 = dummy_vlm_factory(), dummy_vlm_factory()
    apply_global_gaussian_upstream(m1, seed=5, sigma=0.1)
    perturb(m2, seed=5, sigma=0.1)
    assert _states_equal(m1.state_dict(), m2.state_dict())


def test_global_gaussian_upstream_never_touches_vision(dummy_vlm_factory):
    model = dummy_vlm_factory()
    before_visual = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("visual.")}
    apply_global_gaussian_upstream(model, seed=1, sigma=0.5)
    after_visual = {k: v for k, v in model.state_dict().items() if k.startswith("visual.")}
    assert _states_equal(before_visual, after_visual)


def test_global_gaussian_upstream_apply_then_undo_is_close(dummy_vlm_factory):
    model = dummy_vlm_factory()
    before = _clone_state(model)
    record = apply_global_gaussian_upstream(model, seed=3, sigma=0.2)
    undo_global_gaussian_upstream(model, record)
    after = model.state_dict()
    assert all(torch.allclose(before[k], after[k], atol=1e-6, rtol=0) for k in before)


# --- C2/C3: anatomical_relative_l2 exact-norm rescaling ---------------------------------------


def _region_names(model, prefix):
    return [n for n, _ in model.named_parameters() if n.startswith(prefix)]


def test_anatomical_relative_l2_hits_requested_ratio_exactly(dummy_vlm_factory):
    model = dummy_vlm_factory()
    region_names = _region_names(model, "model.layers.")
    record = apply_anatomical_relative_l2(model, region="lm_layers", region_param_names=region_names, seed=11, r=0.05)

    ratio = record.realized_epsilon_l2_norm / record.theta_l2_norm
    assert abs(ratio - 0.05) < 1e-6


def test_anatomical_relative_l2_outside_region_is_exactly_unchanged(dummy_vlm_factory):
    model = dummy_vlm_factory()
    region_names = set(_region_names(model, "model.layers."))
    before = _clone_state(model)

    apply_anatomical_relative_l2(model, region="lm_layers", region_param_names=region_names, seed=11, r=0.2)

    after = model.state_dict()
    for name, tensor in before.items():
        if name not in region_names:
            assert torch.equal(tensor, after[name]), f"{name} outside the perturbed region changed"


def test_anatomical_relative_l2_region_actually_changed(dummy_vlm_factory):
    model = dummy_vlm_factory()
    region_names = _region_names(model, "model.layers.")
    before = _clone_state(model)
    apply_anatomical_relative_l2(model, region="lm_layers", region_param_names=region_names, seed=11, r=0.2)
    after = model.state_dict()
    assert any(not torch.equal(before[n], after[n]) for n in region_names)


def test_anatomical_relative_l2_same_seed_is_deterministic(dummy_vlm_factory):
    m1, m2 = dummy_vlm_factory(), dummy_vlm_factory()
    names = _region_names(m1, "model.layers.")
    apply_anatomical_relative_l2(m1, region="lm_layers", region_param_names=names, seed=99, r=0.1)
    apply_anatomical_relative_l2(m2, region="lm_layers", region_param_names=names, seed=99, r=0.1)
    assert _states_equal(m1.state_dict(), m2.state_dict())


def test_anatomical_relative_l2_different_seed_differs(dummy_vlm_factory):
    m1, m2 = dummy_vlm_factory(), dummy_vlm_factory()
    names = _region_names(m1, "model.layers.")
    apply_anatomical_relative_l2(m1, region="lm_layers", region_param_names=names, seed=1, r=0.1)
    apply_anatomical_relative_l2(m2, region="lm_layers", region_param_names=names, seed=2, r=0.1)
    assert not _states_equal(m1.state_dict(), m2.state_dict())


def test_anatomical_relative_l2_apply_then_undo_restores_region(dummy_vlm_factory):
    model = dummy_vlm_factory()
    names = _region_names(model, "model.layers.")
    before = _clone_state(model)
    record = apply_anatomical_relative_l2(model, region="lm_layers", region_param_names=names, seed=7, r=0.15)
    undo_anatomical_relative_l2(model, record)
    after = model.state_dict()
    for n in names:
        assert torch.allclose(before[n], after[n], atol=1e-5, rtol=0)


def test_anatomical_relative_l2_rejects_empty_region(dummy_vlm_factory):
    model = dummy_vlm_factory()
    try:
        apply_anatomical_relative_l2(model, region="empty", region_param_names=[], seed=1, r=0.1)
        assert False, "expected DegenerateRegionError"
    except DegenerateRegionError:
        pass


def test_anatomical_relative_l2_different_regions_get_different_scale(dummy_vlm_factory):
    """Confirms the exact-rescale design is NOT a single global per-weight sigma shared across
    regions of different dimensionality (spec C2's central requirement) -- two differently
    -sized regions asked for the same r generally get different derived scale factors.
    """
    model = dummy_vlm_factory()
    small_region = _region_names(model, "model.norm.")
    large_region = _region_names(model, "model.layers.")
    small_record = apply_anatomical_relative_l2(model, region="small", region_param_names=small_region, seed=1, r=0.1)
    large_record = apply_anatomical_relative_l2(model, region="large", region_param_names=large_region, seed=1, r=0.1)
    assert small_record.scale != large_record.scale


# --- D: perturbation identity ------------------------------------------------------------------


def _manifest_kwargs(**overrides):
    kwargs = dict(
        seed=1, perturbation_mode="anatomical_relative_l2", model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="hash1", anatomy_region="vision_early", radius=0.05, sigma=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_perturbation_id_same_manifest_fields_same_id():
    m1 = PerturbationManifest(**_manifest_kwargs())
    m2 = PerturbationManifest(**_manifest_kwargs())
    assert m1.perturbation_id == m2.perturbation_id


def test_perturbation_id_different_seed_different_id():
    m1 = PerturbationManifest(**_manifest_kwargs(seed=1))
    m2 = PerturbationManifest(**_manifest_kwargs(seed=2))
    assert m1.perturbation_id != m2.perturbation_id


def test_perturbation_id_different_region_different_id():
    m1 = PerturbationManifest(**_manifest_kwargs(anatomy_region="vision_early"))
    m2 = PerturbationManifest(**_manifest_kwargs(anatomy_region="vision_late"))
    assert m1.perturbation_id != m2.perturbation_id


def test_compute_perturbation_id_matches_manifest_field():
    kwargs = _manifest_kwargs()
    manifest = PerturbationManifest(**kwargs)
    expected = compute_perturbation_id(
        kwargs["seed"], kwargs["perturbation_mode"], kwargs["anatomy_region"], kwargs["radius"], kwargs["sigma"],
        kwargs["model_family"], kwargs["model_scale"], kwargs["model_revision"], kwargs["parameter_mask_hash"],
    )
    assert manifest.perturbation_id == expected


def test_manifest_rejects_unknown_mode():
    import pytest

    from neural_thickets_repro.thicket.perturbation import UnknownPerturbationModeError

    with pytest.raises(UnknownPerturbationModeError):
        PerturbationManifest(**_manifest_kwargs(perturbation_mode="not_a_real_mode"))


# --- D1: shared population across capabilities --------------------------------------------------


def test_generate_perturbation_population_is_reproducible():
    pop_1 = generate_perturbation_population(
        mode="anatomical_relative_l2", n=5, base_seed=42, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="hash1", anatomy_region="vision_early", radius=0.05,
    )
    pop_2 = generate_perturbation_population(
        mode="anatomical_relative_l2", n=5, base_seed=42, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="hash1", anatomy_region="vision_early", radius=0.05,
    )
    assert [m.perturbation_id for m in pop_1] == [m.perturbation_id for m in pop_2]
    assert [m.seed for m in pop_1] == [m.seed for m in pop_2]


def test_generate_perturbation_population_seeds_are_all_distinct():
    pop = generate_perturbation_population(
        mode="anatomical_relative_l2", n=10, base_seed=42, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="hash1", anatomy_region="vision_early", radius=0.05,
    )
    assert len({m.seed for m in pop}) == 10


def test_generate_perturbation_population_differs_across_regions():
    """This is what guarantees perturbation i is a genuinely different perturbation per
    region -- but the two populations are still each independently reproducible, and a driver
    iterating capabilities for a FIXED (mode, region, radius) cell sees the same population
    every time (tested above) -- the cross-capability alignment this spec section requires.
    """
    pop_a = generate_perturbation_population(
        mode="anatomical_relative_l2", n=5, base_seed=42, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="hash1", anatomy_region="vision_early", radius=0.05,
    )
    pop_b = generate_perturbation_population(
        mode="anatomical_relative_l2", n=5, base_seed=42, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="hash1", anatomy_region="vision_late", radius=0.05,
    )
    assert [m.seed for m in pop_a] != [m.seed for m in pop_b]
