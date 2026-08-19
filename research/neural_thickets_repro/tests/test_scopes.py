"""Tests for scopes.py -- pure Python/torch, no GPU/ray/vllm needed."""
import pytest

from neural_thickets_repro.scopes import (
    PERTURBATION_SCOPES,
    ScopeSelectionError,
    build_scope_manifest,
    compute_relative_l2_sigma,
    detect_lm_namespace_convention,
    discover_lm_layer_indices,
    discover_vision_block_indices,
    partition_layers_into_thirds,
    partition_vision_blocks,
    scope_requires_encoder_cache_reset,
)


# --- detect_lm_namespace_convention / discover_lm_layer_indices ---


def test_detects_flat_checkpoint_convention(flat_checkpoint_vlm_factory):
    names = [n for n, _ in flat_checkpoint_vlm_factory().named_parameters()]
    assert detect_lm_namespace_convention(names) == "flat_checkpoint"


def test_detects_runtime_wrapped_convention(runtime_wrapped_vlm_factory):
    names = [n for n, _ in runtime_wrapped_vlm_factory().named_parameters()]
    assert detect_lm_namespace_convention(names) == "runtime_wrapped"


def test_discover_lm_layer_indices_flat_checkpoint(flat_checkpoint_vlm_factory):
    names = [n for n, _ in flat_checkpoint_vlm_factory().named_parameters()]
    convention, indices = discover_lm_layer_indices(names)
    assert convention == "flat_checkpoint"
    assert indices == list(range(12))


def test_discover_lm_layer_indices_runtime_wrapped(runtime_wrapped_vlm_factory):
    names = [n for n, _ in runtime_wrapped_vlm_factory().named_parameters()]
    convention, indices = discover_lm_layer_indices(names)
    assert convention == "runtime_wrapped"
    assert indices == list(range(12))


def test_no_recognized_namespace_hard_fails():
    with pytest.raises(ScopeSelectionError, match="No recognized LM-layer namespace"):
        detect_lm_namespace_convention(["visual.patch_embed.weight", "some.other.thing.weight"])


def test_ambiguous_namespace_hard_fails():
    with pytest.raises(ScopeSelectionError, match="Ambiguous LM-layer namespace"):
        detect_lm_namespace_convention(["model.layers.0.weight", "language_model.model.layers.0.weight"])


def test_non_contiguous_layer_indices_hard_fail():
    with pytest.raises(ScopeSelectionError, match="not a complete contiguous range"):
        discover_lm_layer_indices(["model.layers.0.weight", "model.layers.2.weight"])


# --- partition_layers_into_thirds ---


def test_partition_into_thirds_12_layers():
    thirds = partition_layers_into_thirds(list(range(12)))
    assert thirds == {"early": [0, 1, 2, 3], "middle": [4, 5, 6, 7], "late": [8, 9, 10, 11]}


def test_partition_into_thirds_36_layers_matches_expected_ranges():
    thirds = partition_layers_into_thirds(list(range(36)))
    assert thirds["early"] == list(range(0, 12))
    assert thirds["middle"] == list(range(12, 24))
    assert thirds["late"] == list(range(24, 36))


def test_partition_thirds_mutually_disjoint_and_covers_all():
    thirds = partition_layers_into_thirds(list(range(12)))
    all_selected = thirds["early"] + thirds["middle"] + thirds["late"]
    assert sorted(all_selected) == list(range(12))
    assert set(thirds["early"]) & set(thirds["middle"]) == set()
    assert set(thirds["middle"]) & set(thirds["late"]) == set()
    assert set(thirds["early"]) & set(thirds["late"]) == set()


def test_partition_thirds_hard_fails_when_not_divisible_by_3():
    with pytest.raises(ScopeSelectionError, match="into three equal contiguous thirds"):
        partition_layers_into_thirds(list(range(10)))


# --- build_scope_manifest: per-scope selection correctness, both conventions ---


@pytest.mark.parametrize("factory_name", ["flat_checkpoint_vlm_factory", "runtime_wrapped_vlm_factory"])
def test_vision_encoder_selects_no_lm_or_merger_params(factory_name, request):
    model = request.getfixturevalue(factory_name)()
    manifest = build_scope_manifest("vision_encoder", model.named_parameters())
    assert manifest.selected_param_count > 0
    for name in manifest.selected_param_names:
        assert name.startswith("visual.")
        assert "merger" not in name


@pytest.mark.parametrize("factory_name", ["flat_checkpoint_vlm_factory", "runtime_wrapped_vlm_factory"])
def test_vision_merger_selects_no_encoder_or_lm_params(factory_name, request):
    model = request.getfixturevalue(factory_name)()
    manifest = build_scope_manifest("vision_merger", model.named_parameters())
    assert manifest.selected_param_count > 0
    for name in manifest.selected_param_names:
        assert name.startswith("visual.merger.")


@pytest.mark.parametrize("factory_name", ["flat_checkpoint_vlm_factory", "runtime_wrapped_vlm_factory"])
def test_full_lm_excludes_all_visual(factory_name, request):
    model = request.getfixturevalue(factory_name)()
    manifest = build_scope_manifest("full_lm", model.named_parameters())
    assert manifest.selected_param_count > 0
    for name in manifest.selected_param_names:
        assert not name.startswith(("visual.", "model.visual."))


@pytest.mark.parametrize("factory_name", ["flat_checkpoint_vlm_factory", "runtime_wrapped_vlm_factory"])
def test_full_vlm_contains_all_deduplicated_params(factory_name, request):
    model = request.getfixturevalue(factory_name)()
    all_names = {name for name, _ in model.named_parameters()}
    manifest = build_scope_manifest("full_vlm", model.named_parameters())
    assert set(manifest.selected_param_names) == all_names


@pytest.mark.parametrize("factory_name", ["flat_checkpoint_vlm_factory", "runtime_wrapped_vlm_factory"])
def test_lm_thirds_mutually_disjoint_and_every_layer_in_exactly_one(factory_name, request):
    model = request.getfixturevalue(factory_name)()
    named_params = list(model.named_parameters())
    early = set(build_scope_manifest("lm_early", named_params).selected_param_names)
    middle = set(build_scope_manifest("lm_middle", named_params).selected_param_names)
    late = set(build_scope_manifest("lm_late", named_params).selected_param_names)

    assert early & middle == set()
    assert middle & late == set()
    assert early & late == set()

    layer_pattern_hits = [n for n in (early | middle | late)]
    assert len(layer_pattern_hits) > 0
    # every decoder-layer parameter appears in exactly one of the three thirds
    for n in early | middle | late:
        in_count = sum(n in s for s in (early, middle, late))
        assert in_count == 1


def test_lm_early_middle_late_produce_expected_layer_partition_flat(flat_checkpoint_vlm_factory):
    model = flat_checkpoint_vlm_factory()
    named_params = list(model.named_parameters())
    early = build_scope_manifest("lm_early", named_params).selected_param_names
    middle = build_scope_manifest("lm_middle", named_params).selected_param_names
    late = build_scope_manifest("lm_late", named_params).selected_param_names
    assert all(f"model.layers.{i}." in "".join(early) for i in range(4))
    assert all(f"model.layers.{i}." in "".join(middle) for i in range(4, 8))
    assert all(f"model.layers.{i}." in "".join(late) for i in range(8, 12))


def test_lm_early_middle_late_produce_expected_layer_partition_runtime(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    named_params = list(model.named_parameters())
    early = build_scope_manifest("lm_early", named_params).selected_param_names
    middle = build_scope_manifest("lm_middle", named_params).selected_param_names
    late = build_scope_manifest("lm_late", named_params).selected_param_names
    assert all(f"language_model.model.layers.{i}." in "".join(early) for i in range(4))
    assert all(f"language_model.model.layers.{i}." in "".join(middle) for i in range(4, 8))
    assert all(f"language_model.model.layers.{i}." in "".join(late) for i in range(8, 12))


def test_zero_match_scope_hard_fails():
    # Build a manifest input where NOTHING matches "vision_merger" -- no merger present.
    named_params = [("visual.patch_embed.weight", __import__("torch").zeros(2, 2))]
    with pytest.raises(ScopeSelectionError, match="selected zero parameters"):
        build_scope_manifest("vision_merger", named_params)


def test_unknown_scope_name_raises():
    with pytest.raises(ValueError, match="Unknown perturbation scope"):
        build_scope_manifest("not_a_real_scope", [])


def test_all_ten_scopes_are_registered():
    # 7 coarse scopes + vision_early/middle/late (fine-localization-inside-vision-encoder).
    assert len(PERTURBATION_SCOPES) == 10
    assert {"vision_early", "vision_middle", "vision_late"} <= set(PERTURBATION_SCOPES)


# --- storage dedup / alias reporting ---


def test_tied_parameter_selected_and_counted_exactly_once(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    manifest = build_scope_manifest("full_lm", model.named_parameters())
    # embed_tokens.weight and lm_head.weight are tied (same storage) -- must appear once.
    matching = [n for n in manifest.selected_param_names if n.endswith("embed_tokens.weight") or n.endswith("lm_head.weight")]
    assert len(matching) == 1


def test_tied_parameter_alias_reported_when_untied_view_supplied(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    deduped = list(model.named_parameters())
    undeduped = list(model.named_parameters(remove_duplicate=False))
    manifest = build_scope_manifest("full_lm", deduped, alias_named_parameters=undeduped)
    assert "language_model.lm_head.weight" in manifest.aliases


def test_no_aliases_reported_when_alias_view_not_supplied(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    manifest = build_scope_manifest("full_lm", model.named_parameters())
    assert manifest.aliases == []


def test_element_count_and_l2_norm_unaffected_by_tied_duplicate(runtime_wrapped_vlm_factory):
    """A tied parameter counted twice would double total_element_count/base_l2_norm relative
    to what an untied model with the same shapes would report -- assert this doesn't happen
    by checking the element count matches summing each UNIQUE selected tensor exactly once.
    """
    model = runtime_wrapped_vlm_factory()
    manifest = build_scope_manifest("full_lm", model.named_parameters())
    expected_elements = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("visual."))
    assert manifest.total_element_count == expected_elements


# --- compute_relative_l2_sigma ---


def test_relative_l2_sigma_formula():
    # sigma = r * ||theta||_2 / sqrt(d)
    assert compute_relative_l2_sigma(base_l2_norm=10.0, param_count=100, r=0.1) == pytest.approx(0.1 * 10.0 / 10.0)


def test_relative_l2_sigma_scales_linearly_with_r():
    a = compute_relative_l2_sigma(base_l2_norm=5.0, param_count=25, r=0.01)
    b = compute_relative_l2_sigma(base_l2_norm=5.0, param_count=25, r=0.02)
    assert b == pytest.approx(2 * a)


def test_relative_l2_sigma_rejects_non_positive_param_count():
    with pytest.raises(ValueError):
        compute_relative_l2_sigma(base_l2_norm=1.0, param_count=0, r=0.1)


# --- determinism: same seed+scope+scale => deterministic manifest/selection ---


def test_manifest_selection_deterministic_across_calls(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    named_params = list(model.named_parameters())
    m1 = build_scope_manifest("lm_middle", named_params)
    m2 = build_scope_manifest("lm_middle", named_params)
    assert m1.selected_param_names == m2.selected_param_names
    assert m1.total_element_count == m2.total_element_count
    assert m1.base_l2_norm == m2.base_l2_norm


# --- scope_requires_encoder_cache_reset ---


@pytest.mark.parametrize("scope", ["vision_encoder", "vision_merger", "full_vlm", "vision_early", "vision_middle", "vision_late"])
def test_visual_affecting_scopes_require_encoder_cache_reset(scope):
    assert scope_requires_encoder_cache_reset(scope) is True


@pytest.mark.parametrize("scope", ["full_lm", "lm_early", "lm_middle", "lm_late"])
def test_lm_only_scopes_do_not_require_encoder_cache_reset(scope):
    assert scope_requires_encoder_cache_reset(scope) is False


def test_scope_requires_encoder_cache_reset_covers_every_registered_scope():
    # Every scope in PERTURBATION_SCOPES must have an explicit answer -- none silently
    # defaulted -- proven by checking the function doesn't raise for any of them.
    for scope in PERTURBATION_SCOPES:
        scope_requires_encoder_cache_reset(scope)  # should not raise


def test_scope_requires_encoder_cache_reset_rejects_unknown_scope():
    with pytest.raises(ValueError, match="Unknown perturbation scope"):
        scope_requires_encoder_cache_reset("not_a_real_scope")


# --- discover_vision_block_indices / partition_vision_blocks (vision_early/middle/late) ---


def test_discover_vision_block_indices_finds_complete_32_block_set(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    assert discover_vision_block_indices(names) == list(range(32))


def test_discover_vision_block_indices_hard_fails_on_incomplete_set():
    with pytest.raises(ScopeSelectionError, match="complete 32 contiguous"):
        discover_vision_block_indices(["visual.blocks.0.weight", "visual.blocks.1.weight"])


def test_discover_vision_block_indices_hard_fails_on_zero_matches():
    with pytest.raises(ScopeSelectionError, match="No parameter names matched"):
        discover_vision_block_indices(["model.layers.0.weight", "visual.merger.weight"])


def test_partition_vision_blocks_matches_requested_11_11_10_bounds():
    thirds = partition_vision_blocks(list(range(32)))
    assert thirds["early"] == list(range(0, 11))
    assert thirds["middle"] == list(range(11, 22))
    assert thirds["late"] == list(range(22, 32))


def test_partition_vision_blocks_hard_fails_on_incomplete_set():
    with pytest.raises(ScopeSelectionError, match="requires exactly the complete"):
        partition_vision_blocks(list(range(30)))


# --- vision_early / vision_middle / vision_late scope selection ---


def test_vision_early_selects_patch_embed_rotary_and_blocks_0_to_10(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    manifest = build_scope_manifest("vision_early", model.named_parameters())
    names = manifest.selected_param_names
    assert any(n.startswith("visual.patch_embed.") for n in names)
    # Fixture's rotary_pos_emb is a trainable nn.Linear -- must be assigned to vision_early.
    assert any(n.startswith("visual.rotary_pos_emb.") for n in names)
    for i in range(11):
        assert any(n.startswith(f"visual.blocks.{i}.") for n in names), f"block {i} missing from vision_early"
    for i in range(11, 32):
        assert not any(n.startswith(f"visual.blocks.{i}.") for n in names), f"block {i} leaked into vision_early"


def test_vision_middle_selects_exactly_blocks_11_to_21(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    manifest = build_scope_manifest("vision_middle", model.named_parameters())
    names = manifest.selected_param_names
    assert not any(n.startswith("visual.patch_embed.") for n in names)
    assert not any(n.startswith("visual.rotary_pos_emb.") for n in names)
    for i in range(11, 22):
        assert any(n.startswith(f"visual.blocks.{i}.") for n in names), f"block {i} missing from vision_middle"
    for i in list(range(0, 11)) + list(range(22, 32)):
        assert not any(n.startswith(f"visual.blocks.{i}.") for n in names), f"block {i} leaked into vision_middle"


def test_vision_late_selects_exactly_blocks_22_to_31(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    manifest = build_scope_manifest("vision_late", model.named_parameters())
    names = manifest.selected_param_names
    assert not any(n.startswith("visual.patch_embed.") for n in names)
    assert not any(n.startswith("visual.rotary_pos_emb.") for n in names)
    for i in range(22, 32):
        assert any(n.startswith(f"visual.blocks.{i}.") for n in names), f"block {i} missing from vision_late"
    for i in range(0, 22):
        assert not any(n.startswith(f"visual.blocks.{i}.") for n in names), f"block {i} leaked into vision_late"


def test_vision_thirds_exclude_merger_and_lm(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named_params = list(model.named_parameters())
    for scope in ("vision_early", "vision_middle", "vision_late"):
        manifest = build_scope_manifest(scope, named_params)
        for name in manifest.selected_param_names:
            assert "merger" not in name, f"{scope} leaked merger param {name}"
            assert not name.startswith("language_model."), f"{scope} leaked LM param {name}"


def test_vision_thirds_pairwise_disjoint(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named_params = list(model.named_parameters())
    early = set(build_scope_manifest("vision_early", named_params).selected_param_names)
    middle = set(build_scope_manifest("vision_middle", named_params).selected_param_names)
    late = set(build_scope_manifest("vision_late", named_params).selected_param_names)

    assert early & middle == set()
    assert middle & late == set()
    assert early & late == set()


def test_vision_thirds_union_equals_vision_encoder_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named_params = list(model.named_parameters())
    early = set(build_scope_manifest("vision_early", named_params).selected_param_names)
    middle = set(build_scope_manifest("vision_middle", named_params).selected_param_names)
    late = set(build_scope_manifest("vision_late", named_params).selected_param_names)
    vision_encoder = set(build_scope_manifest("vision_encoder", named_params).selected_param_names)

    assert early | middle | late == vision_encoder


def test_vision_thirds_hard_fail_on_incomplete_block_set(runtime_wrapped_vlm_factory):
    # The 2-block fixture cannot satisfy the fixed 11/11/10 partition's 32-block requirement.
    model = runtime_wrapped_vlm_factory()
    with pytest.raises(ScopeSelectionError, match="complete 32 contiguous"):
        build_scope_manifest("vision_early", model.named_parameters())


def test_vision_early_relative_l2_sigma_derived_from_its_own_manifest(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    named_params = list(model.named_parameters())
    early_manifest = build_scope_manifest("vision_early", named_params)
    encoder_manifest = build_scope_manifest("vision_encoder", named_params)

    # vision_early is a strict subset of vision_encoder, so its own norm/dimension (and
    # therefore its derived sigma at the same r) must differ from vision_encoder's -- proves
    # the sigma is derived from the SUB-scope's manifest, not silently inherited from the
    # parent vision_encoder scope.
    assert early_manifest.total_element_count < encoder_manifest.total_element_count
    sigma_early = compute_relative_l2_sigma(early_manifest.base_l2_norm, early_manifest.total_element_count, r=0.04)
    sigma_encoder = compute_relative_l2_sigma(encoder_manifest.base_l2_norm, encoder_manifest.total_element_count, r=0.04)
    assert sigma_early != pytest.approx(sigma_encoder)
