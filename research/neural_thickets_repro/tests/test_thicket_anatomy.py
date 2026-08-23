import re

import pytest

from neural_thickets_repro.thicket.anatomy import (
    AnatomyDiscoveryError,
    AnatomyValidationError,
    build_anatomy_atlas,
    classify_attention_or_mlp,
    compute_mask_hash,
    discover_contiguous_block_indices,
    partition_into_thirds,
    validate_atlas,
)


# --- partition_into_thirds: the generic, count-agnostic depth-band rule (spec B2) ------------


def test_partition_into_thirds_divisible_by_three():
    result = partition_into_thirds(list(range(12)))
    assert result == {"early": [0, 1, 2, 3], "middle": [4, 5, 6, 7], "late": [8, 9, 10, 11]}


def test_partition_into_thirds_matches_established_32_block_11_11_10_convention():
    """Confirms the generic rule reproduces scopes.py's existing hardcoded vision_early/
    middle/late 11/11/10 split for the real 32-block vision encoder -- not merely
    coincidentally similar, but the exact same boundaries.
    """
    result = partition_into_thirds(list(range(32)))
    assert result["early"] == list(range(0, 11))
    assert result["middle"] == list(range(11, 22))
    assert result["late"] == list(range(22, 32))


def test_partition_into_thirds_remainder_one():
    result = partition_into_thirds(list(range(7)))
    assert [len(result[k]) for k in ("early", "middle", "late")] == [3, 2, 2]
    assert result["early"] + result["middle"] + result["late"] == list(range(7))


def test_partition_into_thirds_is_deterministic_regardless_of_input_order():
    assert partition_into_thirds([5, 1, 3, 0, 2, 4]) == partition_into_thirds([0, 1, 2, 3, 4, 5])


def test_partition_into_thirds_rejects_fewer_than_three():
    with pytest.raises(AnatomyDiscoveryError):
        partition_into_thirds([0, 1])


# --- discover_contiguous_block_indices --------------------------------------------------------


def test_discover_contiguous_block_indices_happy_path():
    pattern = re.compile(r"^visual\.blocks\.(\d+)\.")
    names = [f"visual.blocks.{i}.weight" for i in range(5)] + ["visual.merger.weight"]
    assert discover_contiguous_block_indices(names, pattern) == [0, 1, 2, 3, 4]


def test_discover_contiguous_block_indices_raises_on_no_match():
    pattern = re.compile(r"^visual\.blocks\.(\d+)\.")
    with pytest.raises(AnatomyDiscoveryError):
        discover_contiguous_block_indices(["visual.merger.weight"], pattern)


def test_discover_contiguous_block_indices_raises_on_gap():
    pattern = re.compile(r"^visual\.blocks\.(\d+)\.")
    names = ["visual.blocks.0.weight", "visual.blocks.2.weight"]  # missing index 1
    with pytest.raises(AnatomyDiscoveryError):
        discover_contiguous_block_indices(names, pattern)


# --- classify_attention_or_mlp (Level 3, structural only) -------------------------------------


def test_classify_attention_or_mlp():
    assert classify_attention_or_mlp("visual.blocks.3.attn.qkv.weight") == "attention"
    assert classify_attention_or_mlp("model.layers.5.self_attn.q_proj.weight") == "attention"
    assert classify_attention_or_mlp("model.layers.5.mlp.gate_proj.weight") == "mlp"
    assert classify_attention_or_mlp("model.layers.5.input_layernorm.weight") is None


# --- compute_mask_hash: stable, order-independent -----------------------------------------------


def test_compute_mask_hash_is_order_independent():
    assert compute_mask_hash(["b", "a"]) == compute_mask_hash(["a", "b"])


def test_compute_mask_hash_changes_with_membership():
    assert compute_mask_hash(["a", "b"]) != compute_mask_hash(["a", "b", "c"])


# --- build_anatomy_atlas + validate_atlas: full hierarchy against a real (fixture) VLM --------


def test_build_anatomy_atlas_discovers_full_hierarchy(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]

    atlas = build_anatomy_atlas(names, model_family="qwen2_5_vl")

    assert set(atlas.regions) >= {
        "full_model", "vision", "multimodal_connector_or_merger", "language",
        "vision_early", "vision_middle", "vision_late",
        "language_early", "language_middle", "language_late",
    }
    assert atlas.vision_block_indices == tuple(range(32))
    assert atlas.lm_layer_indices == tuple(range(12))
    assert atlas.lm_namespace_convention == "runtime_wrapped"

    # 32 blocks -> 11/11/10, matching the established convention.
    assert len(atlas.region("vision_early").param_names) >= 11  # >= : early also owns patch_embed/rotary_pos_emb
    assert len(atlas.region("vision_late").param_names) == 10

    # 12 layers -> 4/4/4.
    assert len(atlas.region("language_early").param_names) == 4
    assert len(atlas.region("language_middle").param_names) == 4
    assert len(atlas.region("language_late").param_names) == 4


def test_full_model_contains_every_name(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names)
    assert set(atlas.region("full_model").param_names) == set(names)


def test_level1_regions_partition_full_model_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names)

    union = set(atlas.region("vision").param_names) | set(atlas.region("multimodal_connector_or_merger").param_names) | set(atlas.region("language").param_names)
    assert union == set(names)

    vision = set(atlas.region("vision").param_names)
    connector = set(atlas.region("multimodal_connector_or_merger").param_names)
    language = set(atlas.region("language").param_names)
    assert vision & connector == set()
    assert vision & language == set()
    assert connector & language == set()


def test_level2_vision_bands_are_disjoint_and_within_vision(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names)

    vision = set(atlas.region("vision").param_names)
    early = set(atlas.region("vision_early").param_names)
    middle = set(atlas.region("vision_middle").param_names)
    late = set(atlas.region("vision_late").param_names)
    assert early & middle == set()
    assert early & late == set()
    assert middle & late == set()
    assert (early | middle | late) <= vision


def test_level2_language_bands_are_disjoint_and_within_language(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names)

    language = set(atlas.region("language").param_names)
    early = set(atlas.region("language_early").param_names)
    middle = set(atlas.region("language_middle").param_names)
    late = set(atlas.region("language_late").param_names)
    assert early & middle == set()
    assert early & late == set()
    assert middle & late == set()
    assert (early | middle | late) <= language


def test_atlas_regions_have_stable_mask_hashes(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    atlas_1 = build_anatomy_atlas(names)
    atlas_2 = build_anatomy_atlas(list(reversed(names)))
    for region_name in atlas_1.regions:
        assert atlas_1.region(region_name).mask_hash == atlas_2.region(region_name).mask_hash


def test_build_anatomy_atlas_raises_with_too_few_vision_blocks(flat_checkpoint_vlm_factory):
    """FlatCheckpointVLM's synthetic vision tower has only 2 blocks -- not enough to form
    three non-empty contiguous bands -- must hard-fail, never silently degrade to e.g. 2 bands.
    """
    model = flat_checkpoint_vlm_factory()
    names = [n for n, _ in model.named_parameters()]
    with pytest.raises(AnatomyDiscoveryError):
        build_anatomy_atlas(names)


def test_validate_atlas_passes_on_a_correctly_built_atlas(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names)
    report = validate_atlas(atlas)
    assert report.ok
    assert report.empty_regions == ()
    assert report.sibling_overlaps == {}
    # language's own embed_tokens/norm/lm_head sit outside any numbered layer -> reported.
    assert "language" in report.uncovered_by_parent


def test_validate_atlas_raises_on_unexpectedly_empty_region():
    from neural_thickets_repro.thicket.anatomy import AnatomyAtlas, AnatomyRegion

    empty_region = AnatomyRegion(name="vision", level=1, parent="full_model", param_names=(), mask_hash=compute_mask_hash(()))
    atlas = AnatomyAtlas(model_family="x", regions={"vision": empty_region}, lm_namespace_convention=None, lm_layer_indices=(), vision_block_indices=())
    with pytest.raises(AnatomyValidationError):
        validate_atlas(atlas)


def test_validate_atlas_raises_on_sibling_overlap():
    from neural_thickets_repro.thicket.anatomy import AnatomyAtlas, AnatomyRegion

    a = AnatomyRegion(name="a", level=1, parent="root", param_names=("shared", "only_a"), mask_hash=compute_mask_hash(("shared", "only_a")))
    b = AnatomyRegion(name="b", level=1, parent="root", param_names=("shared", "only_b"), mask_hash=compute_mask_hash(("shared", "only_b")))
    atlas = AnatomyAtlas(model_family="x", regions={"a": a, "b": b}, lm_namespace_convention=None, lm_layer_indices=(), vision_block_indices=())
    with pytest.raises(AnatomyValidationError):
        validate_atlas(atlas)


def test_validate_atlas_deterministic_across_repeated_calls(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names)
    report_1 = validate_atlas(atlas)
    report_2 = validate_atlas(atlas)
    assert report_1 == report_2
