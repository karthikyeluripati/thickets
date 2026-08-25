"""Tests for thicket/anatomy_stage9.py -- the Stage-9 hierarchical (L2) child-region partition.
CPU-only, hand-built parameter-name fixtures (no GPU/torch model construction needed beyond the
existing conftest fixtures already used for L1 anatomy tests).
"""
import pytest
import torch.nn as nn

from neural_thickets_repro.thicket.anatomy_stage9 import (
    STAGE9_CHILD_REGIONS,
    STAGE9_DRILLDOWN_PARENTS,
    Stage9PartitionError,
    build_stage9_hierarchical_partition,
    ensure_stage9_partition_valid,
)


class _DummyVisual32Blocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(32)])
        self.merger = nn.Linear(4, 4, bias=False)
        self.rotary_pos_emb = nn.Linear(4, 4, bias=False)


class _LangLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Linear(4, 4, bias=False)


class _LangModelInner(nn.Module):
    def __init__(self, n=36):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([_LangLayer() for _ in range(n)])
        self.norm = nn.Linear(4, 4, bias=False)


class _LangModel(nn.Module):
    def __init__(self, n=36):
        super().__init__()
        self.model = _LangModelInner(n)


class _VLM(nn.Module):
    def __init__(self, n_layers=36):
        super().__init__()
        self.visual = _DummyVisual32Blocks()
        self.language_model = _LangModel(n_layers)


def _real_shaped_param_names(n_layers=36):
    return [n for n, _ in _VLM(n_layers).named_parameters()]


def test_stage9_produces_exactly_six_frozen_child_regions():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    assert set(children.keys()) == set(STAGE9_CHILD_REGIONS)
    assert len(STAGE9_CHILD_REGIONS) == 6


def test_stage9_drilldown_parents_are_exactly_vision_and_language():
    assert STAGE9_DRILLDOWN_PARENTS == ("vision", "language")
    assert "multimodal_connector_or_merger" not in STAGE9_DRILLDOWN_PARENTS


def test_language_embed_tokens_assigned_to_early():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    assert any("embed_tokens" in n for n in children["language_early"].param_names)
    for band in ("language_mid", "language_late"):
        assert not any("embed_tokens" in n for n in children[band].param_names)


def test_language_final_norm_assigned_to_late():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    # The TOP-level model.norm (not a per-layer norm) must land in language_late.
    top_level_norm = [n for n in children["language_late"].param_names if n.endswith(".model.norm.weight")]
    assert len(top_level_norm) == 1
    for band in ("language_early", "language_mid"):
        assert not any(n.endswith(".model.norm.weight") for n in children[band].param_names)


def test_language_children_union_equals_language_parent():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    assert audits["language"].union_equals_parent is True


def test_language_children_pairwise_disjoint():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    assert audits["language"].children_pairwise_disjoint is True
    early, mid, late = children["language_early"], children["language_mid"], children["language_late"]
    assert not (set(early.param_names) & set(mid.param_names))
    assert not (set(mid.param_names) & set(late.param_names))
    assert not (set(early.param_names) & set(late.param_names))


def test_vision_children_union_equals_vision_parent():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    assert audits["vision"].union_equals_parent is True
    assert audits["vision"].uncovered_tensors == ()  # confirmed zero-gap, audited not assumed


def test_vision_children_pairwise_disjoint():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    assert audits["vision"].children_pairwise_disjoint is True


def test_vision_early_owns_patch_embed_and_rotary_pos_emb():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    early_names = children["vision_early"].param_names
    assert any("patch_embed" in n for n in early_names)
    assert any("rotary_pos_emb" in n for n in early_names)


def test_connector_never_included_in_any_stage9_child():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    for region in children.values():
        assert not any("merger" in n.lower() for n in region.param_names)


def test_ensure_stage9_partition_valid_passes_on_a_correct_partition():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    ensure_stage9_partition_valid(children, audits)  # must not raise


def test_ensure_stage9_partition_valid_detects_missing_region():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    del children["vision_mid"]
    with pytest.raises(Stage9PartitionError):
        ensure_stage9_partition_valid(children, audits)


def test_ensure_stage9_partition_valid_detects_connector_leak():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    from neural_thickets_repro.thicket.anatomy import AnatomyRegion, compute_mask_hash

    leaked_names = tuple(sorted(children["vision_early"].param_names + ("visual.merger.weight",)))
    children["vision_early"] = AnatomyRegion(
        name="vision_early", level=2, parent="vision", param_names=leaked_names, mask_hash=compute_mask_hash(leaked_names),
    )
    with pytest.raises(Stage9PartitionError):
        ensure_stage9_partition_valid(children, audits)


def test_classify_uncovered_tensor_hard_fails_on_an_unclassifiable_name():
    from neural_thickets_repro.thicket.anatomy_stage9 import _classify_uncovered_tensor

    with pytest.raises(Stage9PartitionError):
        _classify_uncovered_tensor("language_model.model.some_mystery_tensor.weight")


def test_partition_is_deterministic_across_repeated_calls():
    names = _real_shaped_param_names()
    children1, audits1 = build_stage9_hierarchical_partition(names)
    children2, audits2 = build_stage9_hierarchical_partition(names)
    for region_name in STAGE9_CHILD_REGIONS:
        assert children1[region_name].param_names == children2[region_name].param_names
        assert children1[region_name].mask_hash == children2[region_name].mask_hash


def test_36_layers_splits_into_exact_thirds_of_12():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names(n_layers=36))
    for band, expected_layer_count in (("language_early", 12), ("language_mid", 12), ("language_late", 12)):
        n_layer_tensors = sum(1 for n in children[band].param_names if ".layers." in n)
        # 2 tensors per layer (self_attn, mlp) in this fixture
        assert n_layer_tensors == expected_layer_count * 2


def test_vision_32_blocks_splits_into_11_11_10():
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    counts = {}
    for band in ("vision_early", "vision_mid", "vision_late"):
        block_indices = {int(n.split("blocks.")[1].split(".")[0]) for n in children[band].param_names if "blocks." in n}
        counts[band] = len(block_indices)
    assert counts == {"vision_early": 11, "vision_mid": 11, "vision_late": 10}
