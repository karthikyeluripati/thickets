"""Tests for thicket.distributed_anatomy_audit -- Stage-11 32B LIVE readiness (task spec Section
11: exact live parameter counts under TP>1, computed WITHOUT any cross-rank all-reduce, from each
parameter's reconstructed global_shape).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from neural_thickets_repro.thicket.distributed_anatomy_audit import (
    AnatomyAuditRankConsensusError,
    report_global_anatomy_audit_rpc,
    verify_anatomy_audit_rank_consensus,
)


def _set_tp_attrs(param: torch.nn.Parameter, *, output_dim=None, input_dim=None):
    if output_dim is not None:
        param.output_dim = output_dim
    if input_dim is not None:
        param.input_dim = input_dim
    return param


class _ColumnParallelLike(nn.Module):
    """Mimics vLLM's ColumnParallelLinear: weight already holds only THIS rank's local shard
    (shape reflects local_size, not global_size), output_dim=0 marks the sharded dim, tp_size/
    tp_rank live on the module.
    """

    def __init__(self, local_out: int, in_features: int, *, tp_size: int, tp_rank: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(local_out, in_features))
        _set_tp_attrs(self.weight, output_dim=0)
        self.tp_size = tp_size
        self.tp_rank = tp_rank


class _ReplicatedNormLike(nn.Module):
    """Mimics an RMSNorm weight -- no output_dim/input_dim attribute, replicated across ranks."""

    def __init__(self, dim: int, *, tp_size: int, tp_rank: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.tp_size = tp_size
        self.tp_rank = tp_rank


def _build_fake_tiny_vlm(*, tp_size: int, tp_rank: int, local_out: int = 4, in_features: int = 8) -> nn.Module:
    """A tiny model matching the real qwen2_5_vl naming convention closely enough for
    build_anatomy_atlas to discover it: 3 vision blocks (visual.blocks.{i}., the minimum
    partition_into_thirds needs), a merger (visual.merger.), and 3 LM layers under the
    "runtime_wrapped" convention (language_model.model.layers.{i}., LM_NAMESPACE_CONVENTIONS).
    Each block/layer/merger holds ONE sharded linear (output_dim=0) plus the model gets one
    replicated norm per top-level region -- enough to exercise both aggregation paths for all
    three regions. `local_out` is THIS rank's local shard size; the TRUE global size is
    `local_out * tp_size`.
    """
    model = nn.Module()
    model.visual = nn.Module()
    model.visual.blocks = nn.ModuleList([_ColumnParallelLike(local_out, in_features, tp_size=tp_size, tp_rank=tp_rank) for _ in range(3)])
    model.visual.norm = _ReplicatedNormLike(in_features, tp_size=tp_size, tp_rank=tp_rank)
    model.visual.merger = nn.Module()
    model.visual.merger.mlp = nn.ModuleList([_ColumnParallelLike(local_out, in_features, tp_size=tp_size, tp_rank=tp_rank)])
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList([_ColumnParallelLike(local_out, in_features, tp_size=tp_size, tp_rank=tp_rank) for _ in range(3)])
    model.language_model.model.norm = _ReplicatedNormLike(in_features, tp_size=tp_size, tp_rank=tp_rank)
    return model


def _fake_worker(model: nn.Module, *, rank: int):
    return SimpleNamespace(model_runner=SimpleNamespace(model=model), rank=rank)


REGION_LABELS = ("vision", "multimodal_connector_or_merger", "language")


def test_global_element_counts_reconstructed_from_shard_metadata_not_local_shard_size():
    """At tp_size=4, local_out=4 -- the TRUE global out-dim per sharded linear is 16, never the
    local 4. This is the core correctness property this module exists for.
    """
    model = _build_fake_tiny_vlm(tp_size=4, tp_rank=2, local_out=4, in_features=8)
    worker = _fake_worker(model, rank=2)

    result = report_global_anatomy_audit_rpc(worker, REGION_LABELS, "qwen2_5_vl")

    per_block_global_elements = 16 * 8  # global_out(16) x in_features(8), never local_out(4) x 8
    vision_replicated_elements = 8  # norm weight, full size regardless of TP
    assert result["regions"]["vision"]["n_elements"] == 3 * per_block_global_elements + vision_replicated_elements

    connector_global_elements = per_block_global_elements
    assert result["regions"]["multimodal_connector_or_merger"]["n_elements"] == connector_global_elements

    language_replicated_elements = 8
    assert result["regions"]["language"]["n_elements"] == 3 * per_block_global_elements + language_replicated_elements


def test_regions_sum_to_total_zero_overlap_zero_unassigned():
    model = _build_fake_tiny_vlm(tp_size=4, tp_rank=0, local_out=4, in_features=8)
    worker = _fake_worker(model, rank=0)

    result = report_global_anatomy_audit_rpc(worker, REGION_LABELS, "qwen2_5_vl")

    region_sum = sum(result["regions"][label]["n_elements"] for label in REGION_LABELS)
    assert region_sum == result["total_model_elements"]
    assert result["union_equals_full_model"] is True
    assert result["pairwise_disjoint"] is True
    assert result["uncovered_by_full_model"] == []


def test_every_rank_computes_identical_global_counts_at_world_size_1():
    """world_size=1 is the degenerate case of the same architectural fact: global_shape ==
    local_shape when tp_size=1, so this reduces to exactly report_scaling_anatomy_audit's own
    (already-tested) TP=1 numbers for element counts.
    """
    model = _build_fake_tiny_vlm(tp_size=1, tp_rank=0, local_out=16, in_features=8)
    worker = _fake_worker(model, rank=0)
    result = report_global_anatomy_audit_rpc(worker, REGION_LABELS, "qwen2_5_vl")
    assert result["regions"]["vision"]["n_elements"] == 3 * (16 * 8) + 8


def test_ranks_report_identical_counts_regardless_of_which_rank_asks():
    """The whole point of the global_shape approach: no cross-rank communication needed because
    every rank reconstructs the SAME true global count independently.
    """
    results = [
        report_global_anatomy_audit_rpc(_fake_worker(_build_fake_tiny_vlm(tp_size=4, tp_rank=r, local_out=4, in_features=8), rank=r), REGION_LABELS, "qwen2_5_vl")
        for r in range(4)
    ]
    consensus = verify_anatomy_audit_rank_consensus(results)
    assert consensus["ok"] is True
    assert consensus["n_ranks"] == 4


def test_rank_consensus_hard_fails_on_disagreement():
    results = [
        report_global_anatomy_audit_rpc(_fake_worker(_build_fake_tiny_vlm(tp_size=4, tp_rank=r, local_out=4, in_features=8), rank=r), REGION_LABELS, "qwen2_5_vl")
        for r in range(2)
    ]
    results[1]["total_model_elements"] += 1  # simulate a genuine mapping-bug disagreement
    with pytest.raises(AnatomyAuditRankConsensusError):
        verify_anatomy_audit_rank_consensus(results)


def test_verify_anatomy_audit_rank_consensus_requires_at_least_one_result():
    with pytest.raises(ValueError):
        verify_anatomy_audit_rank_consensus([])
