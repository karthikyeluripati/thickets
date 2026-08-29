"""Tests for thicket.vllm_shard_mapping -- includes the live-verified correction (real 4xL40S
TP=4 32B run) that VocabParallelEmbedding/ParallelLMHead legitimately set BOTH output_dim and
input_dim, with output_dim authoritative -- see the module's own docstring for the full account.
"""
from __future__ import annotations

import pytest
import torch

from neural_thickets_repro.thicket.vllm_shard_mapping import (
    AmbiguousShardMappingError,
    build_shard_spec_from_attributes,
)


def test_column_parallel_style_output_dim_only():
    spec = build_shard_spec_from_attributes(torch.Size([4, 8]), output_dim=0, input_dim=None, tp_size=4, tp_rank=1, param_name="attn.qkv.weight")
    assert spec.dim == 0
    assert spec.global_shape == torch.Size([16, 8])
    assert spec.local_offset == 4
    assert spec.local_size == 4
    assert spec.world_size == 4
    assert spec.rank == 1
    assert not spec.is_replicated


def test_row_parallel_style_input_dim_only():
    spec = build_shard_spec_from_attributes(torch.Size([8, 4]), output_dim=None, input_dim=1, tp_size=4, tp_rank=2, param_name="attn.o_proj.weight")
    assert spec.dim == 1
    assert spec.global_shape == torch.Size([8, 16])
    assert spec.local_offset == 8
    assert spec.local_size == 4


def test_replicated_no_dim_attributes():
    spec = build_shard_spec_from_attributes(torch.Size([8]), output_dim=None, input_dim=None, tp_size=4, tp_rank=1, param_name="norm.weight")
    assert spec.is_replicated
    assert spec.global_shape == torch.Size([8])
    assert spec.world_size == 4
    assert spec.rank == 1


def test_replicated_at_tp_size_1():
    spec = build_shard_spec_from_attributes(torch.Size([8]), output_dim=None, input_dim=None, tp_size=1, tp_rank=0, param_name="norm.weight")
    assert spec.is_replicated
    assert spec.world_size == 1


def test_both_dims_set_prefers_output_dim_vocab_embedding_case():
    """Live-verified case: language_model.lm_head.weight on a real Qwen2.5-VL-32B TP=4 run
    reported output_dim=0, input_dim=1 -- shape [local_vocab_shard, hidden_size]. output_dim (the
    vocab/row dimension) must be treated as the sharded one, never raise, and never silently pick
    input_dim instead (that would misinterpret hidden_size as the sharded dimension).
    """
    local_vocab_shard, hidden_size, tp_size = 38016, 5120, 4  # realistic proportions, not the real vocab size
    spec = build_shard_spec_from_attributes(
        torch.Size([local_vocab_shard, hidden_size]), output_dim=0, input_dim=1, tp_size=tp_size, tp_rank=2, param_name="language_model.lm_head.weight",
    )
    assert spec.dim == 0
    assert spec.global_shape == torch.Size([local_vocab_shard * tp_size, hidden_size])
    assert spec.local_offset == local_vocab_shard * 2
    assert spec.local_size == local_vocab_shard
    assert not spec.is_replicated


def test_both_dims_set_at_world_size_1_is_also_not_ambiguous():
    spec = build_shard_spec_from_attributes(torch.Size([100, 16]), output_dim=0, input_dim=1, tp_size=1, tp_rank=0, param_name="lm_head.weight")
    assert spec.dim == 0
    assert spec.global_shape == torch.Size([100, 16])
    assert spec.world_size == 1


def test_dim_out_of_range_still_hard_fails():
    with pytest.raises(AmbiguousShardMappingError):
        build_shard_spec_from_attributes(torch.Size([4, 8]), output_dim=5, input_dim=None, tp_size=4, tp_rank=0, param_name="bad.weight")


def test_tp_size_below_1_with_a_dim_set_still_hard_fails():
    with pytest.raises(AmbiguousShardMappingError):
        build_shard_spec_from_attributes(torch.Size([4, 8]), output_dim=0, input_dim=None, tp_size=0, tp_rank=0, param_name="bad.weight")
