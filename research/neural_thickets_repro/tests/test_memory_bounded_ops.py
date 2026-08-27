"""Tests for thicket/memory_bounded_ops.py -- CPU-only. Proves (a) the chunked reductions
compute the SAME mathematical quantity as a full-precision reference computed WITHOUT chunking,
and (b) they never materialize a temporary larger than the configured chunk bound, regardless of
the source tensor's total size (the actual property the Stage-11 7B whole_model OOM fix depends
on) -- proven via an instrumented torch.Tensor.double/float that records the largest tensor any
call in this module ever upcasts.
"""
import torch

from neural_thickets_repro.thicket.memory_bounded_ops import (
    DEFAULT_CHUNK_ELEMENTS,
    chunked_abs_stats,
    chunked_squared_l2_diff_sum,
    chunked_squared_l2_sum,
)


def _max_upcast_tracker(monkeypatch):
    max_seen = {"n": 0}
    original_double = torch.Tensor.double

    def _tracking_double(self, *args, **kwargs):
        max_seen["n"] = max(max_seen["n"], self.numel())
        return original_double(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "double", _tracking_double)
    return max_seen


# =================================================================================================
# Correctness -- same mathematical quantity as a full-precision (non-chunked) reference
# =================================================================================================


def test_chunked_squared_l2_sum_matches_full_precision_reference():
    torch.manual_seed(0)
    tensor = (torch.randn(10_000) * 3.7).to(torch.bfloat16)
    reference = float(tensor.double().pow(2).sum().item())
    result = chunked_squared_l2_sum(tensor, chunk_elements=777)
    assert result == reference  # both computed in float64; chunk boundaries don't change the sum's value here


def test_chunked_squared_l2_sum_hand_computable():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert chunked_squared_l2_sum(tensor, chunk_elements=1) == 1.0 + 4.0 + 9.0 + 16.0
    assert chunked_squared_l2_sum(tensor, chunk_elements=2) == 1.0 + 4.0 + 9.0 + 16.0
    assert chunked_squared_l2_sum(tensor, chunk_elements=1000) == 1.0 + 4.0 + 9.0 + 16.0


def test_chunked_squared_l2_diff_sum_hand_computable():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 0.0, 3.0, 1.0])
    expected = 0.0 + 4.0 + 0.0 + 9.0
    assert chunked_squared_l2_diff_sum(a, b, chunk_elements=1) == expected
    assert chunked_squared_l2_diff_sum(a, b, chunk_elements=3) == expected


def test_chunked_squared_l2_diff_sum_rejects_mismatched_shapes():
    a = torch.zeros(4)
    b = torch.zeros(5)
    try:
        chunked_squared_l2_diff_sum(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_chunked_abs_stats_hand_computable():
    a = torch.tensor([1.0, -2.0, 3.5])
    b = torch.tensor([1.0, 0.0, 1.0])
    max_abs, sum_abs = chunked_abs_stats(a, b, chunk_elements=1)
    assert max_abs == 2.5
    assert sum_abs == 0.0 + 2.0 + 2.5


def test_chunked_abs_stats_matches_full_precision_reference():
    torch.manual_seed(1)
    a = (torch.randn(5000) * 2.0).to(torch.bfloat16)
    b = (torch.randn(5000) * 2.0).to(torch.bfloat16)
    ref_diff = (a.double() - b.double()).abs()
    ref_max, ref_sum = float(ref_diff.max().item()), float(ref_diff.sum().item())
    max_abs, sum_abs = chunked_abs_stats(a, b, chunk_elements=333)
    assert max_abs == ref_max
    assert sum_abs == ref_sum


# =================================================================================================
# Bounded temporary -- the property the OOM fix actually depends on
# =================================================================================================


def test_chunked_squared_l2_sum_never_upcasts_more_than_chunk_elements_at_once(monkeypatch):
    torch.manual_seed(0)
    tensor = torch.randn(500_000)  # far larger than the chunk bound below
    max_seen = _max_upcast_tracker(monkeypatch)
    chunked_squared_l2_sum(tensor, chunk_elements=4096)
    assert 0 < max_seen["n"] <= 4096


def test_chunked_squared_l2_diff_sum_never_upcasts_more_than_chunk_elements_at_once(monkeypatch):
    torch.manual_seed(0)
    a = torch.randn(500_000)
    b = torch.randn(500_000)
    max_seen = _max_upcast_tracker(monkeypatch)
    chunked_squared_l2_diff_sum(a, b, chunk_elements=4096)
    assert 0 < max_seen["n"] <= 4096


def test_chunked_abs_stats_never_upcasts_more_than_chunk_elements_at_once(monkeypatch):
    torch.manual_seed(0)
    a = torch.randn(500_000)
    b = torch.randn(500_000)
    max_seen = _max_upcast_tracker(monkeypatch)
    chunked_abs_stats(a, b, chunk_elements=4096)
    assert 0 < max_seen["n"] <= 4096


def test_default_chunk_elements_temporary_is_small_in_bytes():
    """Documents the actual VRAM bound this module was built for: DEFAULT_CHUNK_ELEMENTS
    float64 elements is a small, fixed number of bytes, regardless of source tensor size.
    """
    bytes_per_chunk = DEFAULT_CHUNK_ELEMENTS * 8  # float64
    assert bytes_per_chunk <= 64 * 1024 * 1024  # <= 64 MiB, far below the ~2 GiB that OOM'd


def test_functions_work_on_a_tensor_far_larger_than_any_realistic_gpu_sized_fp32_clone_would_allow():
    """A 20,000,000-element tensor (160MB at its own float32 size) processed with a tiny chunk
    bound -- proves these functions scale to sizes where "clone the whole thing in fp64" (640MB)
    would itself be wasteful, without requiring an actual multi-GB allocation in this test.
    """
    torch.manual_seed(0)
    tensor = torch.randn(20_000_000, dtype=torch.float32)
    result = chunked_squared_l2_sum(tensor, chunk_elements=DEFAULT_CHUNK_ELEMENTS)
    assert result > 0.0
