"""Tests for thicket.distributed_perturbation.torch_distributed_all_reduce_sum -- specifically
the live-verified device-placement fix (real 4xL40S TP=4 32B run hit `RuntimeError: No backend
type associated with device type cpu` when the reduction tensor defaulted to CPU against vLLM's
NCCL-only default process group). No real distributed runtime is available in CI/local CPU
tests, so `torch.distributed.all_reduce` and CUDA availability are monkeypatched -- this verifies
device SELECTION logic, not real collective communication (that remains live-hardware-only, per
this module's own established convention).
"""
from __future__ import annotations

import torch

from neural_thickets_repro.thicket.distributed_perturbation import torch_distributed_all_reduce_sum


def test_all_reduce_places_tensor_on_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    captured = {}

    def _fake_all_reduce(tensor, op, group):
        captured["device"] = tensor.device
        tensor.mul_(2)  # stand-in for a real SUM reduction across ranks

    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    result = torch_distributed_all_reduce_sum(3.5, process_group=None)
    assert captured["device"].type == "cpu"
    assert result == 7.0


def test_all_reduce_places_tensor_on_current_cuda_device_when_available(monkeypatch):
    """The live bug this test guards against: a bare torch.tensor(...) defaults to CPU even
    inside a GPU worker process, which fails against vLLM's NCCL-only default process group.
    This machine's torch build has no CUDA support at all (real CUDA tensor allocation would
    itself raise), so `torch.tensor` is monkeypatched to capture the requested device without
    actually allocating -- this verifies the device SELECTED, not real CUDA allocation (that part
    is exercised only by the live 4xL40S run).
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    captured = {}
    real_tensor = torch.tensor

    def _fake_tensor(data, *, dtype=None, device=None):
        captured["device"] = device
        return real_tensor(data, dtype=dtype)  # allocate for real on CPU so .item()/.mul_ still work

    monkeypatch.setattr(torch, "tensor", _fake_tensor)

    def _fake_all_reduce(tensor, op, group):
        pass

    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    torch_distributed_all_reduce_sum(1.0, process_group=None)
    assert captured["device"] == torch.device("cuda", 0)


def test_all_reduce_returns_the_reduced_value(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _fake_all_reduce(tensor, op, group):
        tensor.fill_(9.25)

    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    assert torch_distributed_all_reduce_sum(0.0, process_group=None) == 9.25
