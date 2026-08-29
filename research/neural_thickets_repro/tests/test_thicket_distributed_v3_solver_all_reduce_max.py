"""Tests for thicket.distributed_v3_solver.all_reduce_max -- the live-verified device-placement
fix (real 4xL40S TP=4 32B distributed-v3-solver probe hit the identical `RuntimeError: No
backend type associated with device type cpu` already fixed once in distributed_perturbation.
torch_distributed_all_reduce_sum, but missed here since it's a separate function in a separate
module). Mirrors test_thicket_distributed_perturbation_all_reduce.py's own coverage exactly.
"""
from __future__ import annotations

import torch

from neural_thickets_repro.thicket.distributed_v3_solver import all_reduce_max


def test_all_reduce_max_places_tensor_on_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    captured = {}

    def _fake_all_reduce(tensor, op, group):
        captured["device"] = tensor.device
        tensor.fill_(5.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    result = all_reduce_max(3.5, process_group=None)
    assert captured["device"].type == "cpu"
    assert result == 5.0


def test_all_reduce_max_places_tensor_on_current_cuda_device_when_available(monkeypatch):
    """The live bug this test guards against -- see module docstring."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    captured = {}
    real_tensor = torch.tensor

    def _fake_tensor(data, *, dtype=None, device=None):
        captured["device"] = device
        return real_tensor(data, dtype=dtype)

    monkeypatch.setattr(torch, "tensor", _fake_tensor)

    def _fake_all_reduce(tensor, op, group):
        pass

    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    all_reduce_max(1.0, process_group=None)
    assert captured["device"] == torch.device("cuda", 0)
