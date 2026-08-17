"""Synthetic dummy VLM fixture for Gate-0 scaffold tests.

Named to be representative of the real Qwen2.5-VL-3B-Instruct parameter-name shape that we
confirmed by fetching its model.safetensors.index.json from the HF hub (metadata only, no
weight download): vision tower tensors are prefixed `visual.*` (visual.blocks.*,
visual.merger.*, visual.patch_embed.*), everything else is `model.embed_tokens.*` /
`model.layers.*` / `model.norm` (no separate lm_head tensor -- the real checkpoint ties
word embeddings). This fixture is NOT a claim that it matches the real model architecturally
in any other way (sizes, layer count, attention structure) -- it exists purely to exercise
the perturbation-scope/math/selection/voting logic against something CPU-sized.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _DummyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])


class _DummyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])
        self.norm = nn.LayerNorm(4)


class DummyVLM(nn.Module):
    """Mirrors the real checkpoint's top-level split: self.visual.* / self.model.*"""

    def __init__(self):
        super().__init__()
        self.visual = _DummyVisual()
        self.model = _DummyLanguageModel()

    def forward(self, x):  # pragma: no cover - not exercised, just needed to be a valid nn.Module
        return x


@pytest.fixture
def dummy_vlm() -> DummyVLM:
    torch.manual_seed(0)
    return DummyVLM()


@pytest.fixture
def dummy_vlm_factory():
    """Factory so a test can build multiple independent instances with identical init state."""

    def _make() -> DummyVLM:
        torch.manual_seed(0)
        return DummyVLM()

    return _make
