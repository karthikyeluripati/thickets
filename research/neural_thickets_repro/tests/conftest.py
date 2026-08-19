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


# --- Two namespace-convention fixtures for scopes.py (tests/test_scopes.py) ---
#
# SCOPED_PERTURBATION_DESIGN.md's Phase 1 correction: the HF checkpoint's flat safetensors
# keys (model.layers.*, exercised by DummyVLM above) are NOT necessarily what a loaded/
# vLLM-wrapped runtime module's named_parameters() yields -- prior GPU logs show the real
# runtime module nests the LM under language_model.model.*, with a separate lm_head entry
# even though the checkpoint ties the weight. Both conventions are fixtured here, at a layer
# count (12) divisible by 3 so lm_early/middle/late partitioning is actually exercised (the
# 2-layer DummyVLM above can't be split into three non-empty thirds).

_LM_LAYER_COUNT = 12  # divisible by 3: early=0-3, middle=4-7, late=8-11


class _DummyVisualWithMerger(nn.Module):
    """visual.patch_embed / visual.blocks.N / visual.merger -- unprefixed by
    language_model./model. in either observed convention (see scopes.py).
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])
        self.merger = nn.Linear(4, 4, bias=False)


class _DummyLM12Layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(_LM_LAYER_COUNT)])
        self.norm = nn.LayerNorm(4)


class FlatCheckpointVLM(nn.Module):
    """visual.* / model.embed_tokens.* / model.layers.N.* / model.norm.* -- the HF
    checkpoint's flat safetensors-index naming convention (scopes.LM_NAMESPACE_CONVENTIONS
    "flat_checkpoint"). No separate lm_head -- ties are not modeled in this fixture since the
    checkpoint form doesn't expose one as a separate tensor at all.
    """

    def __init__(self):
        super().__init__()
        self.visual = _DummyVisualWithMerger()
        self.model = _DummyLM12Layers()

    def forward(self, x):  # pragma: no cover - not exercised, just needed to be a valid nn.Module
        return x


class _RuntimeLanguageModelWrapper(nn.Module):
    """model.* nested one level under language_model.*, plus a SEPARATE lm_head Linear whose
    weight is explicitly tied (same underlying nn.Parameter / storage) to
    model.embed_tokens.weight -- mirrors the observed real runtime module shape
    (language_model.model.layers.N.* / language_model.lm_head.*) closely enough to exercise
    scopes.py's storage-dedup and alias-reporting logic (tests/test_scopes.py) against an
    actual tied-parameter case, not just a hypothetical one.
    """

    def __init__(self):
        super().__init__()
        self.model = _DummyLM12Layers()
        self.lm_head = nn.Linear(4, 10, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # tied: same nn.Parameter object


class RuntimeWrappedVLM(nn.Module):
    """visual.* / language_model.model.embed_tokens.* / language_model.model.layers.N.* /
    language_model.model.norm.* / language_model.lm_head.* -- the observed real vLLM-loaded
    runtime module naming convention (scopes.LM_NAMESPACE_CONVENTIONS "runtime_wrapped").
    """

    def __init__(self):
        super().__init__()
        self.visual = _DummyVisualWithMerger()
        self.language_model = _RuntimeLanguageModelWrapper()

    def forward(self, x):  # pragma: no cover - not exercised, just needed to be a valid nn.Module
        return x


@pytest.fixture
def flat_checkpoint_vlm_factory():
    def _make() -> FlatCheckpointVLM:
        torch.manual_seed(0)
        return FlatCheckpointVLM()

    return _make


@pytest.fixture
def runtime_wrapped_vlm_factory():
    def _make() -> RuntimeWrappedVLM:
        torch.manual_seed(0)
        return RuntimeWrappedVLM()

    return _make


# --- 32-vision-block fixture (tests/test_scopes.py, test_scoped_perturbation.py,
# test_scope_isolation_gpu_check.py) -- vision_early/middle/late's fixed 11/11/10 partition
# hard-requires the complete visual.blocks.0..31 index set (scopes.discover_vision_block_
# indices), which the 2-block _DummyVisualWithMerger above cannot exercise. Any test that
# iterates ALL of scopes.PERTURBATION_SCOPES (not just a couple of named scopes) must use
# this fixture instead, now that the registry includes the three vision-third scopes.

_VISION_BLOCK_COUNT = 32  # matches the real Qwen2.5-VL-3B-Instruct vision encoder depth


class _DummyVisual32Blocks(nn.Module):
    """visual.patch_embed / visual.blocks.0..31 / visual.merger / visual.rotary_pos_emb --
    the last one stands in for a trainable rotary_pos_emb parameter so vision_early's "assign
    trainable rotary_pos_emb params to vision_early, if any exist" rule is actually exercised
    by at least one test, not only exercised by its absence.
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(_VISION_BLOCK_COUNT)])
        self.merger = nn.Linear(4, 4, bias=False)
        self.rotary_pos_emb = nn.Linear(4, 4, bias=False)


class RuntimeWrappedVLM32Vision(nn.Module):
    """Same LM shape as RuntimeWrappedVLM (language_model.model.layers.0..11 + tied lm_head),
    but with the full 32-block vision encoder.
    """

    def __init__(self):
        super().__init__()
        self.visual = _DummyVisual32Blocks()
        self.language_model = _RuntimeLanguageModelWrapper()

    def forward(self, x):  # pragma: no cover - not exercised, just needed to be a valid nn.Module
        return x


@pytest.fixture
def runtime_wrapped_vlm_32vision_factory():
    def _make() -> RuntimeWrappedVLM32Vision:
        torch.manual_seed(0)
        return RuntimeWrappedVLM32Vision()

    return _make
