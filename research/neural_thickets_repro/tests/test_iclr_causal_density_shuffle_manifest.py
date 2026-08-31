"""Tests for iclr_causal_density.shuffle_manifest -- deterministic shuffled-image derangement
(item 3), no self-maps (item 4), actual image identity changes (item 5), text-only passes no
image (item 6). CPU-only.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from neural_thickets_repro.benchmarks.base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from neural_thickets_repro.benchmarks.image_sanity import make_text_only_variant
from neural_thickets_repro.iclr_causal_density.shuffle_manifest import (
    ShuffleManifestIntegrityError,
    build_frozen_shuffled_variant,
    load_shuffle_manifest,
    write_shuffle_manifest,
)


class _FakeBenchmark(CapabilityBenchmark):
    capability = "fake"
    name = "fake"

    def load_examples(self, cfg: Any) -> List[Example]:
        return []

    def build_prompt(self, example: Example) -> List[dict]:
        return [{"role": "user", "content": "?"}]

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        return ParsedPrediction(parsed=raw_generation, parse_ok=True)

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        return ExampleScore(score=1.0)

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        return {"primary_metric": 1.0, "parser_failure_rate": 0.0}


def _examples(n=30):
    return [Example(example_id=f"ex_{i}", image=f"img_{i}", image_ref=f"img_{i}", prompt_input={}, target=i) for i in range(n)]


def test_shuffled_variant_is_deterministic():
    benchmark = _FakeBenchmark()
    ex = _examples()
    shuffled_1, manifest_1 = build_frozen_shuffled_variant(ex, "fake", "audit", benchmark, seed=7)
    shuffled_2, manifest_2 = build_frozen_shuffled_variant(ex, "fake", "audit", benchmark, seed=7)
    assert [e.image_ref for e in shuffled_1] == [e.image_ref for e in shuffled_2]
    assert manifest_1 == manifest_2


def test_no_shuffled_self_maps():
    benchmark = _FakeBenchmark()
    ex = _examples()
    shuffled, manifest = build_frozen_shuffled_variant(ex, "fake", "audit", benchmark, seed=7)
    for orig, shuf in zip(ex, shuffled):
        assert shuf.image_ref != orig.image_ref  # no example receives its own original image


def test_actual_image_identity_changes_and_is_recorded():
    benchmark = _FakeBenchmark()
    ex = _examples()
    shuffled, manifest = build_frozen_shuffled_variant(ex, "fake", "selection", benchmark, seed=3)
    assert len(manifest.mappings) == len(ex)
    for m in manifest.mappings:
        assert m.shuffled_image_ref != m.original_image_ref
        assert m.shuffled_image_ref.startswith("shuffled_from:")


def test_shuffled_variant_preserves_example_order_and_targets():
    benchmark = _FakeBenchmark()
    ex = _examples()
    shuffled, _ = build_frozen_shuffled_variant(ex, "fake", "audit", benchmark, seed=3)
    assert [e.example_id for e in shuffled] == [e.example_id for e in ex]
    assert [e.target for e in shuffled] == [e.target for e in ex]  # scored against the ORIGINAL target


def test_shuffle_manifest_roundtrip(tmp_path):
    benchmark = _FakeBenchmark()
    ex = _examples()
    _, manifest = build_frozen_shuffled_variant(ex, "fake", "audit", benchmark, seed=3)
    path = tmp_path / "shuffle.json"
    write_shuffle_manifest({"fake:audit": manifest}, path)
    loaded = load_shuffle_manifest(path)
    assert loaded["fake:audit"] == manifest


def test_integrity_error_when_shuffled_image_unchanged(monkeypatch):
    """If the underlying make_shuffled_variant somehow returned a non-shuffled variant (e.g. a
    benchmark whose make_shuffled_image_variant is buggy and echoes the same image), the
    manifest builder must hard-fail, never silently persist a fake 'shuffled' condition.
    """
    class _BrokenBenchmark(_FakeBenchmark):
        def make_shuffled_image_variant(self, example, source_example):
            return example  # BUG: echoes the original, image_ref never changes

    ex = _examples(n=5)
    with pytest.raises(ShuffleManifestIntegrityError):
        build_frozen_shuffled_variant(ex, "fake", "audit", _BrokenBenchmark(), seed=1)


def test_text_only_passes_no_image():
    """Item 6: text-only variant carries image=None, prompt/target unchanged."""
    ex = _examples(n=5)
    text_only = make_text_only_variant(ex)
    assert all(e.image is None for e in text_only)
    assert [e.example_id for e in text_only] == [e.example_id for e in ex]
    assert [e.target for e in text_only] == [e.target for e in ex]
    assert all(e.image_ref == "text_only" for e in text_only)
