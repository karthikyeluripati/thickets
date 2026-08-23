"""Tests for adapters/object_recognition_imagenet.py -- synthetic ImageNet-1K-shaped rows,
no real (gated) dataset download / GPU / ray / vllm needed.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.object_recognition_imagenet import (
    ImageNetGatedAccessError,
    ImageNetObjectRecognitionBenchmark,
)
from neural_thickets_repro.benchmarks.base import Example


def _bench():
    return ImageNetObjectRecognitionBenchmark()


def test_capability_and_name():
    bench = _bench()
    assert bench.capability == "object_recognition"
    assert bench.name == "imagenet1k_val"


def test_known_caveats_documents_gated_access():
    assert "gated" in " ".join(_bench().known_caveats()).lower()


def test_synonym_list_match_not_bare_substring():
    bench = _bench()
    example = Example(example_id="1", target="tench, Tinca tinca")
    parsed = bench.parse_prediction("It's a tench, a type of fish.", example)
    score = bench.score_example(parsed, example)
    assert score.correct is True
    assert score.detail["matched_synonym"] == "tench"


def test_second_synonym_also_matches():
    bench = _bench()
    example = Example(example_id="1", target="tench, Tinca tinca")
    parsed = bench.parse_prediction("This appears to be Tinca tinca.", example)
    score = bench.score_example(parsed, example)
    assert score.correct is True


def test_wrong_class_does_not_match():
    bench = _bench()
    example = Example(example_id="1", target="tench, Tinca tinca")
    parsed = bench.parse_prediction("This is a vulture.", example)
    score = bench.score_example(parsed, example)
    assert score.correct is False


def test_short_generic_word_does_not_spuriously_match_via_raw_substring():
    # target "cat" (hypothetical) must not match merely because "cat" is a raw substring of
    # an unrelated word like "category" -- word-boundary padding must prevent this.
    bench = _bench()
    example = Example(example_id="1", target="cat")
    parsed = bench.parse_prediction("This shows a category of household items.", example)
    score = bench.score_example(parsed, example)
    assert score.correct is False


def test_empty_generation_is_parse_failure():
    bench = _bench()
    example = Example(example_id="1", target="tench, Tinca tinca")
    parsed = bench.parse_prediction("", example)
    assert parsed.parse_ok is False
    score = bench.score_example(parsed, example)
    assert score.score == 0.0


def test_aggregate_metrics_top1_accuracy():
    bench = _bench()
    e1 = Example(example_id="1", target="tench, Tinca tinca")
    e2 = Example(example_id="2", target="vulture")
    scores = [
        bench.score_example(bench.parse_prediction("tench", e1), e1),
        bench.score_example(bench.parse_prediction("eagle", e2), e2),
    ]
    metrics = bench.aggregate_metrics(scores)
    assert metrics["top1_accuracy"] == pytest.approx(0.5)
    assert metrics["primary_metric"] == metrics["top1_accuracy"]


class _FakeHFDataset:
    def __init__(self, features, rows):
        self.features = features
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _install_fake_datasets_module(monkeypatch, load_dataset_fn):
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = load_dataset_fn
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_load_examples_excludes_unlabeled_test_rows_and_uses_int2str(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    label_feature = SimpleNamespace(int2str=lambda idx: {0: "tench, Tinca tinca", -1: "unused"}[idx])
    hf_dataset = _FakeHFDataset(
        features={"label": label_feature},
        rows=[{"image": image, "label": 0}, {"image": image, "label": -1}],
    )
    _install_fake_datasets_module(monkeypatch, lambda source, split, revision: hf_dataset)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="ILSVRC/imagenet-1k", split="validation", revision=None))
    examples = bench.load_examples(cfg)

    assert len(examples) == 1  # the label=-1 row is excluded, not scored as a failure
    assert examples[0].target == "tench, Tinca tinca"


def test_load_examples_gated_access_failure_raises_clear_error_not_silent_substitution(monkeypatch):
    def _raising_load_dataset(source, split, revision):
        raise PermissionError("401 Client Error: access to this resource is gated")

    _install_fake_datasets_module(monkeypatch, _raising_load_dataset)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="ILSVRC/imagenet-1k", split="validation", revision=None))
    with pytest.raises(ImageNetGatedAccessError, match="GATED"):
        bench.load_examples(cfg)
