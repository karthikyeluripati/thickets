"""Tests for adapters/fine_grained_recognition_cub.py -- synthetic CUB-shaped rows, no real
dataset download / GPU / ray / vllm needed.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.fine_grained_recognition_cub import (
    CUBFineGrainedBenchmark,
    CUBSchemaError,
)
from neural_thickets_repro.benchmarks.base import Example


def _bench():
    return CUBFineGrainedBenchmark()


def test_capability_and_name():
    bench = _bench()
    assert bench.capability == "fine_grained_recognition"
    assert bench.name == "cub200_2011_test"


def test_exact_match_scores_correct():
    bench = _bench()
    example = Example(example_id="1", target="Black footed Albatross")
    parsed = bench.parse_prediction("Black footed Albatross", example)
    score = bench.score_example(parsed, example)
    assert score.correct is True
    assert score.score == 1.0


def test_verbose_generation_containing_full_species_phrase_still_matches():
    bench = _bench()
    example = Example(example_id="1", target="Black footed Albatross")
    parsed = bench.parse_prediction("I believe this is a Black-footed Albatross in the photo.", example)
    score = bench.score_example(parsed, example)
    assert score.correct is True


def test_wrong_species_does_not_match():
    bench = _bench()
    example = Example(example_id="1", target="Black footed Albatross")
    parsed = bench.parse_prediction("This looks like a Least Tern", example)
    score = bench.score_example(parsed, example)
    assert score.correct is False


def test_short_generic_word_does_not_spuriously_match_via_substring():
    # target "Tern" must not spuriously match because the word "external"/"internal" etc.
    # contains "tern" as a raw substring -- word-boundary padding must prevent this.
    bench = _bench()
    example = Example(example_id="1", target="Tern")
    parsed = bench.parse_prediction("This is an external view of a bird sanctuary", example)
    score = bench.score_example(parsed, example)
    assert score.correct is False


def test_empty_generation_is_parse_failure():
    bench = _bench()
    example = Example(example_id="1", target="Tern")
    parsed = bench.parse_prediction("", example)
    assert parsed.parse_ok is False
    score = bench.score_example(parsed, example)
    assert score.score == 0.0


def test_aggregate_metrics_top1_accuracy():
    bench = _bench()
    examples = [Example(example_id=str(i), target="Tern") for i in range(2)]
    scores = [
        bench.score_example(bench.parse_prediction("Tern", examples[0]), examples[0]),
        bench.score_example(bench.parse_prediction("Albatross", examples[1]), examples[1]),
    ]
    metrics = bench.aggregate_metrics(scores)
    assert metrics["top1_accuracy"] == pytest.approx(0.5)
    assert metrics["primary_metric"] == metrics["top1_accuracy"]


def test_canonical_species_name_strips_numeric_prefix_and_underscores():
    from neural_thickets_repro.benchmarks.adapters.fine_grained_recognition_cub import _canonical_species_name
    assert _canonical_species_name("001.Black_footed_Albatross") == "Black footed Albatross"
    assert _canonical_species_name("Least_Tern") == "Least Tern"


class _FakeHFDataset:
    """SimpleNamespace can't carry a working __iter__ (dunder methods used by implicit
    protocols like `for` are looked up on the type, not the instance) -- a tiny real class
    instead.
    """
    def __init__(self, features, rows):
        self.features = features
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _install_fake_datasets_module(monkeypatch, hf_dataset):
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, split, revision: hf_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_load_examples_uses_classlabel_names(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    label_feature = SimpleNamespace(names=["001.Black_footed_Albatross", "002.Least_Tern"])
    hf_dataset = _FakeHFDataset(
        features={"label": label_feature},
        rows=[{"image": image, "label": 0}, {"image": image, "label": 1}],
    )
    _install_fake_datasets_module(monkeypatch, hf_dataset)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="bentrevett/caltech-ucsd-birds-200-2011", split="test", revision=None))
    examples = bench.load_examples(cfg)

    assert examples[0].target == "Black footed Albatross"
    assert examples[1].target == "Least Tern"


def test_load_examples_hard_fails_when_no_class_names_exposed(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    hf_dataset = _FakeHFDataset(
        features={"label": SimpleNamespace()},  # no .names attribute
        rows=[{"image": image, "label": 0}],
    )
    _install_fake_datasets_module(monkeypatch, hf_dataset)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="some/mirror", split="test", revision=None))
    with pytest.raises(CUBSchemaError, match="refusing to guess"):
        bench.load_examples(cfg)
