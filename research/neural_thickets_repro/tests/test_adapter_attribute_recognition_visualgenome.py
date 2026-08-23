"""Tests for adapters/attribute_recognition_visualgenome.py -- synthetic VG-shaped rows, no
real dataset download / GPU / ray / vllm needed.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.attribute_recognition_visualgenome import (
    VisualGenomeAttributeBenchmark,
    VisualGenomeSchemaError,
)
from neural_thickets_repro.benchmarks.base import Example


def _bench():
    return VisualGenomeAttributeBenchmark()


def test_capability_and_name():
    bench = _bench()
    assert bench.capability == "attribute_recognition"
    assert bench.name == "visual_genome_attributes"


def test_config_name_is_documented_and_overridable():
    default_bench = _bench()
    assert "attributes" in default_bench.dataset_source()
    custom_bench = VisualGenomeAttributeBenchmark(config_name="attributes_v1.0.0")
    assert "attributes_v1.0.0" in custom_bench.dataset_source()


def test_known_caveats_documents_marker_protocol_and_multi_attribute_targets():
    caveats = " ".join(_bench().known_caveats())
    assert "not naturally occurring VG data" in caveats
    assert "matches ANY of them" in caveats


def test_prepare_image_draws_marker_and_preserves_original(tiny_image_factory):
    bench = _bench()
    image = tiny_image_factory(size=(20, 20), color=(0, 0, 0))
    example = Example(example_id="1", image=image, metadata={"bbox_xywh": [2, 2, 10, 10]})

    marked = bench.prepare_image(example)

    assert marked is not image  # a copy, not the same object
    assert example.image is image  # original untouched
    # Pixel at the drawn outline should now be red (marker color), original was pure black.
    assert marked.getpixel((2, 2)) == (255, 0, 0)
    assert image.getpixel((2, 2)) == (0, 0, 0)
    # Interior of the box (not on the outline) must remain unobscured (still black).
    assert marked.getpixel((6, 6)) == (0, 0, 0)


def test_prepare_image_none_image_returns_none():
    bench = _bench()
    example = Example(example_id="1", image=None, metadata={})
    assert bench.prepare_image(example) is None


def test_score_example_matches_any_of_multiple_valid_attributes():
    bench = _bench()
    example = Example(example_id="1", target=["red", "wooden", "old"])
    parsed = bench.parse_prediction("It looks wooden to me.", example)
    score = bench.score_example(parsed, example)
    assert score.correct is True
    assert score.detail["matched_attribute"] == "wooden"


def test_score_example_no_match_among_valid_attributes():
    bench = _bench()
    example = Example(example_id="1", target=["red", "wooden"])
    parsed = bench.parse_prediction("It's made of metal.", example)
    score = bench.score_example(parsed, example)
    assert score.correct is False


def test_score_example_empty_generation_is_parse_failure():
    bench = _bench()
    example = Example(example_id="1", target=["red"])
    parsed = bench.parse_prediction("", example)
    assert parsed.parse_ok is False
    score = bench.score_example(parsed, example)
    assert score.score == 0.0


def test_aggregate_metrics_accuracy():
    bench = _bench()
    e1 = Example(example_id="1", target=["red"])
    e2 = Example(example_id="2", target=["blue"])
    scores = [
        bench.score_example(bench.parse_prediction("red", e1), e1),
        bench.score_example(bench.parse_prediction("green", e2), e2),
    ]
    metrics = bench.aggregate_metrics(scores)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["primary_metric"] == metrics["accuracy"]


class _FakeHFDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _install_fake_datasets_module(monkeypatch, rows):
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, config_name, split, revision: _FakeHFDataset(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_load_examples_flattens_objects_and_preserves_multi_attribute_targets(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    rows = [{
        "image": image, "image_id": 1,
        "attributes": [
            {"object_id": 10, "names": ["chair"], "attributes": ["red", "wooden"], "x": 1, "y": 2, "w": 3, "h": 4},
            {"object_id": 11, "names": ["table"], "attributes": [], "x": 5, "y": 6, "w": 7, "h": 8},  # no attributes -- skipped
        ],
    }]
    _install_fake_datasets_module(monkeypatch, rows)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="ranjaykrishna/visual_genome", split="train", revision=None))
    examples = bench.load_examples(cfg)

    assert len(examples) == 1  # the no-attribute object is skipped, not scored as a failure
    assert examples[0].example_id == "1_10"
    assert examples[0].target == ["red", "wooden"]
    assert examples[0].metadata["bbox_xywh"] == [1, 2, 3, 4]


def test_load_examples_hard_fails_on_missing_expected_fields(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    rows = [{"image": image, "image_id": 1, "attributes": [{"object_id": 10, "attributes": ["red"]}]}]  # missing x/y/w/h
    _install_fake_datasets_module(monkeypatch, rows)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="ranjaykrishna/visual_genome", split="train", revision=None))
    with pytest.raises(VisualGenomeSchemaError):
        bench.load_examples(cfg)
