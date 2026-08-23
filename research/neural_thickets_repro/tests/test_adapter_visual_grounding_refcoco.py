"""Tests for adapters/visual_grounding_refcoco.py -- synthetic RefCOCO-shaped rows, no real
dataset download / GPU / ray / vllm needed. Coordinate conventions (xywh -> xyxy -> normalized)
get their own dedicated tests here in addition to box_iou.py's own unit tests, since this is
the one place those conversions are chained together against a real image size.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.visual_grounding_refcoco import RefCOCOGroundingBenchmark
from neural_thickets_repro.benchmarks.base import Example


def _bench():
    return RefCOCOGroundingBenchmark()


def test_capability_and_default_name():
    bench = _bench()
    assert bench.capability == "visual_grounding"
    assert bench.name == "refcoco_val"
    assert bench.dataset_source() == "lmms-lab-encoder/RefCOCO"


def test_refcoco_plus_variant_constructor():
    bench = RefCOCOGroundingBenchmark(dataset_repo_id="lmms-lab-encoder/RefCOCO+", variant_name="refcoco_plus_val")
    assert bench.dataset_source() == "lmms-lab-encoder/RefCOCO+"
    assert bench.name == "refcoco_plus_val"
    assert "not independently verified" in " ".join(bench.known_caveats())


def test_text_only_condition_not_supported():
    bench = _bench()
    assert bench.supports_text_only_condition() is False
    assert bench.text_only_unsupported_reason() is not None


@pytest.mark.parametrize("generation,expected_box", [
    ("[0.1, 0.2, 0.5, 0.6]", (0.1, 0.2, 0.5, 0.6)),
    ("The object is located at [0.1, 0.2, 0.5, 0.6] in the image.", (0.1, 0.2, 0.5, 0.6)),
    ("x1=0.1 y1=0.2 x2=0.5 y2=0.6", (0.1, 0.2, 0.5, 0.6)),
])
def test_parser_extracts_four_coordinates_from_realistic_generations(generation, expected_box):
    bench = _bench()
    example = Example(example_id="1", target=(0.1, 0.2, 0.5, 0.6))
    parsed = bench.parse_prediction(generation, example)
    assert parsed.parse_ok is True
    assert parsed.parsed == pytest.approx(expected_box)


@pytest.mark.parametrize("generation", ["I cannot determine the location.", "[0.1, 0.2]", ""])
def test_parser_flags_malformed_generation_as_failure(generation):
    bench = _bench()
    example = Example(example_id="1", target=(0.1, 0.2, 0.5, 0.6))
    parsed = bench.parse_prediction(generation, example)
    assert parsed.parse_ok is False


def test_score_example_identical_boxes_iou_1():
    bench = _bench()
    box = (0.1, 0.2, 0.5, 0.6)
    example = Example(example_id="1", target=box)
    parsed = bench.parse_prediction(f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]", example)
    score = bench.score_example(parsed, example)
    assert score.score == pytest.approx(1.0)
    assert score.correct is True


def test_score_example_non_overlapping_boxes_iou_0():
    bench = _bench()
    example = Example(example_id="1", target=(0.0, 0.0, 0.1, 0.1))
    parsed = bench.parse_prediction("[0.5, 0.5, 0.9, 0.9]", example)
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.correct is False


def test_score_example_parse_failure_scores_zero_iou():
    bench = _bench()
    example = Example(example_id="1", target=(0.1, 0.2, 0.5, 0.6))
    parsed = bench.parse_prediction("no box here", example)
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.detail["reason"] == "parse_failure"


def test_aggregate_metrics_accuracy_and_mean_iou():
    bench = _bench()
    target = (0.0, 0.0, 0.5, 0.5)
    example = Example(example_id="1", target=target)
    perfect = bench.score_example(bench.parse_prediction("[0.0, 0.0, 0.5, 0.5]", example), example)
    miss = bench.score_example(bench.parse_prediction("[0.9, 0.9, 1.0, 1.0]", example), example)

    metrics = bench.aggregate_metrics([perfect, miss])
    assert metrics["accuracy_at_iou_0.5"] == pytest.approx(0.5)
    assert metrics["mean_iou"] == pytest.approx((1.0 + 0.0) / 2)
    assert metrics["primary_metric"] == metrics["accuracy_at_iou_0.5"]


class _FakeImage:
    def __init__(self, size):
        self.size = size


class _FakeHFDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def test_load_examples_converts_xywh_pixels_to_normalized_xyxy(monkeypatch):
    # image 100x200; bbox [x=10, y=20, w=30, h=40] (COCO xywh) -> xyxy pixels (10,20,40,60)
    # -> normalized by (100,200) -> (0.1, 0.1, 0.4, 0.3)
    image = _FakeImage(size=(100, 200))
    fake_rows = [{"image": image, "question": "the red car", "bbox": [10, 20, 30, 40], "question_id": "q1", "file_name": "img1.jpg"}]

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, split, revision: _FakeHFDataset(fake_rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="lmms-lab-encoder/RefCOCO", split="val", revision=None))
    examples = bench.load_examples(cfg)

    assert len(examples) == 1
    assert examples[0].target == pytest.approx((0.1, 0.1, 0.4, 0.3))
    assert examples[0].prompt_input["referring_expression"] == "the red car"
    assert examples[0].metadata["bbox_pixels_xywh"] == [10, 20, 30, 40]


def test_build_prompt_documents_coordinate_convention():
    bench = _bench()
    example = Example(example_id="1", prompt_input={"referring_expression": "the red car"})
    messages = bench.build_prompt(example)
    text = messages[0]["content"][1]["text"]
    assert "the red car" in text
    assert "x1" in text and "y1" in text and "x2" in text and "y2" in text
    assert "0 and 1" in text
