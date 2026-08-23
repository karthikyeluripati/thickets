"""Tests for adapters/visual_grounding_refcoco.py -- synthetic RefCOCO-shaped rows, no real
dataset download / GPU / ray / vllm needed. Coordinate conventions (xywh -> xyxy -> normalized)
get their own dedicated tests here in addition to box_iou.py's own unit tests, since this is
the one place those conversions are chained together against a real image size.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.visual_grounding_refcoco import (
    RefCOCOGroundingBenchmark,
    RefCOCOSchemaError,
)
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


# A degenerate 1x1 "image" makes normalized-[0,1] coordinates numerically identical to pixel
# coordinates (denormalize_xyxy(box, 1, 1) == box) -- lets these unit-level scoring tests stay
# expressed directly in the [0,1] range without needing a realistic image size, while still
# exercising the real score_example() coordinate-canonicalization code path (not bypassing it).
_UNIT_IMAGE_METADATA = {"image_width": 1, "image_height": 1}


def test_score_example_identical_boxes_iou_1():
    bench = _bench()
    box = (0.1, 0.2, 0.5, 0.6)
    example = Example(example_id="1", target=box, metadata=_UNIT_IMAGE_METADATA)
    parsed = bench.parse_prediction(f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]", example)
    score = bench.score_example(parsed, example)
    assert score.score == pytest.approx(1.0)
    assert score.correct is True
    assert score.detail["coordinate_mode"] == "normalized_xyxy_0_1"


def test_score_example_non_overlapping_boxes_iou_0():
    bench = _bench()
    example = Example(example_id="1", target=(0.0, 0.0, 0.1, 0.1), metadata=_UNIT_IMAGE_METADATA)
    parsed = bench.parse_prediction("[0.5, 0.5, 0.9, 0.9]", example)
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.correct is False


def test_score_example_parse_failure_scores_zero_iou():
    bench = _bench()
    example = Example(example_id="1", target=(0.1, 0.2, 0.5, 0.6), metadata=_UNIT_IMAGE_METADATA)
    parsed = bench.parse_prediction("no box here", example)
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.detail["reason"] == "parse_failure"


def test_score_example_unrecognized_coordinate_convention_scores_zero_and_is_flagged():
    bench = _bench()
    # Comfortably larger than any plausible normalized/pixel/qwen-1000 interpretation for a
    # tiny 1x1-metadata "image" -- must not be silently scored as any of them.
    example = Example(example_id="1", target=(0.1, 0.2, 0.5, 0.6), metadata=_UNIT_IMAGE_METADATA)
    parsed = bench.parse_prediction("[5000, 6000, 7000, 8000]", example)
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.detail["reason"] == "unrecognized_coordinate_convention"
    assert score.detail["coordinate_mode"] == "unrecognized"


def test_aggregate_metrics_accuracy_and_mean_iou():
    bench = _bench()
    target = (0.0, 0.0, 0.5, 0.5)
    example = Example(example_id="1", target=target, metadata=_UNIT_IMAGE_METADATA)
    perfect = bench.score_example(bench.parse_prediction("[0.0, 0.0, 0.5, 0.5]", example), example)
    miss = bench.score_example(bench.parse_prediction("[0.9, 0.9, 1.0, 1.0]", example), example)

    metrics = bench.aggregate_metrics([perfect, miss])
    assert metrics["accuracy_at_iou_0.5"] == pytest.approx(0.5)
    assert metrics["mean_iou"] == pytest.approx((1.0 + 0.0) / 2)
    assert metrics["primary_metric"] == metrics["accuracy_at_iou_0.5"]


def test_score_example_pixel_space_prediction_matches_normalized_target_real_example_2():
    """The exact real N=5 smoke-test case: Qwen emits pixel-space coordinates despite the
    prompt asking for [0,1]-normalized ones; score_example() must still recognize the ~0.91
    IoU match instead of scoring near-zero.
    """
    from neural_thickets_repro.benchmarks.box_iou import normalize_xyxy, xywh_to_xyxy

    image_width, image_height = 640, 425
    gt_xywh = (105.70, 196.11, 333.61, 169.56)
    target_normalized = normalize_xyxy(xywh_to_xyxy(gt_xywh), image_width, image_height)
    example = Example(example_id="1", target=target_normalized, metadata={"image_width": image_width, "image_height": image_height})

    bench = _bench()
    parsed = bench.parse_prediction("[112, 189, 444, 362]", example)
    score = bench.score_example(parsed, example)

    assert score.detail["coordinate_mode"] == "pixel_xyxy"
    assert score.score == pytest.approx(0.91, abs=0.02)
    assert score.correct is True


def test_score_example_real_example_471277_pixel_overshoot_is_clipped_not_misclassified():
    """The final real remaining N=5 grounding failure (this repair pass): example_id=471277,
    a 500x375 image. Qwen's pixel prediction [386,0,504,364] overshoots the image width by
    only 4px -- must be recognized as pixel_xyxy (never qwen_normalized_0_1000), clipped to
    [386,0,500,364], and score a high IoU, not the previously-observed IoU=0.
    """
    from neural_thickets_repro.benchmarks.box_iou import normalize_xyxy, xywh_to_xyxy

    image_width, image_height = 500, 375
    gt_xywh = (384.2300, 0.0, 115.7700, 375.0)
    target_normalized = normalize_xyxy(xywh_to_xyxy(gt_xywh), image_width, image_height)
    example = Example(example_id="471277", target=target_normalized, metadata={"image_width": image_width, "image_height": image_height})

    bench = _bench()
    parsed = bench.parse_prediction("[386,0,504,364]", example)
    score = bench.score_example(parsed, example)

    assert score.detail["coordinate_mode"] == "pixel_xyxy"
    assert score.detail["canonical_prediction_box"] == pytest.approx([386, 0, 500, 364])
    assert score.detail["raw_prediction_box"] == pytest.approx([386, 0, 504, 364])  # raw kept unclipped
    assert score.score > 0.9
    assert score.correct is True


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
    # -> normalized by (100,200) -> (0.1, 0.1, 0.4, 0.3). "question" is the fixed
    # region-captioning instruction (real schema, confirmed live) -- NOT the referring
    # expression; "answer" (a list) is where the real referring expression(s) live.
    image = _FakeImage(size=(100, 200))
    fake_rows = [{
        "image": image,
        "question": "Please carefully observe the area circled in the image and come up with a caption for the area.",
        "answer": ["the red car", "a red sedan"],
        "bbox": [10, 20, 30, 40], "question_id": "q1", "file_name": "img1.jpg",
    }]

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, split, revision: _FakeHFDataset(fake_rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="lmms-lab-encoder/RefCOCO", split="val", revision=None))
    examples = bench.load_examples(cfg)

    assert len(examples) == 1
    assert examples[0].target == pytest.approx((0.1, 0.1, 0.4, 0.3))
    assert examples[0].prompt_input["referring_expression"] == "the red car"  # from "answer", NOT the "question" instruction field
    assert examples[0].metadata["bbox_pixels_xywh"] == [10, 20, 30, 40]
    assert examples[0].metadata["all_referring_expressions"] == ["the red car", "a red sedan"]


def test_load_examples_hard_fails_on_empty_answer_list(monkeypatch):
    image = _FakeImage(size=(100, 200))
    fake_rows = [{
        "image": image, "question": "Please carefully observe the area circled in the image and come up with a caption for the area.",
        "answer": [], "bbox": [10, 20, 30, 40], "question_id": "q1", "file_name": "img1.jpg",
    }]
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, split, revision: _FakeHFDataset(fake_rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="lmms-lab-encoder/RefCOCO", split="val", revision=None))
    with pytest.raises(RefCOCOSchemaError, match="empty 'answer' list"):
        bench.load_examples(cfg)


def test_known_caveats_documents_the_question_field_correction():
    caveats = " ".join(_bench().known_caveats())
    assert "region-captioning INSTRUCTION" in caveats
    assert "'answer' field" in caveats


def test_build_prompt_documents_the_explicit_pixel_space_contract():
    """This repair pass: the prompt now explicitly asks for PIXEL coordinates and states the
    image's own real dimensions -- a model-agnostic, reproducible contract, not [0,1]-normalized.
    """
    bench = _bench()
    example = Example(
        example_id="1",
        prompt_input={"referring_expression": "the red car"},
        metadata={"image_width": 640, "image_height": 425},
    )
    messages = bench.build_prompt(example)
    text = messages[0]["content"][1]["text"]
    assert "the red car" in text
    assert "x1" in text and "y1" in text and "x2" in text and "y2" in text
    assert "PIXEL" in text
    assert "640" in text and "425" in text
    assert "0 and 1" not in text
