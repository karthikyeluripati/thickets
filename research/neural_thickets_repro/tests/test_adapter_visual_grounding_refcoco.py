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
    GROUNDING_REPEAT_BOX_IOU_THRESHOLD,
    RefCOCOGroundingBenchmark,
    RefCOCOSchemaError,
)
from neural_thickets_repro.benchmarks.base import Example, ExampleScore, ParsedPrediction
from neural_thickets_repro.benchmarks.runner import PerExampleResult, RunResult


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


# ---------------------------------------------------------------------------------------
# repeatability_verdict() -- grounding-specific measurement-stability criterion (this
# repair pass). Real N=5 finding: parsed_prediction_hash_match=False (3/5 boxes jittered by
# a few pixels between two identical greedy runs) while primary_metric stayed exactly equal
# (1.0 both times) -- exact-token equality is the wrong criterion for a continuous box.
# ---------------------------------------------------------------------------------------

def _grounding_result(boxes, primary_metric, mean_iou, raw_generations=None):
    """boxes: [(example_id, canonical_box_or_None), ...] -- builds a minimal RunResult
    shaped like what run_benchmark() actually produces for this adapter (ExampleScore.detail
    carrying "canonical_prediction_box", aggregate_metrics carrying "mean_iou").
    """
    raw_generations = raw_generations or {}
    per_example = []
    for example_id, box in boxes:
        detail = {"canonical_prediction_box": list(box) if box is not None else None, "iou": 1.0, "coordinate_mode": "pixel_xyxy"}
        parsed = ParsedPrediction(parsed=tuple(box) if box is not None else None, parse_ok=box is not None)
        score = ExampleScore(score=1.0, correct=True, detail=detail)
        raw = raw_generations.get(example_id, f"[{','.join(str(v) for v in box)}]" if box is not None else "")
        per_example.append(PerExampleResult(example_id, "img", raw, parsed, score))
    return RunResult(per_example=per_example, aggregate_metrics={"primary_metric": primary_metric, "mean_iou": mean_iou})


def test_repeatability_verdict_identical_boxes_is_repeatable():
    bench = _bench()
    base = _grounding_result([("1", (100, 100, 200, 200))], primary_metric=1.0, mean_iou=0.91)
    repeat = _grounding_result([("1", (100, 100, 200, 200))], primary_metric=1.0, mean_iou=0.91)

    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is True
    assert diagnostics["mean_repeat_box_iou"] == pytest.approx(1.0)
    assert diagnostics["min_repeat_box_iou"] == pytest.approx(1.0)
    assert diagnostics["repeat_box_equivalence_rate"] == pytest.approx(1.0)
    assert diagnostics["repeat_box_iou_threshold"] == GROUNDING_REPEAT_BOX_IOU_THRESHOLD


def test_repeatability_verdict_one_pixel_jitter_is_repeatable():
    # Real-shaped example: [110,187,444,362] vs [111,186,444,362] -- box IoU ~0.991.
    bench = _bench()
    base = _grounding_result([("1", (110, 187, 444, 362))], primary_metric=1.0, mean_iou=0.91)
    repeat = _grounding_result([("1", (111, 186, 444, 362))], primary_metric=1.0, mean_iou=0.912)

    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is True
    assert diagnostics["mean_repeat_box_iou"] > 0.95


def test_repeatability_verdict_slightly_different_high_overlap_boxes_is_repeatable():
    # IoU = 10000 / 10400 ~= 0.9615 -- just above the 0.95 threshold.
    bench = _bench()
    base = _grounding_result([("1", (0, 0, 100, 100))], primary_metric=1.0, mean_iou=0.9)
    repeat = _grounding_result([("1", (0, 0, 100, 104))], primary_metric=1.0, mean_iou=0.9)

    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is True
    assert diagnostics["mean_repeat_box_iou"] == pytest.approx(10000 / 10400, abs=1e-4)


def test_repeatability_verdict_genuinely_different_boxes_is_not_repeatable():
    bench = _bench()
    base = _grounding_result([("1", (0, 0, 100, 100))], primary_metric=1.0, mean_iou=0.9)
    repeat = _grounding_result([("1", (200, 200, 300, 300))], primary_metric=1.0, mean_iou=0.9)

    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is False
    assert diagnostics["mean_repeat_box_iou"] == pytest.approx(0.0)
    assert diagnostics["repeat_box_equivalence_rate"] == pytest.approx(0.0)


def test_repeatability_verdict_same_accuracy_but_unstable_boxes_is_not_repeatable():
    """The exact scenario this fix must NOT rubber-stamp: identical primary_metric across
    runs does not by itself prove the underlying boxes are stable.
    """
    bench = _bench()
    base = _grounding_result([("1", (0, 0, 100, 100)), ("2", (0, 0, 50, 50))], primary_metric=1.0, mean_iou=0.9)
    repeat = _grounding_result([("1", (200, 200, 300, 300)), ("2", (0, 0, 50, 50))], primary_metric=1.0, mean_iou=0.9)

    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is False  # primary_metric matches (1.0 == 1.0) but example "1"'s box collapsed
    assert diagnostics["min_repeat_box_iou"] == pytest.approx(0.0)
    assert diagnostics["repeat_box_equivalence_rate"] == pytest.approx(0.5)


def test_repeatability_verdict_stable_boxes_with_raw_string_formatting_differences_is_repeatable():
    """Raw generation text differs (formatting only) but the PARSED/canonical box is
    identical -- the verdict must be driven by the canonical box, never by raw string diffs.
    """
    bench = _bench()
    base = _grounding_result([("1", (10, 20, 30, 40))], primary_metric=1.0, mean_iou=0.95, raw_generations={"1": "[10, 20, 30, 40]"})
    repeat = _grounding_result([("1", (10, 20, 30, 40))], primary_metric=1.0, mean_iou=0.95, raw_generations={"1": "The box is [10,20,30,40]."})

    assert base.per_example[0].raw_generation != repeat.per_example[0].raw_generation  # genuinely different raw text
    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is True
    assert diagnostics["mean_repeat_box_iou"] == pytest.approx(1.0)


def test_repeatability_verdict_no_common_ids_is_not_repeatable():
    bench = _bench()
    base = _grounding_result([("1", (0, 0, 100, 100))], primary_metric=1.0, mean_iou=0.9)
    repeat = _grounding_result([("2", (0, 0, 100, 100))], primary_metric=1.0, mean_iou=0.9)

    repeatable, _ = bench.repeatability_verdict(base, repeat)
    assert repeatable is False


def test_repeatability_verdict_missing_canonical_box_counts_as_zero_agreement():
    bench = _bench()
    base = _grounding_result([("1", (0, 0, 100, 100))], primary_metric=1.0, mean_iou=0.9)
    repeat = _grounding_result([("1", None)], primary_metric=1.0, mean_iou=0.9)

    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is False
    assert diagnostics["mean_repeat_box_iou"] == pytest.approx(0.0)
