"""Tests for benchmarks/box_iou.py -- pure Python, no GPU/ray/vllm/pycocotools needed."""
import pytest

from neural_thickets_repro.benchmarks.box_iou import (
    COORD_MODE_NORMALIZED_0_1,
    COORD_MODE_PIXEL,
    COORD_MODE_QWEN_0_1000,
    COORD_MODE_UNRECOGNIZED,
    accuracy_at_iou,
    box_iou,
    canonicalize_prediction_box,
    clip_box_to_image,
    denormalize_xyxy,
    detect_coordinate_mode,
    mean_iou,
    normalize_xyxy,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy_conversion():
    assert xywh_to_xyxy((10, 20, 30, 40)) == (10, 20, 40, 60)


def test_normalize_xyxy_known_values():
    assert normalize_xyxy((10, 20, 40, 60), image_width=100, image_height=200) == (0.1, 0.1, 0.4, 0.3)


def test_normalize_denormalize_round_trip():
    box = (10, 20, 40, 60)
    normalized = normalize_xyxy(box, 100, 200)
    restored = denormalize_xyxy(normalized, 100, 200)
    assert restored == pytest.approx(box)


def test_normalize_xyxy_rejects_non_positive_image_size():
    with pytest.raises(ValueError):
        normalize_xyxy((0, 0, 1, 1), image_width=0, image_height=10)


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_iou_zero_overlap_is_zero():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_known_partial_overlap():
    # a: [0,0,10,10] area=100; b: [5,5,15,15] area=100; intersection: [5,5,10,10] area=25
    # union = 100+100-25=175; iou=25/175
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert box_iou(a, b) == pytest.approx(25 / 175)


def test_iou_degenerate_zero_area_box_returns_zero_no_divide_by_zero():
    zero_area = (5, 5, 5, 5)
    normal = (0, 0, 10, 10)
    assert box_iou(zero_area, normal) == 0.0
    assert box_iou(normal, zero_area) == 0.0
    assert box_iou(zero_area, zero_area) == 0.0


def test_iou_negative_area_box_returns_zero():
    # x2 < x1: a malformed/negative-area box, must not raise or go negative
    assert box_iou((10, 10, 0, 0), (0, 0, 10, 10)) == 0.0


def test_accuracy_at_iou_known_pairs():
    pairs = [
        ((0, 0, 10, 10), (0, 0, 10, 10)),   # iou=1.0 -> hit
        ((0, 0, 10, 10), (20, 20, 30, 30)),  # iou=0.0 -> miss
    ]
    assert accuracy_at_iou(pairs, threshold=0.5) == pytest.approx(0.5)


def test_accuracy_at_iou_empty_raises():
    with pytest.raises(ValueError):
        accuracy_at_iou([])


def test_mean_iou_known_pairs():
    pairs = [
        ((0, 0, 10, 10), (0, 0, 10, 10)),   # iou=1.0
        ((0, 0, 10, 10), (20, 20, 30, 30)),  # iou=0.0
    ]
    assert mean_iou(pairs) == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------
# detect_coordinate_mode / canonicalize_prediction_box (this repair pass -- a real N=5
# Qwen2.5-VL smoke test found the model reliably emitting PIXEL-space boxes despite the
# prompt asking for [0,1]-normalized coordinates, scoring near-zero IoU on predictions that
# were actually ~0.9+ IoU once correctly interpreted).
# ---------------------------------------------------------------------------------------

def test_detect_coordinate_mode_normalized_0_1():
    assert detect_coordinate_mode((0.1, 0.2, 0.5, 0.6), image_width=640, image_height=425) == COORD_MODE_NORMALIZED_0_1


def test_detect_coordinate_mode_pixel_when_it_fits_the_real_image_bounds():
    # values are >1 (not normalized) but comfortably fit a 640x425 image -- must not be
    # misclassified as anything else.
    assert detect_coordinate_mode((112, 189, 444, 362), image_width=640, image_height=425) == COORD_MODE_PIXEL


def test_detect_coordinate_mode_qwen_0_1000_when_it_does_not_fit_pixel_bounds():
    # exceeds the real 300x200 image's pixel bounds but fits within [0, 1000] -- classified
    # as the Qwen-VL-v1-style 0..1000 normalized convention, never silently as pixel space.
    assert detect_coordinate_mode((500, 100, 800, 150), image_width=300, image_height=200) == COORD_MODE_QWEN_0_1000


def test_detect_coordinate_mode_unrecognized_when_nothing_fits():
    assert detect_coordinate_mode((5000, 100, 8000, 150), image_width=300, image_height=200) == COORD_MODE_UNRECOGNIZED


def test_detect_coordinate_mode_never_uses_accuracy_only_range_and_dimensions():
    """Same numeric box, two different image sizes -- must classify purely from the
    deterministic range/dimension rule, never from "which interpretation would score better
    against some target" (which detect_coordinate_mode doesn't even receive).
    """
    box = (10, 20, 90, 80)
    assert detect_coordinate_mode(box, image_width=100, image_height=100) == COORD_MODE_PIXEL
    assert detect_coordinate_mode(box, image_width=1, image_height=1) == COORD_MODE_QWEN_0_1000


def test_canonicalize_prediction_box_normalized_denormalizes():
    canonical, mode = canonicalize_prediction_box((0.1, 0.2, 0.5, 0.6), image_width=100, image_height=200)
    assert mode == COORD_MODE_NORMALIZED_0_1
    assert canonical == pytest.approx((10, 40, 50, 120))


def test_canonicalize_prediction_box_pixel_passes_through():
    canonical, mode = canonicalize_prediction_box((112, 189, 444, 362), image_width=640, image_height=425)
    assert mode == COORD_MODE_PIXEL
    assert canonical == pytest.approx((112, 189, 444, 362))


def test_canonicalize_prediction_box_qwen_0_1000_scales_by_image_dims():
    canonical, mode = canonicalize_prediction_box((500, 100, 800, 150), image_width=300, image_height=200)
    assert mode == COORD_MODE_QWEN_0_1000
    # x*width/1000, y*height/1000 for each coordinate: (500*300/1000, 100*200/1000, 800*300/1000, 150*200/1000)
    assert canonical == pytest.approx((150, 20, 240, 30))


def test_canonicalize_prediction_box_unrecognized_returns_none():
    canonical, mode = canonicalize_prediction_box((5000, 100, 8000, 150), image_width=300, image_height=200)
    assert canonical is None
    assert mode == COORD_MODE_UNRECOGNIZED


# Real examples from the N=5 Qwen2.5-VL smoke test (see the task's own bug report) -- GT
# source xywh converted to pixel xyxy, Qwen's raw (pixel-space) prediction, expected IoU.

def test_real_example_2_pixel_prediction_scores_high_iou_once_canonicalized():
    image_width, image_height = 640, 425
    gt_xywh = (105.70, 196.11, 333.61, 169.56)
    gt_pixel_xyxy = xywh_to_xyxy(gt_xywh)
    qwen_prediction = (112, 189, 444, 362)

    canonical, mode = canonicalize_prediction_box(qwen_prediction, image_width, image_height)
    assert mode == COORD_MODE_PIXEL
    iou = box_iou(canonical, gt_pixel_xyxy)
    assert iou == pytest.approx(0.91, abs=0.02)


def test_real_example_3_pixel_prediction_scores_high_iou_once_canonicalized():
    # Image dimensions weren't stated for this example -- large enough to comfortably contain
    # both boxes, which is all detect_coordinate_mode's pixel-fit rule needs.
    image_width, image_height = 1000, 1000
    gt_xywh = (297.04, 96.10, 198.60, 237.72)
    gt_pixel_xyxy = xywh_to_xyxy(gt_xywh)
    qwen_prediction = (297, 98, 497, 338)

    canonical, mode = canonicalize_prediction_box(qwen_prediction, image_width, image_height)
    assert mode == COORD_MODE_PIXEL
    iou = box_iou(canonical, gt_pixel_xyxy)
    assert iou == pytest.approx(0.97, abs=0.02)


# ---------------------------------------------------------------------------------------
# clip_box_to_image / the explicit pixel-space output contract + tolerance (this repair pass)
# ---------------------------------------------------------------------------------------

def test_clip_box_to_image_clips_each_coordinate_independently():
    clipped = clip_box_to_image((-5, -3, 510, 400), image_width=500, image_height=375)
    assert clipped == (0, 0, 500, 375)


def test_clip_box_to_image_leaves_an_in_bounds_box_unchanged():
    box = (10, 20, 100, 200)
    assert clip_box_to_image(box, image_width=500, image_height=375) == box


def test_real_final_example_pixel_prediction_slightly_overshooting_the_edge():
    """The real remaining N=5 grounding failure this pass fixes: example_id=471277, a
    500x375 image. Qwen's raw pixel prediction [386,0,504,364] overshoots the image width by
    only 4px -- the OLD tolerance (2px) misclassified this as qwen_normalized_0_1000,
    converting it to a tiny box near the top-left corner and scoring IoU=0. Must now be
    recognized as pixel_xyxy, clipped to [386,0,500,364], and score a high IoU.
    """
    image_width, image_height = 500, 375
    gt_xywh = (384.2300, 0.0, 115.7700, 375.0)
    gt_pixel_xyxy = xywh_to_xyxy(gt_xywh)
    qwen_prediction = (386, 0, 504, 364)

    canonical, mode = canonicalize_prediction_box(qwen_prediction, image_width, image_height)

    assert mode == COORD_MODE_PIXEL  # must NEVER become qwen_normalized_0_1000 again
    assert canonical == pytest.approx((386, 0, 500, 364))  # clipped at x2
    iou = box_iou(canonical, gt_pixel_xyxy)
    assert iou > 0.9


def test_canonicalize_prediction_box_clips_a_normalized_box_slightly_past_one():
    canonical, mode = canonicalize_prediction_box((0.0, 0.0, 1.02, 0.5), image_width=100, image_height=100)
    assert mode == COORD_MODE_NORMALIZED_0_1
    assert canonical == pytest.approx((0.0, 0.0, 100.0, 50.0))  # 102 clipped down to 100


def test_pixel_tolerance_does_not_swallow_a_genuinely_too_large_box_on_a_small_image():
    """A guard against over-correcting: a box that is genuinely far outside a SMALL image's
    bounds (not just a few px past the edge) must still be classified as something other than
    pixel_xyxy -- the generous tolerance must not make every large image-space number look
    like "pixel space".
    """
    assert detect_coordinate_mode((500, 100, 800, 150), image_width=300, image_height=200) != COORD_MODE_PIXEL


def test_the_old_broken_behavior_would_have_scored_near_zero():
    """Documents WHY this fix matters: comparing the raw pixel-space prediction directly
    against the NORMALIZED target (the old behavior) gives near-zero IoU for the same
    genuinely-good prediction from test_real_example_2 above.
    """
    image_width, image_height = 640, 425
    gt_xywh = (105.70, 196.11, 333.61, 169.56)
    gt_normalized = normalize_xyxy(xywh_to_xyxy(gt_xywh), image_width, image_height)
    qwen_prediction = (112, 189, 444, 362)  # pixel-space, NOT normalized

    broken_iou = box_iou(qwen_prediction, gt_normalized)
    assert broken_iou < 0.01
