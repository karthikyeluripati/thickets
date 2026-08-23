"""Tests for benchmarks/box_iou.py -- pure Python, no GPU/ray/vllm/pycocotools needed."""
import pytest

from neural_thickets_repro.benchmarks.box_iou import (
    accuracy_at_iou,
    box_iou,
    denormalize_xyxy,
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
