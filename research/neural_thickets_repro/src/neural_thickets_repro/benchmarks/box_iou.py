"""Pure-Python bounding-box IoU utilities for the visual_grounding (RefCOCO/RefCOCO+)
capability. No pycocotools -- box-only IoU arithmetic doesn't need an annotation-format
library, just interval intersection.

Coordinate convention (documented explicitly, per the grounding-adapter's own requirement):
inputs/outputs of box_iou/accuracy_at_iou are [x1, y1, x2, y2] -- top-left/bottom-right
corners. RefCOCO's raw annotation is [x_min, y_min, width, height] (COCO convention);
xywh_to_xyxy converts that to the corner form. normalize_xyxy additionally divides by image
width/height to produce the [0, 1]-normalized representation the visual_grounding adapter
asks the model to output.
"""
from __future__ import annotations

from typing import Iterable, Tuple

Box = Tuple[float, float, float, float]


def xywh_to_xyxy(box: Box) -> Box:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def normalize_xyxy(box: Box, image_width: float, image_height: float) -> Box:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image_width/image_height must be positive, got ({image_width}, {image_height})")
    x1, y1, x2, y2 = box
    return (x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height)


def denormalize_xyxy(box: Box, image_width: float, image_height: float) -> Box:
    x1, y1, x2, y2 = box
    return (x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height)


def box_iou(box_a: Box, box_b: Box) -> float:
    """box_a/box_b in [x1, y1, x2, y2] (either both pixel or both normalized -- IoU is
    scale-invariant as long as both boxes use the same convention). Returns 0.0 for a
    degenerate (zero- or negative-area) box rather than raising or dividing by zero.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def accuracy_at_iou(pairs: Iterable[Tuple[Box, Box]], threshold: float = 0.5) -> float:
    """pairs: iterable of (predicted_box, target_box), same coordinate convention as
    box_iou. Returns the fraction of pairs scoring IoU >= threshold.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("accuracy_at_iou requires at least one pair")
    hits = sum(1 for pred, target in pairs if box_iou(pred, target) >= threshold)
    return hits / len(pairs)


def mean_iou(pairs: Iterable[Tuple[Box, Box]]) -> float:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("mean_iou requires at least one pair")
    return sum(box_iou(pred, target) for pred, target in pairs) / len(pairs)
