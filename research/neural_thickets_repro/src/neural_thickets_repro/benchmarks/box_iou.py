"""Pure-Python bounding-box IoU utilities for the visual_grounding (RefCOCO/RefCOCO+)
capability. No pycocotools -- box-only IoU arithmetic doesn't need an annotation-format
library, just interval intersection.

Coordinate convention (documented explicitly, per the grounding-adapter's own requirement):
inputs/outputs of box_iou/accuracy_at_iou are [x1, y1, x2, y2] -- top-left/bottom-right
corners. RefCOCO's raw annotation is [x_min, y_min, width, height] (COCO convention);
xywh_to_xyxy converts that to the corner form. normalize_xyxy additionally divides by image
width/height to produce the [0, 1]-normalized representation the visual_grounding adapter
asks the model to output.

COORDINATE-CONTRACT BUG, ROUND 1 (fixed in an earlier repair pass): a real N=5 Qwen2.5-VL
smoke test found the model reliably outputting PIXEL-space boxes (e.g. [112, 189, 444, 362]
for a 640x425 image) despite the prompt explicitly asking for [0,1]-normalized coordinates --
the adapter was scoring these directly against a normalized target, giving near-zero IoU on
predictions that were actually ~0.9+ IoU once correctly interpreted as pixel coordinates.

COORDINATE-CONTRACT FIX, ROUND 2 (this repair pass -- EXPLICIT output contract): the grounding
prompt now explicitly asks the model to return PIXEL-space [x1,y1,x2,y2] and states the
image's own width/height in the prompt text (see visual_grounding_refcoco.py's
INSTRUCTION_TEMPLATE) -- a model-agnostic, reproducible contract, not something to rely on
auto-detection to work around. Auto-detection (detect_coordinate_mode below) is kept only as a
backward-compatible FALLBACK for a model that doesn't comply exactly, e.g. a real example this
pass found where a pixel-space prediction ([386,0,504,364] on a 500x375 image) overshot the
image width by only 4px and was WRONGLY classified as qwen_normalized_0_1000 (converting it to
a tiny, wrong box near the image's top-left corner, scoring IoU=0 on what was actually a good
prediction). Fixed via a documented, non-accuracy-tuned tolerance
(_pixel_bound_tolerance() below: the larger of a fixed absolute pixel margin and a small
fraction of the image's own larger dimension) on the pixel-fit check, and via
clip_box_to_image(): a canonicalized box is always clipped into the image's own bounds before
IoU, rather than rejected for a small boundary overshoot. Canonical internal representation
remains pixel-space xyxy.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

Box = Tuple[float, float, float, float]

COORD_MODE_NORMALIZED_0_1 = "normalized_xyxy_0_1"
COORD_MODE_PIXEL = "pixel_xyxy"
COORD_MODE_QWEN_0_1000 = "qwen_normalized_0_1000"
COORD_MODE_UNRECOGNIZED = "unrecognized"

# Tolerances are deliberately generous-but-bounded: enough to absorb a model's off-by-a-few
# rounding without ever letting two genuinely different conventions become ambiguous with
# each other at the boundary. NEVER tuned against any particular example's accuracy -- these
# are fixed, documented margins, chosen before looking at scores.
_NORMALIZED_RANGE_TOLERANCE = 0.02      # e.g. 1.01 still reads as "meant to be normalized"
_PIXEL_BOUND_ABS_TOLERANCE_PX = 10.0    # a box up to ~10px past the image edge is still "pixel space"
_PIXEL_BOUND_RELATIVE_TOLERANCE = 0.02  # ...or up to 2% of the image's own larger dimension, whichever is bigger
_QWEN_1000_RANGE_TOLERANCE = 1.0


def _pixel_bound_tolerance(image_width: float, image_height: float) -> float:
    """The larger of a fixed absolute pixel margin and a small fraction of the image's own
    larger dimension -- generous enough to absorb a genuine pixel-space prediction's small
    boundary overshoot (confirmed real case: +4px on a 500-wide image) without also being so
    generous that a genuinely too-large box gets misclassified as pixel space on a small image.
    """
    return max(_PIXEL_BOUND_ABS_TOLERANCE_PX, _PIXEL_BOUND_RELATIVE_TOLERANCE * max(image_width, image_height))


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


def detect_coordinate_mode(box: Box, image_width: float, image_height: float) -> str:
    """Deterministic, value-range + real-image-dimension classification of which coordinate
    convention `box` (a raw, as-parsed [x1,y1,x2,y2] prediction) is most likely expressed in --
    NEVER inferred from whether one interpretation happens to score better. Checked in this
    order (first match wins), matching the CAPABILITY_BENCHMARK_GATE.md investigation list:

      1. normalized_xyxy_0_1: all four values within [0,1] (+tolerance) -- checked first
         because a box this small could in principle also "fit" inside a large image's pixel
         bounds, and [0,1]-normalized is the more specific/informative interpretation.
      2. pixel_xyxy: the box fits within [0, image_width] x [0, image_height] (+tolerance) --
         this example's own real dimensions, never an assumed fixed size.
      3. qwen_normalized_0_1000: all four values within [0,1000] (+tolerance), for a box that
         did NOT fit the real image's pixel bounds above (so genuinely too large to be pixel
         coordinates for THIS image).
      4. unrecognized: none of the above -- the box cannot be safely canonicalized.
    """
    x1, y1, x2, y2 = box
    values = (x1, y1, x2, y2)

    if all(-_NORMALIZED_RANGE_TOLERANCE <= v <= 1 + _NORMALIZED_RANGE_TOLERANCE for v in values):
        return COORD_MODE_NORMALIZED_0_1

    pixel_tolerance = _pixel_bound_tolerance(image_width, image_height)
    if (
        x1 >= -pixel_tolerance and y1 >= -pixel_tolerance
        and x2 <= image_width + pixel_tolerance and y2 <= image_height + pixel_tolerance
    ):
        return COORD_MODE_PIXEL

    if all(-_QWEN_1000_RANGE_TOLERANCE <= v <= 1000 + _QWEN_1000_RANGE_TOLERANCE for v in values):
        return COORD_MODE_QWEN_0_1000

    return COORD_MODE_UNRECOGNIZED


def clip_box_to_image(box: Box, image_width: float, image_height: float) -> Box:
    """Clips a canonicalized pixel-space box into the image's own bounds -- a prediction
    that slightly overshoots the edge (the explicit output contract's own documented
    tolerance for this) is clipped and scored normally, never rejected outright for a small
    boundary overshoot.
    """
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), image_width), min(max(y1, 0.0), image_height),
        min(max(x2, 0.0), image_width), min(max(y2, 0.0), image_height),
    )


def canonicalize_prediction_box(box: Box, image_width: float, image_height: float) -> Tuple[Optional[Box], str]:
    """Converts `box` into the canonical pixel-xyxy representation for this example's real
    image size, using detect_coordinate_mode()'s deterministic classification, and clips the
    result into the image's own bounds (clip_box_to_image) -- the returned box is always
    ready for IoU as-is. Returns (None, "unrecognized") -- never a guessed conversion --
    when the box cannot be safely canonicalized; the caller must treat that as a scoring
    failure, not silently score it as pixel-space.
    """
    mode = detect_coordinate_mode(box, image_width, image_height)
    if mode == COORD_MODE_NORMALIZED_0_1:
        raw_canonical = denormalize_xyxy(box, image_width, image_height)
    elif mode == COORD_MODE_PIXEL:
        raw_canonical = tuple(box)
    elif mode == COORD_MODE_QWEN_0_1000:
        x1, y1, x2, y2 = box
        raw_canonical = (
            x1 / 1000.0 * image_width, y1 / 1000.0 * image_height,
            x2 / 1000.0 * image_width, y2 / 1000.0 * image_height,
        )
    else:
        return None, mode
    return clip_box_to_image(raw_canonical, image_width, image_height), mode
