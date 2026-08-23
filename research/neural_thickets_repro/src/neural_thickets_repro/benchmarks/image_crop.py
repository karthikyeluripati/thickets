"""Deterministic bounding-box crop utility for the Visual Genome LOCALIZED
attribute-recognition protocol (see adapters/attribute_recognition_visualgenome.py) --
crops an image down to its annotated ground-truth object bbox, with a small FIXED
context-padding percentage, so the model is asked about a specific, visually unambiguous
object instead of the whole scene plus a "which one is this" guess. The bbox is used ONLY
for localization here -- it is never treated as, or allowed to leak, target-label
information (the padding/crop logic below never looks at positive_attributes at all).

Why crop instead of continuing to draw a marker on the full image (the previous protocol):
a real N=50 image-sanity run found the full-image protocol had NO visual dependence
(correct=0.15, shuffled=0.10, text-only=0.15 -- i.e. text-only matched correct-image
exactly) -- the model was answering from the object NAME alone (a strong prior for a
plausible attribute of e.g. "chair") without needing the image at all. Cropping to the
object's own region removes the rest of the scene (and any other objects' attributes) from
consideration, and reusing the identical crop for every condition (correct/repeat/shuffled)
is what makes a later correct-vs-shuffled/text-only comparison a fair test of VISUAL
dependence rather than an artifact of a different transformation per condition.

FIXED PADDING RULE (not tuned against model accuracy): CROP_CONTEXT_PADDING_FRACTION = 0.10
-- 10% of the box's OWN width/height is added on each side, then the result is clipped to the
image's real bounds. Chosen because: (a) VG's annotation boxes are tight around the object,
so a small margin reduces the risk of cropping out a genuinely informative edge cue (a
chair's visible leg, a shirt's collar) that sits just outside the raw box; (b) it is
proportional to the object's own size, not a fixed pixel count, so it behaves consistently
across VG's very wide range of object/image sizes; (c) it is small enough that it does not
reintroduce meaningful surrounding-scene context (unlike, say, padding to a fixed larger
size, which would partially resurrect the original whole-image-context problem). This value
is fixed BEFORE looking at any model output and must not be adjusted to chase a particular
accuracy or image-sanity gap -- see CAPABILITY_BENCHMARK_GATE.md.
"""
from __future__ import annotations

from typing import Sequence, Tuple

CROP_CONTEXT_PADDING_FRACTION = 0.10


class CropError(RuntimeError):
    """The requested bbox cannot be turned into a valid, non-degenerate crop (e.g. a
    non-positive source width/height, or a zero-area result after padding+clipping to the
    image bounds) -- refuses to silently return a garbage/empty crop.
    """


def compute_padded_crop_box(
    bbox_xywh: Sequence[float], image_width: float, image_height: float,
    padding_fraction: float = CROP_CONTEXT_PADDING_FRACTION,
) -> Tuple[int, int, int, int]:
    """Returns (x1, y1, x2, y2) integer PIXEL crop bounds: `bbox_xywh` expanded by
    `padding_fraction` of its OWN width/height on each side, then clipped to
    [0, image_width] x [0, image_height]. Raises CropError for a non-positive source box or
    a degenerate (zero/negative-area) result -- never silently clamped to something usable.
    """
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        raise CropError(f"source bbox has non-positive width/height: {tuple(bbox_xywh)}")

    pad_x = w * padding_fraction
    pad_y = h * padding_fraction
    x1, y1 = x - pad_x, y - pad_y
    x2, y2 = x + w + pad_x, y + h + pad_y

    x1_clipped = max(0, int(round(x1)))
    y1_clipped = max(0, int(round(y1)))
    x2_clipped = min(int(image_width), int(round(x2)))
    y2_clipped = min(int(image_height), int(round(y2)))

    if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
        raise CropError(
            f"padded+clipped crop box is degenerate (zero or negative area): "
            f"bbox={tuple(bbox_xywh)}, image=({image_width}x{image_height}), "
            f"padded=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
            f"clipped=({x1_clipped},{y1_clipped},{x2_clipped},{y2_clipped})"
        )
    return x1_clipped, y1_clipped, x2_clipped, y2_clipped


def crop_to_bbox(image, bbox_xywh: Sequence[float], padding_fraction: float = CROP_CONTEXT_PADDING_FRACTION):
    """Returns (cropped_image, crop_box_xyxy). `image` is never mutated -- PIL's own
    `.crop()` always returns a new Image. Image width/height are read from `image.size`
    itself (never a possibly-stale recorded value), so this always crops whatever image is
    ACTUALLY attached -- correct for the shuffled-image sanity condition, where a different
    example's image (with a different real size) is paired with a bbox for localization.
    """
    image_width, image_height = image.size
    crop_box = compute_padded_crop_box(bbox_xywh, image_width, image_height, padding_fraction)
    return image.crop(crop_box), crop_box
