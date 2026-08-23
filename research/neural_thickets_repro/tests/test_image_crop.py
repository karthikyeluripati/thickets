"""Tests for benchmarks/image_crop.py -- pure Python + real PIL images, no GPU/ray/vllm
needed. Covers the deterministic bbox->crop pipeline for the Visual Genome LOCALIZED
attribute-recognition protocol: xywh->crop bounds, image-bound clipping, fixed padding
behavior, zero-area rejection, and that the original image is never mutated.
"""
import pytest

from neural_thickets_repro.benchmarks.image_crop import (
    CROP_CONTEXT_PADDING_FRACTION,
    CropError,
    compute_padded_crop_box,
    crop_to_bbox,
)


# ---------------------------------------------------------------------------------------
# compute_padded_crop_box -- pure math, no PIL needed
# ---------------------------------------------------------------------------------------

def test_xywh_to_crop_bounds_with_zero_padding():
    box = compute_padded_crop_box((10, 20, 30, 40), image_width=1000, image_height=1000, padding_fraction=0.0)
    assert box == (10, 20, 40, 60)  # x1,y1,x2=x+w,y2=y+h -- exactly the raw bbox, no padding


def test_default_padding_is_ten_percent_of_the_box_own_size():
    assert CROP_CONTEXT_PADDING_FRACTION == 0.10
    # box (100,100,200,200) -- 10% of w=200 is 20, 10% of h=200 is 20
    box = compute_padded_crop_box((100, 100, 200, 200), image_width=1000, image_height=1000)
    assert box == (80, 80, 320, 320)


def test_padding_is_proportional_to_the_box_not_the_image():
    # A tiny box in a huge image still gets padding proportional to ITS OWN size, not the
    # image's -- 10% of a 10px box is 1px, clamped to 0 on the low side (negative coords are
    # not valid pixel coordinates), giving (0,0,11,11), not some image-size-relative margin.
    small_box_in_huge_image = compute_padded_crop_box((0, 0, 10, 10), image_width=10000, image_height=10000)
    assert small_box_in_huge_image == (0, 0, 11, 11)


def test_padding_matches_expected_formula_precisely():
    x, y, w, h = 50, 60, 40, 20
    pad_x, pad_y = w * CROP_CONTEXT_PADDING_FRACTION, h * CROP_CONTEXT_PADDING_FRACTION
    expected = (round(x - pad_x), round(y - pad_y), round(x + w + pad_x), round(y + h + pad_y))
    box = compute_padded_crop_box((x, y, w, h), image_width=1000, image_height=1000)
    assert box == expected


def test_image_bound_clipping_on_all_four_sides():
    # box spills past every edge of a small image once padded.
    box = compute_padded_crop_box((-5, -5, 20, 20), image_width=15, image_height=15, padding_fraction=0.5)
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 15 and y2 <= 15


def test_clipping_never_produces_out_of_bounds_coordinates():
    box = compute_padded_crop_box((900, 900, 300, 300), image_width=1000, image_height=1000, padding_fraction=0.2)
    x1, y1, x2, y2 = box
    assert 0 <= x1 <= 1000
    assert 0 <= y1 <= 1000
    assert 0 <= x2 <= 1000
    assert 0 <= y2 <= 1000


def test_deterministic_given_the_same_inputs():
    args = ((30, 40, 50, 60), 500, 400, 0.1)
    assert compute_padded_crop_box(*args) == compute_padded_crop_box(*args)


# ---------------------------------------------------------------------------------------
# Zero-area / degenerate rejection
# ---------------------------------------------------------------------------------------

def test_zero_width_bbox_raises_crop_error():
    with pytest.raises(CropError, match="non-positive"):
        compute_padded_crop_box((10, 10, 0, 20), image_width=100, image_height=100)


def test_zero_height_bbox_raises_crop_error():
    with pytest.raises(CropError, match="non-positive"):
        compute_padded_crop_box((10, 10, 20, 0), image_width=100, image_height=100)


def test_negative_width_bbox_raises_crop_error():
    with pytest.raises(CropError, match="non-positive"):
        compute_padded_crop_box((10, 10, -5, 20), image_width=100, image_height=100)


def test_bbox_entirely_outside_image_bounds_raises_crop_error():
    # bbox is entirely to the right of a 100-wide image -- clips to a zero-width sliver.
    with pytest.raises(CropError, match="degenerate"):
        compute_padded_crop_box((150, 10, 20, 20), image_width=100, image_height=100)


def test_bbox_touching_only_the_edge_after_clipping_raises_crop_error():
    # bbox starts exactly at the image's right edge -- clipped crop has zero width.
    with pytest.raises(CropError, match="degenerate"):
        compute_padded_crop_box((100, 10, 20, 20), image_width=100, image_height=100, padding_fraction=0.0)


# ---------------------------------------------------------------------------------------
# crop_to_bbox -- real PIL images
# ---------------------------------------------------------------------------------------

def test_crop_to_bbox_returns_image_of_expected_size(tiny_image_factory):
    image = tiny_image_factory(size=(200, 200), color=(10, 20, 30))
    cropped, crop_box = crop_to_bbox(image, (50, 50, 40, 40), padding_fraction=0.0)
    x1, y1, x2, y2 = crop_box
    assert cropped.size == (x2 - x1, y2 - y1)
    assert crop_box == (50, 50, 90, 90)


def test_crop_to_bbox_never_mutates_the_original_image(tiny_image_factory):
    image = tiny_image_factory(size=(200, 200), color=(10, 20, 30))
    original_size = image.size
    original_pixel = image.getpixel((0, 0))

    cropped, _ = crop_to_bbox(image, (50, 50, 40, 40))

    assert image.size == original_size
    assert image.getpixel((0, 0)) == original_pixel
    assert cropped is not image


def test_crop_to_bbox_uses_the_images_own_real_size_not_a_stale_value(tiny_image_factory):
    """The crop must be computed against `image.size` itself -- critical for the shuffled
    sanity condition, where a DIFFERENT image (with a different real size) is paired with a
    bbox via metadata that was never updated to match.
    """
    small_image = tiny_image_factory(size=(50, 50), color=(1, 1, 1))
    large_image = tiny_image_factory(size=(500, 500), color=(2, 2, 2))

    _, small_crop_box = crop_to_bbox(small_image, (10, 10, 20, 20), padding_fraction=0.0)
    _, large_crop_box = crop_to_bbox(large_image, (10, 10, 20, 20), padding_fraction=0.0)

    # Same bbox, same padding, but bounds clipped against each image's OWN real dimensions.
    assert small_crop_box == large_crop_box == (10, 10, 30, 30)  # both comfortably in-bounds here
    # A bbox that would only be out-of-bounds for the SMALL image confirms per-image clipping:
    _, clipped_on_small = crop_to_bbox(small_image, (40, 40, 20, 20), padding_fraction=0.0)
    _, not_clipped_on_large = crop_to_bbox(large_image, (40, 40, 20, 20), padding_fraction=0.0)
    assert clipped_on_small == (40, 40, 50, 50)  # clipped to the 50x50 image's own edge
    assert not_clipped_on_large == (40, 40, 60, 60)  # not clipped in the 500x500 image


def test_crop_to_bbox_raises_crop_error_for_degenerate_result(tiny_image_factory):
    image = tiny_image_factory(size=(50, 50))
    with pytest.raises(CropError):
        crop_to_bbox(image, (200, 200, 20, 20))  # entirely outside the 50x50 image


def test_crop_to_bbox_default_padding_matches_module_constant(tiny_image_factory):
    image = tiny_image_factory(size=(1000, 1000))
    cropped_default, box_default = crop_to_bbox(image, (100, 100, 100, 100))
    cropped_explicit, box_explicit = crop_to_bbox(image, (100, 100, 100, 100), padding_fraction=CROP_CONTEXT_PADDING_FRACTION)
    assert box_default == box_explicit
