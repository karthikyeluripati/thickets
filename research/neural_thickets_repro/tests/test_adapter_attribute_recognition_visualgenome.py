"""Tests for adapters/attribute_recognition_visualgenome.py -- synthetic VG-shaped rows, no
real dataset download / GPU / ray / vllm needed. Covers the LOCALIZED-CROP protocol (this
repair pass): prepare_image() now returns a crop, not a full image + marker overlay.
"""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.attribute_recognition_visualgenome import (
    CROP_CONTEXT_PADDING_FRACTION,
    VisualGenomeAttributeBenchmark,
    VisualGenomeSchemaError,
)
from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.benchmarks.image_crop import CropError


def _bench():
    return VisualGenomeAttributeBenchmark()


def test_capability_and_name():
    bench = _bench()
    assert bench.capability == "attribute_recognition"
    assert bench.name == "visual_genome_attributes"


def test_dataset_source_documents_the_current_source():
    source = _bench().dataset_source()
    assert "AnnaZ1103/visual_genome_revised" in source
    assert "prepare_visual_genome_data.py" in source


def test_known_caveats_documents_the_localized_crop_protocol_and_multi_attribute_targets():
    caveats = " ".join(_bench().known_caveats())
    assert "matches ANY of them" in caveats
    assert "AnnaZ1103/visual_genome_revised" in caveats
    assert "CROP of the annotated bbox" in caveats
    assert "NO measurable visual dependence" in caveats
    assert "never inferred, altered, or treated as attribute/target information" in caveats


def test_known_caveats_documents_the_shuffled_condition_pairs_image_and_bbox_together():
    caveats = " ".join(_bench().known_caveats())
    assert "swaps in a DIFFERENT example's own" in caveats
    assert "(image, bbox) pair together" in caveats


def test_known_caveats_flags_state_action_attributes_for_later_ontology_review():
    caveats = " ".join(_bench().known_caveats())
    assert "walking" in caveats
    assert "hanging" in caveats
    assert "manual ontology review" in caveats
    assert "never filtered, dropped, or reweighted" in caveats


def test_known_caveats_documents_the_value_vs_category_prompt_fix():
    caveats = " ".join(_bench().known_caveats())
    assert "attribute VALUE" in caveats
    assert "bare category label" in caveats


def test_build_prompt_asks_for_value_not_category_and_mentions_object_name():
    bench = _bench()
    example = Example(example_id="1", prompt_input={"object_name": "chair"})
    text = bench.build_prompt(example)[0]["content"][1]["text"]
    assert "VALUE" in text
    assert "category" in text
    assert "Material" in text  # explicit counter-example of what NOT to answer with
    assert "chair" in text  # object name filled into the template


def test_build_prompt_never_mentions_bbox_coordinates_or_this_examples_ground_truth():
    bench = _bench()
    # deliberately distinctive target values that do NOT appear anywhere in the fixed
    # instruction template's own illustrative examples ("brown"/"wooden" do appear there as
    # generic examples, unrelated to any specific example's real target -- that's fine).
    example = Example(example_id="1", prompt_input={"object_name": "chair"}, target=["polka-dotted", "chartreuse"])
    text = bench.build_prompt(example)[0]["content"][1]["text"]
    assert "polka-dotted" not in text
    assert "chartreuse" not in text
    # no raw numeric coordinates leaked into the prompt text
    import re
    assert not re.search(r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+", text)


# ---------------------------------------------------------------------------------------
# prepare_image -- LOCALIZED CROP (replaces the earlier marker-overlay protocol)
# ---------------------------------------------------------------------------------------

def test_prepare_image_returns_a_crop_not_the_full_image(tiny_image_factory):
    bench = _bench()
    image = tiny_image_factory(size=(200, 200), color=(0, 0, 0))
    example = Example(example_id="1", image=image, metadata={"bbox_xywh": [50, 50, 40, 40]})

    cropped = bench.prepare_image(example)

    assert cropped is not image  # a new, cropped image -- not the same object, not merely a copy
    assert cropped.size != image.size
    assert cropped.size[0] < image.size[0]
    assert cropped.size[1] < image.size[1]


def test_prepare_image_crop_size_matches_padded_bbox(tiny_image_factory):
    bench = _bench()
    image = tiny_image_factory(size=(1000, 1000), color=(0, 0, 0))
    bbox = [100, 100, 100, 100]
    example = Example(example_id="1", image=image, metadata={"bbox_xywh": bbox})

    cropped = bench.prepare_image(example)

    pad = 100 * CROP_CONTEXT_PADDING_FRACTION
    expected_side = 100 + 2 * pad
    assert cropped.size == (round(expected_side), round(expected_side))


def test_prepare_image_never_mutates_the_original_image(tiny_image_factory):
    bench = _bench()
    image = tiny_image_factory(size=(200, 200), color=(5, 6, 7))
    original_size = image.size
    original_pixel = image.getpixel((0, 0))
    example = Example(example_id="1", image=image, metadata={"bbox_xywh": [50, 50, 40, 40]})

    bench.prepare_image(example)

    assert image.size == original_size
    assert image.getpixel((0, 0)) == original_pixel


def test_prepare_image_none_image_returns_none():
    bench = _bench()
    example = Example(example_id="1", image=None, metadata={})
    assert bench.prepare_image(example) is None


def test_prepare_image_raises_crop_error_for_a_degenerate_bbox(tiny_image_factory):
    bench = _bench()
    image = tiny_image_factory(size=(50, 50))
    example = Example(example_id="1", image=image, metadata={"bbox_xywh": [200, 200, 20, 20]})
    with pytest.raises(CropError):
        bench.prepare_image(example)


# ---------------------------------------------------------------------------------------
# make_shuffled_image_variant -- shuffled condition = a DIFFERENT example's own (image,
# bbox) pair, paired with THIS example's own prompt/target (this repair pass)
# ---------------------------------------------------------------------------------------

def test_make_shuffled_image_variant_swaps_image_and_bbox_together(tiny_image_factory):
    bench = _bench()
    own_image = tiny_image_factory(size=(100, 100), color=(1, 1, 1))
    source_image = tiny_image_factory(size=(300, 300), color=(2, 2, 2))
    example = Example(
        example_id="own", image=own_image, image_ref="own.jpg",
        prompt_input={"object_name": "chair"}, target=["wooden"],
        metadata={"bbox_xywh": [10, 10, 20, 20], "crop_box_xyxy": [5, 5, 35, 35]},
    )
    source = Example(
        example_id="source", image=source_image, image_ref="source.jpg",
        prompt_input={"object_name": "table"}, target=["metal"],
        metadata={"bbox_xywh": [100, 100, 50, 50], "crop_box_xyxy": [95, 95, 155, 155]},
    )

    shuffled = bench.make_shuffled_image_variant(example, source)

    assert shuffled.example_id == "own"  # keeps THIS example's own id
    assert shuffled.image is source_image  # visual input comes from source
    assert shuffled.metadata["bbox_xywh"] == [100, 100, 50, 50]  # bbox comes from source too
    assert shuffled.metadata["crop_box_xyxy"] == [95, 95, 155, 155]


def test_make_shuffled_image_variant_preserves_original_prompt_and_target(tiny_image_factory):
    bench = _bench()
    example = Example(
        example_id="own", image=tiny_image_factory(size=(100, 100)), prompt_input={"object_name": "chair"}, target=["wooden"],
        metadata={"bbox_xywh": [10, 10, 20, 20]},
    )
    source = Example(
        example_id="source", image=tiny_image_factory(size=(100, 100)), prompt_input={"object_name": "table"}, target=["metal"],
        metadata={"bbox_xywh": [10, 10, 20, 20]},
    )

    shuffled = bench.make_shuffled_image_variant(example, source)

    assert shuffled.prompt_input == {"object_name": "chair"}  # NOT source's prompt
    assert shuffled.target == ["wooden"]  # NOT source's target


def test_make_shuffled_image_variant_records_source_id_for_audit(tiny_image_factory):
    bench = _bench()
    example = Example(example_id="own", image=tiny_image_factory(), prompt_input={}, target=[], metadata={"bbox_xywh": [1, 1, 2, 2]})
    source = Example(example_id="source", image=tiny_image_factory(), prompt_input={}, target=[], metadata={"bbox_xywh": [1, 1, 2, 2]})

    shuffled = bench.make_shuffled_image_variant(example, source)
    assert shuffled.metadata["sanity_shuffle_source_id"] == "source"


def test_make_shuffled_image_variant_never_mutates_either_input_example(tiny_image_factory):
    bench = _bench()
    example = Example(example_id="own", image=tiny_image_factory(), prompt_input={}, target=[], metadata={"bbox_xywh": [1, 1, 2, 2]})
    source = Example(example_id="source", image=tiny_image_factory(), prompt_input={}, target=[], metadata={"bbox_xywh": [9, 9, 9, 9]})

    bench.make_shuffled_image_variant(example, source)

    assert example.metadata["bbox_xywh"] == [1, 1, 2, 2]  # untouched
    assert source.metadata["bbox_xywh"] == [9, 9, 9, 9]  # untouched


def test_prepare_image_on_shuffled_variant_crops_the_swapped_in_image_and_bbox(tiny_image_factory):
    """End-to-end: the shuffled variant's prepare_image() output must be a crop of the
    SOURCE image using the SOURCE bbox -- never the original example's bbox applied to the
    swapped-in image (which would be a misaligned/meaningless crop).
    """
    bench = _bench()
    own_image = tiny_image_factory(size=(50, 50), color=(1, 1, 1))
    source_image = tiny_image_factory(size=(400, 400), color=(2, 2, 2))
    example = Example(example_id="own", image=own_image, prompt_input={}, target=[], metadata={"bbox_xywh": [1000, 1000, 10, 10]})  # would be invalid on source's smaller region if misapplied
    source = Example(example_id="source", image=source_image, prompt_input={}, target=[], metadata={"bbox_xywh": [100, 100, 100, 100]})

    shuffled = bench.make_shuffled_image_variant(example, source)
    cropped = bench.prepare_image(shuffled)

    # A valid crop was produced (the source's own valid bbox was used, not "own"'s bbox,
    # which would be nowhere near the source image's 400x400 bounds in a sensible way).
    assert cropped is not None
    assert cropped.size[0] > 0 and cropped.size[1] > 0


# ---------------------------------------------------------------------------------------
# scoring -- unchanged (still multi-attribute, match-any)
# ---------------------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------------------
# load_examples -- subset/dataset/scoring unchanged; NEW: crop-validity pre-filtering
# ---------------------------------------------------------------------------------------

def _write_prepared_artifact(data_dir, rows, tiny_image_factory, with_images=True, with_image_dims=False, image_size=(200, 200)):
    """Builds a real local vg_attributes.parquet + images/ dir, matching exactly what
    prepare_visual_genome_data.py writes -- load_examples() now reads this local artifact,
    never datasets.load_dataset(), so tests exercise the real pandas/PIL read path directly.
    `image_size` defaults to a generously large 200x200 so ordinary bbox fixtures never
    accidentally trip the (deliberately tested separately) crop-validity exclusion.
    """
    import json as json_module
    import pandas as pd

    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    records = []
    for row in rows:
        record = {
            "image_id": row["image_id"], "example_id": row["example_id"],
            "object_id": row.get("object_id", row["example_id"]), "object_name": row["object_name"],
            "positive_attributes": json_module.dumps(row["positive_attributes"]),
            "bbox_x": row["bbox"][0], "bbox_y": row["bbox"][1], "bbox_w": row["bbox"][2], "bbox_h": row["bbox"][3],
        }
        if with_image_dims:
            record["image_width"] = row.get("image_width", image_size[0])
            record["image_height"] = row.get("image_height", image_size[1])
        records.append(record)
        if with_images:
            tiny_image_factory(size=row.get("image_size", image_size)).save(images_dir / f"{row['image_id']}.jpg")
    pd.DataFrame.from_records(records).to_parquet(data_dir / "vg_attributes.parquet")


def test_load_examples_reads_prepared_local_parquet_and_images(tmp_path, tiny_image_factory):
    rows = [
        {"image_id": "1", "example_id": "vg:1:10:1:2:3:4", "object_id": "10", "object_name": "chair", "positive_attributes": ["red", "wooden"], "bbox": [1, 2, 3, 4]},
        {"image_id": "2", "example_id": "vg:2:11:5:6:7:8", "object_id": "11", "object_name": "table", "positive_attributes": ["blue"], "bbox": [5, 6, 7, 8]},
    ]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert len(examples) == 2
    assert examples[0].example_id == "vg:1:10:1:2:3:4"
    assert examples[0].metadata["object_id"] == "10"  # source object_id preserved, separate from example_id
    assert examples[0].target == ["red", "wooden"]
    assert examples[0].metadata["bbox_xywh"] == [1, 2, 3, 4]
    assert examples[0].image is not None


def test_load_examples_records_crop_box_xyxy_metadata(tmp_path, tiny_image_factory):
    rows = [{"image_id": "1", "example_id": "vg:1:10:1:2:3:4", "object_id": "10", "object_name": "chair", "positive_attributes": ["red"], "bbox": [50, 50, 40, 40]}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory, image_size=(200, 200))

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    pad = 40 * CROP_CONTEXT_PADDING_FRACTION
    expected = [round(50 - pad), round(50 - pad), round(50 + 40 + pad), round(50 + 40 + pad)]
    assert examples[0].metadata["crop_box_xyxy"] == expected


def test_load_examples_excludes_rows_with_a_degenerate_crop(tmp_path, tiny_image_factory):
    """A bbox entirely outside the image's own bounds cannot produce a valid crop -- the
    row must be excluded from the candidate pool, never silently cropped to garbage.
    """
    rows = [
        {"image_id": "good", "example_id": "vg:good", "object_id": "1", "object_name": "chair", "positive_attributes": ["red"], "bbox": [10, 10, 20, 20], "image_size": (100, 100)},
        {"image_id": "bad", "example_id": "vg:bad", "object_id": "2", "object_name": "table", "positive_attributes": ["blue"], "bbox": [500, 500, 20, 20], "image_size": (100, 100)},  # entirely outside a 100x100 image
    ]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    example_ids = {e.example_id for e in examples}
    assert example_ids == {"vg:good"}  # "vg:bad" excluded


def test_load_examples_zero_area_bbox_is_excluded(tmp_path, tiny_image_factory):
    rows = [{"image_id": "1", "example_id": "vg:1", "object_id": "1", "object_name": "chair", "positive_attributes": ["red"], "bbox": [10, 10, 0, 20]}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)
    assert examples == []


def test_load_examples_gives_distinct_example_ids_for_same_object_id_different_bbox(tmp_path, tiny_image_factory):
    """Mirrors the real RunPod finding: image_id=2 has two "building" records sharing
    object_id=22 with different bboxes -- load_examples() must surface two distinct examples,
    both correctly tagged with the shared source object_id in metadata.
    """
    rows = [
        {"image_id": "2", "example_id": "vg:2:22:363:0:146:265", "object_id": "22", "object_name": "building", "positive_attributes": ["brown", "red"], "bbox": [363, 0, 146, 265], "image_size": (800, 800)},
        {"image_id": "2", "example_id": "vg:2:22:108:0:166:205", "object_id": "22", "object_name": "building", "positive_attributes": ["brown", "red"], "bbox": [108, 0, 166, 205], "image_size": (800, 800)},
    ]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert len(examples) == 2
    example_ids = {e.example_id for e in examples}
    assert len(example_ids) == 2, "the two records must remain distinct benchmark examples"
    assert {e.metadata["object_id"] for e in examples} == {"22"}  # shared source object_id preserved on both


def test_load_examples_carries_image_dims_into_metadata_when_present(tmp_path, tiny_image_factory):
    rows = [{"image_id": "1", "example_id": "vg:1:10:1:2:3:4", "object_id": "10", "object_name": "chair", "positive_attributes": ["red"], "bbox": [10, 10, 20, 20], "image_width": 640, "image_height": 480, "image_size": (640, 480)}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory, with_image_dims=True)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert examples[0].metadata["image_width"] == 640
    assert examples[0].metadata["image_height"] == 480


def test_load_examples_strips_target_whitespace_but_preserves_raw_in_metadata(tmp_path, tiny_image_factory):
    """Real N=5 finding: a raw target value had stray trailing whitespace ("wooden "). The
    stored target (used for scoring) must be clean; the exact raw values must still be
    available in metadata for audit.
    """
    rows = [{"image_id": "1", "example_id": "vg:1:10:1:2:3:4", "object_id": "10", "object_name": "chair", "positive_attributes": ["wooden ", "brown"], "bbox": [10, 10, 20, 20]}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert examples[0].target == ["wooden", "brown"]  # stripped
    assert examples[0].metadata["raw_positive_attributes"] == ["wooden ", "brown"]  # exact, unmodified


def test_load_examples_flags_state_action_attributes_without_dropping_them(tmp_path, tiny_image_factory):
    rows = [{"image_id": "1", "example_id": "vg:1:10:1:2:3:4", "object_id": "10", "object_name": "board", "positive_attributes": ["cork", "hanging"], "bbox": [10, 10, 20, 20]}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert examples[0].target == ["cork", "hanging"]  # never dropped
    assert examples[0].metadata["flagged_state_action_attributes"] == ["hanging"]


def test_load_examples_no_flagged_attributes_gives_empty_list(tmp_path, tiny_image_factory):
    rows = [{"image_id": "1", "example_id": "vg:1:10:1:2:3:4", "object_id": "10", "object_name": "chair", "positive_attributes": ["red", "wooden"], "bbox": [10, 10, 20, 20]}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert examples[0].metadata["flagged_state_action_attributes"] == []


def test_load_examples_hard_fails_when_parquet_missing(tmp_path):
    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    with pytest.raises(VisualGenomeSchemaError, match="prepare_visual_genome_data"):
        bench.load_examples(cfg)


def test_load_examples_hard_fails_on_missing_expected_columns(tmp_path):
    import pandas as pd
    tmp_path.mkdir(exist_ok=True)
    pd.DataFrame.from_records([{"image_id": "1", "object_name": "chair"}]).to_parquet(tmp_path / "vg_attributes.parquet")

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    with pytest.raises(VisualGenomeSchemaError, match="missing expected column"):
        bench.load_examples(cfg)


def test_load_examples_missing_image_file_yields_none_image_not_a_crash(tmp_path, tiny_image_factory):
    rows = [{"image_id": "1", "example_id": "vg:1:10:1:2:3:4", "object_id": "10", "object_name": "chair", "positive_attributes": ["red"], "bbox": [10, 10, 20, 20]}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory, with_images=False)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert examples[0].image is None
    assert examples[0].metadata["crop_box_xyxy"] is None  # cannot compute a crop without a real image
