"""Tests for adapters/attribute_recognition_visualgenome.py -- synthetic VG-shaped rows, no
real dataset download / GPU / ray / vllm needed.
"""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.attribute_recognition_visualgenome import (
    VisualGenomeAttributeBenchmark,
    VisualGenomeSchemaError,
)
from neural_thickets_repro.benchmarks.base import Example


def _bench():
    return VisualGenomeAttributeBenchmark()


def test_capability_and_name():
    bench = _bench()
    assert bench.capability == "attribute_recognition"
    assert bench.name == "visual_genome_attributes"


def test_dataset_source_documents_the_two_source_migration():
    source = _bench().dataset_source()
    assert "mikewang/vaw" in source
    assert "prepare_visual_genome_data.py" in source


def test_known_caveats_documents_marker_protocol_and_multi_attribute_targets():
    caveats = " ".join(_bench().known_caveats())
    assert "not naturally occurring VG/VAW data" in caveats
    assert "matches ANY of them" in caveats


def test_prepare_image_draws_marker_and_preserves_original(tiny_image_factory):
    bench = _bench()
    image = tiny_image_factory(size=(20, 20), color=(0, 0, 0))
    example = Example(example_id="1", image=image, metadata={"bbox_xywh": [2, 2, 10, 10]})

    marked = bench.prepare_image(example)

    assert marked is not image  # a copy, not the same object
    assert example.image is image  # original untouched
    # Pixel at the drawn outline should now be red (marker color), original was pure black.
    assert marked.getpixel((2, 2)) == (255, 0, 0)
    assert image.getpixel((2, 2)) == (0, 0, 0)
    # Interior of the box (not on the outline) must remain unobscured (still black).
    assert marked.getpixel((6, 6)) == (0, 0, 0)


def test_prepare_image_none_image_returns_none():
    bench = _bench()
    example = Example(example_id="1", image=None, metadata={})
    assert bench.prepare_image(example) is None


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


def _write_prepared_artifact(data_dir, rows, tiny_image_factory, with_images=True):
    """Builds a real local vg_attributes.parquet + images/ dir, matching exactly what
    prepare_visual_genome_data.py writes -- load_examples() now reads this local artifact,
    never datasets.load_dataset(), so tests exercise the real pandas/PIL read path directly.
    """
    import json as json_module
    import pandas as pd

    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    records = []
    for row in rows:
        records.append({
            "image_id": row["image_id"], "instance_id": row["instance_id"], "object_name": row["object_name"],
            "positive_attributes": json_module.dumps(row["positive_attributes"]),
            "bbox_x": row["bbox"][0], "bbox_y": row["bbox"][1], "bbox_w": row["bbox"][2], "bbox_h": row["bbox"][3],
        })
        if with_images:
            tiny_image_factory().save(images_dir / f"{row['image_id']}.jpg")
    pd.DataFrame.from_records(records).to_parquet(data_dir / "vg_attributes.parquet")


def test_load_examples_reads_prepared_local_parquet_and_images(tmp_path, tiny_image_factory):
    rows = [
        {"image_id": "1", "instance_id": "10", "object_name": "chair", "positive_attributes": ["red", "wooden"], "bbox": [1, 2, 3, 4]},
        {"image_id": "2", "instance_id": "11", "object_name": "table", "positive_attributes": ["blue"], "bbox": [5, 6, 7, 8]},
    ]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert len(examples) == 2
    assert examples[0].example_id == "10"
    assert examples[0].target == ["red", "wooden"]
    assert examples[0].metadata["bbox_xywh"] == [1, 2, 3, 4]
    assert examples[0].image is not None


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
    rows = [{"image_id": "1", "instance_id": "10", "object_name": "chair", "positive_attributes": ["red"], "bbox": [1, 2, 3, 4]}]
    _write_prepared_artifact(tmp_path, rows, tiny_image_factory, with_images=False)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=str(tmp_path)))
    examples = bench.load_examples(cfg)

    assert examples[0].image is None
