"""Requires pandas + Pillow (GPU-prep tooling, not core Gate-0 scaffold logic) -- same
importorskip convention as test_verify_gqa_data.py.
"""
import json

import pytest

pd = pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")

from neural_thickets_repro.verify_visual_genome_data import verify_visual_genome_data  # noqa: E402


def _row(image_id, example_id, object_id=None, object_name="chair", positive_attributes=("red",), bbox=(1, 2, 3, 4), image_width=100, image_height=100):
    """`object_id` defaults to `example_id` for tests that don't care about the distinction --
    tests specifically exercising the real "same object_id, different example_id" case pass
    `object_id` explicitly.
    """
    return {
        "image_id": image_id, "example_id": example_id, "object_id": object_id if object_id is not None else example_id,
        "object_name": object_name,
        "positive_attributes": json.dumps(list(positive_attributes)),
        "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_w": bbox[2], "bbox_h": bbox[3],
        "image_width": image_width, "image_height": image_height, "url": f"http://x/{image_id}.jpg",
    }


def _write_artifact(data_dir, rows, make_images=True, corrupt_ids=(), stats=None):
    images_dir = data_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_parquet(data_dir / "vg_attributes.parquet")
    if stats is not None:
        (data_dir / "vg_prepare_stats.json").write_text(json.dumps(stats))
    if make_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for row in rows:
            img_id = row["image_id"]
            if img_id in seen:
                continue
            seen.add(img_id)
            path = images_dir / f"{img_id}.jpg"
            if img_id in corrupt_ids:
                path.write_bytes(b"not a real jpeg")
            else:
                Image.new("RGB", (4, 4)).save(path, "JPEG")


def test_verify_passes_for_complete_valid_data(tmp_path):
    rows = [_row(f"img{i}", str(i)) for i in range(3)]
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is True
    assert report["row_count"] == 3
    assert report["unique_image_ids"] == 3
    assert report["n_candidate_images"] == 3
    assert report["n_flattened_examples"] == 3
    assert report["n_selected"] == 3
    assert report["dataset_source"].startswith("AnnaZ1103/visual_genome_revised")


def test_verify_fails_on_missing_parquet(tmp_path):
    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert report["parquet_exists"] is False


def test_verify_fails_on_missing_images(tmp_path):
    rows = [_row(f"img{i}", str(i)) for i in range(3)]
    _write_artifact(tmp_path, rows, make_images=False)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert len(report["images_missing"]) == 3


def test_verify_fails_on_corrupt_image(tmp_path):
    rows = [_row(f"img{i}", str(i)) for i in range(3)]
    _write_artifact(tmp_path, rows, corrupt_ids={"img1"})

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    corrupt_ids = {c["image_id"] for c in report["images_corrupt"]}
    assert corrupt_ids == {"img1"}


def test_verify_fails_on_duplicate_example_id(tmp_path):
    rows = [_row("img0", "5"), _row("img1", "5")]  # same example_id twice
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert "5" in report["duplicate_example_ids"]


def test_verify_passes_when_source_object_id_repeats_but_example_id_is_distinct(tmp_path):
    """Mirrors the real RunPod finding: image_id=2 has two distinct "building" records
    sharing object_id=22 with different bboxes. The duplicate check must key on example_id,
    never on the source object_id, which this repackaged dataset does not guarantee unique.
    """
    rows = [
        _row("2", "vg:2:22:363:0:146:265", object_id="22", bbox=(363, 0, 146, 265), image_width=800, image_height=800),
        _row("2", "vg:2:22:108:0:166:205", object_id="22", bbox=(108, 0, 166, 205), image_width=800, image_height=800),
    ]
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is True
    assert report["duplicate_example_ids"] == []


def test_verify_preserves_source_object_id_as_a_column_distinct_from_example_id(tmp_path):
    rows = [_row("img0", "vg:img0:22:1:2:3:4", object_id="22")]
    _write_artifact(tmp_path, rows)

    df = pd.read_parquet(tmp_path / "vg_attributes.parquet")
    assert df.iloc[0]["object_id"] == "22"
    assert df.iloc[0]["example_id"] == "vg:img0:22:1:2:3:4"
    assert df.iloc[0]["object_id"] != df.iloc[0]["example_id"]


def test_verify_fails_on_empty_positive_attributes(tmp_path):
    rows = [_row("img0", "1", positive_attributes=[])]
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert report["rows_with_empty_positive_attributes"] == 1


def test_verify_fails_on_empty_object_name(tmp_path):
    rows = [_row("img0", "1", object_name="")]
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert report["rows_with_empty_object_name"] == 1


def test_verify_fails_on_invalid_bbox(tmp_path):
    rows = [_row("img0", "1", bbox=(90, 0, 50, 10), image_width=100, image_height=100)]  # x+w=140 > 100
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert report["rows_with_invalid_bbox"] == 1


def test_verify_fails_on_missing_required_column(tmp_path):
    data_dir = tmp_path
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([{"image_id": "img0", "object_name": "chair"}]).to_parquet(data_dir / "vg_attributes.parquet")

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert "example_id" in report["missing_columns"]
    assert "object_id" in report["missing_columns"]


def test_verify_reports_attribute_cardinality_distribution(tmp_path):
    rows = [
        _row("img0", "1", positive_attributes=["red"]),
        _row("img1", "2", positive_attributes=["red", "wooden"]),
        _row("img2", "3", positive_attributes=["red", "wooden", "old"]),
    ]
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["attribute_cardinality_distribution"] == {"1": 1, "2": 1, "3": 1}
    assert report["examples_with_multiple_attributes"] == 2


def test_verify_surfaces_prepare_stats_when_present(tmp_path):
    rows = [_row("img0", "1")]
    stats = {"dataset_source": "AnnaZ1103/visual_genome_revised", "n_flattened_examples": 1}
    _write_artifact(tmp_path, rows, stats=stats)

    report = verify_visual_genome_data(tmp_path)
    assert report["prepare_stats"] == stats


def test_verify_prepare_stats_is_none_when_absent(tmp_path):
    rows = [_row("img0", "1")]
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["prepare_stats"] is None
