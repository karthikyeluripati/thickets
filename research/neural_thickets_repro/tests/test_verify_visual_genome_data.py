"""Requires pandas + Pillow (GPU-prep tooling, not core Gate-0 scaffold logic) -- same
importorskip convention as test_verify_gqa_data.py.
"""
import json

import pytest

pd = pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")

from neural_thickets_repro.verify_visual_genome_data import verify_visual_genome_data  # noqa: E402


def _row(image_id, instance_id, positive_attributes=("red",)):
    return {
        "image_id": image_id, "instance_id": instance_id, "object_name": "chair",
        "positive_attributes": json.dumps(list(positive_attributes)),
        "bbox_x": 1, "bbox_y": 2, "bbox_w": 3, "bbox_h": 4,
    }


def _write_artifact(data_dir, rows, make_images=True, corrupt_ids=()):
    images_dir = data_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_parquet(data_dir / "vg_attributes.parquet")
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


def test_verify_fails_on_duplicate_instance_id(tmp_path):
    rows = [_row("img0", "5"), _row("img1", "5")]  # same instance_id twice
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert "5" in report["duplicate_instance_ids"]


def test_verify_fails_on_empty_positive_attributes(tmp_path):
    rows = [_row("img0", "1", positive_attributes=[])]
    _write_artifact(tmp_path, rows)

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert report["rows_with_empty_positive_attributes"] == 1


def test_verify_fails_on_missing_required_column(tmp_path):
    data_dir = tmp_path
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([{"image_id": "img0", "object_name": "chair"}]).to_parquet(data_dir / "vg_attributes.parquet")

    report = verify_visual_genome_data(tmp_path)
    assert report["ok"] is False
    assert "instance_id" in report["missing_columns"]
