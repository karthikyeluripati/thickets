"""Requires pandas + Pillow (already in requirements-gpu.txt; not in requirements-cpu.txt
since verify_gqa_data.py/prepare_gqa_data.py are Gate-1-prep tooling, not core Gate-0
scaffold logic). Skipped automatically if those aren't installed.
"""
import pytest

pd = pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")

from neural_thickets_repro.verify_gqa_data import verify_gqa_data  # noqa: E402


def _write_split(data_dir, name, rows, images_dir, make_images=True, corrupt_ids=()):
    df = pd.DataFrame.from_records(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / f"{name}.parquet")
    if make_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            img_id = row["imageId"]
            path = images_dir / f"{img_id}.jpg"
            if img_id in corrupt_ids:
                path.write_bytes(b"not a real jpeg")
            elif not path.exists():
                Image.new("RGB", (4, 4)).save(path, "JPEG")


def _row(i, image_id):
    return {"id": str(i), "imageId": image_id, "question": "q", "answer": "a", "fullAnswer": "full a"}


def test_verify_passes_for_complete_valid_data(tmp_path):
    images_dir = tmp_path / "images"
    testdev_rows = [_row(i, f"img{i}") for i in range(3)]
    train_rows = [_row(100 + i, f"img{i}") for i in range(2)]  # reuses img0, img1
    _write_split(tmp_path, "testdev", testdev_rows, images_dir)
    _write_split(tmp_path, "train", train_rows, images_dir)

    report = verify_gqa_data(tmp_path, selection_set_size=2, test_set_size=3)
    assert report["overall_ok"] is True
    assert report["testdev"]["row_count"] == 3
    assert report["testdev"]["unique_image_ids"] == 3
    assert report["train_selection"]["row_count"] == 2


def test_verify_fails_on_missing_parquet(tmp_path):
    report = verify_gqa_data(tmp_path, selection_set_size=2, test_set_size=3)
    assert report["overall_ok"] is False
    assert report["testdev"]["parquet_exists"] is False


def test_verify_fails_on_row_count_mismatch(tmp_path):
    images_dir = tmp_path / "images"
    rows = [_row(i, f"img{i}") for i in range(3)]
    _write_split(tmp_path, "testdev", rows, images_dir)
    _write_split(tmp_path, "train", rows[:2], images_dir)

    report = verify_gqa_data(tmp_path, selection_set_size=999, test_set_size=3)
    assert report["overall_ok"] is False
    assert report["train_selection"]["row_count_matches"] is False


def test_verify_fails_on_missing_images(tmp_path):
    images_dir = tmp_path / "images"
    rows = [_row(i, f"img{i}") for i in range(3)]
    _write_split(tmp_path, "testdev", rows, images_dir, make_images=False)
    _write_split(tmp_path, "train", rows[:1], images_dir, make_images=False)

    report = verify_gqa_data(tmp_path, selection_set_size=1, test_set_size=3)
    assert report["overall_ok"] is False
    assert len(report["testdev"]["images_missing"]) == 3


def test_verify_fails_on_corrupt_image(tmp_path):
    images_dir = tmp_path / "images"
    rows = [_row(i, f"img{i}") for i in range(3)]
    _write_split(tmp_path, "testdev", rows, images_dir, corrupt_ids={"img1"})
    _write_split(tmp_path, "train", rows[:1], images_dir)

    report = verify_gqa_data(tmp_path, selection_set_size=1, test_set_size=3)
    assert report["overall_ok"] is False
    corrupt_ids = {c["imageId"] for c in report["testdev"]["images_corrupt"]}
    assert corrupt_ids == {"img1"}


def test_verify_fails_on_missing_required_column(tmp_path):
    images_dir = tmp_path / "images"
    rows = [{"id": "0", "imageId": "img0", "question": "q"}]  # missing answer/fullAnswer
    _write_split(tmp_path, "testdev", rows, images_dir)
    _write_split(tmp_path, "train", rows, images_dir)

    report = verify_gqa_data(tmp_path, selection_set_size=1, test_set_size=1)
    assert report["overall_ok"] is False
    assert "answer" in report["testdev"]["missing_columns"]
