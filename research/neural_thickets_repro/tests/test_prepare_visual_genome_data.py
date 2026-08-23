"""Tests for prepare_visual_genome_data.py's pure logic against the REAL observed
AnnaZ1103/visual_genome_revised schema (confirmed on a real RunPod this session) -- flatten
logic, bbox validation, image-fetch-by-url logic, and parquet round-trip against synthetic
rows. load_attribute_rows/download_file's actual network calls are exercised only via fake
modules / monkeypatched functions. No real dataset download / GPU / ray / vllm needed.
"""
import json
import types
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")  # GPU-prep tooling dependency, not in requirements-cpu.txt -- see test_verify_gqa_data.py's own convention

import neural_thickets_repro.prepare_visual_genome_data as m  # noqa: E402
from neural_thickets_repro.prepare_visual_genome_data import VisualGenomeDataError  # noqa: E402


def _object(object_id=1, names=("chair",), attributes=("red",), x=10, y=10, w=20, h=20):
    return {
        "object_id": object_id, "names": list(names), "attributes": list(attributes),
        "synsets": ["chair.n.01"], "x": x, "y": y, "w": w, "h": h,
    }


def _image_row(image_id=1, url="https://example.com/1.jpg", width=800, height=800, objects=None):
    return {
        "image_id": image_id, "url": url, "width": width, "height": height,
        "coco_id": -1, "flickr_id": -1,
        "attributes": objects if objects is not None else [_object()],
    }


# ---------------------------------------------------------------------------------------
# validate_bbox
# ---------------------------------------------------------------------------------------

def test_validate_bbox_accepts_well_formed_box():
    assert m.validate_bbox(x=10, y=10, w=20, h=20, image_width=100, image_height=100) is True


@pytest.mark.parametrize("x,y,w,h", [(-1, 0, 10, 10), (0, -1, 10, 10), (0, 0, 0, 10), (0, 0, 10, 0)])
def test_validate_bbox_rejects_negative_or_nonpositive(x, y, w, h):
    assert m.validate_bbox(x, y, w, h, image_width=100, image_height=100) is False


def test_validate_bbox_rejects_box_extending_past_image_bounds():
    assert m.validate_bbox(x=90, y=0, w=20, h=10, image_width=100, image_height=100) is False  # x+w=110
    assert m.validate_bbox(x=0, y=90, w=10, h=20, image_width=100, image_height=100) is False  # y+h=110


def test_validate_bbox_allows_the_documented_one_pixel_upper_bound_tolerance():
    assert m.validate_bbox(x=0, y=0, w=101, h=100, image_width=100, image_height=100) is True  # x+w=101, tolerance=1
    assert m.validate_bbox(x=0, y=0, w=102, h=100, image_width=100, image_height=100) is False  # x+w=102, exceeds tolerance


# ---------------------------------------------------------------------------------------
# flatten_attribute_examples
# ---------------------------------------------------------------------------------------

def test_flatten_keeps_objects_with_attributes_and_names_and_valid_bbox():
    rows = [_image_row(image_id=2, objects=[
        _object(object_id=114, names=["sidewalk"], attributes=["brick", "white"], x=204, y=221, w=306, h=162),
        _object(object_id=22, names=["building"], attributes=["brown", "red"], x=363, y=0, w=146, h=265),
    ])]

    flattened, stats = m.flatten_attribute_examples(rows)

    assert len(flattened) == 2
    assert stats == {"n_image_rows": 1, "n_objects_seen": 2, "skipped_no_attributes": 0, "skipped_empty_name": 0, "skipped_invalid_bbox": 0}
    first = flattened[0]
    assert first["instance_id"] == "114"
    assert first["image_id"] == "2"
    assert first["object_name"] == "sidewalk"
    assert first["positive_attributes"] == ["brick", "white"]  # full multi-attribute set preserved
    assert first["bbox_x"] == 204 and first["bbox_w"] == 306


def test_flatten_skips_objects_with_no_positive_attributes():
    rows = [_image_row(objects=[_object(object_id=1, attributes=[])])]
    flattened, stats = m.flatten_attribute_examples(rows)
    assert flattened == []
    assert stats["skipped_no_attributes"] == 1


def test_flatten_skips_objects_with_empty_names():
    rows = [_image_row(objects=[_object(object_id=1, names=[])])]
    flattened, stats = m.flatten_attribute_examples(rows)
    assert flattened == []
    assert stats["skipped_empty_name"] == 1


def test_flatten_skips_objects_with_invalid_bbox():
    rows = [_image_row(width=100, height=100, objects=[_object(object_id=1, x=90, y=0, w=50, h=10)])]
    flattened, stats = m.flatten_attribute_examples(rows)
    assert flattened == []
    assert stats["skipped_invalid_bbox"] == 1


def test_flatten_raises_on_object_missing_required_field():
    bad_object = {"object_id": 1, "names": ["chair"], "attributes": ["red"], "x": 0, "y": 0, "w": 10}  # no "h"
    rows = [_image_row(objects=[bad_object])]
    with pytest.raises(VisualGenomeDataError, match="missing expected field"):
        m.flatten_attribute_examples(rows)


def test_flatten_multiple_objects_can_share_one_image():
    rows = [_image_row(image_id=5, objects=[_object(object_id=1), _object(object_id=2, names=["table"])])]
    flattened, stats = m.flatten_attribute_examples(rows)
    assert {r["image_id"] for r in flattened} == {"5"}
    assert stats["n_image_rows"] == 1
    assert stats["n_objects_seen"] == 2


# ---------------------------------------------------------------------------------------
# needed_images_with_urls / write_attributes_parquet
# ---------------------------------------------------------------------------------------

def test_needed_images_with_urls_deduplicates_by_image_id():
    rows = [
        {"image_id": "1", "url": "http://x/1.jpg"},
        {"image_id": "1", "url": "http://x/1.jpg"},
        {"image_id": "2", "url": "http://x/2.jpg"},
    ]
    assert m.needed_images_with_urls(rows) == {"1": "http://x/1.jpg", "2": "http://x/2.jpg"}


def test_write_attributes_parquet_round_trip(tmp_path):
    rows = [{
        "image_id": "1", "instance_id": "10", "object_name": "chair",
        "positive_attributes": ["red", "wooden"],
        "bbox_x": 1, "bbox_y": 2, "bbox_w": 3, "bbox_h": 4,
        "image_width": 100, "image_height": 100, "url": "http://x/1.jpg",
    }]
    out_path = tmp_path / "vg_attributes.parquet"
    m.write_attributes_parquet(rows, out_path)

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df.iloc[0]["image_id"] == "1"
    assert json.loads(df.iloc[0]["positive_attributes"]) == ["red", "wooden"]
    assert df.iloc[0]["bbox_x"] == 1 and df.iloc[0]["bbox_w"] == 3
    assert df.iloc[0]["url"] == "http://x/1.jpg"


def test_write_prepare_stats_round_trip(tmp_path):
    stats = {"n_flattened_examples": 3, "dataset_source": m.VG_DATASET_NAME}
    out_path = tmp_path / "vg_prepare_stats.json"
    m.write_prepare_stats(stats, out_path)
    assert json.loads(out_path.read_text()) == stats


# ---------------------------------------------------------------------------------------
# fetch_needed_images (per-image-URL download, no zip archive)
# ---------------------------------------------------------------------------------------

def test_fetch_needed_images_skips_download_when_all_already_present(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "1.jpg").write_bytes(b"already here")

    def _fail_download(url, dest):
        raise AssertionError("download_file should not be called when nothing is missing")

    monkeypatch.setattr(m, "download_file", _fail_download)
    m.fetch_needed_images({"1": "http://x/1.jpg"}, images_dir)


def test_fetch_needed_images_downloads_only_missing_ones(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "1.jpg").write_bytes(b"already here")

    calls = []

    def _fake_download(url, dest_path):
        calls.append((url, dest_path))
        dest_path.write_bytes(b"fetched")

    monkeypatch.setattr(m, "download_file", _fake_download)
    m.fetch_needed_images({"1": "http://x/1.jpg", "2": "http://x/2.jpg"}, images_dir)

    assert calls == [("http://x/2.jpg", images_dir / "2.jpg")]
    assert (images_dir / "2.jpg").read_bytes() == b"fetched"


def test_fetch_needed_images_raises_with_details_when_downloads_fail(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"

    def _fake_download(url, dest_path):
        raise VisualGenomeDataError(f"boom: {url}")

    monkeypatch.setattr(m, "download_file", _fake_download)
    with pytest.raises(VisualGenomeDataError, match="failed to download"):
        m.fetch_needed_images({"1": "http://x/1.jpg"}, images_dir)


def test_download_file_wraps_failure_with_actionable_message(tmp_path, monkeypatch):
    def _raising_urlretrieve(url, dest):
        raise OSError("HTTP Error 404")

    monkeypatch.setattr("urllib.request.urlretrieve", _raising_urlretrieve)
    with pytest.raises(VisualGenomeDataError, match="Failed to download image"):
        m.download_file("http://example/1.jpg", tmp_path / "out.jpg")


# ---------------------------------------------------------------------------------------
# load_attribute_rows (fake `datasets` module -- no network)
# ---------------------------------------------------------------------------------------

def _install_fake_datasets_module(monkeypatch, rows, columns):
    class _FakeHFDataset:
        def __init__(self, data):
            self._data = data
            self.column_names = columns

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def select(self, indices):
            return _FakeHFDataset([self._data[i] for i in indices])

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda name, config, split: _FakeHFDataset(rows)
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "datasets", fake_module)


def test_load_attribute_rows_hard_fails_on_missing_columns(monkeypatch):
    _install_fake_datasets_module(monkeypatch, [_image_row()], columns=["image_id", "url"])  # missing width/height/attributes
    with pytest.raises(VisualGenomeDataError, match="missing expected column"):
        m.load_attribute_rows("train", None)


def test_load_attribute_rows_respects_max_candidates(monkeypatch):
    rows = [_image_row(image_id=i) for i in range(20)]
    _install_fake_datasets_module(monkeypatch, rows, columns=list(m.REQUIRED_ROW_COLUMNS))
    result = m.load_attribute_rows("train", max_candidates=5)
    assert len(result) == 5
