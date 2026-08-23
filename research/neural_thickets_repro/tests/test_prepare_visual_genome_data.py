"""Tests for prepare_visual_genome_data.py's pure logic -- zip-member extraction is tested
against a real synthetic in-memory zip (no network), the VAW-row-shaping logic against
synthetic rows, and load_vaw_rows/download_file's actual network calls are exercised only via
fake modules / monkeypatched functions. No real dataset download / GPU / ray / vllm needed.
"""
import json
import types
import zipfile
from io import BytesIO

import pytest

pd = pytest.importorskip("pandas")  # GPU-prep tooling dependency, not in requirements-cpu.txt -- see test_verify_gqa_data.py's own convention

import neural_thickets_repro.prepare_visual_genome_data as m  # noqa: E402
from neural_thickets_repro.prepare_visual_genome_data import VisualGenomeDataError  # noqa: E402


def _row(image_id, instance_id, object_name="chair", positive_attributes=("red",), bbox=(1, 2, 3, 4)):
    return {
        "image_id": image_id, "instance_id": instance_id, "object_name": object_name,
        "positive_attributes": list(positive_attributes), "instance_bbox": list(bbox),
    }


def test_rows_with_positive_attributes_filters_empty_attribute_rows():
    rows = [_row("1", "a", positive_attributes=["red"]), _row("2", "b", positive_attributes=[])]
    kept = m.rows_with_positive_attributes(rows)
    assert [r["instance_id"] for r in kept] == ["a"]


def test_rows_with_positive_attributes_raises_on_malformed_bbox():
    rows = [_row("1", "a", bbox=(1, 2, 3))]  # only 3 elements, not 4
    with pytest.raises(VisualGenomeDataError, match="malformed instance_bbox"):
        m.rows_with_positive_attributes(rows)


def test_needed_image_ids_deduplicates():
    rows = [_row("1", "a"), _row("1", "b"), _row("2", "c")]
    assert m.needed_image_ids(rows) == {"1", "2"}


def test_write_attributes_parquet_round_trip(tmp_path):
    rows = [_row("1", "a", positive_attributes=["red", "wooden"], bbox=(1, 2, 3, 4))]
    out_path = tmp_path / "vg_attributes.parquet"
    m.write_attributes_parquet(rows, out_path)

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df.iloc[0]["image_id"] == "1"
    assert json.loads(df.iloc[0]["positive_attributes"]) == ["red", "wooden"]
    assert df.iloc[0]["bbox_x"] == 1 and df.iloc[0]["bbox_w"] == 3


def _make_zip_bytes(members: dict) -> BytesIO:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf


def test_extract_needed_images_from_zip_extracts_only_needed_members(tmp_path):
    zip_bytes = _make_zip_bytes({
        "VG_100K/1.jpg": b"fake-jpg-bytes-1",
        "VG_100K/2.jpg": b"fake-jpg-bytes-2",
        "VG_100K/3.jpg": b"fake-jpg-bytes-3",
    })
    images_dir = tmp_path / "images"

    found = m.extract_needed_images_from_zip(zip_bytes, {"1", "3", "99"}, images_dir)

    assert found == {"1", "3"}  # "99" genuinely isn't in this zip
    assert (images_dir / "1.jpg").read_bytes() == b"fake-jpg-bytes-1"
    assert (images_dir / "3.jpg").read_bytes() == b"fake-jpg-bytes-3"
    assert not (images_dir / "2.jpg").exists()  # not needed -- never extracted


def test_extract_needed_images_from_zip_empty_needed_set_extracts_nothing(tmp_path):
    zip_bytes = _make_zip_bytes({"VG_100K/1.jpg": b"x"})
    images_dir = tmp_path / "images"
    found = m.extract_needed_images_from_zip(zip_bytes, set(), images_dir)
    assert found == set()
    assert list(images_dir.glob("*.jpg")) == []


def test_fetch_needed_images_skips_download_when_all_already_present(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "1.jpg").write_bytes(b"already here")

    def _fail_download(url, dest):
        raise AssertionError("download_file should not be called when nothing is missing")

    monkeypatch.setattr(m, "download_file", _fail_download)
    m.fetch_needed_images({"1"}, images_dir, tmp_path / "cache", "http://part1", "http://part2")


def test_fetch_needed_images_tries_part2_for_what_part1_lacks(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    part1_zip = _make_zip_bytes({"VG_100K/1.jpg": b"one"})
    part2_zip = _make_zip_bytes({"VG_100K_2/2.jpg": b"two"})

    call_log = []

    def _fake_download(url, dest_path):
        call_log.append(url)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        content = part1_zip.getvalue() if "part1" in url else part2_zip.getvalue()
        dest_path.write_bytes(content)

    monkeypatch.setattr(m, "download_file", _fake_download)
    m.fetch_needed_images({"1", "2"}, images_dir, cache_dir, "http://example/part1.zip", "http://example/part2.zip")

    assert (images_dir / "1.jpg").exists()
    assert (images_dir / "2.jpg").exists()
    assert call_log == ["http://example/part1.zip", "http://example/part2.zip"]


def test_fetch_needed_images_raises_if_still_missing_after_both_archives(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    part1_zip = _make_zip_bytes({"VG_100K/1.jpg": b"one"})
    part2_zip = _make_zip_bytes({})

    def _fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        content = part1_zip.getvalue() if "part1" in url else part2_zip.getvalue()
        dest_path.write_bytes(content)

    monkeypatch.setattr(m, "download_file", _fake_download)
    with pytest.raises(VisualGenomeDataError, match="not found in either archive"):
        m.fetch_needed_images({"1", "999"}, images_dir, tmp_path / "cache", "http://example/part1.zip", "http://example/part2.zip")


def test_download_file_wraps_failure_with_actionable_message(tmp_path, monkeypatch):
    def _raising_urlretrieve(url, dest):
        raise OSError("HTTP Error 404")

    monkeypatch.setattr("urllib.request.urlretrieve", _raising_urlretrieve)
    with pytest.raises(VisualGenomeDataError, match="images-zip-url"):
        m.download_file("http://example/missing.zip", tmp_path / "out.zip")


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
    fake_module.load_dataset = lambda name, split: _FakeHFDataset(rows)
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "datasets", fake_module)


def test_load_vaw_rows_hard_fails_on_missing_columns(monkeypatch):
    _install_fake_datasets_module(monkeypatch, [_row("1", "a")], columns=["image_id", "instance_id"])  # missing object_name/positive_attributes/instance_bbox
    with pytest.raises(VisualGenomeDataError, match="missing expected column"):
        m.load_vaw_rows("train", None)


def test_load_vaw_rows_respects_max_candidates(monkeypatch):
    rows = [_row(str(i), str(i)) for i in range(20)]
    _install_fake_datasets_module(monkeypatch, rows, columns=list(m.REQUIRED_VAW_COLUMNS))
    result = m.load_vaw_rows("train", max_candidates=5)
    assert len(result) == 5
