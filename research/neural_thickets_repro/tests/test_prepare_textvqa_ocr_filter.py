"""Tests for prepare_textvqa_ocr_filter.py -- fake `datasets` module injection, same
convention as elsewhere in this project. No real dataset download / GPU / ray / vllm needed.
"""
import json
import sys
import types

import pytest

import neural_thickets_repro.prepare_textvqa_ocr_filter as m


class _FakeHFDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def select(self, indices):
        return _FakeHFDataset([self._rows[i] for i in indices])


def _row(question_id, answers, ocr_tokens):
    return {"question_id": question_id, "answers": answers, "ocr_tokens": ocr_tokens}


def _install_fake_datasets_module(monkeypatch, rows):
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda name, split: _FakeHFDataset(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_load_textvqa_rows_returns_all_rows(monkeypatch):
    rows = [_row("1", ["stop"] * 10, ["STOP"]), _row("2", ["4"] * 10, [])]
    _install_fake_datasets_module(monkeypatch, rows)
    result = m.load_textvqa_rows("validation")
    assert len(result) == 2


def test_load_textvqa_rows_respects_sample_size(monkeypatch):
    rows = [_row(str(i), ["x"] * 10, []) for i in range(10)]
    _install_fake_datasets_module(monkeypatch, rows)
    result = m.load_textvqa_rows("validation", sample_size=3)
    assert len(result) == 3


def test_build_ocr_grounded_filter_retains_only_ocr_supported_examples():
    rows = [
        _row("macbook", ["macbook air"] * 10, ["Apple", "MacBook", "Air"]),
        _row("wheels", ["4"] * 10, ["FORD"]),
        _row("lithia", ["lithia"] * 10, ["LITHIA"]),
        _row("book", ["yes", "unanswerable"] + ["yes"] * 8, ["PENGUIN"]),
        _row("soup", ["chicken noodle"] * 10, ["Chicken", "Noodle"]),
    ]
    retained, stats = m.build_ocr_grounded_filter(rows)

    assert set(retained) == {"macbook", "lithia", "soup"}
    assert stats == {"total_examples": 5, "retained": 3, "rejected": 2, "percent_retained": pytest.approx(60.0)}


def test_build_ocr_grounded_filter_empty_input_gives_zero_percent_no_crash():
    retained, stats = m.build_ocr_grounded_filter([])
    assert retained == []
    assert stats["percent_retained"] == 0.0


def test_persist_ocr_grounded_ids_writes_both_files(tmp_path):
    ids_path = tmp_path / "textvqa_ocr_grounded_ids.json"
    stats_path = tmp_path / "textvqa_ocr_grounded_stats.json"
    m.persist_ocr_grounded_ids(["b", "a"], {"total_examples": 2, "retained": 2}, ids_path, stats_path)

    assert json.loads(ids_path.read_text()) == ["a", "b"]  # sorted, deterministic
    assert json.loads(stats_path.read_text()) == {"total_examples": 2, "retained": 2}


def test_main_writes_filter_and_stats_and_prints_them(monkeypatch, tmp_path, capsys):
    rows = [
        _row("macbook", ["macbook air"] * 10, ["Apple", "MacBook", "Air"]),
        _row("wheels", ["4"] * 10, ["FORD"]),
    ]
    _install_fake_datasets_module(monkeypatch, rows)
    output_dir = tmp_path / "artifacts"

    rc = m.main(["--output-dir", str(output_dir)])

    assert rc == 0
    ids = json.loads((output_dir / "textvqa_ocr_grounded_ids.json").read_text())
    stats = json.loads((output_dir / "textvqa_ocr_grounded_stats.json").read_text())
    assert ids == ["macbook"]
    assert stats["retained"] == 1
    assert stats["total_examples"] == 2

    captured = capsys.readouterr()
    assert "percent_retained" in captured.out


def test_main_inspect_only_does_not_persist(monkeypatch, tmp_path):
    rows = [_row("macbook", ["macbook air"] * 10, ["MacBook", "Air"])]
    _install_fake_datasets_module(monkeypatch, rows)
    output_dir = tmp_path / "artifacts"

    rc = m.main(["--output-dir", str(output_dir), "--inspect-only"])

    assert rc == 0
    assert not output_dir.exists()
