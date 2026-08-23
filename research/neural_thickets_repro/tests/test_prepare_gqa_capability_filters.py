"""Tests for prepare_gqa_capability_filters.py's pure-logic pieces via a fake `datasets`
module injected into sys.modules -- same convention as elsewhere in this project. No real
network/dataset access needed.
"""
import json
import sys
import types

import pytest

import neural_thickets_repro.prepare_gqa_capability_filters as m


def _install_fake_datasets_module(monkeypatch, rows, split_name="testdev_balanced"):
    fake_module = types.ModuleType("datasets")
    fake_module.get_dataset_split_names = lambda dataset_name, config_name: [split_name]
    fake_module.load_dataset = lambda dataset_name, config_name, split: _FakeHFDataset(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


class _FakeHFDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def select(self, indices):
        return _FakeHFDataset([self._rows[i] for i in indices])


def _relation_row(qid, argument):
    return {
        "id": qid,
        "types": {"semantic": "rel", "structural": "query"},
        "semantic": [{"operation": "relate", "argument": argument}],
    }


def test_load_raw_testdev_rows_returns_all_rows(monkeypatch):
    rows = [_relation_row("1", "left,s"), _relation_row("2", "holding,o")]
    _install_fake_datasets_module(monkeypatch, rows)
    result = m.load_raw_testdev_rows()
    assert len(result) == 2


def test_load_raw_testdev_rows_respects_sample_size(monkeypatch):
    rows = [_relation_row(str(i), "left,s") for i in range(10)]
    _install_fake_datasets_module(monkeypatch, rows)
    result = m.load_raw_testdev_rows(sample_size=3)
    assert len(result) == 3


def test_main_writes_filter_files_and_prints_counts(monkeypatch, tmp_path, capsys):
    rows = [
        _relation_row("s1", "to the left of,s"),
        _relation_row("r1", "holding,o"),
        {"id": "a1", "types": {"semantic": "attr", "structural": "query"}, "semantic": []},
    ]
    _install_fake_datasets_module(monkeypatch, rows)

    config_path = tmp_path / "gqa_repro.yaml"
    config_path.write_text(_MINIMAL_VALID_GQA_CONFIG)
    output_dir = tmp_path / "artifacts"

    rc = m.main(["--config", str(config_path), "--output-dir", str(output_dir)])

    assert rc == 0
    spatial = json.loads((output_dir / "gqa_spatial_ids.json").read_text())
    relational = json.loads((output_dir / "gqa_relational_ids.json").read_text())
    stats = json.loads((output_dir / "gqa_spatial_relational_stats.json").read_text())

    assert spatial == ["s1"]
    assert relational == ["r1"]
    assert stats["n_neither"] == 1

    captured = capsys.readouterr()
    assert "n_spatial" in captured.out


def test_main_inspect_only_does_not_persist_files(monkeypatch, tmp_path):
    rows = [_relation_row("s1", "left,s")]
    _install_fake_datasets_module(monkeypatch, rows)

    config_path = tmp_path / "gqa_repro.yaml"
    config_path.write_text(_MINIMAL_VALID_GQA_CONFIG)
    output_dir = tmp_path / "artifacts"

    rc = m.main(["--config", str(config_path), "--output-dir", str(output_dir), "--inspect-only"])

    assert rc == 0
    assert not output_dir.exists()


def test_main_stops_with_exit_1_when_schema_not_confirmed(monkeypatch, tmp_path):
    rows = [{"id": "1", "no_type_fields_at_all": True}]
    _install_fake_datasets_module(monkeypatch, rows)

    config_path = tmp_path / "gqa_repro.yaml"
    config_path.write_text(_MINIMAL_VALID_GQA_CONFIG)
    output_dir = tmp_path / "artifacts"

    rc = m.main(["--config", str(config_path), "--output-dir", str(output_dir)])

    assert rc == 1
    assert not output_dir.exists()  # never persists an untrustworthy filter


_MINIMAL_VALID_GQA_CONFIG = """
experiment: neural_thickets_gqa_reproduction
model:
  name: Qwen/Qwen2.5-VL-3B-Instruct
  revision: abc123
  architecture: Qwen2_5_VLForConditionalGeneration
  precision: bfloat16
dataset:
  name: GQA
  source: lmms-lab-encoder/GQA
  revision: abc123
  selection_split: train_balanced_instructions[:200]
  test_split: testdev_balanced_instructions
  selection_set_size: 200
  test_set_size: 12578
randopt:
  N: 5000
  K: 50
  sigmas: null
  sigma_candidates: {}
  perturbation_distribution: gaussian
  perturbation_scope: language_model_only
  freeze_visual_encoder: true
  visual_param_prefixes: ["visual."]
evaluation:
  answer_normalization: gqa_flexible_match
  decoding: greedy
  max_tokens: 256
  voting: majority
  tie_break: first_by_selection_score
reproducibility:
  global_seed: 42
gates:
  baseline_tolerance_pp:
    proceed_at_most: 1.0
    investigate_at_most: 3.0
  require_gate1_before_gate2: true
  require_gate2_before_gate3: true
hardware:
  min_free_disk_gb: 25
"""
