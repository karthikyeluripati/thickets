"""Tests for run_capability_benchmark_gate.py's pure-logic pieces (spawn env var forcing,
adapter dotted-path resolution, output-dir construction) plus a full --dry-run integration
test. --dry-run never imports torch/vllm/transformers (checked directly in main()'s control
flow), so it's testable here without GPU -- same import-avoidance convention as
test_eval_base_image_aware_spawn_fix.py.
"""
import importlib
import json
import os

import pytest
import yaml

import neural_thickets_repro.run_capability_benchmark_gate as m
from neural_thickets_repro.benchmarks.adapters.counting_tallyqa import TallyQACountingBenchmark
from neural_thickets_repro.benchmarks.base import Example


def test_importing_module_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    importlib.reload(m)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_assert_spawn_configured_raises_when_unset(monkeypatch):
    importlib.reload(m)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    with pytest.raises(RuntimeError, match="spawn"):
        m._assert_spawn_configured()


def test_load_adapter_resolves_dotted_path():
    adapter = m.load_adapter("neural_thickets_repro.benchmarks.adapters.counting_tallyqa.TallyQACountingBenchmark")
    assert isinstance(adapter, TallyQACountingBenchmark)
    assert adapter.capability == "counting"


def test_load_adapter_raises_on_unknown_class():
    with pytest.raises(AttributeError):
        m.load_adapter("neural_thickets_repro.benchmarks.adapters.counting_tallyqa.NotARealClass")


def test_build_output_dir_nests_by_capability(tmp_path):
    out_dir = m.build_output_dir(tmp_path, "counting")
    assert out_dir == tmp_path / "counting"


_VALID_YAML_TEMPLATE = """
experiment: capability_benchmark_gate_test
model:
  name: Qwen/Qwen2.5-VL-3B-Instruct
  revision: abc123
  architecture: Qwen2_5_VLForConditionalGeneration
  precision: bfloat16
reproducibility:
  global_seed: 42
hardware:
  min_free_disk_gb: 10.0
generation:
  decoding: greedy
  max_tokens: 64
dataset:
  capability: counting
  adapter: neural_thickets_repro.benchmarks.adapters.counting_tallyqa.TallyQACountingBenchmark
  source: HuggingFaceM4/the_cauldron
  revision: null
  split: train
  subset_size: 2
  deviation_reason: "no held-out test split in this repackaging"
  subset_selection_rule: prefix
  subset_seed: null
gates:
  max_parser_failure_rate_pass: 0.02
  max_parser_failure_rate_needs_review: 0.10
  image_sanity_min_gap_pass: 0.05
  image_sanity_subset_size: 40
  floor_ceiling_low: 0.05
  floor_ceiling_high: 0.95
"""


def test_dry_run_loads_examples_builds_subset_and_writes_integrity_report(tmp_path, monkeypatch, tiny_image_factory):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)  # keep the subset-IDs artifact out of the real repo
    image = tiny_image_factory()
    fake_examples = [
        Example(example_id=str(i), image=image, prompt_input={"question": f"q{i}"}, target=i)
        for i in range(5)
    ]
    monkeypatch.setattr(TallyQACountingBenchmark, "load_examples", lambda self, cfg: fake_examples)

    config_path = tmp_path / "counting.yaml"
    config_path.write_text(_VALID_YAML_TEMPLATE)
    output_dir = tmp_path / "results"

    rc = m.main(["--config", str(config_path), "--output-dir", str(output_dir), "--dry-run"])

    assert rc == 0
    integrity_path = output_dir / "counting" / "integrity_report.json"
    assert integrity_path.exists()
    integrity = json.loads(integrity_path.read_text())
    assert integrity["n_loaded"] == 2  # subset_size=2 in the config
    assert integrity["passed"] is True

    subset_ids_path = tmp_path / "artifacts" / "benchmark_subsets" / "tallyqa_the_cauldron_2.json"
    assert subset_ids_path.exists()


def test_subset_size_override_flag(tmp_path, monkeypatch, tiny_image_factory):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    image = tiny_image_factory()
    fake_examples = [Example(example_id=str(i), image=image, prompt_input={"question": f"q{i}"}, target=i) for i in range(10)]
    monkeypatch.setattr(TallyQACountingBenchmark, "load_examples", lambda self, cfg: fake_examples)

    config_path = tmp_path / "counting.yaml"
    config_path.write_text(_VALID_YAML_TEMPLATE)  # config says subset_size: 2
    output_dir = tmp_path / "results"

    rc = m.main(["--config", str(config_path), "--output-dir", str(output_dir), "--subset-size", "5", "--dry-run"])

    assert rc == 0
    integrity = json.loads((output_dir / "counting" / "integrity_report.json").read_text())
    assert integrity["n_loaded"] == 5  # overridden, not the config's own 2

    subset_ids_path = tmp_path / "artifacts" / "benchmark_subsets" / "tallyqa_the_cauldron_5.json"
    assert subset_ids_path.exists()


def test_capability_mismatch_between_config_and_adapter_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    bad_yaml = _VALID_YAML_TEMPLATE.replace("capability: counting", "capability: wrong_capability")
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(bad_yaml)

    with pytest.raises(ValueError, match="does not match"):
        m.main(["--config", str(config_path), "--dry-run"])


def test_integrity_failure_stops_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    # No images at all -> integrity fails -> must return 1 and never reach model loading
    fake_examples = [Example(example_id=str(i), image=None, target=i) for i in range(5)]
    monkeypatch.setattr(TallyQACountingBenchmark, "load_examples", lambda self, cfg: fake_examples)

    config_path = tmp_path / "counting.yaml"
    config_path.write_text(_VALID_YAML_TEMPLATE)
    output_dir = tmp_path / "results"

    rc = m.main(["--config", str(config_path), "--output-dir", str(output_dir)])  # no --dry-run: integrity must still stop it first

    assert rc == 1
    integrity = json.loads((output_dir / "counting" / "integrity_report.json").read_text())
    assert integrity["passed"] is False
