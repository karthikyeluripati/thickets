"""Tests for config.py's Capability Benchmark Gate additions. ExperimentConfig/load_config
(the existing GQA config path) are untouched and not re-tested here -- see test_config.py.
"""
import pytest

from neural_thickets_repro.config import UnresolvedFieldError, load_capability_benchmark_config

_VALID_YAML = """
experiment: capability_benchmark_gate_fake
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
  capability: fake_capability
  adapter: neural_thickets_repro.benchmarks.adapters.fake.FakeBenchmark
  source: fake/dataset
  revision: null
  split: validation
  subset_size: 200
  deviation_reason: null
  subset_selection_rule: shuffled_prefix
  subset_seed: 42
gates:
  max_parser_failure_rate_pass: 0.02
  max_parser_failure_rate_needs_review: 0.10
  image_sanity_min_gap_pass: 0.05
  image_sanity_subset_size: 40
  floor_ceiling_low: 0.05
  floor_ceiling_high: 0.95
"""


def test_load_capability_benchmark_config_round_trip(tmp_path):
    path = tmp_path / "fake.yaml"
    path.write_text(_VALID_YAML)

    cfg = load_capability_benchmark_config(path)

    assert cfg.experiment == "capability_benchmark_gate_fake"
    assert cfg.model.name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert cfg.reproducibility.global_seed == 42
    assert cfg.generation.max_tokens == 64
    assert cfg.dataset.capability == "fake_capability"
    assert cfg.dataset.subset_size == 200
    assert cfg.gates.max_parser_failure_rate_pass == 0.02


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_capability_benchmark_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_missing_section_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: only_this_field\n")
    with pytest.raises(KeyError):
        load_capability_benchmark_config(path)


def test_require_resolved_raises_on_none_field(tmp_path):
    path = tmp_path / "fake.yaml"
    path.write_text(_VALID_YAML)
    cfg = load_capability_benchmark_config(path)

    with pytest.raises(UnresolvedFieldError):
        cfg.require_resolved("dataset.revision")


def test_require_resolved_passes_for_set_field(tmp_path):
    path = tmp_path / "fake.yaml"
    path.write_text(_VALID_YAML)
    cfg = load_capability_benchmark_config(path)

    cfg.require_resolved("model.revision", "dataset.split")  # should not raise
