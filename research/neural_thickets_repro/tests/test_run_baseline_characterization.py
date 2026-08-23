"""Tests for run_baseline_characterization.py's pure-logic pieces (config-path resolution,
model-match assertion, runtime-version reporting) plus a multi-capability --dry-run
integration test. --dry-run never imports torch/vllm/transformers, so it's testable here
without GPU -- same import-avoidance convention as test_run_capability_benchmark_gate.py.
"""
import json
from types import SimpleNamespace

import pytest

import neural_thickets_repro.run_baseline_characterization as m
from neural_thickets_repro.benchmarks.adapters.counting_tallyqa import TallyQACountingBenchmark
from neural_thickets_repro.benchmarks.adapters.fine_grained_recognition_cub import CUBFineGrainedBenchmark
from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.config import CapabilityBenchmarkConfig, load_capability_benchmark_config


def test_default_capability_configs_excludes_imagenet_and_full_textvqa():
    assert "object_recognition.yaml" not in m.DEFAULT_CAPABILITY_CONFIGS
    assert "ocr_text_recognition.yaml" not in m.DEFAULT_CAPABILITY_CONFIGS
    assert "ocr_text_recognition_grounded.yaml" in m.DEFAULT_CAPABILITY_CONFIGS
    assert len(m.DEFAULT_CAPABILITY_CONFIGS) == 7


def test_resolve_config_paths_default_set_resolves_under_configs_benchmarks():
    paths = m.resolve_config_paths()
    assert len(paths) == 7
    for p in paths:
        assert p.exists(), f"{p} should exist in the real repo"
        assert p.parent.name == "benchmarks"


def test_resolve_config_paths_bare_filename():
    paths = m.resolve_config_paths(["counting.yaml"])
    assert paths[0].name == "counting.yaml"
    assert paths[0].exists()


def test_resolve_config_paths_accepts_an_absolute_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("placeholder")
    paths = m.resolve_config_paths([str(custom)])
    assert paths[0] == custom


def _fake_cfg(name, revision="rev1", precision="bfloat16"):
    return CapabilityBenchmarkConfig(
        experiment="e", model=SimpleNamespace(name=name, revision=revision, precision=precision),
        reproducibility=None, hardware=None, generation=None, dataset=None, gates=None,
    )


def test_assert_same_model_passes_for_identical_models():
    configs = [_fake_cfg("Qwen/Qwen2.5-VL-3B-Instruct"), _fake_cfg("Qwen/Qwen2.5-VL-3B-Instruct")]
    m._assert_same_model(configs)  # must not raise


def test_assert_same_model_raises_on_mismatch():
    configs = [_fake_cfg("Qwen/Qwen2.5-VL-3B-Instruct"), _fake_cfg("some-other-model")]
    with pytest.raises(m.ModelMismatchError, match="different models"):
        m._assert_same_model(configs)


def test_assert_same_model_raises_on_revision_mismatch():
    configs = [_fake_cfg("Qwen/Qwen2.5-VL-3B-Instruct", revision="rev1"), _fake_cfg("Qwen/Qwen2.5-VL-3B-Instruct", revision="rev2")]
    with pytest.raises(m.ModelMismatchError):
        m._assert_same_model(configs)


def test_assert_same_model_empty_list_does_not_raise():
    m._assert_same_model([])


def test_runtime_versions_reports_python_and_never_raises():
    # Shared with run_capability_benchmark_gate.py (single implementation, both call sites
    # record identically) -- re-exported here as `m.runtime_versions`.
    versions = m.runtime_versions()
    assert "python" in versions
    assert "torch" in versions  # "not installed" is an acceptable value, just must be present


def test_all_seven_default_configs_load_and_pin_the_same_model():
    """A real-repo consistency check: the shared-engine assumption this whole module exists
    for actually holds for the real configs, not just in a synthetic unit test.
    """
    configs = [load_capability_benchmark_config(p) for p in m.resolve_config_paths()]
    m._assert_same_model(configs)  # must not raise


def test_main_dry_run_across_two_capabilities_writes_integrity_reports(tmp_path, monkeypatch, tiny_image_factory):
    # NOTE: only gate_module.REPO_ROOT is patched (keeps subset-ID artifacts out of the real
    # repo) -- m.REPO_ROOT stays real so --configs bare filenames resolve to the actual
    # configs/benchmarks/*.yaml files this test wants to exercise.
    import neural_thickets_repro.run_capability_benchmark_gate as gate_module
    monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)

    image = tiny_image_factory()
    counting_examples = [Example(example_id=f"c{i}", image=image, prompt_input={"question": f"q{i}"}, target=i) for i in range(5)]
    cub_examples = [Example(example_id=f"b{i}", image=image, prompt_input={"question": f"q{i}"}, target="species") for i in range(5)]
    monkeypatch.setattr(TallyQACountingBenchmark, "load_examples", lambda self, cfg: counting_examples)
    monkeypatch.setattr(CUBFineGrainedBenchmark, "load_examples", lambda self, cfg: cub_examples)

    output_dir = tmp_path / "results"
    rc = m.main(["--configs", "counting.yaml,fine_grained_recognition.yaml", "--output-dir", str(output_dir), "--subset-size", "3", "--dry-run"])

    assert rc == 0
    for capability in ("counting", "fine_grained_recognition"):
        integrity_path = output_dir / capability / "integrity_report.json"
        assert integrity_path.exists()
        integrity = json.loads(integrity_path.read_text())
        assert integrity["n_loaded"] == 3
        assert integrity["passed"] is True

    # --dry-run never writes cards, so no summary is generated.
    assert not (output_dir / "summary.json").exists()
    assert (output_dir / "runtime_versions.json").exists()


def test_main_fails_clearly_and_preserves_completed_outputs_when_a_capability_crashes(tmp_path, monkeypatch, capsys):
    """Simulates a fatal exception raised mid-generation for the SECOND capability (e.g. the
    real vLLM EngineCore IndexError this repair pass fixed) -- the first capability's already
    -written output directory must remain intact, and main() must return a distinct, clearly
    -reported exit code rather than letting a bare traceback propagate with no context.
    """
    call_log = []

    def _fake_run_one_capability(cfg, **kwargs):
        call_log.append(cfg.dataset.capability)
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.dataset.capability == "counting":
            (out_dir / "metrics.json").write_text('{"primary_metric": 0.8}')
            return 0, "fake_llm", "fake_tokenizer"
        raise IndexError("list index out of range")  # simulates the real EngineCore crash

    monkeypatch.setattr(m, "run_one_capability", _fake_run_one_capability)

    output_dir = tmp_path / "results"
    rc = m.main(["--configs", "counting.yaml,fine_grained_recognition.yaml", "--output-dir", str(output_dir)])

    assert rc == m.EXIT_CODE_CAPABILITY_CRASHED
    assert call_log == ["counting", "fine_grained_recognition"]
    # counting's already-written output must remain intact.
    assert (output_dir / "counting" / "metrics.json").exists()
    assert json.loads((output_dir / "counting" / "metrics.json").read_text())["primary_metric"] == 0.8

    captured = capsys.readouterr()
    assert "FATAL" in captured.err
    assert "fine_grained_recognition" in captured.err
    assert "IndexError" in captured.err


def test_main_summary_flags_a_capability_whose_card_was_never_written(tmp_path, monkeypatch):
    """The real observed bug this repair pass fixes: a capability can report rc==0 (success)
    without ever having written a card.json (e.g. some future silent failure inside
    run_one_capability's own card-writing step) -- the FINAL summary must surface this as an
    explicit MISSING row/entry, never silently show fewer rows than capabilities attempted.
    """
    def _fake_run_one_capability(cfg, **kwargs):
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.dataset.capability == "counting":
            (out_dir / "card.json").write_text(json.dumps({
                "capability": "counting", "dataset": "tallyqa", "subset_size": 5,
                "base_metrics": {"primary_metric": 0.8, "parser_failure_rate": 0.0},
                "repeat_metrics": None, "integrity": {"n_valid_images": 5, "n_loaded": 5},
                "image_sanity": None, "status": "NEEDS_REVIEW",
            }))
        # fine_grained_recognition: reports success but never writes a card.json.
        return 0, "fake_llm", "fake_tokenizer"

    monkeypatch.setattr(m, "run_one_capability", _fake_run_one_capability)

    output_dir = tmp_path / "results"
    rc = m.main(["--configs", "counting.yaml,fine_grained_recognition.yaml", "--output-dir", str(output_dir)])

    assert rc == 0
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["missing_capabilities"] == ["fine_grained_recognition"]
    assert summary["n_capabilities"] == 1
    assert summary["n_expected_capabilities"] == 2
    assert summary["all_pass"] is False

    table = (output_dir / "summary.md").read_text()
    assert "counting" in table
    assert "fine_grained_recognition" in table
    assert "MISSING" in table


def test_main_stops_at_first_capability_that_fails_integrity(tmp_path, monkeypatch):
    import neural_thickets_repro.run_capability_benchmark_gate as gate_module
    monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)

    no_image_examples = [Example(example_id=f"c{i}", image=None, target=i) for i in range(5)]
    cub_examples_should_not_run = []

    def _fail_if_called(self, cfg):
        raise AssertionError("fine_grained_recognition should never be reached after counting fails integrity")

    monkeypatch.setattr(TallyQACountingBenchmark, "load_examples", lambda self, cfg: no_image_examples)
    monkeypatch.setattr(CUBFineGrainedBenchmark, "load_examples", _fail_if_called)

    output_dir = tmp_path / "results"
    rc = m.main(["--configs", "counting.yaml,fine_grained_recognition.yaml", "--output-dir", str(output_dir), "--subset-size", "3", "--dry-run"])

    assert rc == 1
    assert (output_dir / "counting" / "integrity_report.json").exists()
    assert not (output_dir / "fine_grained_recognition").exists()
