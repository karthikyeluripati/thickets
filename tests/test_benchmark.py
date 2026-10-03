import json
import math
import pytest
import torch
from thicket_runtime import CandidateSpec, WeightState
from thicket_runtime.benchmark import audit_trace, finite_score, run_trace, summarize
from thicket_runtime.cli import main, write_json
from thicket_runtime.workloads import SyntheticWorkload


def setup(strategy="snapshot-copy", device="cpu"):
    work = SyntheticWorkload(torch.device(device), torch.float32, width=16, depth=2, batch=2)
    state = WeightState(work.model, strategy)
    candidates = [CandidateSpec(state.base_id, i, 0.01) for i in range(3)]
    return work, state, candidates


def test_trace_warmup_repeats_and_strict_json():
    work, state, cs = setup()
    report = run_trace(state, cs, work.infer, work.score, warmup=2, repeats=2)
    assert len(report["rows"]) == 6
    assert all(r["post_trace_drift"]["exact_base"] for r in report["repeats"])
    assert not report["gpu_performance_evidence"]
    assert all(row["apply"]["cuda_interval_ms"] is None for row in report["rows"])
    assert 0 <= report["summary"]["state_wall_fraction"] <= 1
    for row in report["rows"]:
        assert row["total_wall_ms"] >= sum(row[k]["wall_ms"] for k in ["apply", "inference", "restore", "score"])
    json.dumps(report, allow_nan=False)
    assert state.drift()["exact_base"]


def test_snapshot_correctness_audit():
    work, state, cs = setup()
    audit = audit_trace(state, cs, work.infer, work.score)
    assert audit["all_outputs_exact"] and audit["all_resets_exact"] and audit["all_rewards_exact"]


def test_legacy_drift_is_reported_not_hidden():
    work, state, cs = setup("legacy-add-subtract")
    audit = audit_trace(state, cs, work.infer, work.score)
    assert not audit["all_resets_exact"]
    assert state.strategy == "legacy-add-subtract"
    assert state.drift()["exact_base"]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_score_rejected(bad):
    with pytest.raises(ValueError):
        finite_score(bad)


@pytest.mark.parametrize("failure_stage", ["inference", "score"])
def test_failure_recovers_base(failure_stage):
    work, state, cs = setup("legacy-add-subtract")
    def fail(*args):
        raise RuntimeError("deliberate failure")
    with pytest.raises(RuntimeError, match="deliberate"):
        run_trace(state, cs, fail if failure_stage == "inference" else work.infer,
                  fail if failure_stage == "score" else work.score, repeats=1, warmup=0)
    assert state.drift()["exact_base"]


def test_empty_summary_refused():
    with pytest.raises(ValueError):
        summarize([])


def test_cli_artifacts_and_no_overwrite(tmp_path):
    out = tmp_path / "run"
    args = ["--out", str(out), "--candidates", "2", "--repeats", "1", "--warmup", "0",
            "--width", "8", "--depth", "1"]
    assert main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["snapshot-copy"]["exact_semantics_gate"]
    with pytest.raises(FileExistsError):
        main(args)


def test_strict_json_writer_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "out.json", {"bad": float("nan")})
    assert not (tmp_path / "out.json").exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only guard test")
def test_no_silent_cuda_fallback(tmp_path):
    with pytest.raises(SystemExit):
        main(["--device", "cuda", "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_events_smoke():
    work, state, cs = setup(device="cuda:0")
    report = run_trace(state, cs, work.infer, work.score, warmup=1, repeats=1)
    assert report["gpu_performance_evidence"]
    assert all(r["apply"]["cuda_interval_ms"] >= 0 for r in report["rows"])
    assert report["repeats"][0]["memory"]["peak_allocated_bytes"] > 0
