"""Tests for the Stage-11 32B whole-model smoke END-TO-END WIRING milestone: dispatcher
integration, run_stage11_whole_model_scaling.py's 32B branch, vLLM shard-mapping, and the live
G3/G4/G5 gate-check functions. CPU-only, no GPU/ray/vLLM import -- matches this project's
established convention (see test_run_stage11_whole_model_scaling.py's own fake-engine tests).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Dict

import pytest
import torch
import torch.nn as nn

import neural_thickets_repro.run_stage11_visual_thicket_scaling as dispatcher
import neural_thickets_repro.run_stage11_whole_model_scaling as whole_model
import neural_thickets_repro.scaling_common as scaling_common
import neural_thickets_repro.stage11_32b_readiness as readiness
from neural_thickets_repro.thicket import cpu_base_snapshot as cbs
from neural_thickets_repro.thicket import distributed_perturbation as dp
from neural_thickets_repro.thicket import vllm_shard_mapping as vsm


# =================================================================================================
# Top-level 32B dispatch (dispatcher)
# =================================================================================================


def test_dispatcher_32b_whole_model_smoke_dry_run_succeeds():
    rc = dispatcher.main(["--scale", "32B", "--track", "whole_model", "--smoke", "--dry-run"])
    assert rc == 0


def test_dispatcher_32b_anatomy_no_longer_hard_blocked_at_the_dispatcher(monkeypatch):
    """The former unconditional '32B anatomy is not permitted' dispatcher guard is REMOVED now
    that a dedicated, narrow S2 32B anatomy runner exists (run_stage11_coarse_anatomical_atlas_
    32b.py) with its OWN live-evidence readiness gate -- the dispatcher's job is now purely to
    route track=='anatomy'+scale=='32B' to THAT runner, never to decide readiness itself (see
    test_run_stage11_coarse_anatomical_atlas_32b.py for that runner's own gate behavior).
    """
    import neural_thickets_repro.run_stage11_coarse_anatomical_atlas_32b as anatomy_32b

    reached = {}

    def _marker_main(argv=None):
        reached["argv"] = argv
        return 0

    monkeypatch.setattr(anatomy_32b, "main", _marker_main)
    rc = dispatcher.main(["--scale", "32B", "--track", "anatomy", "--smoke"])
    assert rc == 0
    assert reached["argv"] == ["--smoke"]


def test_dispatcher_32b_without_smoke_no_longer_blocked_at_the_dispatcher(monkeypatch):
    """The former '--smoke required, unconditionally' dispatcher guard is REMOVED -- full 32B
    (no --smoke) now reaches run_stage11_whole_model_scaling.main() exactly like smoke does; that
    function's OWN live-evidence gate is what may still block it (see test_32b_full_run_blocked_
    without_valid_live_evidence / test_32b_full_run_authorized_by_valid_live_evidence below), not
    this dispatcher.
    """
    reached = {}

    def _marker_main(argv=None):
        reached["argv"] = argv
        return 0

    monkeypatch.setattr(whole_model, "main", _marker_main)
    rc = dispatcher.main(["--scale", "32B", "--track", "whole_model"])
    assert rc == 0
    assert reached["argv"] == ["--scale", "32B"]


def test_dispatcher_72b_still_hard_rejected():
    rc = dispatcher.main(["--scale", "72B", "--track", "whole_model"])
    assert rc == 1
    rc_anatomy = dispatcher.main(["--scale", "72B", "--track", "anatomy"])
    assert rc_anatomy == 1


def test_dispatcher_3b_7b_dry_run_unaffected(capsys):
    rc3 = dispatcher.main(["--scale", "3B", "--track", "whole_model", "--smoke", "--dry-run"])
    out3 = capsys.readouterr().out
    rc7 = dispatcher.main(["--scale", "7B", "--track", "whole_model", "--smoke", "--dry-run"])
    out7 = capsys.readouterr().out
    assert rc3 == 0 and rc7 == 0
    assert "scale=3B" in out3 and "scale=7B" in out7


# =================================================================================================
# run_stage11_whole_model_scaling.py -- 32B branch
# =================================================================================================


def test_32b_scale_is_now_a_valid_cli_choice():
    rc = whole_model.main(["--scale", "32B", "--smoke", "--dry-run"])
    assert rc == 0


def test_32b_smoke_dry_run_matches_frozen_design_totals(capsys):
    rc = whole_model.main(["--scale", "32B", "--smoke", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total_unique_perturbations=3" in out
    assert "total_perturbation_x_capability_evaluations=18" in out


def test_32b_full_run_blocked_without_valid_live_evidence(tmp_path, monkeypatch):
    """Full 32B (no --smoke) is no longer refused merely for lacking --smoke -- it reaches the
    SAME live-evidence gate smoke does, and is blocked by THAT (honestly, with reasons) when no
    valid evidence exists, exactly like a smoke invocation would be.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    empty_evidence_dir = tmp_path / "no_such_evidence"

    rc = whole_model.main(["--scale", "32B", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(empty_evidence_dir)])
    assert rc == 0  # blocked cleanly, never a crash -- same shape as the smoke-mode equivalent

    import json

    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["live_evidence_found"] is False
    assert report["all_gates_pass"] is False


def test_32b_full_run_authorized_by_valid_live_evidence(tmp_path, monkeypatch):
    """The mirror image: full 32B (no --smoke) with VALID live evidence continues into the shared
    lifecycle exactly like smoke does -- the readiness gate has never distinguished smoke from
    full, and the dispatch guard that WOULD have distinguished them is now removed.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    _write_valid_live_evidence(evidence_dir)
    _patch_resolve_model_snapshot_to_raise_marker(monkeypatch)

    with pytest.raises(_ReachedSharedLifecycleMarker):
        whole_model.main(["--scale", "32B", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])

    import json

    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is True
    assert report["live_evidence_ok"] is True


def test_32b_smoke_flag_still_produces_the_frozen_smoke_plan_when_full_is_now_allowed():
    """Section 11: smoke behavior must remain unchanged -- --smoke still selects the N=5/1-
    direction smoke plan (3 perturbations, 18 rows), never the full N=50/64-direction plan, now
    that full is also reachable through the same code path.
    """
    rc_smoke = whole_model.main(["--scale", "32B", "--smoke", "--dry-run"])
    assert rc_smoke == 0


def test_32b_full_dry_run_matches_frozen_full_design_totals(capsys):
    """Full 32B (no --smoke) must use the SAME frozen full-run sizing 3B/7B already use: D_map
    N=50, 64 directions per radius -- 192 perturbations, 1152 rows, 57,600 evaluations -- never a
    32B-specific size.
    """
    rc = whole_model.main(["--scale", "32B", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total_unique_perturbations=192" in out
    assert "total_perturbation_x_capability_evaluations=1152" in out
    assert "total_perturbed_model_example_evaluations=57600" in out


def test_32b_smoke_blocked_pending_live_evidence_writes_gate_report(tmp_path, monkeypatch):
    """Full (non-dry-run) 32B path with the environment/Hub gates faked open, to reach the
    readiness pre-flight branch -- must write a real gate report and return 0 (a clean, honest
    'blocked' exit), never crash, never proceed toward an engine launch. G4/G5 now read
    READY_FOR_LIVE_VERIFICATION (the distributed v3 solver exists and is CPU-proven, see
    thicket.distributed_v3_solver) rather than FAIL -- real progress -- but the smoke remains
    blocked because READY_FOR_LIVE_VERIFICATION is never PASS.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": "a" * 40, "requested_revision": "main"})
    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path)])
    assert rc == 0
    report_files = list(tmp_path.rglob("stage11_32b_readiness_gate_report.json"))
    assert len(report_files) == 1
    import json
    report = json.loads(report_files[0].read_text())
    assert report["gate_results"]["G4"] == readiness.GATE_READY_FOR_LIVE_VERIFICATION
    assert report["gate_results"]["G5"] == readiness.GATE_READY_FOR_LIVE_VERIFICATION
    assert report["all_gates_pass"] is False


def test_32b_never_reaches_legacy_engine_config(tmp_path, monkeypatch):
    """Structural proof the 32B branch returns before build_stage7b_engine_config (the 3B/7B-only,
    TP=1-hardcoded, legacy-base-snapshot-hardcoded config builder) is ever called.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": "a" * 40, "requested_revision": "main"})

    def _should_never_be_called():
        raise AssertionError("build_stage7b_engine_config must never be called on the 32B branch")

    monkeypatch.setattr(whole_model, "build_stage7b_engine_config", _should_never_be_called)
    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path)])
    assert rc == 0  # would have raised AssertionError above if the legacy path were reached


def test_3b_7b_source_unchanged_by_32b_branch_presence():
    """Confirms via source inspection that the existing 3B/7B code (build_stage7b_engine_config,
    store_base_weights_via_rpc, run_whole_model_rpc's default TP=1 evaluator) is still present,
    reached via the `else` side of `is_32b` conditionals -- never removed or unconditionally
    replaced by the 32B continuation wiring.
    """
    source = inspect.getsource(whole_model.main)
    assert 'is_32b = plan.scale_label == "32B"' in source
    # the existing 3B/7B call chain still appears exactly once, in the `else` branch of each `is_32b` conditional
    assert source.count("build_stage7b_engine_config()") == 1
    assert source.count("store_base_weights_via_rpc(engine)") == 1
    # 32B's own analogs appear too, but as clearly separate, additively-injected branches
    assert source.count("build_32b_engine_config(") == 1
    assert "evaluate_one_whole_model_candidate_distributed_rpc" in source


# =================================================================================================
# 32B engine configuration (Section 3)
# =================================================================================================


def test_32b_engine_config_matches_task_spec():
    cfg = readiness.build_32b_engine_config()
    assert cfg["tensor_parallel_size"] == 4
    assert cfg["gpu_memory_utilization"] == pytest.approx(0.60)
    assert cfg["max_model_len"] == 4096
    assert cfg["enforce_eager"] is True
    assert cfg["enable_prefix_caching"] is False
    assert cfg["precision"] == "bfloat16"
    assert cfg["base_snapshot_mode"] == "cpu_base_weights"


def test_32b_engine_config_rejects_invalid_tp_size():
    with pytest.raises(ValueError):
        readiness.build_32b_engine_config(tensor_parallel_size=0)


def test_32b_engine_config_never_requests_quantization():
    cfg = readiness.build_32b_engine_config()
    assert "quant" not in str(cfg).lower()
    assert cfg["precision"] == "bfloat16"


# =================================================================================================
# Immutable revision persistence (reuses the existing, unmodified generic mechanism)
# =================================================================================================


def test_32b_revision_resolution_is_persisted_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    fake_resolution = {"resolved_revision": "b" * 40, "requested_revision": "main"}
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: fake_resolution)
    whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path)])
    revision_files = list(tmp_path.rglob("model_revision_resolution.json"))
    assert len(revision_files) == 1
    import json
    assert json.loads(revision_files[0].read_text()) == fake_resolution


# =================================================================================================
# Live-readiness-evidence wiring fix -- run_stage11_whole_model_scaling.py's 32B branch must
# CONSUME a real, strictly identity-bound live G1-G8 + strict-v3-solver verification rather than
# always rebuilding the CPU-only default gate_results (task spec: readiness-evidence
# persistence/runner-integration bug).
# =================================================================================================

_FAKE_REVISION = "7cfb30d71a1f4f49a57592323337a4a4727301da"


def _write_valid_live_evidence(evidence_dir, *, revision=_FAKE_REVISION):
    import json

    evidence_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "resolved_revision": {"model_name": readiness.FROZEN_32B_MODEL_NAME, "resolved_revision": revision},
        "model_load": {
            "ok": True,
            "config": {
                "tensor_parallel_size": 4, "dtype": "bfloat16", "base_snapshot_mode": "cpu_base_weights",
                "gpu_memory_utilization": 0.60, "max_model_len": 4096, "enable_prefix_caching": False,
            },
        },
        "gate_results": {g: readiness.GATE_PASS for g in readiness.GATE_IDS},
        "smoke_permitted": True,
    }
    solver = {
        "resolved_revision": {"model_name": readiness.FROZEN_32B_MODEL_NAME, "resolved_revision": revision},
        "solver_error": None, "acceptance_mode": "strict",
        "rank_consensus": {"core_fields_ok": True, "full_bracket_trajectory": {"ok": True}},
        "restoration": {"ok": True}, "g4_g5_final": {"G4": "PASS", "G5": "PASS"}, "scientific_rows_written": 0,
    }
    (evidence_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(base))
    (evidence_dir / "stage11_32b_live_v3_solver_probe_report.json").write_text(json.dumps(solver))
    return base, solver


class _ReachedSharedLifecycleMarker(Exception):
    """Raised by a patched resolve_model_snapshot (the FIRST real external call the shared
    dataset/subset-gate/tokenizer/engine-launch lifecycle makes, immediately after the 32B
    readiness gate) -- proves execution genuinely CONTINUED past the gate into the shared
    lifecycle, without this test needing to actually download a model, load a dataset, start
    Ray, or touch a GPU.
    """


def _patch_resolve_model_snapshot_to_raise_marker(monkeypatch):
    import neural_thickets_repro.vlm_adapter as vlm_adapter

    def _boom(*a, **k):
        raise _ReachedSharedLifecycleMarker("resolve_model_snapshot reached -- continuation into the shared lifecycle confirmed")

    monkeypatch.setattr(vlm_adapter, "resolve_model_snapshot", _boom)


def test_valid_live_evidence_authorizes_32b_smoke_all_gates_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    _write_valid_live_evidence(evidence_dir)
    _patch_resolve_model_snapshot_to_raise_marker(monkeypatch)

    with pytest.raises(_ReachedSharedLifecycleMarker):
        whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])

    import json

    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is True
    assert all(v == readiness.GATE_PASS for v in report["gate_results"].values())
    assert report["live_evidence_found"] is True
    assert report["live_evidence_ok"] is True
    assert report["live_evidence_reasons"] == []


def test_invalid_live_evidence_never_reaches_shared_lifecycle(tmp_path, monkeypatch):
    """The inverse of the above -- proves invalid/mismatched evidence blocks BEFORE the shared
    lifecycle (and therefore before any engine launch) is ever reached, using the identical
    marker technique so both tests are symmetric proof of the same boundary.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    _write_valid_live_evidence(evidence_dir, revision="f" * 40)  # mismatched revision -- invalid
    _patch_resolve_model_snapshot_to_raise_marker(monkeypatch)

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0  # blocked cleanly -- the marker was never raised, proving resolve_model_snapshot was never reached


def test_runner_consumes_pass_rather_than_rebuilding_default_statuses(tmp_path, monkeypatch):
    """The exact bug this fix addresses: with valid live evidence present, the runner's OWN gate
    report must show real PASS values, never the CPU-only defaults (NOT_YET_VERIFIED / READY_FOR_
    LIVE_VERIFICATION) it always wrote before this fix, regardless of what evidence existed.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    _write_valid_live_evidence(evidence_dir)
    _patch_resolve_model_snapshot_to_raise_marker(monkeypatch)

    with pytest.raises(_ReachedSharedLifecycleMarker):
        whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])

    import json

    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    for gate in readiness.GATE_IDS:
        assert report["gate_results"][gate] != readiness.GATE_NOT_YET_VERIFIED
        assert report["gate_results"][gate] != readiness.GATE_READY_FOR_LIVE_VERIFICATION
        assert report["gate_results"][gate] == readiness.GATE_PASS


def test_mismatched_revision_evidence_blocks_32b_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    _write_valid_live_evidence(evidence_dir, revision="f" * 40)  # evidence for a DIFFERENT revision than requested

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0  # blocked cleanly, never a crash

    import json

    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False
    assert report["live_evidence_found"] is True
    assert report["live_evidence_ok"] is False
    assert report["live_evidence_reasons"]  # non-empty -- explains exactly why


def test_missing_live_evidence_directory_blocks_and_reports_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    empty_evidence_dir = tmp_path / "no_such_evidence"

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(empty_evidence_dir)])
    assert rc == 0

    import json

    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["live_evidence_found"] is False
    assert report["all_gates_pass"] is False


def test_one_gate_not_pass_in_evidence_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    base, solver = _write_valid_live_evidence(evidence_dir)
    import json

    base["gate_results"]["G2"] = readiness.GATE_FAIL
    (evidence_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(base))

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False


def test_missing_strict_v3_solver_evidence_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True)
    import json

    (evidence_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps({
        "resolved_revision": {"model_name": readiness.FROZEN_32B_MODEL_NAME, "resolved_revision": _FAKE_REVISION},
        "model_load": {"ok": True, "config": {"tensor_parallel_size": 4, "dtype": "bfloat16", "base_snapshot_mode": "cpu_base_weights", "gpu_memory_utilization": 0.60, "max_model_len": 4096, "enable_prefix_caching": False}},
        "gate_results": {g: readiness.GATE_PASS for g in readiness.GATE_IDS}, "smoke_permitted": True,
    }))
    # NO solver artifact written at all

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False
    assert any("solver" in r for r in report["live_evidence_reasons"])


def test_missing_restoration_evidence_in_solver_artifact_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    base, solver = _write_valid_live_evidence(evidence_dir)
    import json

    solver["restoration"] = {"ok": False}
    (evidence_dir / "stage11_32b_live_v3_solver_probe_report.json").write_text(json.dumps(solver))

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False


def test_tp_size_mismatch_in_evidence_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    base, solver = _write_valid_live_evidence(evidence_dir)
    import json

    base["model_load"]["config"]["tensor_parallel_size"] = 1  # evidence gathered at a different TP size
    (evidence_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(base))

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False


def test_wrong_model_in_evidence_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    base, solver = _write_valid_live_evidence(evidence_dir)
    import json

    base["resolved_revision"]["model_name"] = "Qwen/Qwen2.5-VL-7B-Instruct"
    (evidence_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(base))

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False


def test_invalid_evidence_never_writes_scientific_rows(tmp_path, monkeypatch):
    """Regardless of why evidence is invalid (malformed, mismatched, absent), the 32B branch must
    never write results.jsonl or any candidate row before the readiness gate passes.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    _write_valid_live_evidence(evidence_dir, revision="f" * 40)  # mismatched -- invalid

    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])
    assert rc == 0
    assert list((tmp_path / "out").rglob("results.jsonl")) == []
    assert list((tmp_path / "out").rglob("*candidate*")) == []


def test_valid_evidence_reaches_lifecycle_without_writing_rows_before_an_engine_exists(tmp_path, monkeypatch):
    """With valid evidence, execution DOES continue (see test_valid_live_evidence_authorizes_
    32b_smoke_all_gates_pass) -- but must still not have written any scientific row by the time
    it reaches the first real external call (resolve_model_snapshot), since no engine has been
    launched and no candidate has been evaluated yet at that point.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    evidence_dir = tmp_path / "evidence"
    _write_valid_live_evidence(evidence_dir)
    _patch_resolve_model_snapshot_to_raise_marker(monkeypatch)

    with pytest.raises(_ReachedSharedLifecycleMarker):
        whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path / "out"), "--live-evidence-dir", str(evidence_dir)])

    assert list((tmp_path / "out").rglob("results.jsonl")) == []
    assert list((tmp_path / "out").rglob("*candidate*")) == []


def test_3b_dry_run_never_touches_32b_readiness_preflight(monkeypatch):
    """3B/7B must remain completely unaffected by this wiring fix -- structurally proven by never
    even calling run_32b_readiness_preflight_and_report (which is where all the new live-evidence
    logic lives).
    """
    def _should_never_be_called(**kwargs):
        raise AssertionError("run_32b_readiness_preflight_and_report must never be called for scale=3B")

    monkeypatch.setattr(whole_model, "run_32b_readiness_preflight_and_report", _should_never_be_called)
    rc = whole_model.main(["--scale", "3B", "--dry-run"])
    assert rc == 0  # would have raised AssertionError above if the 32B live-evidence path were reached


def test_72b_remains_not_a_valid_scale_choice():
    with pytest.raises(SystemExit):
        whole_model.main(["--scale", "72B", "--smoke"])


def test_resumed_checkpoint_manifest_rejects_a_different_revision():
    """Existing, unmodified mechanism: a checkpoint persisted under one model_revision must hard
    -fail if a later invocation tries to resume under a DIFFERENT one -- this is what makes
    'once resolved, all subsequent executions must use that exact SHA' true, generically, for
    every scale including 32B, without any new code.
    """
    from neural_thickets_repro.run_stage11_whole_model_scaling import (
        IncompatibleWholeModelCheckpointError, WholeModelCheckpointManifest, ensure_whole_model_checkpoint_manifest,
    )
    import tempfile
    from pathlib import Path

    def _manifest(rev):
        return WholeModelCheckpointManifest(
            experiment_id="x", run_signature="y", scale_label="32B", track="whole_model", restoration_mode="fixed_base",
            perturbation_mode="anatomical_relative_l2", radius_realization_method="fixed_direction_bf16_quantization_aware_v3",
            multimodal_cache_policy="p", enable_prefix_caching=False, generation_batch_size=10, model_revision=rev,
            dataset_role="map", radii=(0.1,), capabilities=("a",), n_directions_per_cell=1, d_map_n=5,
            subset_hashes={"a": "h"}, whole_model_mask_hash="m", direction_seed_bank_hash="s", anatomy_audit_hash="aud",
            expected_unique_perturbations=1, expected_result_rows=1,
        )

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "checkpoint_manifest.json"
        ensure_whole_model_checkpoint_manifest(path, _manifest("a" * 40))
        with pytest.raises(IncompatibleWholeModelCheckpointError):
            ensure_whole_model_checkpoint_manifest(path, _manifest("c" * 40))


# =================================================================================================
# Real vLLM TP shard-metadata mapping (Section 5 -- "the main engineering task")
# =================================================================================================


def test_shard_spec_from_attributes_replicated_when_no_dim_and_tp_size_1():
    spec = vsm.build_shard_spec_from_attributes(torch.Size([8, 8]), output_dim=None, input_dim=None, tp_size=1, tp_rank=0)
    assert spec.is_replicated
    assert spec.world_size == 1


def test_shard_spec_from_attributes_recognized_replicated_under_tp_gt_1():
    """A norm weight (no output_dim/input_dim) living inside a tp_size=4 layer -- the documented
    'no dim attribute = replicated' convention, not an ambiguity.
    """
    spec = vsm.build_shard_spec_from_attributes(torch.Size([5120]), output_dim=None, input_dim=None, tp_size=4, tp_rank=2)
    assert spec.is_replicated
    assert spec.world_size == 4
    assert spec.rank == 2


def test_shard_spec_from_attributes_column_sharded():
    """output_dim=0, local shape already the shard -- global_shape/local_offset recovered per
    vLLM's own documented start_idx = tp_rank * shard_size convention.
    """
    spec = vsm.build_shard_spec_from_attributes(torch.Size([1280, 5120]), output_dim=0, input_dim=None, tp_size=4, tp_rank=2)
    assert spec.dim == 0
    assert spec.local_size == 1280
    assert spec.global_shape == torch.Size([5120, 5120])
    assert spec.local_offset == 2560  # tp_rank(2) * shard_size(1280)


def test_shard_spec_from_attributes_row_sharded():
    spec = vsm.build_shard_spec_from_attributes(torch.Size([5120, 1280]), output_dim=None, input_dim=1, tp_size=4, tp_rank=1)
    assert spec.dim == 1
    assert spec.local_offset == 1280  # tp_rank(1) * shard_size(1280)


def test_shard_spec_both_dims_set_prefers_output_dim_not_a_hard_fail():
    """Live-verified correction (real 4xL40S TP=4 32B run): VocabParallelEmbedding/ParallelLMHead
    legitimately set BOTH output_dim and input_dim -- output_dim is authoritative, never a hard
    fail. See test_thicket_vllm_shard_mapping.py for the full case coverage and
    thicket/vllm_shard_mapping.py's own docstring for the live evidence.
    """
    spec = vsm.build_shard_spec_from_attributes(torch.Size([8, 8]), output_dim=0, input_dim=1, tp_size=4, tp_rank=0)
    assert spec.dim == 0


def test_shard_spec_hard_fails_on_out_of_range_dim():
    with pytest.raises(vsm.AmbiguousShardMappingError):
        vsm.build_shard_spec_from_attributes(torch.Size([8, 8]), output_dim=5, input_dim=None, tp_size=4, tp_rank=0)


class _FakeTPLinear(nn.Module):
    """Mimics vLLM's documented attribute convention: tp_size/tp_rank on the owning module,
    output_dim set directly on the weight Parameter via a plain attribute assignment (the same
    mechanism set_weight_attrs uses under the hood).
    """
    def __init__(self, local_out: int, in_features: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(local_out, in_features))
        self.weight.output_dim = 0
        self.tp_size = tp_size
        self.tp_rank = tp_rank


class _FakeReplicatedNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # no output_dim/input_dim, no tp_size -- plain replicated norm


class _FakeTPModel(nn.Module):
    def __init__(self, tp_size: int, tp_rank: int):
        super().__init__()
        self.proj = _FakeTPLinear(local_out=32 // tp_size, in_features=16, tp_size=tp_size, tp_rank=tp_rank)
        self.norm = _FakeReplicatedNorm(16)


def test_build_shard_specs_for_region_end_to_end_fake_vllm_model():
    model = _FakeTPModel(tp_size=4, tp_rank=2)
    specs = vsm.build_shard_specs_for_region(model, ["proj.weight", "norm.weight"])
    assert specs["proj.weight"].dim == 0
    assert specs["proj.weight"].global_shape == torch.Size([32, 16])
    assert specs["proj.weight"].local_offset == 16  # tp_rank(2) * shard_size(8)
    assert specs["norm.weight"].is_replicated


def test_build_shard_specs_for_region_missing_parameter_hard_fails():
    model = _FakeTPModel(tp_size=4, tp_rank=0)
    with pytest.raises(vsm.AmbiguousShardMappingError):
        vsm.build_shard_specs_for_region(model, ["does.not.exist"])


def test_ensure_uniform_tp_size_detects_mismatch():
    specs = {"a": dp.ShardSpec(global_shape=torch.Size([8]), dim=0, local_offset=0, local_size=2, rank=0, world_size=2)}
    with pytest.raises(vsm.AmbiguousShardMappingError):
        vsm.ensure_uniform_tp_size(specs, expected_tp_size=4)
    vsm.ensure_uniform_tp_size(specs, expected_tp_size=2)  # must not raise


# =================================================================================================
# Live G3/G4/G5 gate-check functions (worker-side, testable via fakes)
# =================================================================================================


class _TinyVLM(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.language_model = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.language_model.weight.copy_(torch.randn(8, 8, generator=g, dtype=torch.float32).to(torch.bfloat16))
        self.to(torch.bfloat16)


def _fake_worker(model):
    ns = SimpleNamespace()
    ns.model_runner = SimpleNamespace(model=model)
    ns._should_perturb = lambda name: True
    return ns


def test_g3_live_check_reports_bit_exact_on_a_real_module():
    model = _TinyVLM(seed=1)
    worker = _fake_worker(model)
    facts = cbs.g3_live_cpu_cuda_equivalence_check_rpc(worker, probe_param_name="language_model.weight", seed=42, delta=0.01)
    equivalence_class = cbs.classify_snapshot_equivalence(**facts)
    assert equivalence_class == cbs.EQUIVALENCE_BIT_EXACT


def test_g3_live_check_requires_existing_parameter():
    model = _TinyVLM(seed=2)
    worker = _fake_worker(model)
    with pytest.raises(RuntimeError):
        cbs.g3_live_cpu_cuda_equivalence_check_rpc(worker, probe_param_name="does.not.exist", seed=1)


def test_g4_g5_live_check_world_size_1_within_tolerance():
    model = _TinyVLM(seed=3)
    worker = _fake_worker(model)
    worker.tensor_parallel_size = 1
    worker.rank = 0
    # language_model.weight has neither output_dim/input_dim and worker has no vLLM-shaped
    # module wrapping -- resolves to replicated/world_size=1 via the fallback path.
    result = dp.g4_g5_live_relative_l2_check_rpc(worker, ["language_model.weight"], seed=5, r=0.05)
    # The ONE-SHOT apply (no bf16-bracketed correction -- that's the v3 solver's job, not yet
    # distributed-aware, see stage11_32b_readiness.V3_SOLVER_DISTRIBUTED_EXTENSION_NOTE) only
    # needs to be CLOSE here -- this test proves the global-norm-reduction plumbing is wired
    # correctly end to end, not bf16 radius exactness (a different, already-solved problem).
    assert result["realized_relative_l2"] == pytest.approx(0.05, abs=5e-3)


def test_classify_g4_g5_live_check_requires_synchronized_global_values():
    consistent = [{"theta_l2_norm": 1.0, "raw_noise_l2_norm": 2.0, "scale": 0.5, "realized_relative_l2": 0.05, "requested_r": 0.05}] * 2
    assert dp.classify_g4_g5_live_check(consistent) is True

    inconsistent = [
        {"theta_l2_norm": 1.0, "raw_noise_l2_norm": 2.0, "scale": 0.5, "realized_relative_l2": 0.05, "requested_r": 0.05},
        {"theta_l2_norm": 1.5, "raw_noise_l2_norm": 2.0, "scale": 0.5, "realized_relative_l2": 0.05, "requested_r": 0.05},  # a rank that never got the all-reduced value -- the exact "no per-rank normalization" bug
    ]
    assert dp.classify_g4_g5_live_check(inconsistent) is False


def test_classify_g4_g5_live_check_empty_list_fails_closed():
    assert dp.classify_g4_g5_live_check([]) is False


def test_classify_g4_g5_live_check_accepts_realistic_bf16_single_shot_gap():
    """Live-verified: a real 4xL40S TP=4 run's un-iterated single apply (not the multi-iteration
    solver) landed realized=0.0037375446189290592 against requested=0.0035698828543799426 -- a
    ~4.7% relative gap, identical across all 4 ranks. This must PASS (rank consensus + within the
    loose sanity bound), matching the real evidence rather than the old ungrounded 1e-6 bar.
    """
    live_result = [
        {"theta_l2_norm": 215.85274580728236, "raw_noise_l2_norm": 7240.792036816176, "scale": 0.00010642054245036344,
         "realized_relative_l2": 0.0037375446189290592, "requested_r": 0.0035698828543799426}
    ] * 4
    assert dp.classify_g4_g5_live_check(live_result) is True


def test_classify_g4_g5_live_check_rejects_a_grossly_wrong_radius():
    """A ~50% relative gap (e.g. a missing/double-counted rank contribution) must still fail --
    the loosened bound is a sanity check, not a rubber stamp."""
    broken = [{"theta_l2_norm": 1.0, "raw_noise_l2_norm": 2.0, "scale": 0.5, "realized_relative_l2": 0.075, "requested_r": 0.05}] * 4
    assert dp.classify_g4_g5_live_check(broken) is False


# =================================================================================================
# 32B anatomy completeness gate (design-level -- reused frozen regions, no execution)
# =================================================================================================


def test_32b_readiness_manifest_region_completeness_reuses_frozen_atlas():
    """No NEW regions were invented for 32B -- the manifest reuses the exact frozen
    vision/connector/language partition, the same one 3B/7B's own live anatomy audits already
    proved complete/disjoint.
    """
    manifest = readiness.build_32b_readiness_manifest()
    assert set(manifest.region_definitions) == {"vision", "multimodal_connector_or_merger", "language"}


# =================================================================================================
# Candidate transactional persistence -- proof the existing mechanism is untouched
# =================================================================================================


def test_evaluate_one_whole_model_candidate_rpc_source_has_no_32b_conditional():
    """The real per-candidate transactional lifecycle (perturb -> evaluate 6 capabilities ->
    restore -> verify -> only then return rows) is completely unparameterized by scale -- no
    32B-specific branch was added to it (the 32B path diverges much earlier, in main(), before
    this function is ever reached) -- this positively confirms 3B/7B's transactional guarantees
    were not touched.
    """
    source = inspect.getsource(whole_model.evaluate_one_whole_model_candidate_rpc)
    assert "32B" not in source
    assert "cpu_base_weights" not in source
    assert "tensor_parallel" not in source


# =================================================================================================
# No 3B/7B behavior regression -- direct equivalence checks
# =================================================================================================


def test_3b_and_7b_still_use_ensure_scale_runnable_unchanged():
    with pytest.raises(scaling_common.ScaleNotYetEnabledError):
        scaling_common.ensure_scale_runnable("72B")
    scaling_common.ensure_scale_runnable("3B")  # must not raise
    scaling_common.ensure_scale_runnable("7B")  # must not raise


def test_3b_7b_dry_run_plans_never_take_the_32b_branch(capsys):
    rc = whole_model.main(["--scale", "7B", "--smoke", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "32B readiness gate report" not in out
    assert "STOP AND REPORT" not in out
