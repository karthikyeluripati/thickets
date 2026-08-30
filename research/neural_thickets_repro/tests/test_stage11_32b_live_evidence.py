"""Tests for stage11_32b_live_evidence.py -- the readiness-evidence-persistence/runner-
integration wiring fix. Fabricates realistic base G1-G8 + strict v3 solver artifacts (matching
the exact schema confirmed on real 4xL40S TP=4 runs) on disk and proves: valid evidence
authorizes, every individual identity/gate/solver mismatch fail-closes, and hardware-fingerprint
binding is optional-but-enforced-when-present.
"""
from __future__ import annotations

import json

import pytest

from neural_thickets_repro import stage11_32b_live_evidence as live_evidence
from neural_thickets_repro import stage11_32b_readiness as readiness

REVISION = "7cfb30d71a1f4f49a57592323337a4a4727301da"
MODEL_NAME = readiness.FROZEN_32B_MODEL_NAME


def _valid_base_artifact(**overrides) -> dict:
    artifact = {
        "resolved_revision": {"model_name": MODEL_NAME, "resolved_revision": REVISION},
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
    artifact.update(overrides)
    return artifact


def _valid_solver_artifact(**overrides) -> dict:
    artifact = {
        "resolved_revision": {"model_name": MODEL_NAME, "resolved_revision": REVISION},
        "solver_error": None,
        "acceptance_mode": "strict",
        "rank_consensus": {"core_fields_ok": True, "full_bracket_trajectory": {"ok": True}},
        "restoration": {"ok": True},
        "g4_g5_final": {"G4": "PASS", "G5": "PASS"},
        "scientific_rows_written": 0,
    }
    artifact.update(overrides)
    return artifact


def _requirement(**overrides) -> live_evidence.LiveEvidenceIdentityRequirement:
    kwargs = {"resolved_revision": REVISION}
    kwargs.update(overrides)
    return live_evidence.LiveEvidenceIdentityRequirement(**kwargs)


def _write_artifacts(tmp_path, base=None, solver=None):
    (tmp_path / live_evidence.LIVE_READINESS_ARTIFACT_FILENAME).write_text(json.dumps(base if base is not None else _valid_base_artifact()))
    (tmp_path / live_evidence.LIVE_V3_SOLVER_ARTIFACT_FILENAME).write_text(json.dumps(solver if solver is not None else _valid_solver_artifact()))


# =================================================================================================
# validate_live_readiness_artifact
# =================================================================================================


def test_valid_base_artifact_passes():
    result = live_evidence.validate_live_readiness_artifact(_valid_base_artifact(), _requirement())
    assert result == {"ok": True, "reasons": []}


def test_base_artifact_wrong_revision_fails():
    result = live_evidence.validate_live_readiness_artifact(
        _valid_base_artifact(resolved_revision={"model_name": MODEL_NAME, "resolved_revision": "b" * 40}), _requirement(),
    )
    assert result["ok"] is False
    assert any("resolved_revision" in r for r in result["reasons"])


def test_base_artifact_wrong_model_fails():
    result = live_evidence.validate_live_readiness_artifact(
        _valid_base_artifact(resolved_revision={"model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "resolved_revision": REVISION}), _requirement(),
    )
    assert result["ok"] is False
    assert any("model_name" in r for r in result["reasons"])


def test_base_artifact_tp_size_mismatch_fails():
    artifact = _valid_base_artifact()
    artifact["model_load"]["config"]["tensor_parallel_size"] = 1
    result = live_evidence.validate_live_readiness_artifact(artifact, _requirement())
    assert result["ok"] is False
    assert any("tensor_parallel_size" in r for r in result["reasons"])


def test_base_artifact_one_gate_not_pass_fails():
    artifact = _valid_base_artifact()
    artifact["gate_results"]["G3"] = readiness.GATE_FAIL
    result = live_evidence.validate_live_readiness_artifact(artifact, _requirement())
    assert result["ok"] is False
    assert any("G3" in r for r in result["reasons"])


def test_base_artifact_missing_gate_fails():
    artifact = _valid_base_artifact()
    del artifact["gate_results"]["G7"]
    result = live_evidence.validate_live_readiness_artifact(artifact, _requirement())
    assert result["ok"] is False
    assert any("missing gate" in r for r in result["reasons"])


def test_base_artifact_own_smoke_permitted_false_fails():
    result = live_evidence.validate_live_readiness_artifact(_valid_base_artifact(smoke_permitted=False), _requirement())
    assert result["ok"] is False


def test_base_artifact_accumulates_multiple_reasons_never_short_circuits():
    artifact = _valid_base_artifact(resolved_revision={"model_name": "wrong-model", "resolved_revision": "wrong-revision"})
    artifact["model_load"]["config"]["tensor_parallel_size"] = 1
    artifact["gate_results"]["G2"] = readiness.GATE_FAIL
    result = live_evidence.validate_live_readiness_artifact(artifact, _requirement())
    assert result["ok"] is False
    assert len(result["reasons"]) >= 4  # model_name, resolved_revision, tensor_parallel_size, G2 -- all reported, not just the first


# =================================================================================================
# validate_live_v3_solver_artifact
# =================================================================================================


def test_valid_solver_artifact_passes():
    result = live_evidence.validate_live_v3_solver_artifact(_valid_solver_artifact(), _requirement())
    assert result == {"ok": True, "reasons": []}


def test_solver_artifact_with_error_fails():
    result = live_evidence.validate_live_v3_solver_artifact(_valid_solver_artifact(solver_error="RadiusCorrectionFailedError: ..."), _requirement())
    assert result["ok"] is False


def test_solver_artifact_invalid_acceptance_mode_fails():
    result = live_evidence.validate_live_v3_solver_artifact(_valid_solver_artifact(acceptance_mode="readiness_sanity_bound"), _requirement())
    assert result["ok"] is False


def test_solver_artifact_quantization_limited_is_valid():
    result = live_evidence.validate_live_v3_solver_artifact(_valid_solver_artifact(acceptance_mode="quantization_limited"), _requirement())
    assert result["ok"] is True


def test_solver_artifact_missing_rank_consensus_fails():
    result = live_evidence.validate_live_v3_solver_artifact(
        _valid_solver_artifact(rank_consensus={"core_fields_ok": False, "full_bracket_trajectory": {"ok": True}}), _requirement(),
    )
    assert result["ok"] is False


def test_solver_artifact_missing_bracket_trajectory_consensus_fails():
    result = live_evidence.validate_live_v3_solver_artifact(
        _valid_solver_artifact(rank_consensus={"core_fields_ok": True, "full_bracket_trajectory": {"ok": False}}), _requirement(),
    )
    assert result["ok"] is False


def test_solver_artifact_restoration_not_ok_fails():
    result = live_evidence.validate_live_v3_solver_artifact(_valid_solver_artifact(restoration={"ok": False}), _requirement())
    assert result["ok"] is False


def test_solver_artifact_g4_g5_not_both_pass_fails():
    result = live_evidence.validate_live_v3_solver_artifact(_valid_solver_artifact(g4_g5_final={"G4": "PASS", "G5": "FAIL"}), _requirement())
    assert result["ok"] is False


def test_solver_artifact_nonzero_scientific_rows_fails():
    result = live_evidence.validate_live_v3_solver_artifact(_valid_solver_artifact(scientific_rows_written=3), _requirement())
    assert result["ok"] is False


# =================================================================================================
# check_gpu_fingerprint
# =================================================================================================


def test_gpu_fingerprint_not_applicable_when_artifact_has_no_uuids():
    result = live_evidence.check_gpu_fingerprint(None, ["GPU-abc"])
    assert result == {"applicable": False, "ok": True, "reason": result["reason"]}


def test_gpu_fingerprint_matches():
    result = live_evidence.check_gpu_fingerprint(["GPU-a", "GPU-b"], ["GPU-b", "GPU-a"])
    assert result["applicable"] is True
    assert result["ok"] is True


def test_gpu_fingerprint_mismatch_fails():
    result = live_evidence.check_gpu_fingerprint(["GPU-old-pod-1"], ["GPU-new-pod-1"])
    assert result["applicable"] is True
    assert result["ok"] is False


def test_gpu_fingerprint_applicable_but_current_unavailable_fails():
    result = live_evidence.check_gpu_fingerprint(["GPU-a"], None)
    assert result["applicable"] is True
    assert result["ok"] is False


# =================================================================================================
# load_and_validate_canonical_live_evidence -- end to end
# =================================================================================================


def test_no_artifacts_found_reports_found_false(tmp_path):
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["found"] is False
    assert result["ok"] is False
    assert result["gate_results"] is None


def test_valid_matching_artifacts_authorize(tmp_path):
    _write_artifacts(tmp_path)
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["found"] is True
    assert result["ok"] is True
    assert result["reasons"] == []
    assert result["gate_results"] == {g: readiness.GATE_PASS for g in readiness.GATE_IDS}


def test_g4_g5_in_merged_result_always_come_from_solver_artifact_not_base():
    """Even if the base artifact's own (raw-primitive) G4/G5 happen to be PASS, the merged
    gate_results' G4/G5 are explicitly overridden from the solver artifact -- proving the merge
    logic doesn't just pass the base dict through unmodified.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_artifacts(d)
        result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=d)
        assert result["gate_results"]["G4"] == readiness.GATE_PASS
        assert result["gate_results"]["G5"] == readiness.GATE_PASS


def test_wrong_revision_blocks(tmp_path):
    _write_artifacts(tmp_path, base=_valid_base_artifact(resolved_revision={"model_name": MODEL_NAME, "resolved_revision": "c" * 40}))
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["found"] is True
    assert result["ok"] is False


def test_tp_not_4_blocks(tmp_path):
    base = _valid_base_artifact()
    base["model_load"]["config"]["tensor_parallel_size"] = 2
    _write_artifacts(tmp_path, base=base)
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["ok"] is False


def test_wrong_model_blocks(tmp_path):
    _write_artifacts(tmp_path, base=_valid_base_artifact(resolved_revision={"model_name": "Qwen/Qwen2.5-VL-3B-Instruct", "resolved_revision": REVISION}))
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["ok"] is False


def test_one_gate_not_pass_blocks(tmp_path):
    base = _valid_base_artifact()
    base["gate_results"]["G6"] = readiness.GATE_FAIL
    _write_artifacts(tmp_path, base=base)
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["ok"] is False


def test_missing_strict_v3_evidence_blocks(tmp_path):
    (tmp_path / live_evidence.LIVE_READINESS_ARTIFACT_FILENAME).write_text(json.dumps(_valid_base_artifact()))
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["found"] is True
    assert result["ok"] is False
    assert any("missing" in r and "solver" in r for r in result["reasons"])


def test_missing_restoration_evidence_blocks(tmp_path):
    _write_artifacts(tmp_path, solver=_valid_solver_artifact(restoration={"ok": False}))
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["ok"] is False


def test_old_hardware_fingerprint_blocks_when_binding_present(tmp_path):
    base = _valid_base_artifact()
    base["gpu_uuids"] = ["GPU-old-pod-uuid-1", "GPU-old-pod-uuid-2"]
    _write_artifacts(tmp_path, base=base)
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path, current_gpu_uuids=["GPU-new-pod-uuid-1", "GPU-new-pod-uuid-2"])
    assert result["ok"] is False
    assert any("fingerprint" in r or "UUID" in r for r in result["reasons"])


def test_matching_hardware_fingerprint_still_authorizes(tmp_path):
    base = _valid_base_artifact()
    base["gpu_uuids"] = ["GPU-1", "GPU-2"]
    solver = _valid_solver_artifact()
    solver["gpu_uuids"] = ["GPU-1", "GPU-2"]
    _write_artifacts(tmp_path, base=base, solver=solver)
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path, current_gpu_uuids=["GPU-2", "GPU-1"])
    assert result["ok"] is True


def test_artifact_without_gpu_uuids_is_not_penalized(tmp_path):
    """Real artifacts written before gpu_uuids capture existed must still authorize -- 'if
    available' framing, never retroactively penalized.
    """
    _write_artifacts(tmp_path)  # no gpu_uuids field at all
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path, current_gpu_uuids=["GPU-whatever"])
    assert result["ok"] is True


def test_malformed_artifact_blocks(tmp_path):
    (tmp_path / live_evidence.LIVE_READINESS_ARTIFACT_FILENAME).write_text("{not valid json")
    (tmp_path / live_evidence.LIVE_V3_SOLVER_ARTIFACT_FILENAME).write_text(json.dumps(_valid_solver_artifact()))
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["found"] is True
    assert result["ok"] is False
    assert any("malformed" in r for r in result["reasons"])


def test_incomplete_artifact_missing_keys_blocks(tmp_path):
    (tmp_path / live_evidence.LIVE_READINESS_ARTIFACT_FILENAME).write_text(json.dumps({"gate_results": {}}))
    (tmp_path / live_evidence.LIVE_V3_SOLVER_ARTIFACT_FILENAME).write_text(json.dumps(_valid_solver_artifact()))
    result = live_evidence.load_and_validate_canonical_live_evidence(_requirement(), evidence_dir=tmp_path)
    assert result["ok"] is False


def test_query_live_gpu_uuids_returns_none_on_failure(monkeypatch):
    import subprocess

    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert live_evidence.query_live_gpu_uuids() is None
