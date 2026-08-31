"""Tests for stage11_32b_s2_live_evidence.py -- the S2 (coarse-anatomy) multi-region live-
readiness evidence consumption. CPU-only, no GPU/ray/vllm import.
"""
from __future__ import annotations

import json

import neural_thickets_repro.stage11_32b_readiness as readiness
import neural_thickets_repro.stage11_32b_s2_live_evidence as s2_evidence
from neural_thickets_repro.stage11_32b_live_evidence import LiveEvidenceIdentityRequirement

_REVISION = "7cfb30d71a1f4f49a57592323337a4a4727301da"


def _base_artifact(revision=_REVISION):
    return {
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


def _one_region_pass(revision=_REVISION):
    return {
        "solver_error": None, "acceptance_mode": "strict",
        "rank_consensus": {"core_fields_ok": True, "full_bracket_trajectory": {"ok": True}},
        "restoration": {"ok": True}, "g4_g5_final": {"G4": "PASS", "G5": "PASS"},
    }


def _s2_artifact(*, revision=_REVISION, tp=4, regions=None, scientific_rows_written=0):
    if regions is None:
        regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS}
    return {
        "resolved_revision": {"model_name": readiness.FROZEN_32B_MODEL_NAME, "resolved_revision": revision},
        "tensor_parallel_size": tp, "regions": regions, "scientific_rows_written": scientific_rows_written,
    }


def _requirement(revision=_REVISION):
    return LiveEvidenceIdentityRequirement(resolved_revision=revision)


# =================================================================================================
# validate_live_s2_solver_probe_artifact
# =================================================================================================


def test_s2_solver_artifact_all_three_regions_present_and_passing_is_ok():
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(), _requirement())
    assert result["ok"] is True
    assert result["reasons"] == []


def test_s2_regions_covers_exactly_vision_connector_language():
    assert set(s2_evidence.S2_REGIONS) == {"vision", "multimodal_connector_or_merger", "language"}
    assert len(s2_evidence.S2_REGIONS) == 3


def test_s2_solver_artifact_missing_one_region_fails():
    """Property 16: S2 readiness must fail if ANY ONE of vision/connector/language is missing."""
    regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS if r != "vision"}
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(regions=regions), _requirement())
    assert result["ok"] is False
    assert any("vision" in reason and "missing" in reason.lower() for reason in result["reasons"])


def test_s2_solver_artifact_one_region_failing_fails_whole_artifact():
    """Property 16 (mirror image): all three present but ONE fails its own G4/G5 -> whole S2
    artifact is not ok, never partial credit.
    """
    regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS}
    regions["language"]["g4_g5_final"] = {"G4": "FAIL", "G5": "FAIL"}
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(regions=regions), _requirement())
    assert result["ok"] is False
    assert any("language" in reason for reason in result["reasons"])


def test_s2_solver_artifact_one_region_with_solver_error_fails():
    regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS}
    regions["multimodal_connector_or_merger"]["solver_error"] = "RadiusCorrectionFailedError: did not converge"
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(regions=regions), _requirement())
    assert result["ok"] is False
    assert any("multimodal_connector_or_merger" in reason and "solver_error" in reason for reason in result["reasons"])


def test_s2_solver_artifact_one_region_rank_disagreement_fails():
    regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS}
    regions["vision"]["rank_consensus"]["core_fields_ok"] = False
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(regions=regions), _requirement())
    assert result["ok"] is False


def test_s2_solver_artifact_one_region_bad_restoration_fails():
    regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS}
    regions["language"]["restoration"]["ok"] = False
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(regions=regions), _requirement())
    assert result["ok"] is False


def test_s2_solver_artifact_wrong_revision_fails():
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(revision="f" * 40), _requirement())
    assert result["ok"] is False
    assert any("resolved_revision" in reason for reason in result["reasons"])


def test_s2_solver_artifact_wrong_tp_size_fails():
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(tp=2), _requirement())
    assert result["ok"] is False


def test_s2_solver_artifact_nonzero_scientific_rows_fails():
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(scientific_rows_written=5), _requirement())
    assert result["ok"] is False


def test_s2_solver_artifact_accumulates_all_reasons_never_short_circuits():
    regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS}
    regions["vision"]["g4_g5_final"] = {"G4": "FAIL", "G5": "FAIL"}
    regions["language"]["restoration"]["ok"] = False
    result = s2_evidence.validate_live_s2_solver_probe_artifact(_s2_artifact(revision="f" * 40, regions=regions), _requirement())
    assert result["ok"] is False
    assert len(result["reasons"]) >= 3  # revision mismatch + vision g4/g5 + language restoration


# =================================================================================================
# load_and_validate_canonical_s2_live_evidence
# =================================================================================================


def test_s2_load_evidence_not_found_when_no_files_exist(tmp_path):
    result = s2_evidence.load_and_validate_canonical_s2_live_evidence(
        _requirement(), base_evidence_dir=tmp_path / "base", s2_evidence_dir=tmp_path / "s2", current_gpu_uuids=None,
    )
    assert result["found"] is False
    assert result["ok"] is False
    assert result["gate_results"] is None


def test_s2_load_evidence_found_but_missing_s2_artifact(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(_base_artifact()))
    result = s2_evidence.load_and_validate_canonical_s2_live_evidence(
        _requirement(), base_evidence_dir=base_dir, s2_evidence_dir=tmp_path / "s2", current_gpu_uuids=None,
    )
    assert result["found"] is True
    assert result["ok"] is False
    assert any("S2" in r for r in result["reasons"])


def test_s2_load_evidence_ok_merges_g4_g5_from_s2_artifact_and_keeps_base_other_gates(tmp_path):
    """The base artifact's OWN precondition is that ALL 8 of its gates already show PASS (its own,
    looser, one-shot-primitive-based G4/G5) -- validate_live_readiness_artifact requires this
    unconditionally, reused BY IMPORT unmodified. What THIS module adds on top: the FINAL merged
    G4/G5 always come from the S2 multi-region solver-probe artifact, never merely inherited from
    the base artifact's own (looser) values -- proven by the companion test below, where a valid
    base + a FAILING S2 artifact still ends up NOT ok despite the base's own G4/G5 already PASS.
    """
    base_dir, s2_dir = tmp_path / "base", tmp_path / "s2"
    base_dir.mkdir()
    s2_dir.mkdir()
    base = _base_artifact()
    (base_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(base))
    (s2_dir / "stage11_32b_s2_live_v3_solver_probe_report.json").write_text(json.dumps(_s2_artifact()))

    result = s2_evidence.load_and_validate_canonical_s2_live_evidence(_requirement(), base_evidence_dir=base_dir, s2_evidence_dir=s2_dir, current_gpu_uuids=None)
    assert result["found"] is True
    assert result["ok"] is True
    assert result["gate_results"]["G4"] == readiness.GATE_PASS
    assert result["gate_results"]["G5"] == readiness.GATE_PASS
    assert result["gate_results"]["G1"] == readiness.GATE_PASS  # from base artifact, untouched


def test_s2_load_evidence_a_passing_base_alone_never_authorizes_when_s2_artifact_fails(tmp_path):
    """The base artifact's own (looser) G4/G5 being PASS is NEVER, by itself, sufficient -- the
    S2 multi-region solver-probe artifact must ALSO validate (all three regions PASS).
    """
    base_dir, s2_dir = tmp_path / "base", tmp_path / "s2"
    base_dir.mkdir()
    s2_dir.mkdir()
    (base_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(_base_artifact()))
    regions = {r: _one_region_pass() for r in s2_evidence.S2_REGIONS}
    regions["vision"]["g4_g5_final"] = {"G4": "FAIL", "G5": "FAIL"}
    (s2_dir / "stage11_32b_s2_live_v3_solver_probe_report.json").write_text(json.dumps(_s2_artifact(regions=regions)))

    result = s2_evidence.load_and_validate_canonical_s2_live_evidence(_requirement(), base_evidence_dir=base_dir, s2_evidence_dir=s2_dir, current_gpu_uuids=None)
    assert result["ok"] is False
    assert result["gate_results"] is None


def test_s2_load_evidence_fails_closed_when_base_gates_not_all_pass(tmp_path):
    base_dir, s2_dir = tmp_path / "base", tmp_path / "s2"
    base_dir.mkdir()
    s2_dir.mkdir()
    base = _base_artifact()
    base["gate_results"]["G2"] = readiness.GATE_FAIL
    (base_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(base))
    (s2_dir / "stage11_32b_s2_live_v3_solver_probe_report.json").write_text(json.dumps(_s2_artifact()))

    result = s2_evidence.load_and_validate_canonical_s2_live_evidence(_requirement(), base_evidence_dir=base_dir, s2_evidence_dir=s2_dir, current_gpu_uuids=None)
    assert result["ok"] is False
    assert result["gate_results"] is None


def test_s2_load_evidence_gpu_fingerprint_binding_when_present():
    from neural_thickets_repro.stage11_32b_live_evidence import check_gpu_fingerprint

    artifact_uuids = ["GPU-aaa", "GPU-bbb"]
    ok = check_gpu_fingerprint(artifact_uuids, ["GPU-aaa", "GPU-bbb"])
    mismatched = check_gpu_fingerprint(artifact_uuids, ["GPU-ccc"])
    assert ok["ok"] is True
    assert mismatched["ok"] is False
