"""Stage-11 32B S2 (coarse anatomy) live-readiness EVIDENCE consumption -- the S2-specific analog
of stage11_32b_live_evidence.py.

WHY S1's EVIDENCE ALONE IS NOT SUFFICIENT FOR S2: G1/G2/G3/G6/G7/G8 are TP/hardware/snapshot/test
facts, not region-scoped -- S2 reuses S1's own base G1-G8 artifact for those UNCHANGED (via
stage11_32b_live_evidence.validate_live_readiness_artifact, reused BY IMPORT, never reinterpreted
or weakened). G4/G5, however, come from a live run of the REAL iterative distributed-v3 solver
against REAL sharded parameters -- and S1's own solver-probe evidence was gathered against ONE
specific parameter subset (multimodal_connector_or_merger; see diagnostics/stage11_32b_live_v3_
solver_probe.py's own "Probe region" log line). A proof that the solver converges correctly on
that subset's shard shapes/sizes says nothing, on its own, about whether it converges correctly on
DIFFERENT subsets (vision, language) with different shard shapes -- the identity-binding in
stage11_32b_live_evidence.LiveEvidenceIdentityRequirement never included `anatomy_region` at all,
so silently reusing S1's evidence to authorize S2 would be scientifically ungrounded. This module
requires its OWN, S2-specific, multi-region solver-probe artifact (diagnostics/stage11_32b_s2_live_
v3_solver_probe.py) proving ALL THREE frozen S2 regions (vision, multimodal_connector_or_merger,
language) pass the strict solver + rank-consensus + restoration checks, gathered in ONE live TP=4
engine session (no per-region model reload), before authorizing ANY S2 candidate.

The already-observed connector-region PASS from the S1 probe may inform expectations, but it does
NOT, by itself, authorize S2 -- this module's own canonical artifact must contain all three
regions for S2 authorization to be explicit and reproducible, exactly as requested.

Merge semantics (mirrors stage11_32b_live_evidence.load_and_validate_canonical_live_evidence
exactly, generalized from "1 solver artifact" to "1 solver artifact spanning 3 regions"):
    gate_results = {G1, G2, G3, G6, G7, G8} from S1's own validated base artifact
                 + {G4: PASS, G5: PASS} ONLY IF the S2 solver-probe artifact validates AND every
                   one of vision/multimodal_connector_or_merger/language reports G4=PASS/G5=PASS
                   internally.
Never partial credit -- one region missing or failing fails S2 authorization entirely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .run_stage8_coarse_anatomical_atlas import STAGE8_REGIONS
from .stage11_32b_readiness import GATE_PASS
from .stage11_32b_live_evidence import (
    DEFAULT_LIVE_READINESS_EVIDENCE_DIR,
    LIVE_READINESS_ARTIFACT_FILENAME,
    LiveEvidenceIdentityRequirement,
    check_gpu_fingerprint,
    query_live_gpu_uuids,
    validate_live_readiness_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# THE fixed, well-known location diagnostics/stage11_32b_s2_live_v3_solver_probe.py writes to --
# separate from S1's own DEFAULT_LIVE_READINESS_EVIDENCE_DIR (never conflated: S1's artifact
# proves ONE region; this artifact proves THREE), and separate from any scientific runner's own
# plan-specific output_dir.
DEFAULT_S2_LIVE_READINESS_EVIDENCE_DIR = REPO_ROOT / "results" / "stage11_32b_s2_live_readiness"
S2_LIVE_V3_SOLVER_ARTIFACT_FILENAME = "stage11_32b_s2_live_v3_solver_probe_report.json"

# The frozen S2 anatomy regions every canonical S2 solver-probe artifact must cover -- reused BY
# IDENTITY from Stage 8/11-7B's own frozen region tuple, never retyped as an independent literal.
S2_REGIONS: Sequence[str] = STAGE8_REGIONS


def _check(reasons: List[str], label: str, expected: Any, actual: Any) -> None:
    if actual != expected:
        reasons.append(f"{label} mismatch: artifact has {actual!r}, current run requires {expected!r}")


def validate_live_s2_solver_probe_artifact(artifact: Dict[str, Any], requirement: LiveEvidenceIdentityRequirement) -> Dict[str, Any]:
    """Validates the S2 multi-region solver-probe artifact (diagnostics/stage11_32b_s2_live_v3_
    solver_probe.py's own report schema: a top-level `regions` dict keyed by region label, each
    value shaped like S1's own per-probe report). Requires EVERY one of S2_REGIONS to be present
    and individually PASS -- a single missing or failing region fails the whole artifact, never
    partial credit. Returns {"ok": bool, "reasons": List[str]} -- never raises.
    """
    reasons: List[str] = []
    resolved = artifact.get("resolved_revision") or {}
    _check(reasons, "model_name (S2 solver artifact)", requirement.model_name, resolved.get("model_name"))
    _check(reasons, "resolved_revision (S2 solver artifact)", requirement.resolved_revision, resolved.get("resolved_revision"))
    _check(reasons, "tensor_parallel_size (S2 solver artifact)", requirement.tensor_parallel_size, artifact.get("tensor_parallel_size"))

    regions = artifact.get("regions") or {}
    missing_regions = [r for r in S2_REGIONS if r not in regions]
    if missing_regions:
        reasons.append(f"S2 solver artifact missing region(s): {missing_regions}")

    for region in S2_REGIONS:
        info = regions.get(region)
        if info is None:
            continue  # already reported as missing above -- do not double-report
        if info.get("solver_error") is not None:
            reasons.append(f"region {region!r}: solver_error present: {info.get('solver_error')}")
        acceptance_mode = info.get("acceptance_mode")
        if acceptance_mode not in ("strict", "quantization_limited"):
            reasons.append(f"region {region!r}: acceptance_mode {acceptance_mode!r} is not a valid scientific acceptance mode")
        rank_consensus = info.get("rank_consensus") or {}
        if rank_consensus.get("core_fields_ok") is not True:
            reasons.append(f"region {region!r}: rank_consensus.core_fields_ok is {rank_consensus.get('core_fields_ok')!r}, expected True")
        trajectory = rank_consensus.get("full_bracket_trajectory") or {}
        if trajectory.get("ok") is not True:
            reasons.append(f"region {region!r}: rank_consensus.full_bracket_trajectory.ok is {trajectory.get('ok')!r}, expected True")
        restoration = info.get("restoration") or {}
        if restoration.get("ok") is not True:
            reasons.append(f"region {region!r}: restoration.ok is {restoration.get('ok')!r}, expected True")
        g4_g5 = info.get("g4_g5_final") or {}
        if g4_g5.get("G4") != GATE_PASS or g4_g5.get("G5") != GATE_PASS:
            reasons.append(f"region {region!r}: g4_g5_final is not both PASS: {g4_g5}")

    if artifact.get("scientific_rows_written") != 0:
        reasons.append(f"S2 solver artifact scientific_rows_written is {artifact.get('scientific_rows_written')!r}, expected 0")

    return {"ok": not reasons, "reasons": reasons}


def load_and_validate_canonical_s2_live_evidence(
    requirement: LiveEvidenceIdentityRequirement, *,
    base_evidence_dir: "str | Path" = DEFAULT_LIVE_READINESS_EVIDENCE_DIR,
    s2_evidence_dir: "str | Path" = DEFAULT_S2_LIVE_READINESS_EVIDENCE_DIR,
    current_gpu_uuids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """THE single function a 32B S2 (anatomy) runner calls before permitting any candidate.

    Returns {"found": bool, "ok": bool, "reasons": List[str], "gate_results": Optional[Dict]} --
    same shape as stage11_32b_live_evidence.load_and_validate_canonical_live_evidence:
      - found=False: NEITHER artifact exists -- a never-verified machine, not an error.
      - found=True, ok=False: artifact(s) exist but fail identity-binding, gate validation, or
        the S2-specific all-three-regions-PASS requirement.
      - ok=True: gate_results = S1's own validated base artifact's G1/G2/G3/G6/G7/G8, with G4/G5
        overridden to PASS from the S2 multi-region solver-probe artifact (never from S1's own,
        single-region, G4/G5).
    """
    base_path = Path(base_evidence_dir) / LIVE_READINESS_ARTIFACT_FILENAME
    s2_path = Path(s2_evidence_dir) / S2_LIVE_V3_SOLVER_ARTIFACT_FILENAME

    if not base_path.exists() and not s2_path.exists():
        return {"found": False, "ok": False, "reasons": [f"no S2 live readiness artifacts found (base={base_path}, s2={s2_path})"], "gate_results": None}

    reasons: List[str] = []
    if not base_path.exists():
        reasons.append(f"missing base G1-G8 artifact at {base_path}")
    if not s2_path.exists():
        reasons.append(f"missing S2 multi-region solver artifact at {s2_path}")
    if reasons:
        return {"found": True, "ok": False, "reasons": reasons, "gate_results": None}

    try:
        base_artifact = json.loads(base_path.read_text())
        s2_artifact = json.loads(s2_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return {"found": True, "ok": False, "reasons": [f"malformed artifact: {exc}"], "gate_results": None}

    base_check = validate_live_readiness_artifact(base_artifact, requirement)
    s2_check = validate_live_s2_solver_probe_artifact(s2_artifact, requirement)
    base_gpu_check = check_gpu_fingerprint(base_artifact.get("gpu_uuids"), current_gpu_uuids)
    s2_gpu_check = check_gpu_fingerprint(s2_artifact.get("gpu_uuids"), current_gpu_uuids)

    all_reasons = list(base_check["reasons"]) + list(s2_check["reasons"])
    if not base_gpu_check["ok"]:
        all_reasons.append(f"base artifact hardware fingerprint: {base_gpu_check['reason']}")
    if not s2_gpu_check["ok"]:
        all_reasons.append(f"S2 solver artifact hardware fingerprint: {s2_gpu_check['reason']}")

    ok = base_check["ok"] and s2_check["ok"] and base_gpu_check["ok"] and s2_gpu_check["ok"]
    gate_results = None
    if ok:
        gate_results = dict(base_artifact["gate_results"])
        gate_results["G4"] = GATE_PASS
        gate_results["G5"] = GATE_PASS

    return {"found": True, "ok": ok, "reasons": all_reasons, "gate_results": gate_results}


__all__ = [
    "DEFAULT_S2_LIVE_READINESS_EVIDENCE_DIR", "S2_LIVE_V3_SOLVER_ARTIFACT_FILENAME", "S2_REGIONS",
    "validate_live_s2_solver_probe_artifact", "load_and_validate_canonical_s2_live_evidence", "query_live_gpu_uuids",
]
