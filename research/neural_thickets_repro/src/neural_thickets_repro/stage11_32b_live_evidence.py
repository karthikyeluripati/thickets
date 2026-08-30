"""Stage-11 32B live-readiness EVIDENCE consumption -- the wiring between a completed live
G1-G8 verification (diagnostics/stage11_32b_live_readiness.py) + the strict distributed-v3
solver probe (diagnostics/stage11_32b_live_v3_solver_probe.py) and the actual 32B smoke runner
(run_stage11_whole_model_scaling.py, via stage11_32b_readiness.run_32b_readiness_preflight_and_
report).

ROOT CAUSE THIS MODULE FIXES (live-discovered, real pod): `run_32b_readiness_preflight_and_report`
previously REBUILT a fresh gate_results dict from scratch on EVERY call -- G1/G2/G3/G6/G7/G8
hardcoded to NOT_YET_VERIFIED, G4/G5 hardcoded to READY_FOR_LIVE_VERIFICATION from the CPU-only
proof alone -- it never read any file from disk at all. Compounding this: even a hypothetical
"read a report" attempt would have looked in the wrong place -- the two live-verification scripts
always write to one FIXED, well-known directory (results/stage11_32b_live_readiness/), while the
scientific runner's OWN gate-report artifact is written to a PLAN-SPECIFIC directory
(results/stage11_whole_model_scaling/<run_signature>/) that differs per invocation and never
coincides with where real evidence lives. Confirmed live: a real G1-G8 PASS + strict v3 solver
PASS on a real 4xL40S pod left the runner's own report at the exact CPU-only defaults
(G1/G2/G3/G6/G7/G8=NOT_YET_VERIFIED, G4/G5=READY_FOR_LIVE_VERIFICATION) and blocked before engine
launch -- the live evidence had NO path to reach the runner at all.

THE FIX: `load_and_validate_canonical_live_evidence` reads BOTH canonical artifacts from their
FIXED, well-known location (independent of whatever plan-specific `output_dir` the current
invocation happens to use), strictly IDENTITY-BINDS them to the CURRENT requested run (model /
revision / TP size / dtype / base_snapshot_mode / gpu_memory_utilization / max_model_len /
enable_prefix_caching, plus GPU UUIDs when persisted), and returns a validated gate_results dict
ONLY when every check passes -- never partial credit, never a silent pass-through of a stale or
mismatched artifact. This is purely ADDITIVE: it does not change what any gate MEANS or how any
gate is computed live -- only WHERE that already-computed live evidence is discovered and how
strictly it is bound to the current run before being trusted.

WHY G4/G5 REQUIRE THE SEPARATE STRICT SOLVER ARTIFACT, NOT JUST THE BASE ONE: the base G1-G8
artifact's own G4/G5 come from the raw one-shot `apply_anatomical_relative_l2_distributed`
primitive (a readiness sanity check, corrected tolerance -- see
thicket.distributed_perturbation.classify_g4_g5_live_check's own docstring). The ACTUAL Stage-11
candidate lifecycle uses the iterative bracket/bisection/plateau solver
(scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed), exercised only by
the separate strict solver probe. Smoke authorization therefore requires BOTH artifacts to
validate, and G4/G5 in the merged gate_results always come from the solver probe, never the base
artifact's own (looser) G4/G5.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .stage11_32b_readiness import FROZEN_32B_MODEL_NAME, GATE_IDS, GATE_PASS

REPO_ROOT = Path(__file__).resolve().parents[2]
# THE fixed, well-known location both diagnostics/stage11_32b_live_readiness.py and diagnostics/
# stage11_32b_live_v3_solver_probe.py already write to -- never the scientific runner's own
# plan-specific output_dir, which differs per invocation (smoke vs full, differing run_signature)
# and can never coincide with a fixed evidence location by construction.
DEFAULT_LIVE_READINESS_EVIDENCE_DIR = REPO_ROOT / "results" / "stage11_32b_live_readiness"
LIVE_READINESS_ARTIFACT_FILENAME = "stage11_32b_live_readiness_report.json"
LIVE_V3_SOLVER_ARTIFACT_FILENAME = "stage11_32b_live_v3_solver_probe_report.json"


@dataclass(frozen=True)
class LiveEvidenceIdentityRequirement:
    """Every field the CURRENT requested 32B smoke invocation must match against a persisted
    live-evidence artifact before that artifact is trusted to authorize THIS run -- the literal
    identity-binding list this wiring fix was asked for. `resolved_revision` has no meaningful
    default (a real one must always be supplied by the caller -- the empty-string default exists
    only so dataclass field ordering works, and would correctly fail-closed against any real
    artifact if a caller ever forgot to override it).
    """
    resolved_revision: str = ""
    model_name: str = FROZEN_32B_MODEL_NAME
    tensor_parallel_size: int = 4
    dtype: str = "bfloat16"
    base_snapshot_mode: str = "cpu_base_weights"
    gpu_memory_utilization: float = 0.60
    max_model_len: int = 4096
    enable_prefix_caching: bool = False


def _check(reasons: List[str], label: str, expected: Any, actual: Any) -> None:
    """Every identity check is independent and ALWAYS runs -- reasons accumulates the full
    picture rather than short-circuiting on the first mismatch, so a caller (and a test) can see
    every reason a stale/wrong artifact was rejected, not just one.
    """
    if actual != expected:
        reasons.append(f"{label} mismatch: artifact has {actual!r}, current run requires {expected!r}")


def validate_live_readiness_artifact(artifact: Dict[str, Any], requirement: LiveEvidenceIdentityRequirement) -> Dict[str, Any]:
    """Validates the BASE G1-G8 artifact (diagnostics/stage11_32b_live_readiness.py's own report
    schema, confirmed by direct inspection of a real run's persisted JSON). Returns
    {"ok": bool, "reasons": List[str]} -- never raises.
    """
    reasons: List[str] = []
    resolved = artifact.get("resolved_revision") or {}
    _check(reasons, "model_name", requirement.model_name, resolved.get("model_name"))
    _check(reasons, "resolved_revision", requirement.resolved_revision, resolved.get("resolved_revision"))

    model_load = artifact.get("model_load") or {}
    if model_load.get("ok") is not True:
        reasons.append(f"model_load.ok is {model_load.get('ok')!r}, expected True")
    config = model_load.get("config") or {}
    _check(reasons, "tensor_parallel_size", requirement.tensor_parallel_size, config.get("tensor_parallel_size"))
    _check(reasons, "dtype", requirement.dtype, config.get("dtype"))
    _check(reasons, "base_snapshot_mode", requirement.base_snapshot_mode, config.get("base_snapshot_mode"))
    _check(reasons, "gpu_memory_utilization", requirement.gpu_memory_utilization, config.get("gpu_memory_utilization"))
    _check(reasons, "max_model_len", requirement.max_model_len, config.get("max_model_len"))
    _check(reasons, "enable_prefix_caching", requirement.enable_prefix_caching, config.get("enable_prefix_caching"))

    gate_results = artifact.get("gate_results") or {}
    missing_gates = [g for g in GATE_IDS if g not in gate_results]
    if missing_gates:
        reasons.append(f"base artifact missing gate result(s): {missing_gates}")
    not_passing = {g: gate_results[g] for g in GATE_IDS if g in gate_results and gate_results[g] != GATE_PASS}
    if not_passing:
        reasons.append(f"base artifact gate(s) not PASS: {not_passing}")
    if artifact.get("smoke_permitted") is not True:
        reasons.append(f"base artifact's own smoke_permitted is {artifact.get('smoke_permitted')!r}, expected True")

    return {"ok": not reasons, "reasons": reasons}


def validate_live_v3_solver_artifact(artifact: Dict[str, Any], requirement: LiveEvidenceIdentityRequirement) -> Dict[str, Any]:
    """Validates the STRICT distributed-v3 solver probe artifact (diagnostics/stage11_32b_live_
    v3_solver_probe.py's own report schema) -- the ONLY acceptable source of a real G4/G5 PASS
    for smoke authorization (see module docstring for why the base artifact's own G4/G5 is not
    sufficient alone). Returns {"ok": bool, "reasons": List[str]} -- never raises.
    """
    reasons: List[str] = []
    resolved = artifact.get("resolved_revision") or {}
    _check(reasons, "model_name (v3 solver artifact)", requirement.model_name, resolved.get("model_name"))
    _check(reasons, "resolved_revision (v3 solver artifact)", requirement.resolved_revision, resolved.get("resolved_revision"))

    if artifact.get("solver_error") is not None:
        reasons.append(f"v3 solver artifact has a solver_error: {artifact.get('solver_error')}")

    acceptance_mode = artifact.get("acceptance_mode")
    if acceptance_mode not in ("strict", "quantization_limited"):
        reasons.append(f"v3 solver artifact acceptance_mode {acceptance_mode!r} is not a valid scientific acceptance mode (strict/quantization_limited)")

    rank_consensus = artifact.get("rank_consensus") or {}
    if rank_consensus.get("core_fields_ok") is not True:
        reasons.append(f"v3 solver artifact rank_consensus.core_fields_ok is {rank_consensus.get('core_fields_ok')!r}, expected True")
    trajectory = rank_consensus.get("full_bracket_trajectory") or {}
    if trajectory.get("ok") is not True:
        reasons.append(f"v3 solver artifact rank_consensus.full_bracket_trajectory.ok is {trajectory.get('ok')!r}, expected True")

    restoration = artifact.get("restoration") or {}
    if restoration.get("ok") is not True:
        reasons.append(f"v3 solver artifact restoration.ok is {restoration.get('ok')!r}, expected True")

    g4_g5_final = artifact.get("g4_g5_final") or {}
    if g4_g5_final.get("G4") != GATE_PASS or g4_g5_final.get("G5") != GATE_PASS:
        reasons.append(f"v3 solver artifact g4_g5_final is not both PASS: {g4_g5_final}")

    if artifact.get("scientific_rows_written") != 0:
        reasons.append(f"v3 solver artifact scientific_rows_written is {artifact.get('scientific_rows_written')!r}, expected 0")

    return {"ok": not reasons, "reasons": reasons}


def check_gpu_fingerprint(artifact_gpu_uuids: Optional[Sequence[str]], current_gpu_uuids: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Hardware-fingerprint binding: 'If GPU UUIDs / pod hardware fingerprint are available, bind
    to them too.' OPTIONAL and non-penalizing when the artifact never persisted GPU UUIDs (every
    artifact produced before this check existed) -- `applicable=False`, `ok=True` -- matching the
    literal 'if available' framing exactly; every OTHER identity/gate/solver check still fully
    gates authorization regardless. Once an artifact DOES persist UUIDs, this becomes a HARD,
    EXACT-SET requirement so a stale artifact from a different pod can never silently authorize
    the current machine.
    """
    if not artifact_gpu_uuids:
        return {"applicable": False, "ok": True, "reason": "artifact has no persisted gpu_uuids -- hardware binding not available for this artifact"}
    if not current_gpu_uuids:
        return {"applicable": True, "ok": False, "reason": "artifact persists gpu_uuids but the current run could not query live GPU UUIDs to compare against"}
    ok = set(artifact_gpu_uuids) == set(current_gpu_uuids)
    return {"applicable": True, "ok": ok, "reason": None if ok else f"GPU UUID mismatch: artifact={sorted(artifact_gpu_uuids)} current={sorted(current_gpu_uuids)}"}


def query_live_gpu_uuids() -> Optional[List[str]]:
    """Live nvidia-smi query for the CURRENT invocation's GPU UUIDs. Returns None (never raises)
    if nvidia-smi is unavailable or the query fails -- a query failure degrades hardware binding
    to 'not applicable' for this specific check (see check_gpu_fingerprint), it must never abort
    the rest of readiness evaluation on its own.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=15).stdout
        return [line.strip() for line in out.strip().splitlines() if line.strip()]
    except Exception:  # noqa: BLE001 -- see docstring
        return None


def load_and_validate_canonical_live_evidence(
    requirement: LiveEvidenceIdentityRequirement, *, evidence_dir: "str | Path" = DEFAULT_LIVE_READINESS_EVIDENCE_DIR,
    current_gpu_uuids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """THE single function stage11_32b_readiness.run_32b_readiness_preflight_and_report calls.

    Returns {"found": bool, "ok": bool, "reasons": List[str], "gate_results": Optional[Dict]}:
      - found=False: NEITHER artifact file exists at `evidence_dir` -- "no valid artifact exists,
        clearly request the live readiness verification" (never an exception; a fresh/never-
        verified machine is an ordinary, expected state, not an error).
      - found=True, ok=False: artifact(s) exist but fail identity-binding or gate/solver
        validation -- reported as exactly that (never conflated with "missing"); smoke_permitted
        is False downstream either way, but the caller can report WHY.
      - ok=True: both artifacts validated against `requirement` -- `gate_results` is the base
        artifact's own gate_results with G4/G5 overridden to PASS from the (separately, more
        strictly validated) solver probe artifact.
    """
    evidence_dir = Path(evidence_dir)
    base_path = evidence_dir / LIVE_READINESS_ARTIFACT_FILENAME
    solver_path = evidence_dir / LIVE_V3_SOLVER_ARTIFACT_FILENAME

    if not base_path.exists() and not solver_path.exists():
        return {"found": False, "ok": False, "reasons": [f"no live readiness artifacts found at {evidence_dir}"], "gate_results": None}

    reasons: List[str] = []
    if not base_path.exists():
        reasons.append(f"missing base G1-G8 artifact at {base_path}")
    if not solver_path.exists():
        reasons.append(f"missing strict distributed-v3 solver artifact at {solver_path}")
    if reasons:
        return {"found": True, "ok": False, "reasons": reasons, "gate_results": None}

    try:
        base_artifact = json.loads(base_path.read_text())
        solver_artifact = json.loads(solver_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return {"found": True, "ok": False, "reasons": [f"malformed artifact: {exc}"], "gate_results": None}

    base_check = validate_live_readiness_artifact(base_artifact, requirement)
    solver_check = validate_live_v3_solver_artifact(solver_artifact, requirement)
    base_gpu_check = check_gpu_fingerprint(base_artifact.get("gpu_uuids"), current_gpu_uuids)
    solver_gpu_check = check_gpu_fingerprint(solver_artifact.get("gpu_uuids"), current_gpu_uuids)

    all_reasons = list(base_check["reasons"]) + list(solver_check["reasons"])
    if not base_gpu_check["ok"]:
        all_reasons.append(f"base artifact hardware fingerprint: {base_gpu_check['reason']}")
    if not solver_gpu_check["ok"]:
        all_reasons.append(f"v3 solver artifact hardware fingerprint: {solver_gpu_check['reason']}")

    ok = base_check["ok"] and solver_check["ok"] and base_gpu_check["ok"] and solver_gpu_check["ok"]
    gate_results = None
    if ok:
        gate_results = dict(base_artifact["gate_results"])
        gate_results["G4"] = GATE_PASS
        gate_results["G5"] = GATE_PASS

    return {"found": True, "ok": ok, "reasons": all_reasons, "gate_results": gate_results}
