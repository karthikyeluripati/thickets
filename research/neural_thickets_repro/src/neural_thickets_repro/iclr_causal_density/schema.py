"""Long-form result row schema for the causal-density pilot -- every field the task spec
requires (Phase 3), persisted one row per (candidate-or-base, capability, condition, example).
Append-only JSONL with flush+fsync (mirrors thicket.schema.ExperimentResultRecord's own
durability discipline exactly -- reused BY PATTERN, not by import, since this schema's field
set is materially different).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = "iclr_causal_density_v1"


@dataclass(frozen=True)
class CausalDensityResultRow:
    source_commit: str
    run_id: str
    model_name: str
    model_revision: str
    capability: str
    dataset_source: str
    subset_role: str          # "selection" or "audit"
    sample_id: str
    original_image_id: str
    evaluated_image_id: str
    condition: str            # "correct_image" | "shuffled_image" | "text_only"
    scope: Optional[str]      # None for base-model rows
    radius: Optional[float]   # None for base-model rows
    seed: Optional[int]       # None for base-model rows
    candidate_id: Optional[str]  # None for base-model rows
    is_base: bool
    prediction: str
    normalized_prediction: str
    target: Any
    per_example_score: float
    aggregate_score: Optional[float]         # filled in at write-time for the owning (candidate, capability, condition) group; None on the raw per-example row
    perturbation_norm: Optional[float]        # realized relative-L2 for this candidate; None for base rows
    norm_verification_ok: Optional[bool]
    scope_isolation_verification_ok: Optional[bool]
    restoration_verification_ok: Optional[bool]
    decoding_config: Dict[str, Any] = field(default_factory=dict)
    adapter_schema_version: str = SCHEMA_VERSION
    failure_status: str = "ok"                # "ok" | "restoration_failed" | "norm_failed" | "isolation_failed" | "eval_error"
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalDensityResultRow":
        return cls(**d)


def append_rows(path: "str | Path", rows: Sequence[CausalDensityResultRow]) -> None:
    """Durable per-candidate persistence -- called ONLY after a candidate's entire
    perturb -> evaluate(all conditions, all capabilities) -> restore -> verify cycle has
    already succeeded, exactly like thicket.schema/append_candidate_rows's own discipline. A
    row appearing here is proof restoration+isolation passed for that candidate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_rows(path: "str | Path") -> List[CausalDensityResultRow]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(CausalDensityResultRow.from_dict(json.loads(line)))
    return rows


def load_completed_candidate_ids(path: "str | Path", *, expected_capabilities: Sequence[str], expected_conditions: Sequence[str]) -> set:
    """A candidate_id is COMPLETE only if rows exist for EVERY (capability, condition) pair --
    mirrors the 32B/Stage-8-9-11 lineage's own load_completed_perturbation_rows discipline
    (exact-set-complete groups only; a partial group is excluded and re-run from scratch on
    resume, never trusted, never duplicated -- see evaluator.py's own docstring for why this
    is safe: append_rows is called only once per candidate, after full success).
    """
    expected = {(cap, cond) for cap in expected_capabilities for cond in expected_conditions}
    seen: Dict[str, set] = {}
    for row in load_rows(path):
        if row.is_base or row.candidate_id is None:
            continue
        seen.setdefault(row.candidate_id, set()).add((row.capability, row.condition))
    return {cid for cid, pairs in seen.items() if expected.issubset(pairs)}
