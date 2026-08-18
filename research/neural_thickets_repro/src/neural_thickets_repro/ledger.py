"""Resumable candidate ledger for a RandOpt population run.

Append-only JSONL, one record per write: candidate_id, seed, sigma, selection_score, rank,
status, runtime_seconds. Later records for the same candidate_id (e.g. "pending" ->
"done") supersede earlier ones on load, so an interrupted N=5000 run can resume without
re-evaluating already-completed candidates.

SCAFFOLD: exercised with synthetic records in tests/test_ledger.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


@dataclass
class CandidateRecord:
    candidate_id: int
    seed: int
    sigma: float
    selection_score: Optional[float]
    rank: Optional[int]
    status: str  # "pending" | "running" | "done" | "failed"
    runtime_seconds: Optional[float]
    # Which WorkerExtension perturb/restore mechanism produced this candidate's score --
    # "released_compat" (perturb_self_weights/restore_self_weights) or "fixed_base"
    # (apply_perturbation/reset_to_base_weights), see run_randopt_image_aware.py. Optional
    # with a None default so this ledger stays usable by generic, restoration-mode-agnostic
    # tests (tests/test_ledger.py) -- callers that DO have a mode (run_randopt_image_aware.py)
    # must always set it explicitly; a record from a real run left as None would mean the
    # score isn't attributable to either restoration mechanism, which should never happen.
    restoration_mode: Optional[str] = None
    # Scoped RandOpt (WACV) fields -- see scopes.py / scoped_perturbation.py /
    # run_scoped_randopt.py. All None for unscoped (run_randopt_image_aware.py) records.
    perturbation_scope: Optional[str] = None  # e.g. "vision_encoder", "lm_middle", ...
    perturbation_scale_mode: Optional[str] = None  # "raw_sigma" | "relative_l2"
    # In relative_l2 mode, the fixed requested r (see candidate_sampling.py -- candidates
    # vary by seed only, r is a run-level constant); None in raw_sigma mode, where `sigma`
    # above is the sampled raw sigma value directly, exactly as in unscoped runs. Never call
    # a sigma_candidate-drawn value "sigma" in relative_l2 output -- none is ever drawn.
    requested_relative_l2: Optional[float] = None
    scope_param_count: Optional[int] = None
    scope_base_l2_norm: Optional[float] = None
    actual_perturbation_l2: Optional[float] = None
    # Always "upstream_per_tensor_reseed" for scoped candidates this milestone -- recorded
    # explicitly (not left implicit) since preserving this exact noise convention, rather
    # than introducing a new one, was a hard requirement for this milestone.
    noise_semantics: Optional[str] = None
    # Thicket-measurement fields (see thicket_metrics.py) -- distinct from scope_param_count
    # (tensor count): this is d_m, the scalar PARAMETER ELEMENT count actually perturbed.
    scope_element_count: Optional[int] = None
    # The exact, explicitly-evaluated unperturbed base model's score on the SAME selection
    # subset this candidate was scored against (never a historical/hardcoded baseline --
    # see run_scoped_randopt.py's base_score computation). Duplicated onto every candidate
    # record (redundant with the run-level value) so each record is self-describing, same
    # convention as restoration_mode/noise_semantics above.
    base_score: Optional[float] = None
    delta_score: Optional[float] = None  # selection_score - base_score
    is_expert: Optional[bool] = None  # delta_score > 0 (strict -- see thicket_metrics.compute_delta_fields)
    is_tie: Optional[bool] = None  # delta_score == 0


class CandidateLedger:
    def __init__(self, path: "str | Path"):
        self.path = Path(path)

    def append(self, record: CandidateRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load_all(self) -> Dict[int, CandidateRecord]:
        """Load records, keeping only the LAST record per candidate_id."""
        records: Dict[int, CandidateRecord] = {}
        if not self.path.exists():
            return records
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records[data["candidate_id"]] = CandidateRecord(**data)
        return records

    def is_done(self, candidate_id: int) -> bool:
        rec = self.load_all().get(candidate_id)
        return rec is not None and rec.status == "done"

    def iter_pending(self, all_candidate_ids: Iterable[int]) -> Iterator[int]:
        done_ids = {cid for cid, rec in self.load_all().items() if rec.status == "done"}
        for cid in all_candidate_ids:
            if cid not in done_ids:
                yield cid
