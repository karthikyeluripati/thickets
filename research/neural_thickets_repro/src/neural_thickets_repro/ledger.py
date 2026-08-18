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
