"""Phase 6 candidate-loop DRIVER -- the outer population/checkpoint/resume orchestration
around evaluate_one_candidate_all_capabilities, mirroring the established pattern this
repository already uses throughout the Stage 6-11 lineage (most recently run_stage11_32b_rpc):
iterate the candidate population, skip already-COMPLETE candidates (schema.
load_completed_candidate_ids -- exact-capability-x-condition-set-complete only), call the
injected per-candidate evaluator, persist its returned rows in ONE atomic append, and record a
compact per-candidate outcome (never the full row contents) for progress reporting.

WHAT THIS MODULE DOES NOT DO: launch a real vllm/ray engine, load real datasets, or resolve a
real model snapshot -- those steps require a live GPU pod to develop and verify safely (see
reports/iclr_causal_density/artifact_audit.md), and are therefore NOT written here. This module
is fully CPU-testable via injected fakes (engine, evaluate_one_candidate, run_benchmark) exactly
like every other test in this package; the real pod-side entry point wires this driver to real
launch_stage6_engine/resolve_model_snapshot/benchmarks.runner calls once a GPU is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .candidates import PerturbationCandidate
from .schema import CausalDensityResultRow, append_rows, load_completed_candidate_ids


class CandidateFailedError(RuntimeError):
    """Raised by run_candidate_population_rpc when fail_fast=True and a candidate's evaluator
    raised -- wraps the original exception with the failing candidate_id for a readable trace.
    """


@dataclass(frozen=True)
class CandidateOutcome:
    candidate_id: str
    scope: str
    radius: float
    seed: int
    status: str  # "completed" | "failed" | "skipped_already_complete"
    n_rows: int
    error: Optional[str] = None


def run_candidate_population_rpc(
    candidates: Sequence[PerturbationCandidate], results_path: "str | Path", *,
    evaluate_one_candidate: Callable[..., List[CausalDensityResultRow]],
    expected_capabilities: Sequence[str], expected_conditions: Sequence[str],
    fail_fast: bool = True, progress_callback: Optional[Callable[[CandidateOutcome], None]] = None,
    capability_supports_condition: Optional[Callable[[str, str], bool]] = None,
) -> List[CandidateOutcome]:
    """Iterates `candidates` in order, skipping any whose candidate_id is already COMPLETE in
    `results_path` (exact expected_capabilities x expected_conditions coverage -- schema.
    load_completed_candidate_ids's own discipline, reused unmodified). For each remaining
    candidate, calls `evaluate_one_candidate(candidate)` (a closure the caller has already
    bound to engine/tokenizer/sampling_params/capability_data/etc. -- this driver knows nothing
    about those), appends the returned rows in ONE atomic call (transactional: a candidate's
    rows only ever reach disk after its own evaluator call has fully succeeded), and records a
    CandidateOutcome. `fail_fast=True` (default) re-raises the first failure immediately,
    wrapped in CandidateFailedError, after recording that candidate's outcome as "failed" --
    Phase 6 must never silently continue past a genuine integrity failure. `fail_fast=False`
    (used only for controlled, explicitly-requested "record every failure and keep going"
    diagnostics runs, never the real scientific pilot) records every failure and continues.

    `capability_supports_condition`: passed straight through to load_completed_candidate_ids --
    see its own docstring for the live resume-duplication bug this fixes. ADDITIVE, opt-in;
    `None` (the default) preserves this function's exact prior behavior.
    """
    results_path = Path(results_path)
    completed_ids = load_completed_candidate_ids(
        results_path, expected_capabilities=expected_capabilities, expected_conditions=expected_conditions,
        capability_supports_condition=capability_supports_condition,
    )

    outcomes: List[CandidateOutcome] = []
    for candidate in candidates:
        if candidate.candidate_id in completed_ids:
            outcome = CandidateOutcome(candidate.candidate_id, candidate.scope, candidate.radius, candidate.seed, "skipped_already_complete", 0)
            outcomes.append(outcome)
            if progress_callback is not None:
                progress_callback(outcome)
            continue

        try:
            rows = evaluate_one_candidate(candidate)
        except Exception as exc:  # noqa: BLE001 -- recorded explicitly, never silently swallowed
            outcome = CandidateOutcome(candidate.candidate_id, candidate.scope, candidate.radius, candidate.seed, "failed", 0, error=f"{type(exc).__name__}: {exc}")
            outcomes.append(outcome)
            if progress_callback is not None:
                progress_callback(outcome)
            if fail_fast:
                raise CandidateFailedError(f"Candidate {candidate.candidate_id!r} failed: {outcome.error}") from exc
            continue

        append_rows(results_path, rows)
        outcome = CandidateOutcome(candidate.candidate_id, candidate.scope, candidate.radius, candidate.seed, "completed", len(rows))
        outcomes.append(outcome)
        if progress_callback is not None:
            progress_callback(outcome)

    return outcomes


@dataclass(frozen=True)
class PopulationCompletionReport:
    expected_candidates: int
    completed_candidates: int
    skipped_already_complete: int
    newly_completed: int
    failed: int
    expected_rows_per_candidate: int
    total_rows_written: int
    run_complete: bool

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def summarize_population_run(candidates: Sequence[PerturbationCandidate], outcomes: Sequence[CandidateOutcome], *, expected_rows_per_candidate: int) -> PopulationCompletionReport:
    skipped = sum(1 for o in outcomes if o.status == "skipped_already_complete")
    newly = sum(1 for o in outcomes if o.status == "completed")
    failed = sum(1 for o in outcomes if o.status == "failed")
    total_rows = sum(o.n_rows for o in outcomes)
    completed_total = skipped + newly
    run_complete = completed_total == len(candidates) == len(outcomes) and failed == 0
    return PopulationCompletionReport(
        expected_candidates=len(candidates), completed_candidates=completed_total, skipped_already_complete=skipped,
        newly_completed=newly, failed=failed, expected_rows_per_candidate=expected_rows_per_candidate,
        total_rows_written=total_rows, run_complete=run_complete,
    )
