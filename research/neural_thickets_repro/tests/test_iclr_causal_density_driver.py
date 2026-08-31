"""Tests for iclr_causal_density.driver -- the outer candidate-population checkpoint/resume
orchestration. Reinforces items 11/12 (resume/idempotency) and 12 (duplicate protection) at the
population-loop level (schema.py's own tests cover the row-level semantics these build on).
"""
from __future__ import annotations

import pytest

from neural_thickets_repro.iclr_causal_density.candidates import PerturbationCandidate
from neural_thickets_repro.iclr_causal_density.driver import CandidateFailedError, run_candidate_population_rpc, summarize_population_run
from neural_thickets_repro.iclr_causal_density.schema import CausalDensityResultRow, load_rows

_CAPS = ["visual_grounding", "counting"]
_CONDS = ["correct_image", "shuffled_image", "text_only"]


def _candidates(n=3):
    return [PerturbationCandidate(candidate_id=f"c{i}", scope="full_lm", radius=0.02, seed=i, seed_index=i) for i in range(n)]


def _fake_rows(candidate_id):
    return [
        CausalDensityResultRow(
            source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40, capability=cap, dataset_source="fake",
            subset_role="audit", sample_id="ex_0", original_image_id="img_0", evaluated_image_id="img_0", condition=cond,
            scope="full_lm", radius=0.02, seed=0, candidate_id=candidate_id, is_base=False, prediction="cat",
            normalized_prediction="cat", target="cat", per_example_score=1.0, aggregate_score=0.8, perturbation_norm=0.1,
            norm_verification_ok=True, scope_isolation_verification_ok=True, restoration_verification_ok=True,
        )
        for cap in _CAPS for cond in _CONDS
    ]


def test_all_candidates_run_when_results_file_empty(tmp_path):
    candidates = _candidates(3)
    calls = []

    def _evaluate(candidate):
        calls.append(candidate.candidate_id)
        return _fake_rows(candidate.candidate_id)

    outcomes = run_candidate_population_rpc(candidates, tmp_path / "results.jsonl", evaluate_one_candidate=_evaluate, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert calls == ["c0", "c1", "c2"]
    assert all(o.status == "completed" for o in outcomes)
    assert all(o.n_rows == 6 for o in outcomes)


def test_resume_skips_already_complete_candidates(tmp_path):
    """Item 11 at the population level: a candidate already fully complete on disk is skipped
    entirely -- its evaluator is never called on resume.
    """
    results_path = tmp_path / "results.jsonl"
    from neural_thickets_repro.iclr_causal_density.schema import append_rows

    append_rows(results_path, _fake_rows("c0"))  # c0 already complete from a prior run

    calls = []

    def _evaluate(candidate):
        calls.append(candidate.candidate_id)
        return _fake_rows(candidate.candidate_id)

    outcomes = run_candidate_population_rpc(_candidates(3), results_path, evaluate_one_candidate=_evaluate, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert "c0" not in calls
    assert calls == ["c1", "c2"]
    assert outcomes[0].status == "skipped_already_complete"


def test_resume_never_duplicates_rows_for_a_completed_candidate(tmp_path):
    results_path = tmp_path / "results.jsonl"
    from neural_thickets_repro.iclr_causal_density.schema import append_rows

    append_rows(results_path, _fake_rows("c0"))

    def _evaluate(candidate):
        return _fake_rows(candidate.candidate_id)

    run_candidate_population_rpc(_candidates(1), results_path, evaluate_one_candidate=_evaluate, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    rows_for_c0 = [r for r in load_rows(results_path) if r.candidate_id == "c0"]
    assert len(rows_for_c0) == 6  # exactly one complete group -- never re-appended


def test_fail_fast_raises_and_records_the_failure(tmp_path):
    def _evaluate(candidate):
        if candidate.candidate_id == "c1":
            raise RuntimeError("restoration failed")
        return _fake_rows(candidate.candidate_id)

    with pytest.raises(CandidateFailedError, match="restoration failed"):
        run_candidate_population_rpc(_candidates(3), tmp_path / "results.jsonl", evaluate_one_candidate=_evaluate, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    # c0 succeeded and was persisted before c1 failed
    assert [r.candidate_id for r in load_rows(tmp_path / "results.jsonl")] == ["c0"] * 6


def test_a_failed_candidate_writes_zero_rows():
    def _evaluate(candidate):
        raise RuntimeError("boom")

    with pytest.raises(CandidateFailedError):
        run_candidate_population_rpc(_candidates(1), "unused_path_never_written.jsonl", evaluate_one_candidate=_evaluate, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    import os

    assert not os.path.exists("unused_path_never_written.jsonl")


def test_summarize_population_run_reports_complete_when_all_candidates_done(tmp_path):
    candidates = _candidates(3)
    outcomes = run_candidate_population_rpc(candidates, tmp_path / "results.jsonl", evaluate_one_candidate=lambda c: _fake_rows(c.candidate_id), expected_capabilities=_CAPS, expected_conditions=_CONDS)
    report = summarize_population_run(candidates, outcomes, expected_rows_per_candidate=6)
    assert report.run_complete is True
    assert report.expected_candidates == 3
    assert report.completed_candidates == 3
    assert report.total_rows_written == 18


def test_summarize_population_run_reports_incomplete_when_a_failure_occurred():
    """Item 17 at the population level: partial progress (one failure among several
    candidates) must never be reported as a complete run.
    """
    from neural_thickets_repro.iclr_causal_density.driver import CandidateOutcome

    candidates = _candidates(3)
    outcomes = [
        CandidateOutcome("c0", "full_lm", 0.02, 0, "completed", 6),
        CandidateOutcome("c1", "full_lm", 0.02, 1, "failed", 0, error="boom"),
        CandidateOutcome("c2", "full_lm", 0.02, 2, "completed", 6),
    ]
    report = summarize_population_run(candidates, outcomes, expected_rows_per_candidate=6)
    assert report.run_complete is False
    assert report.failed == 1
