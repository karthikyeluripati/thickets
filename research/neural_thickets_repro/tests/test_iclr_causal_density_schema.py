"""Tests for iclr_causal_density.schema -- atomic checkpointing (item 14), resume/idempotency
(item 15), duplicate protection (item 16), partial-run detection (item 17). CPU-only.
"""
from __future__ import annotations

from neural_thickets_repro.iclr_causal_density.schema import CausalDensityResultRow, append_rows, load_completed_candidate_ids, load_rows


def _row(candidate_id, capability, condition, **overrides):
    base = dict(
        source_commit="abc123", run_id="run1", model_name="Qwen/Qwen2.5-VL-7B-Instruct", model_revision="a" * 40,
        capability=capability, dataset_source="fake_dataset", subset_role="audit", sample_id="ex_0",
        original_image_id="img_0", evaluated_image_id="img_0", condition=condition, scope="full_lm", radius=0.02,
        seed=42, candidate_id=candidate_id, is_base=False, prediction="cat", normalized_prediction="cat", target="cat",
        per_example_score=1.0, aggregate_score=0.8, perturbation_norm=0.1, norm_verification_ok=True,
        scope_isolation_verification_ok=True, restoration_verification_ok=True,
    )
    base.update(overrides)
    return CausalDensityResultRow(**base)


_CAPS = ["visual_grounding", "counting", "ocr_text_recognition", "spatial_reasoning", "relational_reasoning"]
_CONDS = ["correct_image", "shuffled_image", "text_only"]


def _complete_candidate_rows(candidate_id):
    return [_row(candidate_id, cap, cond) for cap in _CAPS for cond in _CONDS]


def test_atomic_checkpointing_appends_all_rows_for_one_candidate(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = _complete_candidate_rows("c1")
    append_rows(path, rows)
    loaded = load_rows(path)
    assert len(loaded) == len(rows)
    assert path.exists()


def test_resume_is_idempotent_completed_candidate_is_skippable(tmp_path):
    path = tmp_path / "results.jsonl"
    append_rows(path, _complete_candidate_rows("c1"))
    completed = load_completed_candidate_ids(path, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert completed == {"c1"}


def test_duplicate_protection_two_full_appends_are_visible_not_silently_merged(tmp_path):
    """append_rows never deduplicates on write (that responsibility lives in the CALLER, which
    must never call it twice for the same candidate_id -- see evaluator.py's own transactional
    discipline: append_rows is invoked exactly once per successfully-completed candidate). This
    test proves load_completed_candidate_ids still correctly reports the candidate as complete
    even if duplicate rows exist (never crashes, never double-counts it as two candidates).
    """
    path = tmp_path / "results.jsonl"
    append_rows(path, _complete_candidate_rows("c1"))
    append_rows(path, _complete_candidate_rows("c1"))  # simulates a caller bug / crash-recovery double-write
    completed = load_completed_candidate_ids(path, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert completed == {"c1"}  # still exactly one logical candidate, not two
    all_rows = load_rows(path)
    assert len(all_rows) == 2 * len(_CAPS) * len(_CONDS)  # both physical writes ARE visible on disk (auditable), never silently dropped


def test_partial_run_is_never_reported_as_complete(tmp_path):
    """Item 17: a candidate missing even ONE (capability, condition) pair must not appear in
    load_completed_candidate_ids.
    """
    path = tmp_path / "results.jsonl"
    rows = _complete_candidate_rows("c1")
    partial = rows[:-1]  # drop the last (capability, condition) row
    append_rows(path, partial)
    completed = load_completed_candidate_ids(path, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert completed == set()


def test_partial_run_with_wrong_capability_set_not_reported_complete(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [_row("c1", cap, cond) for cap in _CAPS[:-1] for cond in _CONDS]  # missing one capability entirely
    append_rows(path, rows)
    completed = load_completed_candidate_ids(path, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert completed == set()


def test_base_rows_never_counted_as_candidate_completion(tmp_path):
    path = tmp_path / "results.jsonl"
    base_rows = [_row(None, cap, cond, is_base=True, scope=None, radius=None, seed=None) for cap in _CAPS for cond in _CONDS]
    append_rows(path, base_rows)
    completed = load_completed_candidate_ids(path, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert completed == set()


def test_mixed_complete_and_incomplete_candidates(tmp_path):
    path = tmp_path / "results.jsonl"
    append_rows(path, _complete_candidate_rows("c1"))
    append_rows(path, _complete_candidate_rows("c2")[:-1])  # c2 is incomplete
    completed = load_completed_candidate_ids(path, expected_capabilities=_CAPS, expected_conditions=_CONDS)
    assert completed == {"c1"}


def test_load_rows_on_missing_file_returns_empty():
    from pathlib import Path

    assert load_rows(Path("/nonexistent/path/results.jsonl")) == []


def test_row_roundtrip_to_dict_from_dict():
    row = _row("c1", "counting", "correct_image")
    d = row.to_dict()
    reconstructed = CausalDensityResultRow.from_dict(d)
    assert reconstructed == row
