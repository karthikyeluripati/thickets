from neural_thickets_repro.ledger import CandidateLedger, CandidateRecord


def _rec(cid, status, score=None):
    return CandidateRecord(
        candidate_id=cid, seed=1000 + cid, sigma=0.01, selection_score=score,
        rank=None, status=status, runtime_seconds=1.23,
    )


def test_append_and_load_roundtrip(tmp_path):
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec(0, "done", 0.5))
    ledger.append(_rec(1, "done", 0.7))

    records = ledger.load_all()
    assert set(records) == {0, 1}
    assert records[0].status == "done"
    assert records[1].selection_score == 0.7


def test_later_record_supersedes_earlier_for_same_candidate(tmp_path):
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec(0, "pending"))
    ledger.append(_rec(0, "running"))
    ledger.append(_rec(0, "done", 0.9))

    records = ledger.load_all()
    assert records[0].status == "done"
    assert records[0].selection_score == 0.9


def test_is_done(tmp_path):
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec(0, "done"))
    ledger.append(_rec(1, "failed"))
    assert ledger.is_done(0) is True
    assert ledger.is_done(1) is False
    assert ledger.is_done(2) is False  # never appended


def test_resume_skips_completed_candidates_after_simulated_interruption(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = CandidateLedger(path)
    for cid in range(5):
        ledger.append(_rec(cid, "done" if cid < 3 else "failed"))

    # Simulate a fresh process reopening the same ledger file after an interruption.
    resumed_ledger = CandidateLedger(path)
    pending = list(resumed_ledger.iter_pending(range(5)))
    assert pending == [3, 4]


def test_empty_ledger_all_pending(tmp_path):
    ledger = CandidateLedger(tmp_path / "does_not_exist_yet.jsonl")
    assert list(ledger.iter_pending(range(3))) == [0, 1, 2]


def test_restoration_mode_defaults_to_none_for_generic_records():
    # Generic ledger usage (this file's own tests) doesn't need to know about restoration
    # modes -- only run_randopt_image_aware.py's real usage sets it explicitly.
    rec = _rec(0, "done", 0.5)
    assert rec.restoration_mode is None


def test_restoration_mode_round_trips_through_append_and_load(tmp_path):
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")
    ledger.append(CandidateRecord(
        candidate_id=0, seed=111, sigma=0.01, selection_score=0.5,
        rank=None, status="done", runtime_seconds=1.0, restoration_mode="fixed_base",
    ))

    records = ledger.load_all()

    assert records[0].restoration_mode == "fixed_base"


def test_scoped_fields_default_to_none_for_generic_records():
    rec = _rec(0, "done", 0.5)
    assert rec.perturbation_scope is None
    assert rec.perturbation_scale_mode is None
    assert rec.requested_relative_l2 is None
    assert rec.scope_param_count is None
    assert rec.scope_base_l2_norm is None
    assert rec.actual_perturbation_l2 is None
    assert rec.noise_semantics is None


def test_scoped_fields_round_trip_through_append_and_load(tmp_path):
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")
    ledger.append(CandidateRecord(
        candidate_id=0, seed=111, sigma=0.0023, selection_score=0.5,
        rank=None, status="done", runtime_seconds=1.0, restoration_mode="fixed_base",
        perturbation_scope="vision_encoder", perturbation_scale_mode="relative_l2",
        requested_relative_l2=0.02, scope_param_count=42, scope_base_l2_norm=12.5,
        actual_perturbation_l2=0.281, noise_semantics="upstream_per_tensor_reseed",
    ))

    records = ledger.load_all()

    rec = records[0]
    assert rec.perturbation_scope == "vision_encoder"
    assert rec.perturbation_scale_mode == "relative_l2"
    assert rec.requested_relative_l2 == 0.02
    assert rec.scope_param_count == 42
    assert rec.scope_base_l2_norm == 12.5
    assert rec.actual_perturbation_l2 == 0.281
    assert rec.noise_semantics == "upstream_per_tensor_reseed"
