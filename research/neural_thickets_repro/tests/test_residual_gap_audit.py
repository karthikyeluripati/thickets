"""Unit tests for the residual-gap failure classification logic. Uses a minimal fake
handler (not the real GQAHandler, which lives in the external clone and isn't available
locally) that mimics just the two methods _classify() actually calls, so this runs anywhere.
"""
import json

from neural_thickets_repro.diagnostics.residual_gap_audit import _classify, audit


class _FakeHandler:
    """Mimics GQAHandler._normalize_answer / _whole_word_search closely enough to test
    _classify()'s branching -- lowercase+strip normalization, substring-based word search.
    """

    def _normalize_answer(self, text):
        return text.strip().lower()

    def _whole_word_search(self, text, gt):
        return gt in text


def _rec(raw, gt, extracted=""):
    return {"raw_prediction": raw, "reference_answer": gt, "normalized_prediction": extracted}


def test_empty_response_classified_as_degenerate():
    h = _FakeHandler()
    assert _classify(h, _rec("", "yes")) == "empty_or_degenerate_response"
    assert _classify(h, _rec("   ", "yes")) == "empty_or_degenerate_response"


def test_repetitive_response_classified_as_degenerate():
    h = _FakeHandler()
    rec = _rec("the the the the the the the", "yes")
    assert _classify(h, rec) == "empty_or_degenerate_response"


def test_gt_present_in_raw_but_not_scored_is_extraction_failure():
    h = _FakeHandler()
    rec = _rec("Looking closely, I can see a drape on the window.", "drape", extracted="curtain")
    assert _classify(h, rec) == "extraction_or_scoring_failure"


def test_yes_no_disagreement_classified_separately():
    h = _FakeHandler()
    rec = _rec("No, it is not overcast.", "yes", extracted="no")
    assert _classify(h, rec) == "model_wrong_yes_no"


def test_generic_content_disagreement_falls_through():
    h = _FakeHandler()
    rec = _rec("It looks like a bicycle.", "car", extracted="bicycle")
    assert _classify(h, rec) == "model_wrong_other"


def test_audit_reports_consistent_counts(tmp_path):
    handler_records = [
        {"example_id": "1", "question": "q1", "reference_answer": "yes",
         "raw_prediction": "", "normalized_prediction": "", "correct_march_scoring": False},
        {"example_id": "2", "question": "q2", "reference_answer": "car",
         "raw_prediction": "It is a bicycle.", "normalized_prediction": "bicycle", "correct_march_scoring": False},
        {"example_id": "3", "question": "q3", "reference_answer": "car",
         "raw_prediction": "car", "normalized_prediction": "car", "correct_march_scoring": True},
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    with predictions_path.open("w") as f:
        for rec in handler_records:
            f.write(json.dumps(rec) + "\n")

    # audit() calls load_gqa_handler() internally (needs the external clone) -- this test
    # only exercises the file I/O / sampling / counting wiring around _classify(), so it's
    # skipped rather than run against a fake here; _classify() itself is covered above.
    import pytest

    pytest.importorskip("data_handlers", reason="needs external/RandOpt clone, not available locally")
    report = audit(predictions_path, sample_size=10, seed=0, scoring_key="correct_march_scoring")
    assert report["n_total_examples"] == 3
    assert report["n_incorrect_total"] == 2
