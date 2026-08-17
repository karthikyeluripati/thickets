"""Fidelity-gate tests. run_fidelity_gate() needs load_gqa_handler() (the external clone,
not available locally) -- skipped here, but structured to run for real on the pod, which
does have the clone.
"""
import json

import pytest

pytest.importorskip("data_handlers", reason="needs external/RandOpt clone, not available locally")

from neural_thickets_repro.diagnostics.vllm_version_control.fidelity_gate import run_fidelity_gate  # noqa: E402


def _write_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_gate_passes_when_predictions_match_exactly(tmp_path):
    fixed = {"seed": 42, "n": 2, "examples": [
        {"question_id": "1", "image_id": "imgA", "image_path": "a.jpg", "formatted_prompt_text": "q1", "reference_answer": "yes"},
        {"question_id": "2", "image_id": "imgB", "image_path": "b.jpg", "formatted_prompt_text": "q2", "reference_answer": "car"},
    ]}
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed))

    new_path = tmp_path / "new.jsonl"
    _write_jsonl(new_path, [
        {"question_id": "1", "reference_answer": "yes", "raw_prediction": "yes"},
        {"question_id": "2", "reference_answer": "car", "raw_prediction": "car"},
    ])

    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(baseline_path, [
        {"example_id": "1", "reference_answer": "yes", "raw_prediction": "yes", "correct_march_scoring": True},
        {"example_id": "2", "reference_answer": "car", "raw_prediction": "car", "correct_march_scoring": True},
    ])

    report = run_fidelity_gate(fixed_path, new_path, baseline_path)
    assert report["gate_result"] == "PASS"
    assert report["ids_matched"] == 2
    assert report["raw_predictions_identical_count"] == 2
    assert report["correctness_agreement_rate"] == 1.0


def test_gate_fails_when_new_helper_produces_empty_output(tmp_path):
    """Reproduces the reported failure mode's downstream symptom: the harness bug meant
    the new helper's generation was broken, which would show up here as systematic
    disagreement even if it hadn't crashed outright.
    """
    fixed = {"seed": 42, "n": 2, "examples": [
        {"question_id": "1", "image_id": "imgA", "image_path": "a.jpg", "formatted_prompt_text": "q1", "reference_answer": "yes"},
        {"question_id": "2", "image_id": "imgB", "image_path": "b.jpg", "formatted_prompt_text": "q2", "reference_answer": "car"},
    ]}
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed))

    new_path = tmp_path / "new.jsonl"
    _write_jsonl(new_path, [
        {"question_id": "1", "reference_answer": "yes", "raw_prediction": ""},
        {"question_id": "2", "reference_answer": "car", "raw_prediction": ""},
    ])

    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(baseline_path, [
        {"example_id": "1", "reference_answer": "yes", "raw_prediction": "yes", "correct_march_scoring": True},
        {"example_id": "2", "reference_answer": "car", "raw_prediction": "car", "correct_march_scoring": True},
    ])

    report = run_fidelity_gate(fixed_path, new_path, baseline_path)
    assert report["gate_result"] == "FAIL"
    assert report["raw_predictions_identical_count"] == 0
    assert report["n_disagreements"] == 2


def test_gate_reports_missing_ids(tmp_path):
    fixed = {"seed": 42, "n": 2, "examples": [
        {"question_id": "1", "image_id": "imgA", "image_path": "a.jpg", "formatted_prompt_text": "q1", "reference_answer": "yes"},
        {"question_id": "2", "image_id": "imgB", "image_path": "b.jpg", "formatted_prompt_text": "q2", "reference_answer": "car"},
    ]}
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed))

    new_path = tmp_path / "new.jsonl"
    _write_jsonl(new_path, [{"question_id": "1", "reference_answer": "yes", "raw_prediction": "yes"}])  # missing "2"

    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(baseline_path, [
        {"example_id": "1", "reference_answer": "yes", "raw_prediction": "yes", "correct_march_scoring": True},
        {"example_id": "2", "reference_answer": "car", "raw_prediction": "car", "correct_march_scoring": True},
    ])

    report = run_fidelity_gate(fixed_path, new_path, baseline_path)
    assert report["gate_result"] == "FAIL"
    assert report["ids_missing_from_new_helper"] == ["2"]
