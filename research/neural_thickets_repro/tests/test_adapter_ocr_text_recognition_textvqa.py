"""Tests for adapters/ocr_text_recognition_textvqa.py -- synthetic textvqa-shaped rows, no
real dataset download / GPU / ray / vllm needed.
"""
import json
import sys
import types
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.ocr_text_recognition_textvqa import (
    TextVQAOCRBenchmark,
    TextVQAOCRGroundedBenchmark,
)
from neural_thickets_repro.benchmarks.base import Example


def _bench():
    return TextVQAOCRBenchmark()


def test_capability_and_name():
    bench = _bench()
    assert bench.capability == "ocr_text_recognition"
    assert bench.name == "textvqa_validation"


def test_end_to_end_vqa_soft_accuracy_against_10_answer_list():
    bench = _bench()
    answers = ["stop"] * 8 + ["halt"] * 2
    example = Example(example_id="1", prompt_input={"question": "What does the sign say?"}, target=answers)

    parsed = bench.parse_prediction("stop", example)
    assert parsed.parse_ok is True

    score = bench.score_example(parsed, example)
    assert score.score == pytest.approx(1.0)
    assert score.correct is True


def test_partial_match_scores_below_one():
    bench = _bench()
    answers = ["stop"] * 3 + ["halt"] * 7
    example = Example(example_id="1", target=answers)
    parsed = bench.parse_prediction("stop", example)
    score = bench.score_example(parsed, example)
    assert 0.0 < score.score < 1.0


def test_empty_generation_is_parse_failure():
    bench = _bench()
    example = Example(example_id="1", target=["stop"] * 10)
    parsed = bench.parse_prediction("   ", example)
    assert parsed.parse_ok is False
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.detail["reason"] == "parse_failure"


def test_does_not_reduce_target_to_a_single_answer():
    example = Example(example_id="1", target=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    assert isinstance(example.target, list)
    assert len(example.target) == 10


def test_aggregate_metrics_reports_primary_metric_and_parser_failure_rate():
    bench = _bench()
    examples = [Example(example_id=str(i), target=["stop"] * 10) for i in range(3)]
    parsed_ok = bench.parse_prediction("stop", examples[0])
    parsed_wrong = bench.parse_prediction("go", examples[1])
    parsed_empty = bench.parse_prediction("", examples[2])

    scores = [
        bench.score_example(parsed_ok, examples[0]),
        bench.score_example(parsed_wrong, examples[1]),
        bench.score_example(parsed_empty, examples[2]),
    ]
    metrics = bench.aggregate_metrics(scores)
    assert metrics["primary_metric"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert metrics["parser_failure_rate"] == pytest.approx(1 / 3)


def test_load_examples_maps_schema_fields(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    fake_rows = [
        {"question_id": "q1", "question": "What does the sign say?", "image": image,
         "image_id": "img1", "answers": ["stop"] * 10, "ocr_tokens": ["STOP"]},
    ]
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, split, revision: fake_rows
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="lmms-lab-encoder/textvqa", split="validation", revision=None))
    examples = bench.load_examples(cfg)

    assert len(examples) == 1
    assert examples[0].example_id == "q1"
    assert examples[0].image is image
    assert examples[0].target == ["stop"] * 10
    assert examples[0].prompt_input["question"] == "What does the sign say?"
    assert examples[0].metadata["ocr_grounded"] is True  # "stop" recoverable from ocr_tokens=["STOP"]


def test_load_examples_marks_non_ocr_grounded_examples_false(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    fake_rows = [
        {"question_id": "q1", "question": "how many wheels does this van have?", "image": image,
         "image_id": "img1", "answers": ["4"] * 10, "ocr_tokens": ["FORD", "TRANSIT"]},
    ]
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, split, revision: fake_rows
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="lmms-lab-encoder/textvqa", split="validation", revision=None))
    examples = bench.load_examples(cfg)

    assert examples[0].metadata["ocr_grounded"] is False


# ---------------------------------------------------------------------------------------
# TextVQAOCRGroundedBenchmark -- the new EXPERIMENTAL OCR-grounded subset (this repair pass)
# ---------------------------------------------------------------------------------------

def test_grounded_capability_and_name():
    bench = TextVQAOCRGroundedBenchmark(filter_ids_path="/does/not/matter.json")
    assert bench.capability == "ocr_text_recognition_grounded"
    assert bench.name == "textvqa_validation_ocr_grounded"


def test_grounded_known_caveats_documents_experimental_status():
    caveats = " ".join(TextVQAOCRGroundedBenchmark(filter_ids_path="/does/not/matter.json").known_caveats())
    assert "EXPERIMENTAL" in caveats
    assert "NOT an official TextVQA category" in caveats


def test_grounded_load_examples_missing_filter_gives_actionable_error(tmp_path):
    bench = TextVQAOCRGroundedBenchmark(filter_ids_path=tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="prepare_textvqa_ocr_filter"):
        bench.load_examples(SimpleNamespace(dataset=SimpleNamespace(source="lmms-lab-encoder/textvqa", split="validation", revision=None)))


def test_grounded_load_examples_narrows_to_persisted_ids(tiny_image_factory, monkeypatch, tmp_path):
    image = tiny_image_factory()
    fake_rows = [
        {"question_id": "keep1", "question": "what does the sign say?", "image": image, "image_id": "img1", "answers": ["stop"] * 10, "ocr_tokens": ["STOP"]},
        {"question_id": "drop1", "question": "how many wheels?", "image": image, "image_id": "img2", "answers": ["4"] * 10, "ocr_tokens": []},
    ]
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda source, split, revision: fake_rows
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    filter_path = tmp_path / "textvqa_ocr_grounded_ids.json"
    filter_path.write_text(json.dumps(["keep1"]))

    bench = TextVQAOCRGroundedBenchmark(filter_ids_path=filter_path)
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="lmms-lab-encoder/textvqa", split="validation", revision=None))
    examples = bench.load_examples(cfg)

    assert len(examples) == 1
    assert examples[0].example_id == "keep1"


def test_grounded_default_filter_path_resolves_under_artifact_dir(monkeypatch, tmp_path):
    import neural_thickets_repro.benchmarks.adapters.ocr_text_recognition_textvqa as m
    monkeypatch.setattr(m, "DEFAULT_FILTER_IDS_DIR", tmp_path)

    bench = TextVQAOCRGroundedBenchmark()
    assert bench._filter_ids_path == tmp_path / "textvqa_ocr_grounded_ids.json"
