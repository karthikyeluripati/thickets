"""Tests for benchmarks/runner.py -- fake-llm-driven, no GPU/ray/vllm needed. The fake llm
matches vllm.LLM's own .generate(requests, sampling_params, use_tqdm) -> List[output] shape
(output.outputs[0].text), same fake-object convention used throughout this project's tests.
"""
import json
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.benchmarks.runner import MissingImageError, run_benchmark, write_predictions_jsonl


def _fake_output(text):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def _fake_llm(texts):
    calls = []

    def _generate(requests, sampling_params, use_tqdm=True):
        calls.append(requests)
        return [_fake_output(t) for t in texts]

    return SimpleNamespace(generate=_generate, calls=calls)


def _fake_tokenizer():
    return SimpleNamespace(apply_chat_template=lambda messages, add_generation_prompt, tokenize: "TEXT")


def test_image_propagation_reaches_the_generate_call(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = [Example(example_id="1", image=image, prompt_input={"question": "q"}, target="42")]
    llm = _fake_llm(["42"])

    run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)

    requests = llm.calls[0]
    assert requests[0]["multi_modal_data"]["image"] is image


def test_missing_image_raises_by_default(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    examples = [Example(example_id="1", image=None, prompt_input={"question": "q"}, target="42")]
    llm = _fake_llm(["42"])

    with pytest.raises(MissingImageError, match="allow_missing_image=False"):
        run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)


def test_missing_image_allowed_when_flag_set(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    examples = [Example(example_id="1", image=None, prompt_input={"question": "q"}, target="42")]
    llm = _fake_llm(["42"])

    result = run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None, allow_missing_image=True)
    assert result.per_example[0].score.score == 1.0


def test_end_to_end_scoring_and_aggregation(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = [
        Example(example_id="1", image=image, prompt_input={"question": "q1"}, target="42"),
        Example(example_id="2", image=image, prompt_input={"question": "q2"}, target="7"),
    ]
    llm = _fake_llm(["42", "wrong"])

    result = run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)

    assert result.per_example[0].score.score == 1.0
    assert result.per_example[1].score.score == 0.0
    assert result.aggregate_metrics["primary_metric"] == pytest.approx(0.5)
    assert result.aggregate_metrics["parser_failure_rate"] == 0.0


def test_parser_failure_counted_and_reported(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = [Example(example_id="1", image=image, prompt_input={"question": "q"}, target="42")]
    llm = _fake_llm([""])  # empty generation -> parse_ok=False for the fake benchmark

    result = run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)
    assert result.aggregate_metrics["parser_failure_rate"] == 1.0


def test_run_result_generation_hash_deterministic(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = [Example(example_id="1", image=image, prompt_input={"question": "q"}, target="42")]
    llm = _fake_llm(["42"])

    result_a = run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)
    result_b = run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)
    assert result_a.generation_hash() == result_b.generation_hash()
    assert result_a.parsed_prediction_hash() == result_b.parsed_prediction_hash()


def test_generation_hash_differs_when_raw_text_differs_but_parsed_can_still_match():
    from neural_thickets_repro.benchmarks.runner import PerExampleResult, RunResult
    from neural_thickets_repro.benchmarks.base import ExampleScore, ParsedPrediction

    result_a = RunResult(per_example=[PerExampleResult("1", "img", "Answer: 42", ParsedPrediction(42, True), ExampleScore(1.0))], aggregate_metrics={})
    result_b = RunResult(per_example=[PerExampleResult("1", "img", "42", ParsedPrediction(42, True), ExampleScore(1.0))], aggregate_metrics={})

    assert result_a.generation_hash() != result_b.generation_hash()
    assert result_a.parsed_prediction_hash() == result_b.parsed_prediction_hash()


def test_aggregate_metrics_missing_primary_metric_raises(fake_capability_benchmark_factory, tiny_image_factory, monkeypatch):
    bench = fake_capability_benchmark_factory()
    monkeypatch.setattr(bench, "aggregate_metrics", lambda scores: {"accuracy": 1.0})
    image = tiny_image_factory()
    examples = [Example(example_id="1", image=image, prompt_input={"question": "q"}, target="42")]
    llm = _fake_llm(["42"])

    with pytest.raises(ValueError, match="primary_metric"):
        run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)


def test_write_predictions_jsonl_contains_required_fields(fake_capability_benchmark_factory, tiny_image_factory, tmp_path):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = [Example(example_id="1", image=image, image_ref="img1.png", prompt_input={"question": "q"}, target="42", metadata={"m": 1})]
    llm = _fake_llm(["42"])
    result = run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)

    out_path = tmp_path / "predictions.jsonl"
    write_predictions_jsonl(result, examples, out_path)

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    for field in ("example_id", "image_ref", "query", "target", "raw_generation", "parsed_prediction", "parse_ok", "per_example_score", "correct", "metadata"):
        assert field in record
    assert record["example_id"] == "1"
    assert record["raw_generation"] == "42"
    assert record["per_example_score"] == 1.0
