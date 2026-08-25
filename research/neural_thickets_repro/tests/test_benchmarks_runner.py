"""Tests for benchmarks/runner.py -- fake-llm-driven, no GPU/ray/vllm needed. The fake llm
matches vllm.LLM's own .generate(requests, sampling_params, use_tqdm) -> List[output] shape
(output.outputs[0].text), same fake-object convention used throughout this project's tests.
"""
import json
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.benchmarks.runner import (
    MissingImageError,
    bounded_disagreement_examples,
    prediction_disagreement_rate,
    run_benchmark,
    write_predictions_jsonl,
    write_repeat_comparison_jsonl,
)


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


# ---------------------------------------------------------------------------------------
# prediction_disagreement_rate (baseline-characterization prep, this session) -- a
# finer-grained companion to parsed_prediction_hash()'s whole-run boolean equality.
# ---------------------------------------------------------------------------------------

def _result(pairs):
    """pairs: [(example_id, parsed_value), ...] -- builds a minimal RunResult for these tests."""
    from neural_thickets_repro.benchmarks.runner import PerExampleResult, RunResult
    from neural_thickets_repro.benchmarks.base import ExampleScore, ParsedPrediction

    per_example = [
        PerExampleResult(example_id, "img", "raw", ParsedPrediction(parsed_value, True), ExampleScore(1.0))
        for example_id, parsed_value in pairs
    ]
    return RunResult(per_example=per_example, aggregate_metrics={})


def test_prediction_disagreement_rate_zero_when_identical():
    a = _result([("1", "left"), ("2", "right")])
    b = _result([("1", "left"), ("2", "right")])
    assert prediction_disagreement_rate(a, b) == pytest.approx(0.0)


def test_prediction_disagreement_rate_full_when_all_differ():
    a = _result([("1", "left"), ("2", "right")])
    b = _result([("1", "up"), ("2", "down")])
    assert prediction_disagreement_rate(a, b) == pytest.approx(1.0)


def test_prediction_disagreement_rate_partial():
    a = _result([("1", "left"), ("2", "right"), ("3", "up"), ("4", "down")])
    b = _result([("1", "left"), ("2", "WRONG"), ("3", "up"), ("4", "down")])
    assert prediction_disagreement_rate(a, b) == pytest.approx(0.25)


def test_prediction_disagreement_rate_uses_only_common_example_ids():
    a = _result([("1", "left"), ("2", "right"), ("extra_in_a", "x")])
    b = _result([("1", "left"), ("2", "WRONG"), ("extra_in_b", "y")])
    assert prediction_disagreement_rate(a, b) == pytest.approx(0.5)  # 1 disagreement / 2 common ids


def test_prediction_disagreement_rate_raises_on_no_common_ids():
    a = _result([("1", "left")])
    b = _result([("2", "right")])
    with pytest.raises(ValueError, match="no example_ids"):
        prediction_disagreement_rate(a, b)


# ---------------------------------------------------------------------------------------
# write_repeat_comparison_jsonl / bounded_disagreement_examples (baseline-characterization
# repeatability, this repair pass)
# ---------------------------------------------------------------------------------------

def test_write_repeat_comparison_jsonl_contains_base_and_repeat_fields(tmp_path):
    a = _result([("1", "left")])
    b = _result([("1", "right")])

    out_path = tmp_path / "repeat_predictions.jsonl"
    write_repeat_comparison_jsonl(a, b, out_path)

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["example_id"] == "1"
    assert record["base_raw_generation"] == "raw"
    assert record["base_parsed_prediction"] == "left"
    assert record["repeat_parsed_prediction"] == "right"
    assert record["base_score"] == 1.0
    assert record["repeat_score"] == 1.0
    assert record["parsed_prediction_agrees"] is False


def test_write_repeat_comparison_jsonl_computes_box_to_box_iou_when_present(tmp_path):
    from neural_thickets_repro.benchmarks.runner import PerExampleResult, RunResult
    from neural_thickets_repro.benchmarks.base import ExampleScore, ParsedPrediction

    a = RunResult(per_example=[PerExampleResult("1", "img", "raw", ParsedPrediction((0, 0, 100, 100), True), ExampleScore(1.0, detail={"canonical_prediction_box": [0, 0, 100, 100]}))], aggregate_metrics={})
    b = RunResult(per_example=[PerExampleResult("1", "img", "raw", ParsedPrediction((0, 0, 100, 100), True), ExampleScore(1.0, detail={"canonical_prediction_box": [0, 0, 100, 100]}))], aggregate_metrics={})

    out_path = tmp_path / "repeat_predictions.jsonl"
    write_repeat_comparison_jsonl(a, b, out_path)
    record = json.loads(out_path.read_text().strip())
    assert record["box_to_box_iou"] == pytest.approx(1.0)


def test_write_repeat_comparison_jsonl_box_to_box_iou_none_when_no_boxes(tmp_path):
    a = _result([("1", "left")])
    b = _result([("1", "left")])
    out_path = tmp_path / "repeat_predictions.jsonl"
    write_repeat_comparison_jsonl(a, b, out_path)
    record = json.loads(out_path.read_text().strip())
    assert record["box_to_box_iou"] is None


def test_write_repeat_comparison_jsonl_only_common_ids(tmp_path):
    a = _result([("1", "left"), ("extra_a", "x")])
    b = _result([("1", "left"), ("extra_b", "y")])
    out_path = tmp_path / "repeat_predictions.jsonl"
    write_repeat_comparison_jsonl(a, b, out_path)
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["example_id"] == "1"


def test_bounded_disagreement_examples_returns_only_disagreements():
    a = _result([("1", "left"), ("2", "right"), ("3", "up")])
    b = _result([("1", "left"), ("2", "WRONG"), ("3", "up")])
    examples = bounded_disagreement_examples(a, b)
    assert len(examples) == 1
    assert examples[0]["example_id"] == "2"
    assert examples[0]["base_parsed_prediction"] == "right"
    assert examples[0]["repeat_parsed_prediction"] == "WRONG"


def test_bounded_disagreement_examples_empty_when_all_agree():
    a = _result([("1", "left")])
    b = _result([("1", "left")])
    assert bounded_disagreement_examples(a, b) == []


def test_bounded_disagreement_examples_respects_limit():
    a = _result([(str(i), "left") for i in range(20)])
    b = _result([(str(i), "different") for i in range(20)])
    examples = bounded_disagreement_examples(a, b, limit=5)
    assert len(examples) == 5


# =================================================================================================
# max_requests_per_generate / rss_checkpoint (this repair pass -- Stage-8 driver-RSS OOM audit).
# Default (None) must remain byte-identical to the prior single-call behavior for every existing
# caller (Stage 6/7 execution unchanged); batching is an execution-only parameter that must never
# change ordering, raw generations, parsed predictions, scores, or hashes.
# =================================================================================================


def _fake_llm_ordered(all_texts):
    """Returns exactly len(requests) outputs per generate() call, sliced in call order from
    `all_texts` -- correctly models both a single unbatched call and several sequential
    microbatches, since real vLLM's generate() returns one output per request in that SAME call.
    """
    calls = []
    state = {"pos": 0}

    def _generate(requests, sampling_params, use_tqdm=True):
        calls.append(list(requests))
        n = len(requests)
        texts = all_texts[state["pos"]:state["pos"] + n]
        state["pos"] += n
        return [_fake_output(t) for t in texts]

    return SimpleNamespace(generate=_generate, calls=calls)


def _make_n_examples(n, image):
    return [
        Example(example_id=str(i), image=image, prompt_input={"question": f"q{i}"}, target=str(i))
        for i in range(n)
    ]


def test_default_max_requests_per_generate_is_none_and_makes_exactly_one_generate_call(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = _make_n_examples(7, image)
    llm = _fake_llm_ordered([str(i) for i in range(7)])

    run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)

    assert len(llm.calls) == 1
    assert len(llm.calls[0]) == 7


def test_n50_batch_size_10_yields_exactly_5_ordered_batches(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = _make_n_examples(50, image)
    llm = _fake_llm_ordered([str(i) for i in range(50)])

    run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None, max_requests_per_generate=10)

    assert len(llm.calls) == 5
    assert [len(c) for c in llm.calls] == [10, 10, 10, 10, 10]


def test_batching_drops_no_examples_and_duplicates_none(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = _make_n_examples(23, image)  # not evenly divisible by the batch size
    llm = _fake_llm_ordered([str(i) for i in range(23)])

    result = run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None, max_requests_per_generate=10)

    assert len(llm.calls) == 3
    assert [len(c) for c in llm.calls] == [10, 10, 3]
    ids_seen = [r.example_id for r in result.per_example]
    assert ids_seen == [str(i) for i in range(23)]  # original order, no drops/duplicates


def test_unbatched_and_batched_outputs_are_exactly_equivalent(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    # A realistic mix of correct/incorrect/empty generations so score/parsed-prediction/hash
    # equivalence is actually exercised, not just a trivially all-correct run.
    n = 17
    examples = _make_n_examples(n, image)
    texts = [str(i) if i % 3 != 0 else "" for i in range(n)]  # every 3rd generation is empty (parse failure)

    llm_unbatched = _fake_llm_ordered(list(texts))
    result_unbatched = run_benchmark(bench, examples, llm_unbatched, _fake_tokenizer(), sampling_params=None, max_requests_per_generate=None)

    llm_batched = _fake_llm_ordered(list(texts))
    result_batched = run_benchmark(bench, examples, llm_batched, _fake_tokenizer(), sampling_params=None, max_requests_per_generate=4)

    assert [r.example_id for r in result_unbatched.per_example] == [r.example_id for r in result_batched.per_example]
    assert [r.raw_generation for r in result_unbatched.per_example] == [r.raw_generation for r in result_batched.per_example]
    assert [r.parsed.parsed for r in result_unbatched.per_example] == [r.parsed.parsed for r in result_batched.per_example]
    assert [r.score.score for r in result_unbatched.per_example] == [r.score.score for r in result_batched.per_example]
    assert result_unbatched.aggregate_metrics["primary_metric"] == result_batched.aggregate_metrics["primary_metric"]
    assert result_unbatched.generation_hash() == result_batched.generation_hash()
    assert result_unbatched.parsed_prediction_hash() == result_batched.parsed_prediction_hash()


def test_max_requests_per_generate_zero_or_negative_raises(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = _make_n_examples(3, image)
    llm = _fake_llm_ordered(["a", "b", "c"])

    with pytest.raises(ValueError):
        run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None, max_requests_per_generate=0)


def test_rss_checkpoint_default_none_never_called(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = _make_n_examples(2, image)
    llm = _fake_llm_ordered(["1", "2"])
    # No rss_checkpoint passed -- must not raise even though the internal _checkpoint() calls happen.
    run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None)


def test_rss_checkpoint_receives_the_three_expected_labels_in_order(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    examples = _make_n_examples(2, image)
    llm = _fake_llm_ordered(["1", "2"])
    labels = []

    run_benchmark(bench, examples, llm, _fake_tokenizer(), sampling_params=None, rss_checkpoint=labels.append)

    assert labels == ["after_requests", "after_generate", "after_scoring"]


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
