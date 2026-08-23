"""Tests for benchmarks/base.py -- pure Python, no GPU/ray/vllm/datasets needed."""
import pytest

from neural_thickets_repro.benchmarks.base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction


def test_capability_benchmark_is_not_directly_instantiable():
    with pytest.raises(TypeError):
        CapabilityBenchmark()


def test_fake_benchmark_round_trips_example_parsed_prediction_score(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    example = Example(example_id="1", image=None, prompt_input={"question": "q"}, target="42")

    messages = bench.build_prompt(example)
    assert isinstance(messages, list)

    parsed = bench.parse_prediction("42", example)
    assert parsed.parse_ok is True
    assert parsed.parsed == "42"

    score = bench.score_example(parsed, example)
    assert score.score == 1.0
    assert score.correct is True


def test_default_prepare_image_is_passthrough(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    image = tiny_image_factory()
    example = Example(example_id="1", image=image)
    assert bench.prepare_image(example) is image


def test_default_subset_selection_rule_is_shuffled_prefix(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    assert bench.subset_selection_rule() == "shuffled_prefix"
    assert bench.default_subset_size() == 200


def test_default_supports_text_only_condition_true(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    assert bench.supports_text_only_condition() is True
    assert bench.text_only_unsupported_reason() is None


def test_default_known_caveats_empty(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    assert bench.known_caveats() == []


def test_aggregate_metrics_includes_required_keys(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    scores = [ExampleScore(score=1.0, correct=True), ExampleScore(score=0.0, correct=False)]
    metrics = bench.aggregate_metrics(scores)
    assert "primary_metric" in metrics
    assert "parser_failure_rate" in metrics
    assert metrics["primary_metric"] == pytest.approx(0.5)


def test_example_is_frozen():
    example = Example(example_id="1")
    with pytest.raises(Exception):
        example.example_id = "2"


def test_parsed_prediction_and_example_score_defaults():
    parsed = ParsedPrediction(parsed=5, parse_ok=True)
    assert parsed.parse_error is None
    score = ExampleScore(score=0.5)
    assert score.correct is None
    assert score.detail == {}


# ---------------------------------------------------------------------------------------
# default repeatability_verdict() -- discrete exact-match criterion, this repair pass
# ---------------------------------------------------------------------------------------

def _run_result(pairs, primary_metric):
    """pairs: [(example_id, parsed_value), ...] -- builds a minimal RunResult for these tests."""
    from neural_thickets_repro.benchmarks.runner import PerExampleResult, RunResult

    per_example = [
        PerExampleResult(example_id, "img", "raw", ParsedPrediction(parsed_value, True), ExampleScore(1.0))
        for example_id, parsed_value in pairs
    ]
    return RunResult(per_example=per_example, aggregate_metrics={"primary_metric": primary_metric})


def test_default_repeatability_verdict_true_when_exact_match_and_metric_match(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    base = _run_result([("1", "left"), ("2", "right")], primary_metric=0.5)
    repeat = _run_result([("1", "left"), ("2", "right")], primary_metric=0.5)

    repeatable, diagnostics = bench.repeatability_verdict(base, repeat)
    assert repeatable is True
    assert diagnostics == {}


def test_default_repeatability_verdict_false_on_any_parsed_prediction_difference(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    base = _run_result([("1", "left"), ("2", "right")], primary_metric=0.5)
    repeat = _run_result([("1", "left"), ("2", "WRONG")], primary_metric=0.5)

    repeatable, _ = bench.repeatability_verdict(base, repeat)
    assert repeatable is False


def test_default_repeatability_verdict_false_on_metric_mismatch_even_if_predictions_match(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    base = _run_result([("1", "left")], primary_metric=0.5)
    repeat = _run_result([("1", "left")], primary_metric=0.6)

    repeatable, _ = bench.repeatability_verdict(base, repeat)
    assert repeatable is False


def test_default_repeatability_verdict_false_when_no_common_ids(fake_capability_benchmark_factory):
    bench = fake_capability_benchmark_factory()
    base = _run_result([("1", "left")], primary_metric=0.5)
    repeat = _run_result([("2", "left")], primary_metric=0.5)

    repeatable, _ = bench.repeatability_verdict(base, repeat)
    assert repeatable is False
