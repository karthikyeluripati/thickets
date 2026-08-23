"""Tests for adapters/counting_tallyqa.py -- synthetic the_cauldron-shaped rows, no real
dataset download / GPU / ray / vllm needed. Only load_examples() touches datasets.load_dataset;
everything else is exercised against hand-built Example objects.
"""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.counting_tallyqa import TallyQACountingBenchmark
from neural_thickets_repro.benchmarks.base import Example, ParsedPrediction


def _bench():
    return TallyQACountingBenchmark()


def test_capability_and_name():
    bench = _bench()
    assert bench.capability == "counting"
    assert bench.name == "tallyqa_the_cauldron"


def test_known_caveats_documents_repackaging_and_single_split():
    caveats = " ".join(_bench().known_caveats())
    assert "the_cauldron" in caveats
    assert "split" in caveats


@pytest.mark.parametrize("generation,expected", [
    ("There are 4.", 4), ("4 objects", 4), ("The answer is four.", 4), ("17", 17),
])
def test_parser_handles_realistic_generations(generation, expected):
    bench = _bench()
    example = Example(example_id="1", target=expected)
    parsed = bench.parse_prediction(generation, example)
    assert parsed.parse_ok is True
    assert parsed.parsed == expected


def test_parser_flags_non_numeric_generation_as_failure():
    bench = _bench()
    example = Example(example_id="1", target=4)
    parsed = bench.parse_prediction("I don't see any objects", example)
    assert parsed.parse_ok is False
    assert parsed.parsed is None


def test_score_example_exact_match_and_mae():
    bench = _bench()
    example = Example(example_id="1", target=4)
    parsed_correct = ParsedPrediction(parsed=4, parse_ok=True)
    parsed_wrong = ParsedPrediction(parsed=6, parse_ok=True)

    score_correct = bench.score_example(parsed_correct, example)
    score_wrong = bench.score_example(parsed_wrong, example)

    assert score_correct.score == 1.0 and score_correct.correct is True
    assert score_wrong.score == 0.0 and score_wrong.correct is False
    assert score_wrong.detail["abs_error"] == 2


def test_score_example_parse_failure_scores_zero():
    bench = _bench()
    example = Example(example_id="1", target=4)
    parsed = ParsedPrediction(parsed=None, parse_ok=False, parse_error="no integer")
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.detail["reason"] == "parse_failure"


def test_aggregate_metrics_computes_exact_match_and_mae():
    bench = _bench()
    examples = [Example(example_id=str(i), target=4) for i in range(4)]
    parseds = [ParsedPrediction(4, True), ParsedPrediction(6, True), ParsedPrediction(4, True), ParsedPrediction(None, False, "x")]
    scores = [bench.score_example(p, e) for p, e in zip(parseds, examples)]

    metrics = bench.aggregate_metrics(scores)
    assert metrics["exact_match_accuracy"] == pytest.approx(2 / 4)
    assert metrics["mae"] == pytest.approx((0 + 2 + 0) / 3)
    assert metrics["primary_metric"] == metrics["exact_match_accuracy"]
    assert metrics["parser_failure_rate"] == pytest.approx(1 / 4)


def test_aggregate_metrics_empty_scores():
    bench = _bench()
    metrics = bench.aggregate_metrics([])
    assert metrics["primary_metric"] == 0.0
    assert metrics["parser_failure_rate"] == 0.0


def test_build_prompt_includes_question_and_number_instruction():
    bench = _bench()
    example = Example(example_id="1", prompt_input={"question": "How many chairs are there?"})
    messages = bench.build_prompt(example)
    text_block = messages[0]["content"][1]["text"]
    assert "How many chairs are there?" in text_block
    assert "number" in text_block.lower()


def test_load_examples_flattens_multi_turn_rows_and_parses_ground_truth(tiny_image_factory, monkeypatch):
    image = tiny_image_factory()
    fake_rows = [
        {"images": [image], "texts": [
            {"user": "How many dogs?", "assistant": "2.", "source": "TallyQA"},
            {"user": "How many cats?", "assistant": "3.", "source": "TallyQA"},
        ]},
        {"images": [image], "texts": [
            {"user": "How many chairs?", "assistant": "not a number", "source": "TallyQA"},
        ]},
    ]

    def _fake_load_dataset(source, config, split, revision):
        assert config == "tallyqa"
        return fake_rows

    import sys
    import types
    fake_datasets_module = types.ModuleType("datasets")
    fake_datasets_module.load_dataset = _fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)

    bench = _bench()
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="HuggingFaceM4/the_cauldron", split="train", revision=None))
    examples = bench.load_examples(cfg)

    assert len(examples) == 3
    assert examples[0].example_id == "0_0"
    assert examples[0].target == 2
    assert examples[1].example_id == "0_1"
    assert examples[1].target == 3
    assert examples[2].example_id == "1_0"
    assert examples[2].target is None  # malformed ground truth -- a genuine integrity issue
