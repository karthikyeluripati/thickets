"""Tests for adapters/spatial_reasoning_gqa.py (and, by extension, _gqa_filtered_base.py) --
built against a fake GQAHandler double (tests/conftest.py's fake_gqa_handler_factory), no
real external/RandOpt clone or GPU/ray/vllm needed.
"""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.adapters.spatial_reasoning_gqa import GQASpatialReasoningBenchmark


def test_capability_and_name():
    bench = GQASpatialReasoningBenchmark(question_ids={"1"})
    assert bench.capability == "spatial_reasoning"
    assert bench.name == "gqa_testdev_spatial"


def test_subset_selection_rule_matches_existing_gqa_pilot_convention():
    bench = GQASpatialReasoningBenchmark(question_ids={"1"})
    assert bench.subset_selection_rule() == "prefix"


def test_known_caveats_documents_unresolved_schema_confirmation():
    bench = GQASpatialReasoningBenchmark(question_ids={"1"})
    assert "not yet confirmed" in " ".join(bench.known_caveats())


def test_load_examples_filters_to_spatial_ids_only(fake_gqa_handler_factory, tiny_image_factory, tmp_path):
    image_path = tmp_path / "img.png"
    tiny_image_factory().save(image_path)
    records = [
        {"question_id": "1", "image_path": str(image_path), "messages": [{"role": "user", "content": []}], "ground_truth": {"answer": "left"}},
        {"question_id": "2", "image_path": str(image_path), "messages": [{"role": "user", "content": []}], "ground_truth": {"answer": "yes"}},
    ]
    handler = fake_gqa_handler_factory(records)
    bench = GQASpatialReasoningBenchmark(gqa_handler=handler, question_ids={"1"})
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="testdev.parquet", split="test"))

    examples = bench.load_examples(cfg)

    assert len(examples) == 1
    assert examples[0].example_id == "1"
    assert examples[0].image is not None
    assert examples[0].target == {"answer": "left"}


def test_build_prompt_passes_through_gqahandler_messages(fake_gqa_handler_factory):
    bench = GQASpatialReasoningBenchmark(gqa_handler=fake_gqa_handler_factory([]), question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    messages = [{"role": "user", "content": [{"type": "text", "text": "is it left?"}]}]
    example = Example(example_id="1", prompt_input={"messages": messages})
    assert bench.build_prompt(example) is messages


def test_parse_prediction_uses_extract_answer_for_voting(fake_gqa_handler_factory):
    handler = fake_gqa_handler_factory([], extract_fn=lambda response: "left" if "left" in response else None)
    bench = GQASpatialReasoningBenchmark(gqa_handler=handler, question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    example = Example(example_id="1")

    parsed_ok = bench.parse_prediction("The answer is left.", example)
    assert parsed_ok.parse_ok is True
    assert parsed_ok.parsed["extracted"] == "left"

    parsed_fail = bench.parse_prediction("I don't know", example)
    assert parsed_fail.parse_ok is False


def test_score_example_calls_compute_reward_with_raw_generation_not_extracted(fake_gqa_handler_factory):
    calls = []

    def _reward(response, ground_truth):
        calls.append(response)
        return 1.0 if response == "The answer is left." else 0.0

    handler = fake_gqa_handler_factory([], reward_fn=_reward, extract_fn=lambda r: "left")
    bench = GQASpatialReasoningBenchmark(gqa_handler=handler, question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    example = Example(example_id="1", target={"answer": "left"})

    parsed = bench.parse_prediction("The answer is left.", example)
    score = bench.score_example(parsed, example)

    assert calls == ["The answer is left."]  # RAW text reached compute_reward, not "left"
    assert score.score == 1.0
    assert score.correct is True


def test_aggregate_metrics_accuracy_and_parser_failure_rate(fake_gqa_handler_factory):
    handler = fake_gqa_handler_factory([], extract_fn=lambda r: r if r else None)
    bench = GQASpatialReasoningBenchmark(gqa_handler=handler, question_ids=set())
    from neural_thickets_repro.benchmarks.base import ExampleScore

    scores = [ExampleScore(score=1.0, correct=True, detail={"extracted": "left"}),
              ExampleScore(score=0.0, correct=False, detail={"extracted": None})]
    metrics = bench.aggregate_metrics(scores)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["primary_metric"] == metrics["accuracy"]
    assert metrics["parser_failure_rate"] == pytest.approx(0.5)


def test_no_arg_constructor_resolves_to_the_default_artifact_path(monkeypatch, tmp_path):
    """The actual bug fix under test: AdapterClass() (the CLI's uniform, no-arg
    instantiation) must resolve to a real default path, not silently have neither
    question_ids nor filter_ids_path set (the previously-reported "needs either
    question_ids or filter_ids_path" RuntimeError on a fresh pod).
    """
    from neural_thickets_repro.benchmarks.adapters import _gqa_filtered_base
    monkeypatch.setattr(_gqa_filtered_base, "DEFAULT_FILTER_IDS_DIR", tmp_path)

    bench = GQASpatialReasoningBenchmark()
    assert bench._filter_ids_path == tmp_path / "gqa_spatial_ids.json"


def test_missing_filter_artifact_gives_actionable_error(monkeypatch, tmp_path):
    from neural_thickets_repro.benchmarks.adapters import _gqa_filtered_base
    monkeypatch.setattr(_gqa_filtered_base, "DEFAULT_FILTER_IDS_DIR", tmp_path)

    bench = GQASpatialReasoningBenchmark()
    with pytest.raises(RuntimeError, match="prepare_gqa_capability_filters") as exc_info:
        bench._resolve_question_ids()
    assert "no capability filter IDs found" in str(exc_info.value)
    assert str(tmp_path / "gqa_spatial_ids.json") in str(exc_info.value)


def test_explicit_filter_ids_path_overrides_default(tmp_path):
    from neural_thickets_repro.benchmarks.adapters.gqa_raw_schema import persist_filter_ids
    custom_path = tmp_path / "custom_spatial.json"
    persist_filter_ids({"1", "2"}, set(), {}, custom_path, tmp_path / "unused_relational.json", tmp_path / "unused_stats.json")

    bench = GQASpatialReasoningBenchmark(filter_ids_path=custom_path)
    assert bench._resolve_question_ids() == {"1", "2"}
