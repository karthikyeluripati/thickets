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


def test_build_prompt_appends_capability_benchmark_override_without_mutating_original(fake_gqa_handler_factory):
    """PROMPT FIX (this repair pass): build_prompt() no longer passes GQAHandler's messages
    through unchanged -- it appends a brief-reasoning override instruction to the last turn's
    content, addressing a real RunPod token-ceiling truncation, while leaving GQAHandler's own
    messages object untouched (a new list/dicts are returned).
    """
    bench = GQASpatialReasoningBenchmark(gqa_handler=fake_gqa_handler_factory([]), question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    original_content = [{"type": "text", "text": "is it left?"}]
    messages = [{"role": "user", "content": original_content}]
    example = Example(example_id="1", prompt_input={"messages": messages})

    result = bench.build_prompt(example)

    assert result is not messages  # a new list, GQAHandler's own object never mutated
    assert messages[0]["content"] is original_content  # original untouched
    assert len(original_content) == 1  # original content list untouched
    result_texts = [block["text"] for block in result[-1]["content"] if block.get("type") == "text"]
    assert "is it left?" in result_texts
    assert any("boxed" in t and "brief" in t for t in result_texts)


def test_parse_prediction_uses_balanced_brace_boxed_extraction_not_extract_answer_for_voting(fake_gqa_handler_factory):
    """PARSER FIX (this repair pass): parse_prediction() uses this package's own
    gqa_boxed_answer.extract_boxed_answer(), never GQAHandler.extract_answer_for_voting --
    proven here by giving extract_answer_for_voting a fake that would behave differently.
    """
    handler = fake_gqa_handler_factory([], extract_fn=lambda response: "SHOULD NOT BE USED")
    bench = GQASpatialReasoningBenchmark(gqa_handler=handler, question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    example = Example(example_id="1")

    parsed_ok = bench.parse_prediction("The final answer is \\boxed{left}.", example)
    assert parsed_ok.parse_ok is True
    assert parsed_ok.parsed["extracted"] == "left"

    parsed_fail = bench.parse_prediction("I don't know", example)
    assert parsed_fail.parse_ok is False
    assert parsed_fail.parsed["extracted"] is None


def test_parse_prediction_handles_the_real_nested_brace_case(fake_gqa_handler_factory):
    bench = GQASpatialReasoningBenchmark(gqa_handler=fake_gqa_handler_factory([]), question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    example = Example(example_id="1")

    parsed = bench.parse_prediction("\\boxed{\\text{the person in the blue shirt}}", example)
    assert parsed.parse_ok is True
    assert parsed.parsed["extracted"] == "the person in the blue shirt"


def test_parse_prediction_truncated_generation_is_a_real_failure_not_step_step(fake_gqa_handler_factory):
    """The other real RunPod bug: a generation truncated before any \\boxed{} appeared must
    be a genuine parser failure, never a fabricated answer like the observed "step step".
    """
    bench = GQASpatialReasoningBenchmark(gqa_handler=fake_gqa_handler_factory([]), question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    example = Example(example_id="1")

    parsed = bench.parse_prediction("Let me think step by step. First I will look at the", example)
    assert parsed.parse_ok is False
    assert parsed.parsed["extracted"] is None


def test_score_example_calls_compute_reward_with_raw_generation_not_extracted(fake_gqa_handler_factory):
    calls = []

    def _reward(response, ground_truth):
        calls.append(response)
        return 1.0 if response == "The answer is \\boxed{left}." else 0.0

    handler = fake_gqa_handler_factory([], reward_fn=_reward)
    bench = GQASpatialReasoningBenchmark(gqa_handler=handler, question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    example = Example(example_id="1", target={"answer": "left"})

    parsed = bench.parse_prediction("The answer is \\boxed{left}.", example)
    score = bench.score_example(parsed, example)

    assert calls == ["The answer is \\boxed{left}."]  # RAW text reached compute_reward, not "left"
    assert score.score == 1.0
    assert score.correct is True


def test_score_example_skips_compute_reward_entirely_on_parse_failure(fake_gqa_handler_factory):
    def _reward(response, ground_truth):
        raise AssertionError("compute_reward must not be called when there's no extractable answer")

    handler = fake_gqa_handler_factory([], reward_fn=_reward)
    bench = GQASpatialReasoningBenchmark(gqa_handler=handler, question_ids=set())
    from neural_thickets_repro.benchmarks.base import Example
    example = Example(example_id="1", target={"answer": "left"})

    parsed = bench.parse_prediction("no boxed answer here", example)
    score = bench.score_example(parsed, example)
    assert score.score == 0.0
    assert score.correct is False
    assert score.detail["reason"] == "parse_failure"


def test_aggregate_metrics_accuracy_and_parser_failure_rate():
    from neural_thickets_repro.benchmarks.base import ExampleScore

    scores = [ExampleScore(score=1.0, correct=True, detail={"extracted": "left"}),
              ExampleScore(score=0.0, correct=False, detail={"extracted": None, "reason": "parse_failure"})]
    bench = GQASpatialReasoningBenchmark(question_ids=set())
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
    persist_filter_ids({"1", "2"}, set(), set(), {}, custom_path, tmp_path / "unused_relational.json", tmp_path / "unused_mixed.json", tmp_path / "unused_stats.json")

    bench = GQASpatialReasoningBenchmark(filter_ids_path=custom_path)
    assert bench._resolve_question_ids() == {"1", "2"}
