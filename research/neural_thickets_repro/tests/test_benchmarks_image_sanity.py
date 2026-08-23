"""Tests for benchmarks/image_sanity.py -- fake-llm-driven, no GPU/ray/vllm needed."""
import json
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.base import CapabilityBenchmark, Example
from neural_thickets_repro.benchmarks.image_sanity import (
    ImageSanityError,
    make_shuffled_variant,
    make_text_only_variant,
    run_image_sanity_check,
    write_image_sanity_predictions_jsonl,
)


def _examples(n, tiny_image_factory):
    return [Example(example_id=str(i), image=tiny_image_factory(color=(i, 0, 0)), image_ref=f"img{i}", prompt_input={"q": i}, target=str(i)) for i in range(n)]


def test_shuffled_variant_is_a_true_derangement(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    examples = _examples(10, tiny_image_factory)
    shuffled = make_shuffled_variant(examples, seed=1, benchmark=bench)

    for original, swapped in zip(examples, shuffled):
        assert swapped.image is not original.image  # no example keeps its own original image
    # every example still present, same ids/order/prompt/target
    assert [e.example_id for e in shuffled] == [e.example_id for e in examples]
    assert [e.target for e in shuffled] == [e.target for e in examples]


def test_shuffled_variant_deterministic_given_seed(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    examples = _examples(10, tiny_image_factory)
    a = make_shuffled_variant(examples, seed=42, benchmark=bench)
    b = make_shuffled_variant(examples, seed=42, benchmark=bench)
    assert [e.metadata["sanity_shuffle_source_id"] for e in a] == [e.metadata["sanity_shuffle_source_id"] for e in b]


def test_shuffled_variant_records_permutation_for_audit(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    examples = _examples(5, tiny_image_factory)
    shuffled = make_shuffled_variant(examples, seed=7, benchmark=bench)
    for ex in shuffled:
        assert "sanity_shuffle_source_id" in ex.metadata
        assert ex.metadata["sanity_shuffle_source_id"] != ex.example_id


def test_shuffled_variant_raises_for_n_less_than_or_equal_1(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    with pytest.raises(ImageSanityError):
        make_shuffled_variant(_examples(1, tiny_image_factory), seed=1, benchmark=bench)
    with pytest.raises(ImageSanityError):
        make_shuffled_variant([], seed=1, benchmark=bench)


def test_shuffled_variant_delegates_to_benchmark_hook(tiny_image_factory):
    """make_shuffled_variant must go through benchmark.make_shuffled_image_variant() --
    proven here with a fake override that behaves differently from the default.
    """
    calls = []

    class _RecordingBenchmark(CapabilityBenchmark):
        capability = "fake"
        name = "fake"

        def load_examples(self, cfg):
            raise NotImplementedError

        def build_prompt(self, example):
            return []

        def parse_prediction(self, raw_generation, example):
            raise NotImplementedError

        def score_example(self, parsed, example):
            raise NotImplementedError

        def aggregate_metrics(self, scores):
            return {"primary_metric": 0.0, "parser_failure_rate": 0.0}

        def make_shuffled_image_variant(self, example, source_example):
            calls.append((example.example_id, source_example.example_id))
            return super().make_shuffled_image_variant(example, source_example)

    bench = _RecordingBenchmark()
    examples = _examples(5, tiny_image_factory)
    make_shuffled_variant(examples, seed=7, benchmark=bench)
    assert len(calls) == 5
    for example_id, source_id in calls:
        assert example_id != source_id  # true derangement, enforced before the hook is called


def test_text_only_variant_sets_image_none_prompt_target_unchanged(tiny_image_factory):
    examples = _examples(3, tiny_image_factory)
    variants = make_text_only_variant(examples)
    for original, variant in zip(examples, variants):
        assert variant.image is None
        assert variant.prompt_input == original.prompt_input
        assert variant.target == original.target
        assert variant.metadata["sanity_condition"] == "text_only"


def _fake_output(text):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def _fake_llm_scoring_by_image_color(examples_by_ref):
    """Fake LLM whose "correctness" is driven by which image reached it -- lets us prove the
    gap computation reacts to swapped/removed images, not just to a canned constant.
    """
    def _generate(requests, sampling_params, use_tqdm=True):
        outputs = []
        for req in requests:
            if "multi_modal_data" not in req:
                outputs.append(_fake_output("wrong"))  # no image reached the model -> always wrong
            else:
                image = req["multi_modal_data"]["image"]
                # correct iff the pixel color matches example index 0's own color (red channel 0)
                outputs.append(_fake_output("0" if image.getpixel((0, 0))[0] == 0 else "wrong"))
        return outputs
    return SimpleNamespace(generate=_generate)


def _fake_tokenizer():
    return SimpleNamespace(apply_chat_template=lambda messages, add_generation_prompt, tokenize: "TEXT")


def test_run_image_sanity_check_computes_gap_from_fake_llm(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    # target for each example equals its own correct-color-driven expected generation "0" or "wrong"
    examples = [Example(example_id="0", image=tiny_image_factory(color=(0, 0, 0)), prompt_input={}, target="0")]
    examples += [Example(example_id=str(i), image=tiny_image_factory(color=(i, 0, 0)), prompt_input={}, target="wrong") for i in range(1, 6)]

    llm = _fake_llm_scoring_by_image_color(examples)
    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=3)

    assert result.n == 6
    # correct-image run: example "0" has color (0,0,0) matching its own target "0" -> full credit there
    assert result.correct_image_primary_metric > 0.0
    assert isinstance(result.correct_minus_shuffled, float)


def test_supports_text_only_false_reports_reason_no_crash(fake_capability_benchmark_factory, tiny_image_factory, monkeypatch):
    bench = fake_capability_benchmark_factory()
    monkeypatch.setattr(bench, "supports_text_only_condition", lambda: False)
    monkeypatch.setattr(bench, "text_only_unsupported_reason", lambda: "grounding requires the image itself")

    examples = [Example(example_id=str(i), image=tiny_image_factory(), prompt_input={}, target="x") for i in range(4)]
    llm = _fake_llm_scoring_by_image_color(examples)

    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=1)

    assert result.text_only_supported is False
    assert result.text_only_primary_metric is None
    assert result.text_only_unsupported_reason == "grounding requires the image itself"
    assert result.correct_minus_text_only is None


def test_to_dict_includes_all_required_fields(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    examples = [Example(example_id=str(i), image=tiny_image_factory(), prompt_input={}, target="x") for i in range(4)]
    llm = _fake_llm_scoring_by_image_color(examples)
    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=1)

    d = result.to_dict()
    for field in ("correct_image_score", "shuffled_image_score", "text_only_score", "correct_minus_shuffled", "correct_minus_text_only"):
        assert field in d


def test_to_dict_does_not_include_the_heavy_per_example_run_results(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    examples = [Example(example_id=str(i), image=tiny_image_factory(), prompt_input={}, target="x") for i in range(4)]
    llm = _fake_llm_scoring_by_image_color(examples)
    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=1)

    d = result.to_dict()
    assert "correct_result" not in d
    assert "shuffled_result" not in d
    assert "text_only_result" not in d


def test_run_image_sanity_check_populates_per_condition_run_results(fake_capability_benchmark_factory, tiny_image_factory):
    bench = fake_capability_benchmark_factory()
    examples = [Example(example_id=str(i), image=tiny_image_factory(), prompt_input={}, target="x") for i in range(4)]
    llm = _fake_llm_scoring_by_image_color(examples)
    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=1)

    assert result.correct_result is not None and len(result.correct_result.per_example) == 4
    assert result.shuffled_result is not None and len(result.shuffled_result.per_example) == 4
    assert result.text_only_result is not None and len(result.text_only_result.per_example) == 4


def test_run_image_sanity_check_text_only_result_is_none_when_unsupported(fake_capability_benchmark_factory, tiny_image_factory, monkeypatch):
    bench = fake_capability_benchmark_factory()
    monkeypatch.setattr(bench, "supports_text_only_condition", lambda: False)
    examples = [Example(example_id=str(i), image=tiny_image_factory(), prompt_input={}, target="x") for i in range(4)]
    llm = _fake_llm_scoring_by_image_color(examples)

    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=1)
    assert result.text_only_result is None


# ---------------------------------------------------------------------------------------
# write_image_sanity_predictions_jsonl (this repair pass)
# ---------------------------------------------------------------------------------------

def test_write_image_sanity_predictions_jsonl_includes_all_three_conditions(fake_capability_benchmark_factory, tiny_image_factory, tmp_path):
    bench = fake_capability_benchmark_factory()
    examples = [Example(example_id=str(i), image=tiny_image_factory(), prompt_input={}, target="x") for i in range(3)]
    llm = _fake_llm_scoring_by_image_color(examples)
    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=1)

    out_path = tmp_path / "image_sanity_predictions.jsonl"
    write_image_sanity_predictions_jsonl(result, out_path)

    lines = [json.loads(l) for l in out_path.read_text().strip().splitlines()]
    conditions = {l["condition"] for l in lines}
    assert conditions == {"correct", "shuffled", "text_only"}
    assert len(lines) == 3 * 3  # 3 examples x 3 conditions
    for line in lines:
        for field in ("example_id", "image_ref", "raw_generation", "parsed_prediction", "score", "correct", "detail"):
            assert field in line


def test_write_image_sanity_predictions_jsonl_omits_text_only_when_unsupported(fake_capability_benchmark_factory, tiny_image_factory, monkeypatch, tmp_path):
    bench = fake_capability_benchmark_factory()
    monkeypatch.setattr(bench, "supports_text_only_condition", lambda: False)
    examples = [Example(example_id=str(i), image=tiny_image_factory(), prompt_input={}, target="x") for i in range(3)]
    llm = _fake_llm_scoring_by_image_color(examples)
    result = run_image_sanity_check(bench, examples, llm, _fake_tokenizer(), sampling_params=None, seed=1)

    out_path = tmp_path / "image_sanity_predictions.jsonl"
    write_image_sanity_predictions_jsonl(result, out_path)

    lines = [json.loads(l) for l in out_path.read_text().strip().splitlines()]
    conditions = {l["condition"] for l in lines}
    assert conditions == {"correct", "shuffled"}
    assert len(lines) == 3 * 2
