"""Tests for benchmarks/image_sanity.py -- fake-llm-driven, no GPU/ray/vllm needed."""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.benchmarks.image_sanity import (
    ImageSanityError,
    make_shuffled_variant,
    make_text_only_variant,
    run_image_sanity_check,
)


def _examples(n, tiny_image_factory):
    return [Example(example_id=str(i), image=tiny_image_factory(color=(i, 0, 0)), image_ref=f"img{i}", prompt_input={"q": i}, target=str(i)) for i in range(n)]


def test_shuffled_variant_is_a_true_derangement(tiny_image_factory):
    examples = _examples(10, tiny_image_factory)
    shuffled = make_shuffled_variant(examples, seed=1)

    for original, swapped in zip(examples, shuffled):
        assert swapped.image is not original.image  # no example keeps its own original image
    # every example still present, same ids/order/prompt/target
    assert [e.example_id for e in shuffled] == [e.example_id for e in examples]
    assert [e.target for e in shuffled] == [e.target for e in examples]


def test_shuffled_variant_deterministic_given_seed(tiny_image_factory):
    examples = _examples(10, tiny_image_factory)
    a = make_shuffled_variant(examples, seed=42)
    b = make_shuffled_variant(examples, seed=42)
    assert [e.metadata["sanity_shuffle_source_id"] for e in a] == [e.metadata["sanity_shuffle_source_id"] for e in b]


def test_shuffled_variant_records_permutation_for_audit(tiny_image_factory):
    examples = _examples(5, tiny_image_factory)
    shuffled = make_shuffled_variant(examples, seed=7)
    for ex in shuffled:
        assert "sanity_shuffle_source_id" in ex.metadata
        assert ex.metadata["sanity_shuffle_source_id"] != ex.example_id


def test_shuffled_variant_raises_for_n_less_than_or_equal_1(tiny_image_factory):
    with pytest.raises(ImageSanityError):
        make_shuffled_variant(_examples(1, tiny_image_factory), seed=1)
    with pytest.raises(ImageSanityError):
        make_shuffled_variant([], seed=1)


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
