"""Tests for adapters/gqa_boxed_answer.py -- pure Python, no GPU/ray/vllm needed. Covers the
two real RunPod bugs this repairs (a nested-brace \\boxed{\\text{...}} answer mis-extracted,
and a truncated generation whose fallback fabricated "step step").
"""
import pytest

from neural_thickets_repro.benchmarks.adapters.gqa_boxed_answer import extract_boxed_answer


def test_extracts_plain_boxed_answer():
    assert extract_boxed_answer("The answer is \\boxed{left}.") == "left"


def test_extracts_nested_text_wrapped_boxed_answer_real_bug_case():
    """The exact real RunPod failure: the previous (broken) parser produced
    "\\text{the person" -- truncated at the FIRST inner '}' instead of balancing braces.
    """
    generation = "Looking at the image, \\boxed{\\text{the person in the blue shirt}}"
    assert extract_boxed_answer(generation) == "the person in the blue shirt"


def test_truncated_generation_with_no_boxed_answer_is_a_real_parser_failure():
    """The other real RunPod bug: a generation cut off by the token ceiling before any
    \\boxed{} appeared. The OLD fallback fabricated "step step" as if it were a real answer;
    this must return None -- a genuine, honest parser failure.
    """
    truncated = "Let me think step by step. First I will look at the person, then I will reason about"
    assert extract_boxed_answer(truncated) is None


def test_unbalanced_boxed_brace_truncated_mid_answer_returns_none():
    truncated_mid_box = "The final answer is \\boxed{the person in the blue"  # never closes
    assert extract_boxed_answer(truncated_mid_box) is None


def test_no_boxed_marker_at_all_returns_none():
    assert extract_boxed_answer("I don't know the answer.") is None


def test_empty_boxed_content_returns_none():
    assert extract_boxed_answer("\\boxed{}") is None


def test_whitespace_only_boxed_content_returns_none():
    assert extract_boxed_answer("\\boxed{   }") is None


def test_unwraps_mathrm_and_textbf_wrappers():
    assert extract_boxed_answer("\\boxed{\\mathrm{yes}}") == "yes"
    assert extract_boxed_answer("\\boxed{\\textbf{4}}") == "4"


def test_uses_the_last_boxed_occurrence_when_multiple_are_present():
    generation = "An example is \\boxed{wrong} but actually the answer is \\boxed{left}."
    assert extract_boxed_answer(generation) == "left"


def test_nested_wrapper_two_levels_deep():
    assert extract_boxed_answer("\\boxed{\\text{\\textbf{blue}}}") == "blue"


def test_boxed_content_with_no_wrapper_and_surrounding_whitespace_is_stripped():
    assert extract_boxed_answer("\\boxed{  yes  }") == "yes"


def test_boxed_content_containing_unrelated_braces_still_balances_correctly():
    # A generation that happens to mention other brace-containing text before the real answer.
    generation = "Some irrelevant {note} here. \\boxed{\\text{holding a cup}}"
    assert extract_boxed_answer(generation) == "holding a cup"
