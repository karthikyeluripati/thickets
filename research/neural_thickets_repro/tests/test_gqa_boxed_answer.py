"""Tests for adapters/gqa_boxed_answer.py -- pure Python, no GPU/ray/vllm needed. Covers three
real RunPod bugs this repairs: a nested-brace \\boxed{\\text{...}} answer mis-extracted, a
truncated generation whose fallback fabricated "step step", and (this repair pass) a bare
"Yes" with no \\boxed{} being wrongly counted as a parser failure.
"""
import pytest

from neural_thickets_repro.benchmarks.adapters.gqa_boxed_answer import (
    EXTRACTION_MODE_BOXED,
    EXTRACTION_MODE_CONCISE_FALLBACK,
    EXTRACTION_MODE_FAILURE,
    extract_boxed_answer,
    extract_gqa_answer,
)


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


# ---------------------------------------------------------------------------------------
# extract_gqa_answer -- \boxed{} preferred, conservative concise-answer fallback (this
# repair pass; real bug: example_id=201079958, question "Are there drapes to the right of
# the bed?", raw_generation "Yes", target "yes" -- wrongly counted as a parser failure since
# there was no \boxed{} at all).
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("generation,expected_answer,expected_mode", [
    ("\\boxed{No}", "No", EXTRACTION_MODE_BOXED),
    ("\\boxed{\\text{the person in the blue shirt}}", "the person in the blue shirt", EXTRACTION_MODE_BOXED),
    ("Yes", "Yes", EXTRACTION_MODE_CONCISE_FALLBACK),
    ("keyboard", "keyboard", EXTRACTION_MODE_CONCISE_FALLBACK),
])
def test_extract_gqa_answer_required_regression_cases(generation, expected_answer, expected_mode):
    answer, mode = extract_gqa_answer(generation)
    assert answer == expected_answer
    assert mode == expected_mode


def test_extract_gqa_answer_real_bug_case_bare_yes_is_no_longer_a_parser_failure():
    # example_id=201079958: raw_generation "Yes", target "yes" -- must parse successfully.
    answer, mode = extract_gqa_answer("Yes")
    assert answer == "Yes"
    assert mode == EXTRACTION_MODE_CONCISE_FALLBACK


@pytest.mark.parametrize("short_answer", ["No", "keyboard", "blue shirt", "two people"])
def test_extract_gqa_answer_accepts_conservative_short_answer_examples(short_answer):
    answer, mode = extract_gqa_answer(short_answer)
    assert answer == short_answer
    assert mode == EXTRACTION_MODE_CONCISE_FALLBACK


def test_extract_gqa_answer_prefers_boxed_over_the_fallback_when_both_could_apply():
    # The whole generation ("The answer is \boxed{left}.") is itself too long to pass the
    # concise-answer fallback anyway, but boxed extraction must always be tried FIRST.
    answer, mode = extract_gqa_answer("The answer is \\boxed{left}.")
    assert answer == "left"
    assert mode == EXTRACTION_MODE_BOXED


def test_extract_gqa_answer_long_multiline_reasoning_with_no_boxed_answer_fails():
    long_reasoning = (
        "Let me think step by step.\n"
        "First I will look at the bed and its surroundings.\n"
        "There appear to be several objects nearby that could be drapes or curtains."
    )
    answer, mode = extract_gqa_answer(long_reasoning)
    assert answer is None
    assert mode == EXTRACTION_MODE_FAILURE


def test_extract_gqa_answer_preserves_protection_against_the_original_truncated_generation():
    """The exact original truncated generation this package's extract_boxed_answer() already
    protects against (too many tokens to ever pass the concise-answer fallback either) --
    must never resolve to a fabricated short answer like the historically observed
    "step step".
    """
    truncated = "Let me think step by step. First I will look at the person, then I will reason about"
    answer, mode = extract_gqa_answer(truncated)
    assert answer is None
    assert mode == EXTRACTION_MODE_FAILURE
    assert answer != "step step"


def test_extract_gqa_answer_empty_generation_fails():
    answer, mode = extract_gqa_answer("")
    assert answer is None
    assert mode == EXTRACTION_MODE_FAILURE


def test_extract_gqa_answer_whitespace_only_generation_fails():
    answer, mode = extract_gqa_answer("   \n  ")
    assert answer is None
    assert mode == EXTRACTION_MODE_FAILURE


def test_extract_gqa_answer_rejects_a_short_looking_but_multiline_generation():
    # Two short lines -- still rejected, since a genuine short answer is never multi-line.
    answer, mode = extract_gqa_answer("Yes\nbut I am not fully sure")
    assert answer is None
    assert mode == EXTRACTION_MODE_FAILURE


def test_extract_gqa_answer_rejects_a_generation_exceeding_the_token_limit():
    nine_tokens = "one two three four five six seven eight nine"
    answer, mode = extract_gqa_answer(nine_tokens)
    assert answer is None
    assert mode == EXTRACTION_MODE_FAILURE
