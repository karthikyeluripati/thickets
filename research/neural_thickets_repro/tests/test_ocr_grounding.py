"""Tests for benchmarks/ocr_grounding.py -- pure Python, no GPU/ray/vllm needed. Covers the
five real N=5 TextVQA examples the OCR-grounded subset definition was built to separate:
"macbook air"/"chicken noodle" (multi-token, must be recovered) -> keep; "4" (wheels count,
not OCR-supported) -> reject; "lithia" (single-token) -> keep; a book-material yes/no answer
(not OCR-supported) -> reject.
"""
import pytest

from neural_thickets_repro.benchmarks.ocr_grounding import is_ocr_grounded, ocr_tokens_support_answer


def test_single_token_answer_supported_by_matching_ocr_token():
    assert ocr_tokens_support_answer("lithia", ["LITHIA", "PLATES", "AK"]) is True


def test_multi_token_answer_supported_by_contiguous_ocr_tokens():
    assert ocr_tokens_support_answer("macbook air", ["Apple", "MacBook", "Air", "Pro"]) is True
    assert ocr_tokens_support_answer("chicken noodle", ["Campbell's", "Chicken", "Noodle", "Soup"]) is True


def test_multi_token_answer_not_supported_when_tokens_are_not_contiguous():
    assert ocr_tokens_support_answer("macbook air", ["MacBook", "Pro", "Air"]) is False  # "Pro" breaks contiguity


def test_answer_not_supported_by_unrelated_ocr_tokens():
    assert ocr_tokens_support_answer("4", ["STOP", "SIGN"]) is False


def test_empty_ocr_tokens_never_supports_any_answer():
    assert ocr_tokens_support_answer("stop", []) is False


def test_empty_answer_is_never_supported():
    assert ocr_tokens_support_answer("", ["STOP"]) is False


def test_normalization_handles_case_and_punctuation_differences():
    assert ocr_tokens_support_answer("Stop!", ["stop"]) is True


# --- is_ocr_grounded: the five real N=5 observed cases ---

def test_macbook_air_is_kept():
    answers = ["macbook air"] * 10
    ocr_tokens = ["Apple", "MacBook", "Air", "13-inch"]
    assert is_ocr_grounded(answers, ocr_tokens) is True


def test_wheels_count_answer_is_rejected():
    answers = ["4"] * 10
    ocr_tokens = ["FORD", "TRANSIT"]  # no OCR support for the number "4"
    assert is_ocr_grounded(answers, ocr_tokens) is False


def test_lithia_is_kept():
    answers = ["lithia"] * 10
    ocr_tokens = ["LITHIA", "MOTORS"]
    assert is_ocr_grounded(answers, ocr_tokens) is True


def test_book_material_yes_no_style_answer_is_rejected():
    answers = ["yes", "unanswerable", "hardcover", "yes", "yes", "no", "yes", "unanswerable", "yes", "yes"]
    ocr_tokens = ["PENGUIN", "CLASSICS"]
    assert is_ocr_grounded(answers, ocr_tokens) is False


def test_chicken_noodle_is_kept():
    answers = ["chicken noodle"] * 10
    ocr_tokens = ["Campbell's", "Chicken", "Noodle", "Soup"]
    assert is_ocr_grounded(answers, ocr_tokens) is True


def test_is_ocr_grounded_true_if_any_single_answer_among_many_is_recoverable():
    # Only one of the ten (a plausible real-world minority annotation) is OCR-recoverable --
    # is_ocr_grounded is a "does at least one work" check, not a majority-vote one.
    answers = ["not visible"] * 9 + ["lithia"]
    assert is_ocr_grounded(answers, ["LITHIA"]) is True
