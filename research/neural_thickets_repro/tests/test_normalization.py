"""Tests for benchmarks/normalization.py -- pure Python, no GPU/ray/vllm needed."""
import pytest

from neural_thickets_repro.benchmarks.normalization import extract_integer, normalize_answer


def test_lowercases_and_strips_punctuation():
    assert normalize_answer("A Dog!") == "dog"


def test_strips_leading_articles():
    assert normalize_answer("the cat") == "cat"
    assert normalize_answer("an apple") == "apple"


def test_singularization_known_pairs():
    assert normalize_answer("dogs") == "dog"
    assert normalize_answer("boxes") == "box"
    assert normalize_answer("babies") == "baby"
    assert normalize_answer("glass") == "glass"  # ends in "ss", must not be stripped to "gla"


def test_idempotent():
    text = "The Dogs!"
    once = normalize_answer(text)
    twice = normalize_answer(once)
    assert once == twice


def test_extract_integer_from_digit_run():
    assert extract_integer("There are 4.") == 4
    assert extract_integer("4 objects") == 4
    assert extract_integer("  5 items") == 5
    assert extract_integer("17") == 17


def test_extract_integer_from_number_word():
    assert extract_integer("The answer is four.") == 4


def test_extract_integer_returns_none_for_non_numeric():
    assert extract_integer("I don't know") is None
    assert extract_integer("") is None


def test_extract_integer_prefers_digit_run_over_number_word():
    assert extract_integer("4 (four)") == 4
