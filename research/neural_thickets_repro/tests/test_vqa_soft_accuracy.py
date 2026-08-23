"""Tests for benchmarks/vqa_soft_accuracy.py -- pure Python, no GPU/ray/vllm needed."""
import pytest

from neural_thickets_repro.benchmarks.vqa_soft_accuracy import vqa_soft_accuracy


def test_prediction_matching_at_least_3_of_9_in_every_leave_one_out_draw_scores_1():
    # All 10 answers identical -> every leave-one-out draw of 9 has 9 matches -> min(9/3,1)=1
    answers = ["dog"] * 10
    assert vqa_soft_accuracy("dog", answers) == pytest.approx(1.0)


def test_zero_matches_scores_zero():
    answers = ["cat"] * 10
    assert vqa_soft_accuracy("dog", answers) == 0.0


def test_known_partial_match_case():
    # 3 of 10 answers are "dog", 7 are "cat". Leave-one-out over the 3 "dog" draws: 2 dogs
    # remain among the other 9 -> min(2/3,1). Leave-one-out over the 7 "cat" draws: 3 dogs
    # remain among the other 9 -> min(3/3,1)=1.
    answers = ["dog"] * 3 + ["cat"] * 7
    expected = (3 * (2 / 3) + 7 * 1.0) / 10
    assert vqa_soft_accuracy("dog", answers) == pytest.approx(expected)


def test_case_and_punctuation_insensitive_match_still_counts():
    answers = ["Dog."] * 10
    assert vqa_soft_accuracy("dog", answers) == pytest.approx(1.0)


def test_empty_answers_raises():
    with pytest.raises(ValueError):
        vqa_soft_accuracy("dog", [])


def test_single_answer_fallback_exact_match():
    assert vqa_soft_accuracy("dog", ["dog"]) == 1.0
    assert vqa_soft_accuracy("dog", ["cat"]) == 0.0
