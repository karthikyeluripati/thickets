"""Tests for thicket_metrics.py -- pure Python/numpy, no GPU/ray/vllm needed."""
import pytest

from neural_thickets_repro.thicket_metrics import (
    aggregate_thicket_run,
    compute_delta_fields,
    wilson_confidence_interval,
)


# --- compute_delta_fields ---


def test_compute_delta_fields_positive_delta_is_expert():
    delta, is_expert, is_tie = compute_delta_fields(candidate_score=0.6, base_score=0.5)
    assert delta == pytest.approx(0.1)
    assert is_expert is True
    assert is_tie is False


def test_compute_delta_fields_zero_delta_is_tie_not_expert():
    delta, is_expert, is_tie = compute_delta_fields(candidate_score=0.5, base_score=0.5)
    assert delta == 0.0
    assert is_expert is False
    assert is_tie is True


def test_compute_delta_fields_negative_delta_is_neither():
    delta, is_expert, is_tie = compute_delta_fields(candidate_score=0.4, base_score=0.5)
    assert delta == pytest.approx(-0.1)
    assert is_expert is False
    assert is_tie is False


# --- wilson_confidence_interval: known reference cases ---


def test_wilson_ci_zero_successes_known_case():
    # n=10, x=0 -> Wilson 95% CI ~= (0.0, 0.2775), computed by hand from the standard formula.
    lower, upper = wilson_confidence_interval(successes=0, n=10, confidence=0.95)
    assert lower == pytest.approx(0.0, abs=1e-3)
    assert upper == pytest.approx(0.2775, abs=1e-3)


def test_wilson_ci_half_successes_known_case():
    # n=100, x=50 -> Wilson 95% CI ~= (0.4039, 0.5962).
    lower, upper = wilson_confidence_interval(successes=50, n=100, confidence=0.95)
    assert lower == pytest.approx(0.4039, abs=1e-3)
    assert upper == pytest.approx(0.5962, abs=1e-3)


def test_wilson_ci_bounds_are_valid_probabilities():
    lower, upper = wilson_confidence_interval(successes=7, n=20, confidence=0.95)
    assert 0.0 <= lower <= upper <= 1.0


def test_wilson_ci_all_successes():
    lower, upper = wilson_confidence_interval(successes=10, n=10, confidence=0.95)
    assert upper == pytest.approx(1.0, abs=1e-3)
    assert lower < 1.0


def test_wilson_ci_rejects_non_positive_n():
    with pytest.raises(ValueError):
        wilson_confidence_interval(successes=0, n=0)


def test_wilson_ci_rejects_successes_out_of_range():
    with pytest.raises(ValueError):
        wilson_confidence_interval(successes=11, n=10)


def test_wilson_ci_rejects_unsupported_confidence():
    with pytest.raises(ValueError, match="Unsupported confidence"):
        wilson_confidence_interval(successes=5, n=10, confidence=0.99)


# --- aggregate_thicket_run ---


def test_expert_density_calculation_correct():
    base_score = 0.5
    candidates = [(1, 0.6), (2, 0.7), (3, 0.4), (4, 0.3)]  # 2 experts, 0 ties, 2 regressions
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.expert_count == 2
    assert metrics.expert_density == pytest.approx(0.5)


def test_ties_not_counted_as_experts():
    base_score = 0.5
    candidates = [(1, 0.5), (2, 0.5), (3, 0.6)]  # 2 ties, 1 expert
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.tie_count == 2
    assert metrics.expert_count == 1
    assert metrics.expert_density == pytest.approx(1 / 3)


def test_expert_tie_regression_counts_sum_to_n():
    base_score = 0.5
    candidates = [(i, 0.5 + 0.01 * (i - 5)) for i in range(10)]  # mix of above/below/at base
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.expert_count + metrics.tie_count + metrics.regression_count == metrics.n == 10


def test_delta_statistics_correct():
    base_score = 0.5
    candidates = [(1, 0.4), (2, 0.5), (3, 0.6), (4, 0.8)]
    # deltas: -0.1, 0.0, 0.1, 0.3
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.mean_delta == pytest.approx((-0.1 + 0.0 + 0.1 + 0.3) / 4)
    assert metrics.median_delta == pytest.approx((0.0 + 0.1) / 2)
    assert metrics.min_delta == pytest.approx(-0.1)
    assert metrics.max_delta == pytest.approx(0.3)


def test_mean_and_std_score_correct():
    base_score = 0.0
    candidates = [(1, 0.2), (2, 0.4), (3, 0.6), (4, 0.8)]
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.mean_score == pytest.approx(0.5)
    assert metrics.std_score == pytest.approx(0.2236068, abs=1e-6)  # population std of [.2,.4,.6,.8]


def test_score_quantiles_correct():
    base_score = 0.0
    candidates = [(i, float(i)) for i in range(1, 5)]  # scores 1,2,3,4
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.score_quantile_25 == pytest.approx(1.75)
    assert metrics.score_quantile_50 == pytest.approx(2.5)
    assert metrics.score_quantile_75 == pytest.approx(3.25)


def test_best_candidate_picks_max_score_and_matching_seed():
    base_score = 0.0
    candidates = [(11, 0.3), (22, 0.9), (33, 0.5)]
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.best_candidate_score == pytest.approx(0.9)
    assert metrics.best_candidate_seed == 22


def test_best_candidate_tie_picks_first_occurrence_deterministically():
    base_score = 0.0
    candidates = [(1, 0.9), (2, 0.9), (3, 0.1)]
    metrics = aggregate_thicket_run(base_score, candidates)
    assert metrics.best_candidate_seed == 1


def test_expert_density_ci_matches_wilson_formula():
    base_score = 0.5
    candidates = [(i, 0.6) for i in range(5)] + [(i, 0.4) for i in range(5, 10)]  # 5 experts / 10
    metrics = aggregate_thicket_run(base_score, candidates)
    expected_lower, expected_upper = wilson_confidence_interval(5, 10)
    assert metrics.expert_density_ci_lower == pytest.approx(expected_lower)
    assert metrics.expert_density_ci_upper == pytest.approx(expected_upper)


def test_aggregate_thicket_run_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        aggregate_thicket_run(base_score=0.5, candidate_seed_scores=[])


def test_aggregate_uses_the_given_base_score_not_a_hardcoded_one():
    """The exact base score passed in must drive every delta -- proven by re-running with a
    DIFFERENT base_score and checking expert_count/expert_density change accordingly, not by
    reading the implementation.
    """
    candidates = [(1, 0.5), (2, 0.6), (3, 0.7)]
    low_base = aggregate_thicket_run(base_score=0.0, candidate_seed_scores=candidates)
    high_base = aggregate_thicket_run(base_score=0.65, candidate_seed_scores=candidates)
    assert low_base.expert_count == 3  # all candidates beat a base of 0.0
    assert high_base.expert_count == 1  # only 0.7 beats a base of 0.65
    assert low_base.expert_density != high_base.expert_density
