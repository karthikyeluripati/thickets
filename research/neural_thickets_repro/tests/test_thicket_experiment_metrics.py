import numpy as np
import pytest

from neural_thickets_repro.thicket.metrics import (
    best_of_n_expected,
    best_of_n_single_order,
    catastrophic_degradation_rate,
    confirm_expert,
    is_candidate_expert,
    mean_std,
    paired_bootstrap_confidence_interval,
    performance_delta,
    positive_thicket_mass,
    probability_of_degradation,
    probability_of_improvement,
    quantiles,
    solution_density,
    solution_density_confidence_interval,
)


def test_performance_delta():
    assert performance_delta(perturbed_score=0.7, base_score=0.6) == pytest.approx(0.1)


def test_solution_density_supports_arbitrary_margins():
    deltas = [-0.1, 0.0, 0.05, 0.1, 0.2]
    density = solution_density(deltas, margins=[0.0, 0.05, 0.15])
    assert density[0.0] == pytest.approx(4 / 5)  # >= 0
    assert density[0.05] == pytest.approx(3 / 5)
    assert density[0.15] == pytest.approx(1 / 5)


def test_solution_density_confidence_interval_is_a_valid_wilson_interval():
    deltas = [0.1] * 8 + [-0.1] * 2
    lower, upper = solution_density_confidence_interval(deltas, margin=0.0, confidence=0.95)
    assert 0.0 <= lower <= 0.8 <= upper <= 1.0


def test_positive_thicket_mass():
    deltas = [-1.0, 0.0, 1.0, 3.0]
    assert positive_thicket_mass(deltas) == pytest.approx((0 + 0 + 1 + 3) / 4)


def test_positive_thicket_mass_equals_integral_of_solution_density():
    """Numerically verifies spec A4's documented relationship M = integral_0^inf delta(m) dm,
    rather than merely asserting it in a docstring: fine-grid trapezoidal integration of
    solution_density() over an increasing margin grid should match positive_thicket_mass()
    directly, for a synthetic delta population with a known maximum.
    """
    rng = np.random.default_rng(0)
    deltas = rng.normal(loc=0.1, scale=0.5, size=2000)
    deltas = np.clip(deltas, -5, 5)

    margins = np.linspace(0.0, float(deltas.max()), 4000)
    density = solution_density(deltas, margins=margins)
    density_values = np.array([density[float(m)] for m in margins])
    integrated_mass = np.trapezoid(density_values, margins)

    direct_mass = positive_thicket_mass(deltas)
    assert integrated_mass == pytest.approx(direct_mass, abs=1e-3)


def test_probability_of_improvement_and_degradation_are_strict():
    deltas = [0.0, 0.0, 0.1, -0.1]
    assert probability_of_improvement(deltas) == pytest.approx(1 / 4)
    assert probability_of_degradation(deltas) == pytest.approx(1 / 4)


def test_catastrophic_degradation_rate_requires_positive_margin():
    with pytest.raises(ValueError):
        catastrophic_degradation_rate([0.1, -0.1], c=0.0)


def test_catastrophic_degradation_rate():
    deltas = [-0.5, -0.2, 0.0, 0.3]
    assert catastrophic_degradation_rate(deltas, c=0.3) == pytest.approx(1 / 4)


def test_quantiles_supports_arbitrary_qs():
    deltas = list(range(101))  # 0..100
    q = quantiles(deltas, qs=(0.5, 0.9))
    assert q[0.5] == pytest.approx(50, abs=1)
    assert q[0.9] == pytest.approx(90, abs=1)


def test_mean_std():
    mean, std = mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std([1.0, 2.0, 3.0]))


def test_best_of_n_single_order_is_cumulative_max():
    deltas = [0.1, -0.5, 0.3, 0.2, 0.9]
    result = best_of_n_single_order(deltas)
    assert list(result) == [0.1, 0.1, 0.3, 0.3, 0.9]


def test_best_of_n_expected_is_deterministic_given_seed():
    deltas = [0.1, -0.5, 0.3, 0.2, 0.9, -0.1]
    curve_1 = best_of_n_expected(deltas, n_permutations=50, seed=0)
    curve_2 = best_of_n_expected(deltas, n_permutations=50, seed=0)
    assert np.array_equal(curve_1, curve_2)


def test_best_of_n_expected_final_value_equals_the_global_max():
    deltas = [0.1, -0.5, 0.3, 0.2, 0.9, -0.1]
    curve = best_of_n_expected(deltas, n_permutations=50, seed=0)
    assert curve[-1] == pytest.approx(max(deltas))


def test_best_of_n_expected_is_monotonically_non_decreasing():
    deltas = [0.1, -0.5, 0.3, 0.2, 0.9, -0.1]
    curve = best_of_n_expected(deltas, n_permutations=50, seed=0)
    assert all(curve[i] <= curve[i + 1] + 1e-12 for i in range(len(curve) - 1))


def test_paired_bootstrap_confidence_interval_is_deterministic():
    values = [0.1, 0.2, -0.1, 0.3, 0.0]
    ci_1 = paired_bootstrap_confidence_interval(values, seed=0)
    ci_2 = paired_bootstrap_confidence_interval(values, seed=0)
    assert ci_1 == ci_2


def test_paired_bootstrap_confidence_interval_brackets_the_mean_for_clearly_positive_data():
    values = [1.0, 1.1, 0.9, 1.05, 0.95] * 10
    lower, upper = paired_bootstrap_confidence_interval(values, seed=0)
    assert lower > 0.5
    assert lower <= np.mean(values) <= upper


# --- A5: candidate vs. confirmed expert ---------------------------------------------------


def test_is_candidate_expert_is_strict():
    assert is_candidate_expert(0.01) is True
    assert is_candidate_expert(0.0) is False
    assert is_candidate_expert(-0.01) is False


def test_confirm_expert_confirms_a_clearly_positive_held_out_signal():
    held_out = [0.2, 0.25, 0.22, 0.19, 0.21] * 5
    result = confirm_expert(held_out, seed=0)
    assert result.is_confirmed is True
    assert result.ci_lower > 0.0


def test_confirm_expert_does_not_confirm_a_noisy_near_zero_signal():
    held_out = [0.05, -0.06, 0.04, -0.05, 0.06, -0.04] * 5
    result = confirm_expert(held_out, seed=0)
    assert result.is_confirmed is False


def test_confirm_expert_threshold_is_not_hardcoded_by_the_library():
    """Not every Delta > 0 candidate becomes 'confirmed' just by calling the default -- but
    the threshold itself is a caller-supplied argument, not a frozen internal constant.
    """
    held_out = [0.02] * 10
    lenient = confirm_expert(held_out, ci_lower_bound_threshold=-1.0, seed=0)
    strict = confirm_expert(held_out, ci_lower_bound_threshold=1.0, seed=0)
    assert lenient.is_confirmed is True
    assert strict.is_confirmed is False
