"""Tests for iclr_causal_density.metrics -- metric calculations (item 18), paired bootstrap
confidence intervals (item 19), zero visual-density handling (item 20). CPU-only, pure numpy.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_thickets_repro.iclr_causal_density.metrics import (
    CandidateCausalClassification,
    ConditionScores,
    bootstrap_density_ratio_ci,
    bootstrap_g_distribution,
    build_resample_index_matrix,
    classify_candidate,
    compute_density_ratio,
    percentile_ci,
    point_deltas,
)


def _scores(real, text, shuffle):
    return ConditionScores(real=np.array(real, dtype=float), text=np.array(text, dtype=float), shuffle=np.array(shuffle, dtype=float))


# =================================================================================================
# Item 18: metric calculations
# =================================================================================================


def test_point_deltas_hand_computed():
    base = _scores(real=[0.4, 0.4, 0.4, 0.4], text=[0.2, 0.2, 0.2, 0.2], shuffle=[0.2, 0.2, 0.2, 0.2])
    candidate = _scores(real=[0.6, 0.6, 0.6, 0.6], text=[0.25, 0.25, 0.25, 0.25], shuffle=[0.2, 0.2, 0.2, 0.2])
    result = point_deltas(candidate, base)
    assert result["delta_R"] == pytest.approx(0.2)
    assert result["delta_T"] == pytest.approx(0.05)
    assert result["delta_S"] == pytest.approx(0.0)
    # G = delta_R - 0.5*(delta_T + delta_S) = 0.2 - 0.5*(0.05 + 0.0) = 0.175
    assert result["G"] == pytest.approx(0.175)


def test_conventional_expert_requires_positive_delta_r():
    base = _scores([0.5] * 10, [0.5] * 10, [0.5] * 10)
    worse = _scores([0.3] * 10, [0.5] * 10, [0.5] * 10)
    idx = build_resample_index_matrix(10, n_resamples=200, seed=1)
    result = classify_candidate("c1", worse, base, idx)
    assert result.is_conventional_expert is False
    assert result.is_causally_visual_expert is False  # can never be causally visual without being conventional


def test_causally_visual_expert_requires_ci_low_g_positive():
    """A candidate whose real gain is entirely explained by shuffled/text-only gains (a pure
    shortcut) must NOT be classified causally visual, even though it IS conventional.
    """
    rng = np.random.default_rng(0)
    n = 200
    base = _scores(real=rng.normal(0.4, 0.05, n), text=rng.normal(0.4, 0.05, n), shuffle=rng.normal(0.4, 0.05, n))
    # shortcut: candidate improves EQUALLY under all three conditions (image-independent gain)
    shortcut = _scores(real=base.real + 0.1, text=base.text + 0.1, shuffle=base.shuffle + 0.1)
    idx = build_resample_index_matrix(n, n_resamples=2000, seed=1)
    result = classify_candidate("shortcut", shortcut, base, idx)
    assert result.is_conventional_expert is True
    assert result.is_causally_visual_expert is False  # G ~= 0, CI should straddle zero


def test_causally_visual_expert_when_real_gain_is_genuinely_visual():
    rng = np.random.default_rng(0)
    n = 200
    base = _scores(real=rng.normal(0.4, 0.05, n), text=rng.normal(0.4, 0.05, n), shuffle=rng.normal(0.4, 0.05, n))
    # genuinely visual: candidate improves ONLY under correct-image, not shuffled/text-only
    visual = _scores(real=base.real + 0.3, text=base.text.copy(), shuffle=base.shuffle.copy())
    idx = build_resample_index_matrix(n, n_resamples=2000, seed=1)
    result = classify_candidate("visual", visual, base, idx)
    assert result.is_conventional_expert is True
    assert result.is_causally_visual_expert is True


def test_compute_density_ratio_basic_counts():
    classifications = [
        CandidateCausalClassification("c1", 0.1, 0.0, 0.0, 0.1, 0.05, 0.15, True, True),
        CandidateCausalClassification("c2", 0.1, 0.1, 0.1, 0.0, -0.05, 0.05, True, False),
        CandidateCausalClassification("c3", -0.1, 0.0, 0.0, -0.1, -0.15, -0.05, False, False),
    ]
    result = compute_density_ratio(classifications)
    assert result.n_candidates == 3
    assert result.n_conventional == 2
    assert result.n_causally_visual == 1
    assert result.rho_standard == pytest.approx(2 / 3)
    assert result.rho_visual == pytest.approx(1 / 3)
    assert result.D == pytest.approx(2.0)
    assert result.zero_visual_density is False


# =================================================================================================
# Item 19: paired bootstrap confidence intervals
# =================================================================================================


def test_resample_matrix_deterministic_given_seed():
    m1 = build_resample_index_matrix(50, n_resamples=100, seed=7)
    m2 = build_resample_index_matrix(50, n_resamples=100, seed=7)
    assert np.array_equal(m1, m2)


def test_resample_matrix_different_seed_differs():
    m1 = build_resample_index_matrix(50, n_resamples=100, seed=7)
    m2 = build_resample_index_matrix(50, n_resamples=100, seed=8)
    assert not np.array_equal(m1, m2)


def test_percentile_ci_basic():
    values = np.arange(1, 101, dtype=float)  # 1..100
    lo, hi = percentile_ci(values, level=0.95)
    assert lo == pytest.approx(3.475, abs=0.5)
    assert hi == pytest.approx(97.525, abs=0.5)
    assert lo < hi


def test_bootstrap_g_distribution_deterministic_given_shared_matrix():
    base = _scores([0.4] * 20, [0.4] * 20, [0.4] * 20)
    candidate = _scores([0.6] * 20, [0.45] * 20, [0.4] * 20)
    idx = build_resample_index_matrix(20, n_resamples=500, seed=3)
    dist1 = bootstrap_g_distribution(candidate, base, idx)
    dist2 = bootstrap_g_distribution(candidate, base, idx)
    assert np.array_equal(dist1, dist2)
    assert len(dist1) == 500


def test_bootstrap_g_ci_narrows_with_larger_n():
    rng = np.random.default_rng(1)
    base_small = _scores(rng.normal(0.4, 0.1, 20), rng.normal(0.4, 0.1, 20), rng.normal(0.4, 0.1, 20))
    cand_small = _scores(base_small.real + 0.1, base_small.text, base_small.shuffle)
    idx_small = build_resample_index_matrix(20, n_resamples=2000, seed=5)
    dist_small = bootstrap_g_distribution(cand_small, base_small, idx_small)

    base_large = _scores(np.tile(base_small.real, 10), np.tile(base_small.text, 10), np.tile(base_small.shuffle, 10))
    cand_large = _scores(base_large.real + 0.1, base_large.text, base_large.shuffle)
    idx_large = build_resample_index_matrix(200, n_resamples=2000, seed=5)
    dist_large = bootstrap_g_distribution(cand_large, base_large, idx_large)

    assert np.std(dist_large) < np.std(dist_small)  # more (tiled, but still bootstrap-resampled) examples -> narrower CI


# =================================================================================================
# Item 20: zero visual-density handling
# =================================================================================================


def test_zero_visual_density_reported_without_epsilon_smoothing():
    classifications = [
        CandidateCausalClassification("c1", 0.1, 0.1, 0.1, 0.0, -0.1, 0.1, True, False),
        CandidateCausalClassification("c2", 0.2, 0.2, 0.2, 0.0, -0.1, 0.1, True, False),
    ]
    result = compute_density_ratio(classifications)
    assert result.n_causally_visual == 0
    assert result.rho_visual == 0.0
    assert result.D is None  # never an arbitrary large number or epsilon-smoothed ratio
    assert result.zero_visual_density is True
    assert result.rho_standard == 1.0  # rho_standard is still a real, reportable number


def test_bootstrap_density_ratio_ci_undefined_when_all_resamples_have_zero_visual_density():
    rng = np.random.default_rng(0)
    n = 50
    base = _scores(rng.normal(0.4, 0.05, n), rng.normal(0.4, 0.05, n), rng.normal(0.4, 0.05, n))
    # every candidate is a pure shortcut with a ROBUSTLY negative G (never borderline-zero,
    # so floating-point summation-order noise across different resamples can never accidentally
    # push a handful of resamples to the positive side of zero)
    shortcuts = {
        f"c{i}": _scores(base.real + 0.1, base.text + 0.3, base.shuffle + 0.3) for i in range(5)
    }
    idx = build_resample_index_matrix(n, n_resamples=500, seed=2)
    ci_low, ci_high, dist = bootstrap_density_ratio_ci(shortcuts, base, idx)
    assert ci_low is None and ci_high is None
    assert len(dist) == 0


def test_bootstrap_density_ratio_ci_defined_when_some_resamples_have_nonzero_visual_density():
    rng = np.random.default_rng(0)
    n = 200
    base = _scores(rng.normal(0.4, 0.05, n), rng.normal(0.4, 0.05, n), rng.normal(0.4, 0.05, n))
    candidates = {
        "visual_1": _scores(base.real + 0.3, base.text.copy(), base.shuffle.copy()),
        "visual_2": _scores(base.real + 0.3, base.text.copy(), base.shuffle.copy()),
        "shortcut_1": _scores(base.real + 0.1, base.text + 0.1, base.shuffle + 0.1),
        "shortcut_2": _scores(base.real + 0.1, base.text + 0.1, base.shuffle + 0.1),
    }
    idx = build_resample_index_matrix(n, n_resamples=1000, seed=2)
    ci_low, ci_high, dist = bootstrap_density_ratio_ci(candidates, base, idx)
    assert ci_low is not None and ci_high is not None
    assert ci_low <= ci_high
    assert len(dist) > 0
