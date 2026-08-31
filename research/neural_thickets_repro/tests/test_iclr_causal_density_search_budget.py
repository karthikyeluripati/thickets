"""Tests for iclr_causal_density.search_budget -- search-budget resampling determinism (item 21).
CPU-only.
"""
from __future__ import annotations

import pytest

from neural_thickets_repro.iclr_causal_density.metrics import CandidateCausalClassification
from neural_thickets_repro.iclr_causal_density.search_budget import (
    CandidatePoolEntry,
    InsufficientPoolSizeError,
    check_registered_divergence,
    monte_carlo_search_budget_analysis,
)


def _pool(n=100, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n):
        selection_score = float(rng.uniform(0, 1))
        # correlate audit real gain loosely with selection score, and let some high-selection
        # candidates be pure shortcuts (G near 0) to exercise the shortcut-fraction axis
        is_shortcut = rng.random() < 0.3
        delta_r = selection_score * 0.5 + rng.normal(0, 0.02)
        g = 0.01 if is_shortcut else delta_r * 0.8
        classification = CandidateCausalClassification(
            candidate_id=f"c{i}", delta_R=delta_r, delta_T=0.0, delta_S=0.0, G=g,
            g_ci_low=(g - 0.05), g_ci_high=(g + 0.05), is_conventional_expert=delta_r > 0,
            is_causally_visual_expert=(delta_r > 0 and (g - 0.05) > 0),
        )
        entries.append(CandidatePoolEntry(candidate_id=f"c{i}", selection_real_score=selection_score, audit=classification))
    return entries


def test_monte_carlo_analysis_is_deterministic_given_seed():
    pool = _pool()
    r1 = monte_carlo_search_budget_analysis(pool, budgets=(10, 25), n_subsamples=200, seed=42)
    r2 = monte_carlo_search_budget_analysis(pool, budgets=(10, 25), n_subsamples=200, seed=42)
    for N in (10, 25):
        assert r1[N] == r2[N]


def test_monte_carlo_analysis_different_seed_differs():
    pool = _pool()
    r1 = monte_carlo_search_budget_analysis(pool, budgets=(10,), n_subsamples=200, seed=1)
    r2 = monte_carlo_search_budget_analysis(pool, budgets=(10,), n_subsamples=200, seed=2)
    assert r1[10] != r2[10]


def test_all_four_frozen_budgets_produce_results():
    pool = _pool(n=100)
    results = monte_carlo_search_budget_analysis(pool, n_subsamples=100, seed=1)
    assert set(results.keys()) == {10, 25, 50, 100}
    for N, point in results.items():
        assert point.N == N
        assert point.n_subsamples == 100


def test_insufficient_pool_size_raises():
    pool = _pool(n=5)
    with pytest.raises(InsufficientPoolSizeError):
        monte_carlo_search_budget_analysis(pool, budgets=(10,), n_subsamples=10, seed=1)


def test_registered_divergence_detects_monotonic_pattern():
    """A synthetic pool constructed so real gain strictly increases with N (top-10 pools at
    larger N include more strong, but shortcut-heavy, candidates) and G/shortcut-fraction
    trend as registered.
    """
    from neural_thickets_repro.iclr_causal_density.search_budget import SearchBudgetPoint

    results = {
        10: SearchBudgetPoint(N=10, n_subsamples=100, mean_top1_real_gain=0.1, mean_top1_G=0.08, mean_top10_real_gain=0.1, mean_top10_G=0.08, mean_top10_grounded_fraction=0.8, mean_top10_shortcut_fraction=0.1),
        100: SearchBudgetPoint(N=100, n_subsamples=100, mean_top1_real_gain=0.3, mean_top1_G=0.05, mean_top10_real_gain=0.3, mean_top10_G=0.05, mean_top10_grounded_fraction=0.3, mean_top10_shortcut_fraction=0.5),
    }
    divergence = check_registered_divergence(results)
    assert divergence["real_gain_increases"] is True
    assert divergence["g_decreases_or_underproportional"] is True
    assert divergence["shortcut_fraction_increases"] is True
    assert divergence["divergence_confirmed"] is True


def test_registered_divergence_fails_when_g_scales_proportionally():
    from neural_thickets_repro.iclr_causal_density.search_budget import SearchBudgetPoint

    results = {
        10: SearchBudgetPoint(N=10, n_subsamples=100, mean_top1_real_gain=0.1, mean_top1_G=0.1, mean_top10_real_gain=0.1, mean_top10_G=0.1, mean_top10_grounded_fraction=1.0, mean_top10_shortcut_fraction=0.0),
        100: SearchBudgetPoint(N=100, n_subsamples=100, mean_top1_real_gain=0.2, mean_top1_G=0.2, mean_top10_real_gain=0.2, mean_top10_G=0.2, mean_top10_grounded_fraction=1.0, mean_top10_shortcut_fraction=0.0),
    }
    divergence = check_registered_divergence(results)
    assert divergence["divergence_confirmed"] is False  # G scaled EXACTLY proportionally with real gain, and shortcut fraction never increased
