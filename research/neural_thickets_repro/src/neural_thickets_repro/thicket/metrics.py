"""Visual-thicket metrics (spec section G) -- operate ENTIRELY on saved result tables (plain
sequences of Delta_t values, or paired base/perturbed score arrays). No model load, no GPU, no
dependency on how the Delta values were produced. Reuses ..thicket_metrics.wilson_confidence_
interval for binomial-proportion CIs (P(Delta > m) is exactly a proportion) rather than
reimplementing it; adds a generic bootstrap CI for non-proportion statistics (e.g. mean delta,
positive thicket mass), which Wilson's interval does not cover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from ..thicket_metrics import wilson_confidence_interval

DEFAULT_QUANTILES: Tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)


def performance_delta(perturbed_score: float, base_score: float) -> float:
    """Delta_t(epsilon) = S_t(theta_0 + epsilon) - S_t(theta_0) (spec A1)."""
    return perturbed_score - base_score


def empirical_density(deltas: Sequence[float], n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """(bin_edges, counts) histogram representation of p_{t,a,r,s}(Delta) (spec A2) -- purely
    for later visualization; the individual `deltas` themselves (never reduced to one summary
    number) remain the actual scientific object, matching spec A2's explicit instruction not
    to immediately collapse a perturbation population to a single average.
    """
    counts, edges = np.histogram(np.asarray(deltas, dtype=float), bins=n_bins)
    return edges, counts


def solution_density(deltas: Sequence[float], margins: Sequence[float]) -> Dict[float, float]:
    """delta_{t,a,r,s}(m) = P[Delta_t >= m] (spec A3) for every margin `m` in `margins` --
    never hardcoded to m=0 only; caller supplies whatever margin grid it needs.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("solution_density requires at least one delta value")
    return {float(m): float(np.mean(arr >= m)) for m in margins}


def solution_density_confidence_interval(deltas: Sequence[float], margin: float, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson CI for the single-margin proportion P[Delta_t >= margin] -- delta(m) is a
    binomial proportion of the sample, so Wilson's interval (already implemented and tested in
    ..thicket_metrics) applies directly; not reimplemented here.
    """
    arr = np.asarray(deltas, dtype=float)
    successes = int(np.sum(arr >= margin))
    return wilson_confidence_interval(successes, len(arr), confidence)


def positive_thicket_mass(deltas: Sequence[float]) -> float:
    """M_{t,a,r,s} = E[max(Delta_t, 0)] (spec A4). Conceptually M = integral_0^inf delta(m) dm
    -- verified numerically (not merely asserted) in tests/test_thicket_metrics.py by comparing
    this direct computation against a fine-grid trapezoidal integration of solution_density().
    """
    arr = np.asarray(deltas, dtype=float)
    return float(np.mean(np.maximum(arr, 0.0)))


def probability_of_improvement(deltas: Sequence[float]) -> float:
    """P(Delta > 0) -- strict inequality, so a Delta == 0 tie never counts as an improvement
    (matches is_candidate_expert()'s own strict criterion below).
    """
    arr = np.asarray(deltas, dtype=float)
    return float(np.mean(arr > 0.0))


def probability_of_degradation(deltas: Sequence[float]) -> float:
    arr = np.asarray(deltas, dtype=float)
    return float(np.mean(arr < 0.0))


def catastrophic_degradation_rate(deltas: Sequence[float], c: float) -> float:
    """P(Delta <= -c). `c` (a positive margin) is REQUIRED, never defaulted -- spec G.7 makes
    this an explicitly configurable statistic, not a frozen constant.
    """
    if c <= 0:
        raise ValueError(f"c must be a positive degradation margin, got {c}")
    arr = np.asarray(deltas, dtype=float)
    return float(np.mean(arr <= -c))


def quantiles(deltas: Sequence[float], qs: Sequence[float] = DEFAULT_QUANTILES) -> Dict[float, float]:
    arr = np.asarray(deltas, dtype=float)
    values = np.quantile(arr, qs)
    return {float(q): float(v) for q, v in zip(qs, values)}


def mean_std(deltas: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(deltas, dtype=float)
    return float(arr.mean()), float(arr.std())


def best_of_n_single_order(deltas: Sequence[float]) -> np.ndarray:
    """max_{i <= N} Delta_i (spec G.8) for the GIVEN order of `deltas` -- e.g. seed/sampling
    order. A single realized draw of the best-of-N curve, not an expectation over orderings;
    use best_of_n_expected() for the (order-independent) statistic usually reported.
    """
    return np.maximum.accumulate(np.asarray(deltas, dtype=float))


def best_of_n_expected(deltas: Sequence[float], n_permutations: int = 200, seed: int = 0) -> np.ndarray:
    """Expected best-of-N curve: average of best_of_n_single_order() over `n_permutations`
    random re-orderings of `deltas` (seeded, deterministic) -- the usual order-independent
    summary of "how good is the best candidate found after evaluating N perturbations",
    removing sensitivity to whatever arbitrary order the candidates happen to be listed in.
    """
    arr = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    curves = np.empty((n_permutations, arr.size), dtype=float)
    for i in range(n_permutations):
        curves[i] = best_of_n_single_order(rng.permutation(arr))
    return curves.mean(axis=0)


def paired_bootstrap_confidence_interval(
    values: Sequence[float], statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2000, seed: int = 0, confidence: float = 0.95,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary statistic of `values` (e.g. mean paired delta,
    positive thicket mass) -- covers the continuous-statistic case Wilson's interval (binomial
    proportions only) does not. Deterministic given `seed`.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("paired_bootstrap_confidence_interval requires at least one value")
    rng = np.random.default_rng(seed)
    n = arr.size
    resample_stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        resample = arr[rng.integers(0, n, size=n)]
        resample_stats[i] = statistic_fn(resample)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(resample_stats, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


# --- A5: candidate vs. confirmed expert terminology ------------------------------------------


def is_candidate_expert(delta: float) -> bool:
    """Candidate expert (spec A5): a strictly positive improvement on mapping/selection data.
    A tie (delta == 0) is never a candidate expert.
    """
    return delta > 0.0


@dataclass(frozen=True)
class ExpertConfirmation:
    is_confirmed: bool
    ci_lower: float
    ci_upper: float
    ci_lower_bound_threshold: float


def confirm_expert(
    held_out_deltas: Sequence[float], *, ci_lower_bound_threshold: float = 0.0,
    confidence: float = 0.95, n_bootstrap: int = 2000, seed: int = 0,
) -> ExpertConfirmation:
    """Confirmed expert (spec A5): the improvement survives held-out confirmation AND a
    statistical-uncertainty check -- specifically, a paired bootstrap CI (over `held_out_deltas`,
    the SAME candidate's Delta values on D_confirm) whose lower bound exceeds
    `ci_lower_bound_threshold`. The threshold defaults to 0.0 (the natural "still an
    improvement after accounting for sampling uncertainty" boundary) and is NOT a frozen,
    tuned final decision rule -- spec A5 explicitly forbids freezing an arbitrary statistical
    threshold at this stage; callers needing a stricter bar pass a larger threshold explicitly.
    Every Delta > 0 candidate is therefore NOT automatically a confirmed expert: this function
    must be called against separate D_confirm data before that label is ever used.
    """
    ci_lower, ci_upper = paired_bootstrap_confidence_interval(held_out_deltas, np.mean, n_bootstrap, seed, confidence)
    return ExpertConfirmation(is_confirmed=ci_lower > ci_lower_bound_threshold, ci_lower=ci_lower, ci_upper=ci_upper, ci_lower_bound_threshold=ci_lower_bound_threshold)
