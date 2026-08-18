"""Thicket measurement: turns a run's (base_score, per-candidate scores) into the scientific
quantities this experiment actually cares about --

    rho_{m,t}(r) = (1/N) * sum_i 1[S(theta + Delta_{m,i}) > S(theta)]

(expert density) and its distributional context, not merely Top-1/ensemble accuracy. Pure
Python/numpy, no GPU/ray/vllm dependency -- CPU-testable directly against synthetic
(seed, score) lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# Wilson score interval z-values by confidence level. Only 95% is needed for this milestone
# ("a 95% binomial confidence interval ... Wilson interval is fine") -- extend this table
# if another confidence level is needed later rather than pulling in a scipy dependency this
# project doesn't otherwise have for a single constant.
_WILSON_Z_SCORES = {0.95: 1.959963984540054}


def wilson_confidence_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (successes/n) -- preferred over the
    naive normal approximation because it stays well-behaved (non-degenerate, bounded in
    [0, 1]) even at extreme proportions (e.g. successes=0).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, n], got successes={successes}, n={n}")
    if confidence not in _WILSON_Z_SCORES:
        raise ValueError(f"Unsupported confidence level {confidence!r}, only {list(_WILSON_Z_SCORES)} implemented")

    z = _WILSON_Z_SCORES[confidence]
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adjustment = z * ((p * (1 - p) + z ** 2 / (4 * n)) / n) ** 0.5
    lower = (centre - adjustment) / denom
    upper = (centre + adjustment) / denom
    return max(0.0, lower), min(1.0, upper)


def compute_delta_fields(candidate_score: float, base_score: float) -> Tuple[float, bool, bool]:
    """(delta_score, is_expert, is_tie). is_expert is strictly delta > 0 -- a tie (delta == 0)
    is never counted as an expert, matching rho_{m,t}(r)'s strict ">" in the formula.
    """
    delta = candidate_score - base_score
    return delta, delta > 0.0, delta == 0.0


@dataclass
class ThicketRunMetrics:
    n: int
    base_score: float
    expert_count: int
    tie_count: int
    regression_count: int
    expert_density: float
    expert_density_ci_lower: float
    expert_density_ci_upper: float
    mean_score: float
    std_score: float
    mean_delta: float
    median_delta: float
    min_delta: float
    max_delta: float
    score_quantile_25: float
    score_quantile_50: float
    score_quantile_75: float
    best_candidate_score: float
    best_candidate_seed: int


def aggregate_thicket_run(
    base_score: float,
    candidate_seed_scores: Sequence[Tuple[int, float]],
    confidence: float = 0.95,
) -> ThicketRunMetrics:
    """candidate_seed_scores: [(seed, score), ...], one per completed candidate, in any order
    -- expert_count/tie_count/regression_count are computed independently per candidate
    against the SAME base_score, so ordering doesn't matter for any of these statistics
    (best_candidate_seed picks the first candidate achieving the max score, for determinism
    under ties).
    """
    if not candidate_seed_scores:
        raise ValueError("aggregate_thicket_run requires at least one candidate")

    n = len(candidate_seed_scores)
    seeds = [s for s, _ in candidate_seed_scores]
    scores = np.array([s for _, s in candidate_seed_scores], dtype=float)
    deltas = scores - base_score

    expert_count = int(np.sum(deltas > 0.0))
    tie_count = int(np.sum(deltas == 0.0))
    regression_count = int(np.sum(deltas < 0.0))
    assert expert_count + tie_count + regression_count == n, "expert/tie/regression counts must partition N"

    expert_density = expert_count / n
    ci_lower, ci_upper = wilson_confidence_interval(expert_count, n, confidence)

    q25, q50, q75 = np.percentile(scores, [25, 50, 75])
    best_idx = int(np.argmax(scores))

    return ThicketRunMetrics(
        n=n,
        base_score=base_score,
        expert_count=expert_count,
        tie_count=tie_count,
        regression_count=regression_count,
        expert_density=expert_density,
        expert_density_ci_lower=ci_lower,
        expert_density_ci_upper=ci_upper,
        mean_score=float(scores.mean()),
        std_score=float(scores.std()),
        mean_delta=float(deltas.mean()),
        median_delta=float(np.median(deltas)),
        min_delta=float(deltas.min()),
        max_delta=float(deltas.max()),
        score_quantile_25=float(q25),
        score_quantile_50=float(q50),
        score_quantile_75=float(q75),
        best_candidate_score=float(scores[best_idx]),
        best_candidate_seed=int(seeds[best_idx]),
    )
