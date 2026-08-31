"""Phase 8: search-budget analysis. N in {10,25,50,100} candidates ranked by SELECTION-set
correct-image score, evaluated on the (disjoint) AUDIT set's already-computed causal
classification (metrics.py). 1,000 deterministic Monte Carlo subsamples per (capability, scope,
radius) cell, drawn from that cell's 100-candidate pool, using the ONE preregistered
SEARCH_BUDGET_ANALYSIS_SEED. Top-10 is a SELECTED POPULATION (never an ensemble/voting method).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .design import N_MONTE_CARLO_SUBSAMPLES, SEARCH_BUDGETS, SEARCH_BUDGET_ANALYSIS_SEED, TOP_K_POOL_SIZE
from .metrics import CandidateCausalClassification


@dataclass(frozen=True)
class CandidatePoolEntry:
    candidate_id: str
    selection_real_score: float  # aggregate correct-image score on the SELECTION set -- ranking signal only
    audit: CandidateCausalClassification  # audit-set evaluation, from metrics.classify_candidate


@dataclass(frozen=True)
class SearchBudgetPoint:
    N: int
    n_subsamples: int
    mean_top1_real_gain: float
    mean_top1_G: float
    mean_top10_real_gain: float
    mean_top10_G: float
    mean_top10_grounded_fraction: float
    mean_top10_shortcut_fraction: float

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class InsufficientPoolSizeError(RuntimeError):
    """A requested search budget N exceeds the candidate pool size for this cell."""


def monte_carlo_search_budget_analysis(
    pool: Sequence[CandidatePoolEntry], *, budgets: Sequence[int] = SEARCH_BUDGETS,
    n_subsamples: int = N_MONTE_CARLO_SUBSAMPLES, seed: int = SEARCH_BUDGET_ANALYSIS_SEED, top_k_pool_size: int = TOP_K_POOL_SIZE,
) -> Dict[int, SearchBudgetPoint]:
    n_pool = len(pool)
    for N in budgets:
        if N > n_pool:
            raise InsufficientPoolSizeError(f"Requested search budget N={N} exceeds candidate pool size {n_pool}.")

    rng = np.random.default_rng(seed)
    results: Dict[int, SearchBudgetPoint] = {}
    for N in budgets:
        top1_real, top1_g = [], []
        top10_real, top10_g, top10_grounded_frac, top10_shortcut_frac = [], [], [], []
        for _ in range(n_subsamples):
            idx = rng.choice(n_pool, size=N, replace=False)
            subsample = [pool[i] for i in idx]
            ranked = sorted(subsample, key=lambda e: e.selection_real_score, reverse=True)
            top1 = ranked[0]
            top10 = ranked[: min(top_k_pool_size, len(ranked))]

            top1_real.append(top1.audit.delta_R)
            top1_g.append(top1.audit.G)
            top10_real.append(float(np.mean([e.audit.delta_R for e in top10])))
            top10_g.append(float(np.mean([e.audit.G for e in top10])))
            n_conventional = sum(1 for e in top10 if e.audit.is_conventional_expert)
            n_grounded = sum(1 for e in top10 if e.audit.is_causally_visual_expert)
            n_shortcut = n_conventional - n_grounded  # conventional expert that FAILS the causal-visual criterion, by definition
            top10_grounded_frac.append(n_grounded / len(top10))
            top10_shortcut_frac.append(n_shortcut / len(top10))

        results[N] = SearchBudgetPoint(
            N=N, n_subsamples=n_subsamples,
            mean_top1_real_gain=float(np.mean(top1_real)), mean_top1_G=float(np.mean(top1_g)),
            mean_top10_real_gain=float(np.mean(top10_real)), mean_top10_G=float(np.mean(top10_g)),
            mean_top10_grounded_fraction=float(np.mean(top10_grounded_frac)), mean_top10_shortcut_fraction=float(np.mean(top10_shortcut_frac)),
        )
    return results


def check_registered_divergence(results: Dict[int, SearchBudgetPoint]) -> Dict[str, bool]:
    """The REGISTERED divergence (task spec, frozen before any results): comparing the
    SMALLEST vs LARGEST search budget (never a mid-range comparison chosen after seeing data):
      - audit real gain increases
      - G decreases OR fails to increase proportionally (increases by a strictly smaller
        relative amount than real gain does -- covers "increases less than proportionally"
        without requiring an outright decrease)
      - top-10 shortcut fraction increases
    All three sub-checks must hold for `divergence_confirmed` to be True.
    """
    budgets_sorted = sorted(results)
    smallest, largest = results[budgets_sorted[0]], results[budgets_sorted[-1]]

    real_gain_increases = largest.mean_top10_real_gain > smallest.mean_top10_real_gain
    if smallest.mean_top10_real_gain > 0:
        real_gain_relative_increase = (largest.mean_top10_real_gain - smallest.mean_top10_real_gain) / smallest.mean_top10_real_gain
    else:
        real_gain_relative_increase = float("inf") if real_gain_increases else 0.0
    g_decreases_or_underproportional = (
        largest.mean_top10_G <= smallest.mean_top10_G
        or (smallest.mean_top10_G > 0 and (largest.mean_top10_G - smallest.mean_top10_G) / smallest.mean_top10_G < real_gain_relative_increase)
        or (smallest.mean_top10_G <= 0 and largest.mean_top10_G <= smallest.mean_top10_G)
    )
    shortcut_fraction_increases = largest.mean_top10_shortcut_fraction > smallest.mean_top10_shortcut_fraction

    return {
        "real_gain_increases": real_gain_increases,
        "g_decreases_or_underproportional": g_decreases_or_underproportional,
        "shortcut_fraction_increases": shortcut_fraction_increases,
        "divergence_confirmed": real_gain_increases and g_decreases_or_underproportional and shortcut_fraction_increases,
    }
