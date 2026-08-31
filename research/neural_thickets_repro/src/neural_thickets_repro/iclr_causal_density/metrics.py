"""Frozen Phase-7 metrics: per-candidate Delta^R/Delta^T/Delta^S/G_i, paired bootstrap
confidence intervals, rho_standard/rho_visual/D. See design.PREREGISTERED_BOOTSTRAP_METHOD_NOTE
for the exact (preregistered, never-altered-after-seeing-data) bootstrap convention this module
implements. Pure numpy -- no GPU/ray/vllm dependency, fully unit-testable against synthetic
per-example score arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .design import BOOTSTRAP_ANALYSIS_SEED, BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, GROUNDED_COEFFICIENT


def build_resample_index_matrix(n_examples: int, *, n_resamples: int = BOOTSTRAP_N_RESAMPLES, seed: int = BOOTSTRAP_ANALYSIS_SEED) -> np.ndarray:
    """Shape (n_resamples, n_examples) of example indices drawn WITH replacement -- the ONE
    shared matrix every candidate's (and the population-level D's) bootstrap CI is computed
    from. Deterministic given (n_examples, n_resamples, seed).
    """
    if n_examples <= 0:
        raise ValueError(f"n_examples must be positive, got {n_examples}")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_examples, size=(n_resamples, n_examples))


@dataclass(frozen=True)
class ConditionScores:
    """Per-example scores (arrays of length n_audit, in matching example order) for one
    entity (base model or one candidate) under all three visual conditions.
    """
    real: np.ndarray
    text: np.ndarray
    shuffle: np.ndarray

    def __post_init__(self):
        n = len(self.real)
        if not (len(self.text) == n and len(self.shuffle) == n):
            raise ValueError(f"ConditionScores arrays must all be the same length, got real={len(self.real)} text={len(self.text)} shuffle={len(self.shuffle)}")


def point_deltas(candidate: ConditionScores, base: ConditionScores) -> Dict[str, float]:
    delta_r = float(np.mean(candidate.real) - np.mean(base.real))
    delta_t = float(np.mean(candidate.text) - np.mean(base.text))
    delta_s = float(np.mean(candidate.shuffle) - np.mean(base.shuffle))
    g = delta_r - GROUNDED_COEFFICIENT * (delta_t + delta_s)
    return {"delta_R": delta_r, "delta_T": delta_t, "delta_S": delta_s, "G": g}


def bootstrap_g_distribution(candidate: ConditionScores, base: ConditionScores, resample_indices: np.ndarray) -> np.ndarray:
    """Returns the array of G_i^(b) for b=1..B, from the shared resample_indices matrix
    (shape (B, n_examples)) -- vectorized: no Python-level loop over resamples.
    """
    cand_real, cand_text, cand_shuffle = candidate.real[resample_indices], candidate.text[resample_indices], candidate.shuffle[resample_indices]
    base_real, base_text, base_shuffle = base.real[resample_indices], base.text[resample_indices], base.shuffle[resample_indices]
    delta_r_b = cand_real.mean(axis=1) - base_real.mean(axis=1)
    delta_t_b = cand_text.mean(axis=1) - base_text.mean(axis=1)
    delta_s_b = cand_shuffle.mean(axis=1) - base_shuffle.mean(axis=1)
    return delta_r_b - GROUNDED_COEFFICIENT * (delta_t_b + delta_s_b)


def bootstrap_delta_r_distribution(candidate: ConditionScores, base: ConditionScores, resample_indices: np.ndarray) -> np.ndarray:
    cand_real, base_real = candidate.real[resample_indices], base.real[resample_indices]
    return cand_real.mean(axis=1) - base_real.mean(axis=1)


def percentile_ci(values: np.ndarray, *, level: float = BOOTSTRAP_CI_LEVEL) -> "tuple[float, float]":
    alpha = 1.0 - level
    lo = float(np.percentile(values, 100 * (alpha / 2)))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


@dataclass(frozen=True)
class CandidateCausalClassification:
    candidate_id: str
    delta_R: float
    delta_T: float
    delta_S: float
    G: float
    g_ci_low: float
    g_ci_high: float
    is_conventional_expert: bool
    is_causally_visual_expert: bool


def classify_candidate(candidate_id: str, candidate: ConditionScores, base: ConditionScores, resample_indices: np.ndarray) -> CandidateCausalClassification:
    point = point_deltas(candidate, base)
    g_dist = bootstrap_g_distribution(candidate, base, resample_indices)
    ci_low, ci_high = percentile_ci(g_dist)
    is_conventional = point["delta_R"] > 0.0
    is_causally_visual = is_conventional and ci_low > 0.0
    return CandidateCausalClassification(
        candidate_id=candidate_id, delta_R=point["delta_R"], delta_T=point["delta_T"], delta_S=point["delta_S"], G=point["G"],
        g_ci_low=ci_low, g_ci_high=ci_high, is_conventional_expert=is_conventional, is_causally_visual_expert=is_causally_visual,
    )


@dataclass(frozen=True)
class DensityRatioResult:
    n_candidates: int
    n_conventional: int
    n_causally_visual: int
    rho_standard: float
    rho_visual: float
    D: Optional[float]                # None iff rho_visual == 0 -- never epsilon-smoothed
    D_ci_low: Optional[float]
    D_ci_high: Optional[float]
    zero_visual_density: bool

    def to_dict(self) -> Dict:
        return {
            "n_candidates": self.n_candidates, "n_conventional": self.n_conventional, "n_causally_visual": self.n_causally_visual,
            "rho_standard": self.rho_standard, "rho_visual": self.rho_visual, "D": self.D,
            "D_ci_low": self.D_ci_low, "D_ci_high": self.D_ci_high, "zero_visual_density": self.zero_visual_density,
        }


def compute_density_ratio(classifications: Sequence[CandidateCausalClassification]) -> DensityRatioResult:
    n = len(classifications)
    if n == 0:
        raise ValueError("compute_density_ratio requires at least one candidate classification")
    n_conventional = sum(1 for c in classifications if c.is_conventional_expert)
    n_causally_visual = sum(1 for c in classifications if c.is_causally_visual_expert)
    rho_standard = n_conventional / n
    rho_visual = n_causally_visual / n
    zero_visual = rho_visual == 0.0
    d = None if zero_visual else rho_standard / rho_visual
    return DensityRatioResult(
        n_candidates=n, n_conventional=n_conventional, n_causally_visual=n_causally_visual,
        rho_standard=rho_standard, rho_visual=rho_visual, D=d, D_ci_low=None, D_ci_high=None, zero_visual_density=zero_visual,
    )


def bootstrap_density_ratio_ci(
    candidate_scores: Dict[str, ConditionScores], base: ConditionScores, resample_indices: np.ndarray, *, level: float = BOOTSTRAP_CI_LEVEL,
) -> "tuple[Optional[float], Optional[float], np.ndarray]":
    """Population-level D's own 95% CI via the PLUG-IN convention documented in design.
    PREREGISTERED_BOOTSTRAP_METHOD_NOTE: for each resample b, a candidate's per-resample
    classification uses THAT resample's own point Delta_i^R(b)>0 / G_i^(b)>0 (never a nested
    per-resample bootstrap-of-bootstrap). Returns (ci_low, ci_high, D_distribution) --
    ci_low/ci_high are None whenever every single resample has rho_visual^(b)==0 (D undefined
    everywhere in the distribution); D_distribution excludes undefined (rho_visual^(b)==0)
    resamples from the percentile computation, and its own length is reported by the caller
    (n_valid_resamples) so a mostly-undefined distribution is visible, never silently treated
    as a full-precision estimate.
    """
    delta_r_matrix = np.stack([bootstrap_delta_r_distribution(cs, base, resample_indices) for cs in candidate_scores.values()], axis=1)  # (B, n_candidates)
    g_matrix = np.stack([bootstrap_g_distribution(cs, base, resample_indices) for cs in candidate_scores.values()], axis=1)  # (B, n_candidates)

    is_conventional_b = delta_r_matrix > 0.0
    is_causally_visual_b = is_conventional_b & (g_matrix > 0.0)
    n_candidates = delta_r_matrix.shape[1]
    rho_standard_b = is_conventional_b.sum(axis=1) / n_candidates
    rho_visual_b = is_causally_visual_b.sum(axis=1) / n_candidates

    valid = rho_visual_b > 0.0
    if not np.any(valid):
        return None, None, np.array([])
    d_b = rho_standard_b[valid] / rho_visual_b[valid]
    ci_low, ci_high = percentile_ci(d_b, level=level)
    return ci_low, ci_high, d_b
