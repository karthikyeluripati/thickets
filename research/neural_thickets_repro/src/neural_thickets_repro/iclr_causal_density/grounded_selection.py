"""Phase 9: standard vs. grounded selection, compared at identical candidate budgets (top-1,
top-10 pool). Ranking uses the SELECTION set only; evaluation uses the (disjoint) AUDIT set
only -- never the reverse, and never a coefficient other than the frozen GROUNDED_COEFFICIENT
(1/2, never tuned).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .design import GROUNDED_COEFFICIENT, TOP_K_POOL_SIZE
from .metrics import CandidateCausalClassification


@dataclass(frozen=True)
class CandidateSelectionData:
    """One candidate's SELECTION-set aggregate scores (never audit-set data) -- the only
    inputs standard/grounded ranking are allowed to see.
    """
    candidate_id: str
    selection_real: float
    selection_text: float
    selection_shuffle: float


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    ranking_score: float
    audit: CandidateCausalClassification


def standard_rank(pool: Sequence[CandidateSelectionData], audit_by_id: Dict[str, CandidateCausalClassification]) -> List[RankedCandidate]:
    ranked = sorted(pool, key=lambda c: c.selection_real, reverse=True)
    return [RankedCandidate(candidate_id=c.candidate_id, ranking_score=c.selection_real, audit=audit_by_id[c.candidate_id]) for c in ranked]


def grounded_rank(
    pool: Sequence[CandidateSelectionData], audit_by_id: Dict[str, CandidateCausalClassification],
    *, base_selection_real: float, base_selection_text: float, base_selection_shuffle: float, coefficient: float = GROUNDED_COEFFICIENT,
) -> List[RankedCandidate]:
    """Grounded ranking score R_i^grounded = Delta_i^R - coefficient*(Delta_i^T + Delta_i^S),
    ALL computed on the SELECTION set (never the audit set -- see module docstring). Eligible
    ONLY when the selection-set Delta_i^R > 0; ineligible candidates are excluded entirely
    (never ranked with a penalized/clamped score) so top-1/top-10 grounded is always drawn from
    a genuinely eligible pool. May legitimately return fewer than requested if too few
    candidates are eligible -- callers must check len(), never assume a full pool.
    """
    eligible = []
    for c in pool:
        delta_r = c.selection_real - base_selection_real
        if delta_r <= 0.0:
            continue
        delta_t = c.selection_text - base_selection_text
        delta_s = c.selection_shuffle - base_selection_shuffle
        score = delta_r - coefficient * (delta_t + delta_s)
        eligible.append((c.candidate_id, score))
    eligible.sort(key=lambda pair: pair[1], reverse=True)
    return [RankedCandidate(candidate_id=cid, ranking_score=score, audit=audit_by_id[cid]) for cid, score in eligible]


@dataclass(frozen=True)
class PoolEvaluation:
    n: int
    mean_audit_real_gain: float
    mean_audit_G: float
    conventional_density: float
    causally_visual_density: float
    shortcut_fraction: float

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def evaluate_pool(ranked: Sequence[RankedCandidate], k: Optional[int] = None) -> Optional[PoolEvaluation]:
    """Evaluates the top-k (or, if k is None/exceeds len(ranked), the WHOLE given list) on
    their already-computed audit-set classification. Returns None for an EMPTY pool (e.g. zero
    grounded-eligible candidates) -- callers must handle this explicitly, never treat it as a
    zero-valued PoolEvaluation.
    """
    top = ranked[:k] if k is not None else list(ranked)
    if not top:
        return None
    n_conventional = sum(1 for c in top if c.audit.is_conventional_expert)
    n_causally_visual = sum(1 for c in top if c.audit.is_causally_visual_expert)
    return PoolEvaluation(
        n=len(top), mean_audit_real_gain=float(np.mean([c.audit.delta_R for c in top])), mean_audit_G=float(np.mean([c.audit.G for c in top])),
        conventional_density=n_conventional / len(top), causally_visual_density=n_causally_visual / len(top),
        shortcut_fraction=(n_conventional - n_causally_visual) / len(top),
    )


def retention_fraction(grounded_eval: Optional[PoolEvaluation], standard_eval: Optional[PoolEvaluation]) -> Optional[float]:
    """Fraction of standard's audit real-image gain that grounded selection retains --
    evaluated ONLY where standard's gain is positive (task spec); returns None when standard's
    gain is <= 0 (retention is undefined, never reported as a fabricated number) or when
    grounded's pool is empty (zero eligible candidates -- retention is 0.0 in that case, a
    real, reportable number, not undefined).
    """
    if standard_eval is None or standard_eval.mean_audit_real_gain <= 0.0:
        return None
    if grounded_eval is None:
        return 0.0
    return grounded_eval.mean_audit_real_gain / standard_eval.mean_audit_real_gain


@dataclass(frozen=True)
class GroundedSelectionComparison:
    top1_standard: Optional[PoolEvaluation]
    top1_grounded: Optional[PoolEvaluation]
    top10_standard: Optional[PoolEvaluation]
    top10_grounded: Optional[PoolEvaluation]
    top1_retention: Optional[float]
    top10_retention: Optional[float]
    top10_g_materially_improved: bool  # grounded's top-10 mean G strictly greater than standard's -- see decision_gate.py for how this feeds the gate

    def to_dict(self) -> Dict:
        return {
            "top1_standard": self.top1_standard.to_dict() if self.top1_standard else None,
            "top1_grounded": self.top1_grounded.to_dict() if self.top1_grounded else None,
            "top10_standard": self.top10_standard.to_dict() if self.top10_standard else None,
            "top10_grounded": self.top10_grounded.to_dict() if self.top10_grounded else None,
            "top1_retention": self.top1_retention, "top10_retention": self.top10_retention,
            "top10_g_materially_improved": self.top10_g_materially_improved,
        }


def compare_standard_vs_grounded(
    pool: Sequence[CandidateSelectionData], audit_by_id: Dict[str, CandidateCausalClassification], *,
    base_selection_real: float, base_selection_text: float, base_selection_shuffle: float, top_k: int = TOP_K_POOL_SIZE,
) -> GroundedSelectionComparison:
    std_ranked = standard_rank(pool, audit_by_id)
    grd_ranked = grounded_rank(pool, audit_by_id, base_selection_real=base_selection_real, base_selection_text=base_selection_text, base_selection_shuffle=base_selection_shuffle)

    top1_std = evaluate_pool(std_ranked, 1)
    top1_grd = evaluate_pool(grd_ranked, 1)
    top10_std = evaluate_pool(std_ranked, top_k)
    top10_grd = evaluate_pool(grd_ranked, top_k)

    g_improved = (top10_grd is not None and top10_std is not None and top10_grd.mean_audit_G > top10_std.mean_audit_G)

    return GroundedSelectionComparison(
        top1_standard=top1_std, top1_grounded=top1_grd, top10_standard=top10_std, top10_grounded=top10_grd,
        top1_retention=retention_fraction(top1_grd, top1_std), top10_retention=retention_fraction(top10_grd, top10_std),
        top10_g_materially_improved=g_improved,
    )
