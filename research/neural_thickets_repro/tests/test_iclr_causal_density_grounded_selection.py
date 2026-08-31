"""Tests for iclr_causal_density.grounded_selection -- Phase 9 standard-vs-grounded ranking.
CPU-only.
"""
from __future__ import annotations

from neural_thickets_repro.iclr_causal_density.grounded_selection import (
    CandidateSelectionData,
    compare_standard_vs_grounded,
    evaluate_pool,
    grounded_rank,
    retention_fraction,
    standard_rank,
)
from neural_thickets_repro.iclr_causal_density.metrics import CandidateCausalClassification


def _classification(cid, delta_r, g, conventional=None, causal=None):
    conventional = delta_r > 0 if conventional is None else conventional
    causal = (conventional and g > 0) if causal is None else causal
    return CandidateCausalClassification(cid, delta_r, 0.0, 0.0, g, g - 0.05, g + 0.05, conventional, causal)


def test_standard_rank_orders_by_selection_real_score():
    pool = [
        CandidateSelectionData("a", selection_real=0.5, selection_text=0.5, selection_shuffle=0.5),
        CandidateSelectionData("b", selection_real=0.9, selection_text=0.5, selection_shuffle=0.5),
        CandidateSelectionData("c", selection_real=0.3, selection_text=0.5, selection_shuffle=0.5),
    ]
    audit = {"a": _classification("a", 0.1, 0.1), "b": _classification("b", 0.1, 0.1), "c": _classification("c", 0.1, 0.1)}
    ranked = standard_rank(pool, audit)
    assert [r.candidate_id for r in ranked] == ["b", "a", "c"]


def test_grounded_rank_excludes_ineligible_candidates():
    """A candidate whose SELECTION-set real gain is <= 0 is never eligible for grounded
    ranking, even if it has a high raw selection_real score in absolute terms.
    """
    pool = [
        CandidateSelectionData("shortcut", selection_real=0.9, selection_text=0.9, selection_shuffle=0.9),  # equal gain everywhere: eligible (delta_R>0) but low grounded score
        CandidateSelectionData("no_gain", selection_real=0.4, selection_text=0.4, selection_shuffle=0.4),   # base itself -- delta_R == 0, ineligible
        CandidateSelectionData("visual", selection_real=0.9, selection_text=0.4, selection_shuffle=0.4),    # genuinely visual: eligible, high grounded score
    ]
    audit = {c.candidate_id: _classification(c.candidate_id, 0.1, 0.1) for c in pool}
    ranked = grounded_rank(pool, audit, base_selection_real=0.4, base_selection_text=0.4, base_selection_shuffle=0.4)
    ids = [r.candidate_id for r in ranked]
    assert "no_gain" not in ids  # delta_R == 0 -> excluded
    assert ids[0] == "visual"  # highest grounded score ranks first


def test_grounded_rank_returns_empty_when_no_candidate_is_eligible():
    pool = [CandidateSelectionData("a", selection_real=0.4, selection_text=0.4, selection_shuffle=0.4)]
    audit = {"a": _classification("a", 0.1, 0.1)}
    ranked = grounded_rank(pool, audit, base_selection_real=0.4, base_selection_text=0.4, base_selection_shuffle=0.4)
    assert ranked == []
    assert evaluate_pool(ranked, 1) is None


def test_retention_fraction_undefined_when_standard_gain_not_positive():
    from neural_thickets_repro.iclr_causal_density.grounded_selection import PoolEvaluation

    standard = PoolEvaluation(n=1, mean_audit_real_gain=-0.1, mean_audit_G=0.0, conventional_density=0.0, causally_visual_density=0.0, shortcut_fraction=0.0)
    grounded = PoolEvaluation(n=1, mean_audit_real_gain=0.05, mean_audit_G=0.05, conventional_density=1.0, causally_visual_density=1.0, shortcut_fraction=0.0)
    assert retention_fraction(grounded, standard) is None


def test_retention_fraction_zero_when_grounded_pool_empty_but_standard_positive():
    from neural_thickets_repro.iclr_causal_density.grounded_selection import PoolEvaluation

    standard = PoolEvaluation(n=1, mean_audit_real_gain=0.2, mean_audit_G=0.1, conventional_density=1.0, causally_visual_density=0.0, shortcut_fraction=1.0)
    assert retention_fraction(None, standard) == 0.0


def test_compare_standard_vs_grounded_end_to_end():
    pool = [
        CandidateSelectionData("shortcut", selection_real=0.9, selection_text=0.9, selection_shuffle=0.9),
        CandidateSelectionData("visual", selection_real=0.9, selection_text=0.4, selection_shuffle=0.4),
    ]
    audit = {
        "shortcut": _classification("shortcut", 0.1, 0.0, conventional=True, causal=False),
        "visual": _classification("visual", 0.3, 0.3, conventional=True, causal=True),
    }
    comparison = compare_standard_vs_grounded(pool, audit, base_selection_real=0.4, base_selection_text=0.4, base_selection_shuffle=0.4, top_k=2)
    assert comparison.top1_standard is not None
    assert comparison.top1_grounded is not None
    assert comparison.top1_grounded.mean_audit_G >= comparison.top1_standard.mean_audit_G  # grounded should prefer the genuinely visual candidate
