"""Tests for iclr_causal_density.decision_gate -- synthetic CONFIRMED, REJECTED, and
INCONCLUSIVE decisions (item 22). The gate is implemented in code and applied here against
hand-constructed synthetic capability inputs -- never a real-data test (no GPU results exist in
this environment), which is exactly what proves the GATE LOGIC itself is correct and ready
regardless of whether real data is ever available to feed it.
"""
from __future__ import annotations

from neural_thickets_repro.iclr_causal_density.decision_gate import (
    CONFIRMED,
    INCONCLUSIVE,
    INCONCLUSIVE_EXECUTION_BLOCKED,
    REJECTED,
    CapabilityGateInputs,
    evaluate_decision_gate,
    inconclusive_execution_blocked,
)

_CAPS = ["visual_grounding", "counting", "ocr_text_recognition", "spatial_reasoning", "relational_reasoning"]


def _strong_capability(name):
    return CapabilityGateInputs(
        capability=name, D=3.0, D_ci_low=1.5, D_ci_high=6.0, search_budget_divergence_confirmed=True,
        grounded_retention_top10=0.9, grounded_g_improved_top10=True,
    )


def _weak_capability(name):
    return CapabilityGateInputs(
        capability=name, D=1.2, D_ci_low=0.8, D_ci_high=1.8, search_budget_divergence_confirmed=False,
        grounded_retention_top10=0.5, grounded_g_improved_top10=False,
    )


def test_synthetic_confirmed_when_all_five_criteria_hold_in_all_five_capabilities():
    inputs = [_strong_capability(c) for c in _CAPS]
    result = evaluate_decision_gate(inputs, integrity_ok=True)
    assert result.decision == CONFIRMED
    assert all(v["pass"] for v in result.criteria.values())


def test_synthetic_confirmed_when_exactly_the_minimum_thresholds_are_met():
    """4/5 capabilities pass D>=2, 3/5 have CI excluding 1, 4/5 show budget divergence, 4/5
    show G improvement -- exactly the minimum required counts, still CONFIRMED.
    """
    inputs = [
        CapabilityGateInputs("visual_grounding", D=3.0, D_ci_low=1.5, D_ci_high=6.0, search_budget_divergence_confirmed=True, grounded_retention_top10=0.85, grounded_g_improved_top10=True),
        CapabilityGateInputs("counting", D=2.5, D_ci_low=1.2, D_ci_high=5.0, search_budget_divergence_confirmed=True, grounded_retention_top10=0.9, grounded_g_improved_top10=True),
        CapabilityGateInputs("ocr_text_recognition", D=2.1, D_ci_low=1.1, D_ci_high=3.5, search_budget_divergence_confirmed=True, grounded_retention_top10=0.82, grounded_g_improved_top10=True),
        CapabilityGateInputs("spatial_reasoning", D=2.0, D_ci_low=0.9, D_ci_high=4.0, search_budget_divergence_confirmed=True, grounded_retention_top10=0.81, grounded_g_improved_top10=True),
        CapabilityGateInputs("relational_reasoning", D=1.0, D_ci_low=0.5, D_ci_high=2.0, search_budget_divergence_confirmed=False, grounded_retention_top10=0.80, grounded_g_improved_top10=False),
    ]
    result = evaluate_decision_gate(inputs, integrity_ok=True)
    assert result.decision == CONFIRMED


def test_synthetic_rejected_when_density_gap_fails_consistently():
    inputs = [_weak_capability(c) for c in _CAPS]
    result = evaluate_decision_gate(inputs, integrity_ok=True)
    assert result.decision == REJECTED
    assert result.criteria["criterion_1_D_threshold"]["pass"] is False


def test_synthetic_rejected_when_grounded_retention_fails_in_one_capability():
    """Even with strong D/budget-divergence everywhere, ONE capability retaining < 80% of
    standard's positive gain is enough to fail criterion 4 -- REJECTED, never averaged away.
    """
    inputs = [_strong_capability(c) for c in _CAPS[:-1]]
    weak_retention = CapabilityGateInputs(
        capability=_CAPS[-1], D=3.0, D_ci_low=1.5, D_ci_high=6.0, search_budget_divergence_confirmed=True,
        grounded_retention_top10=0.3, grounded_g_improved_top10=True,  # retention far below 80%
    )
    inputs.append(weak_retention)
    result = evaluate_decision_gate(inputs, integrity_ok=True)
    assert result.decision == REJECTED
    assert result.criteria["criterion_4_grounded_retention"]["pass"] is False


def test_synthetic_inconclusive_on_integrity_failure():
    inputs = [_strong_capability(c) for c in _CAPS]  # even with a perfect metric picture
    result = evaluate_decision_gate(inputs, integrity_ok=False, integrity_reasons=["restoration failed for candidate X"])
    assert result.decision == INCONCLUSIVE
    assert "restoration failed" in result.reasons[0]


def test_synthetic_inconclusive_on_missing_capability():
    inputs = [_strong_capability(c) for c in _CAPS[:4]]  # only 4 of 5
    result = evaluate_decision_gate(inputs, integrity_ok=True)
    assert result.decision == INCONCLUSIVE


def test_synthetic_inconclusive_on_inadequate_precision_too_many_undefined_D():
    """rho_visual=0 (D undefined) in 3/5 capabilities -- only 2/5 carry a defined D, below the
    4/5 threshold required to even attempt REJECTED -- must be INCONCLUSIVE, never REJECTED.
    """
    inputs = [
        _weak_capability(_CAPS[0]),
        CapabilityGateInputs(_CAPS[1], D=None, D_ci_low=None, D_ci_high=None, search_budget_divergence_confirmed=False, grounded_retention_top10=None, grounded_g_improved_top10=False),
        CapabilityGateInputs(_CAPS[2], D=None, D_ci_low=None, D_ci_high=None, search_budget_divergence_confirmed=False, grounded_retention_top10=None, grounded_g_improved_top10=False),
        CapabilityGateInputs(_CAPS[3], D=None, D_ci_low=None, D_ci_high=None, search_budget_divergence_confirmed=False, grounded_retention_top10=None, grounded_g_improved_top10=False),
        _weak_capability(_CAPS[4]),
    ]
    result = evaluate_decision_gate(inputs, integrity_ok=True)
    assert result.decision == INCONCLUSIVE


def test_inconclusive_execution_blocked_is_a_distinct_labeled_decision():
    result = inconclusive_execution_blocked("no GPU available in this environment")
    assert result.decision == INCONCLUSIVE_EXECUTION_BLOCKED
    assert result.decision != INCONCLUSIVE  # distinct string -- never conflated with a metric-based INCONCLUSIVE
    assert "no GPU" in result.reasons[0]


def test_decision_gate_never_returns_a_value_outside_the_four_labels():
    from neural_thickets_repro.iclr_causal_density.decision_gate import DECISIONS

    for inputs in ([_strong_capability(c) for c in _CAPS], [_weak_capability(c) for c in _CAPS], [_strong_capability(c) for c in _CAPS[:2]]):
        result = evaluate_decision_gate(inputs, integrity_ok=True)
        assert result.decision in DECISIONS
    assert evaluate_decision_gate([_strong_capability(c) for c in _CAPS], integrity_ok=False).decision in DECISIONS
