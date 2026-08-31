"""Phase 10: the decision gate, implemented in code -- never assigned subjectively. Every
threshold is imported from design.py (frozen before any results exist); this module only
APPLIES them. See design.DECISION_* constants for the frozen numbers.

Precedence (task spec, applied literally):
  1. integrity failure (restoration/isolation/provenance/completeness) -> INCONCLUSIVE, always,
     regardless of any metric.
  2. not exactly 5 capabilities with valid results -> INCONCLUSIVE.
  3. all five CONFIRMED criteria hold -> CONFIRMED.
  4. adequate precision (>=4/5 capabilities have a defined D) but CONFIRMED criteria fail ->
     REJECTED.
  5. otherwise (inadequate precision) -> INCONCLUSIVE.
Never converts inconclusive evidence into confirmation through pooling, and never searches for
an alternative threshold after seeing which branch a given run lands in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .design import (
    DECISION_BUDGET_DIVERGENCE_MIN_CAPABILITIES,
    DECISION_D_CI_EXCLUDES_ONE_MIN_CAPABILITIES,
    DECISION_D_MIN_CAPABILITIES_ABOVE_THRESHOLD,
    DECISION_D_THRESHOLD,
    DECISION_GROUNDED_G_IMPROVEMENT_MIN_CAPABILITIES,
    DECISION_GROUNDED_RETENTION_FRACTION,
)

CONFIRMED = "CONFIRMED"
REJECTED = "REJECTED"
INCONCLUSIVE = "INCONCLUSIVE"
INCONCLUSIVE_EXECUTION_BLOCKED = "INCONCLUSIVE — EXECUTION BLOCKED"
DECISIONS = (CONFIRMED, REJECTED, INCONCLUSIVE, INCONCLUSIVE_EXECUTION_BLOCKED)

_MIN_CAPABILITIES_FOR_ADEQUATE_PRECISION = 4  # of 5 -- fewer defined D's means the evidence cannot support REJECTED either, only INCONCLUSIVE


@dataclass(frozen=True)
class CapabilityGateInputs:
    """One capability's already-computed, capability-pooled (across its 6 scope-radius cells)
    inputs to the gate -- the gate itself does no further metric computation, only comparison
    against the frozen thresholds.
    """
    capability: str
    D: Optional[float]
    D_ci_low: Optional[float]
    D_ci_high: Optional[float]
    search_budget_divergence_confirmed: bool
    grounded_retention_top10: Optional[float]
    grounded_g_improved_top10: bool


@dataclass(frozen=True)
class DecisionGateResult:
    decision: str
    reasons: List[str]
    criteria: Dict[str, object]

    def to_dict(self) -> Dict:
        return {"decision": self.decision, "reasons": self.reasons, "criteria": self.criteria}


def inconclusive_execution_blocked(reason: str) -> DecisionGateResult:
    """The Phase-6 contingency: GPU execution never happened (or was aborted before any
    complete cell existed) in THIS environment -- the gate is never even reached; this is a
    separate, earlier short-circuit, not a metric-based INCONCLUSIVE.
    """
    return DecisionGateResult(decision=INCONCLUSIVE_EXECUTION_BLOCKED, reasons=[reason], criteria={})


def evaluate_decision_gate(capability_inputs: Sequence[CapabilityGateInputs], *, integrity_ok: bool, integrity_reasons: Optional[List[str]] = None) -> DecisionGateResult:
    integrity_reasons = integrity_reasons or []

    if not integrity_ok:
        return DecisionGateResult(INCONCLUSIVE, [f"integrity gate failed: {'; '.join(integrity_reasons) or 'unspecified'}"], {"integrity_ok": False})

    if len(capability_inputs) != 5:
        return DecisionGateResult(INCONCLUSIVE, [f"expected exactly 5 capabilities with valid results, got {len(capability_inputs)}"], {"n_capabilities": len(capability_inputs)})

    n_d_pass = sum(1 for c in capability_inputs if c.D is not None and c.D >= DECISION_D_THRESHOLD)
    criterion_1 = n_d_pass >= DECISION_D_MIN_CAPABILITIES_ABOVE_THRESHOLD

    n_ci_excludes_1 = sum(1 for c in capability_inputs if c.D_ci_low is not None and c.D_ci_low > 1.0)
    criterion_2 = n_ci_excludes_1 >= DECISION_D_CI_EXCLUDES_ONE_MIN_CAPABILITIES

    n_divergence = sum(1 for c in capability_inputs if c.search_budget_divergence_confirmed)
    criterion_3 = n_divergence >= DECISION_BUDGET_DIVERGENCE_MIN_CAPABILITIES

    eligible_for_retention = [c for c in capability_inputs if c.grounded_retention_top10 is not None]
    # "Grounded selection retains at least 80% of standard selection's positive audit real-
    # image gain, evaluated only where standard selection has positive gain" -- applied per
    # capability: EVERY capability where standard has positive gain (eligible_for_retention)
    # must individually retain >= 80%, never merely on average across capabilities (a single
    # capability collapsing to near-zero retention while others compensate on average would
    # not be "retains at least 80%" for that capability).
    criterion_4 = len(eligible_for_retention) > 0 and all(c.grounded_retention_top10 >= DECISION_GROUNDED_RETENTION_FRACTION for c in eligible_for_retention)

    n_g_improved = sum(1 for c in capability_inputs if c.grounded_g_improved_top10)
    criterion_5 = n_g_improved >= DECISION_GROUNDED_G_IMPROVEMENT_MIN_CAPABILITIES

    criteria = {
        "criterion_1_D_threshold": {"pass": criterion_1, "n_capabilities_passing": n_d_pass, "required": DECISION_D_MIN_CAPABILITIES_ABOVE_THRESHOLD, "threshold": DECISION_D_THRESHOLD},
        "criterion_2_D_ci_excludes_1": {"pass": criterion_2, "n_capabilities_passing": n_ci_excludes_1, "required": DECISION_D_CI_EXCLUDES_ONE_MIN_CAPABILITIES},
        "criterion_3_budget_divergence": {"pass": criterion_3, "n_capabilities_passing": n_divergence, "required": DECISION_BUDGET_DIVERGENCE_MIN_CAPABILITIES},
        "criterion_4_grounded_retention": {"pass": criterion_4, "n_eligible": len(eligible_for_retention), "threshold": DECISION_GROUNDED_RETENTION_FRACTION},
        "criterion_5_grounded_g_improved": {"pass": criterion_5, "n_capabilities_passing": n_g_improved, "required": DECISION_GROUNDED_G_IMPROVEMENT_MIN_CAPABILITIES},
    }

    if criterion_1 and criterion_2 and criterion_3 and criterion_4 and criterion_5:
        return DecisionGateResult(CONFIRMED, ["all five CONFIRMED criteria satisfied"], criteria)

    n_defined_D = sum(1 for c in capability_inputs if c.D is not None)
    adequate_precision = n_defined_D >= _MIN_CAPABILITIES_FOR_ADEQUATE_PRECISION

    if not adequate_precision:
        return DecisionGateResult(
            INCONCLUSIVE,
            [f"only {n_defined_D}/5 capabilities have a defined D (rho_visual=0 in the rest) -- too few capabilities carry usable signal to REJECT or CONFIRM"],
            criteria,
        )

    failed = [name for name, c in criteria.items() if not c["pass"]]
    return DecisionGateResult(REJECTED, [f"CONFIRMED criteria failed: {failed}"], criteria)
