"""POST-HOC ADDENDUM to the frozen preregistration -- frozen 2026-09-03, BEFORE the full 6-cell
(scope x radius) search-budget analysis existed for any capability (only vision_encoder's 2
cells were complete for any capability at decision time; the remaining 4 cells' results were
not yet known). User-authorized, recorded here as tested code rather than prose, per this
project's own "implemented in code -- never assigned subjectively" discipline (decision_gate.py's
own docstring).

GAP THIS FILLS: search_budget.py's own docstring is explicit that Monte Carlo search-budget
analysis and its registered-divergence check run PER (capability, scope, radius) CELL -- "1,000
deterministic Monte Carlo subsamples per (capability, scope, radius) cell, drawn from that
cell's 100-candidate pool" (search_budget.py), and the preregistration's own decision-gate
section states the CONFIRMED criterion as "the registered search-budget divergence holds in
>=4/5 capabilities" -- i.e. ONE boolean per CAPABILITY (matching decision_gate.py's
CapabilityGateInputs.search_budget_divergence_confirmed: bool). Neither document specifies how
a capability's SIX per-cell divergence booleans collapse into that one per-capability boolean.

FROZEN RULE (user-authorized): a capability's search_budget_divergence_confirmed is True iff:
  1. At least 4 of its 6 (scope, radius) cells have divergence_confirmed=True
     (search_budget.check_registered_divergence's own output, unmodified), AND
  2. Those passing cells cover BOTH radii (0.02 and 0.04), AND
  3. Those passing cells cover AT LEAST TWO distinct scopes (of the frozen three:
     vision_encoder, full_lm, full_vlm).
This module ONLY implements that aggregation -- it never calls, wraps, or reimplements
search_budget.monte_carlo_search_budget_analysis or check_registered_divergence, and never
touches decision_gate.py's own evaluate_decision_gate. Pure post-hoc bookkeeping over already-
computed per-cell booleans.
"""
from __future__ import annotations

from typing import Dict, Tuple

from .design import RADII, SCOPES

_EXPECTED_CELLS = frozenset((scope, radius) for scope in SCOPES for radius in RADII)
_MIN_PASSING_CELLS = 4
_MIN_SCOPES_COVERED = 2


def capability_search_budget_divergence_confirmed(cell_divergence_by_scope_radius: Dict[Tuple[str, float], bool]) -> bool:
    """`cell_divergence_by_scope_radius`: exactly the 6 frozen (scope, radius) cells for ONE
    capability, each mapped to that cell's own `check_registered_divergence(...)
    ["divergence_confirmed"]` boolean (search_budget.py's own, unmodified output). Raises
    ValueError if the key set doesn't exactly match the frozen 6-cell design -- never silently
    aggregates over a partial or malformed cell set.
    """
    got_cells = set(cell_divergence_by_scope_radius.keys())
    if got_cells != _EXPECTED_CELLS:
        raise ValueError(
            f"Expected exactly the 6 frozen (scope, radius) cells {sorted(_EXPECTED_CELLS)}, "
            f"got {sorted(got_cells)}."
        )
    passing_cells = {cell for cell, confirmed in cell_divergence_by_scope_radius.items() if confirmed}
    if len(passing_cells) < _MIN_PASSING_CELLS:
        return False
    radii_covered = {radius for (_scope, radius) in passing_cells}
    scopes_covered = {scope for (scope, _radius) in passing_cells}
    return len(radii_covered) == len(RADII) and len(scopes_covered) >= _MIN_SCOPES_COVERED
