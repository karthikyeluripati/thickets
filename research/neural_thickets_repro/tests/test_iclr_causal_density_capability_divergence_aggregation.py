"""Tests for the post-hoc, user-authorized capability-level search-budget divergence
aggregation rule (frozen 2026-09-03, before the full 6-cell analysis existed). CPU-only, pure
dict-of-bools logic.
"""
from __future__ import annotations

import pytest

from neural_thickets_repro.iclr_causal_density.capability_divergence_aggregation import capability_search_budget_divergence_confirmed
from neural_thickets_repro.iclr_causal_density.design import RADII, SCOPES

_ALL_CELLS = [(scope, radius) for scope in SCOPES for radius in RADII]


def _cells(passing):
    """passing: iterable of (scope, radius) pairs that should be True; every other frozen cell is False."""
    passing = set(passing)
    return {cell: (cell in passing) for cell in _ALL_CELLS}


def test_raises_on_missing_cell():
    incomplete = _cells([])
    del incomplete[("vision_encoder", 0.02)]
    with pytest.raises(ValueError, match="Expected exactly the 6 frozen"):
        capability_search_budget_divergence_confirmed(incomplete)


def test_raises_on_extra_unknown_cell():
    cells = _cells([])
    cells[("not_a_real_scope", 0.02)] = True
    with pytest.raises(ValueError, match="Expected exactly the 6 frozen"):
        capability_search_budget_divergence_confirmed(cells)


def test_all_six_cells_passing_confirms():
    assert capability_search_budget_divergence_confirmed(_cells(_ALL_CELLS)) is True


def test_zero_cells_passing_does_not_confirm():
    assert capability_search_budget_divergence_confirmed(_cells([])) is False


def test_fewer_than_four_passing_does_not_confirm():
    # 3 passing cells, all conditions on scope/radius coverage satisfied -- still fails the
    # >=4/6 count requirement.
    passing = [("vision_encoder", 0.02), ("vision_encoder", 0.04), ("full_lm", 0.02)]
    assert capability_search_budget_divergence_confirmed(_cells(passing)) is False


def test_three_cells_at_a_single_radius_fails_the_count_clause():
    # This frozen design is 3 scopes x 2 radii -- a single radius can supply at most 3 passing
    # cells (one per scope), so "4 passing cells confined to one radius" is structurally
    # impossible here; the count clause (>=4) fails first, which this test documents directly.
    passing = [("vision_encoder", 0.02), ("full_lm", 0.02), ("full_vlm", 0.02)]
    assert len(passing) == 3
    assert capability_search_budget_divergence_confirmed(_cells(passing)) is False


def test_four_passing_spanning_both_radii_and_two_scopes_confirms():
    passing = [("vision_encoder", 0.02), ("vision_encoder", 0.04), ("full_lm", 0.02), ("full_lm", 0.04)]
    assert capability_search_budget_divergence_confirmed(_cells(passing)) is True


def test_four_passing_both_radii_across_two_scopes_unevenly_still_confirms():
    passing = [("vision_encoder", 0.02), ("vision_encoder", 0.04), ("full_vlm", 0.02), ("full_lm", 0.04)]
    assert capability_search_budget_divergence_confirmed(_cells(passing)) is True


def test_five_of_six_passing_confirms():
    passing = _ALL_CELLS[:-1]
    assert capability_search_budget_divergence_confirmed(_cells(passing)) is True


def test_matches_the_worked_example_from_the_freeze_decision():
    """The exact scenario under discussion when this rule was frozen: only vision_encoder's 2
    cells were known at the time. With only those 2 passing (regardless of the other 4 cells'
    eventual outcome being unknown-but-treated-as-False for this synthetic check), the capability
    must NOT be confirmed yet -- consistent with "we don't know the answer until more cells
    report in," not a premature confirmation.
    """
    passing = [("vision_encoder", 0.02), ("vision_encoder", 0.04)]
    assert capability_search_budget_divergence_confirmed(_cells(passing)) is False
