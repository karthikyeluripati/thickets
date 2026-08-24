import math

import pytest

from neural_thickets_repro.run_global_visual_thicket_pilot import UPSTREAM_SIGMA_GRID
from neural_thickets_repro.thicket.radius_mapping import (
    ANCHOR_LABEL,
    FROZEN_STAGE6_SIGMAS,
    build_sigma_relative_l2_mapping,
    expected_epsilon_l2_norm,
    relative_l2_from_sigma,
    select_common_calibration_radii,
)


def test_frozen_stage6_sigmas_equals_upstream_sigma_grid():
    """Reused, not duplicated -- if Stage 6's own frozen grid ever changed, this module would
    change with it rather than silently drifting out of sync.
    """
    assert FROZEN_STAGE6_SIGMAS == UPSTREAM_SIGMA_GRID


def test_expected_epsilon_l2_norm_formula():
    assert expected_epsilon_l2_norm(0.001, 1_000_000) == pytest.approx(0.001 * 1000.0)


def test_expected_epsilon_l2_norm_rejects_nonpositive_d():
    with pytest.raises(ValueError):
        expected_epsilon_l2_norm(0.001, 0)


def test_expected_epsilon_l2_norm_rejects_negative_sigma():
    with pytest.raises(ValueError):
        expected_epsilon_l2_norm(-0.1, 10)


def test_relative_l2_from_sigma_formula():
    # d=100, sigma=0.01, theta_l2=2.0 -> sqrt(100)=10 -> expected_norm=0.1 -> r_hat=0.05
    assert relative_l2_from_sigma(0.01, 100, 2.0) == pytest.approx(0.05)


def test_relative_l2_from_sigma_rejects_nonpositive_theta_norm():
    with pytest.raises(ValueError):
        relative_l2_from_sigma(0.01, 100, 0.0)


def test_build_sigma_relative_l2_mapping_one_row_per_sigma_labeled_as_anchor():
    rows = build_sigma_relative_l2_mapping(FROZEN_STAGE6_SIGMAS, d=10_000, theta_l2_norm=5.0, scope_label="test_scope")
    assert len(rows) == len(FROZEN_STAGE6_SIGMAS)
    for row, sigma in zip(rows, FROZEN_STAGE6_SIGMAS):
        assert row["sigma"] == sigma
        assert row["scope"] == "test_scope"
        assert row["kind"] == ANCHOR_LABEL
        assert row["r_hat"] == pytest.approx(sigma * math.sqrt(10_000) / 5.0)


def test_build_sigma_relative_l2_mapping_monotonic_in_sigma():
    rows = build_sigma_relative_l2_mapping(FROZEN_STAGE6_SIGMAS, d=10_000, theta_l2_norm=5.0, scope_label="s")
    r_hats = [row["r_hat"] for row in rows]
    assert r_hats == sorted(r_hats)  # FROZEN_STAGE6_SIGMAS is itself ascending


# --- select_common_calibration_radii: mechanical dedup, never accuracy-driven -----------------


def test_select_common_calibration_radii_dedups_near_duplicates():
    values = [0.01, 0.0100000001, 0.05, 0.1]
    kept = select_common_calibration_radii(values, round_sig_figs=6)
    assert kept == [0.01, 0.05, 0.1]


def test_select_common_calibration_radii_drops_nonpositive_and_nonfinite():
    values = [0.0, -0.01, float("nan"), float("inf"), 0.02]
    kept = select_common_calibration_radii(values)
    assert kept == [0.02]


def test_select_common_calibration_radii_preserves_input_order_not_sorted():
    values = [0.05, 0.01, 0.1]
    kept = select_common_calibration_radii(values)
    assert kept == [0.05, 0.01, 0.1]


def test_select_common_calibration_radii_keeps_first_occurrence_on_duplicate():
    values = [0.01, 0.05, 0.01]
    kept = select_common_calibration_radii(values)
    assert kept == [0.01, 0.05]


def test_select_common_calibration_radii_never_filters_by_any_score_input():
    """Mechanical signature check: the function accepts only radii values, nothing that could
    be a capability score/accuracy -- guards against a future edit silently adding a
    score-based filter parameter.
    """
    import inspect

    sig = inspect.signature(select_common_calibration_radii)
    param_names = set(sig.parameters)
    assert param_names <= {"r_hat_values", "round_sig_figs"}
