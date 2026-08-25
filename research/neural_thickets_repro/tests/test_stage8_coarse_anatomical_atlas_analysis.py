"""Tests for analysis/stage8_coarse_anatomical_atlas_analysis.py -- built and verified against
small synthetic ExperimentResultRecord grids, since Stage 8's real 576-perturbation GPU run has
not been executed yet. Every function is a pure transform of already-collected records.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import stage8_coarse_anatomical_atlas_analysis as saa  # noqa: E402

from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_RADII, STAGE8_REGIONS  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

N_DIRECTIONS = 6
BASE_SCORES = {cap: 0.5 for cap in STAGE8_CAPABILITIES}


def _rec(*, capability: str, region: str, radius: float, direction_index: int, delta: float, acceptance_mode: str = "strict", relative_radius_error: float = 0.0) -> ExperimentResultRecord:
    base = BASE_SCORES[capability]
    pid = f"{region}_{radius}_{direction_index}"
    return ExperimentResultRecord(
        experiment_id="stage8_coarse_anatomical_atlas", perturbation_id=pid, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", perturbation_mode="anatomical_relative_l2", anatomy_region=region, radius=radius, sigma=None,
        seed=direction_index, parameter_mask_hash=f"mask_{region}", capability=capability, dataset_role="map",
        subset_hash=f"sub_{capability}", base_score=base, perturbed_score=round(base + delta, 10), delta=delta,
        parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash=f"h_{pid}_{capability}",
        runtime_metadata={
            "direction_family_id": f"{region}:{direction_index}", "direction_seed": direction_index,
            "direction_index": direction_index, "region": region,
            "radius_acceptance_mode": acceptance_mode, "quantization_limited": acceptance_mode == "quantization_limited",
            "requested_relative_l2": radius, "realized_relative_l2": radius * (1.0 + relative_radius_error),
            "relative_radius_error": relative_radius_error,
        },
    )


def _delta_fn(region: str, radius: float, direction_index: int, capability: str) -> float:
    """Deterministic synthetic delta: spatial_reasoning gets a clean useful-then-decaying
    trajectory across radii within 'language'; other (region, capability) combos get small
    fixed non-positive deltas -- gives every analysis function real, hand-verifiable structure.
    """
    radius_rank = STAGE8_RADII.index(radius)
    if region == "language" and capability == "spatial_reasoning":
        return [0.05, 0.02, -0.03][radius_rank] - 0.01 * (direction_index % 3)
    if region == "vision" and capability == "visual_grounding":
        return [0.04, 0.01, -0.02][radius_rank]
    return -0.01 * (radius_rank + 1) - 0.001 * direction_index


def _build_synthetic_records(n_directions: int = N_DIRECTIONS) -> List[ExperimentResultRecord]:
    records = []
    for region in STAGE8_REGIONS:
        for radius in STAGE8_RADII:
            for direction_index in range(n_directions):
                for capability in STAGE8_CAPABILITIES:
                    delta = _delta_fn(region, radius, direction_index, capability)
                    records.append(_rec(capability=capability, region=region, radius=radius, direction_index=direction_index, delta=delta))
    return records


# =================================================================================================
# Section 11: primary measurements
# =================================================================================================


def test_primary_measurements_matches_manual_computation():
    records = _build_synthetic_records()
    result = saa.compute_primary_measurements(records)
    row = result["spatial_reasoning"]["language"][str(STAGE8_RADII[0])]
    expected_deltas = [0.05 - 0.01 * (i % 3) for i in range(N_DIRECTIONS)]
    assert row["n"] == N_DIRECTIONS
    assert row["mean_delta"] == pytest.approx(sum(expected_deltas) / N_DIRECTIONS)
    assert row["density_ge_0.0"] == pytest.approx(sum(1 for d in expected_deltas if d >= 0) / N_DIRECTIONS)


def test_primary_measurements_covers_every_capability_region_radius_cell():
    records = _build_synthetic_records()
    result = saa.compute_primary_measurements(records)
    for cap in STAGE8_CAPABILITIES:
        for region in STAGE8_REGIONS:
            for radius in STAGE8_RADII:
                assert str(radius) in result[cap][region]


# =================================================================================================
# Section 12: anatomical contrasts
# =================================================================================================


def test_anatomical_contrasts_covers_all_three_pairs_per_capability_radius():
    records = _build_synthetic_records()
    result = saa.compute_anatomical_contrasts(records)
    cell = result["spatial_reasoning"][str(STAGE8_RADII[0])]
    assert set(cell.keys()) == {
        "vision_vs_multimodal_connector_or_merger", "vision_vs_language", "multimodal_connector_or_merger_vs_language",
    }


def test_anatomical_contrasts_mean_diff_matches_manual_computation():
    records = _build_synthetic_records()
    result = saa.compute_anatomical_contrasts(records)
    cell = result["spatial_reasoning"][str(STAGE8_RADII[0])]["vision_vs_language"]
    vision_deltas = [_delta_fn("vision", STAGE8_RADII[0], i, "spatial_reasoning") for i in range(N_DIRECTIONS)]
    language_deltas = [_delta_fn("language", STAGE8_RADII[0], i, "spatial_reasoning") for i in range(N_DIRECTIONS)]
    expected = sum(vision_deltas) / N_DIRECTIONS - sum(language_deltas) / N_DIRECTIONS
    assert cell["mean_delta_diff"] == pytest.approx(expected)


def test_anatomical_contrasts_bootstrap_ci_brackets_the_point_estimate():
    records = _build_synthetic_records()
    result = saa.compute_anatomical_contrasts(records)
    cell = result["spatial_reasoning"][str(STAGE8_RADII[0])]["vision_vs_language"]
    lo, hi = cell["mean_delta_diff_95ci_bootstrap"]
    assert lo <= cell["mean_delta_diff"] <= hi


# =================================================================================================
# Section 13: radius trajectories
# =================================================================================================


def test_radius_trajectories_sign_persistence_and_survival_for_the_decaying_spatial_family():
    records = _build_synthetic_records()
    result = saa.compute_radius_trajectories(records)
    # direction_index=0 in language/spatial_reasoning: deltas = [0.05, 0.02, -0.03] -- positive
    # at R_small/R_mid, negative at R_transition: survives to R_mid but not R_transition.
    family = result["trajectories_by_capability"]["spatial_reasoning"]["language:0"]
    deltas = family["delta_by_radius"]
    assert deltas[str(STAGE8_RADII[0])] == pytest.approx(0.05)
    assert deltas[str(STAGE8_RADII[1])] == pytest.approx(0.02)
    assert deltas[str(STAGE8_RADII[2])] == pytest.approx(-0.03)
    # every synthetic trajectory here is monotonically non-increasing across radii
    assert result["monotonic_nonincreasing_fraction"] == pytest.approx(1.0)


def test_radius_trajectories_disappearance_histogram_counts_the_transition_radius():
    records = _build_synthetic_records()
    result = saa.compute_radius_trajectories(records)
    hist = result["positive_direction_disappearance_radius_histogram"]
    # every positive-at-R_small direction in this synthetic set turns negative exactly at
    # R_transition (index 2), never earlier.
    assert hist[str(STAGE8_RADII[2])] > 0
    assert hist[str(STAGE8_RADII[0])] == 0


def test_radius_trajectories_excludes_incomplete_families_never_fabricates():
    records = _build_synthetic_records()
    # Remove one row so one direction-family is incomplete at one radius.
    records = [r for r in records if not (r.anatomy_region == "language" and r.radius == STAGE8_RADII[1] and r.capability == "spatial_reasoning" and r.seed == 0)]
    result = saa.compute_radius_trajectories(records)
    assert "language:0" not in result["trajectories_by_capability"].get("spatial_reasoning", {})


def test_radius_trajectories_rank_stability_present_between_consecutive_radii():
    records = _build_synthetic_records()
    result = saa.compute_radius_trajectories(records)
    rank_stability = result["rank_stability_spearman_between_consecutive_radii"]
    assert "spatial_reasoning" in rank_stability
    assert "language" in rank_stability["spatial_reasoning"]
    keys = rank_stability["spatial_reasoning"]["language"].keys()
    assert f"{STAGE8_RADII[0]}_to_{STAGE8_RADII[1]}" in keys


# =================================================================================================
# Section 14: cross-capability specialization
# =================================================================================================


def test_cross_capability_specialization_is_6x6_per_region_radius_cell():
    records = _build_synthetic_records()
    result = saa.compute_cross_capability_specialization(records)
    cell = result["language"][str(STAGE8_RADII[0])]
    assert len(cell["capabilities"]) == 6
    matrix = cell["spearman_6x6"]
    assert len(matrix) == 6 and all(len(row) == 6 for row in matrix)
    assert cell["n_perturbations"] == N_DIRECTIONS


def test_cross_capability_specialization_improving_histogram_sums_to_n_perturbations():
    records = _build_synthetic_records()
    result = saa.compute_cross_capability_specialization(records)
    cell = result["language"][str(STAGE8_RADII[0])]
    assert sum(cell["improving_count_histogram"].values()) == N_DIRECTIONS


# =================================================================================================
# Section 15: anatomical selectivity atlas -- never collapsed across radius
# =================================================================================================


def test_anatomical_selectivity_atlas_keeps_every_radius_separate():
    records = _build_synthetic_records()
    result = saa.compute_anatomical_selectivity_atlas(records)
    assert set(result.keys()) == {str(r) for r in STAGE8_RADII}
    for radius_key in result:
        assert set(result[radius_key].keys()) == set(STAGE8_CAPABILITIES)
        for cap in STAGE8_CAPABILITIES:
            assert set(result[radius_key][cap].keys()) == set(STAGE8_REGIONS)


def test_anatomical_selectivity_atlas_values_match_primary_measurements():
    records = _build_synthetic_records()
    primary = saa.compute_primary_measurements(records)
    atlas = saa.compute_anatomical_selectivity_atlas(records)
    radius_key = str(STAGE8_RADII[0])
    assert atlas[radius_key]["spatial_reasoning"]["language"]["mean_delta"] == pytest.approx(
        primary["spatial_reasoning"]["language"][radius_key]["mean_delta"]
    )


# =================================================================================================
# Section 16: quantization audit
# =================================================================================================


def test_quantization_audit_counts_strict_vs_quantization_limited():
    records = _build_synthetic_records(n_directions=4)
    # Mark half of vision/R_small's candidates as quantization_limited -- ALL 6 capability rows
    # for the mutated candidates, matching the real invariant that radius_acceptance_mode is
    # computed ONCE per candidate (in apply_result) and written identically into every one of
    # that candidate's capability rows (see evaluate_one_stage8_candidate_rpc).
    for r in records:
        if r.anatomy_region == "vision" and r.radius == STAGE8_RADII[0] and r.seed in (0, 1):
            r.runtime_metadata["radius_acceptance_mode"] = "quantization_limited"
            r.runtime_metadata["quantization_limited"] = True
            r.runtime_metadata["relative_radius_error"] = 5e-4
    result = saa.compute_quantization_audit(records)
    cell = result["vision"][str(STAGE8_RADII[0])]
    assert cell["n_candidates"] == 4
    assert cell["strict_count"] + cell["quantization_limited_count"] == 4
    assert cell["quantization_limited_count"] == 2


def test_quantization_audit_deduplicates_by_perturbation_id_across_capabilities():
    """Every one of the 6 capability rows for the SAME candidate shares one accepted
    perturbation -- n_candidates must count unique perturbation_ids, never 6x too many.
    """
    records = _build_synthetic_records(n_directions=3)
    result = saa.compute_quantization_audit(records)
    cell = result["language"][str(STAGE8_RADII[0])]
    assert cell["n_candidates"] == 3  # not 3 * 6


def test_quantization_audit_reports_realized_over_requested_ratio():
    records = _build_synthetic_records(n_directions=2)
    result = saa.compute_quantization_audit(records)
    cell = result["language"][str(STAGE8_RADII[0])]
    assert cell["mean_realized_over_requested_ratio"] == pytest.approx(1.0)


# =================================================================================================
# No best-radius / no capability-optimization selection logic
# =================================================================================================


def test_no_best_selection_logic_exists():
    import inspect

    source = inspect.getsource(saa)
    for forbidden in ("best_radius", "select_best", "optimal_radius", "best_capability"):
        assert forbidden not in source


def test_full_pipeline_is_deterministic():
    records = _build_synthetic_records()

    def run_once():
        return saa._sanitize({
            "primary": saa.compute_primary_measurements(records),
            "contrasts": saa.compute_anatomical_contrasts(records),
            "trajectories": saa.compute_radius_trajectories(records),
            "specialization": saa.compute_cross_capability_specialization(records),
            "atlas": saa.compute_anatomical_selectivity_atlas(records),
            "quantization": saa.compute_quantization_audit(records),
        })

    import json
    first = json.dumps(run_once(), sort_keys=True)
    second = json.dumps(run_once(), sort_keys=True)
    assert first == second
