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
    """The atlas design (radii/regions/capabilities studied) must never be selected by a
    best-Delta value. Section 7 of this repair pass explicitly requests a PER-DIRECTION
    "best radius" DESCRIPTIVE statistic (never used to redesign the atlas) -- every occurrence
    of "best_radius" in this module must therefore be suffixed "_for_description_only" or
    "_description_only", never a bare selection variable/function.
    """
    import inspect
    import re

    source = inspect.getsource(saa)
    for forbidden in ("select_best", "optimal_radius", "best_capability"):
        assert forbidden not in source
    for match in re.finditer(r"best_radius\w*", source):
        token = match.group(0)
        assert token.endswith("_for_description_only") or token.endswith("_description_only") or token == "best_radius", (
            f"Unexpected best_radius identifier not marked description-only: {token!r}"
        )


def test_best_radius_field_is_never_consumed_by_atlas_design_functions():
    """Cross-module guard: run_stage8_coarse_anatomical_atlas.py's own plan/population/
    checkpoint-building functions (the actual atlas design) must never reference best_radius at
    all -- confirms the descriptive stat computed here never leaks into experiment design.
    """
    import inspect

    import neural_thickets_repro.run_stage8_coarse_anatomical_atlas as runner_module

    assert "best_radius" not in inspect.getsource(runner_module)


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


# =================================================================================================
# Section 1: integrity gate
# =================================================================================================


def _matching_checkpoint(records: List[ExperimentResultRecord]) -> Dict:
    return {
        "d_map_n": saa.STAGE8_D_MAP_N,
        "subset_hashes": {cap: f"sub_{cap}" for cap in STAGE8_CAPABILITIES},
        "region_mask_hashes": {r: f"mask_{r}" for r in STAGE8_REGIONS},
        "enable_prefix_caching": False,
        "multimodal_cache_policy": "full_encoder_reset_vllm011_verified_v2",
        "radius_realization_method": "fixed_direction_bf16_quantization_aware_v3",
        "run_complete": True,
        "expected_result_rows": len(records),
    }


def test_integrity_gate_passes_on_a_full_scale_correct_design():
    records = _build_synthetic_records(n_directions=saa.STAGE8_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    report = saa.run_integrity_gate(records, checkpoint)
    assert report["all_checks_pass"] is True
    assert report["expected_576_unique_perturbations"] is True
    assert report["exactly_64_perturbations_per_anatomy_x_radius"] is True
    assert report["direction_seed_reused_across_all_3_radii_within_anatomy"] is True
    saa.ensure_stage8_integrity(report)  # must not raise


def test_integrity_gate_detects_wrong_row_count():
    records = _build_synthetic_records(n_directions=8)  # not the frozen 64
    checkpoint = _matching_checkpoint(records)
    report = saa.run_integrity_gate(records, checkpoint)
    assert report["all_checks_pass"] is False
    assert report["expected_total_rows_3456"] is False
    with pytest.raises(saa.Stage8IntegrityError):
        saa.ensure_stage8_integrity(report)


def test_integrity_gate_detects_missing_capability_row():
    records = _build_synthetic_records(n_directions=saa.STAGE8_N_DIRECTIONS_PER_CELL)
    del records[0]  # drops exactly one capability row from one perturbation
    checkpoint = _matching_checkpoint(records)
    report = saa.run_integrity_gate(records, checkpoint)
    assert report["all_checks_pass"] is False
    assert report["exactly_6_rows_per_perturbation"] is False


def test_integrity_gate_detects_wrong_cache_policy():
    records = _build_synthetic_records(n_directions=saa.STAGE8_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    checkpoint["multimodal_cache_policy"] = "some_other_policy"
    report = saa.run_integrity_gate(records, checkpoint)
    assert report["cache_policy_correct"] is False
    assert report["all_checks_pass"] is False


def test_integrity_gate_detects_prefix_caching_true():
    records = _build_synthetic_records(n_directions=saa.STAGE8_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    checkpoint["enable_prefix_caching"] = True
    report = saa.run_integrity_gate(records, checkpoint)
    assert report["enable_prefix_caching_false"] is False
    assert report["all_checks_pass"] is False


def test_integrity_gate_no_pairing_of_directions_across_different_anatomies():
    """direction_family_id is always region-qualified (e.g. "vision:0", never bare "0") -- the
    gate itself checks this explicitly, and radius-trajectory grouping (tested separately below)
    keys off the full family id, never the bare seed/index alone.
    """
    records = _build_synthetic_records(n_directions=saa.STAGE8_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    report = saa.run_integrity_gate(records, checkpoint)
    assert report["direction_family_ids_are_region_qualified"] is True


# =================================================================================================
# Section 2: baseline table
# =================================================================================================


def test_baseline_table_reports_canonical_baseline_per_capability():
    records = _build_synthetic_records()
    baseline_scores = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in STAGE8_CAPABILITIES}}
    table = saa.compute_baseline_table(records, baseline_scores)
    for cap in STAGE8_CAPABILITIES:
        assert table[cap]["baseline_score"] == BASE_SCORES[cap]
        assert table[cap]["headroom_1_minus_baseline"] == pytest.approx(1.0 - BASE_SCORES[cap])
        assert table[cap]["canonical_baseline_independent_of_anatomy_radius_direction"] is True
        assert table[cap]["base_score_values_seen_across_all_576_perturbations"] == [BASE_SCORES[cap]]


def test_baseline_table_detects_a_non_canonical_baseline():
    records = _build_synthetic_records()
    # Mutate one row's base_score so the capability is no longer anatomy/radius/direction-independent.
    records[0] = saa.ExperimentResultRecord(**{**records[0].to_dict(), "base_score": 0.99, "perturbed_score": 0.99 + records[0].delta})
    baseline_scores = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in STAGE8_CAPABILITIES}}
    table = saa.compute_baseline_table(records, baseline_scores)
    mutated_cap = records[0].capability
    assert table[mutated_cap]["canonical_baseline_independent_of_anatomy_radius_direction"] is False


# =================================================================================================
# Section 4: BH correction + permutation determinism
# =================================================================================================


def test_benjamini_hochberg_matches_hand_computation():
    pvalues = [0.01, 0.02, 0.03, 0.5]
    q = saa.benjamini_hochberg(pvalues)
    # Standard BH: sorted p = [.01,.02,.03,.5], ranks 1..4, raw q = p*m/rank = [.04,.04,.04,.5],
    # then enforce monotone-nondecreasing from the top: [.04,.04,.04,.5].
    expected_sorted = [0.04, 0.04, 0.04, 0.5]
    sorted_q = sorted(q)
    for a, b in zip(sorted(expected_sorted), sorted_q):
        assert a == pytest.approx(b)


def test_benjamini_hochberg_empty_input():
    assert saa.benjamini_hochberg([]) == []


def test_benjamini_hochberg_all_ones_stays_at_one():
    q = saa.benjamini_hochberg([1.0, 1.0, 1.0])
    assert all(x == pytest.approx(1.0) for x in q)


def test_permutation_p_value_is_deterministic():
    records = _build_synthetic_records()
    c1 = saa.compute_anatomical_contrasts(records)
    c2 = saa.compute_anatomical_contrasts(records)
    p1 = c1["spatial_reasoning"][str(STAGE8_RADII[0])]["vision_vs_language"]["mean_delta_diff_permutation_p"]
    p2 = c2["spatial_reasoning"][str(STAGE8_RADII[0])]["vision_vs_language"]["mean_delta_diff_permutation_p"]
    assert p1 == p2


def test_anatomical_contrasts_bh_correction_adds_q_values_and_significance_flags():
    records = _build_synthetic_records()
    contrasts = saa.compute_anatomical_contrasts(records)
    contrasts = saa.apply_benjamini_hochberg_correction(contrasts)
    cell = contrasts["spatial_reasoning"][str(STAGE8_RADII[0])]["vision_vs_language"]
    for key in ("mean_delta_diff_bh_q", "density_ge_0.02_diff_bh_q", "positive_thicket_mass_diff_bh_q"):
        assert key in cell
        assert 0.0 <= cell[key] <= 1.0
    for key in ("mean_delta_diff_bh_significant_fdr_0.05", "density_ge_0.02_diff_bh_significant_fdr_0.05", "positive_thicket_mass_diff_bh_significant_fdr_0.05"):
        assert isinstance(cell[key], bool)


def test_bh_correction_is_applied_within_each_statistic_family_separately():
    """A cell with an extreme mean_delta p-value must not affect the q-values of the density/
    mass families -- confirms the three families are corrected independently.
    """
    records = _build_synthetic_records()
    contrasts = saa.compute_anatomical_contrasts(records)
    all_cells = [cell for cap_map in contrasts.values() for radius_map in cap_map.values() for cell in radius_map.values()]
    mean_p = [c["mean_delta_diff_permutation_p"] for c in all_cells]
    density_p = [c["density_ge_0.02_diff_permutation_p"] for c in all_cells]
    contrasts = saa.apply_benjamini_hochberg_correction(contrasts)
    all_cells_after = [cell for cap_map in contrasts.values() for radius_map in cap_map.values() for cell in radius_map.values()]
    mean_q_direct = saa.benjamini_hochberg(mean_p)
    density_q_direct = saa.benjamini_hochberg(density_p)
    for cell, mq, dq in zip(all_cells_after, mean_q_direct, density_q_direct):
        assert cell["mean_delta_diff_bh_q"] == pytest.approx(mq)
        assert cell["density_ge_0.02_diff_bh_q"] == pytest.approx(dq)


def test_effect_sizes_present_and_finite_or_none():
    records = _build_synthetic_records()
    contrasts = saa.compute_anatomical_contrasts(records)
    cell = contrasts["spatial_reasoning"][str(STAGE8_RADII[0])]["vision_vs_language"]
    for key in ("mean_delta_effect_size_cohens_d", "density_ge_0.02_effect_size_cohens_h", "positive_thicket_mass_effect_size_standardized"):
        assert key in cell


# =================================================================================================
# Section 6: solution-density curves -- monotonicity in margin m
# =================================================================================================


def test_solution_density_curve_is_non_increasing_in_margin():
    records = _build_synthetic_records()
    curves = saa.compute_solution_density_curves(records)
    for cap, region_map in curves.items():
        for region, radius_map in region_map.items():
            for row in radius_map.values():
                values = row["delta_ge_m"]
                assert all(values[i] >= values[i + 1] - 1e-12 for i in range(len(values) - 1)), f"non-monotonic curve for {cap}/{region}"


def test_solution_density_curve_includes_the_frozen_002_005_thresholds():
    assert 0.02 in saa.SOLUTION_DENSITY_MARGIN_GRID
    assert 0.05 in saa.SOLUTION_DENSITY_MARGIN_GRID


def test_solution_density_curve_at_m_equals_0_matches_density_ge_0():
    records = _build_synthetic_records()
    curves = saa.compute_solution_density_curves(records)
    primary = saa.compute_primary_measurements(records)
    m_index = list(saa.SOLUTION_DENSITY_MARGIN_GRID).index(0.0)
    cap, region, radius_key = "spatial_reasoning", "language", str(STAGE8_RADII[0])
    assert curves[cap][region][radius_key]["delta_ge_m"][m_index] == pytest.approx(primary[cap][region][radius_key]["density_ge_0.0"])


# =================================================================================================
# Section 7: extended radius trajectory metrics
# =================================================================================================


def test_radius_trajectory_extended_fields_present_and_consistent():
    records = _build_synthetic_records()
    t = saa.compute_radius_trajectories(records)
    assert t["r_small"] == STAGE8_RADII[0] and t["r_mid"] == STAGE8_RADII[1] and t["r_transition"] == STAGE8_RADII[2]
    n = t["n_positive_at_small"]
    if n > 0:
        assert 0.0 <= t["positive_at_small_remains_positive_at_mid_rate"] <= 1.0
        assert 0.0 <= t["positive_at_small_remains_positive_at_transition_rate"] <= 1.0
    assert t["n_directions_emerging_only_at_mid"] >= 0
    assert t["n_directions_emerging_only_at_transition"] >= 0
    assert t["monotonic_nonincreasing_fraction"] is not None
    assert t["monotonic_nondecreasing_fraction"] is not None
    assert t["non_monotonic_fraction"] is not None


def test_radius_trajectory_best_radius_is_the_argmax_for_a_hand_built_family():
    def rising_delta(region, radius, direction_index, capability):
        if region == "language" and capability == "spatial_reasoning" and direction_index == 0:
            return {STAGE8_RADII[0]: 0.01, STAGE8_RADII[1]: 0.05, STAGE8_RADII[2]: 0.02}[radius]
        return -0.01

    records = []
    for region in STAGE8_REGIONS:
        for radius in STAGE8_RADII:
            for direction_index in range(2):
                for capability in STAGE8_CAPABILITIES:
                    records.append(_rec(capability=capability, region=region, radius=radius, direction_index=direction_index, delta=rising_delta(region, radius, direction_index, capability)))

    t = saa.compute_radius_trajectories(records)
    family = t["trajectories_by_capability"]["spatial_reasoning"]["language:0"]
    assert family["best_radius_for_description_only"] == STAGE8_RADII[1]


# =================================================================================================
# Section 8: specialization F/G (harm-while-improve, directional transfer)
# =================================================================================================


def test_specialization_directional_transfer_schema():
    records = _build_synthetic_records()
    s = saa.compute_cross_capability_specialization(records)
    cell = s["language"][str(STAGE8_RADII[0])]
    transfer = cell["directional_transfer"]
    assert set(transfer.keys()) == set(STAGE8_CAPABILITIES)
    for source_cap, row in transfer.items():
        assert "n_source_positive" in row
        for target_cap in STAGE8_CAPABILITIES:
            if target_cap == source_cap:
                assert row[target_cap] is None
            elif row[target_cap] is not None:
                assert "mean_delta" in row[target_cap] and "p_delta_gt_0" in row[target_cap]


def test_specialization_tradeoff_fraction_between_0_and_1():
    records = _build_synthetic_records()
    s = saa.compute_cross_capability_specialization(records)
    cell = s["language"][str(STAGE8_RADII[0])]
    assert 0.0 <= cell["fraction_tradeoff_candidates"] <= 1.0
    assert cell["harm_margin_used"] == saa.HARM_MARGIN


def test_specialization_tradeoff_detects_a_hand_built_case():
    """One synthetic candidate (direction_index=0) improves capability[0] by +0.05 and harms
    capability[1] by -0.05 (>= HARM_MARGIN) -- must be counted as a tradeoff candidate. A
    second, neutral candidate (direction_index=1) is included only so build_delta_matrix's
    downstream Spearman/discordance computation isn't degenerate on a single-row matrix.
    """
    caps = STAGE8_CAPABILITIES
    records = []
    tradeoff_deltas = {caps[0]: 0.05, caps[1]: -0.05}
    for cap in caps:
        records.append(_rec(capability=cap, region="language", radius=STAGE8_RADII[0], direction_index=0, delta=tradeoff_deltas.get(cap, 0.0)))
        records.append(_rec(capability=cap, region="language", radius=STAGE8_RADII[0], direction_index=1, delta=0.0))
    s = saa.compute_cross_capability_specialization(records)
    cell = s["language"][str(STAGE8_RADII[0])]
    assert cell["n_tradeoff_candidates_improve_one_harm_another_ge_0.02"] == 1


# =================================================================================================
# Section 5: anatomy x capability interaction -- entropy, dominance, stability
# =================================================================================================


def test_interaction_entropy_and_dominance_schema():
    records = _build_synthetic_records()
    interaction = saa.compute_anatomy_capability_interaction(records)
    cap_direction = interaction["direction_A_capability_to_anatomy"]["spatial_reasoning"]["per_radius"][str(STAGE8_RADII[0])]
    assert 0.0 <= cap_direction["entropy_bits"] <= cap_direction["max_entropy_bits"] + 1e-9
    assert sum(cap_direction["normalized_mass_distribution"].values()) == pytest.approx(1.0)
    assert cap_direction["dominance_margin_over_second_best"] >= 0.0


def test_interaction_dominant_anatomy_matches_manual_argmax():
    def masses_for(region, radius, direction_index, capability):
        if capability != "spatial_reasoning":
            return 0.0
        return {"vision": 0.01, "multimodal_connector_or_merger": 0.02, "language": 0.09}[region]

    records = []
    for region in STAGE8_REGIONS:
        for radius in STAGE8_RADII:
            for direction_index in range(4):
                for capability in STAGE8_CAPABILITIES:
                    records.append(_rec(capability=capability, region=region, radius=radius, direction_index=direction_index, delta=masses_for(region, radius, direction_index, capability)))
    interaction = saa.compute_anatomy_capability_interaction(records)
    cell = interaction["direction_A_capability_to_anatomy"]["spatial_reasoning"]["per_radius"][str(STAGE8_RADII[0])]
    assert cell["dominant_anatomy"] == "language"


def test_interaction_stability_true_when_same_anatomy_dominates_at_2_plus_radii():
    records = _build_synthetic_records()
    interaction = saa.compute_anatomy_capability_interaction(records)
    cap_info = interaction["direction_A_capability_to_anatomy"]["spatial_reasoning"]
    assert isinstance(cap_info["dominance_stable_across_at_least_2_radii"], bool)


def test_interaction_never_claims_final_expert_location():
    """The note may legitimately use the word "final" inside a DISCLAIMER ("NOT a final
    expert-location claim") -- what must never appear is an AFFIRMATIVE claim of finality/
    confirmation.
    """
    note = saa.compute_anatomy_capability_interaction(_build_synthetic_records())["terminology_note"].lower()
    assert "confirmed" not in note
    assert "not a final" in note  # the disclaimer itself must be present
    for affirmative in ("is the final", "final location", "confirmed as", "definitively located"):
        assert affirmative not in note


# =================================================================================================
# Section 10: quantization confound audit
# =================================================================================================


def test_quantization_confound_audit_reports_none_when_no_quantization_limited_candidates():
    records = _build_synthetic_records()  # default acceptance_mode="strict" throughout
    audit = saa.compute_quantization_confound_audit(records)
    cell = audit["language"][str(STAGE8_RADII[0])]
    assert cell["n_quantization_limited_candidates"] == 0
    assert cell["acceptance_mode_associated_with_delta"] is None


def test_quantization_confound_audit_computes_a_real_contrast_when_both_groups_present():
    records = []
    for direction_index in range(8):
        mode = "quantization_limited" if direction_index < 4 else "strict"
        for capability in STAGE8_CAPABILITIES:
            records.append(_rec(capability=capability, region="language", radius=STAGE8_RADII[0], direction_index=direction_index, delta=0.01, acceptance_mode=mode))
    audit = saa.compute_quantization_confound_audit(records)
    cell = audit["language"][str(STAGE8_RADII[0])]
    assert cell["n_strict_candidates"] == 4
    assert cell["n_quantization_limited_candidates"] == 4
    assert cell["mean_delta_diff_strict_minus_quantization_limited"] == pytest.approx(0.0)
    assert cell["acceptance_mode_associated_with_delta"] is False  # identical deltas -- CI must bracket 0


# =================================================================================================
# Section 9: thicket phenotypes
# =================================================================================================


def test_thicket_phenotypes_schema_and_values():
    records = _build_synthetic_records()
    primary = saa.compute_primary_measurements(records)
    specialization = saa.compute_cross_capability_specialization(records)
    phenotypes = saa.compute_thicket_phenotypes(primary, specialization)
    cell = phenotypes["spatial_reasoning"]["language"][str(STAGE8_RADII[0])]
    assert cell["density_ge_0.02"] == primary["spatial_reasoning"]["language"][str(STAGE8_RADII[0])]["density_ge_0.02"]
    assert cell["max_delta"] == primary["spatial_reasoning"]["language"][str(STAGE8_RADII[0])]["max_delta"]
    assert cell["specialization_score_spectral_discordance"] == specialization["language"][str(STAGE8_RADII[0])]["spectral_discordance"]


# =================================================================================================
# Section 11/12/13: CUB check, bridge, Stage-9 recommendation
# =================================================================================================


def test_cub_stability_check_uses_the_same_statistics_no_special_thresholds():
    records = _build_synthetic_records()
    baseline_scores = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in STAGE8_CAPABILITIES}}
    check = saa.compute_cub_stability_check(records, baseline_scores)
    assert check["capability"] == "fine_grained_recognition"
    assert 0.0 <= check["fraction_exact_zero_delta"] <= 1.0
    assert "no special" in check["note"].lower() or "no special/looser" in check["note"].lower()


def test_bridge_never_claims_sigma_radius_numerical_equivalence_and_excludes_invalid_runs():
    records = _build_synthetic_records()
    primary = saa.compute_primary_measurements(records)
    bridge = saa.compute_stage6_stage7b_stage8_bridge(primary, stage7b_calibration_table=None)
    assert "NOT" in bridge["note"] and "equated" in bridge["note"]
    assert bridge["historical_invalid_vision_connector_runs_used"] is False


def test_stage9_recommendation_is_deterministic_and_never_picks_connector_by_default():
    records = _build_synthetic_records()
    primary = saa.compute_primary_measurements(records)
    interaction = saa.compute_anatomy_capability_interaction(records)
    rec1 = saa.compute_stage9_drilldown_recommendation(primary, interaction)
    rec2 = saa.compute_stage9_drilldown_recommendation(primary, interaction)
    assert rec1 == rec2
    assert rec1["connector_action"] in ("keep_whole", "consider_decomposition")
    assert rec1["priority_1_region"] != "multimodal_connector_or_merger"
    assert rec1["priority_2_region"] != "multimodal_connector_or_merger"
