"""Tests for analysis/stage9_hierarchical_localization_analysis.py -- built and verified against
small synthetic ExperimentResultRecord grids (same discipline as
test_stage8_coarse_anatomical_atlas_analysis.py), AND cross-checked against the REAL completed
Stage-9 run (results/stage9_hierarchical_anatomical_atlas/stage9_hierarchical_anatomical_atlas_3b_v1/)
when that directory is present locally.

Most of the underlying statistical machinery (bootstrap CIs, permutation tests, BH correction,
Spearman/spectral-discordance specialization matrices, radius-trajectory pairing, entropy) is
REUSED BY IMPORT from stage8_coarse_anatomical_atlas_analysis.py and already has its own
dedicated test coverage there -- this file focuses on what's NEW for Stage 9: the depth-region
integrity gate, parent/child mapping, the language-depth-answer classifier, parent->child
enrichment (including safe zero-parent handling), and the numerical-patch audit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import stage9_hierarchical_localization_analysis as s9a  # noqa: E402

from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import (  # noqa: E402
    STAGE8_AUTHORITATIVE_SUBSET_HASHES, STAGE9_CAPABILITIES, STAGE9_D_MAP_N, STAGE9_N_DIRECTIONS_PER_CELL, STAGE9_RADII,
)
from neural_thickets_repro.run_stage7b_anatomical_calibration import MULTIMODAL_CACHE_POLICY, RADIUS_REALIZATION_METHOD  # noqa: E402
from neural_thickets_repro.thicket.anatomy_stage9 import STAGE9_CHILD_REGIONS  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

N_DIRECTIONS = 4
BASE_SCORES = {cap: 0.5 for cap in STAGE9_CAPABILITIES}
REAL_RESULTS_DIR = s9a.DEFAULT_RESULTS_DIR


def _rec(*, capability: str, region: str, radius: float, direction_index: int, delta: float, acceptance_mode: str = "strict", relative_radius_error: float = 0.0) -> ExperimentResultRecord:
    base = BASE_SCORES[capability]
    pid = f"{region}_{radius}_{direction_index}"
    return ExperimentResultRecord(
        experiment_id="stage9_hierarchical_anatomical_atlas", perturbation_id=pid, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", perturbation_mode="anatomical_relative_l2", anatomy_region=region, radius=radius, sigma=None,
        seed=direction_index, parameter_mask_hash=f"mask_{region}", capability=capability, dataset_role="map",
        subset_hash=f"sub_{capability}", base_score=base, perturbed_score=round(base + delta, 10), delta=delta,
        parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash=f"h_{pid}_{capability}",
        runtime_metadata={
            "direction_family_id": f"{region}:{direction_index}", "direction_seed": direction_index,
            "direction_index": direction_index, "region": region,
            "radius_acceptance_mode": acceptance_mode, "quantization_limited": acceptance_mode == "quantization_limited",
            "requested_relative_l2": radius, "realized_relative_l2": radius * (1.0 + relative_radius_error),
            "realized_abs_error": abs(radius * relative_radius_error), "relative_radius_error": relative_radius_error,
        },
    )


def _delta_fn(region: str, radius: float, direction_index: int, capability: str) -> float:
    """Deterministic synthetic delta: spatial_reasoning gets a clean, STABLE language_late
    preference across all 3 radii (unlike Stage 8's own fixture, which deliberately decays --
    Stage 9's hero question needs a fixture where a single depth band clearly wins every radius
    to exercise the "stable leader" classification path); everything else gets small fixed
    non-positive deltas.
    """
    if region == "language_late" and capability == "spatial_reasoning":
        return 0.05 - 0.001 * direction_index
    if region in ("language_early", "language_mid") and capability == "spatial_reasoning":
        return -0.01 - 0.001 * direction_index
    if region == "vision_early" and capability == "visual_grounding":
        return 0.05 - 0.001 * direction_index
    return -0.01 - 0.001 * direction_index


def _build_synthetic_records(n_directions: int = N_DIRECTIONS) -> List[ExperimentResultRecord]:
    records = []
    for region in STAGE9_CHILD_REGIONS:
        for radius in STAGE9_RADII:
            for direction_index in range(n_directions):
                for capability in STAGE9_CAPABILITIES:
                    delta = _delta_fn(region, radius, direction_index, capability)
                    records.append(_rec(capability=capability, region=region, radius=radius, direction_index=direction_index, delta=delta))
    return records


def _matching_checkpoint(records: List[ExperimentResultRecord]) -> Dict:
    return {
        "d_map_n": STAGE9_D_MAP_N,
        "subset_hashes": dict(STAGE8_AUTHORITATIVE_SUBSET_HASHES),
        "child_mask_hashes": {r: f"mask_{r}" for r in STAGE9_CHILD_REGIONS},
        "enable_prefix_caching": False,
        "multimodal_cache_policy": MULTIMODAL_CACHE_POLICY,
        "generation_batch_size": 10,
        "radius_realization_method": RADIUS_REALIZATION_METHOD,
        "partition_audit_hash": "hash123",
    }


def _matching_run_manifest(records: List[ExperimentResultRecord]) -> Dict:
    by_pid = set(r.perturbation_id for r in records)
    return {
        "run_complete": True, "partition_audit_hash": "hash123",
        "actual_unique_perturbations": len(by_pid), "expected_unique_perturbations": len(by_pid),
        "actual_result_rows": len(records), "expected_result_rows": len(records),
    }


# =================================================================================================
# Section 1: integrity gate -- exact 1152/6912, 64 directions/cell, 6 capabilities/candidate
# =================================================================================================


def test_integrity_gate_passes_on_a_full_scale_correct_design():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    assert len(records) == 6912
    checkpoint = _matching_checkpoint(records)
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["all_checks_pass"] is True
    assert report["expected_1152_unique_perturbations"] is True
    assert report["exactly_64_perturbations_per_child_x_radius"] is True
    assert report["exactly_6_rows_per_perturbation"] is True
    assert report["six_child_regions"] is True
    s9a.ensure_stage9_analysis_integrity(report)  # must not raise


def test_integrity_gate_detects_wrong_row_count():
    records = _build_synthetic_records(n_directions=8)  # not the frozen 64
    checkpoint = _matching_checkpoint(records)
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["all_checks_pass"] is False
    assert report["expected_total_rows_6912"] is False
    with pytest.raises(s9a.Stage9AnalysisIntegrityError):
        s9a.ensure_stage9_analysis_integrity(report)


def test_integrity_gate_detects_missing_capability_row():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    del records[0]
    checkpoint = _matching_checkpoint(records)
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["exactly_6_rows_per_perturbation"] is False
    assert report["all_checks_pass"] is False


def test_integrity_gate_detects_subset_hash_mismatch_against_stage8_authoritative():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    checkpoint["subset_hashes"] = {**checkpoint["subset_hashes"], "counting": "wrong_hash"}
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["subset_hashes_match_stage8_authoritative"] is False
    assert report["all_checks_pass"] is False


def test_integrity_gate_detects_row_level_mask_hash_inconsistency():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    records[0] = ExperimentResultRecord(**{**records[0].__dict__, "parameter_mask_hash": "corrupted"})
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["row_level_parameter_mask_hash_matches_checkpoint_child_mask_hashes"] is False
    assert report["all_checks_pass"] is False


def test_integrity_gate_detects_partition_audit_hash_self_inconsistency():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    run_manifest = _matching_run_manifest(records)
    run_manifest["partition_audit_hash"] = "different_hash"
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["partition_audit_hash_self_consistent_with_run_manifest"] is False
    assert report["all_checks_pass"] is False


def test_integrity_gate_detects_run_incomplete():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    run_manifest = _matching_run_manifest(records)
    run_manifest["run_complete"] = False
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["run_complete"] is False
    assert report["all_checks_pass"] is False


def test_integrity_gate_detects_wrong_cache_policy():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    checkpoint["multimodal_cache_policy"] = "some_other_policy"
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["cache_policy_correct"] is False


def test_integrity_gate_detects_prefix_caching_true():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    checkpoint["enable_prefix_caching"] = True
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["enable_prefix_caching_false"] is False


def test_integrity_gate_detects_wrong_generation_batch_size():
    records = _build_synthetic_records(n_directions=STAGE9_N_DIRECTIONS_PER_CELL)
    checkpoint = _matching_checkpoint(records)
    checkpoint["generation_batch_size"] = 5
    run_manifest = _matching_run_manifest(records)
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["generation_batch_size_10"] is False


# =================================================================================================
# Child <-> parent membership
# =================================================================================================


def test_child_to_parent_membership_covers_all_six_child_regions():
    assert set(s9a.CHILD_TO_PARENT.keys()) == set(STAGE9_CHILD_REGIONS)
    assert set(s9a.CHILD_TO_PARENT.values()) == {"vision", "language"}


def test_vision_and_language_child_region_groups_partition_all_six():
    assert set(s9a.VISION_CHILD_REGIONS) | set(s9a.LANGUAGE_CHILD_REGIONS) == set(STAGE9_CHILD_REGIONS)
    assert set(s9a.VISION_CHILD_REGIONS) & set(s9a.LANGUAGE_CHILD_REGIONS) == set()
    for region in s9a.VISION_CHILD_REGIONS:
        assert s9a.CHILD_TO_PARENT[region] == "vision"
    for region in s9a.LANGUAGE_CHILD_REGIONS:
        assert s9a.CHILD_TO_PARENT[region] == "language"


def test_depth_pairs_are_within_parent_only():
    for a, b in s9a.VISION_DEPTH_PAIRS:
        assert a in s9a.VISION_CHILD_REGIONS and b in s9a.VISION_CHILD_REGIONS
    for a, b in s9a.LANGUAGE_DEPTH_PAIRS:
        assert a in s9a.LANGUAGE_CHILD_REGIONS and b in s9a.LANGUAGE_CHILD_REGIONS


# =================================================================================================
# Depth contrasts (reused compute_anatomical_contrasts with contrast_pairs override) -- bootstrap
# + BH correction applied separately per parent
# =================================================================================================


def test_depth_contrasts_use_only_within_parent_pairs():
    records = _build_synthetic_records()
    depth_contrasts = s9a.compute_depth_contrasts(records)
    for radius_map in depth_contrasts["vision_depth_contrasts"]["visual_grounding"].values():
        assert set(radius_map.keys()) == {f"{a}_vs_{b}" for a, b in s9a.VISION_DEPTH_PAIRS}
    for radius_map in depth_contrasts["language_depth_contrasts"]["spatial_reasoning"].values():
        assert set(radius_map.keys()) == {f"{a}_vs_{b}" for a, b in s9a.LANGUAGE_DEPTH_PAIRS}


def test_depth_contrasts_language_late_beats_early_and_mid_for_spatial_reasoning():
    """The synthetic fixture gives language_late a clean, stable +0.05-ish advantage for
    spatial_reasoning at every radius -- the mean_delta_diff sign must reflect that directly.
    """
    records = _build_synthetic_records()
    depth_contrasts = s9a.compute_depth_contrasts(records)
    for radius_key, pair_map in depth_contrasts["language_depth_contrasts"]["spatial_reasoning"].items():
        cell = pair_map["language_early_vs_language_late"]
        assert cell["mean_delta_diff"] < 0  # early - late is negative since late is much higher


def test_depth_contrasts_bh_correction_adds_q_values_separately_per_parent():
    records = _build_synthetic_records()
    depth_contrasts = s9a.compute_depth_contrasts(records)
    vision_cell = next(iter(next(iter(depth_contrasts["vision_depth_contrasts"]["visual_grounding"].values())).values()))
    language_cell = next(iter(next(iter(depth_contrasts["language_depth_contrasts"]["spatial_reasoning"].values())).values()))
    for cell in (vision_cell, language_cell):
        assert "mean_delta_diff_bh_q" in cell
        assert "density_ge_0.02_diff_bh_q" in cell
        assert "positive_thicket_mass_diff_bh_q" in cell


def test_summarize_depth_contrast_significance_counts_separately_for_vision_and_language():
    records = _build_synthetic_records()
    depth_contrasts = s9a.compute_depth_contrasts(records)
    summary = s9a.summarize_depth_contrast_significance(depth_contrasts)
    assert set(summary.keys()) == {"vision_depth", "language_depth"}
    assert summary["language_depth"]["n_total_contrasts"] == 6 * 3 * 3  # 6 caps x 3 radii x 3 pairs
    assert summary["vision_depth"]["n_total_contrasts"] == 6 * 3 * 3


# =================================================================================================
# Depth selectivity (within-parent-only normalization via compute_anatomy_capability_interaction)
# =================================================================================================


def test_depth_selectivity_normalizes_within_parent_not_across_all_six():
    records = _build_synthetic_records()
    selectivity = s9a.compute_depth_selectivity(records)
    assert set(selectivity["vision_depth_selectivity"]["regions"]) == set(s9a.VISION_CHILD_REGIONS)
    assert set(selectivity["language_depth_selectivity"]["regions"]) == set(s9a.LANGUAGE_CHILD_REGIONS)


def test_depth_selectivity_spatial_reasoning_dominant_is_language_late():
    records = _build_synthetic_records()
    selectivity = s9a.compute_depth_selectivity(records)
    info = selectivity["language_depth_selectivity"]["direction_A_capability_to_anatomy"]["spatial_reasoning"]
    assert info["dominance_stable_across_at_least_2_radii"] is True
    assert info["stable_dominant_anatomy"] == "language_late"


def test_depth_selectivity_terminology_uses_mapping_scale_language_not_confirmed():
    records = _build_synthetic_records()
    selectivity = s9a.compute_depth_selectivity(records)
    for key in ("vision_depth_selectivity", "language_depth_selectivity"):
        note = selectivity[key]["terminology_note"]
        assert "mapping-scale depth preference" in note
        assert "hierarchical concentration" in note
        assert "confirmed location" not in note.replace("NOT a confirmed location", "")


# =================================================================================================
# Hero question: language-depth answer classifier
# =================================================================================================


def test_language_depth_answer_classifies_stable_late_dominance_as_C():
    records = _build_synthetic_records()
    depth_atlas = s9a.compute_depth_atlas(records)
    depth_contrasts = s9a.compute_depth_contrasts(records)
    answer = s9a.classify_language_depth_answer(depth_atlas, depth_contrasts["language_depth_contrasts"])
    assert answer["answer"] == "C"
    assert answer["answer_meaning"] == "late concentrated"
    assert answer["stable_leader_across_radii"] == "language_late"


def test_language_depth_answer_never_forces_an_answer_when_mixed():
    """A degenerate fixture where every depth band has an identical delta at every radius --
    no leader anywhere -- must report "D" (distributed), never a fabricated single-letter pick.
    """
    records = []
    for region in s9a.LANGUAGE_CHILD_REGIONS:
        for radius in STAGE9_RADII:
            for direction_index in range(N_DIRECTIONS):
                records.append(_rec(capability="spatial_reasoning", region=region, radius=radius, direction_index=direction_index, delta=0.01))
    depth_atlas = s9a.compute_depth_atlas(records)
    depth_contrasts = s9a.compute_depth_contrasts(
        [r for r in records] + _build_synthetic_records()  # pad with the full fixture so vision/other regions still resolve for a well-formed contrasts dict
    )
    answer = s9a.classify_language_depth_answer(depth_atlas, depth_contrasts["language_depth_contrasts"])
    assert answer["answer"] == "D"
    assert answer["stable_leader_across_radii"] is None


# =================================================================================================
# Parent -> child enrichment
# =================================================================================================


def _synthetic_stage8_atlas(records: List[ExperimentResultRecord]) -> Dict:
    """A minimal Stage-8-shaped atlas dict (same schema stage8's compute_primary_measurements
    produces) with a KNOWN, hand-picked parent density/mass per capability x radius, including
    one deliberate zero-density cell to exercise the safe-zero-parent path.
    """
    radii = STAGE9_RADII
    out: Dict = {}
    for cap in STAGE9_CAPABILITIES:
        out[cap] = {
            "vision": {str(r): {"density_ge_0.02": 0.1, "positive_thicket_mass": 0.01} for r in radii},
            "language": {str(r): {"density_ge_0.02": 0.0, "positive_thicket_mass": 0.0} for r in radii},
        }
    return out


def test_parent_child_enrichment_handles_zero_parent_safely():
    records = _build_synthetic_records()
    depth_atlas = s9a.compute_depth_atlas(records)
    stage8_atlas = _synthetic_stage8_atlas(records)
    enrichment = s9a.compute_parent_child_enrichment(depth_atlas, stage8_atlas)
    for radius in STAGE9_RADII:
        cell = enrichment["cells"]["visual_grounding"]["language_early"][str(radius)]
        assert cell["parent_density_ge_0.02"] == 0.0
        assert cell["density_ratio_child_over_parent"] is None  # never a fabricated ratio against a zero denominator
        assert cell["density_diff_child_minus_parent"] == cell["child_density_ge_0.02"]  # diff is still always reported


def test_parent_child_enrichment_summary_counts_are_exhaustive_over_all_cells():
    records = _build_synthetic_records()
    depth_atlas = s9a.compute_depth_atlas(records)
    stage8_atlas = _synthetic_stage8_atlas(records)
    enrichment = s9a.compute_parent_child_enrichment(depth_atlas, stage8_atlas)
    n_cells = sum(len(radius_map) for region_map in enrichment["cells"].values() for radius_map in region_map.values())
    s = enrichment["summary"]
    density_total = s["n_cells_density_enriched_ratio_gt_1"] + s["n_cells_density_diluted_ratio_lt_1"] + s["n_cells_density_ratio_unchanged_exactly_1"] + s["n_cells_density_ratio_unavailable_zero_parent"]
    mass_total = s["n_cells_mass_enriched_ratio_gt_1"] + s["n_cells_mass_diluted_ratio_lt_1"] + s["n_cells_mass_ratio_unchanged_exactly_1"] + s["n_cells_mass_ratio_unavailable_zero_parent"]
    assert density_total == n_cells
    assert mass_total == n_cells
    assert n_cells == len(STAGE9_CAPABILITIES) * len(STAGE9_CHILD_REGIONS) * len(STAGE9_RADII)


def test_parent_child_enrichment_computes_correct_ratio_for_nonzero_parent():
    records = _build_synthetic_records()
    depth_atlas = s9a.compute_depth_atlas(records)
    stage8_atlas = _synthetic_stage8_atlas(records)
    enrichment = s9a.compute_parent_child_enrichment(depth_atlas, stage8_atlas)
    cell = enrichment["cells"]["visual_grounding"]["vision_early"][str(STAGE9_RADII[0])]
    expected_ratio = cell["child_density_ge_0.02"] / 0.1
    assert cell["density_ratio_child_over_parent"] == pytest.approx(expected_ratio)


# =================================================================================================
# Numerical patch audit
# =================================================================================================


def test_numerical_patch_audit_counts_strict_vs_quantization_limited():
    records = [
        _rec(capability=cap, region="language_late", radius=STAGE9_RADII[0], direction_index=0, delta=0.0, acceptance_mode="strict")
        for cap in STAGE9_CAPABILITIES
    ] + [
        _rec(capability=cap, region="language_late", radius=STAGE9_RADII[0], direction_index=1, delta=0.0, acceptance_mode="quantization_limited", relative_radius_error=1e-4)
        for cap in STAGE9_CAPABILITIES
    ]
    audit = s9a.compute_numerical_patch_audit(records)
    assert audit["strict_count"] == 1
    assert audit["quantization_limited_count"] == 1
    assert audit["zero_admissibility_violations"] is True


def test_numerical_patch_audit_detects_admissibility_violation():
    records = [
        _rec(capability=cap, region="vision_early", radius=STAGE9_RADII[0], direction_index=0, delta=0.0, acceptance_mode="quantization_limited", relative_radius_error=0.01)
        for cap in STAGE9_CAPABILITIES  # 0.01 > QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE (1e-3) -- a genuine violation
    ]
    audit = s9a.compute_numerical_patch_audit(records)
    assert audit["zero_admissibility_violations"] is False
    assert audit["n_admissibility_violations"] == 1  # one unique perturbation, not one per capability row


def test_numerical_patch_audit_finds_the_known_reported_candidate():
    records = [
        _rec(
            capability=cap, region=s9a.KNOWN_BRACKET_EXPANSION_CANDIDATE["anatomy_region"], radius=STAGE9_RADII[2],
            direction_index=41, delta=0.0, acceptance_mode="strict",
        )
        for cap in STAGE9_CAPABILITIES
    ]
    for r in records:
        r.runtime_metadata["direction_seed"] = s9a.KNOWN_BRACKET_EXPANSION_CANDIDATE["seed"]
    audit = s9a.compute_numerical_patch_audit(records)
    assert audit["known_reported_bracket_expansion_candidate"]["found_in_results"] is True
    assert len(audit["known_reported_bracket_expansion_candidate"]["rows"]) == len(STAGE9_CAPABILITIES)


def test_numerical_patch_audit_reports_limitation_honestly():
    records = _build_synthetic_records()
    audit = s9a.compute_numerical_patch_audit(records)
    assert "bracket_expansion_used" in audit["limitation_note"]
    assert "not recoverable" in audit["limitation_note"]


# =================================================================================================
# Deterministic outputs
# =================================================================================================


def test_full_pipeline_is_deterministic():
    records = _build_synthetic_records()

    def run_once():
        depth_atlas = s9a.compute_depth_atlas(records)
        depth_contrasts = s9a.compute_depth_contrasts(records)
        return s9a._sanitize({
            "depth_atlas": depth_atlas,
            "depth_contrasts": depth_contrasts,
            "selectivity": s9a.compute_depth_selectivity(records),
            "trajectories": s9a.compute_depth_radius_trajectories(records),
            "specialization": s9a.compute_specialization_by_depth_radius(records),
            "numerical_audit": s9a.compute_numerical_patch_audit(records),
        })

    import json
    first = json.dumps(run_once(), sort_keys=True)
    second = json.dumps(run_once(), sort_keys=True)
    assert first == second


def test_specialization_by_depth_radius_is_6x6_per_child_region_radius_cell():
    records = _build_synthetic_records()
    specialization = s9a.compute_specialization_by_depth_radius(records)
    for region in STAGE9_CHILD_REGIONS:
        for radius in STAGE9_RADII:
            cell = specialization[region][str(radius)]
            matrix = cell["spearman_6x6"]
            assert len(matrix) == 6 and all(len(row) == 6 for row in matrix)


def test_radius_trajectories_pairing_reuses_stage8_direction_family_grouping():
    records = _build_synthetic_records()
    trajectories = s9a.compute_depth_radius_trajectories(records)
    assert trajectories["radii"] == list(STAGE9_RADII)
    assert trajectories["n_complete_trajectories"] == len(STAGE9_CHILD_REGIONS) * N_DIRECTIONS * len(STAGE9_CAPABILITIES)


# =================================================================================================
# Cross-check against the REAL completed Stage-9 run, if present locally -- never fabricated,
# skipped cleanly when the directory doesn't exist (e.g. a fresh checkout without the uploaded
# real results).
# =================================================================================================


@pytest.mark.skipif(not REAL_RESULTS_DIR.exists(), reason="real Stage-9 results directory not present locally")
def test_real_run_passes_the_integrity_gate():
    import json

    records = s9a.load_all(REAL_RESULTS_DIR)
    checkpoint = json.loads((REAL_RESULTS_DIR / "checkpoint_manifest.json").read_text())
    run_manifest = json.loads((REAL_RESULTS_DIR / "run_manifest.json").read_text())
    report = s9a.run_stage9_integrity_gate(records, checkpoint, run_manifest)
    assert report["all_checks_pass"] is True


@pytest.mark.skipif(not REAL_RESULTS_DIR.exists(), reason="real Stage-9 results directory not present locally")
def test_real_run_baselines_match_stage8_authoritative():
    import json

    records = s9a.load_all(REAL_RESULTS_DIR)
    baseline_scores = json.loads((REAL_RESULTS_DIR / "baseline_scores.json").read_text())
    baseline_table = s9a.compute_baseline_table(records, baseline_scores)
    check = s9a.ensure_baselines_match_stage8_authoritative(baseline_table)
    assert check["all_match_stage8_authoritative"] is True


@pytest.mark.skipif(not REAL_RESULTS_DIR.exists(), reason="real Stage-9 results directory not present locally")
def test_real_run_zero_admissibility_violations():
    records = s9a.load_all(REAL_RESULTS_DIR)
    audit = s9a.compute_numerical_patch_audit(records)
    assert audit["zero_admissibility_violations"] is True
    assert audit["known_reported_bracket_expansion_candidate"]["found_in_results"] is True
