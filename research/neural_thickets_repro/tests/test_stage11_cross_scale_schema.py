"""Tests for analysis/stage11_cross_scale_schema.py -- PREPARED, NOT RUN against real data (no
real Stage-11 7B results exist yet). Every function is verified against small synthetic
ExperimentResultRecord grids built for BOTH "scales" (model_scale="3B" and "7B") sharing the same
D_map subset hashes, mirroring the real design's same-example guarantee.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import List

import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import stage11_cross_scale_schema as s11s  # noqa: E402

from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_RADII, STAGE8_REGIONS  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

N_DIRECTIONS = 4
BASE_SCORES = {cap: 0.5 for cap in STAGE8_CAPABILITIES}


def _rec(*, capability: str, region: str, radius: float, direction_index: int, delta: float, model_scale: str = "3B") -> ExperimentResultRecord:
    base = BASE_SCORES[capability]
    pid = f"{model_scale}_{region}_{radius}_{direction_index}"
    return ExperimentResultRecord(
        experiment_id="test", perturbation_id=pid, model_family="qwen2_5_vl", model_scale=model_scale,
        model_revision="rev1", perturbation_mode="anatomical_relative_l2", anatomy_region=region, radius=radius, sigma=None,
        seed=direction_index, parameter_mask_hash=f"mask_{region}_{model_scale}", capability=capability, dataset_role="map",
        subset_hash=f"sub_{capability}", base_score=base, perturbed_score=round(base + delta, 10), delta=delta,
        parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash=f"h_{pid}_{capability}",
        runtime_metadata={
            "direction_family_id": f"{region}:{direction_index}", "direction_seed": direction_index,
            "direction_index": direction_index, "region": region,
            "radius_acceptance_mode": "strict", "quantization_limited": False,
            "requested_relative_l2": radius, "realized_relative_l2": radius,
            "relative_radius_error": 0.0,
        },
    )


def _delta_fn(region: str, radius: float, direction_index: int, capability: str, model_scale: str) -> float:
    radius_rank = STAGE8_RADII.index(radius)
    scale_factor = 1.0 if model_scale == "3B" else 1.3  # 7B slightly more responsive, by construction, so cross-scale diffs are non-trivial
    if region == "language" and capability == "spatial_reasoning":
        return scale_factor * ([0.05, 0.02, -0.03][radius_rank] - 0.01 * (direction_index % 3))
    return scale_factor * (-0.01 * (radius_rank + 1) - 0.001 * direction_index)


def _build_records(model_scale: str, n_directions: int = N_DIRECTIONS) -> List[ExperimentResultRecord]:
    records = []
    for region in STAGE8_REGIONS:
        for radius in STAGE8_RADII:
            for direction_index in range(n_directions):
                for capability in STAGE8_CAPABILITIES:
                    delta = _delta_fn(region, radius, direction_index, capability, model_scale)
                    records.append(_rec(capability=capability, region=region, radius=radius, direction_index=direction_index, delta=delta, model_scale=model_scale))
    return records


# =================================================================================================
# Match-key schema
# =================================================================================================


def test_build_match_keys_covers_the_full_cartesian_product():
    keys = s11s.build_match_keys(STAGE8_REGIONS, STAGE8_RADII, STAGE8_CAPABILITIES)
    assert len(keys) == len(STAGE8_REGIONS) * len(STAGE8_RADII) * len(STAGE8_CAPABILITIES)
    assert len(set(keys)) == len(keys)


def test_ensure_cross_scale_design_matches_passes_for_same_example_records():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    s11s.ensure_cross_scale_design_matches(stage8, stage11)  # must not raise


def _with_subset_hash(record: ExperimentResultRecord, subset_hash: str) -> ExperimentResultRecord:
    import dataclasses
    return dataclasses.replace(record, subset_hash=subset_hash)


def test_ensure_cross_scale_design_matches_detects_different_subset_hashes():
    stage8 = _build_records("3B")
    stage11 = [_with_subset_hash(r, "a_totally_different_subset_hash") for r in _build_records("7B")]
    with pytest.raises(s11s.CrossScaleDesignMismatchError):
        s11s.ensure_cross_scale_design_matches(stage8, stage11)


def test_ensure_cross_scale_design_matches_detects_different_regions():
    stage8 = _build_records("3B")
    stage11 = [r for r in _build_records("7B") if r.anatomy_region != "vision"]
    with pytest.raises(s11s.CrossScaleDesignMismatchError):
        s11s.ensure_cross_scale_design_matches(stage8, stage11)


def test_ensure_cross_scale_design_matches_detects_different_radii():
    stage8 = _build_records("3B")
    stage11 = [r for r in _build_records("7B") if r.radius != STAGE8_RADII[0]]
    with pytest.raises(s11s.CrossScaleDesignMismatchError):
        s11s.ensure_cross_scale_design_matches(stage8, stage11)


# =================================================================================================
# Per-cell comparison record
# =================================================================================================


def test_build_cross_scale_cell_comparisons_covers_every_matched_cell():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    comparisons = s11s.build_cross_scale_cell_comparisons(stage8, stage11)
    assert len(comparisons) == len(STAGE8_REGIONS) * len(STAGE8_RADII) * len(STAGE8_CAPABILITIES)


def test_cross_scale_comparison_diff_matches_manual_computation():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    comparisons = s11s.build_cross_scale_cell_comparisons(stage8, stage11)
    cell = comparisons[f"spatial_reasoning:language:{STAGE8_RADII[0]}"]
    expected_diff = cell.stage11_7b["mean_delta"] - cell.stage8_3b["mean_delta"]
    assert cell.mean_delta_diff_7b_minus_3b == pytest.approx(expected_diff)


def test_cross_scale_comparisons_never_pool_radii():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    comparisons = s11s.build_cross_scale_cell_comparisons(stage8, stage11)
    radii_seen = {c.match_key.radius for c in comparisons.values()}
    assert radii_seen == set(STAGE8_RADII)
    # every individual comparison's own two sides share the SAME radius (never pooled/averaged)
    for c in comparisons.values():
        assert c.stage8_3b["radius"] == c.stage11_7b["radius"] == c.match_key.radius


# =================================================================================================
# Six explicit cross-scale questions (A-F)
# =================================================================================================


def test_question_A_reports_exhaustive_partition_of_all_cells():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    comparisons = s11s.build_cross_scale_cell_comparisons(stage8, stage11)
    result = s11s.question_A_specialist_existence_reproduces(comparisons)
    assert result["n_cells_both_scales"] + result["n_cells_only_3b"] + result["n_cells_only_7b"] + result["n_cells_neither"] == result["n_total"]


def test_question_B_reuses_stage8_anatomy_capability_interaction():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    result = s11s.question_B_coarse_anatomy_interaction_reproduces(stage8, stage11)
    assert set(result["per_capability_agreement"].keys()) == set(STAGE8_CAPABILITIES)
    assert isinstance(result["n_agreeing"], int)


def test_question_C_spatial_reasoning_language_reports_both_scales_by_radius():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    comparisons = s11s.build_cross_scale_cell_comparisons(stage8, stage11)
    result = s11s.question_C_spatial_reasoning_still_language_preferential(comparisons)
    assert result["n_radii"] == len(STAGE8_RADII)
    assert set(result["mean_delta_3b_by_radius"].keys()) == {str(r) for r in STAGE8_RADII}


def test_question_D_reuses_stage8_specialization_machinery():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    result = s11s.question_D_specialization_increases_with_radius(stage8, stage11)
    assert set(result["discordance_3b_by_region_radius"].keys()) == set(STAGE8_REGIONS)
    assert set(result["discordance_7b_by_region_radius"].keys()) == set(STAGE8_REGIONS)


def test_question_E_partitions_cells_by_density_direction():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    comparisons = s11s.build_cross_scale_cell_comparisons(stage8, stage11)
    result = s11s.question_E_useful_expert_density_vs_scale(comparisons)
    total = result["n_cells_density_increased_at_7b"] + result["n_cells_density_decreased_at_7b"] + result["n_cells_unchanged"]
    assert total == len(comparisons)


def test_question_F_reuses_stage8_radius_trajectories():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    result = s11s.question_F_larger_models_tolerate_greater_displacement(stage8, stage11)
    assert "improvement_survival_rate_3b" in result
    assert "improvement_survival_rate_7b" in result


def test_build_cross_scale_report_contains_all_six_questions():
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    report = s11s.build_cross_scale_report(stage8, stage11)
    for letter in "ABCDEF":
        assert any(k.startswith(f"question_{letter}_") for k in report)


# =================================================================================================
# Terminology guard -- never "scaling law" for a two-scale-point comparison
# =================================================================================================


def test_module_source_never_uses_the_forbidden_scaling_law_phrase_as_a_claim():
    source = inspect.getsource(s11s)
    # The phrase may appear ONLY inside the explicit ban/documentation strings that name it as
    # forbidden -- never as an actual claim asserted about the data.
    for line in source.splitlines():
        if s11s.FORBIDDEN_TERM in line.lower():
            assert "never" in line.lower() or "forbidden" in line.lower() or "FORBIDDEN_TERM" in line or "ban" in line.lower()


def test_build_cross_scale_report_uses_only_approved_terminology():
    """The terminology_note is ALLOWED to name the forbidden phrase as part of explicitly
    banning it (e.g. "never constitute a 'scaling law'") -- it must never appear as an actual
    claim, and the approved replacement terms must be present.
    """
    stage8 = _build_records("3B")
    stage11 = _build_records("7B")
    report = s11s.build_cross_scale_report(stage8, stage11)
    note = report["terminology_note"]
    if s11s.FORBIDDEN_TERM in note.lower():
        assert "never" in note.lower()
    assert any(term in note for term in s11s.APPROVED_TERMS)


# =================================================================================================
# Not run against real data yet
# =================================================================================================


def test_main_reports_nothing_to_analyze_when_no_real_stage11_results_exist(tmp_path, capsys):
    empty_stage11_dir = tmp_path / "no_results_here"
    empty_stage11_dir.mkdir()
    exit_code = s11s.main(["--stage11-dir", str(empty_stage11_dir), "--stage8-dir", str(tmp_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "not run" in captured.out.lower() or "no stage-11 results" in captured.out.lower()
