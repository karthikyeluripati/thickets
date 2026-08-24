"""Tests for analysis/stage7b_anatomical_calibration_analysis.py -- pure Python/numpy, no
GPU/ray/vllm dependency. Exercises every function against small, hand-built
ExperimentResultRecord fixtures (never against real pod output), plus one full end-to-end
main() test against a synthetic-but-structurally-complete (real frozen 3x6x8 grid shape) run
written to a tmp_path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import stage7b_anatomical_calibration_analysis as sca  # noqa: E402

from neural_thickets_repro.run_stage7b_anatomical_calibration import (  # noqa: E402
    FULL_CALIBRATION_D_MAP_N,
    FULL_CALIBRATION_N_PER_CELL,
    FULL_CALIBRATION_RADII,
    FULL_CALIBRATION_REGIONS,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    RADIUS_REALIZATION_METHOD,
    REALIZED_RADIUS_TOLERANCE,
    Stage7bCheckpointManifest,
)
from neural_thickets_repro.thicket.schema import ExperimentResultRecord

CAPABILITIES = ("cap_a", "cap_b", "cap_c")


def _record(
    *, perturbation_id: str, region: str, radius: float, seed: int, capability: str,
    delta: float, base_score: float = 0.5, per_example_result_hash: str = "h",
    radius_acceptance_mode: str = "strict", quantization_limited: bool = False,
    absolute_radius_error: float = 1e-8, relative_radius_error: float = 1e-6,
    epsilon_region_l2_norm: float = 1.0, model_revision: str = "rev1", parameter_mask_hash: str = "mh",
    realized_relative_l2: Optional[float] = None,
) -> ExperimentResultRecord:
    realized_relative_l2 = radius if realized_relative_l2 is None else realized_relative_l2
    return ExperimentResultRecord(
        experiment_id="stage7b_anatomical_calibration", perturbation_id=perturbation_id,
        model_family="qwen2_5_vl", model_scale="3B", model_revision=model_revision,
        perturbation_mode="anatomical_relative_l2", anatomy_region=region, radius=radius, sigma=None,
        seed=seed, parameter_mask_hash=parameter_mask_hash, capability=capability, dataset_role="map",
        subset_hash="sh", base_score=base_score, perturbed_score=base_score + delta, delta=delta,
        parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash=per_example_result_hash,
        runtime_metadata={
            "radius_realization_method": RADIUS_REALIZATION_METHOD,
            "radius_acceptance_mode": radius_acceptance_mode, "quantization_limited": quantization_limited,
            "requested_relative_l2": radius, "realized_relative_l2": realized_relative_l2,
            "absolute_radius_error": absolute_radius_error, "relative_radius_error": relative_radius_error,
            "epsilon_region_l2_norm": epsilon_region_l2_norm,
        },
    )


# =============================================================================================
# validate_run_integrity
# =============================================================================================


def _build_full_records(*, contaminate_regions: Sequence[str] = (), model_revision: str = "rev1") -> list:
    """A structurally-complete synthetic run matching the REAL frozen 3x6x8 grid shape (3
    regions x 6 radii x 8 seeds x 3 capabilities = 432 rows), so validate_run_integrity's grid
    check can pass. `contaminate_regions` forces delta=0.0 + a single shared hash for those
    regions (simulating the real encoder-cache artifact); other regions get small varying deltas.
    """
    records = []
    rng = np.random.default_rng(0)
    for region in FULL_CALIBRATION_REGIONS:
        for radius in FULL_CALIBRATION_RADII:
            for seed_idx in range(FULL_CALIBRATION_N_PER_CELL):
                pid = f"{region}|{radius}|{seed_idx}"
                for cap in CAPABILITIES:
                    if region in contaminate_regions:
                        delta = 0.0
                        h = f"const_hash_{region}_{cap}"
                    else:
                        delta = float(rng.normal(0, 0.01))
                        h = f"hash_{region}_{radius}_{seed_idx}_{cap}"
                    records.append(_record(
                        perturbation_id=pid, region=region, radius=radius, seed=seed_idx, capability=cap,
                        delta=delta, per_example_result_hash=h, model_revision=model_revision,
                        parameter_mask_hash=f"mh_{region}",
                    ))
    return records


def _checkpoint_for(records) -> Stage7bCheckpointManifest:
    return Stage7bCheckpointManifest(
        experiment_id="stage7b_anatomical_calibration", run_signature="full_test",
        restoration_mode="fixed_base", perturbation_mode="anatomical_relative_l2",
        radius_realization_method=RADIUS_REALIZATION_METHOD, multimodal_cache_policy="full_reset_on_weight_change_v1",
        model_revision="rev1", dataset_role="map",
        regions=FULL_CALIBRATION_REGIONS, radii=FULL_CALIBRATION_RADII, capabilities=CAPABILITIES,
        n_per_cell=FULL_CALIBRATION_N_PER_CELL, d_map_n=FULL_CALIBRATION_D_MAP_N,
        subset_hashes={c: "sh" for c in CAPABILITIES},
        region_mask_hashes={r: f"mh_{r}" for r in FULL_CALIBRATION_REGIONS},
        expected_unique_perturbations=len(FULL_CALIBRATION_REGIONS) * len(FULL_CALIBRATION_RADII) * FULL_CALIBRATION_N_PER_CELL,
        expected_result_rows=len(FULL_CALIBRATION_REGIONS) * len(FULL_CALIBRATION_RADII) * FULL_CALIBRATION_N_PER_CELL * len(CAPABILITIES),
    )


def test_validate_run_integrity_passes_on_a_complete_3x6x8_grid():
    records = _build_full_records()
    checkpoint = _checkpoint_for(records)
    report = sca.validate_run_integrity(records, checkpoint, run_manifest={"run_complete": True})
    assert report["overall_pass"] is True
    assert report["grid_3x6x8_complete"] is True
    assert report["actual_unique_perturbations"] == 144
    assert report["actual_result_rows"] == 432


def test_validate_run_integrity_detects_exactly_3_capability_rows_per_perturbation():
    records = _build_full_records()
    checkpoint = _checkpoint_for(records)
    # Drop one capability row from one perturbation -- now incomplete.
    victim_pid = records[0].perturbation_id
    broken = [r for r in records if not (r.perturbation_id == victim_pid and r.capability == CAPABILITIES[0])]
    report = sca.validate_run_integrity(broken, checkpoint, run_manifest={"run_complete": False})
    assert report["capability_rows_per_perturbation_complete"] is False
    assert report["overall_pass"] is False


def test_validate_run_integrity_detects_missing_grid_cell():
    records = _build_full_records()
    checkpoint = _checkpoint_for(records)
    victim_pid = records[0].perturbation_id
    broken = [r for r in records if r.perturbation_id != victim_pid]
    report = sca.validate_run_integrity(broken, checkpoint, run_manifest={"run_complete": False})
    assert report["grid_3x6x8_complete"] is False
    assert report["overall_pass"] is False


def test_validate_run_integrity_detects_model_revision_mismatch():
    records = _build_full_records()
    checkpoint = _checkpoint_for(records)
    records[0] = _record(
        perturbation_id=records[0].perturbation_id, region=records[0].anatomy_region, radius=records[0].radius,
        seed=records[0].seed, capability=records[0].capability, delta=0.0, model_revision="rev_DIFFERENT",
        parameter_mask_hash=f"mh_{records[0].anatomy_region}",
    )
    report = sca.validate_run_integrity(records, checkpoint, run_manifest={"run_complete": True})
    assert report["model_revision_uniform_and_matches_checkpoint"] is False
    assert report["overall_pass"] is False


def test_validate_run_integrity_detects_mask_hash_mismatch():
    records = _build_full_records()
    checkpoint = _checkpoint_for(records)
    r0 = records[0]
    records[0] = _record(
        perturbation_id=r0.perturbation_id, region=r0.anatomy_region, radius=r0.radius, seed=r0.seed,
        capability=r0.capability, delta=0.0, parameter_mask_hash="WRONG_HASH",
    )
    report = sca.validate_run_integrity(records, checkpoint, run_manifest={"run_complete": True})
    assert report["region_mask_hashes_match_checkpoint"] is False
    assert report["region_mask_hash_mismatch_count"] >= 1


def test_validate_run_integrity_reports_quantization_limited_acceptance_counts_by_region_radius():
    records = _build_full_records()
    checkpoint = _checkpoint_for(records)
    report = sca.validate_run_integrity(records, checkpoint, run_manifest={"run_complete": True})
    counts = report["quantization_limited_acceptance_counts_by_region_radius"]
    # every synthetic candidate was built with radius_acceptance_mode="strict" by default
    for key, c in counts.items():
        assert c["strict"] == FULL_CALIBRATION_N_PER_CELL
        assert c["quantization_limited"] == 0


# =============================================================================================
# compute_data_integrity_report -- the cache-artifact detector
# =============================================================================================


def test_data_integrity_report_flags_a_genuinely_contaminated_region():
    records = _build_full_records(contaminate_regions=["vision"])
    report = sca.compute_data_integrity_report(records)
    assert "vision" in report["affected_regions"]
    assert "language" not in report["affected_regions"]
    for key, row in report["per_capability_region"].items():
        if row["region"] == "vision":
            assert row["suspected_stale_encoder_cache_artifact"] is True
            assert row["n_unique_per_example_result_hash"] == 1
        if row["region"] == "language":
            assert row["suspected_stale_encoder_cache_artifact"] is False


def test_data_integrity_report_does_not_flag_a_real_zero_perturbation():
    """A region with delta==0.0 because epsilon_region_l2_norm is genuinely 0.0 (no
    perturbation applied at all) must NOT be flagged as a caching artifact -- the detector
    requires BOTH all-zero delta AND a nonzero applied perturbation.
    """
    records = [
        _record(perturbation_id=f"p{i}", region="vision", radius=0.01, seed=i, capability=cap,
                delta=0.0, per_example_result_hash=f"h{i}_{cap}", epsilon_region_l2_norm=0.0)
        for i in range(3) for cap in CAPABILITIES
    ]
    report = sca.compute_data_integrity_report(records)
    assert report["affected_regions"] == []


def test_data_integrity_report_does_not_flag_a_region_with_real_signal():
    records = _build_full_records()  # no contamination
    report = sca.compute_data_integrity_report(records)
    assert report["affected_regions"] == []
    assert "No stale-encoder-cache artifact detected" in report["conclusion"]


def test_data_integrity_report_provenance_fields_for_a_contaminated_run():
    records = _build_full_records(contaminate_regions=["vision", "multimodal_connector_or_merger"])
    report = sca.compute_data_integrity_report(records)
    assert report["scientific_status"] == "partially_invalid"
    assert report["valid_regions"] == ["language"]
    assert report["invalid_regions"] == ["multimodal_connector_or_merger", "vision"]
    assert report["invalid_reason"] == "stale multimodal encoder cache after anatomical weight changes"
    # 2 regions x 6 radii x 8 seeds x 3 capabilities
    assert report["invalid_row_count"] == 2 * 6 * 8 * 3 == 288
    assert report["total_row_count"] == len(records) == 432


def test_data_integrity_report_provenance_fields_for_a_clean_run():
    records = _build_full_records()  # no contamination
    report = sca.compute_data_integrity_report(records)
    assert report["scientific_status"] == "valid"
    assert set(report["valid_regions"]) == set(FULL_CALIBRATION_REGIONS)
    assert report["invalid_regions"] == []
    assert report["invalid_reason"] is None
    assert report["invalid_row_count"] == 0


def test_data_integrity_report_requires_all_three_symptoms_together():
    """Nonzero delta but a collapsed hash (e.g. a scoring quirk, not a cache bug) must not be
    flagged -- all_delta_exactly_zero is required alongside the collapsed hash.
    """
    records = [
        _record(perturbation_id=f"p{i}", region="vision", radius=0.01, seed=i, capability="cap_a",
                delta=0.01 * i, per_example_result_hash="same_hash_for_all")
        for i in range(3)
    ]
    report = sca.compute_data_integrity_report(records)
    assert report["affected_regions"] == []


# =============================================================================================
# unique_candidates_by_region_radius -- dedupe by perturbation, not by row
# =============================================================================================


def test_unique_candidates_by_region_radius_dedupes_capability_rows():
    records = _build_full_records()
    by_cell = sca.unique_candidates_by_region_radius(records)
    for (region, radius), cands in by_cell.items():
        assert len(cands) == FULL_CALIBRATION_N_PER_CELL
        assert len({c.perturbation_id for c in cands}) == FULL_CALIBRATION_N_PER_CELL


# =============================================================================================
# compute_calibration_table
# =============================================================================================


def test_compute_calibration_table_matches_manual_statistics():
    deltas = [0.1, -0.05, 0.0, 0.2, -0.1, 0.02, 0.0, 0.3]
    records = [
        _record(perturbation_id=f"p{i}", region="vision", radius=0.01, seed=i, capability="cap_a", delta=d)
        for i, d in enumerate(deltas)
    ]
    table = sca.compute_calibration_table(records)
    cell = table["cap_a"]["vision"][sca._radius_key(0.01)]
    arr = np.array(deltas)
    assert cell["n"] == 8
    assert cell["mean_delta"] == pytest.approx(float(arr.mean()))
    assert cell["median_delta"] == pytest.approx(float(np.median(arr)))
    assert cell["std_delta"] == pytest.approx(float(arr.std()))
    assert cell["min_delta"] == pytest.approx(-0.1)
    assert cell["max_delta"] == pytest.approx(0.3)
    assert cell["p_delta_gt_0"] == pytest.approx(4 / 8)  # 0.1, 0.2, 0.02, 0.3
    assert cell["p_delta_lt_0"] == pytest.approx(2 / 8)  # -0.05, -0.1
    assert cell["density"]["0.02"] == pytest.approx(float(np.mean(arr >= 0.02)))
    assert cell["density"]["0.05"] == pytest.approx(float(np.mean(arr >= 0.05)))
    assert cell["positive_thicket_mass"] == pytest.approx(float(np.mean(np.maximum(arr, 0.0))))
    lo, hi = cell["mean_delta_95ci_bootstrap"]
    assert lo <= cell["mean_delta"] <= hi


def test_compute_calibration_table_is_deterministic():
    records = _build_full_records()
    t1 = sca.compute_calibration_table(records)
    t2 = sca.compute_calibration_table(records)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)


# =============================================================================================
# compute_matched_radius_comparison
# =============================================================================================


def test_matched_radius_comparison_separates_regions_at_the_same_radius():
    records = [
        _record(perturbation_id="a", region="vision", radius=0.01, seed=0, capability="cap_a", delta=0.5),
        _record(perturbation_id="b", region="language", radius=0.01, seed=0, capability="cap_a", delta=-0.5),
    ]
    out = sca.compute_matched_radius_comparison(records)
    cell = out["cap_a"][sca._radius_key(0.01)]
    assert cell["mean_delta_by_region"]["vision"] == pytest.approx(0.5)
    assert cell["mean_delta_by_region"]["language"] == pytest.approx(-0.5)
    assert cell["p_delta_gt_0_by_region"]["vision"] == 1.0
    assert cell["p_delta_gt_0_by_region"]["language"] == 0.0


# =============================================================================================
# compute_collapse_regime
# =============================================================================================


def test_compute_collapse_regime_severe_margin_and_aggregation():
    deltas = [-0.2, -0.1, -0.05, 0.0, 0.1]  # 2 of 5 are <= -0.10 (severe)
    records = [
        _record(perturbation_id=f"p{i}", region="language", radius=0.5, seed=i, capability="cap_a", delta=d)
        for i, d in enumerate(deltas)
    ]
    out = sca.compute_collapse_regime(records)
    cell = out["language"][sca._radius_key(0.5)]
    assert cell["mean_capability_delta"] == pytest.approx(np.mean(deltas))
    assert cell["fraction_delta_lt_0"] == pytest.approx(3 / 5)
    assert cell["fraction_delta_le_severe"] == pytest.approx(2 / 5)


# =============================================================================================
# classify_regime -- every branch
# =============================================================================================


def test_classify_regime_destructive():
    assert sca.classify_regime(mean_delta=-0.2, p_gt0=0.0, p_lt0=0.9, density_at_02=0.0) == "destructive"


def test_classify_regime_near_base():
    assert sca.classify_regime(mean_delta=0.001, p_gt0=0.05, p_lt0=0.05, density_at_02=0.0) == "near_base"


def test_classify_regime_active():
    assert sca.classify_regime(mean_delta=0.05, p_gt0=0.6, p_lt0=0.1, density_at_02=0.5) == "active"


def test_classify_regime_transition_fallthrough():
    assert sca.classify_regime(mean_delta=0.01, p_gt0=0.3, p_lt0=0.3, density_at_02=0.1) == "transition"


# =============================================================================================
# classify_common_radius_regime / classify_language_only_radius_regime --
# "no region-specific radius optimization"
# =============================================================================================


def test_common_radius_classification_pools_all_regions_not_per_region():
    records = _build_full_records(contaminate_regions=["vision", "multimodal_connector_or_merger"])
    out = sca.classify_common_radius_regime(records)
    # exactly one entry per radius -- never per (region, radius): confirms no region-specific
    # radius selection is happening in this function.
    assert set(out.keys()) == {sca._radius_key(r) for r in FULL_CALIBRATION_RADII}
    for radius, cell in out.items():
        assert cell["n"] == len(FULL_CALIBRATION_REGIONS) * FULL_CALIBRATION_N_PER_CELL * len(CAPABILITIES)


def test_language_only_classification_uses_only_language_rows():
    records = _build_full_records()
    out = sca.classify_language_only_radius_regime(records)
    for radius, cell in out.items():
        assert cell["n"] == FULL_CALIBRATION_N_PER_CELL * len(CAPABILITIES)


def test_no_region_specific_radius_selection_function_exists_in_module():
    """Structural guard: this module must never define a function that picks a DIFFERENT
    'best' radius per region -- the Stage-8 radius set must be COMMON across regions.
    """
    import inspect
    names = [name for name, obj in vars(sca).items() if inspect.isfunction(obj) and obj.__module__ == sca.__name__]
    forbidden_substrings = ("best_radius_for_region", "optimal_radius_per_region", "select_region_specific_radius")
    for name in names:
        for forbidden in forbidden_substrings:
            assert forbidden not in name


# =============================================================================================
# compute_exploratory_anatomy_signal
# =============================================================================================


def test_exploratory_anatomy_signal_filters_to_given_radii_only():
    records = _build_full_records()
    keep_radii = FULL_CALIBRATION_RADII[:2]
    out = sca.compute_exploratory_anatomy_signal(records, keep_radii)
    assert out["non_destructive_radii_used"] == list(keep_radii)
    assert "caveat" in out
    for cap, by_region in out["capability_by_anatomy"].items():
        for region, cell in by_region.items():
            assert cell["n"] == len(keep_radii) * FULL_CALIBRATION_N_PER_CELL


# =============================================================================================
# compute_diversity_by_region_radius
# =============================================================================================


def test_diversity_by_region_radius_matches_direct_thicket_diversity_call():
    from neural_thickets_repro.thicket import diversity as thicket_diversity
    from neural_thickets_repro.run_global_visual_thicket_pilot import build_delta_matrix

    records = [
        _record(perturbation_id=f"p{i}", region="language", radius=0.1, seed=i, capability=cap, delta=float(i * (1 if cap == "cap_a" else -1)))
        for i in range(4) for cap in ("cap_a", "cap_b")
    ]
    out = sca.compute_diversity_by_region_radius(records)
    cell = out["language"][sca._radius_key(0.1)]

    _, _, matrix = build_delta_matrix(records)
    expected_sd = thicket_diversity.spectral_discordance(matrix)
    assert cell["spectral_discordance"] == pytest.approx(expected_sd)
    assert cell["n_perturbations"] == 4


def test_diversity_by_region_radius_flags_spurious_perfect_agreement_for_constant_zero_deltas():
    records = [
        _record(perturbation_id=f"p{i}", region="vision", radius=0.1, seed=i, capability=cap, delta=0.0)
        for i in range(4) for cap in ("cap_a", "cap_b")
    ]
    out = sca.compute_diversity_by_region_radius(records)
    cell = out["vision"][sca._radius_key(0.1)]
    assert cell["spectral_discordance"] == pytest.approx(0.0)


# =============================================================================================
# compute_quantization_audit
# =============================================================================================


def test_quantization_audit_reports_no_violations_for_well_formed_data():
    records = _build_full_records()
    audit = sca.compute_quantization_audit(records)
    assert audit["n_violations"] == 0
    assert audit["all_accepted_candidates_within_v3_admissibility_rule"] is True
    for key, cell in audit["per_region_radius"].items():
        assert cell["n_candidates"] == FULL_CALIBRATION_N_PER_CELL
        assert cell["count_strict"] == FULL_CALIBRATION_N_PER_CELL


def test_quantization_audit_detects_a_strict_violation():
    records = [
        _record(perturbation_id="p0", region="vision", radius=0.01, seed=0, capability=cap, delta=0.0,
                radius_acceptance_mode="strict", absolute_radius_error=REALIZED_RADIUS_TOLERANCE * 10)
        for cap in CAPABILITIES
    ]
    audit = sca.compute_quantization_audit(records)
    assert audit["n_violations"] == 1
    assert audit["all_accepted_candidates_within_v3_admissibility_rule"] is False
    assert audit["violations"][0]["mode"] == "strict"


def test_quantization_audit_detects_a_quantization_limited_violation():
    records = [
        _record(perturbation_id="p0", region="vision", radius=0.01, seed=0, capability=cap, delta=0.0,
                radius_acceptance_mode="quantization_limited", quantization_limited=True,
                relative_radius_error=QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE * 10)
        for cap in CAPABILITIES
    ]
    audit = sca.compute_quantization_audit(records)
    assert audit["n_violations"] == 1
    assert audit["violations"][0]["mode"] == "quantization_limited"


def test_quantization_audit_mean_ratio_and_max_error():
    records = [
        _record(perturbation_id="p0", region="vision", radius=0.02, seed=0, capability=cap, delta=0.0,
                realized_relative_l2=0.02, relative_radius_error=0.0001)
        for cap in CAPABILITIES
    ] + [
        _record(perturbation_id="p1", region="vision", radius=0.02, seed=1, capability=cap, delta=0.0,
                realized_relative_l2=0.021, relative_radius_error=0.0005)
        for cap in CAPABILITIES
    ]
    audit = sca.compute_quantization_audit(records)
    cell = audit["per_region_radius"][f"vision|{sca._radius_key(0.02)}"]
    assert cell["n_candidates"] == 2
    assert cell["mean_realized_over_requested_ratio"] == pytest.approx((1.0 + 1.05) / 2)
    assert cell["max_relative_radius_error"] == pytest.approx(0.0005)


# =============================================================================================
# End-to-end main() -- output files, determinism, integrity gate
# =============================================================================================


def _write_full_synthetic_run(results_dir: Path, *, contaminate_regions: Sequence[str] = ()) -> None:
    records = _build_full_records(contaminate_regions=contaminate_regions)
    checkpoint = _checkpoint_for(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    (results_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint.to_dict(), indent=2))
    (results_dir / "baseline_scores.json").write_text(json.dumps({
        "model_revision": "rev1", "run_signature": "full_test",
        "capabilities": {c: {"score": 0.5, "subset_hash": "sh"} for c in CAPABILITIES},
    }, indent=2))
    (results_dir / "run_manifest.json").write_text(json.dumps({
        "run_complete": True,
        "expected_unique_perturbations": checkpoint.expected_unique_perturbations,
        "actual_unique_perturbations": checkpoint.expected_unique_perturbations,
        "expected_result_rows": checkpoint.expected_result_rows,
        "actual_result_rows": checkpoint.expected_result_rows,
    }, indent=2))


def test_main_writes_all_seven_expected_outputs(tmp_path):
    results_dir = tmp_path / "full_test"
    _write_full_synthetic_run(results_dir)
    rc = sca.main(["--results-dir", str(results_dir)])
    assert rc == 0
    analysis_dir = results_dir / "analysis"
    for name in (
        "calibration_table.json", "matched_radius_anatomy_comparison.json", "radius_regime_summary.json",
        "exploratory_anatomy_signal.json", "diversity_by_region_radius.json", "quantization_audit.json",
        "stage7b_analysis.md",
    ):
        path = analysis_dir / name
        assert path.exists(), f"missing {name}"
        if name.endswith(".json"):
            json.loads(path.read_text())  # must be valid JSON


def test_main_refuses_an_incomplete_run(tmp_path):
    results_dir = tmp_path / "full_test"
    _write_full_synthetic_run(results_dir)
    # Corrupt: delete rows for one perturbation so the grid is incomplete.
    lines = (results_dir / "results.jsonl").read_text().splitlines()
    first_pid = json.loads(lines[0])["perturbation_id"]
    kept = [l for l in lines if json.loads(l)["perturbation_id"] != first_pid]
    (results_dir / "results.jsonl").write_text("\n".join(kept) + "\n")

    with pytest.raises(sca.RunIntegrityError):
        sca.main(["--results-dir", str(results_dir)])


def test_main_outputs_are_deterministic_across_repeated_runs(tmp_path):
    results_dir = tmp_path / "full_test"
    _write_full_synthetic_run(results_dir)
    sca.main(["--results-dir", str(results_dir)])
    analysis_dir = results_dir / "analysis"
    first = {name: (analysis_dir / name).read_text() for name in (
        "calibration_table.json", "matched_radius_anatomy_comparison.json", "radius_regime_summary.json",
        "exploratory_anatomy_signal.json", "diversity_by_region_radius.json", "quantization_audit.json",
        "stage7b_analysis.md",
    )}
    sca.main(["--results-dir", str(results_dir)])
    for name, content in first.items():
        assert (analysis_dir / name).read_text() == content, f"{name} was not deterministic across repeated runs"


def test_main_surfaces_the_data_integrity_warning_when_contaminated(tmp_path):
    results_dir = tmp_path / "full_test"
    _write_full_synthetic_run(results_dir, contaminate_regions=["vision", "multimodal_connector_or_merger"])
    sca.main(["--results-dir", str(results_dir)])
    regime_summary = json.loads((results_dir / "analysis" / "radius_regime_summary.json").read_text())
    warning = regime_summary["data_integrity_warning"]
    assert set(warning["affected_regions"]) == {"vision", "multimodal_connector_or_merger"}
    md = (results_dir / "analysis" / "stage7b_analysis.md").read_text()
    assert "CRITICAL FINDING" in md
