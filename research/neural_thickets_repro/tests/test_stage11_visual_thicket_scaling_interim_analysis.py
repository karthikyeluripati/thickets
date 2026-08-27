"""Tests for analysis/stage11_visual_thicket_scaling_interim_analysis.py -- the first authoritative
3B-vs-7B interim whole-model scale analysis. Covers: authoritative-run discovery (excludes smoke),
per-scale and cross-scale integrity, baseline table, the 36-cell statistics table, solution-density
curves (monotonicity + common margin grid), visual-macro density (candidate-row-preserving
bootstrap), performance-density/Wasserstein/BH, more-vs-stronger classification, radius x scale
landscape, within-scale (never across-scale) radius trajectories, diversity/specialization
bootstrap, headroom sensitivity (secondary only), the two-scale terminology guard, and end-to-end
determinism against real, small, fully-shaped synthetic fixtures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import stage11_visual_thicket_scaling_interim_analysis as sia  # noqa: E402

from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

RADII = sia.RADII
CAPABILITIES = sia.CAPABILITIES
BASE_SCORES = {cap: 0.5 + 0.01 * i for i, cap in enumerate(CAPABILITIES)}


def _rec(
    *, scale_label: str, model_revision: str, mask_hash: str, capability: str, radius: float,
    direction_index: int, delta: float, seed_offset: int = 0,
) -> ExperimentResultRecord:
    base = BASE_SCORES[capability]
    pid = f"whole_model_{radius}_{direction_index}_{scale_label}"
    seed = direction_index + seed_offset
    return ExperimentResultRecord(
        experiment_id="stage11_whole_model_scaling", perturbation_id=pid, model_family="qwen2_5_vl",
        model_scale=scale_label, model_revision=model_revision, perturbation_mode="anatomical_relative_l2",
        anatomy_region="whole_model", radius=radius, sigma=None, seed=seed, parameter_mask_hash=mask_hash,
        capability=capability, dataset_role="map", subset_hash=f"sub_{capability}", base_score=base,
        perturbed_score=round(base + delta, 10), delta=delta, parser_failure_rate=0.0,
        per_example_result_path=None, per_example_result_hash=f"h_{pid}_{capability}",
        runtime_metadata={
            "direction_family_id": f"whole_model:{direction_index}", "direction_seed": seed,
            "direction_index": direction_index, "region": "whole_model",
            "radius_acceptance_mode": "strict", "quantization_limited": False,
            "requested_relative_l2": radius, "realized_relative_l2": radius, "relative_radius_error": 0.0,
        },
    )


def _delta_fn(scale_label: str, radius: float, direction_index: int, capability: str) -> float:
    """Deterministic synthetic delta with a real cross-scale difference baked in via
    `scale_factor` -- 7B gets a systematically larger positive response than 3B for
    visual_grounding (a genuine "expands at 7B" signal to test classification against), and a
    shrinking response for counting (a genuine "contracts at 7B" signal).
    """
    radius_rank = RADII.index(radius)
    scale_factor = 3.0 if scale_label == "7B" else 1.0
    if capability == "visual_grounding":
        if radius_rank == 0:
            # Straddles the 0.02 margin at BOTH scales without saturating either one (density_3B
            # ~0.19, density_7B ~0.625) so the 7B > 3B effect is large enough to survive BH-FDR
            # correction across the full 18-cell grid, not just a raw (uncorrected) p-value.
            jitter = 0.03 * (direction_index % 16) / 15.0
            return round((-0.005 + jitter) * scale_factor, 10)
        base_curve = [None, 0.008, -0.03][radius_rank]
        return round((base_curve - 0.002 * (direction_index % 8)) * scale_factor, 10)
    if capability == "counting":
        base_curve = [0.04, 0.01, -0.04][radius_rank]
        inverse_factor = 1.0 if scale_label == "3B" else 0.3
        return round((base_curve - 0.002 * (direction_index % 8)) * inverse_factor, 10)
    return round(-0.005 * (radius_rank + 1) - 0.0005 * direction_index, 10)


def _build_synthetic_scale_records(scale_label: str, model_revision: str, mask_hash: str, n_directions: int = 64, seed_offset: int = 0) -> List[ExperimentResultRecord]:
    records = []
    for radius in RADII:
        for direction_index in range(n_directions):
            for capability in CAPABILITIES:
                delta = _delta_fn(scale_label, radius, direction_index, capability)
                records.append(_rec(
                    scale_label=scale_label, model_revision=model_revision, mask_hash=mask_hash,
                    capability=capability, radius=radius, direction_index=direction_index, delta=delta,
                    seed_offset=seed_offset,
                ))
    return records


def _full_records_by_scale(n_directions: int = 64) -> Dict[str, List[ExperimentResultRecord]]:
    return {
        "3B": _build_synthetic_scale_records("3B", "rev3b", "mask3b", n_directions=n_directions, seed_offset=0),
        "7B": _build_synthetic_scale_records("7B", "rev7b", "mask7b", n_directions=n_directions, seed_offset=10_000),
    }


def _checkpoint_for(scale_label: str, model_revision: str, mask_hash: str, seed_bank_hash: str, n_directions: int) -> Dict:
    return {
        "experiment_id": "stage11_whole_model_scaling", "run_signature": f"stage11_{scale_label.lower()}_whole_model_test",
        "scale_label": scale_label, "track": "whole_model", "restoration_mode": "fixed_base",
        "perturbation_mode": "anatomical_relative_l2", "radius_realization_method": "fixed_direction_bf16_quantization_aware_v3",
        "multimodal_cache_policy": "full_encoder_reset_vllm011_verified_v2", "enable_prefix_caching": False,
        "model_revision": model_revision, "dataset_role": "map", "radii": list(RADII), "capabilities": list(CAPABILITIES),
        "n_directions_per_cell": n_directions, "d_map_n": sia.EXPECTED_D_MAP_N,
        "subset_hashes": {cap: f"sub_{cap}" for cap in CAPABILITIES},
        "whole_model_mask_hash": mask_hash, "direction_seed_bank_hash": seed_bank_hash,
        "expected_unique_perturbations": len(RADII) * n_directions, "expected_result_rows": len(RADII) * n_directions * len(CAPABILITIES),
    }


def _write_run_dir(root: Path, dirname: str, scale_label: str, model_revision: str, mask_hash: str, seed_bank_hash: str, records: List[ExperimentResultRecord], n_directions: int, run_complete: bool = True) -> Path:
    run_dir = root / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_for(scale_label, model_revision, mask_hash, seed_bank_hash, n_directions)
    manifest = dict(checkpoint)
    by_pid = {r.perturbation_id for r in records}
    manifest["actual_unique_perturbations"] = len(by_pid)
    manifest["actual_result_rows"] = len(records)
    manifest["run_complete"] = run_complete
    (run_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint, indent=2))
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (run_dir / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    return run_dir


@pytest.fixture()
def full_results_root(tmp_path):
    root = tmp_path / "stage11_whole_model_scaling"
    records = _full_records_by_scale(n_directions=64)
    _write_run_dir(root, "stage11_3b_whole_model_test", "3B", "rev3b", "mask3b", "sbh3b", records["3B"], n_directions=64)
    _write_run_dir(root, "stage11_7b_whole_model_test", "7B", "rev7b", "mask7b", "sbh7b", records["7B"], n_directions=64)
    return root


# =================================================================================================
# Section 1: authoritative discovery -- excludes smoke, refuses ambiguity
# =================================================================================================


def test_discover_finds_the_complete_run(full_results_root):
    run_dir = sia.discover_complete_whole_model_run("3B", full_results_root)
    assert run_dir.name == "stage11_3b_whole_model_test"


def test_discover_raises_when_no_results_root():
    with pytest.raises(sia.Stage11InterimDataNotFoundError):
        sia.discover_complete_whole_model_run("3B", Path("/nonexistent/does/not/exist"))


def test_discover_excludes_smoke_run(tmp_path):
    """A smoke run (d_map_n=5, n_directions_per_cell=1) must NEVER be picked up as authoritative,
    even though it lives in the same results root with a plausible directory name.
    """
    root = tmp_path / "stage11_whole_model_scaling"
    smoke_records = _build_synthetic_scale_records("3B", "rev3b", "mask3b", n_directions=1)
    smoke_dir = root / "stage11_3b_whole_model_smoke"
    smoke_dir.mkdir(parents=True)
    checkpoint = _checkpoint_for("3B", "rev3b", "mask3b", "sbh3b", n_directions=1)
    checkpoint["d_map_n"] = 5  # smoke D_map size
    manifest = dict(checkpoint)
    manifest["actual_unique_perturbations"] = len(RADII) * 1
    manifest["actual_result_rows"] = len(RADII) * 1 * len(CAPABILITIES)
    manifest["run_complete"] = True
    (smoke_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint))
    (smoke_dir / "run_manifest.json").write_text(json.dumps(manifest))
    with (smoke_dir / "results.jsonl").open("w") as f:
        for r in smoke_records[: len(RADII) * 1]:
            f.write(json.dumps(r.to_dict()) + "\n")

    with pytest.raises(sia.Stage11InterimDataNotFoundError):
        sia.discover_complete_whole_model_run("3B", root)


def test_discover_raises_on_ambiguous_multiple_complete_runs(full_results_root):
    records = _build_synthetic_scale_records("3B", "rev3b_dup", "mask3b_dup", n_directions=64, seed_offset=99_999)
    _write_run_dir(full_results_root, "stage11_3b_whole_model_test_duplicate", "3B", "rev3b_dup", "mask3b_dup", "sbh3b_dup", records, n_directions=64)
    with pytest.raises(sia.Stage11InterimAmbiguousRunError):
        sia.discover_complete_whole_model_run("3B", full_results_root)


def test_discover_raises_on_incomplete_run(tmp_path):
    root = tmp_path / "stage11_whole_model_scaling"
    records = _build_synthetic_scale_records("3B", "rev3b", "mask3b", n_directions=64)[:-1]  # missing one row
    _write_run_dir(root, "stage11_3b_whole_model_partial", "3B", "rev3b", "mask3b", "sbh3b", records, n_directions=64, run_complete=False)
    with pytest.raises(sia.Stage11InterimDataNotFoundError):
        sia.discover_complete_whole_model_run("3B", root)


# =================================================================================================
# Section 2: integrity gate -- 192/1152, subset hashes, radii, independent-direction invariants
# =================================================================================================


def test_per_scale_integrity_passes_on_well_formed_fixture():
    records_by_scale = _full_records_by_scale()
    checkpoint = {s: _checkpoint_for(s, f"rev{s.lower()}", f"mask{s.lower()}", f"sbh{s.lower()}", 64) for s in sia.SCALES}
    manifest = {}
    for s in sia.SCALES:
        m = dict(checkpoint[s])
        m["actual_unique_perturbations"] = sia.EXPECTED_UNIQUE_PERTURBATIONS
        m["actual_result_rows"] = sia.EXPECTED_ROWS
        m["run_complete"] = True
        manifest[s] = m
    report = sia.run_cross_scale_whole_model_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["all_ok"] is True
    sia.ensure_cross_scale_whole_model_integrity(report)  # must not raise


def test_integrity_192_1152_check_catches_missing_rows():
    records_by_scale = _full_records_by_scale()
    records_by_scale["3B"] = records_by_scale["3B"][:-6]  # drop one whole perturbation's rows
    checkpoint = {s: _checkpoint_for(s, f"rev{s.lower()}", f"mask{s.lower()}", f"sbh{s.lower()}", 64) for s in sia.SCALES}
    manifest = {s: {**checkpoint[s], "actual_unique_perturbations": sia.EXPECTED_UNIQUE_PERTURBATIONS, "actual_result_rows": sia.EXPECTED_ROWS, "run_complete": True} for s in sia.SCALES}
    report = sia.run_cross_scale_whole_model_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["per_scale"]["3B"]["all_checks_pass"] is False
    assert report["all_ok"] is False
    with pytest.raises(sia.Stage11InterimIntegrityError):
        sia.ensure_cross_scale_whole_model_integrity(report)


def test_integrity_requires_same_subset_hashes_across_scales():
    records_by_scale = _full_records_by_scale()
    checkpoint = {s: _checkpoint_for(s, f"rev{s.lower()}", f"mask{s.lower()}", f"sbh{s.lower()}", 64) for s in sia.SCALES}
    checkpoint["7B"]["subset_hashes"]["counting"] = "DIFFERENT_SUBSET_HASH"
    manifest = {s: {**checkpoint[s], "actual_unique_perturbations": sia.EXPECTED_UNIQUE_PERTURBATIONS, "actual_result_rows": sia.EXPECTED_ROWS, "run_complete": True} for s in sia.SCALES}
    report = sia.run_cross_scale_whole_model_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["cross_scale"]["same_d_map_subset_hashes"] is False
    assert report["all_ok"] is False


def test_integrity_requires_same_radii_across_scales():
    records_by_scale = _full_records_by_scale()
    checkpoint = {s: _checkpoint_for(s, f"rev{s.lower()}", f"mask{s.lower()}", f"sbh{s.lower()}", 64) for s in sia.SCALES}
    checkpoint["7B"]["radii"] = [RADII[0], RADII[1], 0.5]
    manifest = {s: {**checkpoint[s], "actual_unique_perturbations": sia.EXPECTED_UNIQUE_PERTURBATIONS, "actual_result_rows": sia.EXPECTED_ROWS, "run_complete": True} for s in sia.SCALES}
    report = sia.run_cross_scale_whole_model_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["cross_scale"]["same_radii"] is False


def test_integrity_requires_independent_direction_seed_bank_and_mask_hash():
    """The whole point of the cross-scale gate: 3B and 7B must have DIFFERENT model_revision,
    whole_model_mask_hash, and direction_seed_bank_hash (independent parameter spaces / seed
    namespaces) -- identical values across scales must FAIL the gate, never pass silently.
    """
    records_by_scale = _full_records_by_scale()
    checkpoint = {s: _checkpoint_for(s, "SAME_REVISION", "SAME_MASK", "SAME_SEED_BANK", 64) for s in sia.SCALES}
    manifest = {s: {**checkpoint[s], "actual_unique_perturbations": sia.EXPECTED_UNIQUE_PERTURBATIONS, "actual_result_rows": sia.EXPECTED_ROWS, "run_complete": True} for s in sia.SCALES}
    report = sia.run_cross_scale_whole_model_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["cross_scale"]["different_model_revision"] is False
    assert report["cross_scale"]["different_whole_model_mask_hash"] is False
    assert report["cross_scale"]["different_direction_seed_bank_hash"] is False
    assert report["all_ok"] is False


# =================================================================================================
# Section 3: baseline table
# =================================================================================================


def test_baseline_table_matches_manual_values_and_headroom():
    records_by_scale = _full_records_by_scale(n_directions=4)
    table = sia.compute_baseline_table(records_by_scale)
    for cap in CAPABILITIES:
        expected = BASE_SCORES[cap]
        assert table[cap]["baseline_3B"] == pytest.approx(expected)
        assert table[cap]["baseline_7B"] == pytest.approx(expected)
        assert table[cap]["headroom_3B"] == pytest.approx(1.0 - expected)
        assert table[cap]["absolute_baseline_difference_7B_minus_3B"] == pytest.approx(0.0)


def test_baseline_table_flags_non_canonical_baseline():
    records_by_scale = _full_records_by_scale(n_directions=4)
    bad = records_by_scale["3B"][0]
    records_by_scale["3B"][0] = ExperimentResultRecord(**{**bad.to_dict(), "base_score": bad.base_score + 0.5, "perturbed_score": bad.perturbed_score + 0.5})
    table = sia.compute_baseline_table(records_by_scale)
    cap = records_by_scale["3B"][0].capability
    assert table[cap]["canonical_baseline_independent_of_radius_direction_3B"] is False
    assert table[cap]["baseline_3B"] is None


# =================================================================================================
# Section 4: 36-cell statistics table
# =================================================================================================


def test_cell_statistics_covers_all_36_cells():
    records_by_scale = _full_records_by_scale(n_directions=8)
    cell_stats = sia.compute_cell_statistics(records_by_scale)
    assert len(cell_stats) == len(sia.SCALES) * len(CAPABILITIES) * len(RADII)
    for scale in sia.SCALES:
        for cap in CAPABILITIES:
            for radius in RADII:
                assert f"{scale}:{cap}:{radius}" in cell_stats


def test_cell_statistics_mean_matches_manual_computation():
    n = 8
    records_by_scale = _full_records_by_scale(n_directions=n)
    cell_stats = sia.compute_cell_statistics(records_by_scale)
    row = cell_stats[f"3B:visual_grounding:{RADII[0]}"]
    expected = [_delta_fn("3B", RADII[0], i, "visual_grounding") for i in range(n)]
    assert row["mean_delta"] == pytest.approx(sum(expected) / n)
    assert row["n"] == n


# =================================================================================================
# Section 5: solution-density curves -- monotonicity + common margin grid
# =================================================================================================


def test_common_margin_grid_contains_required_base_points():
    records_by_scale = _full_records_by_scale(n_directions=8)
    grid = sia.build_common_margin_grid(records_by_scale)
    for m in (0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10):
        assert m in grid


def test_common_margin_grid_extends_beyond_010_when_max_delta_requires_it():
    records_by_scale = _full_records_by_scale(n_directions=8)
    big = records_by_scale["7B"][0]
    records_by_scale["7B"][0] = ExperimentResultRecord(**{**big.to_dict(), "base_score": 0.2, "perturbed_score": 0.5, "delta": 0.3})
    grid = sia.build_common_margin_grid(records_by_scale)
    assert max(grid) >= 0.3


def test_common_margin_grid_is_identical_across_scales_by_construction():
    """The grid is a single object built from BOTH scales together -- there is no separate
    per-scale grid to accidentally diverge.
    """
    records_by_scale = _full_records_by_scale(n_directions=8)
    grid = sia.build_common_margin_grid(records_by_scale)
    curves = sia.compute_solution_density_curves(records_by_scale, grid)
    for scale in sia.SCALES:
        for cap in CAPABILITIES:
            for radius in RADII:
                assert curves["by_scale_capability_radius"][scale][cap][str(radius)]["margins"] == list(grid)


def test_solution_density_curve_is_monotonically_non_increasing_in_margin():
    records_by_scale = _full_records_by_scale(n_directions=8)
    grid = sia.build_common_margin_grid(records_by_scale)
    curves = sia.compute_solution_density_curves(records_by_scale, grid)
    sia.ensure_solution_density_curves_monotonic(curves)  # must not raise
    d = curves["by_scale_capability_radius"]["3B"]["visual_grounding"][str(RADII[0])]["density"]
    assert all(d[i] >= d[i + 1] - 1e-12 for i in range(len(d) - 1))


def test_monotonicity_check_raises_on_a_corrupted_curve():
    curves = {"by_scale_capability_radius": {"3B": {"cap": {"1.0": {"density": [0.1, 0.9, 0.05]}}}}}
    with pytest.raises(ValueError):
        sia.ensure_solution_density_curves_monotonic(curves)


# =================================================================================================
# Section 6: cross-scale solution-density differences + headline-margin tests
# =================================================================================================


def test_density_difference_safe_zero_handling():
    assert sia._safe_ratio(0.0, 0.0) == 1.0 or sia._safe_ratio(0.0, 0.0) is None or sia._safe_ratio(0.0, 0.0) == 0.0
    assert sia._safe_ratio(0.5, 0.0) == float("inf")


def test_headline_margin_tests_deterministic_across_reruns():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out1 = sia.compute_headline_margin_statistical_tests(records_by_scale)
    out2 = sia.compute_headline_margin_statistical_tests(records_by_scale)
    key = f"m={sia.USEFUL_MARGIN}"
    cell_key = f"visual_grounding:{RADII[0]}"
    assert out1[key][cell_key]["difference_95ci_bootstrap"] == out2[key][cell_key]["difference_95ci_bootstrap"]
    assert out1[key][cell_key]["permutation_p_value"] == out2[key][cell_key]["permutation_p_value"]


def test_headline_margin_tests_ci_brackets_the_point_difference():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = sia.compute_headline_margin_statistical_tests(records_by_scale)
    cell = out[f"m={sia.USEFUL_MARGIN}"][f"visual_grounding:{RADII[0]}"]
    lo, hi = cell["difference_95ci_bootstrap"]
    assert lo <= cell["difference_7B_minus_3B"] <= hi


def test_headline_margin_tests_permutation_p_value_in_unit_interval():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = sia.compute_headline_margin_statistical_tests(records_by_scale)
    for cell in out[f"m={sia.USEFUL_MARGIN}"].values():
        assert 0.0 <= cell["permutation_p_value"] <= 1.0


def test_headline_margin_tests_bh_correction_applied_separately_per_margin():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = sia.compute_headline_margin_statistical_tests(records_by_scale)
    n_cells = len(CAPABILITIES) * len(RADII)
    assert len(out[f"m={sia.USEFUL_MARGIN}"]) == n_cells
    assert len(out[f"m={sia.STRONG_MARGIN}"]) == n_cells
    for cell in out[f"m={sia.USEFUL_MARGIN}"].values():
        assert 0.0 <= cell["bh_q_value"] <= 1.0
        assert cell["verdict"] in ("significant_increase", "significant_decrease", "non_significant_trend")


def test_visual_grounding_shows_significant_density_increase_at_smallest_radius():
    """The synthetic fixture bakes a real, large 7B > 3B effect into visual_grounding at the
    smallest radius (scale_factor=1.6 on an already-positive delta) -- this must surface as a
    detected significant increase, not get lost in the machinery.
    """
    records_by_scale = _full_records_by_scale(n_directions=64)
    out = sia.compute_headline_margin_statistical_tests(records_by_scale)
    cell = out[f"m={sia.USEFUL_MARGIN}"][f"visual_grounding:{RADII[0]}"]
    assert cell["difference_7B_minus_3B"] > 0
    assert cell["verdict"] == "significant_increase"


# =================================================================================================
# Section 7: visual-macro solution density -- candidate-row-preserving bootstrap
# =================================================================================================


def test_macro_density_point_estimate_equals_flattened_mean():
    records_by_scale = _full_records_by_scale(n_directions=16)
    grid = sia.build_common_margin_grid(records_by_scale)
    macro = sia.compute_visual_macro_solution_density(records_by_scale, grid)
    _, _, matrix = sia._matrix_for_radius(records_by_scale["3B"], RADII[0])
    expected = float((matrix >= sia.USEFUL_MARGIN).mean())
    actual = macro["by_scale_radius"]["3B"][str(RADII[0])]["by_margin"][str(sia.USEFUL_MARGIN)]["macro_density"]
    assert actual == pytest.approx(expected)


def test_macro_density_bootstrap_preserves_six_capability_row_structure():
    """Row-preserving resampling must draw whole directions (all 6 capabilities together), never
    resample each capability column independently -- verified by checking the bootstrap
    distribution is generated from whole-row draws of the actual delta matrix.
    """
    rng_matrix = np.array([[0.1, -0.1, 0.1, -0.1, 0.1, -0.1]] * 32 + [[-0.1, 0.1, -0.1, 0.1, -0.1, 0.1]] * 32)
    dist = sia._macro_density_bootstrap_distribution(rng_matrix, margin=0.05, seed=1)
    # Every row is either all >=0.05 in 3 of 6 columns or the mirror -- so every possible
    # row-preserving resample's macro density is EXACTLY 0.5, regardless of which rows are drawn.
    assert np.allclose(dist, 0.5)


def test_macro_density_difference_ci_is_deterministic():
    records_by_scale = _full_records_by_scale(n_directions=16)
    grid = sia.build_common_margin_grid(records_by_scale)
    out1 = sia.compute_visual_macro_solution_density(records_by_scale, grid)
    out2 = sia.compute_visual_macro_solution_density(records_by_scale, grid)
    ci1 = out1["difference_7B_minus_3B"][str(RADII[0])]["by_margin"][str(sia.USEFUL_MARGIN)]["difference_95ci_bootstrap"]
    ci2 = out2["difference_7B_minus_3B"][str(RADII[0])]["by_margin"][str(sia.USEFUL_MARGIN)]["difference_95ci_bootstrap"]
    assert ci1 == ci2


# =================================================================================================
# Section 8: performance-density / Wasserstein / BH
# =================================================================================================


def test_wasserstein_1_equal_size_known_example():
    a = [0.0, 1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert sia.wasserstein_1_equal_size(a, b) == pytest.approx(1.0)


def test_wasserstein_1_equal_size_zero_for_identical_samples():
    a = [0.1, -0.2, 0.3, 0.0]
    assert sia.wasserstein_1_equal_size(a, a) == pytest.approx(0.0)


def test_wasserstein_1_equal_size_requires_equal_length():
    with pytest.raises(ValueError):
        sia.wasserstein_1_equal_size([0.0, 1.0], [0.0, 1.0, 2.0])


def test_performance_density_comparison_covers_18_cells_and_bh_correction():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = sia.compute_performance_density_comparison(records_by_scale)
    assert len(out) == len(CAPABILITIES) * len(RADII)
    for cell in out.values():
        assert 0.0 <= cell["wasserstein_1_permutation_p_value"] <= 1.0
        assert 0.0 <= cell["wasserstein_1_bh_q_value"] <= 1.0
        assert cell["shift_pattern"] in (
            "whole_distribution_and_tail_shift", "whole_distribution_shift", "sparse_tail_shift_only", "no_meaningful_shift",
        )


# =================================================================================================
# Section 9: more experts vs stronger experts
# =================================================================================================


def test_more_vs_stronger_classification_covers_18_cells():
    records_by_scale = _full_records_by_scale(n_directions=32)
    density_tests = sia.compute_headline_margin_statistical_tests(records_by_scale)
    strength = sia.compute_strength_contrasts(records_by_scale)
    result = sia.classify_more_vs_stronger(density_tests, strength)
    assert len(result["cells"]) == len(CAPABILITIES) * len(RADII)
    for cell in result["cells"].values():
        assert cell["classification"] in sia.MORE_VS_STRONGER_LABELS


def test_more_vs_stronger_labels_a_strong_expansion_correctly():
    records_by_scale = _full_records_by_scale(n_directions=64)
    density_tests = sia.compute_headline_margin_statistical_tests(records_by_scale)
    strength = sia.compute_strength_contrasts(records_by_scale)
    result = sia.classify_more_vs_stronger(density_tests, strength)
    label = result["cells"][f"visual_grounding:{RADII[0]}"]["classification"]
    assert label in ("more_and_stronger", "more_not_stronger", "stronger_not_more")  # never "decreases"/"neither_clear" given the baked-in positive effect


def test_more_vs_stronger_never_decided_from_point_estimates_without_significance():
    """A cell with a tiny, statistically-indistinguishable difference must NOT be labeled
    more_and_stronger just because the point estimate happens to differ slightly.
    """
    records_by_scale = _full_records_by_scale(n_directions=64)
    # relational_reasoning has an identical tiny non-positive trend at both scales by construction.
    density_tests = sia.compute_headline_margin_statistical_tests(records_by_scale)
    strength = sia.compute_strength_contrasts(records_by_scale)
    result = sia.classify_more_vs_stronger(density_tests, strength)
    label = result["cells"][f"relational_reasoning:{RADII[2]}"]["classification"]
    assert label in ("neither_clear", "decreases")


# =================================================================================================
# Section 10: radius x scale landscape
# =================================================================================================


def test_radius_scale_landscape_covers_all_capabilities():
    records_by_scale = _full_records_by_scale(n_directions=16)
    cell_stats = sia.compute_cell_statistics(records_by_scale)
    landscape = sia.compute_radius_scale_landscape(cell_stats)
    assert set(landscape.keys()) == set(CAPABILITIES)
    for row in landscape.values():
        assert row["question_A_peak_radius_change"] in ("peak_radius_stable", "peak_radius_reorganizes")
        assert row["question_C_broader_or_narrower_useful_neighborhood"] in sia.RADIUS_SCALE_LABELS


def test_radius_scale_landscape_detects_peak_reorganization():
    """visual_grounding's synthetic curve peaks at the small radius for BOTH scales (same shape,
    just scaled), so this should read as peak_radius_stable -- a real, checkable structural fact.
    """
    records_by_scale = _full_records_by_scale(n_directions=32)
    cell_stats = sia.compute_cell_statistics(records_by_scale)
    landscape = sia.compute_radius_scale_landscape(cell_stats)
    assert landscape["visual_grounding"]["peak_radius_3B"] == RADII[0]
    assert landscape["visual_grounding"]["peak_radius_7B"] == RADII[0]
    assert landscape["visual_grounding"]["question_A_peak_radius_change"] == "peak_radius_stable"


# =================================================================================================
# Section 11: within-scale radius trajectories -- NEVER paired across scales
# =================================================================================================


def test_radius_trajectories_by_scale_computed_independently_per_scale():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = sia.compute_radius_trajectories_by_scale(records_by_scale)
    assert "3B" in out and "7B" in out
    assert out["3B"]["n_complete_trajectories"] == len(CAPABILITIES) * 16
    assert out["7B"]["n_complete_trajectories"] == len(CAPABILITIES) * 16


def test_radius_trajectories_summary_comparison_has_no_cross_scale_direction_pairing():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = sia.compute_radius_trajectories_by_scale(records_by_scale)
    assert "no_cross_scale_direction_pairing" not in out  # the note lives under cross_scale_pairing_note, not a fabricated pairing structure
    assert "paired" not in out["cross_scale_pairing_note"].lower() or "not" in out["cross_scale_pairing_note"].lower()
    for field, row in out["summary_comparison_7B_minus_3B"].items():
        assert set(row.keys()) == {"3B", "7B", "difference_7B_minus_3B"}


# =================================================================================================
# Section 12: specialization / diversity -- bootstrap preserves the six-capability candidate vector
# =================================================================================================


def test_diversity_scale_trend_covers_all_three_radii():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = sia.compute_diversity_scale_trend(records_by_scale)
    assert set(out.keys()) == {str(r) for r in RADII}
    for row in out.values():
        assert row["trend"] in ("increases_3B_to_7B", "decreases_3B_to_7B", "no_clear_change")


def test_discordance_bootstrap_distribution_uses_row_resampling_only():
    matrix = np.array([[0.1, 0.2, -0.1, 0.3, 0.0, 0.1]] * 20 + [[-0.2, -0.1, 0.2, -0.3, 0.1, -0.2]] * 20)
    dist = sia._discordance_bootstrap_distribution(matrix, seed=7, n_bootstrap=200)
    assert dist.shape == (200,)
    assert np.all(np.isfinite(dist))


def test_diversity_scale_trend_deterministic_across_reruns():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out1 = sia.compute_diversity_scale_trend(records_by_scale)
    out2 = sia.compute_diversity_scale_trend(records_by_scale)
    assert out1[str(RADII[0])]["difference_95ci_bootstrap"] == out2[str(RADII[0])]["difference_95ci_bootstrap"]


# =================================================================================================
# Section 13/15: raw Delta remains primary; headroom normalization is secondary only
# =================================================================================================


def test_headroom_sensitivity_reports_raw_direction_from_unnormalized_mass():
    records_by_scale = _full_records_by_scale(n_directions=16)
    cell_stats = sia.compute_cell_statistics(records_by_scale)
    baseline_table = sia.compute_baseline_table(records_by_scale)
    out = sia.compute_headroom_sensitivity(records_by_scale, baseline_table, cell_stats)
    row = out[f"visual_grounding:{RADII[0]}"]
    expected_raw_diff = cell_stats[f"7B:visual_grounding:{RADII[0]}"]["positive_thicket_mass"] - cell_stats[f"3B:visual_grounding:{RADII[0]}"]["positive_thicket_mass"]
    assert row["raw_positive_mass_diff_7B_minus_3B"] == pytest.approx(expected_raw_diff)
    assert row["raw_conclusion_direction"] in ("increase", "decrease", "flat")


def test_headroom_sensitivity_marks_not_applicable_when_no_headroom():
    records_by_scale = _full_records_by_scale(n_directions=4)
    cap = CAPABILITIES[0]
    for r in records_by_scale["3B"]:
        if r.capability == cap:
            records_by_scale["3B"][records_by_scale["3B"].index(r)] = ExperimentResultRecord(**{**r.to_dict(), "base_score": 1.0, "perturbed_score": 1.0 + r.delta, "delta": r.delta})
    baseline_table = sia.compute_baseline_table(records_by_scale)
    cell_stats = sia.compute_cell_statistics(records_by_scale)
    out = sia.compute_headroom_sensitivity(records_by_scale, baseline_table, cell_stats)
    row = out[f"{cap}:{RADII[0]}"]
    assert row["normalized_by_scale"]["3B"]["applicable"] is False
    assert row["headroom_sensitivity_verdict"] == "not_applicable"


# =================================================================================================
# Section 20: two-scale terminology guard
# =================================================================================================


def test_terminology_guard_forbids_scaling_law_language_with_two_scales():
    guard = sia.TERMINOLOGY_GUARD
    assert guard["n_scales"] == 2
    assert guard["may_use_scaling_relationship_language"] is False
    assert "scaling law" in guard["disallowed_as_empirical_conclusion"]
    assert set(guard["allowed_terms"]) == {"scale trend", "cross-scale comparison"}


def test_interim_claim_gate_never_claims_scaling_law_established():
    records_by_scale = _full_records_by_scale(n_directions=16)
    cell_stats = sia.compute_cell_statistics(records_by_scale)
    density_tests = sia.compute_headline_margin_statistical_tests(records_by_scale)
    strength = sia.compute_strength_contrasts(records_by_scale)
    more_vs_stronger = sia.classify_more_vs_stronger(density_tests, strength)
    landscape = sia.compute_radius_scale_landscape(cell_stats)
    diversity = sia.compute_diversity_scale_trend(records_by_scale)
    gate = sia.evaluate_interim_claim_gate(cell_stats, density_tests, more_vs_stronger, landscape, diversity)
    claim_keys = (
        "S1_nearby_specialists_exist_both_scales", "S2_solution_density_changes_systematically",
        "S3_specialist_strength_changes", "S4_useful_radius_behavior_changes", "S5_specialization_diversity_changes",
    )
    for v in claim_keys:
        assert gate[v] in sia.CLAIM_VERDICTS
        assert "scaling law" not in gate[v]  # the verdict values themselves never assert a scaling law
    assert "never a valid conclusion" in gate["note"].lower()  # the guard text explicitly forbids it, as a negation


# =================================================================================================
# Section 21: end-to-end determinism against a real fully-shaped fixture on disk
# =================================================================================================


def test_main_runs_end_to_end_and_is_deterministic(full_results_root, tmp_path):
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    rc1 = sia.main(["--results-root", str(full_results_root), "--output-dir", str(out1)])
    rc2 = sia.main(["--results-root", str(full_results_root), "--output-dir", str(out2)])
    assert rc1 == 0 and rc2 == 0

    for name in ("baseline_table.json", "cell_statistics_3b_7b.json", "solution_density_curves.json", "interim_claim_gate.json"):
        assert (out1 / name).read_text() == (out2 / name).read_text()

    integrity = json.loads((out1 / "integrity_report.json").read_text())
    assert integrity["all_ok"] is True

    claim_gate = json.loads((out1 / "interim_claim_gate.json").read_text())
    for v in ("S1_nearby_specialists_exist_both_scales", "S2_solution_density_changes_systematically",
              "S3_specialist_strength_changes", "S4_useful_radius_behavior_changes", "S5_specialization_diversity_changes"):
        assert "scaling law" not in claim_gate[v]  # the verdict values themselves never assert a scaling law

    for name in (
        "baseline_table.json", "cell_statistics_3b_7b.json", "cell_statistics_3b_7b.csv",
        "solution_density_curves.json", "solution_density_curves.csv", "solution_density_scale_differences.json",
        "visual_macro_solution_density.json", "performance_density_comparison.json", "performance_density_comparison.csv",
        "more_vs_stronger_classification.json", "radius_scale_landscape.json", "radius_trajectories_by_scale.json",
        "diversity_scale_trend.json", "headroom_sensitivity.json", "capability_scale_summaries.json",
        "statistical_tests.json", "interim_claim_gate.json", "stage11_interim_3b_7b_summary.md",
    ):
        assert (out1 / name).exists(), f"missing output file {name}"
    for name in (
        "fig_s1a_solution_density_curves.csv", "fig_s1b_visual_macro_scale_trend.csv", "fig_s2_performance_density.csv",
        "fig_s3_radius_scale_matrix.csv", "fig_s4_diversity_scale_trend.csv", "fig_s5_capability_scale_response.csv",
    ):
        assert (out1 / "figure_schemas" / name).exists(), f"missing figure schema {name}"


def test_main_returns_zero_and_does_not_fabricate_when_no_data(tmp_path):
    rc = sia.main(["--results-root", str(tmp_path / "nonexistent"), "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    assert not (tmp_path / "out").exists()
