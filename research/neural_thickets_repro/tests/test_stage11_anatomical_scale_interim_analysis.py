"""Tests for analysis/stage11_anatomical_scale_interim_analysis.py -- the interim 3B-vs-7B
ANATOMY-RESOLVED scale analysis. Covers: authoritative discovery (excludes smoke, per track),
per-scale and cross-scale integrity (576/3456, same N50 subset hashes, same radii/capabilities,
semantic region alignment, independent model revision/seed-bank), the 108-cell table, solution-
density monotonicity, region ranking / anatomical-preference-transition logic, pairwise regional
contrasts + difference-in-differences, region-macro row-preserving bootstrap, specialization
bootstrap structure, density-vs-strength classification, headroom-secondary discipline, the
whole-model/anatomy interpretation's non-additive-causality discipline, the two-scale terminology
guard, and end-to-end determinism against a real, small, fully-shaped synthetic fixture.
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

import stage11_anatomical_scale_interim_analysis as saa  # noqa: E402

from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

REGIONS = saa.REGIONS
RADII = saa.RADII
CAPABILITIES = saa.CAPABILITIES
BASE_SCORES = {cap: 0.5 + 0.01 * i for i, cap in enumerate(CAPABILITIES)}


def _rec(*, scale_label: str, model_revision: str, mask_hash: str, region: str, capability: str, radius: float, direction_index: int, delta: float, seed_offset: int = 0) -> ExperimentResultRecord:
    base = BASE_SCORES[capability]
    pid = f"{region}_{radius}_{direction_index}_{scale_label}"
    return ExperimentResultRecord(
        experiment_id="stage11_anatomy", perturbation_id=pid, model_family="qwen2_5_vl", model_scale=scale_label,
        model_revision=model_revision, perturbation_mode="anatomical_relative_l2", anatomy_region=region, radius=radius,
        sigma=None, seed=direction_index + seed_offset, parameter_mask_hash=mask_hash, capability=capability, dataset_role="map",
        subset_hash=f"sub_{capability}", base_score=base, perturbed_score=round(base + delta, 10), delta=delta,
        parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash=f"h_{pid}_{capability}",
        runtime_metadata={"direction_family_id": f"{region}:{direction_index}", "direction_seed": direction_index + seed_offset, "direction_index": direction_index, "region": region},
    )


def _delta_fn(scale_label: str, region: str, radius: float, direction_index: int, capability: str) -> float:
    """Bakes a real, checkable cross-scale + cross-region signal: vision x visual_grounding gets a
    strong, radius-dependent, scale-amplified positive response (7B > 3B, straddling the 0.02
    margin so density differences aren't ceiling/floor-saturated); everything else gets a small,
    near-identical-across-scale response (a genuine null for testing conservative "diffuse" logic).
    """
    rr = RADII.index(radius)
    scale_factor = 3.0 if scale_label == "7B" else 1.0
    if region == "vision" and capability == "visual_grounding":
        jitter = 0.03 * (direction_index % 16) / 15.0
        return round((-0.005 + jitter) * scale_factor, 10)
    return round(-0.003 * (rr + 1) - 0.0003 * direction_index, 10)


def _build_scale_records(scale_label: str, model_revision: str, mask_hashes: Dict[str, str], n_directions: int = 64, seed_offset: int = 0) -> List[ExperimentResultRecord]:
    records = []
    for region in REGIONS:
        for radius in RADII:
            for direction_index in range(n_directions):
                for capability in CAPABILITIES:
                    delta = _delta_fn(scale_label, region, radius, direction_index, capability)
                    records.append(_rec(scale_label=scale_label, model_revision=model_revision, mask_hash=mask_hashes[region], region=region, capability=capability, radius=radius, direction_index=direction_index, delta=delta, seed_offset=seed_offset))
    return records


def _full_records_by_scale(n_directions: int = 64) -> Dict[str, List[ExperimentResultRecord]]:
    mask3 = {"vision": "maskv", "multimodal_connector_or_merger": "maskc", "language": "maskl3b"}
    mask7 = {"vision": "maskv", "multimodal_connector_or_merger": "maskc", "language": "maskl7b"}  # vision/connector identical across scale (architecturally shared), language differs
    return {
        "3B": _build_scale_records("3B", "rev3b", mask3, n_directions=n_directions, seed_offset=0),
        "7B": _build_scale_records("7B", "rev7b", mask7, n_directions=n_directions, seed_offset=10_000),
    }


def _checkpoint_for(model_revision: str, mask_hashes: Dict[str, str], seed_bank_hash: str, n_directions: int) -> Dict:
    return {
        "regions": list(REGIONS), "radii": list(RADII), "capabilities": list(CAPABILITIES),
        "n_directions_per_cell": n_directions, "d_map_n": saa.EXPECTED_D_MAP_N, "subset_hashes": {c: f"sub_{c}" for c in CAPABILITIES},
        "region_mask_hashes": mask_hashes, "direction_seed_bank_hash": seed_bank_hash, "model_revision": model_revision,
        "perturbation_mode": "anatomical_relative_l2", "radius_realization_method": "fixed_direction_bf16_quantization_aware_v3",
        "restoration_mode": "fixed_base", "multimodal_cache_policy": "full_encoder_reset_vllm011_verified_v2", "enable_prefix_caching": False,
        "expected_unique_perturbations": len(REGIONS) * len(RADII) * n_directions, "expected_result_rows": len(REGIONS) * len(RADII) * n_directions * len(CAPABILITIES),
    }


def _write_run_dir(root: Path, dirname: str, model_revision: str, mask_hashes: Dict[str, str], seed_bank_hash: str, records: List[ExperimentResultRecord], n_directions: int, run_complete: bool = True) -> Path:
    run_dir = root / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_for(model_revision, mask_hashes, seed_bank_hash, n_directions)
    manifest = dict(checkpoint)
    manifest["actual_unique_perturbations"] = len({r.perturbation_id for r in records})
    manifest["actual_result_rows"] = len(records)
    manifest["run_complete"] = run_complete
    (run_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint))
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    with (run_dir / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    baseline_scores = {"capabilities": {c: {"score": BASE_SCORES[c]} for c in CAPABILITIES}}
    (run_dir / "baseline_scores.json").write_text(json.dumps(baseline_scores))
    return run_dir


@pytest.fixture()
def full_roots(tmp_path):
    records = _full_records_by_scale(n_directions=64)
    stage8_root = tmp_path / "stage8_coarse_anatomical_atlas"
    stage11_root = tmp_path / "stage11_coarse_anatomical_atlas_7b"
    mask3 = {"vision": "maskv", "multimodal_connector_or_merger": "maskc", "language": "maskl3b"}
    mask7 = {"vision": "maskv", "multimodal_connector_or_merger": "maskc", "language": "maskl7b"}
    _write_run_dir(stage8_root, "stage8_coarse_anatomical_atlas_3b_v2_batched10", "rev3b", mask3, "sbh3b", records["3B"], n_directions=64)
    _write_run_dir(stage11_root, "stage11_coarse_anatomical_atlas_7b_v1", "rev7b", mask7, "sbh7b", records["7B"], n_directions=64)
    return stage8_root, stage11_root


# =================================================================================================
# Section 1: authoritative discovery -- excludes smoke, refuses ambiguity
# =================================================================================================


def test_discover_finds_the_complete_3b_run(full_roots):
    stage8_root, _ = full_roots
    run_dir = saa.discover_complete_anatomy_run(stage8_root)
    assert run_dir.name == "stage8_coarse_anatomical_atlas_3b_v2_batched10"


def test_discover_finds_the_complete_7b_run(full_roots):
    _, stage11_root = full_roots
    run_dir = saa.discover_complete_anatomy_run(stage11_root)
    assert run_dir.name == "stage11_coarse_anatomical_atlas_7b_v1"


def test_discover_raises_when_no_results_root():
    with pytest.raises(saa.Stage11AnatomyInterimDataNotFoundError):
        saa.discover_complete_anatomy_run(Path("/nonexistent/does/not/exist"))


def test_discover_excludes_smoke_run(tmp_path):
    """A smoke run (d_map_n=5, n_directions_per_cell=1, 9/54 totals) must never be picked up."""
    root = tmp_path / "stage11_coarse_anatomical_atlas_7b"
    smoke_records = _build_scale_records("7B", "rev7b", {"vision": "mv", "multimodal_connector_or_merger": "mc", "language": "ml"}, n_directions=1)
    smoke_dir = root / "stage11_smoke_run"
    smoke_dir.mkdir(parents=True)
    checkpoint = _checkpoint_for("rev7b", {"vision": "mv", "multimodal_connector_or_merger": "mc", "language": "ml"}, "sbh", n_directions=1)
    checkpoint["d_map_n"] = 5
    manifest = dict(checkpoint)
    manifest["actual_unique_perturbations"] = len(REGIONS) * len(RADII) * 1
    manifest["actual_result_rows"] = len(REGIONS) * len(RADII) * 1 * len(CAPABILITIES)
    manifest["run_complete"] = True
    (smoke_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint))
    (smoke_dir / "run_manifest.json").write_text(json.dumps(manifest))
    with (smoke_dir / "results.jsonl").open("w") as f:
        for r in smoke_records:
            f.write(json.dumps(r.to_dict()) + "\n")
    with pytest.raises(saa.Stage11AnatomyInterimDataNotFoundError):
        saa.discover_complete_anatomy_run(root)


def test_discover_raises_on_ambiguous_multiple_complete_runs(full_roots):
    _, stage11_root = full_roots
    records = _build_scale_records("7B", "rev7b_dup", {"vision": "mv2", "multimodal_connector_or_merger": "mc2", "language": "ml2"}, n_directions=64, seed_offset=99_999)
    _write_run_dir(stage11_root, "stage11_coarse_anatomical_atlas_7b_v1_duplicate", "rev7b_dup", {"vision": "mv2", "multimodal_connector_or_merger": "mc2", "language": "ml2"}, "sbh_dup", records, n_directions=64)
    with pytest.raises(saa.Stage11AnatomyInterimAmbiguousRunError):
        saa.discover_complete_anatomy_run(stage11_root)


def test_discover_raises_on_incomplete_run(tmp_path):
    root = tmp_path / "stage8_coarse_anatomical_atlas"
    records = _build_scale_records("3B", "rev3b", {"vision": "mv", "multimodal_connector_or_merger": "mc", "language": "ml"}, n_directions=64)[:-1]
    _write_run_dir(root, "stage8_partial", "rev3b", {"vision": "mv", "multimodal_connector_or_merger": "mc", "language": "ml"}, "sbh", records, n_directions=64, run_complete=False)
    with pytest.raises(saa.Stage11AnatomyInterimDataNotFoundError):
        saa.discover_complete_anatomy_run(root)


# =================================================================================================
# Section 2: integrity gate -- 576/3456, subset hashes, radii, region alignment, independence
# =================================================================================================


def _full_checkpoints_and_manifests():
    mask3 = {"vision": "maskv", "multimodal_connector_or_merger": "maskc", "language": "maskl3b"}
    mask7 = {"vision": "maskv", "multimodal_connector_or_merger": "maskc", "language": "maskl7b"}
    checkpoint = {"3B": _checkpoint_for("rev3b", mask3, "sbh3b", 64), "7B": _checkpoint_for("rev7b", mask7, "sbh7b", 64)}
    manifest = {}
    for s in saa.SCALES:
        m = dict(checkpoint[s])
        m["actual_unique_perturbations"] = saa.EXPECTED_UNIQUE_PERTURBATIONS
        m["actual_result_rows"] = saa.EXPECTED_ROWS
        m["run_complete"] = True
        manifest[s] = m
    return checkpoint, manifest


def test_per_scale_integrity_passes_on_well_formed_fixture():
    records_by_scale = _full_records_by_scale()
    checkpoint, manifest = _full_checkpoints_and_manifests()
    report = saa.run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["all_ok"] is True
    saa.ensure_cross_scale_anatomy_integrity(report)  # must not raise


def test_integrity_576_3456_check_catches_missing_rows():
    records_by_scale = _full_records_by_scale()
    records_by_scale["3B"] = records_by_scale["3B"][:-6]
    checkpoint, manifest = _full_checkpoints_and_manifests()
    report = saa.run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["per_scale"]["3B"]["all_checks_pass"] is False
    with pytest.raises(saa.Stage11AnatomyInterimIntegrityError):
        saa.ensure_cross_scale_anatomy_integrity(report)


def test_integrity_requires_same_subset_hashes_across_scales():
    records_by_scale = _full_records_by_scale()
    checkpoint, manifest = _full_checkpoints_and_manifests()
    checkpoint["7B"]["subset_hashes"]["counting"] = "DIFFERENT"
    report = saa.run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["cross_scale"]["same_d_map_subset_hashes"] is False
    assert report["all_ok"] is False


def test_integrity_requires_same_radii_and_semantic_region_partition():
    records_by_scale = _full_records_by_scale()
    checkpoint, manifest = _full_checkpoints_and_manifests()
    checkpoint["7B"]["radii"] = [RADII[0], RADII[1], 0.5]
    report = saa.run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["cross_scale"]["same_radii"] is False

    checkpoint2, manifest2 = _full_checkpoints_and_manifests()
    checkpoint2["7B"]["regions"] = ["vision", "multimodal_connector_or_merger", "not_language"]
    report2 = saa.run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint2, manifest2)
    assert report2["cross_scale"]["same_semantic_region_partition"] is False


def test_integrity_requires_independent_model_revision_and_seed_bank():
    records_by_scale = _full_records_by_scale()
    checkpoint, manifest = _full_checkpoints_and_manifests()
    checkpoint["7B"]["model_revision"] = checkpoint["3B"]["model_revision"]
    checkpoint["7B"]["direction_seed_bank_hash"] = checkpoint["3B"]["direction_seed_bank_hash"]
    report = saa.run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["cross_scale"]["different_model_revision"] is False
    assert report["cross_scale"]["different_direction_seed_bank_hash"] is False
    assert report["all_ok"] is False


def test_integrity_region_mask_hash_identity_is_informational_not_a_hard_fail():
    """vision/connector legitimately share an identical mask hash across scales in real data
    (shared architecture) -- the gate must NOT hard-fail on that, only report it.
    """
    records_by_scale = _full_records_by_scale()
    checkpoint, manifest = _full_checkpoints_and_manifests()
    report = saa.run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint, manifest)
    assert report["cross_scale"]["region_mask_hash_comparison"]["vision"]["identical"] is True
    assert report["cross_scale"]["region_mask_hash_comparison"]["language"]["identical"] is False
    assert report["all_ok"] is True  # not blocked by the vision/connector identity


# =================================================================================================
# Section 4: 108-cell table
# =================================================================================================


def test_cell_statistics_covers_all_108_cells():
    records_by_scale = _full_records_by_scale(n_directions=8)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    assert len(cell_stats) == len(saa.SCALES) * len(REGIONS) * len(RADII) * len(CAPABILITIES)
    for scale in saa.SCALES:
        for region in REGIONS:
            for radius in RADII:
                for cap in CAPABILITIES:
                    assert f"{scale}:{cap}:{region}:{radius}" in cell_stats


def test_cell_statistics_mean_matches_manual_computation():
    n = 8
    records_by_scale = _full_records_by_scale(n_directions=n)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    row = cell_stats[f"3B:visual_grounding:vision:{RADII[0]}"]
    expected = [_delta_fn("3B", "vision", RADII[0], i, "visual_grounding") for i in range(n)]
    assert row["mean_delta"] == pytest.approx(sum(expected) / n)
    assert row["n"] == n


# =================================================================================================
# Section 5: solution-density curves -- monotonicity, common fixed grid
# =================================================================================================


def test_solution_density_curves_monotonic_and_common_grid():
    records_by_scale = _full_records_by_scale(n_directions=8)
    curves = saa.compute_anatomy_solution_density_curves(records_by_scale)
    saa.ensure_anatomy_curves_monotonic(curves)  # must not raise
    for scale in saa.SCALES:
        for cap in CAPABILITIES:
            for region in REGIONS:
                for radius in RADII:
                    row = curves["by_scale"][scale][cap][region][str(radius)]
                    assert row["margin_grid"] == curves["margin_grid"]  # same fixed grid, every scale/cell


def test_monotonicity_check_raises_on_a_corrupted_curve():
    curves = {"by_scale": {"3B": {"cap": {"vision": {"1.0": {"delta_ge_m": [0.1, 0.9, 0.05]}}}}}}
    with pytest.raises(ValueError):
        saa.ensure_anatomy_curves_monotonic(curves)


# =================================================================================================
# Section 6: cross-scale anatomical differences -- 54 cells, BH-FDR, permutation
# =================================================================================================


def test_headline_density_tests_cover_54_cells_per_margin():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    n_cells = len(CAPABILITIES) * len(REGIONS) * len(RADII)
    assert len(out[f"m={saa.USEFUL_MARGIN}"]) == n_cells == 54
    assert len(out[f"m={saa.STRONG_MARGIN}"]) == n_cells == 54
    for cell in out[f"m={saa.USEFUL_MARGIN}"].values():
        assert 0.0 <= cell["permutation_p_value"] <= 1.0
        assert 0.0 <= cell["bh_q_value"] <= 1.0
        assert cell["verdict"] in ("significant_increase", "significant_decrease", "non_significant_trend")


def test_vision_visual_grounding_shows_significant_density_increase_at_smallest_radius():
    records_by_scale = _full_records_by_scale(n_directions=64)
    out = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    cell = out[f"m={saa.USEFUL_MARGIN}"][f"visual_grounding:vision:{RADII[0]}"]
    assert cell["difference_7B_minus_3B"] > 0
    assert cell["verdict"] == "significant_increase"


def test_headline_density_tests_deterministic_across_reruns():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out1 = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    out2 = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    key = f"visual_grounding:vision:{RADII[0]}"
    assert out1[f"m={saa.USEFUL_MARGIN}"][key]["difference_95ci_bootstrap"] == out2[f"m={saa.USEFUL_MARGIN}"][key]["difference_95ci_bootstrap"]


def test_point_differences_cover_54_cells():
    records_by_scale = _full_records_by_scale(n_directions=8)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    point_diffs = saa.compute_cross_scale_anatomy_point_differences(cell_stats)
    assert len(point_diffs) == 54


# =================================================================================================
# Section 7: region ranking / "where do experts live"
# =================================================================================================


def test_region_ranking_covers_every_scale_capability_radius():
    records_by_scale = _full_records_by_scale(n_directions=8)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    rankings = saa.rank_regions_by_capability_radius_scale(cell_stats)
    assert len(rankings) == len(saa.SCALES) * len(CAPABILITIES) * len(RADII)
    for row in rankings.values():
        assert set(row["ranked_regions"]) == set(REGIONS)


def test_vision_ranked_top_for_visual_grounding_at_small_radius_both_scales():
    records_by_scale = _full_records_by_scale(n_directions=64)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    rankings = saa.rank_regions_by_capability_radius_scale(cell_stats)
    assert rankings[f"3B:visual_grounding:{RADII[0]}"]["ranked_regions"][0] == "vision"
    assert rankings[f"7B:visual_grounding:{RADII[0]}"]["ranked_regions"][0] == "vision"


def test_anatomical_preference_transitions_requires_statistical_support_not_point_estimates():
    """Capabilities with a genuine null signal (near-identical tiny deltas across all 3 regions)
    must be classified diffuse_no_clear_preference, never asserted as having a dominant region
    just because one region's point estimate happens to be marginally higher.
    """
    records_by_scale = _full_records_by_scale(n_directions=64)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    rankings = saa.rank_regions_by_capability_radius_scale(cell_stats)
    transitions = saa.classify_anatomical_preference_transitions(records_by_scale, cell_stats, rankings)
    assert transitions[f"counting:{RADII[0]}"]["classification"] == "diffuse_no_clear_preference"


def test_vision_visual_grounding_preference_is_stable_across_scale():
    records_by_scale = _full_records_by_scale(n_directions=64)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    rankings = saa.rank_regions_by_capability_radius_scale(cell_stats)
    transitions = saa.classify_anatomical_preference_transitions(records_by_scale, cell_stats, rankings)
    row = transitions[f"visual_grounding:{RADII[0]}"]
    assert row["dominant_region_3B"] == row["dominant_region_7B"] == "vision"
    assert row["classification"] == "anatomical_preference_stable"


# =================================================================================================
# Section 8: pairwise regional contrasts (reused from s8a) + difference-in-differences
# =================================================================================================


def test_anatomical_contrasts_by_scale_covers_both_scales_and_three_pairs():
    records_by_scale = _full_records_by_scale(n_directions=16)
    contrasts = saa.compute_anatomical_contrasts_by_scale(records_by_scale)
    assert set(contrasts.keys()) == set(saa.SCALES)
    cell = contrasts["3B"]["visual_grounding"][str(RADII[0])]
    assert set(cell.keys()) == {"vision_vs_multimodal_connector_or_merger", "vision_vs_language", "multimodal_connector_or_merger_vs_language"}


def test_difference_in_differences_covers_every_capability_radius_pair():
    records_by_scale = _full_records_by_scale(n_directions=16)
    did = saa.compute_difference_in_differences(records_by_scale)
    for cap in CAPABILITIES:
        for radius in RADII:
            pairs = did[cap][str(radius)]
            assert set(pairs.keys()) == {"vision_vs_language", "vision_vs_multimodal_connector_or_merger", "multimodal_connector_or_merger_vs_language"}
            for cell in pairs.values():
                assert set(cell["metrics"].keys()) == {"mean_delta", "density_ge_0.02", "positive_thicket_mass"}


def test_difference_in_differences_ci_brackets_the_point_estimate():
    records_by_scale = _full_records_by_scale(n_directions=32)
    did = saa.compute_difference_in_differences(records_by_scale)
    cell = did["visual_grounding"][str(RADII[0])]["vision_vs_language"]["metrics"]["density_ge_0.02"]
    lo, hi = cell["difference_in_differences_95ci_bootstrap"]
    assert lo <= cell["difference_in_differences"] <= hi


def test_difference_in_differences_detects_the_baked_in_vision_language_asymmetry():
    """vision x visual_grounding grows strongly with scale while language x visual_grounding does
    not -- the DiD for that pair/capability/radius must be positive and CI-supported.
    """
    records_by_scale = _full_records_by_scale(n_directions=64)
    did = saa.compute_difference_in_differences(records_by_scale)
    cell = did["visual_grounding"][str(RADII[0])]["vision_vs_language"]["metrics"]["density_ge_0.02"]
    assert cell["difference_in_differences"] > 0
    assert cell["ci_excludes_zero"] is True


# =================================================================================================
# Section 9: anatomical scale-response map
# =================================================================================================


def test_scale_response_map_covers_all_radii_capabilities_regions():
    records_by_scale = _full_records_by_scale(n_directions=8)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    m = saa.compute_anatomical_scale_response_map(cell_stats)
    assert set(m.keys()) == {str(r) for r in RADII}
    for row in m.values():
        assert set(row["density_ge_0.02_diff_matrix"].keys()) == set(CAPABILITIES)
        for region_map in row["density_ge_0.02_diff_matrix"].values():
            assert set(region_map.keys()) == set(REGIONS)


# =================================================================================================
# Section 11: scale x radius x anatomy joint reorganization
# =================================================================================================


def test_radius_scale_anatomy_classification_covers_all_capabilities():
    records_by_scale = _full_records_by_scale(n_directions=16)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    out = saa.compute_radius_scale_anatomy_classification(records_by_scale, cell_stats)
    assert set(out.keys()) == set(CAPABILITIES)
    for row in out.values():
        assert row["classification"] in saa.RADIUS_SCALE_ANATOMY_LABELS


def test_diffuse_capability_classified_insufficient_resolution():
    records_by_scale = _full_records_by_scale(n_directions=64)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    out = saa.compute_radius_scale_anatomy_classification(records_by_scale, cell_stats)
    assert out["counting"]["classification"] == "diffuse_or_insufficient_resolution"


# =================================================================================================
# Section 12: region-level macro trend -- row-preserving bootstrap
# =================================================================================================


def test_region_macro_point_estimate_equals_flattened_mean():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = saa.compute_region_macro_scale_trend(records_by_scale)
    matrix = saa._matrix_for_region_radius(records_by_scale["3B"], "vision", RADII[0])
    expected = float((matrix >= saa.USEFUL_MARGIN).mean())
    actual = out["by_scale_region_radius"]["3B"]["vision"][str(RADII[0])]["macro_density_ge_0.02"]
    assert actual == pytest.approx(expected)


def test_region_macro_difference_ci_deterministic():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out1 = saa.compute_region_macro_scale_trend(records_by_scale)
    out2 = saa.compute_region_macro_scale_trend(records_by_scale)
    ci1 = out1["difference_7B_minus_3B"]["vision"][str(RADII[0])]["difference_95ci_bootstrap"]
    ci2 = out2["difference_7B_minus_3B"]["vision"][str(RADII[0])]["difference_95ci_bootstrap"]
    assert ci1 == ci2


def test_region_macro_row_preserving_bootstrap_structure():
    matrix = np.array([[0.1, -0.1, 0.1, -0.1, 0.1, -0.1]] * 32 + [[-0.1, 0.1, -0.1, 0.1, -0.1, 0.1]] * 32)
    density_stat = lambda m: (m >= 0.05).mean(axis=(1, 2))
    dist = saa._macro_stat_bootstrap_distribution(matrix, density_stat, seed=1)
    assert np.allclose(dist, 0.5)  # every row-preserving resample yields exactly 3-of-6 columns >= margin


# =================================================================================================
# Section 13: specialization by anatomy and scale
# =================================================================================================


def test_specialization_covers_every_region_and_radius():
    records_by_scale = _full_records_by_scale(n_directions=16)
    out = saa.compute_specialization_by_anatomy_scale(records_by_scale)
    assert set(out.keys()) == set(REGIONS)
    for radius_map in out.values():
        assert set(radius_map.keys()) == {str(r) for r in RADII}
        for row in radius_map.values():
            assert row["trend"] in ("increases_3B_to_7B", "decreases_3B_to_7B", "no_clear_change")


def test_discordance_bootstrap_distribution_row_resampling_only():
    matrix = np.array([[0.1, 0.2, -0.1, 0.3, 0.0, 0.1]] * 20 + [[-0.2, -0.1, 0.2, -0.3, 0.1, -0.2]] * 20)
    dist = saa._discordance_bootstrap_distribution(matrix, seed=7, n_bootstrap=200)
    assert dist.shape == (200,)
    assert np.all(np.isfinite(dist))


# =================================================================================================
# Section 14: density vs strength classification
# =================================================================================================


def test_density_vs_strength_covers_54_cells():
    records_by_scale = _full_records_by_scale(n_directions=32)
    density_tests = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    strength = saa.compute_anatomy_strength_contrasts(records_by_scale)
    out = saa.classify_anatomy_density_vs_strength(density_tests, strength)
    assert len(out["cells"]) == 54
    for cell in out["cells"].values():
        assert cell["classification"] in saa.MORE_VS_STRONGER_LABELS


def test_density_vs_strength_labels_vision_visual_grounding_expansion():
    records_by_scale = _full_records_by_scale(n_directions=64)
    density_tests = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    strength = saa.compute_anatomy_strength_contrasts(records_by_scale)
    out = saa.classify_anatomy_density_vs_strength(density_tests, strength)
    label = out["cells"][f"visual_grounding:vision:{RADII[0]}"]["classification"]
    assert label in ("more_and_stronger", "more_not_stronger", "stronger_not_more")


def test_density_vs_strength_null_capability_not_labeled_more_and_stronger():
    records_by_scale = _full_records_by_scale(n_directions=64)
    density_tests = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    strength = saa.compute_anatomy_strength_contrasts(records_by_scale)
    out = saa.classify_anatomy_density_vs_strength(density_tests, strength)
    label = out["cells"][f"counting:language:{RADII[2]}"]["classification"]
    assert label in ("neither_clear", "decreases")


# =================================================================================================
# Section 15: headroom sensitivity -- secondary only
# =================================================================================================


def test_headroom_sensitivity_reports_raw_direction_from_unnormalized_mass():
    records_by_scale = _full_records_by_scale(n_directions=16)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    baseline_table = saa.compute_merged_baseline_table(records_by_scale, {"3B": {"capabilities": {c: {"score": BASE_SCORES[c]} for c in CAPABILITIES}}, "7B": {"capabilities": {c: {"score": BASE_SCORES[c]} for c in CAPABILITIES}}})
    out = saa.compute_anatomy_headroom_sensitivity(records_by_scale, baseline_table, cell_stats)
    row = out[f"visual_grounding:vision:{RADII[0]}"]
    expected = cell_stats[f"7B:visual_grounding:vision:{RADII[0]}"]["positive_thicket_mass"] - cell_stats[f"3B:visual_grounding:vision:{RADII[0]}"]["positive_thicket_mass"]
    assert row["raw_positive_mass_diff_7B_minus_3B"] == pytest.approx(expected)
    assert row["raw_conclusion_direction"] in ("increase", "decrease", "flat")


def test_headroom_sensitivity_marks_not_applicable_when_no_headroom():
    records_by_scale = _full_records_by_scale(n_directions=4)
    cap = CAPABILITIES[0]
    for scale in saa.SCALES:
        for i, r in enumerate(records_by_scale[scale]):
            if r.capability == cap:
                records_by_scale[scale][i] = ExperimentResultRecord(**{**r.to_dict(), "base_score": 1.0, "perturbed_score": 1.0 + r.delta, "delta": r.delta})
    baseline_scores = {"3B": {"capabilities": {c: {"score": 1.0 if c == cap else BASE_SCORES[c]} for c in CAPABILITIES}}, "7B": {"capabilities": {c: {"score": 1.0 if c == cap else BASE_SCORES[c]} for c in CAPABILITIES}}}
    baseline_table = saa.compute_merged_baseline_table(records_by_scale, baseline_scores)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    out = saa.compute_anatomy_headroom_sensitivity(records_by_scale, baseline_table, cell_stats)
    row = out[f"{cap}:vision:{RADII[0]}"]
    assert row["normalized_by_scale"]["3B"]["applicable"] is False
    assert row["headroom_sensitivity_verdict"] == "not_applicable"


# =================================================================================================
# Whole-model-to-anatomy interpretation -- non-additive-causality discipline
# =================================================================================================


def test_whole_model_interpretation_never_claims_exact_causal_decomposition():
    records_by_scale = _full_records_by_scale(n_directions=64)
    density_tests = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    s1_summaries = {"visual_grounding": {"classification": "thicket_expands_3B_to_7B"}, "counting": {"classification": "thicket_contracts_3B_to_7B"}}
    out = saa.build_whole_model_to_anatomy_interpretation(s1_summaries, density_tests)
    serialized = json.dumps(out)
    assert "exact causal" not in serialized.lower() or "never" in serialized.lower()
    for row in out["by_capability"].values():
        text = row["interpretation"]
        assert not text.lower().startswith("caused by")
        assert not text.lower().startswith("is the cause of")


def test_whole_model_interpretation_localizes_the_baked_in_visual_grounding_expansion():
    records_by_scale = _full_records_by_scale(n_directions=64)
    density_tests = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    s1_summaries = {"visual_grounding": {"classification": "thicket_expands_3B_to_7B"}}
    out = saa.build_whole_model_to_anatomy_interpretation(s1_summaries, density_tests)
    row = out["by_capability"]["visual_grounding"]
    assert row["interpretation"].startswith("consistent with the vision region")


def test_whole_model_interpretation_handles_missing_s1_summary_gracefully():
    records_by_scale = _full_records_by_scale(n_directions=8)
    density_tests = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    out = saa.build_whole_model_to_anatomy_interpretation(None, density_tests)
    assert out["s1_summary_available"] is False
    for row in out["by_capability"].values():
        assert row["s1_whole_model_classification"] is None


# =================================================================================================
# Terminology guard + claim gate
# =================================================================================================


def test_terminology_guard_forbids_scaling_law_language_with_two_scales():
    guard = saa.TERMINOLOGY_GUARD
    assert guard["n_scales"] == 2
    assert guard["may_use_scaling_relationship_language"] is False
    assert "scaling law" in guard["disallowed_as_empirical_conclusion"]


def test_claim_gate_never_claims_scaling_law_established():
    records_by_scale = _full_records_by_scale(n_directions=16)
    cell_stats = saa.compute_anatomy_cell_statistics(records_by_scale)
    density_tests = saa.compute_cross_scale_anatomy_density_tests(records_by_scale)
    rankings = saa.rank_regions_by_capability_radius_scale(cell_stats)
    transitions = saa.classify_anatomical_preference_transitions(records_by_scale, cell_stats, rankings)
    strength = saa.compute_anatomy_strength_contrasts(records_by_scale)
    density_vs_strength = saa.classify_anatomy_density_vs_strength(density_tests, strength)
    radius_scale_anatomy = saa.compute_radius_scale_anatomy_classification(records_by_scale, cell_stats)
    specialization = saa.compute_specialization_by_anatomy_scale(records_by_scale)
    gate = saa.evaluate_anatomy_interim_claim_gate(cell_stats, density_tests, transitions, density_vs_strength, radius_scale_anatomy, specialization)
    for k in ("A1_coarse_anatomy_structures_density_both_scales", "A2_anatomical_distribution_changes", "A3_scale_effects_capability_dependent", "A4_scale_effects_anatomically_non_uniform", "A5_radius_and_scale_jointly_reorganize", "A6_specialization_changes_differently_by_region"):
        assert gate[k] in saa.CLAIM_VERDICTS
        assert "scaling law" not in gate[k]


# =================================================================================================
# End-to-end determinism against a real fully-shaped fixture on disk
# =================================================================================================


def test_main_runs_end_to_end_and_is_deterministic(full_roots, tmp_path):
    stage8_root, stage11_root = full_roots
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    missing_s1 = str(tmp_path / "no_such_s1_summary.json")
    rc1 = saa.main(["--stage8-dir", str(stage8_root), "--stage11-anatomy-dir", str(stage11_root), "--output-dir", str(out1), "--s1-summary", missing_s1])
    rc2 = saa.main(["--stage8-dir", str(stage8_root), "--stage11-anatomy-dir", str(stage11_root), "--output-dir", str(out2), "--s1-summary", missing_s1])
    assert rc1 == 0 and rc2 == 0

    for name in ("anatomy_cell_statistics.json", "solution_density_curves.json", "interim_claim_gate.json", "anatomical_contrasts.json"):
        assert (out1 / name).read_text() == (out2 / name).read_text()

    integrity = json.loads((out1 / "integrity_report.json").read_text())
    assert integrity["all_ok"] is True

    for name in (
        "integrity_report.json", "anatomy_cell_statistics.json", "anatomy_cell_statistics.csv",
        "solution_density_curves.json", "solution_density_curves.csv", "cross_scale_anatomy_differences.json",
        "anatomy_preference_transitions.json", "anatomical_contrasts.json", "anatomical_difference_in_differences.json",
        "radius_scale_anatomy.json", "region_macro_scale_trend.json", "specialization_by_anatomy_scale.json",
        "density_vs_strength_classification.json", "headroom_sensitivity.json", "whole_model_to_anatomy_interpretation.json",
        "statistical_tests.json", "interim_claim_gate.json", "stage11_interim_3b_7b_anatomy_summary.md",
    ):
        assert (out1 / name).exists(), f"missing output file {name}"
    for name in ("fig_a_atlas_3b_7b.csv", "fig_b_scale_response_atlas.csv", "fig_c_solution_density_by_anatomy.csv", "fig_d_anatomy_preference_transitions.csv", "fig_e_macro_anatomy_scale.csv", "fig_f_specialization_anatomy_scale.csv", "fig_g_anatomical_contrasts.csv"):
        assert (out1 / "figure_schemas" / name).exists(), f"missing figure schema {name}"


def test_main_returns_zero_and_does_not_fabricate_when_no_data(tmp_path):
    rc = saa.main(["--stage8-dir", str(tmp_path / "nope8"), "--stage11-anatomy-dir", str(tmp_path / "nope11"), "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    assert not (tmp_path / "out").exists()
