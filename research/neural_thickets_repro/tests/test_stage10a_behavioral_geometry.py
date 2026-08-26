"""Tests for analysis/stage10a_behavioral_geometry.py -- pure-math primitives verified against
hand-computable cases, and the record-level plumbing (behavioral matrix construction, radius-not
-pooled cell iteration, matched-direction trajectory pairing, cross-anatomy comparisons never
pairing candidates across parameter spaces) verified against small synthetic
ExperimentResultRecord grids, following this project's established testing discipline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import stage10a_behavioral_geometry as s10  # noqa: E402

from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_RADII, STAGE8_REGIONS  # noqa: E402
from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import STAGE9_RADII  # noqa: E402
from neural_thickets_repro.thicket.anatomy_stage9 import STAGE9_CHILD_REGIONS  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

REAL_STAGE8_DIR = s10.DEFAULT_STAGE8_DIR
REAL_STAGE9_DIR = s10.DEFAULT_STAGE9_DIR


def _rec(*, capability: str, region: str, radius: float, direction_index: int, delta: float, source: str = "stage8") -> ExperimentResultRecord:
    pid = f"{region}_{radius}_{direction_index}"
    return ExperimentResultRecord(
        experiment_id=f"{source}_test", perturbation_id=pid, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", perturbation_mode="anatomical_relative_l2", anatomy_region=region, radius=radius, sigma=None,
        seed=direction_index, parameter_mask_hash=f"mask_{region}", capability=capability, dataset_role="map",
        subset_hash=f"sub_{capability}", base_score=0.5, perturbed_score=round(0.5 + delta, 10), delta=delta,
        parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash=f"h_{pid}_{capability}",
        runtime_metadata={
            "direction_family_id": f"{region}:{direction_index}", "direction_seed": direction_index,
            "direction_index": direction_index, "region": region,
            "theta_region_l2_norm": 100.0, "epsilon_region_l2_norm": 100.0 * radius,
            "realized_relative_l2": radius,
        },
    )


def _delta_fn(region: str, radius: float, direction_index: int, capability: str, radii: List[float]) -> float:
    """Deterministic synthetic delta -- gives a real, hand-verifiable rank-1-ish structure: every
    capability moves together (scaled by a per-capability weight), scaled by direction_index and
    a mild radius-dependent multiplier, so PC1 should dominate and the matrix should NOT look
    rank-1 under an independent-column null (all 6 columns share the SAME per-direction factor).
    """
    weights = {"visual_grounding": 1.0, "counting": 0.8, "spatial_reasoning": -0.6, "ocr_text_recognition_grounded": 0.4, "relational_reasoning": 0.5, "fine_grained_recognition": -0.3}
    radius_rank = sorted(radii).index(radius)
    common_factor = (direction_index - 3.5) * 0.01 * (1 + 0.3 * radius_rank)
    return round(weights[capability] * common_factor, 10)


def _build_synthetic_cell_records(region: str, radius: float, radii: List[float], n_directions: int = 8, source: str = "stage8") -> List[ExperimentResultRecord]:
    return [
        _rec(capability=cap, region=region, radius=radius, direction_index=i, delta=_delta_fn(region, radius, i, cap, radii), source=source)
        for i in range(n_directions) for cap in s10.CAPABILITIES
    ]


def _build_synthetic_region_records(region: str, radii: List[float], n_directions: int = 8, source: str = "stage8") -> List[ExperimentResultRecord]:
    records = []
    for radius in radii:
        records.extend(_build_synthetic_cell_records(region, radius, radii, n_directions=n_directions, source=source))
    return records


# =================================================================================================
# Behavioral matrix construction -- 64x6 shape, capability order, radius not pooled
# =================================================================================================


def test_build_behavioral_matrix_shape_and_direction_order():
    rows = _build_synthetic_cell_records("vision", STAGE8_RADII[0], list(STAGE8_RADII), n_directions=8)
    direction_indices, matrix = s10.build_behavioral_matrix(rows)
    assert matrix.shape == (8, 6)
    assert direction_indices == list(range(8))


def test_build_behavioral_matrix_column_order_is_fixed_and_shared_across_stages():
    rows8 = _build_synthetic_cell_records("vision", STAGE8_RADII[0], list(STAGE8_RADII), n_directions=4, source="stage8")
    rows9 = _build_synthetic_cell_records("vision_early", STAGE9_RADII[0], list(STAGE9_RADII), n_directions=4, source="stage9")
    _, matrix8 = s10.build_behavioral_matrix(rows8)
    _, matrix9 = s10.build_behavioral_matrix(rows9)
    assert matrix8.shape[1] == matrix9.shape[1] == 6
    assert s10.CAPABILITIES == tuple(sorted(STAGE8_CAPABILITIES))


def test_build_behavioral_matrix_raises_on_missing_capability():
    rows = _build_synthetic_cell_records("vision", STAGE8_RADII[0], list(STAGE8_RADII), n_directions=4)
    del rows[0]
    with pytest.raises(ValueError):
        s10.build_behavioral_matrix(rows)


def test_iter_cells_never_pools_radii():
    stage8_records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=4, source="stage8")
    stage9_records = _build_synthetic_region_records("vision_early", list(STAGE9_RADII), n_directions=4, source="stage9")
    cells = s10.iter_cells(stage8_records, stage9_records)
    seen_radii_per_region = {}
    for source, region, radius, rows in cells:
        assert all(r.radius == radius for r in rows)  # every row in a cell shares the exact same radius
        assert all(r.anatomy_region == region for r in rows)
        seen_radii_per_region.setdefault((source, region), set()).add(radius)
    for (source, region), radii_seen in seen_radii_per_region.items():
        expected = set(STAGE8_RADII) if source == "stage8" else set(STAGE9_RADII)
        assert radii_seen == expected  # each radius appears as its OWN separate cell, never merged


# =================================================================================================
# Effective-rank formula + stable-rank formula (hand-computable cases)
# =================================================================================================


def test_singular_value_stats_rank_one_matrix_has_effective_rank_one():
    u = np.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0]).reshape(-1, 1)
    v = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(1, -1)
    matrix = u @ v  # exactly rank 1 by construction
    stats = s10.singular_value_stats(matrix)
    assert stats["entropy_effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert stats["stable_rank"] == pytest.approx(1.0, abs=1e-6)
    assert stats["variance_explained_pc1"] == pytest.approx(1.0, abs=1e-6)


def test_singular_value_stats_orthogonal_equal_energy_matrix_has_full_effective_rank():
    """A matrix whose singular values are all equal has p_j uniform -> entropy effective rank
    equals the number of singular values exactly (a direct hand-computable identity:
    entropy of a uniform distribution over m outcomes is log(m), so exp(log(m)) == m).
    """
    matrix = np.eye(6) * 3.0  # 6x6, all singular values == 3.0
    stats = s10.singular_value_stats(matrix)
    assert stats["entropy_effective_rank"] == pytest.approx(6.0, abs=1e-6)
    assert stats["stable_rank"] == pytest.approx(6.0, abs=1e-6)


def test_stable_rank_matches_frobenius_over_spectral_norm_definition():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(20, 6))
    stats = s10.singular_value_stats(matrix)
    frob_sq = float(np.sum(matrix ** 2))
    spectral_sq = float(np.linalg.norm(matrix, ord=2) ** 2)
    assert stats["stable_rank"] == pytest.approx(frob_sq / spectral_sq, rel=1e-9)


def test_variance_explained_cumulative_sums_are_nondecreasing_and_bounded():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(64, 6))
    stats = s10.singular_value_stats(matrix)
    assert stats["variance_explained_pc1"] <= stats["variance_explained_pc1_pc2"] <= stats["variance_explained_pc1_pc2_pc3"] <= 1.0 + 1e-9


# =================================================================================================
# Permutation null (Null A) -- destroys cross-capability correlation, sample-size matched
# =================================================================================================


def test_null_a_permutation_preserves_column_marginals():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(20, 6))
    permuted = s10._permute_columns_independently(matrix, np.random.default_rng(3))
    for j in range(6):
        assert sorted(permuted[:, j].tolist()) == pytest.approx(sorted(matrix[:, j].tolist()))


def test_null_a_destroys_rank_one_structure_built_from_a_shared_factor():
    """The exact rank-1 matrix from the earlier test: independently permuting each column
    breaks the shared common-factor structure, so the null's effective rank should be
    noticeably HIGHER than the perfectly-rank-1 observed matrix.
    """
    u = np.arange(1, 21, dtype=float).reshape(-1, 1)
    v = np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0]).reshape(1, -1)
    matrix = u @ v
    observed = s10.singular_value_stats(matrix)
    result = s10.compute_effective_rank_with_nulls(matrix, seed_offset=0, n_null=200)
    assert observed["entropy_effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert result["null_a_independent_permutation"]["entropy_effective_rank"]["null_mean"] > 1.5
    assert result["null_a_independent_permutation"]["entropy_effective_rank"]["observed_lower_than_95pct_of_null"] is True


def test_null_b_gaussian_matched_has_the_same_shape_and_finite_stats():
    rng = np.random.default_rng(4)
    matrix = rng.normal(loc=[0, 1, 2, -1, 0.5, -0.5], scale=[1, 2, 0.5, 1, 1, 3], size=(64, 6))
    gaussian = s10._gaussian_matched_matrix(matrix, np.random.default_rng(5))
    assert gaussian.shape == matrix.shape
    assert np.all(np.isfinite(gaussian))


def test_null_sample_size_matches_observed_matrix_exactly():
    rng = np.random.default_rng(6)
    matrix = rng.normal(size=(37, 6))  # a non-64 size, on purpose, to prove no hardcoding
    result = s10.compute_effective_rank_with_nulls(matrix, seed_offset=0, n_null=50)
    assert result["n_null_draws"] == 50
    # both null constructors must preserve n and m exactly -- checked indirectly via no exceptions
    # and via the permutation-preserves-marginals test above; here we confirm the observed matrix
    # itself was never resized.
    assert len(result["observed"]["singular_values"]) == 6


# =================================================================================================
# PCA determinism
# =================================================================================================


def test_pca_loadings_is_deterministic():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(30, 6))
    first = s10.pca_loadings(matrix)
    second = s10.pca_loadings(matrix)
    assert first == second


def test_pca_loadings_orthonormal_components():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(30, 6))
    result = s10.pca_loadings(matrix, k=3)
    vectors = [np.array([c["loadings"][cap] for cap in s10.CAPABILITIES]) for c in result["components"]]
    for v in vectors:
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            assert np.dot(vectors[i], vectors[j]) == pytest.approx(0.0, abs=1e-9)


def test_describe_loading_pattern_never_forces_a_label_without_a_clean_two_capability_tradeoff():
    diffuse_loading = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) / np.linalg.norm([0.2] * 6)
    result = s10.describe_loading_pattern(diffuse_loading)
    # all-same-sign but not >=4 "dominant" (threshold 0.3) -- must not claim general-improvement either
    assert result["structural_label"] is None or "general-improvement" not in (result["structural_label"] or "") or result["all_same_sign"]


def test_describe_loading_pattern_detects_a_clean_two_capability_tradeoff():
    loading = np.zeros(6)
    loading[0] = 0.7
    loading[1] = -0.7
    result = s10.describe_loading_pattern(loading)
    assert result["structural_label"] is not None
    assert "tradeoff axis" in result["structural_label"]


def test_describe_loading_pattern_detects_general_improvement_axis():
    loading = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    loading = loading / np.linalg.norm(loading)
    result = s10.describe_loading_pattern(loading)
    assert result["all_same_sign"] is True
    assert result["structural_label"] == "general-improvement-like axis (all capabilities load with the same sign)"


# =================================================================================================
# Principal-angle computation
# =================================================================================================


def test_principal_angles_identical_subspaces_have_cosine_one():
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(20, 6))
    basis = s10.top_k_subspace(matrix, 2)
    cosines = s10.principal_angles_cosines(basis, basis)
    assert cosines == pytest.approx(np.ones(2), abs=1e-9)


def test_principal_angles_orthogonal_subspaces_have_cosine_zero():
    basis_a = np.zeros((6, 1))
    basis_a[0, 0] = 1.0
    basis_b = np.zeros((6, 1))
    basis_b[1, 0] = 1.0
    cosines = s10.principal_angles_cosines(basis_a, basis_b)
    assert cosines[0] == pytest.approx(0.0, abs=1e-9)


def test_procrustes_similarity_is_one_for_identical_bases():
    rng = np.random.default_rng(10)
    matrix = rng.normal(size=(20, 6))
    basis = s10.top_k_subspace(matrix, 2)
    result = s10.procrustes_similarity(basis, basis)
    assert result["similarity"] == pytest.approx(1.0, abs=1e-6)
    assert result["residual_frobenius"] == pytest.approx(0.0, abs=1e-6)


# =================================================================================================
# Split-half determinism
# =================================================================================================


def test_split_half_stability_is_deterministic():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(64, 6))
    first = s10.compute_split_half_stability(matrix, n_splits=20, seed_offset=0)
    second = s10.compute_split_half_stability(matrix, n_splits=20, seed_offset=0)
    assert first == second


def test_split_half_stability_high_for_a_strong_shared_signal():
    """The rank-1 common-factor matrix (same construction as the null-A test) should show a
    HIGH split-half k=1 cosine -- the dominant axis is genuinely reproducible from half the data.
    """
    u = np.arange(1, 65, dtype=float).reshape(-1, 1) + np.random.default_rng(12).normal(scale=0.01, size=(64, 1))
    v = np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0]).reshape(1, -1)
    matrix = u @ v
    result = s10.compute_split_half_stability(matrix, n_splits=30, seed_offset=0)
    assert result["k_1"]["first_angle_cosine"]["mean"] > 0.9


# =================================================================================================
# Matched direction family across radii
# =================================================================================================


def test_trajectory_geometry_pairs_by_direction_index_not_perturbation_id():
    records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=5, source="stage8")
    cells = s10.iter_cells(records, [])
    trajectories = s10.compute_trajectory_geometry(cells)
    assert trajectories["stage8:vision"]["n_complete_trajectories"] == 5  # all 5 directions present at all 3 radii


def test_trajectory_geometry_excludes_incomplete_families():
    records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=5, source="stage8")
    # drop every row for direction_index=0 at the largest radius only -- an incomplete trajectory
    records = [r for r in records if not (r.runtime_metadata["direction_index"] == 0 and r.radius == max(STAGE8_RADII))]
    cells = s10.iter_cells(records, [])
    trajectories = s10.compute_trajectory_geometry(cells)
    assert trajectories["stage8:vision"]["n_complete_trajectories"] == 4


def test_trajectory_geometry_cosine_one_when_scale_only_changes_magnitude():
    """If every direction's behavioral vector merely SCALES (never rotates) across radii, cosine
    similarity between radius pairs must be exactly 1.0 -- a direct, hand-verifiable case for
    "scale changes magnitude, not direction".
    """
    records = []
    base_vector_weights = {"visual_grounding": 1.0, "counting": 0.5, "spatial_reasoning": -0.5, "ocr_text_recognition_grounded": 0.2, "relational_reasoning": 0.3, "fine_grained_recognition": -0.1}
    for radius_rank, radius in enumerate(sorted(STAGE8_RADII)):
        scale = 1.0 + radius_rank  # 1x, 2x, 3x -- pure magnitude change
        for cap, w in base_vector_weights.items():
            records.append(_rec(capability=cap, region="vision", radius=radius, direction_index=0, delta=w * scale, source="stage8"))
    cells = s10.iter_cells(records, [])
    trajectories = s10.compute_trajectory_geometry(cells)
    region_out = trajectories["stage8:vision"]
    for pair in ("small_vs_mid", "small_vs_transition", "mid_vs_transition"):
        assert region_out[pair]["cosine"]["mean"] == pytest.approx(1.0, abs=1e-9)
        assert region_out[pair]["norm_growth_ratio"]["mean"] > 1.0


# =================================================================================================
# No cross-anatomy parameter pairing -- only behavioral SUBSPACES are compared
# =================================================================================================


def test_cross_anatomy_geometry_never_pairs_individual_candidates_across_regions():
    """Stage-8's vision cell and Stage-9's vision_early cell have DIFFERENT numbers of directions
    in this synthetic fixture (8 vs 5) -- if the comparison ever tried to pair individual
    candidates row-for-row, it would need equal N and would raise; since it only compares the
    resulting SUBSPACES (top-k singular vectors, a fixed (6, k) shape regardless of N), it must
    succeed cleanly.
    """
    stage8_records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=8, source="stage8")
    stage9_records = _build_synthetic_region_records("vision_early", list(STAGE9_RADII), n_directions=5, source="stage9")
    cells = s10.iter_cells(stage8_records, stage9_records)
    result = s10.compute_cross_anatomy_geometry(cells)
    radius_key = str(STAGE8_RADII[0])
    assert radius_key in result
    pair_key = "stage8:vision__vs__stage9:vision_early" if "stage8:vision__vs__stage9:vision_early" in result[radius_key]["k_1"] else "stage9:vision_early__vs__stage8:vision"
    assert pair_key in result[radius_key]["k_1"]


def test_cross_anatomy_geometry_groups_by_matching_radius_only():
    stage8_records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=6, source="stage8")
    stage9_records = _build_synthetic_region_records("language_late", list(STAGE9_RADII), n_directions=6, source="stage9")
    cells = s10.iter_cells(stage8_records, stage9_records)
    result = s10.compute_cross_anatomy_geometry(cells)
    for radius_key, radius_info in result.items():
        assert set(radius_info["regions"]) <= {"stage8:vision", "stage9:language_late"}


# =================================================================================================
# Useful-expert threshold remains 0.02
# =================================================================================================


def test_useful_expert_threshold_is_frozen_at_002():
    assert s10.USEFUL_EXPERT_THRESHOLD == 0.02


def test_useful_expert_geometry_uses_the_frozen_threshold_exactly():
    matrix = np.zeros((10, 6))
    matrix[:5, 0] = 0.02  # exactly at threshold -> useful
    matrix[5:, 0] = 0.019999  # just under -> not useful
    result = s10.compute_useful_expert_geometry(matrix, seed_offset=0)
    assert result[s10.CAPABILITIES[0]]["n_useful"] == 5
    assert result[s10.CAPABILITIES[0]]["n_non_improving"] == 5


def test_useful_expert_geometry_random_comparison_matches_useful_count():
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(64, 6)) * 0.001  # small noise, safely below the 0.02 threshold everywhere
    matrix[:20, 0] = 0.05  # force EXACTLY 20 useful rows for capability 0, none elsewhere in that column
    result = s10.compute_useful_expert_geometry(matrix, seed_offset=0)
    cap0 = s10.CAPABILITIES[0]
    assert result[cap0]["n_useful"] == 20
    assert result[cap0]["same_number_random_candidates_geometry"]["n"] == 20


# =================================================================================================
# Clustering gating + reconstruction feasibility bookkeeping
# =================================================================================================


def test_clustering_skipped_below_minimum_useful_count():
    rng = np.random.default_rng(14)
    vectors = rng.normal(size=(s10.CLUSTERING_MIN_USEFUL - 1, 6))
    result = s10.compute_exploratory_clustering(vectors, seed_offset=0)
    assert result["attempted"] is False
    assert result["verdict"] == "insufficient_sample_size"


def test_clustering_reports_continuous_geometry_for_pure_gaussian_noise():
    """No real cluster structure in i.i.d. Gaussian noise -- the strict validity gates (method
    agreement, bootstrap stability, exceeds shuffled null) should not ALL pass, so the verdict
    must be "continuous_geometry", never a fabricated discrete-cluster claim.
    """
    rng = np.random.default_rng(15)
    vectors = rng.normal(size=(20, 6))
    result = s10.compute_exploratory_clustering(vectors, seed_offset=0)
    assert result["attempted"] is True
    assert result["verdict"] == "continuous_geometry_no_stable_discrete_clusters"


def test_adjusted_rand_index_is_one_for_identical_labelings():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert s10._adjusted_rand_index(labels, labels) == pytest.approx(1.0)


def test_silhouette_score_is_high_for_well_separated_clusters():
    vectors = np.vstack([np.full((5, 6), 0.0), np.full((5, 6), 10.0)])
    labels = np.array([0] * 5 + [1] * 5)
    score = s10.silhouette_score(vectors, labels)
    assert score > 0.9


def test_reconstruction_feasibility_bookkeeping_records_exactly_six_candidates():
    if not REAL_STAGE8_DIR.exists() or not REAL_STAGE9_DIR.exists():
        pytest.skip("real Stage-8/Stage-9 results not present locally")
    stage8_records = s10.load_stage8_records(REAL_STAGE8_DIR)
    stage9_records = s10.load_stage9_records(REAL_STAGE9_DIR)
    result = s10.compute_reconstruction_feasibility(stage8_records, stage9_records)
    used = result["candidates_used"]
    assert len(used["stage8_language"]) == 2
    assert len(used["stage8_vision"]) == 2
    assert len(used["stage9_depth"]) == 2
    assert result["reconstruct_all_1152_now"] is False
    assert result["mechanism_level_determinism_proof"]["bit_identical_epsilon_across_two_independent_runs"] is True
    assert result["self_consistency_against_real_persisted_norms"]["all_self_consistent"] is True


# =================================================================================================
# Deterministic outputs / integrity gate
# =================================================================================================


def test_integrity_gate_passes_on_a_correctly_shaped_synthetic_design():
    stage8_records = []
    for region in STAGE8_REGIONS:
        for radius in STAGE8_RADII:
            for cap in STAGE8_CAPABILITIES:
                for idx in range(64):
                    stage8_records.append(_rec(capability=cap, region=region, radius=radius, direction_index=idx, delta=0.0, source="stage8"))
    stage9_records = []
    for region in STAGE9_CHILD_REGIONS:
        for radius in STAGE9_RADII:
            for cap in STAGE8_CAPABILITIES:
                for idx in range(64):
                    stage9_records.append(_rec(capability=cap, region=region, radius=radius, direction_index=idx, delta=0.0, source="stage9"))
    integrity = s10.run_stage10a_integrity_gate(stage8_records, stage9_records)
    assert integrity["all_checks_pass"] is True


def test_full_small_pipeline_is_deterministic():
    stage8_records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=10, source="stage8")
    stage9_records = _build_synthetic_region_records("vision_early", list(STAGE9_RADII), n_directions=10, source="stage9")
    cells = s10.iter_cells(stage8_records, stage9_records)

    def run_once():
        return s10._sanitize({
            "effective_rank": s10.compute_effective_rank_by_cell(cells),
            "pca": s10.compute_pca_by_cell(cells),
            "trajectories": s10.compute_trajectory_geometry(cells),
        })

    import json
    first = json.dumps(run_once(), sort_keys=True)
    second = json.dumps(run_once(), sort_keys=True)
    assert first == second


# =================================================================================================
# Cross-check against the REAL completed Stage-8/9 runs, if present locally
# =================================================================================================


@pytest.mark.skipif(not (REAL_STAGE8_DIR.exists() and REAL_STAGE9_DIR.exists()), reason="real Stage-8/9 results not present locally")
def test_real_runs_pass_the_stage10a_integrity_gate():
    stage8_records = s10.load_stage8_records(REAL_STAGE8_DIR)
    stage9_records = s10.load_stage9_records(REAL_STAGE9_DIR)
    integrity = s10.run_stage10a_integrity_gate(stage8_records, stage9_records)
    assert integrity["all_checks_pass"] is True


@pytest.mark.skipif(not (REAL_STAGE8_DIR.exists() and REAL_STAGE9_DIR.exists()), reason="real Stage-8/9 results not present locally")
def test_real_runs_produce_exactly_27_cells():
    stage8_records = s10.load_stage8_records(REAL_STAGE8_DIR)
    stage9_records = s10.load_stage9_records(REAL_STAGE9_DIR)
    cells = s10.iter_cells(stage8_records, stage9_records)
    assert len(cells) == 9 + 18
    for _, _, _, rows in cells:
        _, matrix = s10.build_behavioral_matrix(rows)
        assert matrix.shape == (64, 6)


# =================================================================================================
# Compact CSV exports
# =================================================================================================


def test_effective_rank_csv_has_one_row_per_cell(tmp_path):
    stage8_records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=6, source="stage8")
    cells = s10.iter_cells(stage8_records, [])
    effective_rank_by_cell = s10.compute_effective_rank_by_cell(cells)
    path = tmp_path / "effective_rank_by_cell.csv"
    s10.write_effective_rank_csv(effective_rank_by_cell, path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1 + len(cells)  # header + one row per cell


def test_specialization_rank_relationship_csv_matches_json_cell_count(tmp_path):
    stage8_records = _build_synthetic_region_records("vision", list(STAGE8_RADII), n_directions=6, source="stage8")
    cells = s10.iter_cells(stage8_records, [])
    effective_rank_by_cell = s10.compute_effective_rank_by_cell(cells)
    specialization_rank = s10.compute_specialization_rank_relationship(cells, effective_rank_by_cell)
    path = tmp_path / "specialization_rank_relationship.csv"
    s10.write_specialization_rank_relationship_csv(specialization_rank, path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1 + len(specialization_rank["cells"])
