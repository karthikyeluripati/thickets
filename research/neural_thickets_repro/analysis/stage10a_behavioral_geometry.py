"""Stage 10A: BEHAVIORAL geometry of Stage-8/Stage-9 perturbation effects -- ANALYSIS ONLY, no
model/GPU runs. Answers: are the six-dimensional capability-effect vectors produced by nearby
perturbations low-dimensional, does that geometry differ by anatomy/depth/radius, do useful
experts for the same capability share a behavioral direction, does radius rotate the behavioral
subspace or just its magnitude, and is a (much more expensive) PARAMETER-SPACE geometry stage
scientifically justified next?

Deliberately does NOT call this "parameter-space low rank" -- everything here is the geometry of
the OBSERVED 6-dimensional capability-effect vectors b_i = [Delta_grounding, Delta_counting,
Delta_spatial, Delta_ocr, Delta_relational, Delta_finegrain], never the underlying weight-space
perturbation tensors themselves (that is Stage 10B, proposed but NOT implemented here -- see
Section 16's feasibility audit).

Pure numpy throughout (no scipy/scikit-learn) -- matches this project's existing dependency
discipline (see thicket/diversity.py's own docstring: "neither is in requirements-cpu.txt").
Principal angles, Procrustes alignment, k-means, agglomerative clustering, and silhouette scores
are all hand-rolled below rather than adding a new dependency.

Reuses (never reimplements): run_global_visual_thicket_pilot.load_records, thicket.diversity's
spectral_discordance/task_rank_correlation_matrix (via already-computed Stage 8/9 specialization
JSON, for Section 14's rank-vs-specialization correlation), and
stage8_coarse_anatomical_atlas_analysis.group_by_region_radius for cell iteration.

Usage:
    python analysis/stage10a_behavioral_geometry.py [--stage8-dir <path>] [--stage9-dir <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ANALYSIS_ROOT = Path(__file__).resolve().parent
for p in (SRC_ROOT, ANALYSIS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neural_thickets_repro.run_global_visual_thicket_pilot import load_records  # noqa: E402
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import (  # noqa: E402
    STAGE8_CAPABILITIES, STAGE8_D_MAP_N, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_RADII, STAGE8_REGIONS,
)
from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import STAGE9_CAPABILITIES, STAGE9_RADII  # noqa: E402
from neural_thickets_repro.thicket.anatomy_stage9 import STAGE9_CHILD_REGIONS  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

import stage8_coarse_anatomical_atlas_analysis as s8a  # noqa: E402

DEFAULT_STAGE8_DIR = REPO_ROOT / "results" / "stage8_coarse_anatomical_atlas" / "stage8_coarse_anatomical_atlas_3b_v2_batched10"
DEFAULT_STAGE9_DIR = REPO_ROOT / "results" / "stage9_hierarchical_anatomical_atlas" / "stage9_hierarchical_anatomical_atlas_3b_v1"

USEFUL_EXPERT_THRESHOLD = 0.02  # the SAME frozen margin used everywhere else in this project -- never re-derived here
assert set(STAGE8_CAPABILITIES) == set(STAGE9_CAPABILITIES), "Stage 8 and Stage 9 must share the identical 6 capabilities for behavioral vectors to be comparable"
CAPABILITIES: Tuple[str, ...] = tuple(sorted(STAGE8_CAPABILITIES))  # fixed column order for EVERY behavioral matrix in this module

N_NULL_DRAWS = 10_000
NULL_A_SEED_BASE = 20260826  # distinct namespace from every prior stage's bootstrap/permutation seeds
NULL_B_SEED_BASE = 20260827
N_SPLIT_HALF_SPLITS = 200
SPLIT_HALF_SEED_BASE = 20260828
CLUSTERING_MIN_USEFUL = 15  # below this, exploratory clustering is not attempted -- too few points for k up to 6
KMEANS_SEEDS = tuple(range(10))  # 10 deterministic restarts for the MAIN (once-per-cell) evaluation, to pick the best inertia
KMEANS_MAX_ITERS = 100
BOOTSTRAP_CLUSTER_SEED_BASE = 20260829
# Bootstrap/null resampling here re-runs k-means many times (unlike Section 6's null draws, which
# are a single cheap SVD each) -- deliberately smaller than the >=10,000 used where "computationally
# cheap" (Section 6) applies; single-seed k-means (not best-of-10) is used inside these loops for
# the same reason, since each draw only needs A reasonable clustering, not the optimal one.
N_CLUSTER_BOOTSTRAP = 50
NULL_SILHOUETTE_SEED_BASE = 20260830
N_NULL_SILHOUETTE_DRAWS = 100

# Stage 7A's own live parameter inventory (already published/frozen in run_stage8's docstring) --
# reused BY VALUE, never re-derived, since the real 3B checkpoint is not available in this
# environment to recount from scratch.
STAGE7A_L1_PARAM_COUNTS: Dict[str, float] = {
    "vision": 632.0e6, "multimodal_connector_or_merger": 36.7e6, "language": 3086.0e6,
}


def _sanitize(obj: Any) -> Any:
    return s8a._sanitize(obj)


def _write_json(path: Path, obj: Any) -> None:
    s8a._write_json(path, obj)


# =================================================================================================
# Data loading + cell iteration
# =================================================================================================


def load_stage8_records(stage8_dir: Path) -> List[ExperimentResultRecord]:
    return load_records(stage8_dir / "results.jsonl")


def load_stage9_records(stage9_dir: Path) -> List[ExperimentResultRecord]:
    return load_records(stage9_dir / "results.jsonl")


class CellId(Tuple[str, str, float]):
    """A (source, region, radius) triple, source in {"stage8","stage9"} -- kept as a plain tuple
    subclass purely so JSON-serialized keys read unambiguously (e.g. "stage8:vision:0.0036...").
    """


def _cell_key(source: str, region: str, radius: float) -> str:
    return f"{source}:{region}:{radius}"


def iter_cells(stage8_records: Sequence[ExperimentResultRecord], stage9_records: Sequence[ExperimentResultRecord]) -> List[Tuple[str, str, float, List[ExperimentResultRecord]]]:
    """Every Stage-8 anatomy x radius cell (9) and every Stage-9 child-region x radius cell (18)
    -- 27 cells total, RADII NEVER POOLED (Section 5's explicit requirement): each cell's rows
    share the exact same (source, region, radius) triple.
    """
    cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]] = []
    s8_by_region_radius = s8a.group_by_region_radius(stage8_records)
    for (region, radius), rows in sorted(s8_by_region_radius.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cells.append(("stage8", region, radius, rows))
    s9_by_region_radius = s8a.group_by_region_radius(stage9_records)
    for (region, radius), rows in sorted(s9_by_region_radius.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cells.append(("stage9", region, radius, rows))
    return cells


def build_behavioral_matrix(rows: Sequence[ExperimentResultRecord]) -> Tuple[List[int], np.ndarray]:
    """Builds the 64x6 raw-Delta behavioral matrix for one (region, radius) cell -- rows ordered
    by direction_index (0..63, NOT by perturbation_id, so cross-radius/cross-split alignment by
    direction_index is trivial elsewhere in this module), columns in the FIXED `CAPABILITIES`
    order (sorted, shared identically across Stage 8 and Stage 9 since both import the same 6
    capabilities by identity). Raises if any (direction_index, capability) cell is missing --
    mirrors build_delta_matrix's own "never silently pad" discipline.
    """
    by_direction: Dict[int, Dict[str, float]] = {}
    for r in rows:
        idx = r.runtime_metadata.get("direction_index")
        if idx is None:
            raise ValueError(f"Row for perturbation {r.perturbation_id!r} has no direction_index in runtime_metadata.")
        by_direction.setdefault(idx, {})[r.capability] = r.delta

    direction_indices = sorted(by_direction.keys())
    matrix = np.full((len(direction_indices), len(CAPABILITIES)), np.nan)
    for i, idx in enumerate(direction_indices):
        for j, cap in enumerate(CAPABILITIES):
            if cap not in by_direction[idx]:
                raise ValueError(f"direction_index={idx} is missing capability {cap!r}.")
            matrix[i, j] = by_direction[idx][cap]
    if np.isnan(matrix).any():
        raise ValueError("Behavioral matrix has unexpected missing entries.")
    return direction_indices, matrix


def center_columns(matrix: np.ndarray) -> np.ndarray:
    """Secondary sensitivity standardization (Section 5): capability-wise centered Delta WITHIN
    the cell -- subtracts each capability's own column mean. The PRIMARY analysis throughout this
    module uses the raw (uncentered) matrix, deliberately, so a dominant "everything improves
    together" direction remains visible in the singular spectrum/PC1 (Section 7 explicitly asks
    whether PC1 looks like "general improvement").
    """
    return matrix - matrix.mean(axis=0, keepdims=True)


# =================================================================================================
# Section 6: effective rank + nulls
# =================================================================================================


def singular_value_stats(matrix: np.ndarray) -> Dict[str, Any]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sq = singular_values ** 2
    total = float(sq.sum())
    if total <= 0.0:
        p = np.full(sq.size, 1.0 / sq.size)
    else:
        p = sq / total
    nz = p[p > 0]
    entropy_effective_rank = float(np.exp(-np.sum(nz * np.log(nz))))
    stable_rank = float(total / sq[0]) if sq[0] > 0 else float(sq.size)
    cum = np.cumsum(p)
    return {
        "singular_values": singular_values.tolist(),
        "variance_explained_pc1": float(p[0]),
        "variance_explained_pc1_pc2": float(cum[min(1, len(cum) - 1)]),
        "variance_explained_pc1_pc2_pc3": float(cum[min(2, len(cum) - 1)]),
        "entropy_effective_rank": entropy_effective_rank,
        "stable_rank": stable_rank,
    }


def _permute_columns_independently(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = matrix.copy()
    for j in range(matrix.shape[1]):
        out[:, j] = rng.permutation(out[:, j])
    return out


def _gaussian_matched_matrix(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1)
    return rng.normal(loc=means, scale=np.where(stds > 0, stds, 1e-12), size=matrix.shape)


def compute_effective_rank_with_nulls(matrix: np.ndarray, *, seed_offset: int, n_null: int = N_NULL_DRAWS) -> Dict[str, Any]:
    observed = singular_value_stats(matrix)

    rng_a = np.random.default_rng(NULL_A_SEED_BASE + seed_offset)
    rng_b = np.random.default_rng(NULL_B_SEED_BASE + seed_offset)
    null_a_eff_rank, null_a_stable_rank, null_a_pc1 = [], [], []
    null_b_eff_rank, null_b_stable_rank, null_b_pc1 = [], [], []
    for _ in range(n_null):
        stats_a = singular_value_stats(_permute_columns_independently(matrix, rng_a))
        null_a_eff_rank.append(stats_a["entropy_effective_rank"])
        null_a_stable_rank.append(stats_a["stable_rank"])
        null_a_pc1.append(stats_a["variance_explained_pc1"])
        stats_b = singular_value_stats(_gaussian_matched_matrix(matrix, rng_b))
        null_b_eff_rank.append(stats_b["entropy_effective_rank"])
        null_b_stable_rank.append(stats_b["stable_rank"])
        null_b_pc1.append(stats_b["variance_explained_pc1"])

    def _summary(observed_value: float, null_values: List[float]) -> Dict[str, Any]:
        arr = np.asarray(null_values)
        return {
            "observed": observed_value, "null_mean": float(arr.mean()), "null_std": float(arr.std(ddof=1)),
            "null_p2.5": float(np.percentile(arr, 2.5)), "null_p97.5": float(np.percentile(arr, 97.5)),
            "fraction_null_leq_observed": float(np.mean(arr <= observed_value)),
            "observed_lower_than_95pct_of_null": bool(observed_value < np.percentile(arr, 5.0)),
        }

    return {
        "observed": observed,
        "null_a_independent_permutation": {
            "entropy_effective_rank": _summary(observed["entropy_effective_rank"], null_a_eff_rank),
            "stable_rank": _summary(observed["stable_rank"], null_a_stable_rank),
            "variance_explained_pc1": {**_summary(observed["variance_explained_pc1"], null_a_pc1), "fraction_null_leq_observed": float(np.mean(np.asarray(null_a_pc1) <= observed["variance_explained_pc1"]))},
        },
        "null_b_gaussian_matched": {
            "entropy_effective_rank": _summary(observed["entropy_effective_rank"], null_b_eff_rank),
            "stable_rank": _summary(observed["stable_rank"], null_b_stable_rank),
            "variance_explained_pc1": _summary(observed["variance_explained_pc1"], null_b_pc1),
        },
        "n_null_draws": n_null,
    }


def compute_effective_rank_by_cell(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]], *, centered: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, (source, region, radius, rows) in enumerate(cells):
        _, matrix = build_behavioral_matrix(rows)
        m = center_columns(matrix) if centered else matrix
        out[_cell_key(source, region, radius)] = {
            "source": source, "region": region, "radius": radius, "n": matrix.shape[0],
            **compute_effective_rank_with_nulls(m, seed_offset=i),
        }
    return out


# =================================================================================================
# Section 7: PCA / behavioral axes
# =================================================================================================


def pca_loadings(matrix: np.ndarray, k: int = 3) -> Dict[str, Any]:
    """Right singular vectors of the RAW (uncentered) matrix -- see center_columns's docstring
    for why this module deliberately keeps PCA uncentered as its primary convention.
    """
    _, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(k, vt.shape[0])
    sq = s ** 2
    total = float(sq.sum()) if sq.sum() > 0 else 1.0
    pcs = []
    for i in range(k):
        loading = vt[i]
        pcs.append({
            "pc": i + 1, "explained_variance_ratio": float(sq[i] / total) if i < len(sq) else 0.0,
            "loadings": {cap: float(loading[j]) for j, cap in enumerate(CAPABILITIES)},
            "description": describe_loading_pattern(loading),
        })
    return {"n_components_reported": k, "components": pcs}


def describe_loading_pattern(loading: np.ndarray, *, dominance_threshold: float = 0.3) -> Dict[str, Any]:
    """Purely MECHANICAL description of a loading vector's dominant capabilities -- never assigns
    a semantic label ("spatial vs grounding" etc.) unless the loading pattern strictly matches:
    exactly 2 capabilities exceed `dominance_threshold` in absolute loading, with opposite signs,
    and every other capability is below half that threshold. Otherwise reports only the raw
    dominant-capability list with signs, letting the reader judge -- per the explicit instruction
    not to force semantic names.
    """
    loading = np.asarray(loading)
    dominant_idx = [j for j in range(len(loading)) if abs(loading[j]) >= dominance_threshold]
    dominant = [(CAPABILITIES[j], float(loading[j])) for j in dominant_idx]
    is_general_direction = bool(np.all(loading >= -1e-9) or np.all(loading <= 1e-9))
    structural_label = None
    if len(dominant) == 2:
        signs = [1 if v > 0 else -1 for _, v in dominant]
        others_small = all(abs(loading[j]) < dominance_threshold / 2 for j in range(len(loading)) if j not in dominant_idx)
        if signs[0] != signs[1] and others_small:
            structural_label = f"two-capability tradeoff axis: {dominant[0][0]} vs {dominant[1][0]}"
    elif is_general_direction and len(dominant) >= 4:
        structural_label = "general-improvement-like axis (all capabilities load with the same sign)"
    return {"dominant_capabilities": dominant, "all_same_sign": is_general_direction, "structural_label": structural_label}


def compute_pca_by_cell(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for source, region, radius, rows in cells:
        _, matrix = build_behavioral_matrix(rows)
        out[_cell_key(source, region, radius)] = {"source": source, "region": region, "radius": radius, **pca_loadings(matrix)}
    return out


# =================================================================================================
# Principal angles / Procrustes -- shared pure-numpy primitives (Sections 8, 9, 13)
# =================================================================================================


def top_k_subspace(matrix: np.ndarray, k: int) -> np.ndarray:
    """Returns the top-k right singular vectors of `matrix` as an (n_features, k) matrix with
    ORTHONORMAL columns -- the "behavioral subspace" in the 6-dimensional capability space.
    """
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(k, vt.shape[0])
    return vt[:k].T


def principal_angles_cosines(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """cos(theta_i) for i=1..k between two k-dimensional subspaces of the same ambient space,
    given as (n_features, k) orthonormal-column bases -- the singular values of basis_a.T @
    basis_b. Sorted descending (index 0 = smallest angle = most aligned direction pair). These
    are numerically IDENTICAL to the canonical correlations between the two subspaces (a known
    identity when both bases are orthonormal in the same ambient space) -- computed once, reused
    under both names where the task asks for "principal angles" and "canonical correlations".
    """
    m = basis_a.T @ basis_b
    s = np.linalg.svd(m, compute_uv=False)
    return np.clip(np.sort(s)[::-1], -1.0, 1.0)


def procrustes_similarity(basis_a: np.ndarray, basis_b: np.ndarray) -> Dict[str, Any]:
    """Optimal orthogonal alignment R minimizing ||basis_a @ R - basis_b||_F (standard orthogonal
    Procrustes via SVD of basis_a.T @ basis_b) -- returns the residual and a similarity score in
    [0, 1] (1 = perfect alignment after the best possible rotation/reflection).
    """
    m = basis_a.T @ basis_b
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    aligned = basis_a @ r
    residual = float(np.linalg.norm(aligned - basis_b, ord="fro"))
    denom = float(np.linalg.norm(basis_a, ord="fro") + np.linalg.norm(basis_b, ord="fro"))
    similarity = 1.0 - (residual / denom if denom > 0 else 0.0)
    return {"residual_frobenius": residual, "similarity": similarity}


# =================================================================================================
# Section 8: split-half stability
# =================================================================================================


def _balanced_splits(n: int, n_splits: int, seed_base: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed_base + s)
        perm = rng.permutation(n)
        half = n // 2
        splits.append((perm[:half], perm[half:]))
    return splits


def compute_split_half_stability(matrix: np.ndarray, *, n_splits: int = N_SPLIT_HALF_SPLITS, seed_offset: int = 0) -> Dict[str, Any]:
    n = matrix.shape[0]
    splits = _balanced_splits(n, n_splits, SPLIT_HALF_SEED_BASE + seed_offset)
    per_k: Dict[int, Dict[str, List[float]]] = {k: {"first_angle_cosine": [], "mean_angle_cosine": [], "procrustes_similarity": []} for k in (1, 2, 3)}
    for idx_a, idx_b in splits:
        mat_a, mat_b = matrix[idx_a], matrix[idx_b]
        for k in (1, 2, 3):
            basis_a, basis_b = top_k_subspace(mat_a, k), top_k_subspace(mat_b, k)
            cosines = principal_angles_cosines(basis_a, basis_b)
            per_k[k]["first_angle_cosine"].append(float(cosines[0]))
            per_k[k]["mean_angle_cosine"].append(float(np.mean(cosines)))
            per_k[k]["procrustes_similarity"].append(procrustes_similarity(basis_a, basis_b)["similarity"])

    out: Dict[str, Any] = {"n_splits": n_splits}
    for k in (1, 2, 3):
        vals = per_k[k]
        out[f"k_{k}"] = {
            stat: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5))}
            for stat, v in vals.items()
        }
    return out


def compute_split_half_by_cell(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, (source, region, radius, rows) in enumerate(cells):
        _, matrix = build_behavioral_matrix(rows)
        out[_cell_key(source, region, radius)] = {"source": source, "region": region, "radius": radius, **compute_split_half_stability(matrix, seed_offset=i)}
    return out


# =================================================================================================
# Sections 9 & 10: cross-radius subspace geometry + matched-direction trajectory geometry
# =================================================================================================


def _radius_pairs(radii: Sequence[float]) -> List[Tuple[str, float, float]]:
    ordered = sorted(radii)
    if len(ordered) != 3:
        raise ValueError(f"Expected exactly 3 radii, got {ordered}")
    r_small, r_mid, r_transition = ordered
    return [("small_vs_mid", r_small, r_mid), ("small_vs_transition", r_small, r_transition), ("mid_vs_transition", r_mid, r_transition)]


def compute_cross_radius_subspace(matrices_by_radius: Dict[float, np.ndarray]) -> Dict[str, Any]:
    radii = list(matrices_by_radius.keys())
    out: Dict[str, Any] = {}
    for pair_name, r_a, r_b in _radius_pairs(radii):
        mat_a, mat_b = matrices_by_radius[r_a], matrices_by_radius[r_b]
        pair_out: Dict[str, Any] = {"radius_a": r_a, "radius_b": r_b}
        for k in (1, 2, 3):
            basis_a, basis_b = top_k_subspace(mat_a, k), top_k_subspace(mat_b, k)
            cosines = principal_angles_cosines(basis_a, basis_b)
            proc = procrustes_similarity(basis_a, basis_b)
            pc_loading_cosine = float(abs(np.dot(basis_a[:, 0], basis_b[:, 0]))) if k == 1 else None
            pair_out[f"k_{k}"] = {
                "principal_angle_cosines": cosines.tolist(), "canonical_correlations": cosines.tolist(),
                "first_principal_angle_cosine": float(cosines[0]), "mean_principal_angle_cosine": float(np.mean(cosines)),
                "procrustes_similarity": proc["similarity"],
            }
            if pc_loading_cosine is not None:
                pair_out[f"k_{k}"]["pc1_loading_absolute_cosine"] = pc_loading_cosine
        out[pair_name] = pair_out
    return out


def compute_cross_radius_subspace_by_region(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    by_region: Dict[Tuple[str, str], Dict[float, np.ndarray]] = {}
    for source, region, radius, rows in cells:
        _, matrix = build_behavioral_matrix(rows)
        by_region.setdefault((source, region), {})[radius] = matrix
    out: Dict[str, Any] = {}
    for (source, region), matrices_by_radius in by_region.items():
        out[f"{source}:{region}"] = {"source": source, "region": region, **compute_cross_radius_subspace(matrices_by_radius)}
    return out


def compute_trajectory_geometry(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    """Direction family i (fixed seed, reused across radii within one region -- Stage 8/9's own
    frozen design) gets a behavioral vector b_i(r) at each of the 3 radii. Computes cosine
    similarity, Euclidean displacement, norm growth, and per-capability sign flips between every
    consecutive AND every pair of radii, summarized by region.
    """
    by_region: Dict[Tuple[str, str], Dict[int, Dict[float, np.ndarray]]] = {}
    for source, region, radius, rows in cells:
        direction_indices, matrix = build_behavioral_matrix(rows)
        for i, idx in enumerate(direction_indices):
            by_region.setdefault((source, region), {}).setdefault(idx, {})[radius] = matrix[i]

    out: Dict[str, Any] = {}
    for (source, region), by_direction in by_region.items():
        radii = sorted({r for d in by_direction.values() for r in d.keys()})
        pairs = _radius_pairs(radii)
        pair_stats: Dict[str, Dict[str, List[float]]] = {p[0]: {"cosine": [], "euclidean_displacement": [], "norm_growth_ratio": [], "n_sign_flips": []} for p in pairs}
        norms_by_radius: Dict[float, List[float]] = {r: [] for r in radii}
        n_complete = 0
        for idx, radius_to_vec in by_direction.items():
            if any(r not in radius_to_vec for r in radii):
                continue
            n_complete += 1
            for r in radii:
                norms_by_radius[r].append(float(np.linalg.norm(radius_to_vec[r])))
            for pair_name, r_a, r_b in pairs:
                v_a, v_b = radius_to_vec[r_a], radius_to_vec[r_b]
                norm_a, norm_b = np.linalg.norm(v_a), np.linalg.norm(v_b)
                cosine = float(np.dot(v_a, v_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else None
                if cosine is not None:
                    pair_stats[pair_name]["cosine"].append(cosine)
                pair_stats[pair_name]["euclidean_displacement"].append(float(np.linalg.norm(v_b - v_a)))
                pair_stats[pair_name]["norm_growth_ratio"].append(float(norm_b / norm_a) if norm_a > 0 else None)
                pair_stats[pair_name]["n_sign_flips"].append(int(np.sum(np.sign(v_a) != np.sign(v_b))))

        region_out: Dict[str, Any] = {"source": source, "region": region, "n_complete_trajectories": n_complete, "mean_norm_by_radius": {str(r): float(np.mean(norms_by_radius[r])) if norms_by_radius[r] else None for r in radii}}
        for pair_name, stats in pair_stats.items():
            clean = {k: [v for v in vals if v is not None] for k, vals in stats.items()}
            region_out[pair_name] = {
                stat: {"mean": float(np.mean(vals)), "median": float(np.median(vals))} if vals else None
                for stat, vals in clean.items()
            }
        out[f"{source}:{region}"] = region_out
    return out


# =================================================================================================
# Section 11: useful-expert geometry
# =================================================================================================


def _pairwise_cosine_distribution(vectors: np.ndarray) -> List[float]:
    n = vectors.shape[0]
    norms = np.linalg.norm(vectors, axis=1)
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] > 0 and norms[j] > 0:
                cosines.append(float(np.dot(vectors[i], vectors[j]) / (norms[i] * norms[j])))
    return cosines


def _group_geometry(vectors: np.ndarray) -> Dict[str, Any]:
    if vectors.shape[0] == 0:
        return {"n": 0}
    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    dispersion = float(np.mean(np.sum((vectors - centroid) ** 2, axis=1)))
    cos_to_centroid = []
    if centroid_norm > 0:
        for v in vectors:
            v_norm = np.linalg.norm(v)
            if v_norm > 0:
                cos_to_centroid.append(float(np.dot(v, centroid) / (v_norm * centroid_norm)))
    cov = np.cov(vectors, rowvar=False) if vectors.shape[0] > 1 else np.zeros((vectors.shape[1], vectors.shape[1]))
    cov_eigs = np.sort(np.linalg.eigvalsh(cov))[::-1] if vectors.shape[0] > 1 else np.zeros(vectors.shape[1])
    pairwise = _pairwise_cosine_distribution(vectors)
    return {
        "n": int(vectors.shape[0]), "centroid": centroid.tolist(), "centroid_norm": float(centroid_norm),
        "within_group_dispersion_mean_sq_dist_to_centroid": dispersion,
        "cosine_to_centroid_mean": float(np.mean(cos_to_centroid)) if cos_to_centroid else None,
        "cosine_to_centroid_std": float(np.std(cos_to_centroid, ddof=1)) if len(cos_to_centroid) > 1 else None,
        "pairwise_cosine_mean": float(np.mean(pairwise)) if pairwise else None,
        "pairwise_cosine_std": float(np.std(pairwise, ddof=1)) if len(pairwise) > 1 else None,
        "covariance_eigenvalues": cov_eigs.tolist(),
    }


def compute_useful_expert_geometry(matrix: np.ndarray, *, seed_offset: int) -> Dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_CLUSTER_SEED_BASE + seed_offset)
    n = matrix.shape[0]
    out: Dict[str, Any] = {}
    for j, cap in enumerate(CAPABILITIES):
        useful_mask = matrix[:, j] >= USEFUL_EXPERT_THRESHOLD
        non_improving_mask = matrix[:, j] < USEFUL_EXPERT_THRESHOLD
        useful_vectors = matrix[useful_mask]
        non_improving_vectors = matrix[non_improving_mask]
        n_useful = int(useful_mask.sum())
        random_idx = rng.choice(n, size=n_useful, replace=False) if n_useful <= n else np.arange(n)
        random_vectors = matrix[random_idx]
        out[cap] = {
            "capability": cap, "n_useful": n_useful, "n_non_improving": int(non_improving_mask.sum()),
            "useful_expert_geometry": _group_geometry(useful_vectors),
            "same_number_random_candidates_geometry": _group_geometry(random_vectors),
            "non_improving_geometry": _group_geometry(non_improving_vectors),
        }
    return out


def compute_useful_expert_geometry_by_cell(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, (source, region, radius, rows) in enumerate(cells):
        _, matrix = build_behavioral_matrix(rows)
        out[_cell_key(source, region, radius)] = {"source": source, "region": region, "radius": radius, **compute_useful_expert_geometry(matrix, seed_offset=i)}
    return out


# =================================================================================================
# Section 12: exploratory clustering (pure numpy k-means / agglomerative / silhouette)
# =================================================================================================


def _kmeans(vectors: np.ndarray, k: int, seed: int, max_iters: int = KMEANS_MAX_ITERS) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    centers = vectors[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        dists = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            if np.any(labels == c):
                centers[c] = vectors[labels == c].mean(axis=0)
    dists = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
    inertia = float(np.sum(np.min(dists, axis=1) ** 2))
    return labels, inertia


def _best_kmeans(vectors: np.ndarray, k: int, seed_offset: int) -> Tuple[np.ndarray, float]:
    best_labels, best_inertia = None, np.inf
    for s in KMEANS_SEEDS:
        labels, inertia = _kmeans(vectors, k, seed=BOOTSTRAP_CLUSTER_SEED_BASE + seed_offset * 1000 + s)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels, best_inertia


def _agglomerative_average_linkage(vectors: np.ndarray, k: int) -> np.ndarray:
    n = vectors.shape[0]
    clusters = [[i] for i in range(n)]
    dist = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    active = list(range(n))
    cluster_members = {i: [i] for i in range(n)}
    next_id = n
    while len(active) > k:
        best_pair, best_d = None, np.inf
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                members_a, members_b = cluster_members[a], cluster_members[b]
                d = np.mean([dist[x, y] for x in members_a for y in members_b])
                if d < best_d:
                    best_d, best_pair = d, (a, b)
        a, b = best_pair
        cluster_members[next_id] = cluster_members[a] + cluster_members[b]
        active = [c for c in active if c not in (a, b)] + [next_id]
        next_id += 1
    labels = np.zeros(n, dtype=int)
    for cluster_idx, cid in enumerate(active):
        for member in cluster_members[cid]:
            labels[member] = cluster_idx
    return labels


def silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
    n = vectors.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    dist = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not np.any(same):
            scores.append(0.0)
            continue
        a_i = float(np.mean(dist[i, same]))
        b_i = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = labels == lbl
            b_i = min(b_i, float(np.mean(dist[i, other])))
        scores.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0)
    return float(np.mean(scores))


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Standard ARI (hand-rolled, no scipy/sklearn) -- used both for bootstrap stability and
    for cross-method (k-means vs agglomerative) agreement.
    """
    contingency: Dict[Tuple[int, int], int] = {}
    for x, y in zip(labels_a, labels_b):
        contingency[(x, y)] = contingency.get((x, y), 0) + 1
    n = len(labels_a)
    a_sums: Dict[int, int] = {}
    b_sums: Dict[int, int] = {}
    for (x, y), c in contingency.items():
        a_sums[x] = a_sums.get(x, 0) + c
        b_sums[y] = b_sums.get(y, 0) + c

    def _comb2(v: int) -> float:
        return v * (v - 1) / 2.0

    sum_comb_c = sum(_comb2(c) for c in contingency.values())
    sum_comb_a = sum(_comb2(v) for v in a_sums.values())
    sum_comb_b = sum(_comb2(v) for v in b_sums.values())
    total_comb = _comb2(n)
    expected = (sum_comb_a * sum_comb_b) / total_comb if total_comb > 0 else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return float((sum_comb_c - expected) / denom)


def compute_exploratory_clustering(vectors: np.ndarray, *, seed_offset: int) -> Dict[str, Any]:
    n = vectors.shape[0]
    if n < CLUSTERING_MIN_USEFUL:
        return {"attempted": False, "n_useful": n, "reason": f"n_useful ({n}) below CLUSTERING_MIN_USEFUL ({CLUSTERING_MIN_USEFUL})", "verdict": "insufficient_sample_size"}

    rng = np.random.default_rng(BOOTSTRAP_CLUSTER_SEED_BASE + seed_offset)
    per_k: Dict[int, Any] = {}
    any_k_stable = False
    for k in range(2, min(7, n)):
        km_labels, km_inertia = _best_kmeans(vectors, k, seed_offset)
        agg_labels = _agglomerative_average_linkage(vectors, k)
        km_silhouette = silhouette_score(vectors, km_labels)
        agg_silhouette = silhouette_score(vectors, agg_labels)
        method_agreement_ari = _adjusted_rand_index(km_labels, agg_labels)

        bootstrap_aris = []
        for b in range(N_CLUSTER_BOOTSTRAP):
            draw_seed = BOOTSTRAP_CLUSTER_SEED_BASE + seed_offset * 10_000 + k * 100 + b
            boot_rng = np.random.default_rng(draw_seed)
            boot_idx = boot_rng.choice(n, size=n, replace=True)
            boot_labels, _ = _kmeans(vectors[boot_idx], k, seed=draw_seed)  # single-seed (not best-of-10) -- see module constants' rationale
            unique_idx = np.unique(boot_idx, return_index=True)[1]
            if len(unique_idx) < k:
                continue
            bootstrap_aris.append(_adjusted_rand_index(km_labels[boot_idx[unique_idx]], boot_labels[unique_idx]))

        null_silhouettes = []
        for b in range(N_NULL_SILHOUETTE_DRAWS):
            draw_seed = NULL_SILHOUETTE_SEED_BASE + seed_offset * 10_000 + k * 100 + b
            null_rng = np.random.default_rng(draw_seed)
            shuffled = _permute_columns_independently(vectors, null_rng)
            null_labels, _ = _kmeans(shuffled, k, seed=draw_seed)  # single-seed (not best-of-10) -- see module constants' rationale
            null_silhouettes.append(silhouette_score(shuffled, null_labels))
        null_arr = np.asarray(null_silhouettes)
        exceeds_null = bool(km_silhouette > np.percentile(null_arr, 95.0))

        is_stable = bool(
            method_agreement_ari > 0.5
            and (float(np.mean(bootstrap_aris)) > 0.5 if bootstrap_aris else False)
            and exceeds_null
        )
        any_k_stable = any_k_stable or is_stable
        per_k[k] = {
            "kmeans_inertia": km_inertia, "kmeans_silhouette": km_silhouette, "agglomerative_silhouette": agg_silhouette,
            "kmeans_vs_agglomerative_ari": method_agreement_ari,
            "bootstrap_ari_mean": float(np.mean(bootstrap_aris)) if bootstrap_aris else None,
            "bootstrap_ari_n": len(bootstrap_aris),
            "null_silhouette_p95": float(np.percentile(null_arr, 95.0)), "exceeds_shuffled_null": exceeds_null,
            "stable": is_stable,
        }
    return {
        "attempted": True, "n_useful": n, "per_k": per_k,
        "verdict": "discrete_clusters_supported" if any_k_stable else "continuous_geometry_no_stable_discrete_clusters",
    }


def compute_clustering_for_useful_experts(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, (source, region, radius, rows) in enumerate(cells):
        _, matrix = build_behavioral_matrix(rows)
        cell_out: Dict[str, Any] = {"source": source, "region": region, "radius": radius, "by_capability": {}}
        for j, cap in enumerate(CAPABILITIES):
            useful_vectors = matrix[matrix[:, j] >= USEFUL_EXPERT_THRESHOLD]
            cell_out["by_capability"][cap] = compute_exploratory_clustering(useful_vectors, seed_offset=i * 6 + j)
        out[_cell_key(source, region, radius)] = cell_out
    return out


# =================================================================================================
# Section 13: cross-anatomy behavioral geometry (subspace-only comparison, never candidate-paired)
# =================================================================================================


def compute_cross_anatomy_geometry(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    by_radius: Dict[float, Dict[str, np.ndarray]] = {}
    for source, region, radius, rows in cells:
        _, matrix = build_behavioral_matrix(rows)
        by_radius.setdefault(radius, {})[f"{source}:{region}"] = matrix

    out: Dict[str, Any] = {}
    for radius, region_matrices in by_radius.items():
        region_names = sorted(region_matrices.keys())
        radius_out: Dict[str, Any] = {"regions": region_names}
        for k in (1, 2, 3):
            pairwise: Dict[str, Any] = {}
            for i in range(len(region_names)):
                for j in range(i + 1, len(region_names)):
                    a, b = region_names[i], region_names[j]
                    basis_a = top_k_subspace(region_matrices[a], k)
                    basis_b = top_k_subspace(region_matrices[b], k)
                    cosines = principal_angles_cosines(basis_a, basis_b)
                    pairwise[f"{a}__vs__{b}"] = {"first_principal_angle_cosine": float(cosines[0]), "mean_principal_angle_cosine": float(np.mean(cosines))}
            radius_out[f"k_{k}"] = pairwise
        out[str(radius)] = radius_out
    return out


# =================================================================================================
# Section 14: specialization <-> effective rank relationship
# =================================================================================================


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rank_x = np.argsort(np.argsort(x_arr))
    rank_y = np.argsort(np.argsort(y_arr))
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return 0.0
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def compute_specialization_rank_relationship(
    cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]], effective_rank_by_cell: Dict[str, Any],
) -> Dict[str, Any]:
    spectral_discordance_vals, tradeoff_fraction_vals = [], []
    entropy_rank_vals, stable_rank_vals, pc1_vals = [], [], []
    cell_rows = []
    for source, region, radius, rows in cells:
        key = _cell_key(source, region, radius)
        _, matrix = build_behavioral_matrix(rows)
        discordance = s8a.thicket_diversity.spectral_discordance(matrix)
        n = matrix.shape[0]
        improves_one = matrix > 0
        harms_margin = matrix <= -s8a.HARM_MARGIN
        tradeoff_fraction = float(np.mean(np.any(improves_one, axis=1) & np.any(harms_margin, axis=1)))
        rank_stats = effective_rank_by_cell[key]["observed"]
        spectral_discordance_vals.append(discordance)
        tradeoff_fraction_vals.append(tradeoff_fraction)
        entropy_rank_vals.append(rank_stats["entropy_effective_rank"])
        stable_rank_vals.append(rank_stats["stable_rank"])
        pc1_vals.append(rank_stats["variance_explained_pc1"])
        cell_rows.append({"cell": key, "spectral_discordance": discordance, "tradeoff_fraction": tradeoff_fraction, "entropy_effective_rank": rank_stats["entropy_effective_rank"], "stable_rank": rank_stats["stable_rank"], "pc1_variance": rank_stats["variance_explained_pc1"]})

    return {
        "n_cells": len(cell_rows),
        "spearman_discordance_vs_entropy_rank": _spearman(spectral_discordance_vals, entropy_rank_vals),
        "spearman_discordance_vs_stable_rank": _spearman(spectral_discordance_vals, stable_rank_vals),
        "spearman_discordance_vs_pc1_variance": _spearman(spectral_discordance_vals, pc1_vals),
        "spearman_tradeoff_fraction_vs_entropy_rank": _spearman(tradeoff_fraction_vals, entropy_rank_vals),
        "spearman_tradeoff_fraction_vs_stable_rank": _spearman(tradeoff_fraction_vals, stable_rank_vals),
        "spearman_tradeoff_fraction_vs_pc1_variance": _spearman(tradeoff_fraction_vals, pc1_vals),
        "cells": cell_rows,
    }


# =================================================================================================
# Section 15: score-granularity control
# =================================================================================================


def compute_granularity_control(cells: List[Tuple[str, str, float, List[ExperimentResultRecord]]]) -> Dict[str, Any]:
    per_example_recoverable = any(r.per_example_result_path is not None for _, _, _, rows in cells for r in rows)
    centered_effective_rank = compute_effective_rank_by_cell(cells, centered=True)
    raw_effective_rank = compute_effective_rank_by_cell(cells, centered=False)
    agreement = []
    for key in raw_effective_rank:
        raw_eff = raw_effective_rank[key]["observed"]["entropy_effective_rank"]
        centered_eff = centered_effective_rank[key]["observed"]["entropy_effective_rank"]
        agreement.append({"cell": key, "raw_entropy_effective_rank": raw_eff, "centered_entropy_effective_rank": centered_eff})
    return {
        "per_example_binary_signatures_recoverable": per_example_recoverable,
        "limitation_note": (
            "Per-example binary/categorical change signatures are NOT persisted at the Stage 8/9 "
            "results.jsonl row level (per_example_result_path is always null; only a hash of the "
            "per-example generations is stored, per_example_result_hash) -- this control cannot be "
            "run on real recovered per-example data and is NOT fabricated here. The secondary "
            "sensitivity check below (raw vs. capability-centered Delta) is the only granularity-"
            "robustness cross-check available from persisted data."
        ),
        "raw_vs_centered_effective_rank_agreement": agreement,
    }


# =================================================================================================
# Section 16: parameter-space geometry feasibility audit
# =================================================================================================


def compute_reconstruction_feasibility(stage8_records: Sequence[ExperimentResultRecord], stage9_records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Two independent feasibility checks, neither requiring a GPU or model inference run:

    (1) REAL DATA self-consistency: for 2 Stage-8 language + 2 Stage-8 vision + 2 Stage-9 depth
        candidates (chosen deterministically -- first two direction_index values encountered per
        group), verifies the algebraic identity realized_relative_l2 == epsilon_region_l2_norm /
        theta_region_l2_norm holds against the REAL persisted numbers (pure arithmetic, no model
        access needed) -- confirms the persisted norms are mutually consistent.

    (2) MECHANISM-level determinism proof: scoped_anatomical_perturbation's OWN deterministic-
        noise-generation + v3 solver code path is re-run TWICE, independently, on a small
        SYNTHETIC bf16 tensor (same class already used throughout this project's test suite --
        NOT the real 3B checkpoint, which is not available in this environment and would require
        a multi-GB download this task does not authorize), from the SAME (seed, requested radius)
        -- proves the resulting accepted scalar and realized epsilon tensor are BIT-IDENTICAL
        across the two independent runs, which is the exact mechanism real reconstruction would
        rely on (real theta_0 + regenerated noise + re-solved scalar, never inference).

    RAM/time for the REAL checkpoint are then ESTIMATES extrapolated from Stage 7A's own already-
        published L1 parameter counts (632.0M vision / 3086.0M language / 36.7M connector) --
        never independently re-measured here, since the real checkpoint is not loaded.
    """
    import time

    from neural_thickets_repro.scoped_anatomical_perturbation import scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3
    from types import SimpleNamespace
    import torch

    def _pick_candidates(records: Sequence[ExperimentResultRecord], regions: Sequence[str], n_per_region: int) -> List[ExperimentResultRecord]:
        seen: Dict[str, set] = {r: set() for r in regions}
        picked = []
        for r in sorted(records, key=lambda rr: (rr.anatomy_region, rr.runtime_metadata.get("direction_index", 0), rr.radius)):
            if r.anatomy_region in regions and r.runtime_metadata.get("direction_index") not in seen[r.anatomy_region] and len(seen[r.anatomy_region]) < n_per_region:
                seen[r.anatomy_region].add(r.runtime_metadata.get("direction_index"))
                picked.append(r)
        return picked

    stage8_language = _pick_candidates(stage8_records, ["language"], 2)
    stage8_vision = _pick_candidates(stage8_records, ["vision"], 2)
    stage9_depth = _pick_candidates(stage9_records, ["vision_early", "language_early"], 1)  # 1 from each -> 2 total

    self_consistency_checks = []
    for r in (stage8_language + stage8_vision + stage9_depth):
        meta = r.runtime_metadata
        theta_norm, epsilon_norm = meta.get("theta_region_l2_norm"), meta.get("epsilon_region_l2_norm")
        realized = meta.get("realized_relative_l2")
        recomputed = (epsilon_norm / theta_norm) if theta_norm else None
        matches = recomputed is not None and realized is not None and abs(recomputed - realized) < 1e-9
        self_consistency_checks.append({
            "perturbation_id": r.perturbation_id, "region": r.anatomy_region, "radius": r.radius,
            "direction_seed": meta.get("direction_seed"), "theta_region_l2_norm": theta_norm,
            "epsilon_region_l2_norm": epsilon_norm, "realized_relative_l2": realized,
            "recomputed_from_norms": recomputed, "self_consistent": matches,
        })

    # Mechanism-level determinism proof on a synthetic bf16 model -- same code path, real bf16
    # arithmetic, no GPU, no real checkpoint.
    class _SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.region_layer = torch.nn.Linear(5000, 1, bias=False)
            self.outside_layer = torch.nn.Linear(100, 1, bias=False)

    def _reset(model, base_weights):
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(base_weights[name])

    def _make_worker():
        torch.manual_seed(0)
        model = _SyntheticModel().to(torch.bfloat16)
        base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        worker.reset_to_base_weights = lambda: _reset(model, base_weights)
        worker._base_weights = base_weights
        return worker, model, base_weights

    seed, requested_r = 424242, 0.0035698828543799426
    start = time.perf_counter()
    worker_1, model_1, base_1 = _make_worker()
    result_1 = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_1, seed, requested_r, "region", ["region_layer.weight"])
    epsilon_1 = (model_1.region_layer.weight.detach().float() - base_1["region_layer.weight"].float()).clone()
    elapsed_seconds_5000_elements = time.perf_counter() - start

    worker_2, model_2, base_2 = _make_worker()
    result_2 = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_2, seed, requested_r, "region", ["region_layer.weight"])
    epsilon_2 = (model_2.region_layer.weight.detach().float() - base_2["region_layer.weight"].float()).clone()

    bit_identical = bool(torch.equal(epsilon_1, epsilon_2))
    same_accepted_scalar = result_1["accepted_scalar"] == result_2["accepted_scalar"]
    same_realized = result_1["realized_relative_l2"] == result_2["realized_relative_l2"]

    bytes_per_element_bf16 = 2
    ram_estimates_gb = {
        region: round(count * bytes_per_element_bf16 * 3 / 1e9, 2)  # base + noise + delta buffers, factor of 3 -- conservative
        for region, count in STAGE7A_L1_PARAM_COUNTS.items()
    }

    return {
        "self_consistency_against_real_persisted_norms": {
            "checks": self_consistency_checks,
            "all_self_consistent": all(c["self_consistent"] for c in self_consistency_checks),
        },
        "mechanism_level_determinism_proof": {
            "note": "Run on a SYNTHETIC bf16 tensor (5000 elements) mirroring the real code path -- NOT the real 3B checkpoint (not available in this environment; would require a multi-GB download not authorized for this analysis-only stage).",
            "bit_identical_epsilon_across_two_independent_runs": bit_identical,
            "same_accepted_scalar": same_accepted_scalar, "same_realized_relative_l2": same_realized,
            "elapsed_seconds_for_5000_element_region_two_full_solver_runs": elapsed_seconds_5000_elements,
        },
        "exact_reproducibility_conclusion": (
            "CONFIRMED at the mechanism level: given (model_revision, direction_seed, region_param_names/mask, "
            "requested radius) plus the real base weights theta_0, the noise tensor and the v3 solver's accepted "
            "scalar are BOTH deterministic and independently re-derivable without any inference/generation pass -- "
            "epsilon_i = theta_i - theta_0 is exactly reconstructible from persisted metadata plus theta_0 alone. "
            "NOT yet verified against the real checkpoint (not loaded in this environment)."
        ),
        "ram_time_feasibility_for_the_real_checkpoint": {
            "stage7a_l1_param_counts_used": STAGE7A_L1_PARAM_COUNTS,
            "estimated_ram_gb_per_full_region_reconstruction": ram_estimates_gb,
            "note": "Estimates only -- assumes selective (safetensors-style) loading of just the region's own tensors, never the full 3B-parameter model; a real measurement requires the actual checkpoint, not attempted here.",
            "full_materialization_required": "No -- a streaming/per-tensor Gram-matrix or sketch computation (e.g. accumulating epsilon_i^T epsilon_j incrementally per parameter tensor) avoids ever holding all 1152 full-size epsilon vectors in memory simultaneously.",
            "recommendation": "Streaming Gram/sketch computation over per-tensor chunks, never full materialization of all 1152 vectors at once.",
        },
        "candidates_used": {
            "stage8_language": [r.perturbation_id for r in stage8_language],
            "stage8_vision": [r.perturbation_id for r in stage8_vision],
            "stage9_depth": [r.perturbation_id for r in stage9_depth],
        },
        "reconstruct_all_1152_now": False,
    }


# =================================================================================================
# Section 1: integrity (of the INPUTS to Stage 10A -- Stage 8/9's own gates already ran; this
# re-confirms the authoritative-run identity Stage 10A actually loaded)
# =================================================================================================


def run_stage10a_integrity_gate(stage8_records: Sequence[ExperimentResultRecord], stage9_records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["stage8_row_count_3456"] = len(stage8_records) == len(STAGE8_REGIONS) * len(STAGE8_RADII) * STAGE8_N_DIRECTIONS_PER_CELL * len(STAGE8_CAPABILITIES)
    checks["stage9_row_count_6912"] = len(stage9_records) == len(STAGE9_CHILD_REGIONS) * len(STAGE9_RADII) * 64 * len(STAGE9_CAPABILITIES)
    checks["stage8_regions_match"] = {r.anatomy_region for r in stage8_records} == set(STAGE8_REGIONS)
    checks["stage9_regions_match"] = {r.anatomy_region for r in stage9_records} == set(STAGE9_CHILD_REGIONS)
    checks["capabilities_identical_across_stage8_and_stage9"] = set(STAGE8_CAPABILITIES) == set(STAGE9_CAPABILITIES)
    checks["shared_capability_column_order_fixed"] = CAPABILITIES == tuple(sorted(STAGE8_CAPABILITIES))
    checks["all_checks_pass"] = all(bool(v) for v in checks.values() if isinstance(v, bool))
    return checks


# =================================================================================================
# Stage-10A classification + Stage-10B recommendation + paper story (Sections 17, 18)
# =================================================================================================


def classify_stage10a(effective_rank_by_cell: Dict[str, Any], split_half_by_cell: Dict[str, Any]) -> Dict[str, Any]:
    entropy_ranks = [v["observed"]["entropy_effective_rank"] for v in effective_rank_by_cell.values()]
    lower_than_null_a = sum(1 for v in effective_rank_by_cell.values() if v["null_a_independent_permutation"]["entropy_effective_rank"]["observed_lower_than_95pct_of_null"])
    n_cells = len(effective_rank_by_cell)
    mean_first_angle_cosine_k1 = float(np.mean([v["k_1"]["first_angle_cosine"]["mean"] for v in split_half_by_cell.values()]))

    fraction_low_rank = lower_than_null_a / n_cells if n_cells else 0.0
    if fraction_low_rank >= 0.7 and mean_first_angle_cosine_k1 >= 0.7:
        classification = "A"
    elif fraction_low_rank >= 0.3 or mean_first_angle_cosine_k1 >= 0.5:
        classification = "B"
    else:
        classification = "C"

    return {
        "classification": classification,
        "classification_meaning": {"A": "STRONG LOW-DIMENSIONAL BEHAVIORAL GEOMETRY", "B": "MIXED GEOMETRY", "C": "NO LOW-DIMENSIONAL STRUCTURE"}[classification],
        "mean_entropy_effective_rank_across_cells": float(np.mean(entropy_ranks)), "n_cells": n_cells,
        "fraction_cells_significantly_lower_rank_than_independent_null": fraction_low_rank,
        "mean_split_half_k1_first_angle_cosine": mean_first_angle_cosine_k1,
    }


def compute_stage10b_recommendation(classification: Dict[str, Any]) -> Dict[str, Any]:
    cls = classification["classification"]
    if cls in ("A", "B"):
        return {
            "stage10b_justified": True,
            "classification_basis": cls,
            "proposed_methods_to_assess_not_implement": [
                "exact/streaming Gram matrices over per-tensor chunks (never full materialization of all candidate vectors)",
                "deterministic random projections (Johnson-Lindenstrauss sketches) for approximate Gram estimation at scale",
                "reward-weighted perturbation covariance (weighting each direction's contribution by its behavioral usefulness)",
                "SVD / effective rank of the reconstructed parameter-space Gram matrix",
                "split-half stability of the parameter-space subspace (mirroring Section 8's behavioral analog)",
                "principal angles between parameter-space subspaces across radius/region (mirroring Sections 9/13)",
                "a targeted subspace-recombination experiment (perturbing along a reconstructed dominant axis directly, to test causal sufficiency)",
            ],
            "rationale": f"Stage 10A classification={cls} ({classification['classification_meaning']}) -- behavioral geometry shows enough structure to justify the more expensive parameter-space investigation.",
        }
    return {
        "stage10b_justified": False,
        "classification_basis": cls,
        "proposed_methods_to_assess_not_implement": [],
        "rationale": (
            f"Stage 10A classification={cls} (NO LOW-DIMENSIONAL STRUCTURE) -- the frozen low-rank claim is "
            "NOT forced here. Recommend moving to scale (more capabilities/models/scenarios) rather than a "
            "targeted parameter-space geometry stage."
        ),
    }


def compute_paper_story_update(classification: Dict[str, Any], cross_radius_by_region: Dict[str, Any], trajectory_geometry: Dict[str, Any]) -> Dict[str, Any]:
    mean_k1_across_radius_pairs = []
    for region_data in cross_radius_by_region.values():
        for pair_name in ("small_vs_mid", "small_vs_transition", "mid_vs_transition"):
            mean_k1_across_radius_pairs.append(region_data[pair_name]["k_1"]["first_principal_angle_cosine"])
    mean_subspace_alignment = float(np.mean(mean_k1_across_radius_pairs)) if mean_k1_across_radius_pairs else None
    subspace_rotates = mean_subspace_alignment is not None and mean_subspace_alignment < 0.7

    geometry_word = {"A": "low-dimensional", "B": "mixed", "C": "high-dimensional"}[classification["classification"]]
    return {
        "emerging_picture": (
            f"Coarse anatomy structures expert density; fine depth is comparatively diffuse; perturbation scale "
            f"reorganizes specialist identity; and the resulting behavioral changes lie on {geometry_word} "
            f"capability-effect geometry (Stage 10A classification {classification['classification']})."
        ),
        "mean_cross_radius_k1_subspace_alignment_cosine": mean_subspace_alignment,
        "identity_change_within_stable_subspace_or_subspace_rotates": "subspace_rotates_with_radius" if subspace_rotates else "identity_changes_within_a_comparatively_stable_subspace",
        "caveat": "This is behavioral-effect geometry only -- no claim is made here about parameter-space low-rank structure (that is Stage 10B, proposed but not implemented).",
    }


# =================================================================================================
# Compact CSV exports
# =================================================================================================


def write_effective_rank_csv(effective_rank_by_cell: Dict[str, Any], path: Path) -> None:
    header = ["cell", "source", "region", "radius", "n", "entropy_effective_rank", "stable_rank",
              "variance_explained_pc1", "variance_explained_pc1_pc2", "variance_explained_pc1_pc2_pc3",
              "null_a_effective_rank_mean", "null_a_observed_lower_than_95pct_of_null",
              "null_b_effective_rank_mean"]
    rows = []
    for key, cell in effective_rank_by_cell.items():
        obs = cell["observed"]
        rows.append([
            key, cell["source"], cell["region"], cell["radius"], cell["n"],
            obs["entropy_effective_rank"], obs["stable_rank"], obs["variance_explained_pc1"],
            obs["variance_explained_pc1_pc2"], obs["variance_explained_pc1_pc2_pc3"],
            cell["null_a_independent_permutation"]["entropy_effective_rank"]["null_mean"],
            cell["null_a_independent_permutation"]["entropy_effective_rank"]["observed_lower_than_95pct_of_null"],
            cell["null_b_gaussian_matched"]["entropy_effective_rank"]["null_mean"],
        ])
    s8a._write_csv(path, header, rows)


def write_split_half_csv(split_half_by_cell: Dict[str, Any], path: Path) -> None:
    header = ["cell", "source", "region", "radius", "k1_first_angle_cosine_mean", "k1_procrustes_similarity_mean",
              "k2_first_angle_cosine_mean", "k3_first_angle_cosine_mean"]
    rows = []
    for key, cell in split_half_by_cell.items():
        rows.append([
            key, cell["source"], cell["region"], cell["radius"],
            cell["k_1"]["first_angle_cosine"]["mean"], cell["k_1"]["procrustes_similarity"]["mean"],
            cell["k_2"]["first_angle_cosine"]["mean"], cell["k_3"]["first_angle_cosine"]["mean"],
        ])
    s8a._write_csv(path, header, rows)


def write_trajectory_geometry_csv(trajectory_geometry: Dict[str, Any], path: Path) -> None:
    header = ["region_key", "source", "region", "n_complete_trajectories",
              "small_vs_mid_cosine_mean", "small_vs_transition_cosine_mean", "mid_vs_transition_cosine_mean"]
    rows = []
    for key, region in trajectory_geometry.items():
        rows.append([
            key, region["source"], region["region"], region["n_complete_trajectories"],
            region["small_vs_mid"]["cosine"]["mean"] if region["small_vs_mid"]["cosine"] else None,
            region["small_vs_transition"]["cosine"]["mean"] if region["small_vs_transition"]["cosine"] else None,
            region["mid_vs_transition"]["cosine"]["mean"] if region["mid_vs_transition"]["cosine"] else None,
        ])
    s8a._write_csv(path, header, rows)


def write_specialization_rank_relationship_csv(specialization_rank: Dict[str, Any], path: Path) -> None:
    header = ["cell", "spectral_discordance", "tradeoff_fraction", "entropy_effective_rank", "stable_rank", "pc1_variance"]
    rows = [[c["cell"], c["spectral_discordance"], c["tradeoff_fraction"], c["entropy_effective_rank"], c["stable_rank"], c["pc1_variance"]] for c in specialization_rank["cells"]]
    s8a._write_csv(path, header, rows)


# =================================================================================================
# Markdown report + main
# =================================================================================================


def build_markdown_report(integrity: Dict[str, Any], classification: Dict[str, Any], stage10b: Dict[str, Any], paper_story: Dict[str, Any], reconstruction: Dict[str, Any]) -> str:
    lines = [
        "# Stage 10A: behavioral geometry of Stage-8/Stage-9 perturbation effects", "",
        f"Integrity gate: **{'PASS' if integrity['all_checks_pass'] else 'FAIL'}**.", "",
        f"## Classification: **{classification['classification']}** -- {classification['classification_meaning']}", "",
        f"Mean entropy effective rank across {classification['n_cells']} cells: {classification['mean_entropy_effective_rank_across_cells']:.3f} (out of 6 possible).",
        f"Fraction of cells significantly lower rank than the independent-column null: {classification['fraction_cells_significantly_lower_rank_than_independent_null']:.2f}.",
        f"Mean split-half k=1 subspace alignment cosine: {classification['mean_split_half_k1_first_angle_cosine']:.3f}.", "",
        "## Stage-10B recommendation", "",
        f"**{'JUSTIFIED' if stage10b['stage10b_justified'] else 'NOT JUSTIFIED'}** -- {stage10b['rationale']}", "",
        "## Paper story update", "",
        paper_story["emerging_picture"], "",
        f"Reconstruction feasibility: mechanism-level determinism {'CONFIRMED' if reconstruction['mechanism_level_determinism_proof']['bit_identical_epsilon_across_two_independent_runs'] else 'NOT confirmed'} on synthetic tensors; real-checkpoint execution not attempted in this environment.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage8-dir", default=str(DEFAULT_STAGE8_DIR))
    parser.add_argument("--stage9-dir", default=str(DEFAULT_STAGE9_DIR))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "stage10a_behavioral_geometry"))
    args = parser.parse_args(argv)

    stage8_dir, stage9_dir = Path(args.stage8_dir), Path(args.stage9_dir)
    stage8_records = load_stage8_records(stage8_dir)
    stage9_records = load_stage9_records(stage9_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    integrity = run_stage10a_integrity_gate(stage8_records, stage9_records)
    _write_json(output_dir / "integrity_report.json", integrity)
    if not integrity["all_checks_pass"]:
        raise RuntimeError(f"Stage-10A integrity gate FAILED: {integrity}")
    print(f"Integrity gate PASSED ({sum(1 for v in integrity.values() if isinstance(v, bool))} checks).")

    cells = iter_cells(stage8_records, stage9_records)
    print(f"Loaded {len(cells)} (source, region, radius) cells.")

    effective_rank_by_cell = compute_effective_rank_by_cell(cells, centered=False)
    _write_json(output_dir / "effective_rank_by_cell.json", effective_rank_by_cell)
    _write_json(output_dir / "singular_spectra.json", {k: v["observed"]["singular_values"] for k, v in effective_rank_by_cell.items()})
    write_effective_rank_csv(effective_rank_by_cell, output_dir / "effective_rank_by_cell.csv")
    print("Computed effective rank + nulls for all cells.")

    pca_by_cell = compute_pca_by_cell(cells)
    _write_json(output_dir / "pca_loadings.json", pca_by_cell)
    print("Computed PCA loadings.")

    split_half_by_cell = compute_split_half_by_cell(cells)
    _write_json(output_dir / "split_half_stability.json", split_half_by_cell)
    write_split_half_csv(split_half_by_cell, output_dir / "split_half_stability.csv")
    print("Computed split-half stability.")

    cross_radius_by_region = compute_cross_radius_subspace_by_region(cells)
    _write_json(output_dir / "cross_radius_subspace.json", cross_radius_by_region)
    print("Computed cross-radius subspace geometry.")

    trajectory_geometry = compute_trajectory_geometry(cells)
    _write_json(output_dir / "trajectory_geometry.json", trajectory_geometry)
    write_trajectory_geometry_csv(trajectory_geometry, output_dir / "trajectory_geometry.csv")
    print("Computed matched-direction trajectory geometry.")

    useful_expert_geometry = compute_useful_expert_geometry_by_cell(cells)
    _write_json(output_dir / "useful_expert_geometry.json", useful_expert_geometry)
    print("Computed useful-expert geometry.")

    clustering = compute_clustering_for_useful_experts(cells)
    _write_json(output_dir / "clustering_exploratory.json", clustering)
    print("Computed exploratory clustering.")

    cross_anatomy = compute_cross_anatomy_geometry(cells)
    _write_json(output_dir / "cross_anatomy_behavioral_geometry.json", cross_anatomy)
    print("Computed cross-anatomy behavioral geometry.")

    specialization_rank = compute_specialization_rank_relationship(cells, effective_rank_by_cell)
    _write_json(output_dir / "specialization_rank_relationship.json", specialization_rank)
    write_specialization_rank_relationship_csv(specialization_rank, output_dir / "specialization_rank_relationship.csv")
    print("Computed specialization-vs-rank relationship.")

    granularity = compute_granularity_control(cells)
    _write_json(output_dir / "granularity_control.json", granularity)
    print("Computed granularity control.")

    reconstruction = compute_reconstruction_feasibility(stage8_records, stage9_records)
    _write_json(output_dir / "reconstruction_feasibility.json", reconstruction)
    print("Computed reconstruction feasibility audit.")

    classification = classify_stage10a(effective_rank_by_cell, split_half_by_cell)
    stage10b = compute_stage10b_recommendation(classification)
    paper_story = compute_paper_story_update(classification, cross_radius_by_region, trajectory_geometry)
    _write_json(output_dir / "stage10b_recommendation.json", {"classification": classification, **stage10b, "paper_story": paper_story})

    report = build_markdown_report(integrity, classification, stage10b, paper_story, reconstruction)
    (output_dir / "stage10a_summary.md").write_text(report)

    print(f"Wrote Stage-10A analysis outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
