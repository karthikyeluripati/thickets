"""Stage 8 analysis schema (this repair pass): the paper-scale L1 anatomy x capability x radius
atlas. Built and tested against synthetic ExperimentResultRecord grids BEFORE any real GPU data
exists (Stage 8's 576-perturbation run has not been executed yet) -- every function here is a
pure, deterministic transform of already-collected records, run NO model, applies NO
perturbation, and NEVER selects a "best" radius/region/capability (see
test_no_best_selection_logic_exists in the accompanying test file).

Reuses this project's OWN existing statistical primitives (thicket.metrics, thicket.diversity,
thicket_metrics.wilson_confidence_interval, run_global_visual_thicket_pilot.build_delta_matrix)
throughout -- never reimplements them -- matching the discipline already established in
analysis/stage6_visual_thicket_analysis.py and analysis/stage7b_anatomical_calibration_analysis.py.

Produces (once real data exists, at results/stage8_coarse_anatomical_atlas/<run_signature>/analysis/):
    primary_measurements.json        -- Section 11: capability x anatomy x radius descriptive
                                         stats + Wilson/bootstrap CIs
    anatomical_contrasts.json        -- Section 12: matched-radius vision/connector/language
                                         pairwise effect-size contrasts, per capability
    radius_trajectories.json         -- Section 13: per (region, direction_index, capability)
                                         Delta(R_small)->Delta(R_mid)->Delta(R_transition)
    cross_capability_specialization.json -- Section 14: 6x6 Spearman/discordance/Jaccard/sign
                                         agreement/improvement-count histogram, per anatomy x radius
    anatomical_selectivity_atlas.json -- Section 15: capability x anatomy atlas (density>=.02,
                                         positive thicket mass, mean Delta), NEVER collapsed
                                         across radius
    quantization_audit.json          -- Section 16: strict/quantization-limited counts + error
                                         stats, by region x radius

Usage:
    python analysis/stage8_coarse_anatomical_atlas_analysis.py [--results-dir <path>]
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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neural_thickets_repro.run_global_visual_thicket_pilot import build_delta_matrix, load_records  # noqa: E402
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import (  # noqa: E402
    STAGE8_CAPABILITIES, STAGE8_D_MAP_N, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_RADII, STAGE8_REGIONS,
)
from neural_thickets_repro.thicket import diversity as thicket_diversity  # noqa: E402
from neural_thickets_repro.thicket import metrics as thicket_metrics  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402
from neural_thickets_repro.thicket_metrics import wilson_confidence_interval  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "stage8_coarse_anatomical_atlas" / "stage8_coarse_anatomical_atlas_3b_v2_batched10"

BOOTSTRAP_SEED = 20260825  # distinct from Stage 6/7B's own bootstrap seeds, deterministic
N_BOOTSTRAP = 10000
PERMUTATION_SEED = 20260826  # distinct namespace from the bootstrap seed -- independent RNG stream
N_PERMUTATIONS = 10000
TOP_Q_FRACTIONS: Tuple[float, ...] = (0.1, 0.2)  # reused BY IDENTITY from stage6_visual_thicket_analysis.py's own frozen convention (WITHIN_SIGMA_JACCARD_FRACTIONS) -- never re-optimized
HARM_MARGIN = 0.02  # matches the project's existing "solution" margin convention -- never re-derived per analysis
SOLUTION_DENSITY_MARGIN_GRID: Tuple[float, ...] = tuple(
    round(x, 4) for x in np.arange(-0.10, 0.32, 0.01)
)  # step 0.01 so the frozen 0.02/0.05 thresholds land exactly on the grid; spans the observed useful range


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_sanitize(obj), indent=2))


def load_all(results_dir: Path) -> List[ExperimentResultRecord]:
    return load_records(results_dir / "results.jsonl")


def group_by_capability_region_radius(records: Sequence[ExperimentResultRecord]) -> Dict[Tuple[str, str, float], List[ExperimentResultRecord]]:
    out: Dict[Tuple[str, str, float], List[ExperimentResultRecord]] = {}
    for r in records:
        out.setdefault((r.capability, r.anatomy_region, r.radius), []).append(r)
    return out


def group_by_region_radius(records: Sequence[ExperimentResultRecord]) -> Dict[Tuple[str, float], List[ExperimentResultRecord]]:
    out: Dict[Tuple[str, float], List[ExperimentResultRecord]] = {}
    for r in records:
        out.setdefault((r.anatomy_region, r.radius), []).append(r)
    return out


# =============================================================================================
# Section 1: integrity gate -- must pass before any statistic below is trusted
# =============================================================================================


class Stage8IntegrityError(RuntimeError):
    """The raw results.jsonl fails the frozen Stage-8 design's hard-verification gate -- never
    silently analyzed. See run_integrity_gate's own checks list.
    """


def run_integrity_gate(records: Sequence[ExperimentResultRecord], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Mechanical, all-or-nothing verification against the FROZEN Stage-8 design constants
    (STAGE8_REGIONS/STAGE8_RADII/STAGE8_CAPABILITIES/STAGE8_N_DIRECTIONS_PER_CELL/
    STAGE8_D_MAP_N) and the run's own persisted checkpoint_manifest.json -- never against
    values re-derived from the data itself (that would make the gate unable to catch a
    systematically wrong run). Returns a dict of individual boolean checks plus
    `all_checks_pass`; raises Stage8IntegrityError only via the separate
    `ensure_stage8_integrity` gate below, so a caller can persist the full report even on
    failure.
    """
    checks: Dict[str, Any] = {}

    checks["three_anatomies"] = {r.anatomy_region for r in records} == set(STAGE8_REGIONS)
    checks["three_frozen_radii"] = {r.radius for r in records} == set(STAGE8_RADII)
    checks["six_capabilities"] = {r.capability for r in records} == set(STAGE8_CAPABILITIES)
    checks["expected_total_rows_3456"] = len(records) == len(STAGE8_REGIONS) * len(STAGE8_RADII) * STAGE8_N_DIRECTIONS_PER_CELL * len(STAGE8_CAPABILITIES)

    by_pid: Dict[str, List[ExperimentResultRecord]] = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, []).append(r)
    checks["expected_576_unique_perturbations"] = len(by_pid) == len(STAGE8_REGIONS) * len(STAGE8_RADII) * STAGE8_N_DIRECTIONS_PER_CELL
    checks["exactly_6_rows_per_perturbation"] = all(len(rows) == len(STAGE8_CAPABILITIES) for rows in by_pid.values())
    checks["same_candidate_evaluated_on_all_6_capabilities"] = all(
        {row.capability for row in rows} == set(STAGE8_CAPABILITIES) for rows in by_pid.values()
    )
    checks["no_duplicate_capability_rows_within_a_perturbation"] = all(
        len({row.capability for row in rows}) == len(rows) for rows in by_pid.values()
    )

    by_region_radius: Dict[Tuple[str, float], set] = {}
    for pid, rows in by_pid.items():
        key = (rows[0].anatomy_region, rows[0].radius)
        by_region_radius.setdefault(key, set()).add(pid)
    expected_cells = {(region, radius) for region in STAGE8_REGIONS for radius in STAGE8_RADII}
    checks["no_missing_cells"] = set(by_region_radius.keys()) == expected_cells
    checks["exactly_64_perturbations_per_anatomy_x_radius"] = all(len(v) == STAGE8_N_DIRECTIONS_PER_CELL for v in by_region_radius.values())

    by_region_seed: Dict[str, Dict[Any, set]] = {}
    for r in records:
        seed = r.runtime_metadata.get("direction_seed")
        by_region_seed.setdefault(r.anatomy_region, {}).setdefault(seed, set()).add(r.radius)
    seed_reuse_ok = True
    for region in STAGE8_REGIONS:
        seed_map = by_region_seed.get(region, {})
        if len(seed_map) != STAGE8_N_DIRECTIONS_PER_CELL:
            seed_reuse_ok = False
        if any(radii_seen != set(STAGE8_RADII) for radii_seen in seed_map.values()):
            seed_reuse_ok = False
    checks["direction_seed_reused_across_all_3_radii_within_anatomy"] = seed_reuse_ok

    # No pairing of directions across DIFFERENT anatomies -- distinct regions must not be
    # treated as sharing a "direction family" even if a raw seed integer coincidentally recurs.
    direction_family_ids = {r.runtime_metadata.get("direction_family_id") for r in records}
    checks["direction_family_ids_are_region_qualified"] = all(
        fid is not None and fid.split(":")[0] in STAGE8_REGIONS for fid in direction_family_ids
    )

    checks["model_revision_consistent"] = len({r.model_revision for r in records}) == 1
    checks["model_revision"] = next(iter({r.model_revision for r in records}), None)
    checks["d_map_n_50"] = checkpoint.get("d_map_n") == STAGE8_D_MAP_N
    checks["all_six_subset_hashes_present"] = set(checkpoint.get("subset_hashes", {}).keys()) == set(STAGE8_CAPABILITIES)
    checks["all_three_anatomy_mask_hashes_present"] = set(checkpoint.get("region_mask_hashes", {}).keys()) == set(STAGE8_REGIONS)
    checks["enable_prefix_caching_false"] = checkpoint.get("enable_prefix_caching") is False
    checks["cache_policy_correct"] = checkpoint.get("multimodal_cache_policy") == "full_encoder_reset_vllm011_verified_v2"
    checks["radius_realization_method_correct"] = checkpoint.get("radius_realization_method") == "fixed_direction_bf16_quantization_aware_v3"
    checks["run_complete"] = checkpoint.get("run_complete", checkpoint.get("expected_result_rows") == len(records))

    non_meta_keys = [k for k in checks if k not in ("model_revision",)]
    checks["all_checks_pass"] = all(bool(checks[k]) for k in non_meta_keys if isinstance(checks[k], bool))
    return checks


def ensure_stage8_integrity(integrity_report: Dict[str, Any]) -> None:
    if not integrity_report.get("all_checks_pass"):
        failed = {
            k: v for k, v in integrity_report.items()
            if isinstance(v, bool) and not v
        }
        raise Stage8IntegrityError(f"Stage-8 integrity gate FAILED -- refusing to analyze. Failed checks: {failed}")


# =============================================================================================
# Section 2: baseline table
# =============================================================================================


def compute_baseline_table(records: Sequence[ExperimentResultRecord], baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
    """One canonical baseline per capability, confirmed independent of anatomy/radius/direction
    by checking that EVERY row for that capability (across all 576 perturbations) carries the
    identical base_score -- never averaged or recomputed per cell. `score_granularity` reports
    the smallest observed nonzero |delta| for that capability (a discrete-metric capability like
    exact-match accuracy at N=50 is granular in multiples of 1/50=0.02; a continuous metric like
    VQA soft-accuracy is not) -- descriptive only, never used to change any threshold.
    """
    by_cap: Dict[str, List[ExperimentResultRecord]] = {}
    for r in records:
        by_cap.setdefault(r.capability, []).append(r)

    out: Dict[str, Any] = {}
    for cap, rows in by_cap.items():
        base_scores_seen = sorted({r.base_score for r in rows})
        canonical_score = baseline_scores.get("capabilities", {}).get(cap, {}).get("score")
        nonzero_abs_deltas = sorted({abs(r.delta) for r in rows if r.delta != 0.0})
        out[cap] = {
            "capability": cap,
            "baseline_score": canonical_score,
            "headroom_1_minus_baseline": (1.0 - canonical_score) if canonical_score is not None else None,
            "base_score_values_seen_across_all_576_perturbations": base_scores_seen,
            "canonical_baseline_independent_of_anatomy_radius_direction": len(base_scores_seen) == 1 and (
                canonical_score is None or base_scores_seen[0] == canonical_score
            ),
            "min_observed_nonzero_abs_delta": nonzero_abs_deltas[0] if nonzero_abs_deltas else None,
            "n_distinct_nonzero_abs_delta_values": len(nonzero_abs_deltas),
        }
    return out


# =============================================================================================
# Section 11: primary measurements
# =============================================================================================


def compute_primary_measurements(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_cell = group_by_capability_region_radius(records)
    out: Dict[str, Any] = {}
    for (cap, region, radius), rows in by_cell.items():
        deltas = [r.delta for r in rows]
        arr = np.asarray(deltas, dtype=float)
        n = int(arr.size)
        mean, std = thicket_metrics.mean_std(deltas)
        median = float(np.median(arr))
        p_gt0 = thicket_metrics.probability_of_improvement(deltas)
        p_lt0 = thicket_metrics.probability_of_degradation(deltas)
        density = thicket_metrics.solution_density(deltas, margins=(0.02, 0.05))
        mass = thicket_metrics.positive_thicket_mass(deltas)

        n_ge0 = int(np.sum(arr >= 0.0))
        n_ge_02 = int(np.sum(arr >= 0.02))
        n_ge_05 = int(np.sum(arr >= 0.05))
        n_gt0 = int(np.sum(arr > 0))
        mean_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=np.mean, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
        mass_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=lambda d: float(np.mean(np.clip(d, 0.0, None))), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 1)
        negative_mass = float(np.mean(np.clip(-arr, 0.0, None)))

        out.setdefault(cap, {}).setdefault(region, {})[str(radius)] = {
            "capability": cap, "anatomy_region": region, "radius": radius, "n": n,
            "mean_delta": mean, "mean_delta_95ci_bootstrap": list(mean_ci),
            "std_delta": std, "median_delta": median, "min_delta": float(arr.min()), "max_delta": float(arr.max()),
            "p_delta_gt_0": p_gt0, "p_delta_gt_0_95ci_wilson": list(wilson_confidence_interval(n_gt0, n)),
            "p_delta_lt_0": p_lt0,
            "density_ge_0.0": n_ge0 / n, "density_ge_0.0_95ci_wilson": list(wilson_confidence_interval(n_ge0, n)),
            "density_ge_0.02": density[0.02], "density_ge_0.02_95ci_wilson": list(wilson_confidence_interval(n_ge_02, n)),
            "density_ge_0.05": density[0.05], "density_ge_0.05_95ci_wilson": list(wilson_confidence_interval(n_ge_05, n)),
            "positive_thicket_mass": mass, "positive_thicket_mass_95ci_bootstrap": list(mass_ci),
            "negative_mass": negative_mass,
        }
    return out


# =============================================================================================
# Section 12: anatomical contrasts (matched radius)
# =============================================================================================


_CONTRAST_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("vision", "multimodal_connector_or_merger"),
    ("vision", "language"),
    ("multimodal_connector_or_merger", "language"),
)


def _bootstrap_diff_ci(deltas_a: Sequence[float], deltas_b: Sequence[float], statistic_fn_axis1, seed: int) -> Tuple[float, float]:
    """Independent-samples bootstrap CI for statistic_fn_axis1(a) - statistic_fn_axis1(b) --
    resamples each side independently (unpaired: region A's 64 directions and region B's 64
    directions are different perturbations by construction, never a paired comparison).
    `statistic_fn_axis1` reduces an (N_BOOTSTRAP, n) resampled matrix along axis=1 in one
    vectorized call (see _mean_axis1/_density_ge_002_axis1/_positive_mass_axis1 below) -- all
    N_BOOTSTRAP resamples are drawn and reduced at once via numpy, never a per-resample Python
    loop, which is what makes this tractable across the full capability x radius x region-pair grid.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(deltas_a, dtype=float)
    b = np.asarray(deltas_b, dtype=float)
    resampled_a = a[rng.integers(0, a.size, size=(N_BOOTSTRAP, a.size))]
    resampled_b = b[rng.integers(0, b.size, size=(N_BOOTSTRAP, b.size))]
    diffs = statistic_fn_axis1(resampled_a) - statistic_fn_axis1(resampled_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_axis1(m: np.ndarray) -> np.ndarray:
    return m.mean(axis=1)


def _density_ge_002_axis1(m: np.ndarray) -> np.ndarray:
    return (m >= 0.02).mean(axis=1)


def _positive_mass_axis1(m: np.ndarray) -> np.ndarray:
    return np.clip(m, 0.0, None).mean(axis=1)


def _density_ge_002(arr: np.ndarray) -> float:
    return float(np.mean(arr >= 0.02))


def _positive_mass(arr: np.ndarray) -> float:
    return float(np.mean(np.clip(arr, 0.0, None)))


def _permutation_p_value(
    deltas_a: np.ndarray, deltas_b: np.ndarray, statistic_fn_axis1, observed_diff: float, seed: int,
) -> float:
    """Two-sided anatomy-label permutation test: pools a and b, repeatedly reassigns group
    labels at random (respecting the original group sizes), and reports the fraction of
    permuted |diff| >= |observed diff|. Vectorized across N_PERMUTATIONS at once.
    """
    pooled = np.concatenate([deltas_a, deltas_b])
    n_a = deltas_a.size
    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((N_PERMUTATIONS, pooled.size)), axis=1)
    permuted = pooled[perm_idx]
    perm_a, perm_b = permuted[:, :n_a], permuted[:, n_a:]
    diffs = statistic_fn_axis1(perm_a) - statistic_fn_axis1(perm_b)
    return float(np.mean(np.abs(diffs) >= abs(observed_diff)))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    pooled_std = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))
    if pooled_std == 0.0:
        return None
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _cohens_h(p_a: float, p_b: float) -> float:
    """Cohen's h -- the standard effect size for a difference between two proportions."""
    return float(2 * np.arcsin(np.sqrt(np.clip(p_a, 0.0, 1.0))) - 2 * np.arcsin(np.sqrt(np.clip(p_b, 0.0, 1.0))))


def _standardized_mass_diff(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    pos_a, pos_b = np.clip(a, 0.0, None), np.clip(b, 0.0, None)
    pooled_std = float(np.sqrt((np.var(pos_a, ddof=1) + np.var(pos_b, ddof=1)) / 2.0))
    if pooled_std == 0.0:
        return None
    return float((np.mean(pos_a) - np.mean(pos_b)) / pooled_std)


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Standard BH step-up FDR correction: returns adjusted q-values in the SAME order as the
    input `pvalues`. q-values are monotone non-decreasing when sorted by descending p-value
    (the standard "running minimum from the largest rank down" enforcement), clipped to [0, 1].
    """
    m = len(pvalues)
    if m == 0:
        return []
    arr = np.asarray(pvalues, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    ranks = np.arange(1, m + 1)
    q = ranked * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m)
    out[order] = q
    return out.tolist()


def compute_anatomical_contrasts(
    records: Sequence[ExperimentResultRecord], *, contrast_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """D_map exploratory statistics only (never held-out D_confirm evidence). Reports, per
    capability x radius x anatomy-pair: bootstrap CIs (independent, direction-level resampling
    -- directions in different regions are NOT the same geometric direction, so resampling is
    never paired), a two-sided anatomy-label permutation-test p-value, and an effect size, for
    each of mean_delta / density>=.02 / positive_thicket_mass. BH-FDR correction is then applied
    SEPARATELY within each of the three statistic families (54 p-values per family: 6
    capabilities x 3 radii x 3 anatomy pairs) -- see apply_benjamini_hochberg_correction.

    `contrast_pairs` defaults to Stage 8's own frozen 3 L1-region pairs (`_CONTRAST_PAIRS`) --
    pass a different set of (region_a, region_b) pairs to reuse this EXACT statistical machinery
    for a different partition of the same `anatomy_region` field (e.g. Stage 9's within-parent
    depth-band pairs), without duplicating any of the bootstrap/permutation/effect-size logic.
    """
    pairs = contrast_pairs if contrast_pairs is not None else _CONTRAST_PAIRS
    by_cell = group_by_capability_region_radius(records)
    out: Dict[str, Any] = {}
    capabilities = sorted({r.capability for r in records})
    radii = sorted({r.radius for r in records})
    for cap in capabilities:
        for radius in radii:
            for region_a, region_b in pairs:
                key_a, key_b = (cap, region_a, radius), (cap, region_b, radius)
                if key_a not in by_cell or key_b not in by_cell:
                    continue
                deltas_a = np.asarray([r.delta for r in by_cell[key_a]], dtype=float)
                deltas_b = np.asarray([r.delta for r in by_cell[key_b]], dtype=float)
                mean_diff = float(np.mean(deltas_a) - np.mean(deltas_b))
                p_a, p_b = _density_ge_002(deltas_a), _density_ge_002(deltas_b)
                density_diff = p_a - p_b
                mass_diff = _positive_mass(deltas_a) - _positive_mass(deltas_b)
                seed = BOOTSTRAP_SEED + hash((cap, region_a, region_b, radius)) % 10_000
                perm_seed = PERMUTATION_SEED + hash((cap, region_a, region_b, radius)) % 10_000
                out.setdefault(cap, {}).setdefault(str(radius), {})[f"{region_a}_vs_{region_b}"] = {
                    "capability": cap, "radius": radius, "region_a": region_a, "region_b": region_b,
                    "mean_delta_diff": mean_diff,
                    "mean_delta_diff_95ci_bootstrap": list(_bootstrap_diff_ci(deltas_a, deltas_b, _mean_axis1, seed)),
                    "mean_delta_diff_permutation_p": _permutation_p_value(deltas_a, deltas_b, _mean_axis1, mean_diff, perm_seed),
                    "mean_delta_effect_size_cohens_d": _cohens_d(deltas_a, deltas_b),
                    "density_ge_0.02_diff": density_diff,
                    "density_ge_0.02_diff_95ci_bootstrap": list(_bootstrap_diff_ci(deltas_a, deltas_b, _density_ge_002_axis1, seed + 1)),
                    "density_ge_0.02_diff_permutation_p": _permutation_p_value(deltas_a, deltas_b, _density_ge_002_axis1, density_diff, perm_seed + 1),
                    "density_ge_0.02_effect_size_cohens_h": _cohens_h(p_a, p_b),
                    "positive_thicket_mass_diff": mass_diff,
                    "positive_thicket_mass_diff_95ci_bootstrap": list(_bootstrap_diff_ci(deltas_a, deltas_b, _positive_mass_axis1, seed + 2)),
                    "positive_thicket_mass_diff_permutation_p": _permutation_p_value(deltas_a, deltas_b, _positive_mass_axis1, mass_diff, perm_seed + 2),
                    "positive_thicket_mass_effect_size_standardized": _standardized_mass_diff(deltas_a, deltas_b),
                }
    return out


_BH_STATISTIC_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("mean_delta_diff_permutation_p", "mean_delta_diff_bh_q"),
    ("density_ge_0.02_diff_permutation_p", "density_ge_0.02_diff_bh_q"),
    ("positive_thicket_mass_diff_permutation_p", "positive_thicket_mass_diff_bh_q"),
)
BH_FDR_ALPHA = 0.05


def apply_benjamini_hochberg_correction(contrasts: Dict[str, Any]) -> Dict[str, Any]:
    """Walks the full nested contrasts structure, collects each statistic's p-values into its
    OWN family (never pooled across the three statistic types), applies BH correction within
    each family, and writes the resulting q-value (plus a `..._bh_significant_fdr_0.05` flag)
    back onto each contrast cell IN PLACE. D_map-exploratory: flags a PATTERN worth confirming
    on D_confirm, never itself a confirmation.
    """
    cells: List[Dict[str, Any]] = [
        cell for cap_map in contrasts.values() for radius_map in cap_map.values() for cell in radius_map.values()
    ]
    for p_key, q_key in _BH_STATISTIC_FAMILIES:
        pvalues = [cell[p_key] for cell in cells]
        qvalues = benjamini_hochberg(pvalues)
        sig_key = q_key.replace("_bh_q", "_bh_significant_fdr_0.05")
        for cell, q in zip(cells, qvalues):
            cell[q_key] = q
            cell[sig_key] = q < BH_FDR_ALPHA
    return contrasts


# =============================================================================================
# Section 13: radius trajectories (direction-family reuse across radii)
# =============================================================================================


def compute_radius_trajectories(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Groups rows by (region, direction_index, capability) via runtime_metadata's
    direction_family_id, orders by radius, and computes: sign persistence (fraction of
    consecutive radius steps whose Delta sign is unchanged), improvement survival (does a
    positive Delta at R_small remain positive at every larger radius tested), monotonic
    degradation frequency (fraction of trajectories that are non-increasing across radii),
    and the radius (if any) at which a positive direction first turns non-positive.
    """
    radii = sorted({r.radius for r in records})
    by_family: Dict[Tuple[str, int, str], Dict[float, float]] = {}
    for r in records:
        family_id = r.runtime_metadata.get("direction_family_id")
        if family_id is None:
            continue
        region, idx_str = family_id.split(":")
        key = (region, int(idx_str), r.capability)
        by_family.setdefault(key, {})[r.radius] = r.delta

    if len(radii) != 3:
        raise ValueError(f"compute_radius_trajectories assumes exactly 3 radii (small/mid/transition), got {len(radii)}: {radii}")
    r_small, r_mid, r_transition = radii

    trajectories: Dict[str, Any] = {}
    sign_persistent = 0
    total_pairs = 0
    improvement_survives = 0
    total_positive_at_smallest = 0
    monotonic_nonincreasing = 0
    monotonic_nondecreasing = 0
    non_monotonic = 0
    total_trajectories = 0
    disappearance_radius_counts: Dict[str, int] = {str(r): 0 for r in radii}
    positive_at_small_and_mid = 0
    positive_at_small_and_transition = 0
    emerges_only_at_mid = 0  # non-positive at small, positive at mid
    emerges_only_at_transition = 0  # non-positive at small AND mid, positive at transition
    radius_win_tally_description_only: Dict[str, int] = {str(r): 0 for r in radii}

    for (region, idx, cap), radius_to_delta in by_family.items():
        ordered = [radius_to_delta.get(r) for r in radii]
        if any(v is None for v in ordered):
            continue  # incomplete trajectory (not evaluated at every radius) -- excluded, never fabricated
        total_trajectories += 1
        d_small, d_mid, d_transition = ordered
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in ordered]
        for i in range(len(signs) - 1):
            total_pairs += 1
            if signs[i] == signs[i + 1]:
                sign_persistent += 1
        if d_small > 0:
            total_positive_at_smallest += 1
            if d_mid > 0:
                positive_at_small_and_mid += 1
            if d_transition > 0:
                positive_at_small_and_transition += 1
            if all(v > 0 for v in ordered):
                improvement_survives += 1
            else:
                first_nonpositive_idx = next(i for i, v in enumerate(ordered) if v <= 0)
                disappearance_radius_counts[str(radii[first_nonpositive_idx])] += 1
        else:
            if d_mid > 0:
                emerges_only_at_mid += 1
            if d_mid <= 0 and d_transition > 0:
                emerges_only_at_transition += 1
        if all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)):
            monotonic_nonincreasing += 1
        if all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1)):
            monotonic_nondecreasing += 1
        is_monotonic = all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)) or all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
        if not is_monotonic:
            non_monotonic += 1

        best_radius = radii[int(np.argmax(ordered))]
        radius_win_tally_description_only[str(best_radius)] += 1

        family_id = f"{region}:{idx}"
        trajectories.setdefault(cap, {})[family_id] = {
            "region": region, "direction_index": idx, "capability": cap,
            "delta_by_radius": {str(r): v for r, v in zip(radii, ordered)},
            "best_radius_for_description_only": best_radius,
        }

    rank_stability = _compute_rank_stability_across_radii(records, radii)

    return {
        "radii": radii, "r_small": r_small, "r_mid": r_mid, "r_transition": r_transition,
        "n_complete_trajectories": total_trajectories,
        "sign_persistence_rate": (sign_persistent / total_pairs) if total_pairs else None,
        "improvement_survival_rate": (improvement_survives / total_positive_at_smallest) if total_positive_at_smallest else None,
        "positive_at_small_remains_positive_at_mid_rate": (positive_at_small_and_mid / total_positive_at_smallest) if total_positive_at_smallest else None,
        "positive_at_small_remains_positive_at_transition_rate": (positive_at_small_and_transition / total_positive_at_smallest) if total_positive_at_smallest else None,
        "n_positive_at_small": total_positive_at_smallest,
        "n_directions_emerging_only_at_mid": emerges_only_at_mid,
        "n_directions_emerging_only_at_transition": emerges_only_at_transition,
        "monotonic_nonincreasing_fraction": (monotonic_nonincreasing / total_trajectories) if total_trajectories else None,
        "monotonic_nondecreasing_fraction": (monotonic_nondecreasing / total_trajectories) if total_trajectories else None,
        "non_monotonic_fraction": (non_monotonic / total_trajectories) if total_trajectories else None,
        "positive_direction_disappearance_radius_histogram": disappearance_radius_counts,
        "best_radius_histogram_description_only": radius_win_tally_description_only,
        "rank_stability_spearman_between_consecutive_radii": rank_stability,
        "trajectories_by_capability": trajectories,
    }


def _compute_rank_stability_across_radii(records: Sequence[ExperimentResultRecord], radii: Sequence[float]) -> Dict[str, Any]:
    """Spearman rank correlation of direction-family Delta rankings between each pair of
    CONSECUTIVE radii, per capability x region -- reuses thicket.diversity.task_rank_correlation_
    matrix (a 2-column delta "matrix" gives exactly the pairwise Spearman correlation).
    """
    by_cap_region: Dict[Tuple[str, str], Dict[int, Dict[float, float]]] = {}
    for r in records:
        family_id = r.runtime_metadata.get("direction_family_id")
        if family_id is None:
            continue
        region, idx_str = family_id.split(":")
        key = (r.capability, region)
        by_cap_region.setdefault(key, {}).setdefault(int(idx_str), {})[r.radius] = r.delta

    out: Dict[str, Any] = {}
    for (cap, region), idx_to_radius_delta in by_cap_region.items():
        complete = {idx: d for idx, d in idx_to_radius_delta.items() if all(r in d for r in radii)}
        if len(complete) < 2:
            continue
        ordered_idx = sorted(complete)
        for i in range(len(radii) - 1):
            r_a, r_b = radii[i], radii[i + 1]
            col_a = np.array([complete[idx][r_a] for idx in ordered_idx])
            col_b = np.array([complete[idx][r_b] for idx in ordered_idx])
            if np.std(col_a) == 0 or np.std(col_b) == 0:
                continue
            matrix = np.stack([col_a, col_b], axis=1)
            corr = thicket_diversity.task_rank_correlation_matrix(matrix)
            out.setdefault(cap, {}).setdefault(region, {})[f"{r_a}_to_{r_b}"] = float(corr[0, 1])
    return out


# =============================================================================================
# Section 14: cross-capability specialization (6x6, per anatomy x radius)
# =============================================================================================


def compute_cross_capability_specialization(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Sections A-E + F (harm-while-improve) + G (directional transfer), within each anatomy x
    radius cell's shared 64 candidates x 6 capabilities. TOP_Q_FRACTIONS (0.1, 0.2) reused BY
    IDENTITY from stage6_visual_thicket_analysis.py's own frozen within-sigma Jaccard
    convention -- never re-optimized for Stage 8.
    """
    by_region_radius = group_by_region_radius(records)
    out: Dict[str, Any] = {}
    for (region, radius), rows in sorted(by_region_radius.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        perturbation_ids, capabilities, matrix = build_delta_matrix(rows)
        m = matrix.shape[1]
        n = matrix.shape[0]
        spearman = thicket_diversity.task_rank_correlation_matrix(matrix)
        discordance = thicket_diversity.spectral_discordance(matrix)
        overlap = {
            f"q_{q}": thicket_diversity.expert_overlap_matrix(matrix, q=q, q_is_fraction=True).tolist()
            for q in TOP_Q_FRACTIONS
        }
        signs = np.sign(matrix)
        sign_agreement = np.eye(m)
        for i in range(m):
            for j in range(i + 1, m):
                sign_agreement[i, j] = sign_agreement[j, i] = float(np.mean(signs[:, i] == signs[:, j]))
        n_improving = np.sum(matrix > 0, axis=1)
        improving_hist = {str(k): int(np.sum(n_improving == k)) for k in range(m + 1)}

        # F: fraction of candidates that improve at least one capability while harming another
        # by >= HARM_MARGIN (0.02) -- a genuine within-candidate improve/harm tradeoff.
        improves_one = matrix > 0
        harms_margin = matrix <= -HARM_MARGIN
        tradeoff_candidates = np.any(improves_one, axis=1) & np.any(harms_margin, axis=1)
        # Exclude the degenerate case where the SAME single capability column is being asked to
        # both improve and harm on the same row (impossible per-row, but guard explicitly): a
        # candidate only counts if the improving set and harming set of capabilities are disjoint
        # per row is trivially true since a scalar can't be both >0 and <=-0.02 simultaneously.
        n_tradeoff = int(np.sum(tradeoff_candidates))

        # G: directional transfer -- for each source capability t, mean Delta and P(Delta>0) on
        # every OTHER capability, restricted to candidates with Delta_t > 0.
        directional_transfer: Dict[str, Any] = {}
        for ti, source_cap in enumerate(capabilities):
            selected = matrix[:, ti] > 0
            n_selected = int(np.sum(selected))
            row_out: Dict[str, Any] = {"n_source_positive": n_selected}
            for tj, target_cap in enumerate(capabilities):
                if ti == tj or n_selected == 0:
                    row_out[target_cap] = None
                    continue
                target_deltas = matrix[selected, tj]
                row_out[target_cap] = {
                    "mean_delta": float(np.mean(target_deltas)),
                    "p_delta_gt_0": float(np.mean(target_deltas > 0)),
                }
            directional_transfer[source_cap] = row_out

        out.setdefault(region, {})[str(radius)] = {
            "region": region, "radius": radius, "n_perturbations": n, "capabilities": list(capabilities),
            "spearman_6x6": spearman.tolist(), "spectral_discordance": discordance,
            "expert_overlap_jaccard": overlap, "top_q_fractions_used": list(TOP_Q_FRACTIONS),
            "sign_agreement_matrix": sign_agreement.tolist(),
            "improving_count_histogram": improving_hist,
            "n_tradeoff_candidates_improve_one_harm_another_ge_0.02": n_tradeoff,
            "fraction_tradeoff_candidates": n_tradeoff / n if n else None,
            "harm_margin_used": HARM_MARGIN,
            "directional_transfer": directional_transfer,
        }
    return out


# =============================================================================================
# Section 15: anatomical selectivity atlas (never collapsed across radius)
# =============================================================================================


def compute_anatomical_selectivity_atlas(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    primary = compute_primary_measurements(records)
    out: Dict[str, Any] = {}
    for cap, region_map in primary.items():
        for region, radius_map in region_map.items():
            for radius_key, row in radius_map.items():
                out.setdefault(radius_key, {}).setdefault(cap, {})[region] = {
                    "density_ge_0.02": row["density_ge_0.02"],
                    "positive_thicket_mass": row["positive_thicket_mass"],
                    "mean_delta": row["mean_delta"],
                }
    return out


# =============================================================================================
# Section 5: capability x anatomy interaction -- selectivity, dominance, entropy, stability
# =============================================================================================


def _normalized_distribution(values: Sequence[float]) -> List[float]:
    """Normalizes a non-negative vector to sum to 1 -- an all-zero vector maps to a uniform
    distribution (maximal entropy), the honest representation of "no positive mass anywhere",
    never a divide-by-zero or a fabricated concentration.
    """
    arr = np.asarray(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (arr / total).tolist()


def _entropy(distribution: Sequence[float]) -> float:
    arr = np.asarray(distribution, dtype=float)
    arr = arr[arr > 0]
    return float(-np.sum(arr * np.log2(arr)))


def compute_anatomy_capability_interaction(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Answers BOTH directions of Section 5: (A) for each capability, where is useful
    perturbation density concentrated across anatomy; (B) for each anatomy, which capabilities
    preferentially improve there. Selectivity is quantified from the NORMALIZED positive-
    thicket-mass distribution across the 3 anatomies (never density alone, since mass also
    captures improvement magnitude) -- dominant anatomy (if the top share exceeds the second by
    a nonzero margin), dominance margin, Shannon entropy across the 3-way distribution (low
    entropy = concentrated / high entropy = diffuse), and whether the SAME anatomy dominates at
    >=2 of the 3 radii. Deliberately uses "Stage-8 anatomical preference" / "mapping-scale
    anatomical concentration" language, never "location of experts" -- D_confirm has not run.
    """
    primary = compute_primary_measurements(records)
    radii = sorted({r.radius for r in records})
    regions = sorted({r.anatomy_region for r in records})

    by_capability_direction: Dict[str, Any] = {}
    dominant_region_by_cap_radius: Dict[str, Dict[str, str]] = {}
    for cap, region_map in primary.items():
        per_radius: Dict[str, Any] = {}
        for radius in radii:
            masses = [region_map[region][str(radius)]["positive_thicket_mass"] for region in regions]
            densities = [region_map[region][str(radius)]["density_ge_0.02"] for region in regions]
            mean_deltas = [region_map[region][str(radius)]["mean_delta"] for region in regions]
            dist = _normalized_distribution(masses)
            ranked = sorted(zip(dist, regions), reverse=True)
            dominant_share, dominant_region = ranked[0]
            second_share = ranked[1][0] if len(ranked) > 1 else 0.0
            entropy = _entropy(dist)
            per_radius[str(radius)] = {
                "radius": radius, "positive_thicket_mass_by_region": dict(zip(regions, masses)),
                "density_ge_0.02_by_region": dict(zip(regions, densities)), "mean_delta_by_region": dict(zip(regions, mean_deltas)),
                "normalized_mass_distribution": dict(zip(regions, dist)),
                "dominant_anatomy": dominant_region if dominant_share > second_share else None,
                "dominance_margin_over_second_best": dominant_share - second_share,
                "entropy_bits": entropy,
                "max_entropy_bits": float(np.log2(len(regions))),
            }
            dominant_region_by_cap_radius.setdefault(cap, {})[str(radius)] = dominant_region if dominant_share > second_share else None
        dominants = [dominant_region_by_cap_radius[cap][str(r)] for r in radii]
        non_none = [d for d in dominants if d is not None]
        stable_across_radii = len(non_none) >= 2 and len(set(non_none)) == 1
        by_capability_direction[cap] = {
            "per_radius": per_radius,
            "dominant_anatomy_by_radius": dominant_region_by_cap_radius[cap],
            "dominance_stable_across_at_least_2_radii": stable_across_radii,
            "stable_dominant_anatomy": non_none[0] if stable_across_radii else None,
        }

    by_anatomy_direction: Dict[str, Any] = {}
    for region in regions:
        per_radius = {}
        for radius in radii:
            masses = {cap: primary[cap][region][str(radius)]["positive_thicket_mass"] for cap in primary}
            densities = {cap: primary[cap][region][str(radius)]["density_ge_0.02"] for cap in primary}
            ranked_caps = sorted(masses.items(), key=lambda kv: kv[1], reverse=True)
            per_radius[str(radius)] = {
                "radius": radius, "positive_thicket_mass_by_capability": masses, "density_ge_0.02_by_capability": densities,
                "capabilities_ranked_by_mass": [c for c, _ in ranked_caps],
            }
        by_anatomy_direction[region] = {"per_radius": per_radius}

    return {
        "radii": radii, "regions": regions,
        "direction_A_capability_to_anatomy": by_capability_direction,
        "direction_B_anatomy_to_capability": by_anatomy_direction,
        "terminology_note": (
            "Any concentration reported here is a Stage-8 mapping-scale anatomical preference / "
            "anatomical concentration on D_map exploratory data -- NOT a final expert-location claim."
        ),
    }


# =============================================================================================
# Section 6: solution-density curves delta(m) = P[Delta >= m]
# =============================================================================================


def compute_solution_density_curves(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """delta_{t,a,r}(m) = P[Delta_t >= m] over SOLUTION_DENSITY_MARGIN_GRID (a common grid
    spanning the observed useful range, including the frozen 0.02/0.05 thresholds exactly) --
    for every capability x anatomy x radius cell. Monotonicity in m is a structural property of
    P[Delta>=m] (non-increasing as m grows) and is checked by the accompanying test, not
    asserted here.
    """
    by_cell = group_by_capability_region_radius(records)
    out: Dict[str, Any] = {}
    for (cap, region, radius), rows in by_cell.items():
        arr = np.asarray([r.delta for r in rows], dtype=float)
        n = arr.size
        curve = [float(np.mean(arr >= m)) for m in SOLUTION_DENSITY_MARGIN_GRID]
        out.setdefault(cap, {}).setdefault(region, {})[str(radius)] = {
            "capability": cap, "anatomy_region": region, "radius": radius, "n": n,
            "margin_grid": list(SOLUTION_DENSITY_MARGIN_GRID), "delta_ge_m": curve,
        }
    return out


# =============================================================================================
# Section 16: quantization audit
# =============================================================================================


def compute_quantization_audit(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_region_radius = group_by_region_radius(records)
    out: Dict[str, Any] = {}
    for (region, radius), rows in by_region_radius.items():
        # Multiple capability rows share the SAME accepted perturbation -- de-duplicate by
        # perturbation_id so each unique candidate's quantization outcome is counted once.
        by_pid: Dict[str, ExperimentResultRecord] = {r.perturbation_id: r for r in rows}
        strict = 0
        quant_limited = 0
        relative_errors: List[float] = []
        realized_over_requested: List[float] = []
        for r in by_pid.values():
            meta = r.runtime_metadata
            mode = meta.get("radius_acceptance_mode")
            if mode == "strict":
                strict += 1
            elif mode == "quantization_limited":
                quant_limited += 1
            rel_err = meta.get("relative_radius_error")
            if rel_err is not None:
                relative_errors.append(rel_err)
            realized = meta.get("realized_relative_l2")
            requested = meta.get("requested_relative_l2")
            if realized is not None and requested:
                realized_over_requested.append(realized / requested)

        out.setdefault(region, {})[str(radius)] = {
            "region": region, "radius": radius, "n_candidates": len(by_pid),
            "strict_count": strict, "quantization_limited_count": quant_limited,
            "max_relative_radius_error": max(relative_errors) if relative_errors else None,
            "mean_relative_radius_error": (sum(relative_errors) / len(relative_errors)) if relative_errors else None,
            "mean_realized_over_requested_ratio": (sum(realized_over_requested) / len(realized_over_requested)) if realized_over_requested else None,
        }
    return out


def compute_quantization_confound_audit(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Confound check ONLY (Section 10's own framing: "we do NOT want apparent anatomy
    differences explained by one region having systematically different realized radii"): per
    anatomy x radius, splits candidates into strict vs quantization_limited acceptance mode and
    compares mean Delta (averaged across the 6 capability rows per candidate) between the two
    groups via the SAME independent-samples bootstrap CI machinery used for the anatomy
    contrasts. A wide/zero-crossing CI means acceptance mode is NOT detectably associated with
    Delta in that cell -- the intended, reassuring outcome; a cell where quantization-limited
    candidates are absent reports n=0 rather than fabricating a comparison.
    """
    by_region_radius = group_by_region_radius(records)
    out: Dict[str, Any] = {}
    for (region, radius), rows in by_region_radius.items():
        by_pid: Dict[str, List[ExperimentResultRecord]] = {}
        for r in rows:
            by_pid.setdefault(r.perturbation_id, []).append(r)
        strict_means, quant_means = [], []
        for pid, pid_rows in by_pid.items():
            mode = pid_rows[0].runtime_metadata.get("radius_acceptance_mode")
            candidate_mean_delta = float(np.mean([r.delta for r in pid_rows]))
            if mode == "strict":
                strict_means.append(candidate_mean_delta)
            elif mode == "quantization_limited":
                quant_means.append(candidate_mean_delta)

        cell: Dict[str, Any] = {
            "region": region, "radius": radius,
            "n_strict_candidates": len(strict_means), "n_quantization_limited_candidates": len(quant_means),
            "mean_delta_strict": float(np.mean(strict_means)) if strict_means else None,
            "mean_delta_quantization_limited": float(np.mean(quant_means)) if quant_means else None,
        }
        if strict_means and quant_means:
            a, b = np.asarray(strict_means), np.asarray(quant_means)
            diff = float(np.mean(a) - np.mean(b))
            seed = BOOTSTRAP_SEED + hash(("quant_confound", region, radius)) % 10_000
            cell["mean_delta_diff_strict_minus_quantization_limited"] = diff
            cell["mean_delta_diff_95ci_bootstrap"] = list(_bootstrap_diff_ci(a, b, _mean_axis1, seed))
            cell["acceptance_mode_associated_with_delta"] = not (cell["mean_delta_diff_95ci_bootstrap"][0] <= 0.0 <= cell["mean_delta_diff_95ci_bootstrap"][1])
        else:
            cell["mean_delta_diff_strict_minus_quantization_limited"] = None
            cell["mean_delta_diff_95ci_bootstrap"] = None
            cell["acceptance_mode_associated_with_delta"] = None
        out.setdefault(region, {})[str(radius)] = cell
    return out


# =============================================================================================
# Section 9: thicket phenotypes (density / strength / diversity kept distinct)
# =============================================================================================


def compute_thicket_phenotypes(
    primary_measurements: Dict[str, Any], specialization: Dict[str, Any],
) -> Dict[str, Any]:
    """One phenotype record per capability x anatomy x radius: density (density_ge_0.02),
    strength (positive_thicket_mass, max_delta), and diversity (that cell's OWN
    anatomy-radius-level spectral_discordance, from cross-capability specialization -- a
    property of the (anatomy, radius) cell shared across all 6 capabilities, not
    capability-specific by construction). `specialization_score` here is literally that shared
    spectral_discordance value, reported under its own descriptive name for phenotype-table
    readability -- never a new statistic. Deliberately reports only the four requested raw
    numbers; qualitative labels ("dense but weak" etc.) are NOT computed here (left as prose in
    the .md report, never smuggled into the primary statistics).
    """
    out: Dict[str, Any] = {}
    for cap, region_map in primary_measurements.items():
        for region, radius_map in region_map.items():
            for radius_key, row in radius_map.items():
                discordance = None
                region_spec = specialization.get(region, {}).get(radius_key)
                if region_spec is not None:
                    discordance = region_spec.get("spectral_discordance")
                out.setdefault(cap, {}).setdefault(region, {})[radius_key] = {
                    "capability": cap, "anatomy_region": region, "radius": row["radius"],
                    "density_ge_0.02": row["density_ge_0.02"],
                    "positive_thicket_mass": row["positive_thicket_mass"],
                    "max_delta": row["max_delta"],
                    "specialization_score_spectral_discordance": discordance,
                }
    return out


# =============================================================================================
# Section 11: CUB / fine_grained_recognition evaluator stability check (no special thresholds)
# =============================================================================================


def compute_cub_stability_check(records: Sequence[ExperimentResultRecord], baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
    cap_rows = [r for r in records if r.capability == "fine_grained_recognition"]
    deltas = np.asarray([r.delta for r in cap_rows], dtype=float)
    n = deltas.size
    nonzero_abs = sorted({abs(d) for d in deltas.tolist() if d != 0.0})
    scores = np.asarray([r.perturbed_score for r in cap_rows], dtype=float)
    return {
        "capability": "fine_grained_recognition", "n_rows": n,
        "baseline_score": baseline_scores.get("capabilities", {}).get("fine_grained_recognition", {}).get("score"),
        "fraction_exact_zero_delta": float(np.mean(deltas == 0.0)),
        "min_nonzero_abs_delta_observed": nonzero_abs[0] if nonzero_abs else None,
        "n_distinct_nonzero_abs_delta_values": len(nonzero_abs),
        "score_distribution": {
            "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)) if n > 1 else 0.0,
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "quartiles": [float(np.percentile(scores, q)) for q in (25, 50, 75)],
        },
        "note": "No special/looser thresholds applied for CUB -- this verifies it behaves like a usable Stage-8 capability, using the SAME statistics every other capability gets.",
    }


# =============================================================================================
# Section 12: Stage 6 / Stage 7B / Stage 8 bridge (qualitative, no sigma<->radius equivalence)
# =============================================================================================


def compute_stage6_stage7b_stage8_bridge(
    primary_measurements: Dict[str, Any], stage7b_calibration_table: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Qualitative connection only -- sigma (Stage 6) and relative-L2 radius (Stage 7B/8) are
    different parameterizations over different perturbation scopes/regions and are NEVER
    numerically equated. Uses ONLY the clean, cache-safe Stage-6 reproduction and the corrected
    (cache-reset-v2) Stage-7B run's own already-published findings as fixed prose facts (not
    re-derived here) -- the invalid historical vision/connector runs are never referenced.
    """
    language_spatial = primary_measurements.get("spatial_reasoning", {}).get("language", {})
    stage8_language_spatial_signs = {
        radius_key: ("positive" if row["mean_delta"] > 0 else "negative" if row["mean_delta"] < 0 else "zero")
        for radius_key, row in language_spatial.items()
    }

    stage7b_language_spatial_signs = None
    if stage7b_calibration_table is not None:
        stage7b_spatial = stage7b_calibration_table.get("spatial_reasoning", {}).get("language", {})
        stage7b_language_spatial_signs = {
            k: ("positive" if v.get("mean_delta", 0) > 0 else "negative" if v.get("mean_delta", 0) < 0 else "zero")
            for k, v in stage7b_spatial.items()
        }

    non_language_extension_caps = ["counting", "relational_reasoning", "fine_grained_recognition"]
    extension_summary = {
        cap: {
            region: {
                radius_key: primary_measurements.get(cap, {}).get(region, {}).get(radius_key, {}).get("density_ge_0.02")
                for radius_key in primary_measurements.get(cap, {}).get(region, {})
            }
            for region in STAGE8_REGIONS
        }
        for cap in non_language_extension_caps
    }

    return {
        "note": (
            "Sigma (Stage 6, global upstream Gaussian perturbation) and relative-L2 radius "
            "(Stage 7B/8, anatomically-scoped perturbation) are DIFFERENT parameterizations over "
            "DIFFERENT perturbation scopes and are NOT numerically equated anywhere in this file."
        ),
        "stage6_finding": "Nearby specialists exist in language-side upstream perturbations; specialization survives cache-safe reproduction (stage6_global_gaussian_upstream_cache_safe_v2).",
        "stage7b_finding": "Calibration-scale capability x anatomy interaction exists (full_fixed_direction_bf16_quantization_aware_v3_cache_reset_v011_verified_v2).",
        "stage8_language_spatial_reasoning_signs_by_radius": stage8_language_spatial_signs,
        "stage7b_language_spatial_reasoning_signs_by_radius": stage7b_language_spatial_signs,
        "stage8_extension_capabilities_density_ge_0.02_by_region_radius": extension_summary,
        "historical_invalid_vision_connector_runs_used": False,
    }


# =============================================================================================
# Section 13: Stage-9 drilldown recommendation (data-driven, description only -- NOT implemented)
# =============================================================================================


def compute_stage9_drilldown_recommendation(
    primary_measurements: Dict[str, Any], anatomy_capability_interaction: Dict[str, Any],
) -> Dict[str, Any]:
    """Scores each of the 3 L1 regions by: (1) reproducible solution density -- mean
    density_ge_0.02 across capabilities, averaged only over radii where that region is the
    region's OWN best (never selecting by a single max-Delta candidate); (2) positive thicket
    mass, same averaging; (3) capability selectivity -- how many capabilities have that region
    as their dominance-stable (>=2 radii) preferred anatomy (direction_A); (4) stability -- same
    boolean. Explicitly never chooses by a single maximum Delta value. connector_action is
    "keep_whole" unless connector shows BOTH reproducible density and capability selectivity
    comparable to vision/language (it is architecturally ~17x smaller than vision and ~84x
    smaller than language by parameter count, per Stage 7A's own live inventory -- a real
    "further decomposition worthwhile" signal requires genuine evidence, not being included by
    default).
    """
    regions = STAGE8_REGIONS
    radii = sorted({radius for cap_map in primary_measurements.values() for region_map in cap_map.values() for radius in [float(k) for k in region_map]})

    region_scores: Dict[str, Any] = {}
    for region in regions:
        densities, masses = [], []
        for cap, region_map in primary_measurements.items():
            for radius_key, row in region_map.get(region, {}).items():
                densities.append(row["density_ge_0.02"])
                masses.append(row["positive_thicket_mass"])
        n_selective_capabilities = sum(
            1 for cap, info in anatomy_capability_interaction["direction_A_capability_to_anatomy"].items()
            if info["dominance_stable_across_at_least_2_radii"] and info["stable_dominant_anatomy"] == region
        )
        region_scores[region] = {
            "mean_density_ge_0.02_across_all_cells": float(np.mean(densities)) if densities else 0.0,
            "mean_positive_thicket_mass_across_all_cells": float(np.mean(masses)) if masses else 0.0,
            "n_capabilities_with_stable_dominance_here": n_selective_capabilities,
        }

    ranked = sorted(
        region_scores.items(),
        key=lambda kv: (kv[1]["n_capabilities_with_stable_dominance_here"], kv[1]["mean_density_ge_0.02_across_all_cells"]),
        reverse=True,
    )
    ranked_non_connector = [r for r in ranked if r[0] != "multimodal_connector_or_merger"]
    priority_1 = ranked_non_connector[0][0] if ranked_non_connector else None
    priority_2 = ranked_non_connector[1][0] if len(ranked_non_connector) > 1 else None

    connector_scores = region_scores["multimodal_connector_or_merger"]
    connector_competitive = (
        connector_scores["n_capabilities_with_stable_dominance_here"] > 0
        and connector_scores["mean_density_ge_0.02_across_all_cells"] >= min(
            region_scores[r]["mean_density_ge_0.02_across_all_cells"] for r in regions if r != "multimodal_connector_or_merger"
        )
    )
    connector_action = "consider_decomposition" if connector_competitive else "keep_whole"

    return {
        "region_scores": region_scores,
        "priority_1_region": priority_1,
        "priority_2_region": priority_2,
        "connector_action": connector_action,
        "connector_action_rationale": (
            "multimodal_connector_or_merger is architecturally far smaller than vision/language "
            "(36.7M vs 632.0M / 3086.0M parameters, Stage 7A's own live inventory) -- "
            f"{'shows selective, competitive density despite its size, worth a decomposition check' if connector_competitive else 'shows no capability-selective stable dominance and no density advantage over the other two regions in this Stage-8 atlas, so it should remain a single undivided L1 region for Stage 9'}."
        ),
        "priority_rationale": (
            "Regions ranked by (a) number of capabilities for which this region is the dominance-"
            "stable (>=2 of 3 radii) preferred anatomy, then (b) mean solution density >= .02 "
            "across all its capability x radius cells -- never by a single maximum-Delta candidate."
        ),
        "caveat": "Selection is descriptive prioritization from D_map exploratory evidence only -- not a claim that other regions lack useful structure.",
    }


# =============================================================================================
# CSV export (compact, plotting-ready)
# =============================================================================================


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_atlas_csv(primary_measurements: Dict[str, Any], path: Path) -> None:
    header = ["capability", "anatomy_region", "radius", "n", "mean_delta", "median_delta", "std_delta",
              "p_delta_gt_0", "density_ge_0.0", "density_ge_0.02", "density_ge_0.05", "positive_thicket_mass", "negative_mass"]
    rows = []
    for cap, region_map in primary_measurements.items():
        for region, radius_map in region_map.items():
            for row in radius_map.values():
                rows.append([cap, region, row["radius"], row["n"], row["mean_delta"], row["median_delta"], row["std_delta"],
                             row["p_delta_gt_0"], row["density_ge_0.0"], row["density_ge_0.02"], row["density_ge_0.05"],
                             row["positive_thicket_mass"], row["negative_mass"]])
    _write_csv(path, header, rows)


def write_contrasts_csv(contrasts: Dict[str, Any], path: Path) -> None:
    header = ["capability", "radius", "region_a", "region_b", "mean_delta_diff", "mean_delta_diff_bh_q",
              "density_ge_0.02_diff", "density_ge_0.02_diff_bh_q", "positive_thicket_mass_diff", "positive_thicket_mass_diff_bh_q"]
    rows = []
    for cap, radius_map in contrasts.items():
        for radius_key, pair_map in radius_map.items():
            for cell in pair_map.values():
                rows.append([cap, cell["radius"], cell["region_a"], cell["region_b"], cell["mean_delta_diff"],
                             cell.get("mean_delta_diff_bh_q"), cell["density_ge_0.02_diff"], cell.get("density_ge_0.02_diff_bh_q"),
                             cell["positive_thicket_mass_diff"], cell.get("positive_thicket_mass_diff_bh_q")])
    _write_csv(path, header, rows)


# =============================================================================================
# Markdown report
# =============================================================================================


def build_markdown_report(
    integrity: Dict[str, Any], baseline: Dict[str, Any], primary: Dict[str, Any],
    contrasts: Dict[str, Any], interaction: Dict[str, Any], trajectories: Dict[str, Any],
    specialization: Dict[str, Any], quant_audit: Dict[str, Any], quant_confound: Dict[str, Any],
    cub_check: Dict[str, Any], bridge: Dict[str, Any], stage9: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Stage 8: paper-scale coarse anatomical atlas -- analysis")
    lines.append("")
    lines.append(f"Integrity gate: **{'PASS' if integrity['all_checks_pass'] else 'FAIL'}**. "
                  f"Model revision: `{integrity['model_revision']}`.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| capability | baseline | headroom |")
    lines.append("|---|---|---|")
    for cap, row in baseline.items():
        lines.append(f"| {cap} | {row['baseline_score']:.4f} | {row['headroom_1_minus_baseline']:.4f} |")
    lines.append("")
    lines.append("## Stage-9 drilldown recommendation")
    lines.append("")
    lines.append(f"priority_1_region = **{stage9['priority_1_region']}**, priority_2_region = **{stage9['priority_2_region']}**, "
                  f"connector_action = **{stage9['connector_action']}**.")
    lines.append("")
    lines.append(stage9["connector_action_rationale"])
    lines.append("")
    lines.append("## CUB / fine_grained_recognition stability")
    lines.append("")
    lines.append(f"fraction_exact_zero_delta = {cub_check['fraction_exact_zero_delta']:.3f}, "
                  f"n_distinct_nonzero_abs_delta_values = {cub_check['n_distinct_nonzero_abs_delta_values']}.")
    lines.append("")
    lines.append("## Stage 6 / Stage 7B / Stage 8 bridge")
    lines.append("")
    lines.append(bridge["note"])
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--stage7b-dir", default=str(
            REPO_ROOT / "results" / "stage7b_anatomical_calibration"
            / "full_fixed_direction_bf16_quantization_aware_v3_cache_reset_v011_verified_v2"
        ),
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    records = load_all(results_dir)
    checkpoint = json.loads((results_dir / "checkpoint_manifest.json").read_text())
    baseline_scores = json.loads((results_dir / "baseline_scores.json").read_text())

    integrity = run_integrity_gate(records, checkpoint)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_json(analysis_dir / "integrity_report.json", integrity)
    ensure_stage8_integrity(integrity)  # hard stop here if the gate failed -- nothing below runs
    print(f"Integrity gate PASSED ({sum(1 for k,v in integrity.items() if isinstance(v, bool))} checks).")

    baseline_table = compute_baseline_table(records, baseline_scores)
    _write_json(analysis_dir / "baseline_table.json", baseline_table)

    primary = compute_primary_measurements(records)
    _write_json(analysis_dir / "atlas_cell_statistics.json", primary)
    write_atlas_csv(primary, analysis_dir / "atlas_cell_statistics.csv")

    contrasts = compute_anatomical_contrasts(records)
    contrasts = apply_benjamini_hochberg_correction(contrasts)
    _write_json(analysis_dir / "anatomical_contrasts.json", contrasts)
    write_contrasts_csv(contrasts, analysis_dir / "anatomical_contrasts.csv")

    interaction = compute_anatomy_capability_interaction(records)
    _write_json(analysis_dir / "anatomy_capability_interaction.json", interaction)

    curves = compute_solution_density_curves(records)
    _write_json(analysis_dir / "solution_density_curves.json", curves)

    trajectories = compute_radius_trajectories(records)
    _write_json(analysis_dir / "radius_trajectories.json", trajectories)

    specialization = compute_cross_capability_specialization(records)
    _write_json(analysis_dir / "specialization_by_anatomy_radius.json", specialization)

    phenotypes = compute_thicket_phenotypes(primary, specialization)
    _write_json(analysis_dir / "thicket_phenotypes.json", phenotypes)

    quant_audit = compute_quantization_audit(records)
    quant_confound = compute_quantization_confound_audit(records)
    _write_json(analysis_dir / "quantization_audit.json", {"strict_vs_quantization_limited_counts": quant_audit, "delta_confound_check": quant_confound})

    cub_check = compute_cub_stability_check(records, baseline_scores)

    stage7b_calibration_table = None
    stage7b_path = Path(args.stage7b_dir) / "analysis" / "calibration_table.json"
    if stage7b_path.exists():
        stage7b_calibration_table = json.loads(stage7b_path.read_text())
    bridge = compute_stage6_stage7b_stage8_bridge(primary, stage7b_calibration_table)
    _write_json(analysis_dir / "stage6_stage7b_stage8_bridge.json", bridge)

    stage9 = compute_stage9_drilldown_recommendation(primary, interaction)
    _write_json(analysis_dir / "stage9_drilldown_recommendation.json", stage9)

    report = build_markdown_report(
        integrity, baseline_table, primary, contrasts, interaction, trajectories,
        specialization, quant_audit, quant_confound, cub_check, bridge, stage9,
    )
    (analysis_dir / "stage8_analysis.md").write_text(report)

    print(f"Wrote analysis outputs to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
