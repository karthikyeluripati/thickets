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
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_RADII, STAGE8_REGIONS  # noqa: E402
from neural_thickets_repro.thicket import diversity as thicket_diversity  # noqa: E402
from neural_thickets_repro.thicket import metrics as thicket_metrics  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402
from neural_thickets_repro.thicket_metrics import wilson_confidence_interval  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "stage8_coarse_anatomical_atlas" / "stage8_coarse_anatomical_atlas_3b_v1"

BOOTSTRAP_SEED = 20260825  # distinct from Stage 6/7B's own bootstrap seeds, deterministic
N_BOOTSTRAP = 10000


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

        out.setdefault(cap, {}).setdefault(region, {})[str(radius)] = {
            "capability": cap, "anatomy_region": region, "radius": radius, "n": n,
            "mean_delta": mean, "mean_delta_95ci_bootstrap": list(mean_ci),
            "std_delta": std, "median_delta": median, "min_delta": float(arr.min()), "max_delta": float(arr.max()),
            "p_delta_gt_0": p_gt0, "p_delta_gt_0_95ci_wilson": list(wilson_confidence_interval(n_gt0, n)),
            "p_delta_lt_0": p_lt0,
            "density_ge_0.0": n_ge0 / n, "density_ge_0.0_95ci_wilson": list(wilson_confidence_interval(n_ge0, n)),
            "density_ge_0.02": density[0.02], "density_ge_0.02_95ci_wilson": list(wilson_confidence_interval(n_ge_02, n)),
            "density_ge_0.05": density[0.05], "density_ge_0.05_95ci_wilson": list(wilson_confidence_interval(n_ge_05, n)),
            "positive_thicket_mass": mass,
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


def compute_anatomical_contrasts(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_cell = group_by_capability_region_radius(records)
    out: Dict[str, Any] = {}
    capabilities = sorted({r.capability for r in records})
    radii = sorted({r.radius for r in records})
    for cap in capabilities:
        for radius in radii:
            for region_a, region_b in _CONTRAST_PAIRS:
                key_a, key_b = (cap, region_a, radius), (cap, region_b, radius)
                if key_a not in by_cell or key_b not in by_cell:
                    continue
                deltas_a = np.asarray([r.delta for r in by_cell[key_a]], dtype=float)
                deltas_b = np.asarray([r.delta for r in by_cell[key_b]], dtype=float)
                mean_diff = float(np.mean(deltas_a) - np.mean(deltas_b))
                density_diff = _density_ge_002(deltas_a) - _density_ge_002(deltas_b)
                mass_diff = _positive_mass(deltas_a) - _positive_mass(deltas_b)
                seed = BOOTSTRAP_SEED + hash((cap, region_a, region_b, radius)) % 10_000
                out.setdefault(cap, {}).setdefault(str(radius), {})[f"{region_a}_vs_{region_b}"] = {
                    "capability": cap, "radius": radius, "region_a": region_a, "region_b": region_b,
                    "mean_delta_diff": mean_diff,
                    "mean_delta_diff_95ci_bootstrap": list(_bootstrap_diff_ci(deltas_a, deltas_b, _mean_axis1, seed)),
                    "density_ge_0.02_diff": density_diff,
                    "density_ge_0.02_diff_95ci_bootstrap": list(_bootstrap_diff_ci(deltas_a, deltas_b, _density_ge_002_axis1, seed + 1)),
                    "positive_thicket_mass_diff": mass_diff,
                    "positive_thicket_mass_diff_95ci_bootstrap": list(_bootstrap_diff_ci(deltas_a, deltas_b, _positive_mass_axis1, seed + 2)),
                }
    return out


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

    trajectories: Dict[str, Any] = {}
    sign_persistent = 0
    total_pairs = 0
    improvement_survives = 0
    total_positive_at_smallest = 0
    monotonic_nonincreasing = 0
    total_trajectories = 0
    disappearance_radius_counts: Dict[str, int] = {str(r): 0 for r in radii}

    for (region, idx, cap), radius_to_delta in by_family.items():
        ordered = [radius_to_delta.get(r) for r in radii]
        if any(v is None for v in ordered):
            continue  # incomplete trajectory (not evaluated at every radius) -- excluded, never fabricated
        total_trajectories += 1
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in ordered]
        for i in range(len(signs) - 1):
            total_pairs += 1
            if signs[i] == signs[i + 1]:
                sign_persistent += 1
        if ordered[0] > 0:
            total_positive_at_smallest += 1
            if all(v > 0 for v in ordered):
                improvement_survives += 1
            else:
                first_nonpositive_idx = next(i for i, v in enumerate(ordered) if v <= 0)
                disappearance_radius_counts[str(radii[first_nonpositive_idx])] += 1
        if all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)):
            monotonic_nonincreasing += 1

        family_id = f"{region}:{idx}"
        trajectories.setdefault(cap, {})[family_id] = {
            "region": region, "direction_index": idx, "capability": cap,
            "delta_by_radius": {str(r): v for r, v in zip(radii, ordered)},
        }

    rank_stability = _compute_rank_stability_across_radii(records, radii)

    return {
        "radii": radii,
        "n_complete_trajectories": total_trajectories,
        "sign_persistence_rate": (sign_persistent / total_pairs) if total_pairs else None,
        "improvement_survival_rate": (improvement_survives / total_positive_at_smallest) if total_positive_at_smallest else None,
        "monotonic_nonincreasing_fraction": (monotonic_nonincreasing / total_trajectories) if total_trajectories else None,
        "positive_direction_disappearance_radius_histogram": disappearance_radius_counts,
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
    by_region_radius = group_by_region_radius(records)
    out: Dict[str, Any] = {}
    for (region, radius), rows in sorted(by_region_radius.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        perturbation_ids, capabilities, matrix = build_delta_matrix(rows)
        m = matrix.shape[1]
        spearman = thicket_diversity.task_rank_correlation_matrix(matrix)
        discordance = thicket_diversity.spectral_discordance(matrix)
        overlap = {
            f"q_{q}": thicket_diversity.expert_overlap_matrix(matrix, q=q, q_is_fraction=True).tolist()
            for q in (0.1, 0.2)
        }
        signs = np.sign(matrix)
        sign_agreement = np.eye(m)
        for i in range(m):
            for j in range(i + 1, m):
                sign_agreement[i, j] = sign_agreement[j, i] = float(np.mean(signs[:, i] == signs[:, j]))
        n_improving = np.sum(matrix > 0, axis=1)
        improving_hist = {str(k): int(np.sum(n_improving == k)) for k in range(m + 1)}

        out.setdefault(region, {})[str(radius)] = {
            "region": region, "radius": radius, "n_perturbations": matrix.shape[0], "capabilities": list(capabilities),
            "spearman_6x6": spearman.tolist(), "spectral_discordance": discordance,
            "expert_overlap_jaccard": overlap, "sign_agreement_matrix": sign_agreement.tolist(),
            "improving_count_histogram": improving_hist,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    records = load_all(results_dir)

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(analysis_dir / "primary_measurements.json", compute_primary_measurements(records))
    _write_json(analysis_dir / "anatomical_contrasts.json", compute_anatomical_contrasts(records))
    _write_json(analysis_dir / "radius_trajectories.json", compute_radius_trajectories(records))
    _write_json(analysis_dir / "cross_capability_specialization.json", compute_cross_capability_specialization(records))
    _write_json(analysis_dir / "anatomical_selectivity_atlas.json", compute_anatomical_selectivity_atlas(records))
    _write_json(analysis_dir / "quantization_audit.json", compute_quantization_audit(records))

    print(f"Wrote analysis outputs to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
