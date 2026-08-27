"""Stage 11 S1 interim 3B-vs-7B whole-model scale analysis -- the FIRST authoritative cross-scale
readout for "Scaling Laws of Visual Neural Thickets", run against real, COMPLETE Stage-11
whole-model results at 3B and 7B. With exactly two scale points this is NEVER a scaling-law
claim: only "3B-to-7B scale trend" / "cross-scale comparison" language is used (enforced via
stage11_cross_scale_schema.classify_terminology_context, reused by import). The 32B/72B design
is frozen and untouched by anything in this module.

Reuses this project's own validated primitives throughout -- never reimplements them:
    run_global_visual_thicket_pilot.{load_records, build_delta_matrix}
    thicket.metrics.{quantiles, mean_std, solution_density, positive_thicket_mass,
        probability_of_improvement, probability_of_degradation, paired_bootstrap_confidence_interval}
    thicket_metrics.wilson_confidence_interval
    thicket.diversity.{task_rank_correlation_matrix, spectral_discordance}
    stage8_coarse_anatomical_atlas_analysis (as s8a): group_by_capability_region_radius,
        compute_cross_capability_specialization, compute_radius_trajectories,
        benjamini_hochberg, _permutation_p_value, _bootstrap_diff_ci, _positive_mass_axis1,
        _write_csv, _sanitize
    stage11_cross_scale_schema.classify_terminology_context

Key discipline points (see the accompanying task spec):
  - whole_model is a SINGLE anatomy_region value (unlike the S2 anatomy track's 3 regions), so
    the region dimension is trivial here -- s8a's region-aware machinery collapses cleanly.
  - 3B and 7B direction index i are NEVER geometrically paired (different parameter spaces).
    Cross-scale inference is always INDEPENDENT-SAMPLE (unpaired bootstrap/permutation).
  - The sampling unit for thicket density is the perturbation DIRECTION (n=64 per radius per
    scale), never the N=50 evaluation examples baked into each single delta.
  - Visual-macro / specialization bootstraps resample DIRECTION ROWS of the (64 x 6) delta
    matrix, never each capability column independently -- this preserves each direction's own
    six-capability outcome vector.

Usage (once real Stage-11 S1 3B and 7B whole-model data exists -- this module discovers it
structurally, never by a hardcoded directory name):
    python analysis/stage11_visual_thicket_scaling_interim_analysis.py [--results-root <path>] [--output-dir <path>]
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
for _p in (SRC_ROOT, ANALYSIS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from neural_thickets_repro.run_global_visual_thicket_pilot import build_delta_matrix, load_records  # noqa: E402
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import (  # noqa: E402
    STAGE8_CAPABILITIES, STAGE8_D_MAP_N, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_RADII,
)
from neural_thickets_repro.thicket import diversity as thicket_diversity  # noqa: E402
from neural_thickets_repro.thicket import metrics as thicket_metrics  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402
from neural_thickets_repro.thicket_metrics import wilson_confidence_interval  # noqa: E402

import stage8_coarse_anatomical_atlas_analysis as s8a  # noqa: E402
from stage11_cross_scale_schema import classify_terminology_context  # noqa: E402

# =================================================================================================
# Frozen constants -- reused BY IDENTITY from Stage 8 (the whole-model track deliberately reuses
# the exact same radii/capabilities/candidate-budget/D_map size, never recalibrated per scale).
# =================================================================================================

WHOLE_MODEL_REGION = "whole_model"
SCALES: Tuple[str, ...] = ("3B", "7B")
CAPABILITIES: Tuple[str, ...] = STAGE8_CAPABILITIES
RADII: Tuple[float, ...] = STAGE8_RADII
RADIUS_LABELS: Dict[float, str] = {RADII[0]: "small", RADII[1]: "mid", RADII[2]: "transition"}
EXPECTED_D_MAP_N = STAGE8_D_MAP_N
EXPECTED_N_DIRECTIONS = STAGE8_N_DIRECTIONS_PER_CELL
EXPECTED_UNIQUE_PERTURBATIONS = len(RADII) * EXPECTED_N_DIRECTIONS  # 192
EXPECTED_ROWS = EXPECTED_UNIQUE_PERTURBATIONS * len(CAPABILITIES)  # 1152

BOOTSTRAP_SEED = 20260827  # distinct namespace from Stage 8 (20260825) / Stage 9 / cross_scale_schema
N_BOOTSTRAP = 10_000
PERMUTATION_SEED = 20260828
N_PERMUTATIONS = 10_000

USEFUL_MARGIN = 0.02
STRONG_MARGIN = 0.05
HEADLINE_MARGINS: Tuple[float, ...] = (USEFUL_MARGIN, STRONG_MARGIN)
QUANTILE_LEVELS: Tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.95)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "stage11_whole_model_scaling"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage11_visual_thicket_scaling_analysis" / "interim_3b_7b_s1"

TERMINOLOGY_GUARD = classify_terminology_context(n_scales=len(SCALES))


# =================================================================================================
# Section 1: authoritative-only input discovery
# =================================================================================================


class Stage11InterimDataNotFoundError(RuntimeError):
    """No structurally-complete Stage-11 S1 whole-model run exists for the requested scale --
    never silently falls back to a smoke run or a partial run.
    """


class Stage11InterimAmbiguousRunError(RuntimeError):
    """More than one candidate run under results_root structurally qualifies as the authoritative
    complete run for a scale -- refuses to guess which one is real.
    """


def _looks_like_complete_whole_model_run(checkpoint: Dict[str, Any], manifest: Dict[str, Any], scale_label: str) -> bool:
    """Structural validation against FROZEN constants (never against values re-derived from the
    directory name itself) -- this is what actually excludes smoke runs (d_map_n=5,
    n_directions_per_cell=1) regardless of what a directory happens to be named.
    """
    return bool(
        checkpoint.get("track") == "whole_model"
        and checkpoint.get("scale_label") == scale_label
        and checkpoint.get("d_map_n") == EXPECTED_D_MAP_N
        and checkpoint.get("n_directions_per_cell") == EXPECTED_N_DIRECTIONS
        and checkpoint.get("expected_unique_perturbations") == EXPECTED_UNIQUE_PERTURBATIONS
        and checkpoint.get("expected_result_rows") == EXPECTED_ROWS
        and manifest.get("run_complete") is True
        and manifest.get("actual_unique_perturbations") == EXPECTED_UNIQUE_PERTURBATIONS
        and manifest.get("actual_result_rows") == EXPECTED_ROWS
    )


def discover_complete_whole_model_run(scale_label: str, results_root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    if not results_root.exists():
        raise Stage11InterimDataNotFoundError(
            f"No Stage-11 whole-model results root at {results_root} -- cannot locate a complete "
            f"{scale_label} run. This analysis PREPARES its full pipeline but refuses to fabricate "
            f"results without real data."
        )
    candidates: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        checkpoint_path = child / "checkpoint_manifest.json"
        manifest_path = child / "run_manifest.json"
        results_path = child / "results.jsonl"
        if not (checkpoint_path.exists() and manifest_path.exists() and results_path.exists()):
            continue
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if _looks_like_complete_whole_model_run(checkpoint, manifest, scale_label):
            candidates.append(child)
    if not candidates:
        raise Stage11InterimDataNotFoundError(
            f"No structurally-complete Stage-11 whole-model run found for scale={scale_label!r} "
            f"under {results_root} (require track=whole_model, d_map_n={EXPECTED_D_MAP_N}, "
            f"n_directions_per_cell={EXPECTED_N_DIRECTIONS}, run_complete=true, "
            f"{EXPECTED_UNIQUE_PERTURBATIONS} unique perturbations, {EXPECTED_ROWS} rows). "
            f"Smoke runs are structurally excluded by this check, never by directory name."
        )
    if len(candidates) > 1:
        raise Stage11InterimAmbiguousRunError(
            f"Multiple structurally-complete Stage-11 whole-model runs found for scale={scale_label!r} "
            f"under {results_root}: {[str(c) for c in candidates]} -- refusing to guess which is "
            f"authoritative."
        )
    return candidates[0]


def load_complete_whole_model_records(
    scale_label: str, results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Tuple[List[ExperimentResultRecord], Dict[str, Any], Dict[str, Any]]:
    run_dir = discover_complete_whole_model_run(scale_label, results_root)
    checkpoint = json.loads((run_dir / "checkpoint_manifest.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    records = load_records(run_dir / "results.jsonl")
    return records, checkpoint, manifest


# =================================================================================================
# Section 2: cross-scale integrity gate
# =================================================================================================


class Stage11InterimIntegrityError(RuntimeError):
    """The whole-model results fail per-scale or cross-scale hard verification -- never analyzed."""


def _per_scale_integrity(scale_label: str, records: Sequence[ExperimentResultRecord], checkpoint: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    checks["all_rows_whole_model_region"] = all(r.anatomy_region == WHOLE_MODEL_REGION for r in records)
    checks["all_rows_correct_scale"] = all(r.model_scale == scale_label for r in records)
    checks["checkpoint_track_whole_model"] = checkpoint.get("track") == "whole_model"
    checks["checkpoint_scale_matches"] = checkpoint.get("scale_label") == scale_label
    checks["three_frozen_radii"] = {r.radius for r in records} == set(RADII)
    checks["six_frozen_capabilities"] = {r.capability for r in records} == set(CAPABILITIES)
    checks["expected_total_rows"] = len(records) == EXPECTED_ROWS

    by_pid: Dict[str, List[ExperimentResultRecord]] = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, []).append(r)
    checks["expected_192_unique_perturbations"] = len(by_pid) == EXPECTED_UNIQUE_PERTURBATIONS
    checks["exactly_6_rows_per_perturbation"] = all(len(rows) == len(CAPABILITIES) for rows in by_pid.values())
    checks["same_candidate_evaluated_on_all_6_capabilities"] = all(
        {row.capability for row in rows} == set(CAPABILITIES) for rows in by_pid.values()
    )
    checks["no_duplicate_capability_rows_within_a_perturbation"] = all(
        len({row.capability for row in rows}) == len(rows) for rows in by_pid.values()
    )

    by_radius: Dict[float, set] = {}
    for pid, rows in by_pid.items():
        by_radius.setdefault(rows[0].radius, set()).add(pid)
    checks["no_missing_radius_cells"] = set(by_radius.keys()) == set(RADII)
    checks["exactly_64_perturbations_per_radius"] = all(len(v) == EXPECTED_N_DIRECTIONS for v in by_radius.values())
    checks["no_duplicate_perturbation_ids_across_run"] = len(set(by_pid.keys())) == EXPECTED_UNIQUE_PERTURBATIONS

    checks["d_map_n_50"] = checkpoint.get("d_map_n") == EXPECTED_D_MAP_N
    checks["run_complete_flag_true"] = manifest.get("run_complete") is True
    checks["actual_counts_match_expected"] = (
        manifest.get("actual_unique_perturbations") == EXPECTED_UNIQUE_PERTURBATIONS
        and manifest.get("actual_result_rows") == EXPECTED_ROWS
    )

    checks["perturbation_mode_anatomical_relative_l2"] = checkpoint.get("perturbation_mode") == "anatomical_relative_l2"
    checks["radius_realization_method_correct"] = checkpoint.get("radius_realization_method") == "fixed_direction_bf16_quantization_aware_v3"
    checks["restoration_mode_fixed_base"] = checkpoint.get("restoration_mode") == "fixed_base"
    checks["cache_policy_correct"] = checkpoint.get("multimodal_cache_policy") == "full_encoder_reset_vllm011_verified_v2"
    checks["enable_prefix_caching_false"] = checkpoint.get("enable_prefix_caching") is False

    model_revisions = {r.model_revision for r in records}
    checks["model_revision_consistent"] = len(model_revisions) == 1
    checks["model_revision"] = next(iter(model_revisions), None)
    mask_hashes = {r.parameter_mask_hash for r in records}
    checks["whole_model_mask_hash_consistent"] = len(mask_hashes) == 1
    checks["whole_model_mask_hash"] = next(iter(mask_hashes), None)
    checks["checkpoint_mask_hash_matches_records"] = checkpoint.get("whole_model_mask_hash") in mask_hashes

    non_meta_keys = [k for k in checks if k not in ("model_revision", "whole_model_mask_hash")]
    checks["all_checks_pass"] = all(bool(checks[k]) for k in non_meta_keys if isinstance(checks[k], bool))
    return checks


def run_cross_scale_whole_model_integrity_gate(
    records_by_scale: Dict[str, List[ExperimentResultRecord]],
    checkpoint_by_scale: Dict[str, Dict[str, Any]],
    manifest_by_scale: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    per_scale = {
        scale: _per_scale_integrity(scale, records_by_scale[scale], checkpoint_by_scale[scale], manifest_by_scale[scale])
        for scale in SCALES
    }
    ck3, ck7 = checkpoint_by_scale["3B"], checkpoint_by_scale["7B"]

    cross: Dict[str, Any] = {}
    cross["same_capability_set"] = set(ck3.get("capabilities", [])) == set(ck7.get("capabilities", [])) == set(CAPABILITIES)
    cross["same_capability_ordering"] = list(ck3.get("capabilities", [])) == list(ck7.get("capabilities", []))
    cross["same_d_map_subset_hashes"] = ck3.get("subset_hashes") == ck7.get("subset_hashes")
    cross["same_radii"] = list(ck3.get("radii", [])) == list(ck7.get("radii", [])) == list(RADII)
    cross["same_candidate_budget"] = ck3.get("n_directions_per_cell") == ck7.get("n_directions_per_cell") == EXPECTED_N_DIRECTIONS
    cross["same_d_map_n"] = ck3.get("d_map_n") == ck7.get("d_map_n") == EXPECTED_D_MAP_N
    cross["different_model_revision"] = ck3.get("model_revision") != ck7.get("model_revision") and ck3.get("model_revision") and ck7.get("model_revision")
    cross["different_whole_model_mask_hash"] = ck3.get("whole_model_mask_hash") != ck7.get("whole_model_mask_hash")
    cross["different_direction_seed_bank_hash"] = ck3.get("direction_seed_bank_hash") != ck7.get("direction_seed_bank_hash")
    cross["cross_scale_inference_mode"] = "independent_sample_never_paired"
    cross["all_ok"] = all(bool(v) for k, v in cross.items() if isinstance(v, bool))

    report = {
        "per_scale": per_scale,
        "cross_scale": cross,
        "all_ok": bool(per_scale["3B"]["all_checks_pass"] and per_scale["7B"]["all_checks_pass"] and cross["all_ok"]),
    }
    return report


def ensure_cross_scale_whole_model_integrity(report: Dict[str, Any]) -> None:
    if not report.get("all_ok"):
        failed = {}
        for scale in SCALES:
            failed[scale] = {k: v for k, v in report["per_scale"][scale].items() if isinstance(v, bool) and not v}
        failed["cross_scale"] = {k: v for k, v in report["cross_scale"].items() if isinstance(v, bool) and not v}
        raise Stage11InterimIntegrityError(f"Stage-11 S1 cross-scale integrity gate FAILED -- refusing to analyze. Failed checks: {failed}")


# =================================================================================================
# Section 3: baseline table
# =================================================================================================


def compute_baseline_table(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        row: Dict[str, Any] = {"capability": cap}
        canonical: Dict[str, Optional[float]] = {}
        for scale in SCALES:
            seen = sorted({r.base_score for r in records_by_scale[scale] if r.capability == cap})
            row[f"base_score_values_seen_{scale}"] = seen
            row[f"canonical_baseline_independent_of_radius_direction_{scale}"] = len(seen) == 1
            canonical[scale] = seen[0] if len(seen) == 1 else None
        row["baseline_3B"] = canonical["3B"]
        row["baseline_7B"] = canonical["7B"]
        row["absolute_baseline_difference_7B_minus_3B"] = (
            (canonical["7B"] - canonical["3B"]) if canonical["3B"] is not None and canonical["7B"] is not None else None
        )
        row["headroom_3B"] = (1.0 - canonical["3B"]) if canonical["3B"] is not None else None
        row["headroom_7B"] = (1.0 - canonical["7B"]) if canonical["7B"] is not None else None
        out[cap] = row
    return out


# =================================================================================================
# Section 4: primary 36-cell table (2 scales x 3 radii x 6 capabilities)
# =================================================================================================


def compute_cell_statistics(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scale in SCALES:
        by_cap_radius: Dict[Tuple[str, float], List[ExperimentResultRecord]] = {}
        for r in records_by_scale[scale]:
            by_cap_radius.setdefault((r.capability, r.radius), []).append(r)
        for (cap, radius), rows in by_cap_radius.items():
            deltas = [r.delta for r in rows]
            arr = np.asarray(deltas, dtype=float)
            n = int(arr.size)
            mean, std = thicket_metrics.mean_std(deltas)
            q = thicket_metrics.quantiles(deltas, qs=QUANTILE_LEVELS)
            p_gt0 = thicket_metrics.probability_of_improvement(deltas)
            p_lt0 = thicket_metrics.probability_of_degradation(deltas)
            density = thicket_metrics.solution_density(deltas, margins=(0.0, USEFUL_MARGIN, STRONG_MARGIN))
            pos_mass = thicket_metrics.positive_thicket_mass(deltas)
            neg_mass = float(np.mean(np.clip(-arr, 0.0, None)))

            n_ge0 = int(np.sum(arr >= 0.0))
            n_ge_useful = int(np.sum(arr >= USEFUL_MARGIN))
            n_ge_strong = int(np.sum(arr >= STRONG_MARGIN))
            n_gt0 = int(np.sum(arr > 0))

            mean_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=np.mean, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            pos_mass_ci = thicket_metrics.paired_bootstrap_confidence_interval(
                deltas, statistic_fn=lambda d: float(np.mean(np.clip(d, 0.0, None))), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 1,
            )
            neg_mass_ci = thicket_metrics.paired_bootstrap_confidence_interval(
                deltas, statistic_fn=lambda d: float(np.mean(np.clip(-d, 0.0, None))), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 2,
            )
            q90_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=lambda d: float(np.quantile(d, 0.9)), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 3)
            q95_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=lambda d: float(np.quantile(d, 0.95)), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 4)

            key = f"{scale}:{cap}:{radius}"
            out[key] = {
                "scale": scale, "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius], "n": n,
                "mean_delta": mean, "mean_delta_95ci_bootstrap": list(mean_ci),
                "median_delta": q[0.5], "std_delta": std, "min_delta": float(arr.min()), "max_delta": float(arr.max()),
                "p_delta_gt_0": p_gt0, "p_delta_gt_0_95ci_wilson": list(wilson_confidence_interval(n_gt0, n)),
                "p_delta_lt_0": p_lt0,
                "density_ge_0.0": density[0.0], "density_ge_0.0_95ci_wilson": list(wilson_confidence_interval(n_ge0, n)),
                "density_ge_0.02": density[USEFUL_MARGIN], "density_ge_0.02_95ci_wilson": list(wilson_confidence_interval(n_ge_useful, n)),
                "density_ge_0.05": density[STRONG_MARGIN], "density_ge_0.05_95ci_wilson": list(wilson_confidence_interval(n_ge_strong, n)),
                "positive_thicket_mass": pos_mass, "positive_thicket_mass_95ci_bootstrap": list(pos_mass_ci),
                "negative_mass": neg_mass, "negative_mass_95ci_bootstrap": list(neg_mass_ci),
                "q25": q[0.25], "q50": q[0.5], "q75": q[0.75],
                "q90": q[0.9], "q90_95ci_bootstrap": list(q90_ci),
                "q95": q[0.95], "q95_95ci_bootstrap": list(q95_ci),
            }
    return out


def write_cell_statistics_csv(cell_stats: Dict[str, Any], path: Path) -> None:
    header = [
        "scale", "capability", "radius", "radius_label", "n", "mean_delta", "median_delta", "std_delta",
        "min_delta", "max_delta", "p_delta_gt_0", "p_delta_lt_0", "density_ge_0.0", "density_ge_0.02",
        "density_ge_0.05", "positive_thicket_mass", "negative_mass", "q25", "q50", "q75", "q90", "q95",
    ]
    rows = []
    for row in cell_stats.values():
        rows.append([row[h] for h in header])
    s8a._write_csv(path, header, rows)


# =================================================================================================
# Section 5: solution-density curves (PRIMARY) -- one common margin grid across both scales
# =================================================================================================


def build_common_margin_grid(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Tuple[float, ...]:
    base = [0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]
    max_delta = max(
        (r.delta for records in records_by_scale.values() for r in records if r.delta > 0.0),
        default=0.0,
    )
    grid = list(base)
    m = 0.10
    while m < max_delta - 1e-9:
        m = round(m + 0.02, 10)
        grid.append(m)
    return tuple(sorted(set(round(x, 10) for x in grid)))


def compute_solution_density_curves(records_by_scale: Dict[str, List[ExperimentResultRecord]], margin_grid: Sequence[float]) -> Dict[str, Any]:
    curves: Dict[str, Any] = {"margin_grid": list(margin_grid), "by_scale_capability_radius": {}}
    for scale in SCALES:
        by_cap_radius: Dict[Tuple[str, float], List[float]] = {}
        for r in records_by_scale[scale]:
            by_cap_radius.setdefault((r.capability, r.radius), []).append(r.delta)
        for (cap, radius), deltas in by_cap_radius.items():
            density_map = thicket_metrics.solution_density(deltas, margins=margin_grid)
            densities = [density_map[float(m)] for m in margin_grid]
            curves["by_scale_capability_radius"].setdefault(scale, {}).setdefault(cap, {})[str(radius)] = {
                "scale": scale, "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                "margins": list(margin_grid), "density": densities,
            }
    return curves


def ensure_solution_density_curves_monotonic(curves: Dict[str, Any]) -> None:
    for scale, cap_map in curves["by_scale_capability_radius"].items():
        for cap, radius_map in cap_map.items():
            for radius_key, row in radius_map.items():
                d = row["density"]
                if any(d[i] < d[i + 1] - 1e-12 for i in range(len(d) - 1)):
                    raise ValueError(f"Solution-density curve non-monotonic at scale={scale} cap={cap} radius={radius_key}: {d}")


def write_solution_density_curves_csv(curves: Dict[str, Any], path: Path) -> None:
    header = ["scale", "capability", "radius", "radius_label", "margin", "density"]
    rows = []
    for scale, cap_map in curves["by_scale_capability_radius"].items():
        for cap, radius_map in cap_map.items():
            for row in radius_map.values():
                for m, d in zip(row["margins"], row["density"]):
                    rows.append([scale, cap, row["radius"], row["radius_label"], m, d])
    s8a._write_csv(path, header, rows)


# =================================================================================================
# Section 6: cross-scale solution-density differences + headline-margin statistical tests
# =================================================================================================


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0.0:
        return None if numerator == 0.0 else float("inf")
    return numerator / denominator


def compute_cross_scale_solution_density_differences(curves: Dict[str, Any], margin_grid: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    by_scale = curves["by_scale_capability_radius"]
    for cap in CAPABILITIES:
        for radius in RADII:
            d3 = by_scale["3B"][cap][str(radius)]["density"]
            d7 = by_scale["7B"][cap][str(radius)]["density"]
            per_margin = []
            for m, v3, v7 in zip(margin_grid, d3, d7):
                per_margin.append({
                    "margin": m, "density_3B": v3, "density_7B": v7,
                    "difference_7B_minus_3B": v7 - v3,
                    "ratio_7B_over_3B": _safe_ratio(v7, v3),
                })
            out.setdefault(cap, {})[str(radius)] = {"capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius], "per_margin": per_margin}
    return out


def _density_ge_axis1(m: float):
    return lambda mat: (mat >= m).mean(axis=1)


def compute_headline_margin_statistical_tests(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in HEADLINE_MARGINS:
        cells: Dict[str, Any] = {}
        stat_fn = _density_ge_axis1(m)
        for cap in CAPABILITIES:
            for radius in RADII:
                a = np.asarray([r.delta for r in records_by_scale["3B"] if r.capability == cap and r.radius == radius], dtype=float)
                b = np.asarray([r.delta for r in records_by_scale["7B"] if r.capability == cap and r.radius == radius], dtype=float)
                density_a, density_b = float(np.mean(a >= m)), float(np.mean(b >= m))
                diff = density_b - density_a
                seed_key = (cap, radius, m)
                seed = BOOTSTRAP_SEED + hash(seed_key) % 10_000
                perm_seed = PERMUTATION_SEED + hash(seed_key) % 10_000
                ci = s8a._bootstrap_diff_ci(b, a, stat_fn, seed)
                p_value = s8a._permutation_p_value(b, a, stat_fn, diff, perm_seed)
                key = f"{cap}:{radius}"
                cells[key] = {
                    "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius], "margin": m,
                    "density_3B": density_a, "density_7B": density_b, "difference_7B_minus_3B": diff,
                    "difference_95ci_bootstrap": list(ci), "permutation_p_value": p_value,
                }
        pvalues = [cell["permutation_p_value"] for cell in cells.values()]
        qvalues = s8a.benjamini_hochberg(pvalues)
        for cell, q in zip(cells.values(), qvalues):
            cell["bh_q_value"] = q
            cell["bh_significant_fdr_0.05"] = q < 0.05
            if cell["bh_significant_fdr_0.05"] and cell["difference_7B_minus_3B"] > 0:
                cell["verdict"] = "significant_increase"
            elif cell["bh_significant_fdr_0.05"] and cell["difference_7B_minus_3B"] < 0:
                cell["verdict"] = "significant_decrease"
            else:
                cell["verdict"] = "non_significant_trend"
        out[f"m={m}"] = cells
    return out


# =================================================================================================
# Section 7: visual-macro solution density (equal-weight across 6 capabilities, candidate-row-
# preserving bootstrap -- resamples the 64 DIRECTION rows, never each capability column separately)
# =================================================================================================


def _matrix_for_radius(records: Sequence[ExperimentResultRecord], radius: float) -> Tuple[Tuple[str, ...], Tuple[str, ...], np.ndarray]:
    subset = [r for r in records if r.radius == radius]
    return build_delta_matrix(subset)


def _macro_density_bootstrap_distribution(matrix: np.ndarray, margin: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    resampled = matrix[idx]  # (N_BOOTSTRAP, n, n_capabilities)
    return (resampled >= margin).mean(axis=(1, 2))


def compute_visual_macro_solution_density(records_by_scale: Dict[str, List[ExperimentResultRecord]], margin_grid: Sequence[float]) -> Dict[str, Any]:
    matrices: Dict[str, Dict[float, np.ndarray]] = {scale: {} for scale in SCALES}
    for scale in SCALES:
        for radius in RADII:
            _, _, matrix = _matrix_for_radius(records_by_scale[scale], radius)
            matrices[scale][radius] = matrix

    by_scale_radius: Dict[str, Any] = {}
    for scale in SCALES:
        for radius in RADII:
            matrix = matrices[scale][radius]
            by_margin = {}
            for m in margin_grid:
                point = float((matrix >= m).mean())
                boot_seed = BOOTSTRAP_SEED + hash((scale, radius, m)) % 10_000
                dist = _macro_density_bootstrap_distribution(matrix, m, boot_seed)
                lo, hi = np.percentile(dist, [2.5, 97.5])
                by_margin[str(m)] = {"macro_density": point, "ci_95_bootstrap": [float(lo), float(hi)]}
            by_scale_radius.setdefault(scale, {})[str(radius)] = {"radius": radius, "radius_label": RADIUS_LABELS[radius], "by_margin": by_margin}

    difference: Dict[str, Any] = {}
    for radius in RADII:
        by_margin = {}
        for m in margin_grid:
            point3 = by_scale_radius["3B"][str(radius)]["by_margin"][str(m)]["macro_density"]
            point7 = by_scale_radius["7B"][str(radius)]["by_margin"][str(m)]["macro_density"]
            seed3 = BOOTSTRAP_SEED + hash(("3B", radius, m)) % 10_000
            seed7 = BOOTSTRAP_SEED + hash(("7B", radius, m)) % 10_000
            dist3 = _macro_density_bootstrap_distribution(matrices["3B"][radius], m, seed3)
            dist7 = _macro_density_bootstrap_distribution(matrices["7B"][radius], m, seed7)
            diff_dist = dist7 - dist3
            lo, hi = np.percentile(diff_dist, [2.5, 97.5])
            by_margin[str(m)] = {
                "difference_7B_minus_3B": point7 - point3,
                "difference_95ci_bootstrap": [float(lo), float(hi)],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
        difference[str(radius)] = {"radius": radius, "radius_label": RADIUS_LABELS[radius], "by_margin": by_margin}

    return {"by_scale_radius": by_scale_radius, "difference_7B_minus_3B": difference, "note": "visual-macro scale trend, NOT a scaling law (2 scale points)"}


# =================================================================================================
# Section 8: performance-density / full-distribution shift (exact 1-D Wasserstein-1, permutation)
# =================================================================================================


def wasserstein_1_equal_size(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact closed-form 1-D optimal-transport distance for two EQUAL-SIZE samples: monotone
    sorted-to-sorted coupling is provably optimal in 1-D, so no scipy/linear-programming needed.
    """
    arr_a, arr_b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    if arr_a.size != arr_b.size:
        raise ValueError(f"wasserstein_1_equal_size requires equal-size samples, got {arr_a.size} and {arr_b.size}")
    return float(np.mean(np.abs(arr_a - arr_b)))


def _wasserstein_permutation_p_value(a: np.ndarray, b: np.ndarray, observed_w1: float, seed: int) -> float:
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((N_PERMUTATIONS, pooled.size)), axis=1)
    permuted = pooled[perm_idx]
    perm_a, perm_b = np.sort(permuted[:, :n_a], axis=1), np.sort(permuted[:, n_a:], axis=1)
    perm_w1 = np.mean(np.abs(perm_a - perm_b), axis=1)
    return float(np.mean(perm_w1 >= observed_w1))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    support = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.array([np.mean(a <= x) for x in support])
    cdf_b = np.array([np.mean(b <= x) for x in support])
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compute_performance_density_comparison(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for radius in RADII:
            a = np.asarray([r.delta for r in records_by_scale["3B"] if r.capability == cap and r.radius == radius], dtype=float)
            b = np.asarray([r.delta for r in records_by_scale["7B"] if r.capability == cap and r.radius == radius], dtype=float)
            mean_diff = float(np.mean(b) - np.mean(a))
            median_diff = float(np.median(b) - np.median(a))
            pos_mass_diff = float(np.mean(np.clip(b, 0, None)) - np.mean(np.clip(a, 0, None)))
            neg_mass_diff = float(np.mean(np.clip(-b, 0, None)) - np.mean(np.clip(-a, 0, None)))
            q90_diff = float(np.quantile(b, 0.9) - np.quantile(a, 0.9))
            q95_diff = float(np.quantile(b, 0.95) - np.quantile(a, 0.95))
            w1 = wasserstein_1_equal_size(a, b)
            perm_seed = PERMUTATION_SEED + hash((cap, radius, "w1")) % 10_000
            w1_perm_p = _wasserstein_permutation_p_value(a, b, w1, perm_seed)
            ks = _ks_statistic(a, b)

            whole = abs(median_diff) >= USEFUL_MARGIN or abs(mean_diff) >= USEFUL_MARGIN
            tail = abs(q90_diff) >= USEFUL_MARGIN or abs(q95_diff) >= USEFUL_MARGIN
            if whole and tail:
                shift_pattern = "whole_distribution_and_tail_shift"
            elif whole:
                shift_pattern = "whole_distribution_shift"
            elif tail:
                shift_pattern = "sparse_tail_shift_only"
            else:
                shift_pattern = "no_meaningful_shift"

            key = f"{cap}:{radius}"
            cells[key] = {
                "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                "mean_delta_diff_7B_minus_3B": mean_diff, "median_delta_diff_7B_minus_3B": median_diff,
                "positive_mass_diff_7B_minus_3B": pos_mass_diff, "negative_mass_diff_7B_minus_3B": neg_mass_diff,
                "q90_diff_7B_minus_3B": q90_diff, "q95_diff_7B_minus_3B": q95_diff,
                "wasserstein_1_distance": w1, "wasserstein_1_permutation_p_value": w1_perm_p,
                "ks_statistic_secondary_only": ks,
                "shift_pattern": shift_pattern,
            }
    pvalues = [cell["wasserstein_1_permutation_p_value"] for cell in cells.values()]
    qvalues = s8a.benjamini_hochberg(pvalues)
    for cell, q in zip(cells.values(), qvalues):
        cell["wasserstein_1_bh_q_value"] = q
        cell["distribution_shift_significant_fdr_0.05"] = q < 0.05
    return cells


def write_performance_density_comparison_csv(cells: Dict[str, Any], path: Path) -> None:
    header = [
        "capability", "radius", "radius_label", "mean_delta_diff_7B_minus_3B", "median_delta_diff_7B_minus_3B",
        "positive_mass_diff_7B_minus_3B", "negative_mass_diff_7B_minus_3B", "q90_diff_7B_minus_3B",
        "q95_diff_7B_minus_3B", "wasserstein_1_distance", "wasserstein_1_bh_q_value",
        "distribution_shift_significant_fdr_0.05", "shift_pattern",
    ]
    rows = [[cell[h] for h in header] for cell in cells.values()]
    s8a._write_csv(path, header, rows)


# =================================================================================================
# Section 9: more experts vs stronger experts (bootstrap-CI-gated classification, FROZEN logic)
# =================================================================================================


MORE_VS_STRONGER_LABELS: Tuple[str, ...] = ("more_and_stronger", "more_not_stronger", "stronger_not_more", "neither_clear", "decreases")

DECISION_LOGIC_NOTE = (
    "more := density(m=0.02) bootstrap-CI-supported significant increase 7B>3B; "
    "stronger := positive_thicket_mass bootstrap-CI-supported significant increase 7B>3B; "
    "less_dense / weaker := the mirrored significant-decrease conditions. "
    "more_and_stronger if both hold; more_not_stronger / stronger_not_more if exactly one holds; "
    "decreases if less_dense or weaker holds without a compensating significant increase; "
    "neither_clear otherwise. Never decided from point estimates alone."
)


def compute_strength_contrasts(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for radius in RADII:
            a = np.asarray([r.delta for r in records_by_scale["3B"] if r.capability == cap and r.radius == radius], dtype=float)
            b = np.asarray([r.delta for r in records_by_scale["7B"] if r.capability == cap and r.radius == radius], dtype=float)
            mass_diff = float(np.mean(np.clip(b, 0, None)) - np.mean(np.clip(a, 0, None)))
            q90_diff = float(np.quantile(b, 0.9) - np.quantile(a, 0.9))
            seed = BOOTSTRAP_SEED + hash((cap, radius, "strength")) % 10_000
            perm_seed = PERMUTATION_SEED + hash((cap, radius, "strength")) % 10_000
            mass_ci = s8a._bootstrap_diff_ci(b, a, s8a._positive_mass_axis1, seed)
            mass_p = s8a._permutation_p_value(b, a, s8a._positive_mass_axis1, mass_diff, perm_seed)
            q90_axis1 = lambda m: np.quantile(m, 0.9, axis=1)  # noqa: E731
            q90_ci = s8a._bootstrap_diff_ci(b, a, q90_axis1, seed + 1)
            q90_p = s8a._permutation_p_value(b, a, q90_axis1, q90_diff, perm_seed + 1)
            key = f"{cap}:{radius}"
            cells[key] = {
                "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                "positive_mass_diff_7B_minus_3B": mass_diff, "positive_mass_diff_95ci_bootstrap": list(mass_ci),
                "positive_mass_diff_permutation_p": mass_p,
                "q90_diff_7B_minus_3B": q90_diff, "q90_diff_95ci_bootstrap": list(q90_ci),
                "q90_diff_permutation_p": q90_p,
            }
    mass_p = [c["positive_mass_diff_permutation_p"] for c in cells.values()]
    q90_p = [c["q90_diff_permutation_p"] for c in cells.values()]
    mass_q = s8a.benjamini_hochberg(mass_p)
    q90_q = s8a.benjamini_hochberg(q90_p)
    for cell, mq, qq in zip(cells.values(), mass_q, q90_q):
        cell["positive_mass_diff_bh_q"] = mq
        cell["q90_diff_bh_q"] = qq
        mass_sig = mq < 0.05
        q90_sig = qq < 0.05
        if mass_sig and cell["positive_mass_diff_7B_minus_3B"] > 0:
            cell["mass_verdict"] = "significant_increase"
        elif mass_sig and cell["positive_mass_diff_7B_minus_3B"] < 0:
            cell["mass_verdict"] = "significant_decrease"
        else:
            cell["mass_verdict"] = "non_significant"
        if q90_sig and cell["q90_diff_7B_minus_3B"] > 0:
            cell["q90_verdict"] = "significant_increase"
        elif q90_sig and cell["q90_diff_7B_minus_3B"] < 0:
            cell["q90_verdict"] = "significant_decrease"
        else:
            cell["q90_verdict"] = "non_significant"
        strength_signals = {cell["mass_verdict"], cell["q90_verdict"]}
        if "significant_increase" in strength_signals and "significant_decrease" not in strength_signals:
            cell["strength_verdict"] = "significant_increase"
        elif "significant_decrease" in strength_signals and "significant_increase" not in strength_signals:
            cell["strength_verdict"] = "significant_decrease"
        else:
            cell["strength_verdict"] = "non_significant"
    return cells


def classify_more_vs_stronger(density_tests: Dict[str, Any], strength_contrasts: Dict[str, Any]) -> Dict[str, Any]:
    density_m002 = density_tests[f"m={USEFUL_MARGIN}"]
    out: Dict[str, Any] = {"decision_logic": DECISION_LOGIC_NOTE}
    cells: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for radius in RADII:
            key = f"{cap}:{radius}"
            density_verdict = density_m002[key]["verdict"]
            strength_verdict = strength_contrasts[key]["strength_verdict"]
            more = density_verdict == "significant_increase"
            less_dense = density_verdict == "significant_decrease"
            stronger = strength_verdict == "significant_increase"
            weaker = strength_verdict == "significant_decrease"
            if more and stronger:
                label = "more_and_stronger"
            elif more and not stronger:
                label = "more_not_stronger"
            elif stronger and not more:
                label = "stronger_not_more"
            elif less_dense or weaker:
                label = "decreases"
            else:
                label = "neither_clear"
            cells[key] = {
                "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                "density_verdict": density_verdict, "strength_verdict": strength_verdict, "classification": label,
            }
    out["cells"] = cells
    return out


# =================================================================================================
# Section 10: radius x scale landscape (conservative classification, only 3 radii available)
# =================================================================================================


RADIUS_SCALE_LABELS: Tuple[str, ...] = (
    "peak_radius_stable", "peak_radius_reorganizes",
    "evidence_of_broader_useful_neighborhood", "evidence_of_narrower_useful_neighborhood",
    "insufficient_resolution",
)


def compute_radius_scale_landscape(cell_stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        matrix: Dict[str, Any] = {}
        peak_radius: Dict[str, float] = {}
        useful_radius_count: Dict[str, int] = {}
        for scale in SCALES:
            best_density, best_radius = -1.0, None
            n_useful = 0
            for radius in RADII:
                row = cell_stats[f"{scale}:{cap}:{radius}"]
                matrix.setdefault(scale, {})[RADIUS_LABELS[radius]] = {
                    "density_ge_0.02": row["density_ge_0.02"], "density_ge_0.05": row["density_ge_0.05"],
                    "positive_thicket_mass": row["positive_thicket_mass"], "negative_mass": row["negative_mass"],
                }
                if row["density_ge_0.02"] > best_density:
                    best_density, best_radius = row["density_ge_0.02"], radius
                if row["density_ge_0.02"] > 0.0:
                    n_useful += 1
            peak_radius[scale] = best_radius
            useful_radius_count[scale] = n_useful

        transition_radius = RADII[-1]
        neg_mass_3B = cell_stats[f"3B:{cap}:{transition_radius}"]["negative_mass"]
        neg_mass_7B = cell_stats[f"7B:{cap}:{transition_radius}"]["negative_mass"]
        neg_mass_diff = neg_mass_7B - neg_mass_3B
        if abs(neg_mass_diff) < 0.005:
            question_b = "no_clear_difference"
        elif neg_mass_diff > 0:
            question_b = "more_destructive_at_7B"
        else:
            question_b = "less_destructive_at_7B"

        if peak_radius["3B"] == peak_radius["7B"]:
            question_a = "peak_radius_stable"
        else:
            question_a = "peak_radius_reorganizes"

        if useful_radius_count["7B"] > useful_radius_count["3B"]:
            question_c = "evidence_of_broader_useful_neighborhood"
        elif useful_radius_count["7B"] < useful_radius_count["3B"]:
            question_c = "evidence_of_narrower_useful_neighborhood"
        else:
            question_c = "insufficient_resolution"

        out[cap] = {
            "capability": cap, "matrix": matrix,
            "peak_radius_3B": peak_radius["3B"], "peak_radius_7B": peak_radius["7B"],
            "question_A_peak_radius_change": question_a,
            "question_B_relative_destructiveness_at_transition_radius": question_b,
            "negative_mass_diff_at_transition_radius_7B_minus_3B": neg_mass_diff,
            "question_C_broader_or_narrower_useful_neighborhood": question_c,
            "n_useful_radii_3B": useful_radius_count["3B"], "n_useful_radii_7B": useful_radius_count["7B"],
        }
    return out


# =================================================================================================
# Section 11: radius trajectories WITHIN each scale (reuse s8a.compute_radius_trajectories per
# scale -- whole_model's direction_family_id is "whole_model:<idx>", matching s8a's expected
# "<region>:<idx>" shape directly). Direction index i is NEVER paired across scales.
# =================================================================================================


def compute_radius_trajectories_by_scale(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    traj = {scale: s8a.compute_radius_trajectories(records_by_scale[scale]) for scale in SCALES}
    scalar_fields = (
        "sign_persistence_rate", "improvement_survival_rate",
        "positive_at_small_remains_positive_at_mid_rate", "positive_at_small_remains_positive_at_transition_rate",
        "monotonic_nonincreasing_fraction", "monotonic_nondecreasing_fraction", "non_monotonic_fraction",
    )
    comparison = {}
    for field in scalar_fields:
        v3, v7 = traj["3B"].get(field), traj["7B"].get(field)
        comparison[field] = {"3B": v3, "7B": v7, "difference_7B_minus_3B": (v7 - v3) if v3 is not None and v7 is not None else None}
    return {
        "3B": traj["3B"], "7B": traj["7B"],
        "summary_comparison_7B_minus_3B": comparison,
        "cross_scale_pairing_note": "Direction index i is NOT paired across 3B and 7B (different parameter spaces); only the SUMMARY statistics above are compared, never individual trajectories.",
    }


# =================================================================================================
# Section 12: specialization / diversity scale trend (Spectral Discordance, per radius)
# =================================================================================================


def _discordance_bootstrap_distribution(matrix: np.ndarray, seed: int, n_bootstrap: int = N_BOOTSTRAP) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    out = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        out[i] = thicket_diversity.spectral_discordance(matrix[idx])
    return out


def compute_diversity_scale_trend(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    spec_by_scale = {scale: s8a.compute_cross_capability_specialization(records_by_scale[scale]) for scale in SCALES}
    out: Dict[str, Any] = {}
    for radius in RADII:
        d3 = spec_by_scale["3B"][WHOLE_MODEL_REGION][str(radius)]
        d7 = spec_by_scale["7B"][WHOLE_MODEL_REGION][str(radius)]
        _, _, matrix3 = _matrix_for_radius(records_by_scale["3B"], radius)
        _, _, matrix7 = _matrix_for_radius(records_by_scale["7B"], radius)
        seed3 = BOOTSTRAP_SEED + hash(("3B", radius, "discordance")) % 10_000
        seed7 = BOOTSTRAP_SEED + hash(("7B", radius, "discordance")) % 10_000
        dist3 = _discordance_bootstrap_distribution(matrix3, seed3)
        dist7 = _discordance_bootstrap_distribution(matrix7, seed7)
        diff_dist = dist7 - dist3
        lo, hi = np.percentile(diff_dist, [2.5, 97.5])
        diff = d7["spectral_discordance"] - d3["spectral_discordance"]
        if lo > 0:
            trend = "increases_3B_to_7B"
        elif hi < 0:
            trend = "decreases_3B_to_7B"
        else:
            trend = "no_clear_change"
        out[str(radius)] = {
            "radius": radius, "radius_label": RADIUS_LABELS[radius],
            "spectral_discordance_3B": d3["spectral_discordance"], "spectral_discordance_7B": d7["spectral_discordance"],
            "difference_7B_minus_3B": diff, "difference_95ci_bootstrap": [float(lo), float(hi)],
            "spearman_6x6_3B": d3["spearman_6x6"], "spearman_6x6_7B": d7["spearman_6x6"],
            "sign_agreement_matrix_3B": d3["sign_agreement_matrix"], "sign_agreement_matrix_7B": d7["sign_agreement_matrix"],
            "improving_count_histogram_3B": d3["improving_count_histogram"], "improving_count_histogram_7B": d7["improving_count_histogram"],
            "trend": trend,
        }
    return out


# =================================================================================================
# Section 13: headroom sensitivity (SECONDARY ONLY -- raw Delta remains primary throughout)
# =================================================================================================


def compute_headroom_sensitivity(records_by_scale: Dict[str, List[ExperimentResultRecord]], baseline_table: Dict[str, Any], cell_stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for radius in RADII:
            row: Dict[str, Any] = {"capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius]}
            normalized_stats: Dict[str, Any] = {}
            for scale in SCALES:
                headroom = baseline_table[cap][f"headroom_{scale}"]
                deltas = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.radius == radius], dtype=float)
                if headroom is None or headroom <= 0.0:
                    normalized_stats[scale] = {"applicable": False, "reason": "no_remaining_headroom"}
                    continue
                normalized = deltas / headroom
                normalized_stats[scale] = {
                    "applicable": True,
                    "normalized_positive_mass": float(np.mean(np.clip(normalized, 0, None))),
                    "normalized_q90": float(np.quantile(normalized, 0.9)),
                    "normalized_q95": float(np.quantile(normalized, 0.95)),
                }
            row["normalized_by_scale"] = normalized_stats

            raw_mass_diff = cell_stats[f"7B:{cap}:{radius}"]["positive_thicket_mass"] - cell_stats[f"3B:{cap}:{radius}"]["positive_thicket_mass"]
            raw_direction = "increase" if raw_mass_diff > 1e-9 else ("decrease" if raw_mass_diff < -1e-9 else "flat")
            row["raw_positive_mass_diff_7B_minus_3B"] = raw_mass_diff
            row["raw_conclusion_direction"] = raw_direction

            if normalized_stats["3B"].get("applicable") and normalized_stats["7B"].get("applicable"):
                norm_diff = normalized_stats["7B"]["normalized_positive_mass"] - normalized_stats["3B"]["normalized_positive_mass"]
                norm_direction = "increase" if norm_diff > 1e-9 else ("decrease" if norm_diff < -1e-9 else "flat")
                row["normalized_positive_mass_diff_7B_minus_3B"] = norm_diff
                row["normalized_conclusion_direction"] = norm_direction
                if norm_direction == raw_direction:
                    row["headroom_sensitivity_verdict"] = "raw_conclusion_persists"
                elif norm_direction == "flat" or raw_direction == "flat":
                    row["headroom_sensitivity_verdict"] = "raw_conclusion_weakens"
                else:
                    row["headroom_sensitivity_verdict"] = "raw_conclusion_reverses"
            else:
                row["normalized_conclusion_direction"] = None
                row["headroom_sensitivity_verdict"] = "not_applicable"
            out[f"{cap}:{radius}"] = row
    return out


# =================================================================================================
# Section 14: capability-by-capability scale trend
# =================================================================================================


CAPABILITY_SCALE_RESPONSE_LABELS: Tuple[str, ...] = (
    "thicket_expands_3B_to_7B", "thicket_strengthens_3B_to_7B", "mixed_scale_response",
    "little_change", "thicket_contracts_3B_to_7B",
)


def compute_capability_scale_summaries(
    baseline_table: Dict[str, Any], cell_stats: Dict[str, Any], more_vs_stronger: Dict[str, Any],
    radius_landscape: Dict[str, Any], headroom: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        labels_by_radius = [more_vs_stronger["cells"][f"{cap}:{radius}"]["classification"] for radius in RADII]
        expands = sum(1 for lbl in labels_by_radius if lbl in ("more_and_stronger", "more_not_stronger"))
        strengthens_only = sum(1 for lbl in labels_by_radius if lbl == "stronger_not_more")
        contracts = sum(1 for lbl in labels_by_radius if lbl == "decreases")
        little = sum(1 for lbl in labels_by_radius if lbl == "neither_clear")

        counts = {"expands": expands, "strengthens_only": strengthens_only, "contracts": contracts, "little_change": little}
        majority_label, majority_count = max(counts.items(), key=lambda kv: kv[1])
        if majority_count >= 2:
            classification = {
                "expands": "thicket_expands_3B_to_7B", "strengthens_only": "thicket_strengthens_3B_to_7B",
                "contracts": "thicket_contracts_3B_to_7B", "little_change": "little_change",
            }[majority_label]
        else:
            classification = "mixed_scale_response"

        density_change = {str(radius): cell_stats[f"7B:{cap}:{radius}"]["density_ge_0.02"] - cell_stats[f"3B:{cap}:{radius}"]["density_ge_0.02"] for radius in RADII}
        density_change_strong = {str(radius): cell_stats[f"7B:{cap}:{radius}"]["density_ge_0.05"] - cell_stats[f"3B:{cap}:{radius}"]["density_ge_0.05"] for radius in RADII}
        positive_mass_change = {str(radius): cell_stats[f"7B:{cap}:{radius}"]["positive_thicket_mass"] - cell_stats[f"3B:{cap}:{radius}"]["positive_thicket_mass"] for radius in RADII}
        upper_tail_change = {str(radius): cell_stats[f"7B:{cap}:{radius}"]["q90"] - cell_stats[f"3B:{cap}:{radius}"]["q90"] for radius in RADII}

        out[cap] = {
            "capability": cap,
            "baseline_change_7B_minus_3B": baseline_table[cap]["absolute_baseline_difference_7B_minus_3B"],
            "density_ge_0.02_change_by_radius": density_change,
            "density_ge_0.05_change_by_radius": density_change_strong,
            "positive_mass_change_by_radius": positive_mass_change,
            "upper_tail_q90_change_by_radius": upper_tail_change,
            "radius_preference_change": radius_landscape[cap]["question_A_peak_radius_change"],
            "peak_radius_3B": radius_landscape[cap]["peak_radius_3B"], "peak_radius_7B": radius_landscape[cap]["peak_radius_7B"],
            "headroom_sensitivity_by_radius": {str(radius): headroom[f"{cap}:{radius}"]["headroom_sensitivity_verdict"] for radius in RADII},
            "more_vs_stronger_by_radius": {str(radius): lbl for radius, lbl in zip(RADII, labels_by_radius)},
            "classification": classification,
        }
    return out


# =================================================================================================
# Section 16: interim claim gate (S1-S5, never "scaling law established")
# =================================================================================================


CLAIM_VERDICTS: Tuple[str, ...] = ("strongly_supported_3B_to_7B", "supported_3B_to_7B", "mixed", "unsupported")


def evaluate_interim_claim_gate(
    cell_stats: Dict[str, Any], density_tests: Dict[str, Any], more_vs_stronger: Dict[str, Any],
    radius_landscape: Dict[str, Any], diversity_trend: Dict[str, Any],
) -> Dict[str, Any]:
    n_cells = len(CAPABILITIES) * len(RADII)  # 18

    # S1: nearby whole-model visual specialists exist at both scales
    n_scales_with_any_density = 0
    for scale in SCALES:
        n_positive_cells = sum(1 for cap in CAPABILITIES for radius in RADII if cell_stats[f"{scale}:{cap}:{radius}"]["density_ge_0.02"] > 0.0)
        if n_positive_cells >= n_cells // 2:
            n_scales_with_any_density += 1
    if n_scales_with_any_density == 2:
        s1 = "strongly_supported_3B_to_7B"
    elif n_scales_with_any_density == 1:
        s1 = "supported_3B_to_7B"
    else:
        s1 = "unsupported"

    # S2: solution density changes systematically 3B -> 7B
    density_cells = density_tests[f"m={USEFUL_MARGIN}"]
    n_significant = sum(1 for c in density_cells.values() if c["verdict"] != "non_significant_trend")
    n_increase = sum(1 for c in density_cells.values() if c["verdict"] == "significant_increase")
    n_decrease = sum(1 for c in density_cells.values() if c["verdict"] == "significant_decrease")
    if n_significant >= n_cells * 0.5 and max(n_increase, n_decrease) >= n_significant * 0.7:
        s2 = "strongly_supported_3B_to_7B"
    elif n_significant >= 1:
        s2 = "supported_3B_to_7B" if max(n_increase, n_decrease) >= n_significant * 0.6 else "mixed"
    else:
        s2 = "unsupported"

    # S3: specialist strength changes 3B -> 7B
    n_stronger = sum(1 for cap in CAPABILITIES for radius in RADII if more_vs_stronger["cells"][f"{cap}:{radius}"]["classification"] in ("more_and_stronger", "stronger_not_more"))
    n_weaker = sum(1 for cap in CAPABILITIES for radius in RADII if more_vs_stronger["cells"][f"{cap}:{radius}"]["classification"] == "decreases")
    if n_stronger >= n_cells * 0.5:
        s3 = "strongly_supported_3B_to_7B"
    elif n_stronger >= 1 or n_weaker >= 1:
        s3 = "supported_3B_to_7B" if max(n_stronger, n_weaker) >= 3 else "mixed"
    else:
        s3 = "unsupported"

    # S4: useful-radius behavior changes 3B -> 7B
    reorganizes = sum(1 for cap in CAPABILITIES if radius_landscape[cap]["question_A_peak_radius_change"] == "peak_radius_reorganizes")
    broader_or_narrower = sum(1 for cap in CAPABILITIES if radius_landscape[cap]["question_C_broader_or_narrower_useful_neighborhood"] != "insufficient_resolution")
    if reorganizes + broader_or_narrower >= len(CAPABILITIES):
        s4 = "strongly_supported_3B_to_7B"
    elif reorganizes + broader_or_narrower >= 1:
        s4 = "supported_3B_to_7B"
    else:
        s4 = "unsupported"

    # S5: specialization/diversity changes 3B -> 7B
    n_diversity_change = sum(1 for row in diversity_trend.values() if row["trend"] != "no_clear_change")
    if n_diversity_change == len(RADII):
        s5 = "strongly_supported_3B_to_7B"
    elif n_diversity_change >= 1:
        s5 = "supported_3B_to_7B"
    else:
        s5 = "unsupported"

    return {
        "S1_nearby_specialists_exist_both_scales": s1,
        "S2_solution_density_changes_systematically": s2,
        "S3_specialist_strength_changes": s3,
        "S4_useful_radius_behavior_changes": s4,
        "S5_specialization_diversity_changes": s5,
        "terminology_guard": TERMINOLOGY_GUARD,
        "note": "Two scale points only -- 'scaling law established' is NEVER a valid conclusion under any outcome above.",
    }


# =================================================================================================
# Section 17: pre-7B-anatomy questions A-H
# =================================================================================================


def answer_pre_anatomy_questions(
    density_tests: Dict[str, Any], more_vs_stronger: Dict[str, Any], performance_density: Dict[str, Any],
    radius_landscape: Dict[str, Any], diversity_trend: Dict[str, Any], headroom: Dict[str, Any],
    capability_summaries: Dict[str, Any],
) -> Dict[str, Any]:
    density_m002 = density_tests[f"m={USEFUL_MARGIN}"]
    n_denser = sum(1 for c in density_m002.values() if c["verdict"] == "significant_increase")
    n_sparser = sum(1 for c in density_m002.values() if c["verdict"] == "significant_decrease")
    answer_a = {"denser_at_7B": n_denser > n_sparser, "n_cells_significantly_denser_at_7B": n_denser, "n_cells_significantly_sparser_at_7B": n_sparser, "n_cells_total": len(density_m002)}

    n_stronger = sum(1 for c in more_vs_stronger["cells"].values() if c["classification"] in ("more_and_stronger", "stronger_not_more"))
    n_weaker = sum(1 for c in more_vs_stronger["cells"].values() if c["classification"] == "decreases")
    answer_b = {"stronger_at_7B": n_stronger > n_weaker, "n_cells_stronger": n_stronger, "n_cells_weaker": n_weaker}

    n_tail_grows = sum(1 for c in performance_density.values() if c["q90_diff_7B_minus_3B"] > 0 or c["q95_diff_7B_minus_3B"] > 0)
    n_tail_shrinks = sum(1 for c in performance_density.values() if c["q90_diff_7B_minus_3B"] < 0 and c["q95_diff_7B_minus_3B"] < 0)
    answer_c = {"positive_tail_grows_at_7B": n_tail_grows > n_tail_shrinks, "n_cells_tail_grows": n_tail_grows, "n_cells_tail_shrinks": n_tail_shrinks}

    n_neg_mass_grows = sum(1 for c in performance_density.values() if c["negative_mass_diff_7B_minus_3B"] > 0)
    n_neg_mass_shrinks = sum(1 for c in performance_density.values() if c["negative_mass_diff_7B_minus_3B"] < 0)
    answer_d = {"destructive_tail_grows_at_7B": n_neg_mass_grows > n_neg_mass_shrinks, "n_cells_grows": n_neg_mass_grows, "n_cells_shrinks": n_neg_mass_shrinks}

    n_broader = sum(1 for cap in CAPABILITIES if radius_landscape[cap]["question_C_broader_or_narrower_useful_neighborhood"] == "evidence_of_broader_useful_neighborhood")
    n_narrower = sum(1 for cap in CAPABILITIES if radius_landscape[cap]["question_C_broader_or_narrower_useful_neighborhood"] == "evidence_of_narrower_useful_neighborhood")
    answer_e = {"useful_radius_moves_outward_at_7B": n_broader > n_narrower, "n_capabilities_broader": n_broader, "n_capabilities_narrower": n_narrower}

    n_more_specialized = sum(1 for row in diversity_trend.values() if row["trend"] == "increases_3B_to_7B")
    n_less_specialized = sum(1 for row in diversity_trend.values() if row["trend"] == "decreases_3B_to_7B")
    answer_f = {"specialization_increases_at_7B": n_more_specialized > n_less_specialized, "n_radii_increases": n_more_specialized, "n_radii_decreases": n_less_specialized}

    classifications = {cap: capability_summaries[cap]["classification"] for cap in CAPABILITIES}
    n_distinct_classifications = len(set(classifications.values()))
    answer_g = {"uniform_across_capability": n_distinct_classifications == 1, "n_distinct_response_classes": n_distinct_classifications, "classification_by_capability": classifications}

    n_persists = sum(1 for row in headroom.values() if row["headroom_sensitivity_verdict"] == "raw_conclusion_persists")
    n_reverses = sum(1 for row in headroom.values() if row["headroom_sensitivity_verdict"] == "raw_conclusion_reverses")
    answer_h = {"headroom_explains_main_findings": n_reverses > n_persists, "n_cells_conclusion_persists": n_persists, "n_cells_conclusion_reverses": n_reverses}

    return {
        "A_denser_at_equal_radius": answer_a, "B_stronger": answer_b, "C_positive_tail_grows": answer_c,
        "D_destructive_tail_shrinks_or_grows": answer_d, "E_useful_radius_moves_outward": answer_e,
        "F_specialization_increases": answer_f, "G_uniform_or_heterogeneous": answer_g, "H_headroom_explains_findings": answer_h,
        "note": "Answers derived from real 3B/7B data. Does NOT alter the frozen 32B/72B design.",
    }


# =================================================================================================
# Section 18: figure data (non-publication-styled)
# =================================================================================================


def build_fig_s1a(curves: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["scale", "capability", "radius", "radius_label", "margin", "density"]
    rows = []
    for scale, cap_map in curves["by_scale_capability_radius"].items():
        for cap, radius_map in cap_map.items():
            for row in radius_map.values():
                for m, d in zip(row["margins"], row["density"]):
                    rows.append([scale, cap, row["radius"], row["radius_label"], m, d])
    return header, rows


def build_fig_s1b(macro: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["scale", "radius", "radius_label", "margin", "macro_density", "ci_lo", "ci_hi"]
    rows = []
    for scale, radius_map in macro["by_scale_radius"].items():
        for radius_key, row in radius_map.items():
            for m, cell in row["by_margin"].items():
                rows.append([scale, row["radius"], row["radius_label"], m, cell["macro_density"], cell["ci_95_bootstrap"][0], cell["ci_95_bootstrap"][1]])
    return header, rows


def build_fig_s2(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Tuple[List[str], List[List[Any]]]:
    header = ["scale", "capability", "radius", "radius_label", "delta_sorted", "ecdf"]
    rows = []
    for scale in SCALES:
        for cap in CAPABILITIES:
            for radius in RADII:
                deltas = sorted(r.delta for r in records_by_scale[scale] if r.capability == cap and r.radius == radius)
                n = len(deltas)
                for i, d in enumerate(deltas):
                    rows.append([scale, cap, radius, RADIUS_LABELS[radius], d, (i + 1) / n])
    return header, rows


def build_fig_s3(radius_landscape: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["capability", "radius_label", "scale", "density_ge_0.02", "density_ge_0.05", "positive_thicket_mass", "negative_mass"]
    rows = []
    for cap, entry in radius_landscape.items():
        for scale, radius_map in entry["matrix"].items():
            for radius_label, cell in radius_map.items():
                rows.append([cap, radius_label, scale, cell["density_ge_0.02"], cell["density_ge_0.05"], cell["positive_thicket_mass"], cell["negative_mass"]])
    return header, rows


def build_fig_s4(diversity_trend: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["radius", "radius_label", "scale", "spectral_discordance"]
    rows = []
    for radius_key, row in diversity_trend.items():
        rows.append([row["radius"], row["radius_label"], "3B", row["spectral_discordance_3B"]])
        rows.append([row["radius"], row["radius_label"], "7B", row["spectral_discordance_7B"]])
    return header, rows


def build_fig_s5(density_tests: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["capability", "radius", "radius_label", "density_diff_7B_minus_3B"]
    rows = []
    for cell in density_tests[f"m={USEFUL_MARGIN}"].values():
        rows.append([cell["capability"], cell["radius"], cell["radius_label"], cell["difference_7B_minus_3B"]])
    return header, rows


# =================================================================================================
# Markdown summary + main orchestration
# =================================================================================================


def build_markdown_summary(
    integrity: Dict[str, Any], baseline: Dict[str, Any], claim_gate: Dict[str, Any], capability_summaries: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Stage 11 S1: interim 3B-vs-7B whole-model scale analysis")
    lines.append("")
    lines.append(f"Cross-scale integrity gate: **{'PASS' if integrity['all_ok'] else 'FAIL'}**.")
    lines.append("")
    lines.append("This is NOT a scaling-law claim (only 2 scale points). Terminology guard: "
                  f"{TERMINOLOGY_GUARD['allowed_terms']}.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| capability | baseline_3B | baseline_7B | headroom_3B | headroom_7B |")
    lines.append("|---|---|---|---|---|")
    for cap, row in baseline.items():
        lines.append(f"| {cap} | {row['baseline_3B']:.4f} | {row['baseline_7B']:.4f} | {row['headroom_3B']:.4f} | {row['headroom_7B']:.4f} |")
    lines.append("")
    lines.append("## Interim claim gate (S1-S5)")
    lines.append("")
    for k in ("S1_nearby_specialists_exist_both_scales", "S2_solution_density_changes_systematically",
              "S3_specialist_strength_changes", "S4_useful_radius_behavior_changes", "S5_specialization_diversity_changes"):
        lines.append(f"- {k}: **{claim_gate[k]}**")
    lines.append("")
    lines.append("## Capability-by-capability scale response")
    lines.append("")
    lines.append("| capability | classification |")
    lines.append("|---|---|")
    for cap, row in capability_summaries.items():
        lines.append(f"| {cap} | {row['classification']} |")
    lines.append("")
    lines.append("DO NOT START 7B ANATOMY. DO NOT ENABLE 32B. DO NOT ENABLE 72B. "
                  "DO NOT CHANGE THE FROZEN SCALE EXPERIMENT.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    try:
        records_by_scale: Dict[str, List[ExperimentResultRecord]] = {}
        checkpoint_by_scale: Dict[str, Dict[str, Any]] = {}
        manifest_by_scale: Dict[str, Dict[str, Any]] = {}
        for scale in SCALES:
            records, checkpoint, manifest = load_complete_whole_model_records(scale, results_root)
            records_by_scale[scale] = records
            checkpoint_by_scale[scale] = checkpoint
            manifest_by_scale[scale] = manifest
    except Stage11InterimDataNotFoundError as exc:
        print(f"PREPARED, NOT RUN: {exc}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    integrity = run_cross_scale_whole_model_integrity_gate(records_by_scale, checkpoint_by_scale, manifest_by_scale)
    (output_dir / "integrity_report.json").write_text(json.dumps(s8a._sanitize(integrity), indent=2))
    ensure_cross_scale_whole_model_integrity(integrity)
    print("Cross-scale integrity gate PASSED.")

    baseline_table = compute_baseline_table(records_by_scale)
    (output_dir / "baseline_table.json").write_text(json.dumps(s8a._sanitize(baseline_table), indent=2))

    cell_stats = compute_cell_statistics(records_by_scale)
    (output_dir / "cell_statistics_3b_7b.json").write_text(json.dumps(s8a._sanitize(cell_stats), indent=2))
    write_cell_statistics_csv(cell_stats, output_dir / "cell_statistics_3b_7b.csv")

    margin_grid = build_common_margin_grid(records_by_scale)
    curves = compute_solution_density_curves(records_by_scale, margin_grid)
    ensure_solution_density_curves_monotonic(curves)
    (output_dir / "solution_density_curves.json").write_text(json.dumps(s8a._sanitize(curves), indent=2))
    write_solution_density_curves_csv(curves, output_dir / "solution_density_curves.csv")

    density_diffs = compute_cross_scale_solution_density_differences(curves, margin_grid)
    density_tests = compute_headline_margin_statistical_tests(records_by_scale)
    (output_dir / "solution_density_scale_differences.json").write_text(
        json.dumps(s8a._sanitize({"per_margin_differences": density_diffs, "headline_margin_tests": density_tests}), indent=2)
    )

    macro_density = compute_visual_macro_solution_density(records_by_scale, margin_grid)
    (output_dir / "visual_macro_solution_density.json").write_text(json.dumps(s8a._sanitize(macro_density), indent=2))

    performance_density = compute_performance_density_comparison(records_by_scale)
    (output_dir / "performance_density_comparison.json").write_text(json.dumps(s8a._sanitize(performance_density), indent=2))
    write_performance_density_comparison_csv(performance_density, output_dir / "performance_density_comparison.csv")

    strength_contrasts = compute_strength_contrasts(records_by_scale)
    more_vs_stronger = classify_more_vs_stronger(density_tests, strength_contrasts)
    (output_dir / "more_vs_stronger_classification.json").write_text(json.dumps(s8a._sanitize(more_vs_stronger), indent=2))

    radius_landscape = compute_radius_scale_landscape(cell_stats)
    (output_dir / "radius_scale_landscape.json").write_text(json.dumps(s8a._sanitize(radius_landscape), indent=2))

    radius_trajectories = compute_radius_trajectories_by_scale(records_by_scale)
    (output_dir / "radius_trajectories_by_scale.json").write_text(json.dumps(s8a._sanitize(radius_trajectories), indent=2))

    diversity_trend = compute_diversity_scale_trend(records_by_scale)
    (output_dir / "diversity_scale_trend.json").write_text(json.dumps(s8a._sanitize(diversity_trend), indent=2))

    headroom = compute_headroom_sensitivity(records_by_scale, baseline_table, cell_stats)
    (output_dir / "headroom_sensitivity.json").write_text(json.dumps(s8a._sanitize(headroom), indent=2))

    capability_summaries = compute_capability_scale_summaries(baseline_table, cell_stats, more_vs_stronger, radius_landscape, headroom)
    (output_dir / "capability_scale_summaries.json").write_text(json.dumps(s8a._sanitize(capability_summaries), indent=2))

    statistical_tests = {"headline_margin_density_tests": density_tests, "strength_contrasts": strength_contrasts, "performance_density_comparison": performance_density}
    (output_dir / "statistical_tests.json").write_text(json.dumps(s8a._sanitize(statistical_tests), indent=2))

    claim_gate = evaluate_interim_claim_gate(cell_stats, density_tests, more_vs_stronger, radius_landscape, diversity_trend)
    (output_dir / "interim_claim_gate.json").write_text(json.dumps(s8a._sanitize(claim_gate), indent=2))

    pre_anatomy = answer_pre_anatomy_questions(density_tests, more_vs_stronger, performance_density, radius_landscape, diversity_trend, headroom, capability_summaries)
    (output_dir / "pre_anatomy_questions_a_to_h.json").write_text(json.dumps(s8a._sanitize(pre_anatomy), indent=2))

    figure_dir = output_dir / "figure_schemas"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, builder, args_ in (
        ("fig_s1a_solution_density_curves.csv", build_fig_s1a, (curves,)),
        ("fig_s1b_visual_macro_scale_trend.csv", build_fig_s1b, (macro_density,)),
        ("fig_s2_performance_density.csv", build_fig_s2, (records_by_scale,)),
        ("fig_s3_radius_scale_matrix.csv", build_fig_s3, (radius_landscape,)),
        ("fig_s4_diversity_scale_trend.csv", build_fig_s4, (diversity_trend,)),
        ("fig_s5_capability_scale_response.csv", build_fig_s5, (density_tests,)),
    ):
        header, rows = builder(*args_)
        s8a._write_csv(figure_dir / name, header, rows)

    summary_md = build_markdown_summary(integrity, baseline_table, claim_gate, capability_summaries)
    (output_dir / "stage11_interim_3b_7b_summary.md").write_text(summary_md)

    print(f"Stage-11 S1 interim 3B-vs-7B analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
