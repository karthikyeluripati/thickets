"""Stage 11 S2 interim 3B-vs-7B ANATOMY-RESOLVED scale analysis -- the first authoritative
cross-scale readout for "Where Do Visual Experts Live? Mapping Neural Thickets in
Vision-Language Models", run against the real, COMPLETE Stage-8 (3B) and Stage-11 (7B) coarse
anatomical atlas runs. Two scale points only: NEVER "scaling law"/"anisotropic scaling law"/
"universal trend"/"monotonic scaling" -- only "anatomical scale trend" / "3B-to-7B anatomical
change" / "scale-dependent anatomy" / "heterogeneous anatomical response".

Reuses stage8_coarse_anatomical_atlas_analysis.py (s8a) heavily -- Stage 8's OWN analysis module
was already built for exactly this shape (3 anatomy regions x 3 radii x 6 capabilities, single
scale), so most of the per-scale anatomy-resolved machinery here is s8a's own functions called
once per scale, never reimplemented:
    s8a.group_by_capability_region_radius / group_by_region_radius
    s8a.compute_baseline_table
    s8a.compute_solution_density_curves (fixed SOLUTION_DENSITY_MARGIN_GRID -- already common
        across any two calls, by construction, since it never depends on the data)
    s8a.compute_anatomical_contrasts / apply_benjamini_hochberg_correction (the exact pairwise
        vision/connector/language contrast machinery Section 8 asks for)
    s8a.compute_cross_capability_specialization (the exact 6x6/discordance/sign-agreement/
        histogram/tradeoff/transfer machinery Section 13 asks for)
    s8a.benjamini_hochberg / _permutation_p_value / _bootstrap_diff_ci / _positive_mass_axis1 /
        _write_csv / _sanitize
    thicket.metrics / thicket_metrics.wilson_confidence_interval / thicket.diversity
    run_global_visual_thicket_pilot.{load_records, build_delta_matrix}

Cross-scale independence discipline (unchanged from the S1 whole-model interim analysis):
3B direction i and 7B direction i are NEVER geometrically paired -- cross-scale comparisons are
always independent-sample (unpaired bootstrap/permutation). The sampling unit for thicket density
is the perturbation DIRECTION (n=64 per region/radius/scale), never the N=50 evaluation examples.
Region-level macro/specialization bootstraps resample DIRECTION ROWS of the (64 x 6) delta matrix
within a (region, radius, scale) cell, never each capability column independently.

Usage (once real Stage-8 3B and Stage-11 7B anatomy data exists -- discovered structurally under
each track's own results root, never by a hardcoded run-signature string):
    python analysis/stage11_anatomical_scale_interim_analysis.py [--stage8-dir <path>] [--stage11-anatomy-dir <path>] [--s1-summary <path>] [--output-dir <path>]
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
    STAGE8_CAPABILITIES, STAGE8_D_MAP_N, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_RADII, STAGE8_REGIONS,
)
from neural_thickets_repro.thicket import diversity as thicket_diversity  # noqa: E402
from neural_thickets_repro.thicket import metrics as thicket_metrics  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402
from neural_thickets_repro.thicket_metrics import wilson_confidence_interval  # noqa: E402

import stage8_coarse_anatomical_atlas_analysis as s8a  # noqa: E402
from stage11_cross_scale_schema import classify_terminology_context  # noqa: E402

# =================================================================================================
# Frozen constants -- reused BY IDENTITY from Stage 8
# =================================================================================================

SCALES: Tuple[str, ...] = ("3B", "7B")
REGIONS: Tuple[str, ...] = STAGE8_REGIONS
CAPABILITIES: Tuple[str, ...] = STAGE8_CAPABILITIES
RADII: Tuple[float, ...] = STAGE8_RADII
RADIUS_LABELS: Dict[float, str] = {RADII[0]: "small", RADII[1]: "mid", RADII[2]: "transition"}
EXPECTED_D_MAP_N = STAGE8_D_MAP_N
EXPECTED_N_DIRECTIONS = STAGE8_N_DIRECTIONS_PER_CELL
EXPECTED_UNIQUE_PERTURBATIONS = len(REGIONS) * len(RADII) * EXPECTED_N_DIRECTIONS  # 576
EXPECTED_ROWS = EXPECTED_UNIQUE_PERTURBATIONS * len(CAPABILITIES)  # 3456

BOOTSTRAP_SEED = 20260829  # distinct namespace from every prior stage's bootstrap seed
N_BOOTSTRAP = 10_000
DISCORDANCE_N_BOOTSTRAP = 3_000  # pure-Python-loop cost (SVD-free but still per-replicate); reduced from N_BOOTSTRAP purely for runtime, still deterministic and still >> what's needed for a 95% CI
PERMUTATION_SEED = 20260830
N_PERMUTATIONS = 10_000

USEFUL_MARGIN = 0.02
STRONG_MARGIN = 0.05
HEADLINE_MARGINS: Tuple[float, ...] = (USEFUL_MARGIN, STRONG_MARGIN)
QUANTILE_LEVELS: Tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.95)

DEFAULT_STAGE8_DIR = REPO_ROOT / "results" / "stage8_coarse_anatomical_atlas"
DEFAULT_STAGE11_ANATOMY_DIR = REPO_ROOT / "results" / "stage11_coarse_anatomical_atlas_7b"
DEFAULT_S1_SUMMARY_PATH = REPO_ROOT / "results" / "stage11_visual_thicket_scaling_analysis" / "interim_3b_7b_s1" / "capability_scale_summaries.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage11_visual_thicket_scaling_analysis" / "interim_3b_7b_anatomy"

TERMINOLOGY_GUARD = classify_terminology_context(n_scales=len(SCALES))


# =================================================================================================
# Section 1: authoritative-only discovery (structural, never a hardcoded run-signature string --
# excludes smoke by construction since smoke has d_map_n=5/n_directions_per_cell=1/totals 9,54)
# =================================================================================================


class Stage11AnatomyInterimDataNotFoundError(RuntimeError):
    """No structurally-complete anatomy run exists under the given results root."""


class Stage11AnatomyInterimAmbiguousRunError(RuntimeError):
    """More than one candidate run structurally qualifies as authoritative -- refuses to guess."""


def _looks_like_complete_anatomy_run(checkpoint: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
    return bool(
        set(checkpoint.get("regions", [])) == set(REGIONS)
        and checkpoint.get("d_map_n") == EXPECTED_D_MAP_N
        and checkpoint.get("n_directions_per_cell") == EXPECTED_N_DIRECTIONS
        and checkpoint.get("expected_unique_perturbations") == EXPECTED_UNIQUE_PERTURBATIONS
        and checkpoint.get("expected_result_rows") == EXPECTED_ROWS
        and manifest.get("run_complete") is True
        and manifest.get("actual_unique_perturbations") == EXPECTED_UNIQUE_PERTURBATIONS
        and manifest.get("actual_result_rows") == EXPECTED_ROWS
    )


def discover_complete_anatomy_run(results_root: Path) -> Path:
    if not results_root.exists():
        raise Stage11AnatomyInterimDataNotFoundError(
            f"No anatomy results root at {results_root} -- cannot locate a complete run. This "
            f"analysis PREPARES its full pipeline but refuses to fabricate results without real data."
        )
    candidates: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        checkpoint_path, manifest_path, results_path = child / "checkpoint_manifest.json", child / "run_manifest.json", child / "results.jsonl"
        if not (checkpoint_path.exists() and manifest_path.exists() and results_path.exists()):
            continue
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if _looks_like_complete_anatomy_run(checkpoint, manifest):
            candidates.append(child)
    if not candidates:
        raise Stage11AnatomyInterimDataNotFoundError(
            f"No structurally-complete anatomy run found under {results_root} (require 3 regions, "
            f"d_map_n={EXPECTED_D_MAP_N}, n_directions_per_cell={EXPECTED_N_DIRECTIONS}, "
            f"run_complete=true, {EXPECTED_UNIQUE_PERTURBATIONS} unique perturbations, "
            f"{EXPECTED_ROWS} rows). Smoke runs are structurally excluded, never by directory name."
        )
    if len(candidates) > 1:
        raise Stage11AnatomyInterimAmbiguousRunError(f"Multiple structurally-complete anatomy runs found under {results_root}: {[str(c) for c in candidates]}")
    return candidates[0]


def load_complete_anatomy_records(results_root: Path) -> Tuple[List[ExperimentResultRecord], Dict[str, Any], Dict[str, Any]]:
    run_dir = discover_complete_anatomy_run(results_root)
    checkpoint = json.loads((run_dir / "checkpoint_manifest.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    records = load_records(run_dir / "results.jsonl")
    return records, checkpoint, manifest


def load_baseline_scores(results_root: Path) -> Dict[str, Any]:
    run_dir = discover_complete_anatomy_run(results_root)
    path = run_dir / "baseline_scores.json"
    return json.loads(path.read_text()) if path.exists() else {}


# =================================================================================================
# Section 2: cross-scale integrity gate
# =================================================================================================


class Stage11AnatomyInterimIntegrityError(RuntimeError):
    pass


def _per_scale_integrity(scale_label: str, records: Sequence[ExperimentResultRecord], checkpoint: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    checks["all_rows_correct_scale"] = all(r.model_scale == scale_label for r in records)
    checks["three_regions"] = {r.anatomy_region for r in records} == set(REGIONS)
    checks["three_frozen_radii"] = {r.radius for r in records} == set(RADII)
    checks["six_frozen_capabilities"] = {r.capability for r in records} == set(CAPABILITIES)
    checks["expected_total_rows"] = len(records) == EXPECTED_ROWS

    by_pid: Dict[str, List[ExperimentResultRecord]] = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, []).append(r)
    checks["expected_576_unique_perturbations"] = len(by_pid) == EXPECTED_UNIQUE_PERTURBATIONS
    checks["exactly_6_rows_per_perturbation"] = all(len(rows) == len(CAPABILITIES) for rows in by_pid.values())
    checks["same_candidate_evaluated_on_all_6_capabilities"] = all({row.capability for row in rows} == set(CAPABILITIES) for rows in by_pid.values())
    checks["no_duplicate_capability_rows_within_a_perturbation"] = all(len({row.capability for row in rows}) == len(rows) for rows in by_pid.values())

    by_region_radius: Dict[Tuple[str, float], set] = {}
    for pid, rows in by_pid.items():
        by_region_radius.setdefault((rows[0].anatomy_region, rows[0].radius), set()).add(pid)
    expected_cells = {(region, radius) for region in REGIONS for radius in RADII}
    checks["no_missing_cells"] = set(by_region_radius.keys()) == expected_cells
    checks["exactly_64_perturbations_per_region_x_radius"] = all(len(v) == EXPECTED_N_DIRECTIONS for v in by_region_radius.values())

    checks["d_map_n_50"] = checkpoint.get("d_map_n") == EXPECTED_D_MAP_N
    checks["run_complete_flag_true"] = manifest.get("run_complete") is True
    checks["actual_counts_match_expected"] = (
        manifest.get("actual_unique_perturbations") == EXPECTED_UNIQUE_PERTURBATIONS and manifest.get("actual_result_rows") == EXPECTED_ROWS
    )
    checks["perturbation_mode_anatomical_relative_l2"] = checkpoint.get("perturbation_mode") == "anatomical_relative_l2"
    checks["radius_realization_method_correct"] = checkpoint.get("radius_realization_method") == "fixed_direction_bf16_quantization_aware_v3"
    checks["restoration_mode_fixed_base"] = checkpoint.get("restoration_mode") == "fixed_base"
    checks["cache_policy_correct"] = checkpoint.get("multimodal_cache_policy") == "full_encoder_reset_vllm011_verified_v2"
    checks["enable_prefix_caching_false"] = checkpoint.get("enable_prefix_caching") is False

    model_revisions = {r.model_revision for r in records}
    checks["model_revision_consistent"] = len(model_revisions) == 1
    checks["model_revision"] = next(iter(model_revisions), None)

    region_mask_hashes: Dict[str, set] = {}
    for r in records:
        region_mask_hashes.setdefault(r.anatomy_region, set()).add(r.parameter_mask_hash)
    checks["one_mask_hash_per_region"] = all(len(v) == 1 for v in region_mask_hashes.values())
    checks["region_mask_hashes"] = {region: next(iter(v)) for region, v in region_mask_hashes.items()}
    checks["checkpoint_region_mask_hashes_match_records"] = all(
        checkpoint.get("region_mask_hashes", {}).get(region) == checks["region_mask_hashes"].get(region) for region in REGIONS
    )

    non_meta_keys = [k for k in checks if k not in ("model_revision", "region_mask_hashes")]
    checks["all_checks_pass"] = all(bool(checks[k]) for k in non_meta_keys if isinstance(checks[k], bool))
    return checks


def run_cross_scale_anatomy_integrity_gate(
    records_by_scale: Dict[str, List[ExperimentResultRecord]], checkpoint_by_scale: Dict[str, Dict[str, Any]], manifest_by_scale: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    per_scale = {scale: _per_scale_integrity(scale, records_by_scale[scale], checkpoint_by_scale[scale], manifest_by_scale[scale]) for scale in SCALES}
    ck3, ck7 = checkpoint_by_scale["3B"], checkpoint_by_scale["7B"]

    cross: Dict[str, Any] = {}
    cross["same_capability_set"] = set(ck3.get("capabilities", [])) == set(ck7.get("capabilities", [])) == set(CAPABILITIES)
    cross["same_capability_ordering"] = list(ck3.get("capabilities", [])) == list(ck7.get("capabilities", []))
    cross["same_semantic_region_partition"] = set(ck3.get("regions", [])) == set(ck7.get("regions", [])) == set(REGIONS)
    cross["same_d_map_subset_hashes"] = ck3.get("subset_hashes") == ck7.get("subset_hashes")
    cross["same_radii"] = list(ck3.get("radii", [])) == list(ck7.get("radii", [])) == list(RADII)
    cross["same_candidate_budget"] = ck3.get("n_directions_per_cell") == ck7.get("n_directions_per_cell") == EXPECTED_N_DIRECTIONS
    cross["same_d_map_n"] = ck3.get("d_map_n") == ck7.get("d_map_n") == EXPECTED_D_MAP_N
    cross["different_model_revision"] = bool(ck3.get("model_revision") and ck7.get("model_revision") and ck3.get("model_revision") != ck7.get("model_revision"))
    cross["different_direction_seed_bank_hash"] = ck3.get("direction_seed_bank_hash") != ck7.get("direction_seed_bank_hash")
    # Region mask hashes are INFORMATIONAL, never a hard-fail condition: vision/connector are often
    # architecturally IDENTICAL in size across a model family's scales (e.g. a shared ViT encoder),
    # while language legitimately differs -- "require distinct... where architecture dimensions
    # differ" (task spec) means a hard requirement would be scientifically wrong here.
    region_mask_hashes_3b = per_scale["3B"].get("region_mask_hashes", {})
    region_mask_hashes_7b = per_scale["7B"].get("region_mask_hashes", {})
    cross["region_mask_hash_comparison"] = {
        region: {
            "hash_3B": region_mask_hashes_3b.get(region), "hash_7B": region_mask_hashes_7b.get(region),
            "identical": region_mask_hashes_3b.get(region) == region_mask_hashes_7b.get(region),
        }
        for region in REGIONS
    }
    cross["cross_scale_inference_mode"] = "independent_sample_never_paired"
    cross["all_ok"] = all(bool(v) for k, v in cross.items() if isinstance(v, bool))

    return {
        "per_scale": per_scale, "cross_scale": cross,
        "all_ok": bool(per_scale["3B"]["all_checks_pass"] and per_scale["7B"]["all_checks_pass"] and cross["all_ok"]),
    }


def ensure_cross_scale_anatomy_integrity(report: Dict[str, Any]) -> None:
    if not report.get("all_ok"):
        failed = {scale: {k: v for k, v in report["per_scale"][scale].items() if isinstance(v, bool) and not v} for scale in SCALES}
        failed["cross_scale"] = {k: v for k, v in report["cross_scale"].items() if isinstance(v, bool) and not v}
        raise Stage11AnatomyInterimIntegrityError(f"Stage-11 S2 cross-scale anatomy integrity gate FAILED -- refusing to analyze. Failed checks: {failed}")


# =================================================================================================
# Baseline / headroom (merged view across scales, reusing s8a.compute_baseline_table per scale)
# =================================================================================================


def compute_merged_baseline_table(records_by_scale: Dict[str, List[ExperimentResultRecord]], baseline_scores_by_scale: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_scale = {scale: s8a.compute_baseline_table(records_by_scale[scale], baseline_scores_by_scale[scale]) for scale in SCALES}
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        b3 = per_scale["3B"].get(cap, {}).get("baseline_score")
        b7 = per_scale["7B"].get(cap, {}).get("baseline_score")
        out[cap] = {
            "capability": cap, "baseline_3B": b3, "baseline_7B": b7,
            "headroom_3B": (1.0 - b3) if b3 is not None else None, "headroom_7B": (1.0 - b7) if b7 is not None else None,
            "absolute_baseline_difference_7B_minus_3B": (b7 - b3) if b3 is not None and b7 is not None else None,
        }
    return out


# =================================================================================================
# Section 4: primary 108-cell table (2 scales x 3 regions x 3 radii x 6 capabilities)
# =================================================================================================


def compute_anatomy_cell_statistics(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scale in SCALES:
        by_cell = s8a.group_by_capability_region_radius(records_by_scale[scale])
        for (cap, region, radius), rows in by_cell.items():
            deltas = [r.delta for r in rows]
            arr = np.asarray(deltas, dtype=float)
            n = int(arr.size)
            mean, std = thicket_metrics.mean_std(deltas)
            q = thicket_metrics.quantiles(deltas, qs=QUANTILE_LEVELS)
            p_gt0 = thicket_metrics.probability_of_improvement(deltas)
            density = thicket_metrics.solution_density(deltas, margins=(0.0, USEFUL_MARGIN, STRONG_MARGIN))
            pos_mass = thicket_metrics.positive_thicket_mass(deltas)
            neg_mass = float(np.mean(np.clip(-arr, 0.0, None)))

            n_gt0 = int(np.sum(arr > 0))
            n_ge_useful, n_ge_strong = int(np.sum(arr >= USEFUL_MARGIN)), int(np.sum(arr >= STRONG_MARGIN))

            mean_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=np.mean, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            pos_mass_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=lambda d: float(np.mean(np.clip(d, 0.0, None))), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 1)
            neg_mass_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=lambda d: float(np.mean(np.clip(-d, 0.0, None))), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 2)
            q90_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=lambda d: float(np.quantile(d, 0.9)), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 3)
            q95_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=lambda d: float(np.quantile(d, 0.95)), n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + 4)

            key = f"{scale}:{cap}:{region}:{radius}"
            out[key] = {
                "scale": scale, "capability": cap, "region": region, "radius": radius, "radius_label": RADIUS_LABELS[radius], "n": n,
                "mean_delta": mean, "mean_delta_95ci_bootstrap": list(mean_ci),
                "median_delta": q[0.5], "std_delta": std, "min_delta": float(arr.min()), "max_delta": float(arr.max()),
                "p_delta_gt_0": p_gt0, "p_delta_gt_0_95ci_wilson": list(wilson_confidence_interval(n_gt0, n)),
                "density_ge_0.0": density[0.0], "density_ge_0.0_95ci_wilson": list(wilson_confidence_interval(int(np.sum(arr >= 0.0)), n)),
                "density_ge_0.02": density[USEFUL_MARGIN], "density_ge_0.02_95ci_wilson": list(wilson_confidence_interval(n_ge_useful, n)),
                "density_ge_0.05": density[STRONG_MARGIN], "density_ge_0.05_95ci_wilson": list(wilson_confidence_interval(n_ge_strong, n)),
                "positive_thicket_mass": pos_mass, "positive_thicket_mass_95ci_bootstrap": list(pos_mass_ci),
                "negative_mass": neg_mass, "negative_mass_95ci_bootstrap": list(neg_mass_ci),
                "q25": q[0.25], "q50": q[0.5], "q75": q[0.75],
                "q90": q[0.9], "q90_95ci_bootstrap": list(q90_ci),
                "q95": q[0.95], "q95_95ci_bootstrap": list(q95_ci),
            }
    return out


def write_anatomy_cell_statistics_csv(cell_stats: Dict[str, Any], path: Path) -> None:
    header = [
        "scale", "capability", "region", "radius", "radius_label", "n", "mean_delta", "median_delta", "std_delta",
        "min_delta", "max_delta", "p_delta_gt_0", "density_ge_0.0", "density_ge_0.02", "density_ge_0.05",
        "positive_thicket_mass", "negative_mass", "q25", "q50", "q75", "q90", "q95",
    ]
    rows = [[row[h] for h in header] for row in cell_stats.values()]
    s8a._write_csv(path, header, rows)


# =================================================================================================
# Section 5: anatomy-resolved solution-density curves -- reuse s8a's fixed common margin grid
# (SOLUTION_DENSITY_MARGIN_GRID is a module-level constant, so calling s8a's function once per
# scale is trivially "one common grid across both scales" -- never data-dependent, never favors
# either scale).
# =================================================================================================


def compute_anatomy_solution_density_curves(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    return {"margin_grid": list(s8a.SOLUTION_DENSITY_MARGIN_GRID), "by_scale": {scale: s8a.compute_solution_density_curves(records_by_scale[scale]) for scale in SCALES}}


def ensure_anatomy_curves_monotonic(curves: Dict[str, Any]) -> None:
    for scale, cap_map in curves["by_scale"].items():
        for cap, region_map in cap_map.items():
            for region, radius_map in region_map.items():
                for radius_key, row in radius_map.items():
                    d = row["delta_ge_m"]
                    if any(d[i] < d[i + 1] - 1e-12 for i in range(len(d) - 1)):
                        raise ValueError(f"Solution-density curve non-monotonic at scale={scale} cap={cap} region={region} radius={radius_key}: {d}")


def write_anatomy_solution_density_curves_csv(curves: Dict[str, Any], path: Path) -> None:
    header = ["scale", "capability", "region", "radius", "margin", "density"]
    rows = []
    for scale, cap_map in curves["by_scale"].items():
        for cap, region_map in cap_map.items():
            for region, radius_map in region_map.items():
                for row in radius_map.values():
                    for m, d in zip(row["margin_grid"], row["delta_ge_m"]):
                        rows.append([scale, cap, region, row["radius"], m, d])
    s8a._write_csv(path, header, rows)


# =================================================================================================
# Section 6: cross-scale anatomical differences (54 = 6 capabilities x 3 regions x 3 radii)
# =================================================================================================


def _density_ge_axis1(m: float):
    return lambda mat: (mat >= m).mean(axis=1)


def compute_cross_scale_anatomy_density_tests(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in HEADLINE_MARGINS:
        cells: Dict[str, Any] = {}
        stat_fn = _density_ge_axis1(m)
        for cap in CAPABILITIES:
            for region in REGIONS:
                for radius in RADII:
                    a = np.asarray([r.delta for r in records_by_scale["3B"] if r.capability == cap and r.anatomy_region == region and r.radius == radius], dtype=float)
                    b = np.asarray([r.delta for r in records_by_scale["7B"] if r.capability == cap and r.anatomy_region == region and r.radius == radius], dtype=float)
                    density_a, density_b = float(np.mean(a >= m)), float(np.mean(b >= m))
                    diff = density_b - density_a
                    seed_key = (cap, region, radius, m)
                    seed, perm_seed = BOOTSTRAP_SEED + hash(seed_key) % 10_000, PERMUTATION_SEED + hash(seed_key) % 10_000
                    ci = s8a._bootstrap_diff_ci(b, a, stat_fn, seed)
                    p_value = s8a._permutation_p_value(b, a, stat_fn, diff, perm_seed)
                    key = f"{cap}:{region}:{radius}"
                    cells[key] = {
                        "capability": cap, "region": region, "radius": radius, "radius_label": RADIUS_LABELS[radius], "margin": m,
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


def compute_cross_scale_anatomy_point_differences(cell_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Plain (non-bootstrapped) 7B-3B point differences for mean_delta/positive_mass/negative_mass/
    Q90/Q95 -- the task spec requires the full bootstrap+permutation+BH treatment only for the two
    headline density margins (compute_cross_scale_anatomy_density_tests above); these other metrics
    are reported as point differences, each side's own per-scale bootstrap CI already available in
    cell_stats.
    """
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for region in REGIONS:
            for radius in RADII:
                row3, row7 = cell_stats[f"3B:{cap}:{region}:{radius}"], cell_stats[f"7B:{cap}:{region}:{radius}"]
                key = f"{cap}:{region}:{radius}"
                out[key] = {
                    "capability": cap, "region": region, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                    "mean_delta_diff_7B_minus_3B": row7["mean_delta"] - row3["mean_delta"],
                    "positive_mass_diff_7B_minus_3B": row7["positive_thicket_mass"] - row3["positive_thicket_mass"],
                    "negative_mass_diff_7B_minus_3B": row7["negative_mass"] - row3["negative_mass"],
                    "q90_diff_7B_minus_3B": row7["q90"] - row3["q90"], "q95_diff_7B_minus_3B": row7["q95"] - row3["q95"],
                }
    return out


# =================================================================================================
# Section 7: "Where do experts live?" -- core region-ranking analysis
# =================================================================================================


def rank_regions_by_capability_radius_scale(cell_stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scale in SCALES:
        for cap in CAPABILITIES:
            for radius in RADII:
                ranked = sorted(REGIONS, key=lambda region: (
                    cell_stats[f"{scale}:{cap}:{region}:{radius}"]["density_ge_0.02"],
                    cell_stats[f"{scale}:{cap}:{region}:{radius}"]["positive_thicket_mass"],
                    cell_stats[f"{scale}:{cap}:{region}:{radius}"]["mean_delta"],
                ), reverse=True)
                out[f"{scale}:{cap}:{radius}"] = {"scale": scale, "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius], "ranked_regions": ranked}
    return out


def classify_anatomical_preference_transitions(records_by_scale: Dict[str, List[ExperimentResultRecord]], cell_stats: Dict[str, Any], rankings: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for radius in RADII:
            per_scale_dominant: Dict[str, Any] = {}
            for scale in SCALES:
                ranked = rankings[f"{scale}:{cap}:{radius}"]["ranked_regions"]
                top1, top2 = ranked[0], ranked[1]
                a = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.anatomy_region == top1 and r.radius == radius], dtype=float)
                b = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.anatomy_region == top2 and r.radius == radius], dtype=float)
                seed = BOOTSTRAP_SEED + hash((cap, radius, scale, "region_gap")) % 10_000
                gap_ci = s8a._bootstrap_diff_ci(a, b, s8a._density_ge_002_axis1, seed)
                well_supported = gap_ci[0] > 0  # top1's density>=.02 significantly exceeds top2's
                per_scale_dominant[scale] = {
                    "dominant_region": top1 if well_supported else None, "top1_region": top1, "top2_region": top2,
                    "density_gap_top1_minus_top2_95ci_bootstrap": list(gap_ci), "well_supported": well_supported,
                }
            d3, d7 = per_scale_dominant["3B"]["dominant_region"], per_scale_dominant["7B"]["dominant_region"]
            if d3 is None or d7 is None:
                classification = "diffuse_no_clear_preference"
            elif d3 == d7:
                classification = "anatomical_preference_stable"
            else:
                classification = "anatomical_preference_reorganizes"
            out[f"{cap}:{radius}"] = {
                "capability": cap, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                "per_scale": per_scale_dominant, "dominant_region_3B": d3, "dominant_region_7B": d7, "classification": classification,
            }
    return out


# =================================================================================================
# Section 8: anatomical contrasts (reuse s8a wholesale) + difference-in-differences
# =================================================================================================


def compute_anatomical_contrasts_by_scale(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    return {scale: s8a.apply_benjamini_hochberg_correction(s8a.compute_anatomical_contrasts(records_by_scale[scale])) for scale in SCALES}


def _stat_mean_axis1(m: np.ndarray) -> np.ndarray:
    return m.mean(axis=1)


_DID_STATISTIC_AXIS1 = {"mean_delta": s8a._mean_axis1, "density_ge_0.02": s8a._density_ge_002_axis1, "positive_thicket_mass": s8a._positive_mass_axis1}


def _did_bootstrap_ci(a3: np.ndarray, b3: np.ndarray, a7: np.ndarray, b7: np.ndarray, statistic_fn_axis1, seed: int) -> Tuple[float, float]:
    """Difference-in-differences CI: [(stat(A_7B)-stat(B_7B)) - (stat(A_3B)-stat(B_3B))], each of
    the four groups resampled INDEPENDENTLY (never paired -- 3B/7B are independent samples, and
    region A/B are independent seed namespaces even within one scale).
    """
    rng = np.random.default_rng(seed)
    resampled_a3 = a3[rng.integers(0, a3.size, size=(N_BOOTSTRAP, a3.size))]
    resampled_b3 = b3[rng.integers(0, b3.size, size=(N_BOOTSTRAP, b3.size))]
    resampled_a7 = a7[rng.integers(0, a7.size, size=(N_BOOTSTRAP, a7.size))]
    resampled_b7 = b7[rng.integers(0, b7.size, size=(N_BOOTSTRAP, b7.size))]
    did = (statistic_fn_axis1(resampled_a7) - statistic_fn_axis1(resampled_b7)) - (statistic_fn_axis1(resampled_a3) - statistic_fn_axis1(resampled_b3))
    lo, hi = np.percentile(did, [2.5, 97.5])
    return float(lo), float(hi)


_REGION_PAIRS: Tuple[Tuple[str, str], ...] = (("vision", "language"), ("vision", "multimodal_connector_or_merger"), ("multimodal_connector_or_merger", "language"))


def compute_difference_in_differences(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for radius in RADII:
            for region_a, region_b in _REGION_PAIRS:
                deltas = {}
                for scale in SCALES:
                    deltas[(scale, region_a)] = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.anatomy_region == region_a and r.radius == radius], dtype=float)
                    deltas[(scale, region_b)] = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.anatomy_region == region_b and r.radius == radius], dtype=float)
                pair_key = f"{region_a}_vs_{region_b}"
                metrics_out: Dict[str, Any] = {}
                for metric_name, stat_fn in _DID_STATISTIC_AXIS1.items():
                    point_stat = {(scale, region): float(stat_fn(deltas[(scale, region)].reshape(1, -1))[0]) for scale in SCALES for region in (region_a, region_b)}
                    contrast_3b = point_stat[("3B", region_a)] - point_stat[("3B", region_b)]
                    contrast_7b = point_stat[("7B", region_a)] - point_stat[("7B", region_b)]
                    did_point = contrast_7b - contrast_3b
                    seed = BOOTSTRAP_SEED + hash((cap, radius, pair_key, metric_name)) % 10_000
                    ci = _did_bootstrap_ci(deltas[("3B", region_a)], deltas[("3B", region_b)], deltas[("7B", region_a)], deltas[("7B", region_b)], stat_fn, seed)
                    metrics_out[metric_name] = {
                        "contrast_3B": contrast_3b, "contrast_7B": contrast_7b, "difference_in_differences": did_point,
                        "difference_in_differences_95ci_bootstrap": list(ci), "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                    }
                out.setdefault(cap, {}).setdefault(str(radius), {})[pair_key] = {"capability": cap, "radius": radius, "region_a": region_a, "region_b": region_b, "metrics": metrics_out}
    return out


# =================================================================================================
# Section 9: anatomical scale-response map (trivial derivation from cell_stats)
# =================================================================================================


def compute_anatomical_scale_response_map(cell_stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for radius in RADII:
        density_matrix, mass_matrix, q90_matrix = {}, {}, {}
        for cap in CAPABILITIES:
            density_matrix[cap] = {region: cell_stats[f"7B:{cap}:{region}:{radius}"]["density_ge_0.02"] - cell_stats[f"3B:{cap}:{region}:{radius}"]["density_ge_0.02"] for region in REGIONS}
            mass_matrix[cap] = {region: cell_stats[f"7B:{cap}:{region}:{radius}"]["positive_thicket_mass"] - cell_stats[f"3B:{cap}:{region}:{radius}"]["positive_thicket_mass"] for region in REGIONS}
            q90_matrix[cap] = {region: cell_stats[f"7B:{cap}:{region}:{radius}"]["q90"] - cell_stats[f"3B:{cap}:{region}:{radius}"]["q90"] for region in REGIONS}
        out[str(radius)] = {"radius": radius, "radius_label": RADIUS_LABELS[radius], "density_ge_0.02_diff_matrix": density_matrix, "positive_mass_diff_matrix": mass_matrix, "q90_diff_matrix": q90_matrix}
    return out


# =================================================================================================
# Section 11: scale x radius x anatomy joint reorganization classification
# =================================================================================================


RADIUS_SCALE_ANATOMY_LABELS: Tuple[str, ...] = (
    "stable_anatomy_stable_radius", "stable_anatomy_radius_reorganizes", "anatomy_reorganizes_radius_stable", "anatomy_and_radius_reorganize", "diffuse_or_insufficient_resolution",
)


def compute_radius_scale_anatomy_classification(records_by_scale: Dict[str, List[ExperimentResultRecord]], cell_stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        per_scale_best: Dict[str, Any] = {}
        for scale in SCALES:
            cells = [(region, radius, cell_stats[f"{scale}:{cap}:{region}:{radius}"]["density_ge_0.02"]) for region in REGIONS for radius in RADII]
            cells_sorted = sorted(cells, key=lambda t: t[2], reverse=True)
            (top_region, top_radius, top_density), (_, _, second_density) = cells_sorted[0], cells_sorted[1]
            top_deltas = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.anatomy_region == top_region and r.radius == top_radius], dtype=float)
            second_region, second_radius = cells_sorted[1][0], cells_sorted[1][1]
            second_deltas = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.anatomy_region == second_region and r.radius == second_radius], dtype=float)
            seed = BOOTSTRAP_SEED + hash((cap, scale, "radius_anatomy_gap")) % 10_000
            gap_ci = s8a._bootstrap_diff_ci(top_deltas, second_deltas, s8a._density_ge_002_axis1, seed)
            well_supported = gap_ci[0] > 0
            per_scale_best[scale] = {"region": top_region if well_supported else None, "radius": top_radius if well_supported else None, "well_supported": well_supported}
        r3, r7 = per_scale_best["3B"], per_scale_best["7B"]
        if not r3["well_supported"] or not r7["well_supported"]:
            classification = "diffuse_or_insufficient_resolution"
        else:
            same_region, same_radius = r3["region"] == r7["region"], r3["radius"] == r7["radius"]
            if same_region and same_radius:
                classification = "stable_anatomy_stable_radius"
            elif same_region and not same_radius:
                classification = "stable_anatomy_radius_reorganizes"
            elif not same_region and same_radius:
                classification = "anatomy_reorganizes_radius_stable"
            else:
                classification = "anatomy_and_radius_reorganize"
        out[cap] = {"capability": cap, "per_scale_best": per_scale_best, "classification": classification}
    return out


# =================================================================================================
# Section 12: region-level macro trend (row-preserving bootstrap, per scale x region x radius)
# =================================================================================================


def _matrix_for_region_radius(records: Sequence[ExperimentResultRecord], region: str, radius: float) -> np.ndarray:
    subset = [r for r in records if r.anatomy_region == region and r.radius == radius]
    _, _, matrix = build_delta_matrix(subset)
    return matrix


def _macro_stat_bootstrap_distribution(matrix: np.ndarray, statistic_fn, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    resampled = matrix[idx]  # (N_BOOTSTRAP, n, n_capabilities)
    return statistic_fn(resampled)


def compute_region_macro_scale_trend(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    density_stat = lambda m: (m >= USEFUL_MARGIN).mean(axis=(1, 2))  # noqa: E731
    density_strong_stat = lambda m: (m >= STRONG_MARGIN).mean(axis=(1, 2))  # noqa: E731
    mass_stat = lambda m: np.clip(m, 0.0, None).mean(axis=(1, 2))  # noqa: E731

    by_scale_region_radius: Dict[str, Any] = {}
    matrices: Dict[Tuple[str, str, float], np.ndarray] = {}
    for scale in SCALES:
        for region in REGIONS:
            for radius in RADII:
                matrix = _matrix_for_region_radius(records_by_scale[scale], region, radius)
                matrices[(scale, region, radius)] = matrix
                point = {"macro_density_ge_0.02": float((matrix >= USEFUL_MARGIN).mean()), "macro_density_ge_0.05": float((matrix >= STRONG_MARGIN).mean()), "macro_positive_mass": float(np.mean(np.clip(matrix, 0.0, None)))}
                by_scale_region_radius.setdefault(scale, {}).setdefault(region, {})[str(radius)] = {"radius": radius, "radius_label": RADIUS_LABELS[radius], **point}

    difference: Dict[str, Any] = {}
    for region in REGIONS:
        for radius in RADII:
            seed3 = BOOTSTRAP_SEED + hash(("3B", region, radius, "macro")) % 10_000
            seed7 = BOOTSTRAP_SEED + hash(("7B", region, radius, "macro")) % 10_000
            dist3 = _macro_stat_bootstrap_distribution(matrices[("3B", region, radius)], density_stat, seed3)
            dist7 = _macro_stat_bootstrap_distribution(matrices[("7B", region, radius)], density_stat, seed7)
            diff_dist = dist7 - dist3
            lo, hi = np.percentile(diff_dist, [2.5, 97.5])
            point3 = by_scale_region_radius["3B"][region][str(radius)]["macro_density_ge_0.02"]
            point7 = by_scale_region_radius["7B"][region][str(radius)]["macro_density_ge_0.02"]
            difference.setdefault(region, {})[str(radius)] = {
                "radius": radius, "radius_label": RADIUS_LABELS[radius], "macro_density_ge_0.02_3B": point3, "macro_density_ge_0.02_7B": point7,
                "difference_7B_minus_3B": point7 - point3, "difference_95ci_bootstrap": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
    return {"by_scale_region_radius": by_scale_region_radius, "difference_7B_minus_3B": difference}


# =================================================================================================
# Section 13: specialization by anatomy and scale (reuse s8a.compute_cross_capability_specialization)
# =================================================================================================


def _discordance_bootstrap_distribution(matrix: np.ndarray, seed: int, n_bootstrap: int = DISCORDANCE_N_BOOTSTRAP) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    out = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        out[i] = thicket_diversity.spectral_discordance(matrix[idx])
    return out


def compute_specialization_by_anatomy_scale(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    spec_by_scale = {scale: s8a.compute_cross_capability_specialization(records_by_scale[scale]) for scale in SCALES}
    out: Dict[str, Any] = {}
    for region in REGIONS:
        for radius in RADII:
            d3 = spec_by_scale["3B"][region][str(radius)]
            d7 = spec_by_scale["7B"][region][str(radius)]
            m3, m7 = _matrix_for_region_radius(records_by_scale["3B"], region, radius), _matrix_for_region_radius(records_by_scale["7B"], region, radius)
            seed3, seed7 = BOOTSTRAP_SEED + hash(("3B", region, radius, "discordance")) % 10_000, BOOTSTRAP_SEED + hash(("7B", region, radius, "discordance")) % 10_000
            dist3, dist7 = _discordance_bootstrap_distribution(m3, seed3), _discordance_bootstrap_distribution(m7, seed7)
            diff_dist = dist7 - dist3
            lo, hi = np.percentile(diff_dist, [2.5, 97.5])
            trend = "increases_3B_to_7B" if lo > 0 else ("decreases_3B_to_7B" if hi < 0 else "no_clear_change")
            out.setdefault(region, {})[str(radius)] = {
                "region": region, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                "spectral_discordance_3B": d3["spectral_discordance"], "spectral_discordance_7B": d7["spectral_discordance"],
                "difference_7B_minus_3B": d7["spectral_discordance"] - d3["spectral_discordance"], "difference_95ci_bootstrap": [float(lo), float(hi)],
                "improving_count_histogram_3B": d3["improving_count_histogram"], "improving_count_histogram_7B": d7["improving_count_histogram"],
                "fraction_tradeoff_candidates_3B": d3["fraction_tradeoff_candidates"], "fraction_tradeoff_candidates_7B": d7["fraction_tradeoff_candidates"],
                "trend": trend,
            }
    return out


# =================================================================================================
# Section 14: anatomical density vs strength classification (54 cells)
# =================================================================================================


MORE_VS_STRONGER_LABELS: Tuple[str, ...] = ("more_and_stronger", "more_not_stronger", "stronger_not_more", "neither_clear", "decreases")
DECISION_LOGIC_NOTE = (
    "more := density(m=0.02) bootstrap-CI-supported significant increase 7B>3B; "
    "stronger := positive_thicket_mass OR Q90 bootstrap-CI-supported significant increase 7B>3B; "
    "less_dense / weaker := the mirrored significant-decrease conditions. "
    "more_and_stronger if both hold; more_not_stronger / stronger_not_more if exactly one holds; "
    "decreases if less_dense or weaker holds without a compensating significant increase; neither_clear otherwise."
)


def compute_anatomy_strength_contrasts(records_by_scale: Dict[str, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for region in REGIONS:
            for radius in RADII:
                a = np.asarray([r.delta for r in records_by_scale["3B"] if r.capability == cap and r.anatomy_region == region and r.radius == radius], dtype=float)
                b = np.asarray([r.delta for r in records_by_scale["7B"] if r.capability == cap and r.anatomy_region == region and r.radius == radius], dtype=float)
                mass_diff = float(np.mean(np.clip(b, 0, None)) - np.mean(np.clip(a, 0, None)))
                q90_diff = float(np.quantile(b, 0.9) - np.quantile(a, 0.9))
                seed, perm_seed = BOOTSTRAP_SEED + hash((cap, region, radius, "strength")) % 10_000, PERMUTATION_SEED + hash((cap, region, radius, "strength")) % 10_000
                mass_ci = s8a._bootstrap_diff_ci(b, a, s8a._positive_mass_axis1, seed)
                mass_p = s8a._permutation_p_value(b, a, s8a._positive_mass_axis1, mass_diff, perm_seed)
                q90_axis1 = lambda m: np.quantile(m, 0.9, axis=1)  # noqa: E731
                q90_ci = s8a._bootstrap_diff_ci(b, a, q90_axis1, seed + 1)
                q90_p = s8a._permutation_p_value(b, a, q90_axis1, q90_diff, perm_seed + 1)
                key = f"{cap}:{region}:{radius}"
                cells[key] = {
                    "capability": cap, "region": region, "radius": radius, "radius_label": RADIUS_LABELS[radius],
                    "positive_mass_diff_7B_minus_3B": mass_diff, "positive_mass_diff_95ci_bootstrap": list(mass_ci), "positive_mass_diff_permutation_p": mass_p,
                    "q90_diff_7B_minus_3B": q90_diff, "q90_diff_95ci_bootstrap": list(q90_ci), "q90_diff_permutation_p": q90_p,
                }
    mass_p, q90_p = [c["positive_mass_diff_permutation_p"] for c in cells.values()], [c["q90_diff_permutation_p"] for c in cells.values()]
    mass_q, q90_q = s8a.benjamini_hochberg(mass_p), s8a.benjamini_hochberg(q90_p)
    for cell, mq, qq in zip(cells.values(), mass_q, q90_q):
        cell["positive_mass_diff_bh_q"], cell["q90_diff_bh_q"] = mq, qq
        mass_sig, q90_sig = mq < 0.05, qq < 0.05
        cell["mass_verdict"] = "significant_increase" if (mass_sig and cell["positive_mass_diff_7B_minus_3B"] > 0) else ("significant_decrease" if (mass_sig and cell["positive_mass_diff_7B_minus_3B"] < 0) else "non_significant")
        cell["q90_verdict"] = "significant_increase" if (q90_sig and cell["q90_diff_7B_minus_3B"] > 0) else ("significant_decrease" if (q90_sig and cell["q90_diff_7B_minus_3B"] < 0) else "non_significant")
        signals = {cell["mass_verdict"], cell["q90_verdict"]}
        if "significant_increase" in signals and "significant_decrease" not in signals:
            cell["strength_verdict"] = "significant_increase"
        elif "significant_decrease" in signals and "significant_increase" not in signals:
            cell["strength_verdict"] = "significant_decrease"
        else:
            cell["strength_verdict"] = "non_significant"
    return cells


def classify_anatomy_density_vs_strength(density_tests: Dict[str, Any], strength_contrasts: Dict[str, Any]) -> Dict[str, Any]:
    density_m002 = density_tests[f"m={USEFUL_MARGIN}"]
    cells: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for region in REGIONS:
            for radius in RADII:
                key = f"{cap}:{region}:{radius}"
                density_verdict, strength_verdict = density_m002[key]["verdict"], strength_contrasts[key]["strength_verdict"]
                more, less_dense = density_verdict == "significant_increase", density_verdict == "significant_decrease"
                stronger, weaker = strength_verdict == "significant_increase", strength_verdict == "significant_decrease"
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
                cells[key] = {"capability": cap, "region": region, "radius": radius, "radius_label": RADIUS_LABELS[radius], "density_verdict": density_verdict, "strength_verdict": strength_verdict, "classification": label}
    return {"decision_logic": DECISION_LOGIC_NOTE, "cells": cells}


# =================================================================================================
# Section 15: headroom sensitivity (secondary only; raw Delta remains primary)
# =================================================================================================


def compute_anatomy_headroom_sensitivity(records_by_scale: Dict[str, List[ExperimentResultRecord]], baseline_table: Dict[str, Any], cell_stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        for region in REGIONS:
            for radius in RADII:
                row: Dict[str, Any] = {"capability": cap, "region": region, "radius": radius, "radius_label": RADIUS_LABELS[radius]}
                normalized_stats: Dict[str, Any] = {}
                for scale in SCALES:
                    headroom = baseline_table[cap][f"headroom_{scale}"]
                    deltas = np.asarray([r.delta for r in records_by_scale[scale] if r.capability == cap and r.anatomy_region == region and r.radius == radius], dtype=float)
                    if headroom is None or headroom <= 0.0:
                        normalized_stats[scale] = {"applicable": False, "reason": "no_remaining_headroom"}
                        continue
                    normalized = deltas / headroom
                    normalized_stats[scale] = {"applicable": True, "normalized_positive_mass": float(np.mean(np.clip(normalized, 0, None))), "normalized_q90": float(np.quantile(normalized, 0.9)), "normalized_q95": float(np.quantile(normalized, 0.95))}
                row["normalized_by_scale"] = normalized_stats
                raw_mass_diff = cell_stats[f"7B:{cap}:{region}:{radius}"]["positive_thicket_mass"] - cell_stats[f"3B:{cap}:{region}:{radius}"]["positive_thicket_mass"]
                raw_direction = "increase" if raw_mass_diff > 1e-9 else ("decrease" if raw_mass_diff < -1e-9 else "flat")
                row["raw_positive_mass_diff_7B_minus_3B"], row["raw_conclusion_direction"] = raw_mass_diff, raw_direction
                if normalized_stats["3B"].get("applicable") and normalized_stats["7B"].get("applicable"):
                    norm_diff = normalized_stats["7B"]["normalized_positive_mass"] - normalized_stats["3B"]["normalized_positive_mass"]
                    norm_direction = "increase" if norm_diff > 1e-9 else ("decrease" if norm_diff < -1e-9 else "flat")
                    row["normalized_positive_mass_diff_7B_minus_3B"], row["normalized_conclusion_direction"] = norm_diff, norm_direction
                    if norm_direction == raw_direction:
                        row["headroom_sensitivity_verdict"] = "raw_conclusion_persists"
                    elif norm_direction == "flat" or raw_direction == "flat":
                        row["headroom_sensitivity_verdict"] = "raw_conclusion_weakens"
                    else:
                        row["headroom_sensitivity_verdict"] = "raw_conclusion_reverses"
                else:
                    row["normalized_conclusion_direction"], row["headroom_sensitivity_verdict"] = None, "not_applicable"
                out[f"{cap}:{region}:{radius}"] = row
    return out


# =================================================================================================
# Section 10: relate anatomy back to the whole-model S1 result
# =================================================================================================


S1_TO_ANATOMY_HEDGE_NOTE = (
    "A whole-model perturbation is NOT algebraically equivalent to summing the separate region "
    "perturbations -- these are 'consistent with' / 'the strongest anatomical correlate is' "
    "statements, never an exact causal decomposition of the S1 whole-model effect."
)


def load_s1_capability_summaries(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_whole_model_to_anatomy_interpretation(s1_summaries: Optional[Dict[str, Any]], density_tests: Dict[str, Any]) -> Dict[str, Any]:
    density_m002 = density_tests[f"m={USEFUL_MARGIN}"]
    out: Dict[str, Any] = {"note": S1_TO_ANATOMY_HEDGE_NOTE, "s1_summary_available": s1_summaries is not None, "by_capability": {}}
    for cap in CAPABILITIES:
        s1_classification = s1_summaries.get(cap, {}).get("classification") if s1_summaries else None
        per_region_signal = {}
        for region in REGIONS:
            sig_cells = [density_m002[f"{cap}:{region}:{radius}"] for radius in RADII]
            n_increase = sum(1 for c in sig_cells if c["verdict"] == "significant_increase")
            n_decrease = sum(1 for c in sig_cells if c["verdict"] == "significant_decrease")
            per_region_signal[region] = {"n_significant_increase_of_3_radii": n_increase, "n_significant_decrease_of_3_radii": n_decrease}
        if s1_classification in ("thicket_expands_3B_to_7B", "thicket_strengthens_3B_to_7B"):
            best_region = max(REGIONS, key=lambda r: per_region_signal[r]["n_significant_increase_of_3_radii"])
            best_count = per_region_signal[best_region]["n_significant_increase_of_3_radii"]
            interpretation = (f"consistent with the {best_region} region driving the S1 whole-model expansion" if best_count > 0 else "no region shows a statistically significant expansion at m=0.02 -- the S1 whole-model effect does not have a clear anatomical correlate at this margin")
        elif s1_classification in ("thicket_contracts_3B_to_7B",):
            best_region = max(REGIONS, key=lambda r: per_region_signal[r]["n_significant_decrease_of_3_radii"])
            best_count = per_region_signal[best_region]["n_significant_decrease_of_3_radii"]
            interpretation = (f"consistent with the {best_region} region driving the S1 whole-model contraction" if best_count > 0 else "no region shows a statistically significant contraction at m=0.02 -- the S1 whole-model effect does not have a clear anatomical correlate at this margin")
        else:
            interpretation = f"S1 whole-model classification is {s1_classification!r} (little_change/mixed) -- no single-region anatomical correlate is asserted"
        out["by_capability"][cap] = {"s1_whole_model_classification": s1_classification, "per_region_significant_density_change_count": per_region_signal, "interpretation": interpretation}
    return out


# =================================================================================================
# Section 16: claim gates A1-A6
# =================================================================================================


CLAIM_VERDICTS: Tuple[str, ...] = ("strongly_supported_3B_to_7B", "supported_3B_to_7B", "mixed", "unsupported")


def evaluate_anatomy_interim_claim_gate(
    cell_stats: Dict[str, Any], density_tests: Dict[str, Any], preference_transitions: Dict[str, Any], density_vs_strength: Dict[str, Any],
    radius_scale_anatomy: Dict[str, Any], specialization: Dict[str, Any],
) -> Dict[str, Any]:
    n_cells_54 = len(CAPABILITIES) * len(REGIONS) * len(RADII)

    # A1: coarse anatomy structures expert density at BOTH scales (nonzero density somewhere, both scales)
    n_scales_with_signal = sum(1 for scale in SCALES if sum(1 for cap in CAPABILITIES for region in REGIONS for radius in RADII if cell_stats[f"{scale}:{cap}:{region}:{radius}"]["density_ge_0.02"] > 0.0) >= n_cells_54 * 0.3)
    a1 = "strongly_supported_3B_to_7B" if n_scales_with_signal == 2 else ("supported_3B_to_7B" if n_scales_with_signal == 1 else "unsupported")

    # A2: anatomical distribution of expertise changes 3B->7B (preference transitions reorganize)
    n_reorganize = sum(1 for row in preference_transitions.values() if row["classification"] == "anatomical_preference_reorganizes")
    n_diffuse = sum(1 for row in preference_transitions.values() if row["classification"] == "diffuse_no_clear_preference")
    n_total_pref = len(preference_transitions)
    a2 = "strongly_supported_3B_to_7B" if n_reorganize >= n_total_pref * 0.5 else ("supported_3B_to_7B" if n_reorganize >= 1 else ("mixed" if n_diffuse >= n_total_pref * 0.5 else "unsupported"))

    # A3: scale effects are capability-dependent (heterogeneous classification across capabilities)
    density_m002 = density_tests[f"m={USEFUL_MARGIN}"]
    per_cap_direction = {}
    for cap in CAPABILITIES:
        n_inc = sum(1 for region in REGIONS for radius in RADII if density_m002[f"{cap}:{region}:{radius}"]["verdict"] == "significant_increase")
        n_dec = sum(1 for region in REGIONS for radius in RADII if density_m002[f"{cap}:{region}:{radius}"]["verdict"] == "significant_decrease")
        per_cap_direction[cap] = "increase" if n_inc > n_dec else ("decrease" if n_dec > n_inc else "flat")
    n_distinct_directions = len(set(per_cap_direction.values()))
    a3 = "strongly_supported_3B_to_7B" if n_distinct_directions >= 3 else ("supported_3B_to_7B" if n_distinct_directions == 2 else "unsupported")

    # A4: scale effects are anatomically non-uniform (density_vs_strength classification varies by region within a capability)
    n_caps_with_region_variation = 0
    for cap in CAPABILITIES:
        labels = {density_vs_strength["cells"][f"{cap}:{region}:{radius}"]["classification"] for region in REGIONS for radius in RADII}
        if len(labels) >= 2:
            n_caps_with_region_variation += 1
    a4 = "strongly_supported_3B_to_7B" if n_caps_with_region_variation >= len(CAPABILITIES) * 0.5 else ("supported_3B_to_7B" if n_caps_with_region_variation >= 1 else "unsupported")

    # A5: radius and scale jointly reorganize anatomical expertise
    n_reorg_radius_anatomy = sum(1 for row in radius_scale_anatomy.values() if row["classification"] in ("stable_anatomy_radius_reorganizes", "anatomy_reorganizes_radius_stable", "anatomy_and_radius_reorganize"))
    a5 = "strongly_supported_3B_to_7B" if n_reorg_radius_anatomy >= len(CAPABILITIES) * 0.5 else ("supported_3B_to_7B" if n_reorg_radius_anatomy >= 1 else "unsupported")

    # A6: specialization changes with scale DIFFERENTLY across regions
    trends_by_region = {region: {specialization[region][str(radius)]["trend"] for radius in RADII} for region in REGIONS}
    n_distinct_region_trend_sets = len({frozenset(v) for v in trends_by_region.values()})
    any_change = any(t != "no_clear_change" for trends in trends_by_region.values() for t in trends)
    a6 = "strongly_supported_3B_to_7B" if (n_distinct_region_trend_sets >= 2 and any_change) else ("supported_3B_to_7B" if any_change else "unsupported")

    return {
        "A1_coarse_anatomy_structures_density_both_scales": a1, "A2_anatomical_distribution_changes": a2, "A3_scale_effects_capability_dependent": a3,
        "A4_scale_effects_anatomically_non_uniform": a4, "A5_radius_and_scale_jointly_reorganize": a5, "A6_specialization_changes_differently_by_region": a6,
        "terminology_guard": TERMINOLOGY_GUARD, "note": "Two scale points only -- 'scaling law established' is NEVER a valid conclusion under any outcome above.",
    }


# =================================================================================================
# Section 17: critical questions A-J
# =================================================================================================


def answer_anatomy_critical_questions(
    rankings: Dict[str, Any], preference_transitions: Dict[str, Any], anatomical_scale_response_map: Dict[str, Any], density_tests: Dict[str, Any],
    radius_scale_anatomy: Dict[str, Any], specialization: Dict[str, Any], whole_model_interpretation: Dict[str, Any], headroom: Dict[str, Any],
) -> Dict[str, Any]:
    def _dominant_counts(scale: str) -> Dict[str, int]:
        counts = {region: 0 for region in REGIONS}
        for cap in CAPABILITIES:
            for radius in RADII:
                top = rankings[f"{scale}:{cap}:{radius}"]["ranked_regions"][0]
                counts[top] += 1
        return counts

    answer_a = {"dominant_region_counts_across_cap_x_radius_3B": _dominant_counts("3B")}
    answer_b = {"dominant_region_counts_across_cap_x_radius_7B": _dominant_counts("7B")}

    n_reorganize = sum(1 for row in preference_transitions.values() if row["classification"] == "anatomical_preference_reorganizes")
    answer_c = {"coarse_anatomical_address_changes": n_reorganize > 0, "n_capability_radius_pairs_reorganizing": n_reorganize, "n_total": len(preference_transitions)}

    region_totals = {region: 0.0 for region in REGIONS}
    for radius_row in anatomical_scale_response_map.values():
        for cap_row in radius_row["density_ge_0.02_diff_matrix"].values():
            for region, v in cap_row.items():
                region_totals[region] += v
    answer_d = {"region_with_largest_total_density_increase": max(region_totals, key=region_totals.get), "region_totals_summed_density_diff": region_totals}
    answer_e = {"region_with_largest_total_density_decrease": min(region_totals, key=region_totals.get), "region_totals_summed_density_diff": region_totals}

    density_m002 = density_tests[f"m={USEFUL_MARGIN}"]
    per_cap_direction = {}
    for cap in CAPABILITIES:
        n_inc = sum(1 for region in REGIONS for radius in RADII if density_m002[f"{cap}:{region}:{radius}"]["verdict"] == "significant_increase")
        n_dec = sum(1 for region in REGIONS for radius in RADII if density_m002[f"{cap}:{region}:{radius}"]["verdict"] == "significant_decrease")
        per_cap_direction[cap] = "increase" if n_inc > n_dec else ("decrease" if n_dec > n_inc else "flat")
    answer_f = {"uniform_across_capabilities": len(set(per_cap_direction.values())) == 1, "direction_by_capability": per_cap_direction}

    n_radius_relevant = sum(1 for row in radius_scale_anatomy.values() if row["classification"] != "stable_anatomy_stable_radius")
    answer_g = {"radius_alters_anatomical_scale_response": n_radius_relevant > 0, "n_capabilities_where_radius_matters": n_radius_relevant, "n_total": len(radius_scale_anatomy)}

    any_specialization_change = any(specialization[region][str(radius)]["trend"] != "no_clear_change" for region in REGIONS for radius in RADII)
    answer_h = {"specialization_changes_anatomically_with_scale": any_specialization_change}

    n_localized = sum(1 for row in whole_model_interpretation["by_capability"].values() if row["interpretation"].startswith("consistent with"))
    answer_i = {"n_capabilities_with_anatomical_correlate": n_localized, "n_total": len(whole_model_interpretation["by_capability"])}

    n_persists = sum(1 for row in headroom.values() if row["headroom_sensitivity_verdict"] == "raw_conclusion_persists")
    n_reverses = sum(1 for row in headroom.values() if row["headroom_sensitivity_verdict"] == "raw_conclusion_reverses")
    answer_j = {"headroom_explains_anatomical_differences": n_reverses > n_persists, "n_cells_conclusion_persists": n_persists, "n_cells_conclusion_reverses": n_reverses}

    return {
        "A_where_experts_live_3B": answer_a, "B_where_experts_live_7B": answer_b, "C_coarse_address_changes": answer_c,
        "D_largest_density_increase_region": answer_d, "E_largest_density_decrease_region": answer_e, "F_uniform_across_capabilities": answer_f,
        "G_radius_alters_response": answer_g, "H_specialization_changes_anatomically": answer_h, "I_s1_localized": answer_i, "J_headroom_explains": answer_j,
    }


# =================================================================================================
# Section 18: figure data (non-publication-styled)
# =================================================================================================


def build_fig_a(cell_stats: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["capability", "region", "radius", "radius_label", "scale", "density_ge_0.02"]
    return header, [[row["capability"], row["region"], row["radius"], row["radius_label"], row["scale"], row["density_ge_0.02"]] for row in cell_stats.values()]


def build_fig_b(anatomical_scale_response_map: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["radius", "radius_label", "capability", "region", "density_ge_0.02_diff_7B_minus_3B"]
    rows = []
    for radius_key, row in anatomical_scale_response_map.items():
        for cap, region_map in row["density_ge_0.02_diff_matrix"].items():
            for region, v in region_map.items():
                rows.append([row["radius"], row["radius_label"], cap, region, v])
    return header, rows


def build_fig_c(curves: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["scale", "capability", "region", "radius", "margin", "density"]
    rows = []
    for scale, cap_map in curves["by_scale"].items():
        for cap, region_map in cap_map.items():
            for region, radius_map in region_map.items():
                for row in radius_map.values():
                    for m, d in zip(row["margin_grid"], row["delta_ge_m"]):
                        rows.append([scale, cap, region, row["radius"], m, d])
    return header, rows


def build_fig_d(preference_transitions: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["capability", "radius", "radius_label", "dominant_region_3B", "dominant_region_7B", "classification"]
    return header, [[row["capability"], row["radius"], row["radius_label"], row["dominant_region_3B"], row["dominant_region_7B"], row["classification"]] for row in preference_transitions.values()]


def build_fig_e(region_macro: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["scale", "region", "radius", "radius_label", "macro_density_ge_0.02"]
    rows = []
    for scale, region_map in region_macro["by_scale_region_radius"].items():
        for region, radius_map in region_map.items():
            for row in radius_map.values():
                rows.append([scale, region, row["radius"], row["radius_label"], row["macro_density_ge_0.02"]])
    return header, rows


def build_fig_f(specialization: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["region", "radius", "radius_label", "scale", "spectral_discordance"]
    rows = []
    for region, radius_map in specialization.items():
        for row in radius_map.values():
            rows.append([region, row["radius"], row["radius_label"], "3B", row["spectral_discordance_3B"]])
            rows.append([region, row["radius"], row["radius_label"], "7B", row["spectral_discordance_7B"]])
    return header, rows


def build_fig_g(did: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["capability", "radius", "region_pair", "metric", "contrast_3B", "contrast_7B", "difference_in_differences", "ci_excludes_zero"]
    rows = []
    for cap, radius_map in did.items():
        for radius_key, pair_map in radius_map.items():
            for pair_key, cell in pair_map.items():
                for metric_name, m in cell["metrics"].items():
                    rows.append([cap, cell["radius"], pair_key, metric_name, m["contrast_3B"], m["contrast_7B"], m["difference_in_differences"], m["ci_excludes_zero"]])
    return header, rows


# =================================================================================================
# Markdown summary + main orchestration
# =================================================================================================


def build_markdown_summary(integrity: Dict[str, Any], claim_gate: Dict[str, Any], preference_transitions: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Stage 11 S2: interim 3B-vs-7B anatomy-resolved scale analysis")
    lines.append("")
    lines.append(f"Cross-scale integrity gate: **{'PASS' if integrity['all_ok'] else 'FAIL'}**.")
    lines.append("")
    lines.append(f"This is NOT a scaling-law claim (only 2 scale points). Terminology guard: {TERMINOLOGY_GUARD['allowed_terms']}.")
    lines.append("")
    lines.append("## Claim gate (A1-A6)")
    lines.append("")
    for k in ("A1_coarse_anatomy_structures_density_both_scales", "A2_anatomical_distribution_changes", "A3_scale_effects_capability_dependent", "A4_scale_effects_anatomically_non_uniform", "A5_radius_and_scale_jointly_reorganize", "A6_specialization_changes_differently_by_region"):
        lines.append(f"- {k}: **{claim_gate[k]}**")
    lines.append("")
    lines.append("## Anatomical preference transitions (dominant region, per capability x radius)")
    lines.append("")
    lines.append("| capability | radius | dominant 3B | dominant 7B | classification |")
    lines.append("|---|---|---|---|---|")
    for row in preference_transitions.values():
        lines.append(f"| {row['capability']} | {row['radius_label']} | {row['dominant_region_3B']} | {row['dominant_region_7B']} | {row['classification']} |")
    lines.append("")
    lines.append("DO NOT START 32B. DO NOT START 72B. DO NOT REDESIGN THE FROZEN SCALE EXPERIMENT. DO NOT START ATLAS-GUIDED SEARCH YET.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage8-dir", default=str(DEFAULT_STAGE8_DIR))
    parser.add_argument("--stage11-anatomy-dir", default=str(DEFAULT_STAGE11_ANATOMY_DIR))
    parser.add_argument("--s1-summary", default=str(DEFAULT_S1_SUMMARY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    results_roots = {"3B": Path(args.stage8_dir), "7B": Path(args.stage11_anatomy_dir)}
    output_dir = Path(args.output_dir)

    try:
        records_by_scale: Dict[str, List[ExperimentResultRecord]] = {}
        checkpoint_by_scale: Dict[str, Dict[str, Any]] = {}
        manifest_by_scale: Dict[str, Dict[str, Any]] = {}
        baseline_scores_by_scale: Dict[str, Dict[str, Any]] = {}
        for scale in SCALES:
            records, checkpoint, manifest = load_complete_anatomy_records(results_roots[scale])
            records_by_scale[scale], checkpoint_by_scale[scale], manifest_by_scale[scale] = records, checkpoint, manifest
            baseline_scores_by_scale[scale] = load_baseline_scores(results_roots[scale])
    except Stage11AnatomyInterimDataNotFoundError as exc:
        print(f"PREPARED, NOT RUN: {exc}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    integrity = run_cross_scale_anatomy_integrity_gate(records_by_scale, checkpoint_by_scale, manifest_by_scale)
    (output_dir / "integrity_report.json").write_text(json.dumps(s8a._sanitize(integrity), indent=2))
    ensure_cross_scale_anatomy_integrity(integrity)
    print("Cross-scale anatomy integrity gate PASSED.")

    baseline_table = compute_merged_baseline_table(records_by_scale, baseline_scores_by_scale)

    cell_stats = compute_anatomy_cell_statistics(records_by_scale)
    (output_dir / "anatomy_cell_statistics.json").write_text(json.dumps(s8a._sanitize(cell_stats), indent=2))
    write_anatomy_cell_statistics_csv(cell_stats, output_dir / "anatomy_cell_statistics.csv")

    curves = compute_anatomy_solution_density_curves(records_by_scale)
    ensure_anatomy_curves_monotonic(curves)
    (output_dir / "solution_density_curves.json").write_text(json.dumps(s8a._sanitize(curves), indent=2))
    write_anatomy_solution_density_curves_csv(curves, output_dir / "solution_density_curves.csv")

    density_tests = compute_cross_scale_anatomy_density_tests(records_by_scale)
    point_diffs = compute_cross_scale_anatomy_point_differences(cell_stats)
    (output_dir / "cross_scale_anatomy_differences.json").write_text(json.dumps(s8a._sanitize({"headline_margin_tests": density_tests, "point_differences": point_diffs}), indent=2))

    rankings = rank_regions_by_capability_radius_scale(cell_stats)
    preference_transitions = classify_anatomical_preference_transitions(records_by_scale, cell_stats, rankings)
    (output_dir / "anatomy_preference_transitions.json").write_text(json.dumps(s8a._sanitize({"rankings": rankings, "transitions": preference_transitions}), indent=2))

    contrasts_by_scale = compute_anatomical_contrasts_by_scale(records_by_scale)
    (output_dir / "anatomical_contrasts.json").write_text(json.dumps(s8a._sanitize(contrasts_by_scale), indent=2))

    did = compute_difference_in_differences(records_by_scale)
    (output_dir / "anatomical_difference_in_differences.json").write_text(json.dumps(s8a._sanitize(did), indent=2))

    scale_response_map = compute_anatomical_scale_response_map(cell_stats)
    radius_scale_anatomy = compute_radius_scale_anatomy_classification(records_by_scale, cell_stats)
    (output_dir / "radius_scale_anatomy.json").write_text(json.dumps(s8a._sanitize(radius_scale_anatomy), indent=2))

    region_macro = compute_region_macro_scale_trend(records_by_scale)
    (output_dir / "region_macro_scale_trend.json").write_text(json.dumps(s8a._sanitize(region_macro), indent=2))

    specialization = compute_specialization_by_anatomy_scale(records_by_scale)
    (output_dir / "specialization_by_anatomy_scale.json").write_text(json.dumps(s8a._sanitize(specialization), indent=2))

    strength_contrasts = compute_anatomy_strength_contrasts(records_by_scale)
    density_vs_strength = classify_anatomy_density_vs_strength(density_tests, strength_contrasts)
    (output_dir / "density_vs_strength_classification.json").write_text(json.dumps(s8a._sanitize(density_vs_strength), indent=2))

    headroom = compute_anatomy_headroom_sensitivity(records_by_scale, baseline_table, cell_stats)
    (output_dir / "headroom_sensitivity.json").write_text(json.dumps(s8a._sanitize(headroom), indent=2))

    s1_summaries = load_s1_capability_summaries(Path(args.s1_summary))
    whole_model_interpretation = build_whole_model_to_anatomy_interpretation(s1_summaries, density_tests)
    (output_dir / "whole_model_to_anatomy_interpretation.json").write_text(json.dumps(s8a._sanitize(whole_model_interpretation), indent=2))

    statistical_tests = {"headline_margin_density_tests": density_tests, "strength_contrasts": strength_contrasts, "difference_in_differences": did}
    (output_dir / "statistical_tests.json").write_text(json.dumps(s8a._sanitize(statistical_tests), indent=2))

    claim_gate = evaluate_anatomy_interim_claim_gate(cell_stats, density_tests, preference_transitions, density_vs_strength, radius_scale_anatomy, specialization)
    (output_dir / "interim_claim_gate.json").write_text(json.dumps(s8a._sanitize(claim_gate), indent=2))

    critical_answers = answer_anatomy_critical_questions(rankings, preference_transitions, scale_response_map, density_tests, radius_scale_anatomy, specialization, whole_model_interpretation, headroom)
    (output_dir / "critical_questions_a_to_j.json").write_text(json.dumps(s8a._sanitize(critical_answers), indent=2))

    figure_dir = output_dir / "figure_schemas"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, builder, fargs in (
        ("fig_a_atlas_3b_7b.csv", build_fig_a, (cell_stats,)),
        ("fig_b_scale_response_atlas.csv", build_fig_b, (scale_response_map,)),
        ("fig_c_solution_density_by_anatomy.csv", build_fig_c, (curves,)),
        ("fig_d_anatomy_preference_transitions.csv", build_fig_d, (preference_transitions,)),
        ("fig_e_macro_anatomy_scale.csv", build_fig_e, (region_macro,)),
        ("fig_f_specialization_anatomy_scale.csv", build_fig_f, (specialization,)),
        ("fig_g_anatomical_contrasts.csv", build_fig_g, (did,)),
    ):
        header, rows = builder(*fargs)
        s8a._write_csv(figure_dir / name, header, rows)

    summary_md = build_markdown_summary(integrity, claim_gate, preference_transitions)
    (output_dir / "stage11_interim_3b_7b_anatomy_summary.md").write_text(summary_md)

    print(f"Stage-11 S2 interim 3B-vs-7B anatomy analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
