"""Stage 7B rigorous scientific analysis (post-hoc, read-only). Reads ONLY the already
-completed full run's existing output -- results/stage7b_anatomical_calibration/
full_fixed_direction_bf16_quantization_aware_v3/{results.jsonl, checkpoint_manifest.json,
baseline_scores.json, run_manifest.json} -- and refuses to proceed if the run's own recorded
expectations don't match what's actually on disk (same discipline as
analysis/stage6_visual_thicket_analysis.py's main()).

Runs NO model, applies NO new perturbation, alters NO existing result. Reuses this project's own
metrics/diversity implementations (thicket.metrics, thicket.diversity, thicket_metrics.
wilson_confidence_interval, run_global_visual_thicket_pilot.build_delta_matrix,
run_stage7b_anatomical_calibration.Stage7bCheckpointManifest) rather than reimplementing any of
them.

CRITICAL DATA-INTEGRITY FINDING (see compute_data_integrity_report() and its docstring):
every vision-region and multimodal_connector_or_merger-region row in this run has delta EXACTLY
0.0, and within each (capability, region) group of 48 rows spanning all 6 radii and 8 seeds,
per_example_result_hash collapses to a SINGLE value -- generation output is completely invariant
to how large a vision/connector perturbation was applied. This is not a "near-base" scientific
finding: it is the identical symptom GATE2_CACHE_SAFETY_REVIEW.md documents for vLLM's
multimodal-encoder-OUTPUT cache, whose safety argument there explicitly depends on "the visual
encoder is never perturbed" -- a precondition Stage 7B violates (it perturbs both vision and
connector/merger regions) without ever calling the reset_vllm_encoder_cache_full() /
ensure_full_encoder_cache_reset_exposed() mechanism vlm_adapter.py already provides for exactly
this situation (confirmed by direct grep: zero references to either name anywhere in
run_stage7b_anatomical_calibration.py). Language-region rows show real, substantial,
radius-dependent deltas and many distinct per_example_result_hash values, consistent with no
caching confound on that side (GATE2_CACHE_SAFETY_REVIEW.md section 1/3). Vision/connector
results in this run must be treated as scientifically invalid until the cache-reset call is
wired into the evaluation RPC path and Stage 7B is re-run for those two regions.

Produces (results/stage7b_anatomical_calibration/full_fixed_direction_bf16_quantization_aware_v3/analysis/):
    calibration_table.json               -- per capability x region x radius descriptive stats + Wilson/bootstrap CIs
    matched_radius_anatomy_comparison.json -- per capability x radius: mean Delta / P(>0) / positive thicket mass BY REGION
    radius_regime_summary.json           -- common-radius regime classification, region x radius collapse scores,
                                             language-only supplementary classification, and the data-integrity report
    exploratory_anatomy_signal.json      -- capability x anatomy matrix at non-destructive radii (EXPLORATORY / CALIBRATION-SCALE)
    diversity_by_region_radius.json      -- per region x radius Spearman / Spectral Discordance / top-perturbation overlap / sign agreement
    quantization_audit.json              -- per-candidate + per region x radius BF16 realization audit
    stage7b_analysis.md                  -- the human-readable writeup, referencing the above numbers

Usage:
    python analysis/stage7b_anatomical_calibration_analysis.py [--results-dir <path>]
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
from neural_thickets_repro.thicket import diversity as thicket_diversity  # noqa: E402
from neural_thickets_repro.thicket import metrics as thicket_metrics  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402
from neural_thickets_repro.thicket_metrics import wilson_confidence_interval  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "stage7b_anatomical_calibration" / "full_fixed_direction_bf16_quantization_aware_v3"

# Fixed, deterministic -- matches analysis/stage6_visual_thicket_analysis.py's own
# BOOTSTRAP_SEED discipline.
BOOTSTRAP_SEED = 20260824
N_BOOTSTRAP = 10000

DENSITY_MARGINS: Tuple[float, ...] = (0.0, 0.02, 0.05)
SEVERE_DEGRADATION_MARGIN = 0.10  # Delta <= -0.10, for the collapse/destructive regime report only
TOP_OVERLAP_FRACTION = 0.25  # top-2-of-8 -- documented, small-N diagnostic (section 8), not a final estimate


class RunIntegrityError(RuntimeError):
    """The on-disk Stage 7B run does not match its own recorded expectations -- refuses to
    analyze an incomplete, mismatched, or misidentified run.
    """


def _sanitize(obj: Any) -> Any:
    """Recursively replaces NaN/Inf with None so every JSON file this script writes is valid
    JSON. A degenerate all-equal delta column (exactly what the vision/connector cache-artifact
    rows produce -- every delta EXACTLY 0.0) makes np.corrcoef divide by a zero column std,
    producing NaN -- an EXPECTED, not exceptional, case here, since it is the direct numeric
    signature of the data-integrity finding this script exists to surface.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_sanitize(obj), indent=2))


def _radius_key(radius: float) -> str:
    return f"{radius:.10g}"


# =============================================================================================
# Load + group
# =============================================================================================


def load_all(results_dir: Path) -> Tuple[List[ExperimentResultRecord], Stage7bCheckpointManifest, Dict[str, Any], Dict[str, Any]]:
    records = load_records(results_dir / "results.jsonl")
    checkpoint = Stage7bCheckpointManifest.from_dict(json.loads((results_dir / "checkpoint_manifest.json").read_text()))
    baseline = json.loads((results_dir / "baseline_scores.json").read_text())
    run_manifest_path = results_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text()) if run_manifest_path.exists() else {}
    return records, checkpoint, baseline, run_manifest


def group_by_region_radius(records: Sequence[ExperimentResultRecord]) -> Dict[Tuple[str, float], List[ExperimentResultRecord]]:
    out: Dict[Tuple[str, float], List[ExperimentResultRecord]] = {}
    for r in records:
        out.setdefault((r.anatomy_region, r.radius), []).append(r)
    return out


def group_by_capability_region_radius(records: Sequence[ExperimentResultRecord]) -> Dict[Tuple[str, str, float], List[float]]:
    out: Dict[Tuple[str, str, float], List[float]] = {}
    for r in records:
        out.setdefault((r.capability, r.anatomy_region, r.radius), []).append(r.delta)
    return out


def group_by_capability_radius(records: Sequence[ExperimentResultRecord]) -> Dict[Tuple[str, float], List[ExperimentResultRecord]]:
    out: Dict[Tuple[str, float], List[ExperimentResultRecord]] = {}
    for r in records:
        out.setdefault((r.capability, r.radius), []).append(r)
    return out


def group_by_radius(records: Sequence[ExperimentResultRecord]) -> Dict[float, List[ExperimentResultRecord]]:
    out: Dict[float, List[ExperimentResultRecord]] = {}
    for r in records:
        out.setdefault(r.radius, []).append(r)
    return out


def unique_candidates_by_region_radius(records: Sequence[ExperimentResultRecord]) -> Dict[Tuple[str, float], List[ExperimentResultRecord]]:
    """One record per (region, radius, perturbation_id) -- radius-realization/acceptance fields
    are identical across a perturbation's 3 capability rows (set once, at apply time), so
    counting/averaging them per-row would triple-count every candidate.
    """
    seen: Dict[Tuple[str, float, str], ExperimentResultRecord] = {}
    for r in records:
        seen[(r.anatomy_region, r.radius, r.perturbation_id)] = r
    out: Dict[Tuple[str, float], List[ExperimentResultRecord]] = {}
    for (region, radius, _pid), r in seen.items():
        out.setdefault((region, radius), []).append(r)
    return out


# =============================================================================================
# Section 1: run integrity (+ the data-integrity / encoder-cache finding)
# =============================================================================================


def validate_run_integrity(
    records: Sequence[ExperimentResultRecord], checkpoint: Stage7bCheckpointManifest, run_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    n_rows = len(records)
    perturbation_ids = {r.perturbation_id for r in records}
    n_unique = len(perturbation_ids)

    rows_per_perturbation = {}
    for r in records:
        rows_per_perturbation.setdefault(r.perturbation_id, set()).add(r.capability)
    capability_counts_ok = all(len(caps) == len(checkpoint.capabilities) for caps in rows_per_perturbation.values())

    cell_counts: Dict[Tuple[str, float], int] = {}
    for r in records:
        cell_counts[(r.anatomy_region, r.radius)] = cell_counts.get((r.anatomy_region, r.radius), 0) + 1
    expected_cell_n = FULL_CALIBRATION_N_PER_CELL * len(checkpoint.capabilities)
    expected_cells = {(region, radius) for region in FULL_CALIBRATION_REGIONS for radius in FULL_CALIBRATION_RADII}
    grid_ok = (
        set(cell_counts.keys()) == expected_cells
        and all(n == expected_cell_n for n in cell_counts.values())
    )

    model_revisions = {r.model_revision for r in records}
    model_revision_ok = model_revisions == {checkpoint.model_revision}

    mask_hash_ok = True
    mask_hash_mismatches: List[str] = []
    for r in records:
        expected_hash = checkpoint.region_mask_hashes.get(r.anatomy_region)
        if expected_hash is None or r.parameter_mask_hash != expected_hash:
            mask_hash_ok = False
            mask_hash_mismatches.append(r.perturbation_id)

    method_values = {r.runtime_metadata.get("radius_realization_method") for r in records}
    method_ok = method_values == {RADIUS_REALIZATION_METHOD} == {checkpoint.radius_realization_method}

    run_complete = bool(run_manifest.get("run_complete", False)) if run_manifest else (
        n_unique == checkpoint.expected_unique_perturbations and n_rows == checkpoint.expected_result_rows
    )

    acceptance_counts_by_region_radius: Dict[str, Dict[str, int]] = {}
    for (region, radius), cands in sorted(unique_candidates_by_region_radius(records).items()):
        key = f"{region}|{_radius_key(radius)}"
        counts = {"strict": 0, "quantization_limited": 0}
        for c in cands:
            mode = c.runtime_metadata["radius_acceptance_mode"]
            counts[mode] = counts.get(mode, 0) + 1
        acceptance_counts_by_region_radius[key] = counts

    max_relative_radius_error = max(
        (r.runtime_metadata["relative_radius_error"] for r in records), default=None,
    )

    report = {
        "expected_unique_perturbations": checkpoint.expected_unique_perturbations,
        "actual_unique_perturbations": n_unique,
        "expected_result_rows": checkpoint.expected_result_rows,
        "actual_result_rows": n_rows,
        "capability_rows_per_perturbation_complete": capability_counts_ok,
        "grid_3x6x8_complete": grid_ok,
        "n_grid_cells": len(cell_counts),
        "expected_grid_cells": len(expected_cells),
        "model_revision_uniform_and_matches_checkpoint": model_revision_ok,
        "region_mask_hashes_match_checkpoint": mask_hash_ok,
        "region_mask_hash_mismatch_count": len(mask_hash_mismatches),
        "radius_realization_method_matches_frozen_v3": method_ok,
        "radius_realization_method": sorted(method_values),
        "run_complete": run_complete,
        "quantization_limited_acceptance_counts_by_region_radius": acceptance_counts_by_region_radius,
        "max_actual_relative_radius_error": max_relative_radius_error,
        "overall_pass": bool(
            n_unique == checkpoint.expected_unique_perturbations and n_rows == checkpoint.expected_result_rows
            and capability_counts_ok and grid_ok and model_revision_ok and mask_hash_ok and method_ok and run_complete
        ),
    }
    return report


def compute_data_integrity_report(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Detects the encoder-cache artifact directly from the data: for each (capability, region)
    group, is delta identically 0.0 across every radius/seed, AND does per_example_result_hash
    collapse to a single value? A single collapsed hash across every requested radius/seed for a
    REGION THAT WAS ACTUALLY PERTURBED (region_param_count > 0, epsilon_region_l2_norm > 0 --
    checked below, not merely assumed) is only explicable by generation reusing a cached
    multimodal-encoder output computed under the UNPERTURBED base weights -- see this module's
    docstring and GATE2_CACHE_SAFETY_REVIEW.md for the full mechanism and root-cause citation.
    """
    by_cap_region: Dict[Tuple[str, str], List[ExperimentResultRecord]] = {}
    for r in records:
        by_cap_region.setdefault((r.capability, r.anatomy_region), []).append(r)

    per_group: Dict[str, Dict[str, Any]] = {}
    suspect_region_capability_pairs: List[str] = []
    for (cap, region), rows in sorted(by_cap_region.items()):
        deltas = [r.delta for r in rows]
        hashes = {r.per_example_result_hash for r in rows}
        epsilon_norms = [r.runtime_metadata.get("epsilon_region_l2_norm", 0.0) for r in rows]
        all_delta_zero = all(d == 0.0 for d in deltas)
        collapsed_to_one_hash = len(hashes) == 1
        perturbation_was_nontrivial = all(e is not None and e > 0.0 for e in epsilon_norms)
        suspect = all_delta_zero and collapsed_to_one_hash and perturbation_was_nontrivial
        key = f"{cap}|{region}"
        per_group[key] = {
            "capability": cap, "region": region, "n_rows": len(rows),
            "all_delta_exactly_zero": all_delta_zero,
            "n_unique_per_example_result_hash": len(hashes),
            "perturbation_epsilon_l2_norm_min": min(epsilon_norms) if epsilon_norms else None,
            "perturbation_epsilon_l2_norm_max": max(epsilon_norms) if epsilon_norms else None,
            "suspected_stale_encoder_cache_artifact": suspect,
        }
        if suspect:
            suspect_region_capability_pairs.append(key)

    affected_regions = sorted({key.split("|")[1] for key in suspect_region_capability_pairs})
    invalid_row_count = sum(row["n_rows"] for row in per_group.values() if row["suspected_stale_encoder_cache_artifact"])
    valid_regions = sorted(set(r.anatomy_region for r in records) - set(affected_regions))

    return {
        # Explicit, machine-readable provenance block (never let a downstream reader infer this
        # from the free-text "finding"/"conclusion" strings below) -- Stage 8 or any future
        # analysis MUST check scientific_status before consuming this run's region-level data.
        "scientific_status": "partially_invalid" if affected_regions else "valid",
        "valid_regions": valid_regions,
        "invalid_regions": affected_regions,
        "invalid_reason": (
            "stale multimodal encoder cache after anatomical weight changes" if affected_regions else None
        ),
        "invalid_row_count": invalid_row_count,
        "total_row_count": len(records),
        "finding": (
            "Every (capability, region) row-group for region in {vision, multimodal_connector_or_merger} "
            "has delta EXACTLY 0.0 across all 6 radii x 8 seeds AND collapses to a single "
            "per_example_result_hash, despite a real, nonzero perturbation being applied "
            "(epsilon_region_l2_norm > 0 confirmed per-candidate). Generation output is completely "
            "invariant to vision/connector perturbation magnitude. Root cause (confirmed by source "
            "inspection, not assumed): run_stage7b_anatomical_calibration.py launches its engine via "
            "launch_stage6_engine()/build_stage6_engine_config() -- the exact path "
            "GATE2_CACHE_SAFETY_REVIEW.md analyzed and declared safe ONLY because 'the visual "
            "encoder is never perturbed' under Stage 6. Stage 7B perturbs both vision and connector "
            "regions, violating that precondition, and never calls "
            "vlm_adapter.ensure_full_encoder_cache_reset_exposed() / "
            "vlm_adapter.reset_vllm_encoder_cache_full() anywhere (confirmed by direct grep: zero "
            "references to either name in run_stage7b_anatomical_calibration.py) -- vLLM's cached "
            "multimodal-encoder output for the fixed image inputs is therefore never invalidated, "
            "so every generation call under a vision/connector-perturbed candidate silently reuses "
            "the BASE model's cached image embeddings. language-region rows show real, "
            "radius-dependent deltas and many distinct hashes and are NOT affected (no analogous "
            "caching layer sits between language weights and the token-generation forward pass, "
            "per GATE2_CACHE_SAFETY_REVIEW.md section 1/3)."
        ),
        "affected_regions": affected_regions,
        "per_capability_region": per_group,
        "conclusion": (
            "vision and multimodal_connector_or_merger results in THIS run are SCIENTIFICALLY "
            "INVALID -- an instrumentation artifact, not a near-base finding. The cache-lifecycle "
            "fix (reset_vllm_encoder_cache_full wired into evaluate_one_calibration_candidate_rpc, "
            "multimodal_cache_policy=full_reset_on_weight_change_v1) has since been implemented in "
            "run_stage7b_anatomical_calibration.py, but THIS specific run predates that fix and "
            "must be preserved as no-cache-reset PROVENANCE only, never consumed by Stage 8 or any "
            "later anatomical analysis -- a corrected run must be executed under the NEW "
            "full_fixed_direction_bf16_quantization_aware_v3_cache_reset_v1 run_signature/output_dir "
            "before vision/connector conclusions can be drawn. language-region results in this run "
            "are NOT affected by this bug and may be used as-is."
            if affected_regions else
            "No stale-encoder-cache artifact detected in this run."
        ),
    }


# =============================================================================================
# Section 2: baseline / headroom
# =============================================================================================


def compute_baseline_headroom(baseline: Dict[str, Any]) -> Dict[str, Any]:
    return {
        cap: {"baseline_score": info["score"], "headroom_1_minus_baseline": 1.0 - info["score"]}
        for cap, info in baseline["capabilities"].items()
    }


# =============================================================================================
# Section 3: capability x region x radius calibration table
# =============================================================================================


def compute_calibration_table(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_cap_region_radius = group_by_capability_region_radius(records)
    out: Dict[str, Any] = {}
    for (cap, region, radius), deltas in by_cap_region_radius.items():
        arr = np.asarray(deltas, dtype=float)
        n = int(arr.size)
        mean, std = thicket_metrics.mean_std(deltas)
        median = float(np.median(arr))
        p_gt0 = thicket_metrics.probability_of_improvement(deltas)
        p_lt0 = thicket_metrics.probability_of_degradation(deltas)
        density = thicket_metrics.solution_density(deltas, margins=DENSITY_MARGINS)
        mass = thicket_metrics.positive_thicket_mass(deltas)

        density_cis = {}
        for m in DENSITY_MARGINS:
            n_ge = int(np.sum(arr >= m))
            density_cis[str(m)] = list(wilson_confidence_interval(n_ge, n))
        mean_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=np.mean, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

        out.setdefault(cap, {}).setdefault(region, {})[_radius_key(radius)] = {
            "radius": radius, "n": n,
            "mean_delta": mean, "mean_delta_95ci_bootstrap": list(mean_ci),
            "median_delta": median, "std_delta": std,
            "min_delta": float(arr.min()), "max_delta": float(arr.max()),
            "p_delta_gt_0": p_gt0, "p_delta_lt_0": p_lt0,
            "density": {str(m): density[m] for m in DENSITY_MARGINS},
            "density_95ci_wilson": density_cis,
            "positive_thicket_mass": mass,
        }
    return out


# =============================================================================================
# Section 4: matched-radius region comparison
# =============================================================================================


def compute_matched_radius_comparison(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_cap_radius = group_by_capability_radius(records)
    out: Dict[str, Any] = {}
    for (cap, radius), rows in by_cap_radius.items():
        by_region: Dict[str, List[float]] = {}
        for r in rows:
            by_region.setdefault(r.anatomy_region, []).append(r.delta)
        mean_by_region = {region: thicket_metrics.mean_std(d)[0] for region, d in by_region.items()}
        p_gt0_by_region = {region: thicket_metrics.probability_of_improvement(d) for region, d in by_region.items()}
        mass_by_region = {region: thicket_metrics.positive_thicket_mass(d) for region, d in by_region.items()}
        out.setdefault(cap, {})[_radius_key(radius)] = {
            "radius": radius,
            "mean_delta_by_region": mean_by_region,
            "p_delta_gt_0_by_region": p_gt0_by_region,
            "positive_thicket_mass_by_region": mass_by_region,
        }
    return out


# =============================================================================================
# Section 5: collapse / destructive regime (region x radius, aggregate across capability+seed)
# =============================================================================================


def compute_collapse_regime(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_region_radius = group_by_region_radius(records)
    out: Dict[str, Any] = {}
    for (region, radius), rows in by_region_radius.items():
        deltas = [r.delta for r in rows]
        arr = np.asarray(deltas, dtype=float)
        out.setdefault(region, {})[_radius_key(radius)] = {
            "radius": radius, "n": int(arr.size),
            "mean_capability_delta": float(arr.mean()),
            "fraction_delta_lt_0": float(np.mean(arr < 0.0)),
            "fraction_delta_le_severe": float(np.mean(arr <= -SEVERE_DEGRADATION_MARGIN)),
            "severe_margin": -SEVERE_DEGRADATION_MARGIN,
        }
    return out


# =============================================================================================
# Section 6: common-radius regime classification
# =============================================================================================


def classify_regime(mean_delta: float, p_gt0: float, p_lt0: float, density_at_02: float) -> str:
    """Purely descriptive, applied MECHANICALLY and IDENTICALLY to every radius cell using only
    that cell's own already-computed statistics -- never a cross-cell comparison, never a "pick
    the best" selection. Deliberately the SAME fixed rule as analysis/stage6_visual_thicket_
    analysis.py's classify_regime (renaming "useful" -> "active" to match this task's requested
    label set), for cross-stage consistency of what "active"/"destructive" mean:

        destructive: P(Delta<0) >= 0.5 and mean_delta <= -0.05
        near_base:   P(Delta>0) < 0.1 and P(Delta<0) < 0.1
        active:      mean_delta > 0 and density(>=0.02) >= 0.3 and P(Delta<0) < 0.5
        transition:  otherwise
    """
    if p_lt0 >= 0.5 and mean_delta <= -0.05:
        return "destructive"
    if p_gt0 < 0.1 and p_lt0 < 0.1:
        return "near_base"
    if mean_delta > 0 and density_at_02 >= 0.3 and p_lt0 < 0.5:
        return "active"
    return "transition"


def _regime_row(arr: np.ndarray) -> Dict[str, Any]:
    n = int(arr.size)
    mean = float(arr.mean())
    p_gt0 = float(np.mean(arr > 0.0))
    p_lt0 = float(np.mean(arr < 0.0))
    density_02 = float(np.mean(arr >= 0.02))
    n_gt0 = int(np.sum(arr > 0.0))
    return {
        "n": n, "mean_delta": mean, "p_delta_gt_0": p_gt0, "p_delta_gt_0_95ci_wilson": list(wilson_confidence_interval(n_gt0, n)),
        "p_delta_lt_0": p_lt0, "density_ge_0.02": density_02,
        "regime": classify_regime(mean, p_gt0, p_lt0, density_02),
    }


def classify_common_radius_regime(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Pools ALL THREE regions (and all 3 capabilities) together per radius -- this is the
    literal "COMMON radius, behavior across all three anatomical regions" classification the
    task requests. NOTE: given compute_data_integrity_report()'s finding, 2 of 3 regions'
    contribution to this pooled statistic is a constant-zero instrumentation artifact, not a
    real physical near-base measurement -- see radius_regime_summary.json's
    "data_integrity_warning" key and stage7b_analysis.md for why this table must NOT be read as
    a trustworthy common-radius decision on its own.
    """
    by_radius = group_by_radius(records)
    out: Dict[str, Any] = {}
    for radius, rows in by_radius.items():
        arr = np.asarray([r.delta for r in rows], dtype=float)
        out[_radius_key(radius)] = {"radius": radius, **_regime_row(arr)}
    return out


def classify_language_only_radius_regime(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """Supplementary table: the SAME classification rule restricted to language-region rows only
    (the one region confirmed NOT affected by the encoder-cache artifact) -- the only currently
    trustworthy per-radius regime signal in this run.
    """
    by_radius = group_by_radius([r for r in records if r.anatomy_region == "language"])
    out: Dict[str, Any] = {}
    for radius, rows in by_radius.items():
        arr = np.asarray([r.delta for r in rows], dtype=float)
        out[_radius_key(radius)] = {"radius": radius, **_regime_row(arr)}
    return out


def build_radius_regime_summary(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    return {
        "data_integrity_warning": compute_data_integrity_report(records),
        "common_radius_classification_pooled_all_regions": classify_common_radius_regime(records),
        "language_only_radius_classification_supplementary": classify_language_only_radius_regime(records),
        "region_radius_collapse_scores": compute_collapse_regime(records),
    }


# =============================================================================================
# Section 7: exploratory anatomical signal (CALIBRATION-SCALE / EXPLORATORY)
# =============================================================================================


def compute_exploratory_anatomy_signal(records: Sequence[ExperimentResultRecord], non_destructive_radii: Sequence[float]) -> Dict[str, Any]:
    """capability x anatomy matrix (mean Delta, P(Delta>0)) restricted to `non_destructive_radii`
    -- callers MUST derive this list from classify_language_only_radius_regime(), never
    classify_common_radius_regime(): the pooled/common classification is diluted by the two
    constant-zero artifact regions (2/3 of its pooled sample), which pulls P(Delta<0) below the
    destructive threshold even at radii where language itself is clearly collapsing (e.g. mean
    Delta=-0.37/-0.79) -- using it here would silently pool language's destructive-regime rows
    into this "informative" table. EXPLORATORY / CALIBRATION-SCALE only (n=8 per region x
    radius x capability cell) -- see compute_data_integrity_report() for why the
    vision/multimodal_connector_or_merger columns below reflect a caching artifact, not genuine
    anatomical non-response, and must not be read as an "experts live in language only" claim.
    """
    filtered = [r for r in records if r.radius in set(non_destructive_radii)]
    by_cap_region: Dict[Tuple[str, str], List[float]] = {}
    for r in filtered:
        by_cap_region.setdefault((r.capability, r.anatomy_region), []).append(r.delta)

    matrix: Dict[str, Dict[str, Any]] = {}
    for (cap, region), deltas in by_cap_region.items():
        arr = np.asarray(deltas, dtype=float)
        matrix.setdefault(cap, {})[region] = {
            "n": int(arr.size), "mean_delta": float(arr.mean()), "p_delta_gt_0": float(np.mean(arr > 0.0)),
        }

    return {
        "label": "CALIBRATION-SCALE / EXPLORATORY -- N=8 per cell, not a paper-final claim",
        "non_destructive_radii_used": list(non_destructive_radii),
        "capability_by_anatomy": matrix,
        "caveat": (
            "vision and multimodal_connector_or_merger columns are contaminated by the stale "
            "encoder-cache artifact documented in radius_regime_summary.json's "
            "data_integrity_warning -- their mean_delta=0.0 / p_delta_gt_0=0.0 values reflect "
            "the caching bug, not an anatomical finding about where grounding/OCR/spatial "
            "reasoning expertise resides. Only the language column is currently interpretable."
        ),
    }


# =============================================================================================
# Section 8: same-direction cross-capability diversity (region x radius)
# =============================================================================================


def _compute_sign_agreement_matrix(matrix: np.ndarray) -> np.ndarray:
    signs = np.sign(matrix)
    m = matrix.shape[1]
    agreement = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            frac = float(np.mean(signs[:, i] == signs[:, j]))
            agreement[i, j] = agreement[j, i] = frac
    return agreement


def compute_diversity_by_region_radius(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_region_radius = group_by_region_radius(records)
    out: Dict[str, Any] = {}
    for (region, radius), rows in sorted(by_region_radius.items()):
        perturbation_ids, capabilities, matrix = build_delta_matrix(rows)
        try:
            spearman = thicket_diversity.task_rank_correlation_matrix(matrix).tolist()
        except thicket_diversity.DiversityInputError:
            spearman = None
        try:
            discordance = thicket_diversity.spectral_discordance(matrix)
        except thicket_diversity.DiversityInputError:
            discordance = None
        overlap = thicket_diversity.expert_overlap_matrix(matrix, q=TOP_OVERLAP_FRACTION, q_is_fraction=True).tolist()
        sign_agreement = _compute_sign_agreement_matrix(matrix).tolist()

        out.setdefault(region, {})[_radius_key(radius)] = {
            "radius": radius, "n_perturbations": matrix.shape[0], "capabilities": list(capabilities),
            "task_rank_correlation_matrix_spearman": spearman,
            "spectral_discordance": discordance,
            "top_perturbation_overlap_jaccard_q0.25": overlap,
            "sign_agreement_matrix": sign_agreement,
            "note": (
                "n=8 -- diagnostic, not a final specialization estimate. For "
                "vision/multimodal_connector_or_merger, EVERY delta is identically 0.0 (the "
                "encoder-cache artifact), which makes task_rank_correlation_matrix_spearman "
                "come out as an EXACT, SPURIOUS 1.0 everywhere (percentile_rank_matrix's stable "
                "tie-break assigns the same arbitrary rank order to every column when the "
                "underlying values are all tied, so identical tie-broken orderings correlate "
                "perfectly) and spectral_discordance an exact 0.0 -- this reads as 'perfect "
                "agreement across capabilities', which is scientifically meaningless here: there "
                "is no real variation to agree or disagree about. Not evidence of low "
                "specialization; see the data_integrity_warning."
            ),
        }
    return out


# =============================================================================================
# Section 9: quantization audit
# =============================================================================================


def compute_quantization_audit(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_region_radius = unique_candidates_by_region_radius(records)
    per_cell: Dict[str, Any] = {}
    violations: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []

    for (region, radius), cands in sorted(by_region_radius.items()):
        strict = [c for c in cands if c.runtime_metadata["radius_acceptance_mode"] == "strict"]
        quant = [c for c in cands if c.runtime_metadata["radius_acceptance_mode"] == "quantization_limited"]
        ratios = [c.runtime_metadata["realized_relative_l2"] / c.radius for c in cands]
        rel_errors = [c.runtime_metadata["relative_radius_error"] for c in cands]

        for c in strict:
            if c.runtime_metadata["absolute_radius_error"] > REALIZED_RADIUS_TOLERANCE:
                violations.append({"perturbation_id": c.perturbation_id, "region": region, "radius": radius, "mode": "strict", "absolute_radius_error": c.runtime_metadata["absolute_radius_error"]})
        for c in quant:
            if c.runtime_metadata["relative_radius_error"] > QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE:
                violations.append({"perturbation_id": c.perturbation_id, "region": region, "radius": radius, "mode": "quantization_limited", "relative_radius_error": c.runtime_metadata["relative_radius_error"]})

        key = f"{region}|{_radius_key(radius)}"
        per_cell[key] = {
            "region": region, "radius": radius, "n_candidates": len(cands),
            "count_strict": len(strict), "count_quantization_limited": len(quant),
            "mean_realized_over_requested_ratio": float(np.mean(ratios)),
            "max_relative_radius_error": float(np.max(rel_errors)),
        }
        for c in cands:
            all_candidates.append({
                "perturbation_id": c.perturbation_id, "region": region, "radius": radius,
                "requested_relative_l2": c.runtime_metadata["requested_relative_l2"],
                "realized_relative_l2": c.runtime_metadata["realized_relative_l2"],
                "radius_acceptance_mode": c.runtime_metadata["radius_acceptance_mode"],
                "quantization_limited": c.runtime_metadata["quantization_limited"],
                "absolute_radius_error": c.runtime_metadata["absolute_radius_error"],
                "relative_radius_error": c.runtime_metadata["relative_radius_error"],
            })

    return {
        "strict_tolerance": REALIZED_RADIUS_TOLERANCE,
        "quantization_plateau_relative_tolerance": QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
        "per_region_radius": per_cell,
        "per_candidate": all_candidates,
        "n_violations": len(violations),
        "violations": violations,
        "all_accepted_candidates_within_v3_admissibility_rule": len(violations) == 0,
    }


# =============================================================================================
# Markdown report
# =============================================================================================


def _fmt(x: Optional[float], digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def build_markdown_report(
    *, integrity: Dict[str, Any], baseline_headroom: Dict[str, Any], calibration_table: Dict[str, Any],
    matched_radius: Dict[str, Any], regime_summary: Dict[str, Any], exploratory: Dict[str, Any],
    diversity: Dict[str, Any], quant_audit: Dict[str, Any], checkpoint: Stage7bCheckpointManifest,
) -> str:
    capabilities = sorted(calibration_table.keys())
    regions = sorted(checkpoint.regions)
    radii = sorted(checkpoint.radii)

    lines: List[str] = []
    lines.append("# Stage 7B Analysis: Anatomical Calibration (full run, v3 quantization-aware)")
    lines.append("")
    lines.append(
        f"Source: `results.jsonl` ({checkpoint.expected_result_rows} rows, "
        f"{checkpoint.expected_unique_perturbations} unique perturbations, "
        f"radius_realization_method={checkpoint.radius_realization_method}, "
        f"restoration_mode={checkpoint.restoration_mode}). Analysis only -- no model run, no "
        f"perturbation applied, no existing result altered."
    )
    lines.append("")

    lines.append("## Stage disambiguation")
    lines.append("")
    lines.append("- **Stage 6**: language-only global Gaussian landscape (vision encoder frozen, never perturbed).")
    lines.append("- **Stage 7B** (this document): norm-controlled anatomical calibration -- 3 regions (vision, "
                  "multimodal_connector_or_merger, language) x 6 common relative-L2 radii x 8 perturbations x 3 "
                  "capabilities, D_map N=20 per capability. Calibration-scale evidence, not the paper atlas.")
    lines.append("- **Stage 8** (future, NOT implemented here): paper-scale anatomical atlas, built on the radius set this "
                  "document recommends.")
    lines.append("")

    lines.append("## CRITICAL FINDING: stale multimodal-encoder cache invalidates vision/connector results")
    lines.append("")
    di = regime_summary["data_integrity_warning"]
    lines.append(
        f"**scientific_status = `{di['scientific_status']}`** -- "
        f"valid_regions = {di['valid_regions']}, invalid_regions = {di['invalid_regions']}, "
        f"invalid_reason = {di['invalid_reason']!r}, "
        f"**invalid_row_count = {di['invalid_row_count']} of {di['total_row_count']} total rows** "
        f"(NOT all {di['total_row_count']} rows -- only the vision + connector rows)."
    )
    lines.append("")
    lines.append(di["finding"])
    lines.append("")
    lines.append(f"**Affected regions**: {', '.join(di['affected_regions']) if di['affected_regions'] else 'none'}.")
    lines.append("")
    lines.append(f"**Conclusion**: {di['conclusion']}")
    lines.append("")
    lines.append("| capability | region | n_rows | all delta==0 | unique hashes | suspected artifact |")
    lines.append("|---|---|---|---|---|---|")
    for key in sorted(di["per_capability_region"].keys()):
        row = di["per_capability_region"][key]
        lines.append(
            f"| {row['capability']} | {row['region']} | {row['n_rows']} | {row['all_delta_exactly_zero']} | "
            f"{row['n_unique_per_example_result_hash']} | {row['suspected_stale_encoder_cache_artifact']} |"
        )
    lines.append("")

    lines.append("## 1) Run integrity")
    lines.append("")
    lines.append(f"`overall_pass={integrity['overall_pass']}`: "
                  f"{integrity['actual_unique_perturbations']}/{integrity['expected_unique_perturbations']} unique perturbations, "
                  f"{integrity['actual_result_rows']}/{integrity['expected_result_rows']} rows, "
                  f"3x6x8 grid complete={integrity['grid_3x6x8_complete']}, "
                  f"model_revision consistent={integrity['model_revision_uniform_and_matches_checkpoint']}, "
                  f"mask hashes consistent={integrity['region_mask_hashes_match_checkpoint']}, "
                  f"method=={checkpoint.radius_realization_method}: {integrity['radius_realization_method_matches_frozen_v3']}, "
                  f"run_complete={integrity['run_complete']}.")
    lines.append("")
    lines.append(f"Max actual relative-radius error observed across all accepted candidates: "
                  f"**{_fmt(integrity['max_actual_relative_radius_error'], 6)}** (admissibility bound: "
                  f"{QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE}).")
    lines.append("")
    lines.append("Quantization-limited acceptance counts by region x radius:")
    lines.append("")
    lines.append("| region | radius | strict | quantization_limited |")
    lines.append("|---|---|---|---|")
    for key in sorted(integrity["quantization_limited_acceptance_counts_by_region_radius"].keys()):
        region, radius_str = key.split("|")
        counts = integrity["quantization_limited_acceptance_counts_by_region_radius"][key]
        lines.append(f"| {region} | {radius_str} | {counts.get('strict', 0)} | {counts.get('quantization_limited', 0)} |")
    lines.append("")

    lines.append("## 2) Baseline scores and headroom")
    lines.append("")
    lines.append("| capability | baseline_score | headroom (1 - baseline) |")
    lines.append("|---|---|---|")
    for cap in capabilities:
        h = baseline_headroom[cap]
        lines.append(f"| {cap} | {h['baseline_score']:.4f} | {h['headroom_1_minus_baseline']:.4f} |")
    lines.append("")
    lines.append("Raw Delta (never headroom-normalized) is the metric used throughout every other table in this document.")
    lines.append("")

    lines.append("## 3) Capability x region x radius calibration table (compact: mean Delta / P(>0) / density>=.02)")
    lines.append("")
    for cap in capabilities:
        lines.append(f"### {cap}")
        lines.append("")
        lines.append("| region | radius | mean | P(>0) | P(<0) | d>=.02 | mass | regime (common) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        common = regime_summary["common_radius_classification_pooled_all_regions"]
        for region in regions:
            for radius in radii:
                cell = calibration_table.get(cap, {}).get(region, {}).get(_radius_key(radius))
                if cell is None:
                    continue
                regime = common.get(_radius_key(radius), {}).get("regime", "n/a")
                lines.append(
                    f"| {region} | {radius:.6f} | {cell['mean_delta']:+.4f} | {cell['p_delta_gt_0']:.3f} | "
                    f"{cell['p_delta_lt_0']:.3f} | {cell['density']['0.02']:.3f} | {cell['positive_thicket_mass']:.4f} | {regime} |"
                )
        lines.append("")

    lines.append("## 4) Matched-radius region comparison")
    lines.append("")
    lines.append(
        "At the SAME relative-L2 radius, mean Delta by region (relative-L2 normalization is already "
        "the cross-region control -- no separate parameter-count correction applied)."
    )
    lines.append("")
    for cap in capabilities:
        lines.append(f"### {cap}")
        lines.append("")
        lines.append("| radius | " + " | ".join(regions) + " |")
        lines.append("|---|" + "---|" * len(regions))
        for radius in radii:
            row = matched_radius.get(cap, {}).get(_radius_key(radius))
            if row is None:
                continue
            vals = " | ".join(f"{row['mean_delta_by_region'].get(region, float('nan')):+.4f}" for region in regions)
            lines.append(f"| {radius:.6f} | {vals} |")
        lines.append("")

    lines.append("## 5) Collapse / destructive regime by region x radius")
    lines.append("")
    lines.append("| region | radius | mean capability Delta | P(Delta<0) | P(Delta<=-0.10) |")
    lines.append("|---|---|---|---|---|")
    collapse = regime_summary["region_radius_collapse_scores"]
    for region in regions:
        for radius in radii:
            cell = collapse.get(region, {}).get(_radius_key(radius))
            if cell is None:
                continue
            lines.append(
                f"| {region} | {radius:.6f} | {cell['mean_capability_delta']:+.4f} | "
                f"{cell['fraction_delta_lt_0']:.3f} | {cell['fraction_delta_le_severe']:.3f} |"
            )
    lines.append("")

    lines.append("## 6) Common radius regime classification (pooled across all 3 regions)")
    lines.append("")
    lines.append(
        "**WARNING**: this pooled classification currently averages 2 contaminated "
        "(constant-zero, see the critical finding above) regions together with the 1 real "
        "(language) region -- it is diluted, not a valid common-radius decision, until the "
        "encoder-cache bug is fixed and vision/connector are re-run. Shown for completeness; "
        "the language-only table immediately below is the currently trustworthy signal."
    )
    lines.append("")
    lines.append("| radius | mean (pooled, contaminated) | P(>0) | P(<0) | d>=.02 | regime (pooled) |")
    lines.append("|---|---|---|---|---|---|")
    common = regime_summary["common_radius_classification_pooled_all_regions"]
    for radius in radii:
        cell = common.get(_radius_key(radius))
        if cell is None:
            continue
        lines.append(f"| {radius:.6f} | {cell['mean_delta']:+.4f} | {cell['p_delta_gt_0']:.3f} | {cell['p_delta_lt_0']:.3f} | {cell['density_ge_0.02']:.3f} | {cell['regime']} |")
    lines.append("")

    lines.append("### Language-only radius classification (supplementary, currently the trustworthy signal)")
    lines.append("")
    lines.append("| radius | mean | P(>0) | P(<0) | d>=.02 | regime |")
    lines.append("|---|---|---|---|---|---|")
    lang_only = regime_summary["language_only_radius_classification_supplementary"]
    for radius in radii:
        cell = lang_only.get(_radius_key(radius))
        if cell is None:
            continue
        lines.append(f"| {radius:.6f} | {cell['mean_delta']:+.4f} | {cell['p_delta_gt_0']:.3f} | {cell['p_delta_lt_0']:.3f} | {cell['density_ge_0.02']:.3f} | {cell['regime']} |")
    lines.append("")

    lines.append("## 7) Exploratory anatomical signal (CALIBRATION-SCALE / EXPLORATORY)")
    lines.append("")
    lines.append(f"Radii used (non-destructive per the language-only classification -- the only trustworthy per-radius signal): {exploratory['non_destructive_radii_used']}.")
    lines.append("")
    lines.append(exploratory["caveat"])
    lines.append("")
    lines.append("| capability | " + " | ".join(regions) + " (mean Delta) |")
    lines.append("|---|" + "---|" * len(regions))
    for cap in capabilities:
        row = exploratory["capability_by_anatomy"].get(cap, {})
        vals = " | ".join(f"{row.get(region, {}).get('mean_delta', float('nan')):+.4f}" for region in regions)
        lines.append(f"| {cap} | {vals} |")
    lines.append("")

    lines.append("## 8) Same-direction cross-capability diversity (region x radius, N=8 diagnostic)")
    lines.append("")
    lines.append("| region | radius | spectral discordance |")
    lines.append("|---|---|---|")
    for region in regions:
        for radius in radii:
            cell = diversity.get(region, {}).get(_radius_key(radius))
            if cell is None:
                continue
            sd = cell["spectral_discordance"]
            sd_str = "n/a (degenerate)" if sd is None else f"{sd:.4f}"
            lines.append(f"| {region} | {radius:.6f} | {sd_str} |")
    lines.append("")
    lines.append(
        "Full Spearman rank correlation matrices, top-perturbation-overlap Jaccard, and sign-agreement "
        "matrices are in `diversity_by_region_radius.json`. The exact 0.0 values for "
        "vision/multimodal_connector_or_merger above are a SPURIOUS perfect-agreement artifact of "
        "ranking constant-zero columns (see that file's own per-cell `note` field), not evidence of "
        "low specialization -- there is no real variation in those columns to agree or disagree about."
    )
    lines.append("")

    lines.append("## 9) Quantization audit")
    lines.append("")
    lines.append(f"All accepted candidates within v3 admissibility rule: "
                  f"**{quant_audit['all_accepted_candidates_within_v3_admissibility_rule']}** "
                  f"({quant_audit['n_violations']} violations).")
    lines.append("")
    lines.append("| region | radius | n | strict | quantization_limited | mean realized/requested | max rel. error |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(quant_audit["per_region_radius"].keys()):
        c = quant_audit["per_region_radius"][key]
        lines.append(
            f"| {c['region']} | {c['radius']:.6f} | {c['n_candidates']} | {c['count_strict']} | "
            f"{c['count_quantization_limited']} | {c['mean_realized_over_requested_ratio']:.6f} | "
            f"{c['max_relative_radius_error']:.6f} |"
        )
    lines.append("")

    lines.append("## 11) Stage-8 recommendation")
    lines.append("")
    lines.append(
        "**D) Does calibration give enough evidence to proceed?** **NO, not yet.** 2 of 3 anatomical "
        "regions (vision, multimodal_connector_or_merger) in this run are contaminated by the stale "
        "encoder-cache artifact documented above; a genuinely COMMON radius set cannot be chosen "
        "across all three regions from this data. **E) Issue that would invalidate Stage 8**: exactly "
        "this bug, if Stage 8 were launched on the current codebase, would silently repeat -- Stage 8 "
        "must not launch until `reset_vllm_encoder_cache_full()` is wired into "
        "`evaluate_one_calibration_candidate_rpc`'s RPC path (or equivalent) and vision/connector are "
        "re-run and re-validated with this same analysis."
    )
    lines.append("")
    lang_regimes = {radius: lang_only[_radius_key(radius)]["regime"] for radius in radii}
    active_radii = [r for r, reg in lang_regimes.items() if reg == "active"]
    near_base_radii = [r for r, reg in lang_regimes.items() if reg == "near_base"]
    lines.append(
        f"**A/B/C (language region only, the one trustworthy signal in this run)**: near_base radii = "
        f"{[f'{r:.6f}' for r in near_base_radii]}, active radii = {[f'{r:.6f}' for r in active_radii]}. "
        f"A principled COMMON radius set, once vision/connector are re-run and confirmed to behave "
        f"consistently with language's regime boundaries, should retain one near-base radius and one "
        f"active radius from this set (dropping {[f'{r:.6f}' for r, reg in lang_regimes.items() if reg == 'destructive']} "
        f"as destructive) -- see the RETURN summary for the specific proposal. This is an interim, "
        f"language-only-informed proposal, not a final Stage-8 decision."
    )
    lines.append("")

    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    records, checkpoint, baseline, run_manifest = load_all(results_dir)

    integrity = validate_run_integrity(records, checkpoint, run_manifest)
    if not integrity["overall_pass"]:
        raise RunIntegrityError(f"Stage 7B run integrity check failed: {json.dumps(integrity, indent=2, default=str)}")

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    calibration_table = compute_calibration_table(records)
    _write_json(analysis_dir / "calibration_table.json", calibration_table)

    matched_radius = compute_matched_radius_comparison(records)
    _write_json(analysis_dir / "matched_radius_anatomy_comparison.json", matched_radius)

    regime_summary = build_radius_regime_summary(records)
    _write_json(analysis_dir / "radius_regime_summary.json", regime_summary)

    lang_only_for_filter = regime_summary["language_only_radius_classification_supplementary"]
    non_destructive_radii = sorted(cell["radius"] for cell in lang_only_for_filter.values() if cell["regime"] != "destructive")
    exploratory = compute_exploratory_anatomy_signal(records, non_destructive_radii)
    _write_json(analysis_dir / "exploratory_anatomy_signal.json", exploratory)

    diversity = compute_diversity_by_region_radius(records)
    _write_json(analysis_dir / "diversity_by_region_radius.json", diversity)

    quant_audit = compute_quantization_audit(records)
    _write_json(analysis_dir / "quantization_audit.json", quant_audit)

    baseline_headroom = compute_baseline_headroom(baseline)

    report = build_markdown_report(
        integrity=integrity, baseline_headroom=baseline_headroom, calibration_table=calibration_table,
        matched_radius=matched_radius, regime_summary=regime_summary, exploratory=exploratory,
        diversity=diversity, quant_audit=quant_audit, checkpoint=checkpoint,
    )
    (analysis_dir / "stage7b_analysis.md").write_text(report)

    print(f"Wrote analysis outputs to {analysis_dir}")
    for name in (
        "calibration_table.json", "matched_radius_anatomy_comparison.json", "radius_regime_summary.json",
        "exploratory_anatomy_signal.json", "diversity_by_region_radius.json", "quantization_audit.json",
        "stage7b_analysis.md",
    ):
        print(f"  - {analysis_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
