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
from collections import Counter
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

    cache_policy_values = {r.runtime_metadata.get("multimodal_cache_policy") for r in records}
    cache_policy_ok = cache_policy_values == {checkpoint.multimodal_cache_policy} and checkpoint.multimodal_cache_policy is not None

    enable_prefix_caching_ok = checkpoint.enable_prefix_caching is False

    duplicate_row_keys = [
        key for key, count in Counter((r.perturbation_id, r.capability) for r in records).items() if count > 1
    ]
    no_duplicate_rows = len(duplicate_row_keys) == 0

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
        "multimodal_cache_policy_consistent_and_matches_checkpoint": cache_policy_ok,
        "multimodal_cache_policy": sorted(v for v in cache_policy_values if v is not None),
        "enable_prefix_caching_is_false": enable_prefix_caching_ok,
        "enable_prefix_caching": checkpoint.enable_prefix_caching,
        "no_duplicate_perturbation_capability_rows": no_duplicate_rows,
        "duplicate_row_count": len(duplicate_row_keys),
        "run_complete": run_complete,
        "quantization_limited_acceptance_counts_by_region_radius": acceptance_counts_by_region_radius,
        "max_actual_relative_radius_error": max_relative_radius_error,
        "overall_pass": bool(
            n_unique == checkpoint.expected_unique_perturbations and n_rows == checkpoint.expected_result_rows
            and capability_counts_ok and grid_ok and model_revision_ok and mask_hash_ok and method_ok
            and cache_policy_ok and enable_prefix_caching_ok and no_duplicate_rows and run_complete
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
    old_cache_artifact_reproduced = bool(affected_regions)

    if affected_regions:
        finding = (
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
        )
        conclusion = (
            "vision and multimodal_connector_or_merger results in THIS run are SCIENTIFICALLY "
            "INVALID -- an instrumentation artifact, not a near-base finding. The cache-lifecycle "
            "fix (reset_vllm_encoder_cache_full wired into evaluate_one_calibration_candidate_rpc, "
            "multimodal_cache_policy=full_reset_on_weight_change_v1 or later) has since been "
            "implemented in run_stage7b_anatomical_calibration.py, but THIS specific run predates "
            "that fix and must be preserved as no-cache-reset PROVENANCE only, never consumed by "
            "Stage 8 or any later anatomical analysis -- a corrected run must be executed under a "
            "cache-policy-suffixed run_signature/output_dir before vision/connector conclusions "
            "can be drawn. language-region results in this run are NOT affected by this bug and "
            "may be used as-is."
        )
    else:
        # REGRESSION-CHECK PASS (this repair pass, applied to the cache-safe re-run under
        # multimodal_cache_policy=full_encoder_reset_vllm011_verified_v2): reports the ACTUAL
        # positive evidence the old pathology disappeared -- real per-region hash diversity and
        # nonzero deltas -- never merely "no exception raised" / "field absent".
        region_hash_diversity = {
            region: min(
                (row["n_unique_per_example_result_hash"] for row in per_group.values() if row["region"] == region),
                default=None,
            )
            for region in sorted({row["region"] for row in per_group.values()})
        }
        finding = (
            "CACHE-ARTIFACT REGRESSION CHECK: PASSED. The same detector applied to the prior "
            "no-cache-reset run (0 of 432 rows flagged here, vs. 288 of 432 there) found every "
            "(capability, region) group's delta NOT identically zero and per_example_result_hash "
            "NOT collapsed to a single value for every region, including vision and "
            "multimodal_connector_or_merger -- minimum distinct per_example_result_hash count "
            f"observed across any (capability, region) group, by region: {region_hash_diversity}. "
            "Generation output is no longer invariant to vision/connector perturbation magnitude."
        )
        conclusion = "No stale-encoder-cache artifact detected in this run."

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
        # Direct answer to "did the stale-cache artifact disappear?" -- True means the OLD
        # pathology (delta==0 + collapsed hash despite real weight displacement) was found AGAIN
        # in this run; False (the expected value for a cache-safe run) means it was not.
        "old_cache_artifact_reproduced": old_cache_artifact_reproduced,
        "finding": finding,
        "affected_regions": affected_regions,
        "per_capability_region": per_group,
        "conclusion": conclusion,
    }


# =============================================================================================
# Section 2: baseline / headroom
# =============================================================================================


def compute_baseline_headroom(baseline: Dict[str, Any]) -> Dict[str, Any]:
    return {
        cap: {"baseline_score": info["score"], "headroom_1_minus_baseline": 1.0 - info["score"]}
        for cap, info in baseline["capabilities"].items()
    }


def validate_baseline_consistency_across_regions(
    records: Sequence[ExperimentResultRecord], baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """A baseline is computed ONCE against theta_0, before any candidate loop, and must not
    depend on which anatomy region a given candidate happens to perturb -- verifies every
    persisted record's own base_score field (regardless of anatomy_region) equals that
    capability's single canonical baseline_scores.json score exactly.
    """
    canonical = {cap: info["score"] for cap, info in baseline["capabilities"].items()}
    by_capability_region: Dict[Tuple[str, str], set] = {}
    for r in records:
        by_capability_region.setdefault((r.capability, r.anatomy_region), set()).add(r.base_score)

    mismatches: List[Dict[str, Any]] = []
    for (cap, region), scores in sorted(by_capability_region.items()):
        if len(scores) != 1 or next(iter(scores)) != canonical.get(cap):
            mismatches.append({"capability": cap, "region": region, "observed_base_scores": sorted(scores), "canonical_baseline": canonical.get(cap)})

    return {
        "canonical_baseline_by_capability": canonical,
        "consistent_across_all_regions": len(mismatches) == 0,
        "mismatches": mismatches,
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


def build_stage8_radius_recommendation(records: Sequence[ExperimentResultRecord], data_integrity_report: Dict[str, Any]) -> Dict[str, Any]:
    """Recommends 2-3 COMMON radii spanning distinct behavioral regimes -- selection is a
    STRUCTURAL/coverage criterion over classify_regime's own already-computed, UNCHANGED
    (never re-tuned) labels, never a "pick the highest-accuracy radius" search. Hard-gates on
    data_integrity_report["scientific_status"]: proceed_to_stage8 is False whenever the run is
    not fully valid, regardless of how clean the radius regime span looks.
    """
    if data_integrity_report["scientific_status"] != "valid":
        return {
            "selected_common_radii": [], "classifications": {}, "rationale": {}, "excluded_radii": {},
            "proceed_to_stage8": False,
            "blocking_issue": (
                f"data_integrity_report.scientific_status={data_integrity_report['scientific_status']!r} "
                f"(invalid_regions={data_integrity_report['invalid_regions']}) -- a Stage-8 radius "
                f"recommendation requires a fully valid, cache-safe run for ALL three regions."
            ),
        }

    common = classify_common_radius_regime(records)
    ordered = sorted(common.values(), key=lambda c: c["radius"])
    classifications = {str(c["radius"]): c["regime"] for c in ordered}
    destructive_radii = [c["radius"] for c in ordered if c["regime"] == "destructive"]
    non_destructive = [c for c in ordered if c["regime"] != "destructive"]

    if not non_destructive:
        return {
            "selected_common_radii": [], "classifications": classifications, "rationale": {},
            "excluded_radii": {str(r): "destructive" for r in destructive_radii},
            "proceed_to_stage8": False,
            "blocking_issue": "Every frozen radius classified destructive under the pooled common-radius regime -- no non-destructive anchor exists to build a Stage-8 radius set from.",
        }

    # R_small: the smallest frozen radius (always the gentlest perturbation tested) -- the
    # near-base/weakly-active anchor, whatever its own regime label turns out to be.
    r_small = non_destructive[0]

    # R_active: the LARGEST radius still classified "active" -- maximizes regime SEPARATION
    # from R_small (broadest coverage of the active regime while remaining non-destructive), a
    # structural criterion, never a max-accuracy selection.
    active_candidates = [c for c in non_destructive if c["regime"] == "active"]
    r_active = active_candidates[-1] if active_candidates else None

    # R_transition (optional): the LARGEST radius classified "transition" -- the radius closest
    # to (but not past) the active/destructive boundary, useful for demarcating it.
    transition_candidates = [c for c in non_destructive if c["regime"] == "transition"]
    r_transition = transition_candidates[-1] if transition_candidates else None

    selected = [r_small]
    rationale = {
        str(r_small["radius"]): (
            f"R_small: smallest frozen radius ({r_small['radius']:.6g}), classified "
            f"{r_small['regime']!r} under the pooled common-radius regime -- the near-base/"
            f"weakly-active anchor, representing the gentlest perturbation regime tested."
        ),
    }
    if r_active is not None and r_active["radius"] != r_small["radius"]:
        selected.append(r_active)
        rationale[str(r_active["radius"])] = (
            f"R_active: largest frozen radius still classified 'active' ({r_active['radius']:.6g}) "
            f"-- clearly behaviorally active and broadly non-destructive, chosen for maximal "
            f"regime separation from R_small, never for maximizing any single capability's score."
        )
    if r_transition is not None and r_transition["radius"] not in {c["radius"] for c in selected}:
        selected.append(r_transition)
        rationale[str(r_transition["radius"])] = (
            f"R_transition (optional): largest frozen radius classified 'transition' "
            f"({r_transition['radius']:.6g}) -- demarcates the boundary between the active and "
            f"destructive regimes, included only because a genuine transition-labeled radius "
            f"exists in this run's own classification."
        )

    selected_sorted = sorted(selected, key=lambda c: c["radius"])
    proceed = len(selected_sorted) >= 2

    return {
        "selected_common_radii": [c["radius"] for c in selected_sorted],
        "classifications": classifications,
        "rationale": rationale,
        "excluded_radii": {str(r): "destructive" for r in destructive_radii},
        "proceed_to_stage8": proceed,
        "blocking_issue": None if proceed else (
            "Fewer than 2 distinct non-destructive regimes could be identified from the pooled "
            "common-radius classification -- not enough regime span to recommend a Stage-8 "
            "radius set yet."
        ),
    }


# =============================================================================================
# Section 7: exploratory anatomical signal (CALIBRATION-SCALE / EXPLORATORY)
# =============================================================================================


def compute_exploratory_anatomy_signal(
    records: Sequence[ExperimentResultRecord], non_destructive_radii: Sequence[float],
    *, contaminated_regions: Sequence[str] = (),
) -> Dict[str, Any]:
    """capability x anatomy matrix (mean Delta, P(Delta>0)) restricted to `non_destructive_radii`.
    `contaminated_regions` (from compute_data_integrity_report()'s own affected_regions, passed
    by the caller -- this function stays pure/provenance-agnostic otherwise) controls the
    caveat text only: when non-empty, callers MUST have derived `non_destructive_radii` from
    classify_language_only_radius_regime(), never classify_common_radius_regime() (the pooled/
    common classification would be diluted by the contaminated regions' constant-zero rows,
    pulling P(Delta<0) below the destructive threshold even where the real region is clearly
    collapsing); when empty (a fully valid, cache-safe run), the pooled classification is the
    correct, non-diluted source and every region's column below is genuinely interpretable.
    EXPLORATORY / CALIBRATION-SCALE only (n=8 per region x radius x capability cell) either way.
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

    if contaminated_regions:
        caveat = (
            f"{', '.join(sorted(contaminated_regions))} column(s) are contaminated by the stale "
            "encoder-cache artifact documented in radius_regime_summary.json's "
            "data_integrity_warning -- their mean_delta=0.0 / p_delta_gt_0=0.0 values reflect "
            "the caching bug, not an anatomical finding about where grounding/OCR/spatial "
            "reasoning expertise resides. Only the uncontaminated column(s) are currently "
            "interpretable."
        )
    else:
        caveat = (
            "All three anatomical regions are scientifically valid in this run (no cache "
            "artifact detected) -- every column below is genuinely interpretable, "
            "calibration-scale anatomical signal (still N=8 per cell, still exploratory, not a "
            "paper-final 'experts live in X' claim)."
        )

    return {
        "label": "CALIBRATION-SCALE / EXPLORATORY -- N=8 per cell, not a paper-final claim",
        "non_destructive_radii_used": list(non_destructive_radii),
        "capability_by_anatomy": matrix,
        "caveat": caveat,
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


def _compute_improving_count_histogram(matrix: np.ndarray) -> Dict[str, int]:
    """Same logic as analysis/stage6_visual_thicket_analysis.py's own
    compute_improving_count_histogram -- number of perturbations improving exactly k of the M
    capabilities simultaneously (k=0..M), for M=3: none / exactly 1 / exactly 2 / all 3.
    """
    n_improving = np.sum(matrix > 0, axis=1)
    m = matrix.shape[1]
    return {str(k): int(np.sum(n_improving == k)) for k in range(m + 1)}


_DEGENERATE_ARTIFACT_NOTE = (
    "n=8 -- diagnostic, not a final specialization estimate. EVERY delta in this cell is "
    "identically 0.0 (a suspected stale-encoder-cache artifact -- see the data_integrity_warning), "
    "which makes task_rank_correlation_matrix_spearman come out as an EXACT, SPURIOUS 1.0 "
    "everywhere (percentile_rank_matrix's stable tie-break assigns the same arbitrary rank order "
    "to every column when the underlying values are all tied, so identical tie-broken orderings "
    "correlate perfectly) and spectral_discordance an exact 0.0 -- this reads as 'perfect "
    "agreement across capabilities', which is scientifically meaningless here: there is no real "
    "variation to agree or disagree about. Not evidence of low specialization."
)
_NORMAL_DIVERSITY_NOTE = "n=8 -- diagnostic, calibration-scale evidence only, not a final specialization estimate."


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
        improving_histogram = _compute_improving_count_histogram(matrix)
        cell_is_degenerate_all_zero = bool(np.all(matrix == 0.0))

        out.setdefault(region, {})[_radius_key(radius)] = {
            "radius": radius, "n_perturbations": matrix.shape[0], "capabilities": list(capabilities),
            "task_rank_correlation_matrix_spearman": spearman,
            "spectral_discordance": discordance,
            "top_perturbation_overlap_jaccard_q0.25": overlap,
            "sign_agreement_matrix": sign_agreement,
            "improving_count_histogram": improving_histogram,
            "note": _DEGENERATE_ARTIFACT_NOTE if cell_is_degenerate_all_zero else _NORMAL_DIVERSITY_NOTE,
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
    *, integrity: Dict[str, Any], baseline_headroom: Dict[str, Any], baseline_consistency: Dict[str, Any],
    calibration_table: Dict[str, Any], matched_radius: Dict[str, Any], regime_summary: Dict[str, Any],
    exploratory: Dict[str, Any], diversity: Dict[str, Any], quant_audit: Dict[str, Any],
    stage8_recommendation: Dict[str, Any], checkpoint: Stage7bCheckpointManifest,
) -> str:
    capabilities = sorted(calibration_table.keys())
    regions = sorted(checkpoint.regions)
    radii = sorted(checkpoint.radii)
    di = regime_summary["data_integrity_warning"]
    is_valid_run = di["scientific_status"] == "valid"

    lines: List[str] = []
    lines.append("# Stage 7B Analysis: Anatomical Calibration (full run, v3 quantization-aware)")
    lines.append("")
    lines.append(
        f"Source: `results.jsonl` ({checkpoint.expected_result_rows} rows, "
        f"{checkpoint.expected_unique_perturbations} unique perturbations, "
        f"radius_realization_method={checkpoint.radius_realization_method}, "
        f"multimodal_cache_policy={checkpoint.multimodal_cache_policy}, "
        f"enable_prefix_caching={checkpoint.enable_prefix_caching}, "
        f"restoration_mode={checkpoint.restoration_mode}). Analysis only -- no model run, no "
        f"perturbation applied, no existing result altered."
    )
    lines.append("")

    lines.append("## Stage disambiguation")
    lines.append("")
    lines.append("- **Stage 6**: historical `global_gaussian_upstream` run -- now proven (see `thicket.anatomy`'s "
                  "own exclusion of `visual.`/`model.visual.`-prefixed parameters) to perturb exactly the "
                  "language region, not a separate 'language-only' protocol by original design.")
    lines.append("- **Stage 7B** (this document): norm-controlled anatomical calibration -- 3 regions (vision, "
                  "multimodal_connector_or_merger, language) x 6 common relative-L2 radii x 8 perturbations x 3 "
                  "capabilities, D_map N=20 per capability, exact-norm-controlled `anatomical_relative_l2` "
                  "(distinct from Stage 6's own raw-sigma Gaussian protocol -- radii and sigmas are NEVER "
                  "compared as numerically identical anywhere in this document). Calibration-scale evidence, "
                  "not the paper atlas.")
    lines.append("- **Stage 8** (future, NOT implemented here): paper-scale anatomical atlas, built on the radius set this "
                  "document recommends.")
    lines.append("")

    if is_valid_run:
        lines.append("## Cache-artifact regression check: PASSED")
        lines.append("")
        lines.append(
            f"**scientific_status = `{di['scientific_status']}`** -- "
            f"old_cache_artifact_reproduced = **{di['old_cache_artifact_reproduced']}** -- "
            f"valid_regions = {di['valid_regions']} (all 3), invalid_regions = {di['invalid_regions']} (none)."
        )
        lines.append("")
        lines.append(di["finding"])
        lines.append("")
        lines.append(f"**Conclusion**: {di['conclusion']}")
        lines.append("")
        lines.append(
            "For comparison, the SAME detector applied to the prior no-cache-reset run at "
            "`results/stage7b_anatomical_calibration/full_fixed_direction_bf16_quantization_aware_v3/` "
            "found `old_cache_artifact_reproduced=True`, `invalid_row_count=288` of 432 -- that run "
            "remains on disk, marked `scientific_status=partially_invalid`, as no-cache-reset "
            "PROVENANCE only; its vision/connector rows are never mixed into this analysis, and its "
            "language rows are reference-only, never merged with this run's own language rows."
        )
        lines.append("")
    else:
        lines.append("## CRITICAL FINDING: stale multimodal-encoder cache invalidates vision/connector results")
        lines.append("")
        lines.append(
            f"**scientific_status = `{di['scientific_status']}`** -- "
            f"old_cache_artifact_reproduced = **{di['old_cache_artifact_reproduced']}** -- "
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
    lines.append(
        f"**Baseline consistency across regions**: {baseline_consistency['consistent_across_all_regions']} "
        f"-- every candidate row's own `base_score` (regardless of which anatomy region it perturbs) was "
        f"checked against the single canonical baseline in `baseline_scores.json` "
        f"({baseline_consistency['canonical_baseline_by_capability']}); a baseline is computed exactly "
        f"once against theta_0, before any candidate loop, so it must not depend on anatomy region."
    )
    if baseline_consistency["mismatches"]:
        lines.append("")
        lines.append(f"**MISMATCHES FOUND** ({len(baseline_consistency['mismatches'])}): {baseline_consistency['mismatches']}")
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
    if is_valid_run:
        lines.append(
            "All 3 regions are scientifically valid in this run -- this pooled classification is "
            "the authoritative COMMON-radius signal (never diluted by a contaminated region). "
            "Uses `classify_regime`, byte-identical/UNCHANGED from the prior (contaminated-run) "
            "analysis -- no threshold was retuned after seeing these corrected results."
        )
    else:
        lines.append(
            "**WARNING**: this pooled classification currently averages 2 contaminated "
            "(constant-zero, see the critical finding above) regions together with the 1 real "
            "(language) region -- it is diluted, not a valid common-radius decision, until the "
            "encoder-cache bug is fixed and vision/connector are re-run. Shown for completeness; "
            "the language-only table immediately below is the currently trustworthy signal."
        )
    lines.append("")
    lines.append(f"| radius | mean{' (pooled, contaminated)' if not is_valid_run else ''} | P(>0) | P(<0) | d>=.02 | regime (pooled) |")
    lines.append("|---|---|---|---|---|---|")
    common = regime_summary["common_radius_classification_pooled_all_regions"]
    for radius in radii:
        cell = common.get(_radius_key(radius))
        if cell is None:
            continue
        lines.append(f"| {radius:.6f} | {cell['mean_delta']:+.4f} | {cell['p_delta_gt_0']:.3f} | {cell['p_delta_lt_0']:.3f} | {cell['density_ge_0.02']:.3f} | {cell['regime']} |")
    lines.append("")

    lines.append("### Language-only radius classification (supplementary" + ("" if is_valid_run else ", currently the trustworthy signal") + ")")
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
    lines.append(f"Radii used (non-destructive per the {'pooled common' if is_valid_run else 'language-only'} classification): {exploratory['non_destructive_radii_used']}.")
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
    lines.append("| region | radius | spectral discordance | improving: none | 1 cap | 2 caps | all 3 |")
    lines.append("|---|---|---|---|---|---|---|")
    any_degenerate_cell = False
    for region in regions:
        for radius in radii:
            cell = diversity.get(region, {}).get(_radius_key(radius))
            if cell is None:
                continue
            sd = cell["spectral_discordance"]
            sd_str = "n/a (degenerate)" if sd is None else f"{sd:.4f}"
            hist = cell.get("improving_count_histogram", {})
            if cell["note"] == _DEGENERATE_ARTIFACT_NOTE:
                any_degenerate_cell = True
            lines.append(f"| {region} | {radius:.6f} | {sd_str} | {hist.get('0', '-')} | {hist.get('1', '-')} | {hist.get('2', '-')} | {hist.get('3', '-')} |")
    lines.append("")
    lines.append(
        "Full Spearman rank correlation matrices, top-perturbation-overlap Jaccard (top-2-of-8), and "
        "sign-agreement matrices are in `diversity_by_region_radius.json`, alongside this same "
        "improving-count breakdown per cell -- 'general improvement' cells (mass concentrated in the "
        "'all 3' / '2 caps' columns) vs. 'specialized' cells (mass concentrated in the '1 cap' column, "
        "with high Spectral Discordance) is the direct evidence for section 9's specialization question."
    )
    if any_degenerate_cell:
        lines.append(
            " Any cell whose `note` field flags it as degenerate (exact 0.0 spectral discordance from "
            "ranking constant-zero columns) is a caching-artifact signature, not evidence of low "
            "specialization -- see `radius_regime_summary.json`'s `data_integrity_warning`."
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

    lines.append("## 10) Comparison to Stage 6 (qualitative only -- sigma and radius are NEVER numerically compared)")
    lines.append("")
    lines.append(
        "Stage 6's `global_gaussian_upstream` protocol is now proven (via `thicket.anatomy`'s own "
        "parameter exclusion) to perturb exactly the language region, not a separate protocol -- so "
        "the only valid comparison is qualitative: does Stage 6's spatial-reasoning finding "
        "(`results/visual_thicket_global_3b_pilot/full/analysis/stage6_analysis.md`: 'a dense useful "
        "nearby thicket' at sigma in {0.0001, 0.0005, 0.001, 0.002}, `useful`/active regime, density"
        "(>=0.02) peaking at sigma=0.001) reproduce here, in the language-region row of THIS run's own "
        "regime table (section 6)?"
    )
    lines.append("")
    spatial_lang_rows = calibration_table.get("spatial_reasoning", {}).get("language", {})
    lang_regimes_current = {radius: lang_only[_radius_key(radius)]["regime"] for radius in radii}
    reproduces = any(
        lang_regimes_current.get(radius) == "active" and spatial_lang_rows.get(_radius_key(radius), {}).get("density", {}).get("0.02", 0.0) >= 0.3
        for radius in radii
    )
    lines.append(
        f"**Reproduces: {reproduces}.** Language-region spatial_reasoning regime by radius (this run): "
        f"{ {f'{r:.6g}': lang_regimes_current.get(r) for r in radii} }. "
        f"Stage 6's own sigma-indexed radii are NOT the same numeric scale as Stage 7B's relative-L2 "
        f"radii (raw Gaussian sigma vs. exact-norm-controlled relative-L2 -- see the Stage disambiguation "
        f"section above), so only the QUALITATIVE pattern (a dense, active/useful, non-destructive "
        f"small-radius language-side neighborhood for spatial reasoning) is being checked, never a "
        f"sigma==radius numeric identity."
    )
    lines.append("")

    lines.append("## 11) Stage-8 recommendation")
    lines.append("")
    if stage8_recommendation["proceed_to_stage8"]:
        lines.append(
            f"**proceed_to_stage8 = True.** Selected COMMON radii: "
            f"{[f'{r:.6g}' for r in stage8_recommendation['selected_common_radii']]}."
        )
        lines.append("")
        for radius_str, reason in stage8_recommendation["rationale"].items():
            lines.append(f"- `{radius_str}`: {reason}")
        lines.append("")
        lines.append(f"Excluded as destructive: {list(stage8_recommendation['excluded_radii'].keys())}.")
    else:
        lines.append(f"**proceed_to_stage8 = False.** {stage8_recommendation['blocking_issue']}")
    lines.append("")
    lines.append("Full machine-readable recommendation (selected_common_radii, classifications, rationale, "
                  "excluded_radii, proceed_to_stage8, blocking_issue) is in `stage8_radius_recommendation.json`.")
    lines.append("")

    lines.append("## 12) Decision gate")
    lines.append("")
    b_distinguishable = any(
        matched_radius.get(cap, {}).get(_radius_key(radius), {}).get("mean_delta_by_region", {}).get(regions[0])
        != matched_radius.get(cap, {}).get(_radius_key(radius), {}).get("mean_delta_by_region", {}).get(regions[-1])
        for cap in capabilities for radius in radii
        if matched_radius.get(cap, {}).get(_radius_key(radius)) is not None
    )
    lines.append(f"**A. Did the stale-cache artifact disappear?** {not di['old_cache_artifact_reproduced']} "
                  f"(`old_cache_artifact_reproduced={di['old_cache_artifact_reproduced']}`).")
    lines.append(f"**B. Do vision/connector/language now show distinguishable behavioral landscapes?** "
                  f"{b_distinguishable if is_valid_run else 'N/A -- run not fully valid'} "
                  f"(see section 5's matched-radius matrices for the exact per-cell values).")
    lines.append(f"**C. Evidence of capability x anatomy interaction?** See section 7 (exploratory "
                  f"anatomical signal) and section 8's improving-count histograms -- calibration-scale "
                  f"signal only (N=8), not yet a paper-final claim.")
    lines.append(f"**D. Does the spatial language-side thicket reproduce?** {reproduces} (see section 10).")
    lines.append(f"**E. Which COMMON radii should Stage 8 use?** "
                  f"{[f'{r:.6g}' for r in stage8_recommendation['selected_common_radii']] if stage8_recommendation['proceed_to_stage8'] else 'none recommended yet'} "
                  f"(see section 11).")
    lines.append(f"**F. Which radii are destructive and should be dropped?** "
                  f"{list(stage8_recommendation['excluded_radii'].keys())}.")
    lines.append(f"**G. Is Stage 7B strong enough to proceed to Stage 8?** "
                  f"{stage8_recommendation['proceed_to_stage8']} "
                  f"({'see section 11 rationale' if stage8_recommendation['proceed_to_stage8'] else stage8_recommendation['blocking_issue']}).")
    lines.append(f"**H. Remaining instrumentation concern?** "
                  f"{'None identified in this analysis pass.' if is_valid_run else 'YES -- see the critical finding above; this run is not fully valid.'} "
                  f"Quantization admissibility: {quant_audit['all_accepted_candidates_within_v3_admissibility_rule']} "
                  f"({quant_audit['n_violations']} violations); baseline region-independence: "
                  f"{baseline_consistency['consistent_across_all_regions']}.")
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

    # Non-destructive-radii filter source: the pooled common classification (classify_common_
    # radius_regime, UNCHANGED) whenever the run is fully valid -- language-only was a
    # dilution WORKAROUND for a run where 2 of 3 regions were a contaminated constant-zero
    # artifact (see compute_data_integrity_report); that precondition no longer holds for a
    # scientific_status=="valid" run, so the pooled classification is now the correct,
    # non-diluted source. This is a provenance-conditional CHOICE OF INPUT, never a change to
    # classify_regime's own thresholds/logic, which stays byte-identical either way.
    scientific_status = regime_summary["data_integrity_warning"]["scientific_status"]
    if scientific_status == "valid":
        filter_source = regime_summary["common_radius_classification_pooled_all_regions"]
    else:
        filter_source = regime_summary["language_only_radius_classification_supplementary"]
    non_destructive_radii = sorted(cell["radius"] for cell in filter_source.values() if cell["regime"] != "destructive")
    exploratory = compute_exploratory_anatomy_signal(
        records, non_destructive_radii, contaminated_regions=regime_summary["data_integrity_warning"]["affected_regions"],
    )
    _write_json(analysis_dir / "exploratory_anatomy_signal.json", exploratory)

    diversity = compute_diversity_by_region_radius(records)
    _write_json(analysis_dir / "diversity_by_region_radius.json", diversity)

    quant_audit = compute_quantization_audit(records)
    _write_json(analysis_dir / "quantization_audit.json", quant_audit)

    baseline_headroom = compute_baseline_headroom(baseline)
    baseline_consistency = validate_baseline_consistency_across_regions(records, baseline)

    stage8_recommendation = build_stage8_radius_recommendation(records, regime_summary["data_integrity_warning"])
    _write_json(analysis_dir / "stage8_radius_recommendation.json", stage8_recommendation)

    report = build_markdown_report(
        integrity=integrity, baseline_headroom=baseline_headroom, baseline_consistency=baseline_consistency,
        calibration_table=calibration_table, matched_radius=matched_radius, regime_summary=regime_summary,
        exploratory=exploratory, diversity=diversity, quant_audit=quant_audit,
        stage8_recommendation=stage8_recommendation, checkpoint=checkpoint,
    )
    (analysis_dir / "stage7b_analysis.md").write_text(report)

    print(f"Wrote analysis outputs to {analysis_dir}")
    for name in (
        "calibration_table.json", "matched_radius_anatomy_comparison.json", "radius_regime_summary.json",
        "exploratory_anatomy_signal.json", "diversity_by_region_radius.json", "quantization_audit.json",
        "stage8_radius_recommendation.json", "stage7b_analysis.md",
    ):
        print(f"  - {analysis_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
