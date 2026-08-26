"""Stage 9 analysis: does Stage-8's coarse L1 anatomy (vision / language) resolve into sharper
early/mid/late depth localization? Built and run against the REAL, complete Stage-9 full run
(results/stage9_hierarchical_anatomical_atlas/stage9_hierarchical_anatomical_atlas_3b_v1/,
1152/1152 perturbations, 6912/6912 rows, run_complete=true).

Reuses stage8_coarse_anatomical_atlas_analysis.py's OWN already-tested statistical machinery BY
IMPORT, never reimplemented -- `ExperimentResultRecord.anatomy_region` is a generic field, and
Stage 9's records populate it with the CHILD region name (e.g. "vision_early") exactly the way
Stage 8's records populate it with the L1 region name (e.g. "vision") -- so every one of Stage
8's region-agnostic functions (compute_primary_measurements, compute_anatomical_contrasts [now
accepting an optional `contrast_pairs` override, added additively for this reuse],
apply_benjamini_hochberg_correction, compute_radius_trajectories,
compute_cross_capability_specialization, compute_anatomy_capability_interaction,
compute_quantization_audit, compute_quantization_confound_audit, compute_baseline_table) work
UNCHANGED on Stage-9 records, satisfying the task's "use the SAME definitions as Stage 8"
requirement by literal code reuse rather than parallel reimplementation.

NUMERICAL PATCH AUDIT LIMITATION (stated here, not glossed over): `runtime_metadata` as
persisted by run_stage9_hierarchical_anatomical_atlas.py's `evaluate_one_stage9_candidate_rpc`
does NOT carry a `bracket_expansion_used` flag per row (only `radius_acceptance_mode`,
`quantization_limited`, `relative_radius_error`, etc.) -- that field exists on the raw
`scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3` return dict but was never
added to the whitelisted keys written into each ExperimentResultRecord. This means EXACT
per-candidate identification of "did bracket expansion specifically fire" is not recoverable
from results.jsonl alone. `compute_numerical_patch_audit` below reports what IS honestly
recoverable: the one specific candidate named in the bug report (region=language_late,
seed=980336641146292533), full strict-vs-quantization_limited counts (a strict superset of
bracket-expansion-resolved candidates), and a hard admissibility check across all 6912 rows.

Usage:
    python analysis/stage9_hierarchical_localization_analysis.py [--results-dir <path>] [--stage8-analysis-dir <path>]
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
from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import (  # noqa: E402
    STAGE8_AUTHORITATIVE_BASELINE, STAGE8_AUTHORITATIVE_SUBSET_HASHES, STAGE9_CAPABILITIES,
    STAGE9_D_MAP_N, STAGE9_N_DIRECTIONS_PER_CELL, STAGE9_RADII,
)
from neural_thickets_repro.run_stage7b_anatomical_calibration import (  # noqa: E402
    ENABLE_PREFIX_CACHING, MULTIMODAL_CACHE_POLICY, RADIUS_REALIZATION_METHOD,
)
from neural_thickets_repro.thicket.anatomy_stage9 import STAGE9_CHILD_REGIONS  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402
from neural_thickets_repro.scoped_anatomical_perturbation import (  # noqa: E402
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE, RADIUS_REALIZATION_TOLERANCE,
)

import stage8_coarse_anatomical_atlas_analysis as s8a  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "stage9_hierarchical_anatomical_atlas" / "stage9_hierarchical_anatomical_atlas_3b_v1"
DEFAULT_STAGE8_ANALYSIS_DIR = (
    REPO_ROOT / "results" / "stage8_coarse_anatomical_atlas" / "stage8_coarse_anatomical_atlas_3b_v2_batched10" / "analysis"
)

VISION_CHILD_REGIONS: Tuple[str, ...] = ("vision_early", "vision_mid", "vision_late")
LANGUAGE_CHILD_REGIONS: Tuple[str, ...] = ("language_early", "language_mid", "language_late")
VISION_DEPTH_PAIRS: Tuple[Tuple[str, str], ...] = (("vision_early", "vision_mid"), ("vision_early", "vision_late"), ("vision_mid", "vision_late"))
LANGUAGE_DEPTH_PAIRS: Tuple[Tuple[str, str], ...] = (("language_early", "language_mid"), ("language_early", "language_late"), ("language_mid", "language_late"))
CHILD_TO_PARENT: Dict[str, str] = {
    "vision_early": "vision", "vision_mid": "vision", "vision_late": "vision",
    "language_early": "language", "language_mid": "language", "language_late": "language",
}
KNOWN_BRACKET_EXPANSION_CANDIDATE = {"anatomy_region": "language_late", "seed": 980336641146292533}


def _sanitize(obj: Any) -> Any:
    return s8a._sanitize(obj)


def _write_json(path: Path, obj: Any) -> None:
    s8a._write_json(path, obj)


def load_all(results_dir: Path) -> List[ExperimentResultRecord]:
    return load_records(results_dir / "results.jsonl")


# =================================================================================================
# Section 1: integrity
# =================================================================================================


class Stage9AnalysisIntegrityError(RuntimeError):
    """The raw Stage-9 results.jsonl fails the frozen design's hard-verification gate -- never
    silently analyzed further.
    """


def run_stage9_integrity_gate(records: Sequence[ExperimentResultRecord], checkpoint: Dict[str, Any], run_manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    checks["six_child_regions"] = {r.anatomy_region for r in records} == set(STAGE9_CHILD_REGIONS)
    checks["three_frozen_radii"] = {r.radius for r in records} == set(STAGE9_RADII)
    checks["six_capabilities"] = {r.capability for r in records} == set(STAGE9_CAPABILITIES)
    checks["expected_total_rows_6912"] = len(records) == len(STAGE9_CHILD_REGIONS) * len(STAGE9_RADII) * STAGE9_N_DIRECTIONS_PER_CELL * len(STAGE9_CAPABILITIES)

    by_pid: Dict[str, List[ExperimentResultRecord]] = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, []).append(r)
    checks["expected_1152_unique_perturbations"] = len(by_pid) == len(STAGE9_CHILD_REGIONS) * len(STAGE9_RADII) * STAGE9_N_DIRECTIONS_PER_CELL
    checks["exactly_6_rows_per_perturbation"] = all(len(rows) == len(STAGE9_CAPABILITIES) for rows in by_pid.values())
    checks["same_candidate_evaluated_on_all_6_capabilities"] = all(
        {row.capability for row in rows} == set(STAGE9_CAPABILITIES) for rows in by_pid.values()
    )
    checks["no_duplicate_capability_rows_within_a_perturbation"] = all(
        len({row.capability for row in rows}) == len(rows) for rows in by_pid.values()
    )

    by_region_radius: Dict[Tuple[str, float], set] = {}
    for pid, rows in by_pid.items():
        key = (rows[0].anatomy_region, rows[0].radius)
        by_region_radius.setdefault(key, set()).add(pid)
    expected_cells = {(region, radius) for region in STAGE9_CHILD_REGIONS for radius in STAGE9_RADII}
    checks["no_missing_cells"] = set(by_region_radius.keys()) == expected_cells
    checks["exactly_64_perturbations_per_child_x_radius"] = all(v == STAGE9_N_DIRECTIONS_PER_CELL for v in {k: len(v) for k, v in by_region_radius.items()}.values())

    by_region_seed: Dict[str, Dict[Any, set]] = {}
    for r in records:
        seed = r.runtime_metadata.get("direction_seed")
        by_region_seed.setdefault(r.anatomy_region, {}).setdefault(seed, set()).add(r.radius)
    seed_reuse_ok = True
    for region in STAGE9_CHILD_REGIONS:
        seed_map = by_region_seed.get(region, {})
        if len(seed_map) != STAGE9_N_DIRECTIONS_PER_CELL:
            seed_reuse_ok = False
        if any(radii_seen != set(STAGE9_RADII) for radii_seen in seed_map.values()):
            seed_reuse_ok = False
    checks["direction_seed_reused_across_all_3_radii_within_child_region"] = seed_reuse_ok

    direction_family_ids = {r.runtime_metadata.get("direction_family_id") for r in records}
    checks["direction_family_ids_are_child_region_qualified"] = all(
        fid is not None and fid.split(":")[0] in STAGE9_CHILD_REGIONS for fid in direction_family_ids
    )

    checks["model_revision_consistent"] = len({r.model_revision for r in records}) == 1
    checks["model_revision"] = next(iter({r.model_revision for r in records}), None)
    checks["d_map_n_50"] = checkpoint.get("d_map_n") == STAGE9_D_MAP_N
    checks["subset_hashes_match_stage8_authoritative"] = checkpoint.get("subset_hashes") == STAGE8_AUTHORITATIVE_SUBSET_HASHES
    checks["all_six_child_mask_hashes_present"] = set(checkpoint.get("child_mask_hashes", {}).keys()) == set(STAGE9_CHILD_REGIONS)
    # Mechanical self-consistency: every row's own parameter_mask_hash for a given child region
    # must equal the checkpoint's own recorded hash for that region (no external ground truth
    # for "correct" mask contents is recomputable here without the real model's live parameter
    # names -- that hard gate already ran, and hard-failed the run if violated, during the real
    # Stage-9 execution itself; this check only confirms the persisted artifacts remain
    # internally consistent after the fact).
    child_mask_hashes = checkpoint.get("child_mask_hashes", {})
    checks["row_level_parameter_mask_hash_matches_checkpoint_child_mask_hashes"] = all(
        r.parameter_mask_hash == child_mask_hashes.get(r.anatomy_region) for r in records
    )
    checks["partition_audit_hash_present"] = bool(checkpoint.get("partition_audit_hash"))
    checks["partition_audit_hash_self_consistent_with_run_manifest"] = checkpoint.get("partition_audit_hash") == run_manifest.get("partition_audit_hash")
    checks["enable_prefix_caching_false"] = checkpoint.get("enable_prefix_caching") is False and ENABLE_PREFIX_CACHING is False
    checks["cache_policy_correct"] = checkpoint.get("multimodal_cache_policy") == MULTIMODAL_CACHE_POLICY
    checks["generation_batch_size_10"] = checkpoint.get("generation_batch_size") == 10
    checks["radius_realization_method_correct"] = checkpoint.get("radius_realization_method") == RADIUS_REALIZATION_METHOD
    checks["run_complete"] = bool(run_manifest.get("run_complete"))
    checks["actual_unique_perturbations_matches_expected"] = run_manifest.get("actual_unique_perturbations") == run_manifest.get("expected_unique_perturbations") == 1152
    checks["actual_result_rows_matches_expected"] = run_manifest.get("actual_result_rows") == run_manifest.get("expected_result_rows") == 6912

    non_meta_keys = [k for k in checks if k not in ("model_revision",)]
    checks["all_checks_pass"] = all(bool(checks[k]) for k in non_meta_keys if isinstance(checks[k], bool))
    return checks


def ensure_stage9_analysis_integrity(integrity_report: Dict[str, Any]) -> None:
    if not integrity_report.get("all_checks_pass"):
        failed = {k: v for k, v in integrity_report.items() if isinstance(v, bool) and not v}
        raise Stage9AnalysisIntegrityError(f"Stage-9 analysis integrity gate FAILED -- refusing to analyze. Failed checks: {failed}")


# =================================================================================================
# Section 2: baselines (reused from Stage 8 by identity -- compute_baseline_table is generic)
# =================================================================================================


def compute_baseline_table(records: Sequence[ExperimentResultRecord], baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
    return s8a.compute_baseline_table(records, baseline_scores)


def ensure_baselines_match_stage8_authoritative(baseline_table: Dict[str, Any]) -> Dict[str, Any]:
    mismatches = {
        cap: {"stage9_observed": row["baseline_score"], "stage8_authoritative": STAGE8_AUTHORITATIVE_BASELINE.get(cap)}
        for cap, row in baseline_table.items()
        if row["baseline_score"] != STAGE8_AUTHORITATIVE_BASELINE.get(cap)
    }
    return {"all_match_stage8_authoritative": len(mismatches) == 0, "mismatches": mismatches}


# =================================================================================================
# Section 3: primary 108-cell depth atlas (reused from Stage 8 by identity)
# =================================================================================================


def compute_depth_atlas(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    """compute_primary_measurements is region-agnostic (keys off ExperimentResultRecord.
    anatomy_region, whatever string that happens to be) -- reused UNCHANGED, producing exactly
    the SAME per-cell statistic set Stage 8 used, now over 6 child regions x 3 radii x 6
    capabilities = 108 cells instead of Stage 8's 3 x 3 x 6 = 54.
    """
    return s8a.compute_primary_measurements(records)


# =================================================================================================
# Section 6: depth selectivity (within-parent only) -- reuses compute_anatomy_capability_
# interaction by calling it separately on vision-only and language-only record subsets, so
# positive-thicket-mass normalization happens WITHIN each parent's 3 depth bands, never across
# all 6 child regions at once.
# =================================================================================================


def _filter_records(records: Sequence[ExperimentResultRecord], regions: Sequence[str]) -> List[ExperimentResultRecord]:
    region_set = set(regions)
    return [r for r in records if r.anatomy_region in region_set]


_DEPTH_TERMINOLOGY_NOTE = (
    "Any concentration reported here is a Stage-9 mapping-scale depth preference / hierarchical "
    "concentration on D_map exploratory data -- NOT a confirmed location claim."
)


def compute_depth_selectivity(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    vision_records = _filter_records(records, VISION_CHILD_REGIONS)
    language_records = _filter_records(records, LANGUAGE_CHILD_REGIONS)
    vision_interaction = s8a.compute_anatomy_capability_interaction(vision_records)
    language_interaction = s8a.compute_anatomy_capability_interaction(language_records)
    vision_interaction["terminology_note"] = _DEPTH_TERMINOLOGY_NOTE
    language_interaction["terminology_note"] = _DEPTH_TERMINOLOGY_NOTE
    return {"vision_depth_selectivity": vision_interaction, "language_depth_selectivity": language_interaction}


# =================================================================================================
# Section 8: depth contrasts (within-parent pairwise, reused compute_anatomical_contrasts with
# the new `contrast_pairs` override, BH-corrected SEPARATELY per parent)
# =================================================================================================


def compute_depth_contrasts(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    vision_contrasts = s8a.compute_anatomical_contrasts(records, contrast_pairs=VISION_DEPTH_PAIRS)
    vision_contrasts = s8a.apply_benjamini_hochberg_correction(vision_contrasts)
    language_contrasts = s8a.compute_anatomical_contrasts(records, contrast_pairs=LANGUAGE_DEPTH_PAIRS)
    language_contrasts = s8a.apply_benjamini_hochberg_correction(language_contrasts)
    return {"vision_depth_contrasts": vision_contrasts, "language_depth_contrasts": language_contrasts}


def _count_significant_contrasts(contrasts: Dict[str, Any]) -> Dict[str, int]:
    cells = [cell for cap_map in contrasts.values() for radius_map in cap_map.values() for cell in radius_map.values()]
    return {
        "mean_delta_diff": sum(1 for c in cells if c.get("mean_delta_diff_bh_significant_fdr_0.05")),
        "density_ge_0.02_diff": sum(1 for c in cells if c.get("density_ge_0.02_diff_bh_significant_fdr_0.05")),
        "positive_thicket_mass_diff": sum(1 for c in cells if c.get("positive_thicket_mass_diff_bh_significant_fdr_0.05")),
        "n_total_contrasts": len(cells),
    }


def summarize_depth_contrast_significance(depth_contrasts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vision_depth": _count_significant_contrasts(depth_contrasts["vision_depth_contrasts"]),
        "language_depth": _count_significant_contrasts(depth_contrasts["language_depth_contrasts"]),
    }


# =================================================================================================
# Section 4: hero question -- spatial_reasoning language-depth zoom
# =================================================================================================

_LANGUAGE_DEPTH_ANSWER_MARGIN = 0.02  # same solution-margin convention used everywhere else in this project


def classify_language_depth_answer(depth_atlas: Dict[str, Any], language_contrasts: Dict[str, Any]) -> Dict[str, Any]:
    """Answers the A/B/C/D/E question from the actual density_ge_0.02 ranking of
    language_early/mid/late at each radius, plus whether that ranking is stable across radii and
    whether any pairwise depth contrast is BH-significant -- never forced to a single answer if
    the ranking changes across radii or no band is a clear leader.
    """
    cap = "spatial_reasoning"
    per_radius = depth_atlas.get(cap, {})
    radii = list(STAGE9_RADII)
    leaders_by_radius: Dict[str, Optional[str]] = {}
    detail_by_radius: Dict[str, Any] = {}
    for radius in radii:
        radius_key = str(radius)
        rows = {region: per_radius.get(region, {}).get(radius_key) for region in LANGUAGE_CHILD_REGIONS}
        if any(v is None for v in rows.values()):
            leaders_by_radius[radius_key] = None
            continue
        densities = {region: rows[region]["density_ge_0.02"] for region in LANGUAGE_CHILD_REGIONS}
        ranked = sorted(densities.items(), key=lambda kv: kv[1], reverse=True)
        top_region, top_density = ranked[0]
        second_density = ranked[1][1]
        leader = top_region if (top_density - second_density) >= _LANGUAGE_DEPTH_ANSWER_MARGIN else None
        leaders_by_radius[radius_key] = leader
        detail_by_radius[radius_key] = {
            "density_ge_0.02_by_depth": densities,
            "mean_delta_by_depth": {region: rows[region]["mean_delta"] for region in LANGUAGE_CHILD_REGIONS},
            "positive_thicket_mass_by_depth": {region: rows[region]["positive_thicket_mass"] for region in LANGUAGE_CHILD_REGIONS},
            "max_delta_by_depth": {region: rows[region]["max_delta"] for region in LANGUAGE_CHILD_REGIONS},
            "leading_depth_this_radius": leader,
            "leader_margin_over_second": top_density - second_density,
        }

    non_none_leaders = [v for v in leaders_by_radius.values() if v is not None]
    stable = len(non_none_leaders) >= 2 and len(set(non_none_leaders)) == 1

    pair_significance = {
        pair_key: {
            "density_ge_0.02_diff": cell.get("density_ge_0.02_diff"),
            "density_ge_0.02_diff_bh_significant_fdr_0.05": cell.get("density_ge_0.02_diff_bh_significant_fdr_0.05"),
            "mean_delta_diff_bh_significant_fdr_0.05": cell.get("mean_delta_diff_bh_significant_fdr_0.05"),
        }
        for radius_key, pair_map in language_contrasts.get(cap, {}).items()
        for pair_key, cell in pair_map.items()
    }

    if stable:
        answer_label = {"language_early": "A", "language_mid": "B", "language_late": "C"}[non_none_leaders[0]]
    elif len(non_none_leaders) == 0:
        answer_label = "D"
    else:
        answer_label = "E"

    return {
        "capability": cap, "leading_depth_by_radius": leaders_by_radius,
        "detail_by_radius": detail_by_radius,
        "stable_leader_across_radii": non_none_leaders[0] if stable else None,
        "answer": answer_label,
        "answer_meaning": {
            "A": "early concentrated", "B": "mid concentrated", "C": "late concentrated",
            "D": "distributed across depth (no clear leader)", "E": "changes substantially with radius",
        }[answer_label],
        "pairwise_depth_contrasts": pair_significance,
    }


# =================================================================================================
# Section 5: vision depth summary (descriptive, capability-by-capability)
# =================================================================================================


def compute_vision_depth_summary(depth_atlas: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cap in STAGE9_CAPABILITIES:
        per_region_radius = depth_atlas.get(cap, {})
        by_radius: Dict[str, Any] = {}
        for radius in STAGE9_RADII:
            radius_key = str(radius)
            rows = {region: per_region_radius.get(region, {}).get(radius_key) for region in VISION_CHILD_REGIONS}
            if any(v is None for v in rows.values()):
                continue
            densities = {region: rows[region]["density_ge_0.02"] for region in VISION_CHILD_REGIONS}
            ranked = sorted(densities.items(), key=lambda kv: kv[1], reverse=True)
            top_region, top_density = ranked[0]
            second_density = ranked[1][1]
            by_radius[radius_key] = {
                "density_ge_0.02_by_depth": densities,
                "mean_delta_by_depth": {region: rows[region]["mean_delta"] for region in VISION_CHILD_REGIONS},
                "positive_thicket_mass_by_depth": {region: rows[region]["positive_thicket_mass"] for region in VISION_CHILD_REGIONS},
                "leading_depth_this_radius": top_region if (top_density - second_density) >= _LANGUAGE_DEPTH_ANSWER_MARGIN else None,
            }
        leaders = [v["leading_depth_this_radius"] for v in by_radius.values() if v["leading_depth_this_radius"] is not None]
        stable = len(leaders) >= 2 and len(set(leaders)) == 1
        out[cap] = {"by_radius": by_radius, "stable_leading_depth_across_at_least_2_radii": leaders[0] if stable else None}
    return out


# =================================================================================================
# Section 7: Stage-8 parent -> Stage-9 child enrichment
# =================================================================================================


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def compute_parent_child_enrichment(depth_atlas: Dict[str, Any], stage8_atlas: Dict[str, Any]) -> Dict[str, Any]:
    """Compares each Stage-9 child cell against its Stage-8 WHOLE-PARENT cell at the SAME
    capability x radius. Zero/near-zero parent density or mass is handled by reporting the ratio
    as None (never a fabricated infinite/undefined number) while the (child - parent) DIFFERENCE
    is always reported regardless.
    """
    out: Dict[str, Any] = {}
    n_density_enriched = n_density_diluted = n_density_unavailable = n_density_unchanged = 0
    n_mass_enriched = n_mass_diluted = n_mass_unavailable = n_mass_unchanged = 0
    for cap in STAGE9_CAPABILITIES:
        for child in STAGE9_CHILD_REGIONS:
            parent = CHILD_TO_PARENT[child]
            for radius in STAGE9_RADII:
                radius_key = str(radius)
                child_row = depth_atlas.get(cap, {}).get(child, {}).get(radius_key)
                parent_row = stage8_atlas.get(cap, {}).get(parent, {}).get(radius_key)
                if child_row is None or parent_row is None:
                    continue
                child_density, parent_density = child_row["density_ge_0.02"], parent_row["density_ge_0.02"]
                child_mass, parent_mass = child_row["positive_thicket_mass"], parent_row["positive_thicket_mass"]
                density_ratio = _safe_ratio(child_density, parent_density)
                mass_ratio = _safe_ratio(child_mass, parent_mass)
                if density_ratio is None:
                    n_density_unavailable += 1
                elif density_ratio > 1.0:
                    n_density_enriched += 1
                elif density_ratio < 1.0:
                    n_density_diluted += 1
                else:
                    n_density_unchanged += 1
                if mass_ratio is None:
                    n_mass_unavailable += 1
                elif mass_ratio > 1.0:
                    n_mass_enriched += 1
                elif mass_ratio < 1.0:
                    n_mass_diluted += 1
                else:
                    n_mass_unchanged += 1
                out.setdefault(cap, {}).setdefault(child, {})[radius_key] = {
                    "capability": cap, "child_region": child, "parent_region": parent, "radius": radius,
                    "child_density_ge_0.02": child_density, "parent_density_ge_0.02": parent_density,
                    "density_diff_child_minus_parent": child_density - parent_density,
                    "density_ratio_child_over_parent": density_ratio,
                    "child_positive_thicket_mass": child_mass, "parent_positive_thicket_mass": parent_mass,
                    "mass_diff_child_minus_parent": child_mass - parent_mass,
                    "mass_ratio_child_over_parent": mass_ratio,
                }
    summary = {
        "n_cells_density_enriched_ratio_gt_1": n_density_enriched, "n_cells_density_diluted_ratio_lt_1": n_density_diluted,
        "n_cells_density_ratio_unchanged_exactly_1": n_density_unchanged,
        "n_cells_density_ratio_unavailable_zero_parent": n_density_unavailable,
        "n_cells_mass_enriched_ratio_gt_1": n_mass_enriched, "n_cells_mass_diluted_ratio_lt_1": n_mass_diluted,
        "n_cells_mass_ratio_unchanged_exactly_1": n_mass_unchanged,
        "n_cells_mass_ratio_unavailable_zero_parent": n_mass_unavailable,
    }
    return {"cells": out, "summary": summary}


# =================================================================================================
# Section 9: specialization within depth (reused from Stage 8 by identity -- region-agnostic,
# groups by (anatomy_region, radius) which is already the child region for Stage 9 rows)
# =================================================================================================


def compute_specialization_by_depth_radius(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    return s8a.compute_cross_capability_specialization(records)


# =================================================================================================
# Section 10: radius trajectories (reused from Stage 8 by identity)
# =================================================================================================


def compute_depth_radius_trajectories(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    return s8a.compute_radius_trajectories(records)


# =================================================================================================
# Section 11: numerical patch audit
# =================================================================================================


def compute_numerical_patch_audit(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_pid: Dict[str, List[ExperimentResultRecord]] = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, []).append(r)

    strict_count = quant_limited_count = 0
    admissibility_violations: List[Dict[str, Any]] = []
    known_candidate_rows: List[Dict[str, Any]] = []
    per_cell_quant_limited: Dict[Tuple[str, float], int] = {}

    for pid, rows in by_pid.items():
        row = rows[0]
        meta = row.runtime_metadata
        mode = meta.get("radius_acceptance_mode")
        if mode == "strict":
            strict_count += 1
            if meta.get("realized_abs_error", 0.0) > RADIUS_REALIZATION_TOLERANCE:
                admissibility_violations.append({"perturbation_id": pid, "region": row.anatomy_region, "radius": row.radius, "mode": mode, "realized_abs_error": meta.get("realized_abs_error")})
        elif mode == "quantization_limited":
            quant_limited_count += 1
            per_cell_quant_limited[(row.anatomy_region, row.radius)] = per_cell_quant_limited.get((row.anatomy_region, row.radius), 0) + 1
            if meta.get("relative_radius_error", 0.0) > QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE:
                admissibility_violations.append({"perturbation_id": pid, "region": row.anatomy_region, "radius": row.radius, "mode": mode, "relative_radius_error": meta.get("relative_radius_error")})
        else:
            admissibility_violations.append({"perturbation_id": pid, "region": row.anatomy_region, "radius": row.radius, "mode": mode, "reason": "unknown_radius_acceptance_mode"})

        if row.anatomy_region == KNOWN_BRACKET_EXPANSION_CANDIDATE["anatomy_region"] and meta.get("direction_seed") == KNOWN_BRACKET_EXPANSION_CANDIDATE["seed"]:
            known_candidate_rows = [
                {
                    "capability": rr.capability, "perturbation_id": pid, "region": rr.anatomy_region, "radius": rr.radius,
                    "direction_seed": meta.get("direction_seed"), "direction_index": meta.get("direction_index"),
                    "radius_acceptance_mode": rr.runtime_metadata.get("radius_acceptance_mode"),
                    "realized_relative_l2": rr.runtime_metadata.get("realized_relative_l2"),
                    "relative_radius_error": rr.runtime_metadata.get("relative_radius_error"),
                    "delta": rr.delta,
                }
                for rr in rows
            ]

    confound = s8a.compute_quantization_confound_audit(records)

    return {
        "limitation_note": (
            "runtime_metadata does not persist a per-row bracket_expansion_used flag -- exact "
            "per-candidate bracket-expansion identification is not recoverable from results.jsonl "
            "alone. quantization_limited_count below is a STRICT SUPERSET of bracket-expansion- "
            "resolved candidates (most quantization_limited candidates resolve within the original "
            "<=20-attempt v2/v3 search, exactly as in Stage 8; only candidates that ALSO had no "
            "bracket within the original 20 attempts needed expansion)."
        ),
        "n_unique_perturbations": len(by_pid),
        "strict_count": strict_count, "quantization_limited_count": quant_limited_count,
        "quantization_limited_count_by_child_x_radius": {f"{k[0]}@{k[1]}": v for k, v in per_cell_quant_limited.items()},
        "known_reported_bracket_expansion_candidate": {
            "region": KNOWN_BRACKET_EXPANSION_CANDIDATE["anatomy_region"], "seed": KNOWN_BRACKET_EXPANSION_CANDIDATE["seed"],
            "found_in_results": len(known_candidate_rows) > 0, "rows": known_candidate_rows,
        },
        "n_admissibility_violations": len(admissibility_violations),
        "admissibility_violations": admissibility_violations,
        "zero_admissibility_violations": len(admissibility_violations) == 0,
        "quantization_limited_vs_strict_delta_confound_check": confound,
    }


# =================================================================================================
# Section 13: figure data
# =================================================================================================


def build_figure_data(
    depth_atlas: Dict[str, Any], stage8_atlas: Dict[str, Any], enrichment: Dict[str, Any],
    specialization: Dict[str, Any], trajectories: Dict[str, Any],
) -> Dict[str, Any]:
    hierarchical_atlas = {
        "stage8_l1": {
            cap: {region: {radius_key: row["density_ge_0.02"] for radius_key, row in region_map.items()} for region, region_map in stage8_atlas.get(cap, {}).items()}
            for cap in STAGE9_CAPABILITIES
        },
        "stage9_depth": {
            cap: {region: {radius_key: row["density_ge_0.02"] for radius_key, row in region_map.items()} for region, region_map in depth_atlas.get(cap, {}).items()}
            for cap in STAGE9_CAPABILITIES
        },
    }
    strength_hierarchy = {
        cap: {region: {radius_key: row["positive_thicket_mass"] for radius_key, row in region_map.items()} for region, region_map in depth_atlas.get(cap, {}).items()}
        for cap in STAGE9_CAPABILITIES
    }
    spatial_reasoning_language_zoom = {
        region: depth_atlas.get("spatial_reasoning", {}).get(region, {}) for region in LANGUAGE_CHILD_REGIONS
    }
    enrichment_forest = enrichment["cells"]
    specialization_matrices = {
        region: {radius_key: cell.get("spearman_6x6") for radius_key, cell in radius_map.items()}
        for region, radius_map in specialization.items()
    }
    radius_trajectory_summary = {
        "sign_persistence_rate": trajectories.get("sign_persistence_rate"),
        "improvement_survival_rate": trajectories.get("improvement_survival_rate"),
        "monotonic_nonincreasing_fraction": trajectories.get("monotonic_nonincreasing_fraction"),
        "monotonic_nondecreasing_fraction": trajectories.get("monotonic_nondecreasing_fraction"),
        "non_monotonic_fraction": trajectories.get("non_monotonic_fraction"),
    }
    return {
        "A_hierarchical_atlas_density_ge_0.02": hierarchical_atlas,
        "B_strength_hierarchy_positive_thicket_mass": strength_hierarchy,
        "C_spatial_reasoning_language_zoom": spatial_reasoning_language_zoom,
        "D_parent_child_enrichment_forest": enrichment_forest,
        "E_depth_specialization_matrices": specialization_matrices,
        "F_radius_trajectories_summary": radius_trajectory_summary,
    }


# =================================================================================================
# Section 14: paper claim gate
# =================================================================================================


def compute_paper_claim_gate(
    depth_atlas: Dict[str, Any], depth_selectivity: Dict[str, Any], depth_contrast_significance: Dict[str, Any],
    enrichment: Dict[str, Any], language_depth_answer: Dict[str, Any], specialization: Dict[str, Any],
    trajectories: Dict[str, Any],
) -> Dict[str, Any]:
    any_positive_mass = any(
        row["positive_thicket_mass"] > 0
        for cap_map in depth_atlas.values() for region_map in cap_map.values() for row in region_map.values()
    )
    c1 = "strongly_supported" if any_positive_mass else "unsupported"

    n_stable_capability_depth = sum(
        1 for cap, info in depth_selectivity["vision_depth_selectivity"]["direction_A_capability_to_anatomy"].items() if info["dominance_stable_across_at_least_2_radii"]
    ) + sum(
        1 for cap, info in depth_selectivity["language_depth_selectivity"]["direction_A_capability_to_anatomy"].items() if info["dominance_stable_across_at_least_2_radii"]
    )
    total_significant_depth_contrasts = (
        depth_contrast_significance["vision_depth"]["density_ge_0.02_diff"] + depth_contrast_significance["language_depth"]["density_ge_0.02_diff"]
    )
    c2a = "strongly_supported"  # already established at Stage 8's L1 level (spatial_reasoning/language finding) -- Stage 9 doesn't re-litigate this, only refines it
    if total_significant_depth_contrasts >= 5:
        c2b = "strongly_supported"
    elif total_significant_depth_contrasts >= 1 or n_stable_capability_depth >= 2:
        c2b = "supported"
    elif n_stable_capability_depth >= 1:
        c2b = "mixed"
    else:
        c2b = "unsupported"

    spearman_values = [
        cell["spearman_6x6"][i][j]
        for region_map in specialization.values() for cell in region_map.values()
        for i in range(len(cell["capabilities"])) for j in range(i + 1, len(cell["capabilities"]))
    ]
    any_specialization = any(v < 0.9 for v in spearman_values) if spearman_values else False
    c3 = "supported" if any_specialization else "mixed"

    non_monotonic_fraction = trajectories.get("non_monotonic_fraction")
    if non_monotonic_fraction is not None and non_monotonic_fraction >= 0.3:
        c_radius = "strongly_supported"
    elif non_monotonic_fraction is not None and non_monotonic_fraction > 0.0:
        c_radius = "supported"
    else:
        c_radius = "mixed"

    return {
        "C1_nearby_visual_specialists_exist": c1,
        "C2a_expert_density_strength_depends_on_coarse_anatomy": c2a,
        "C2b_expert_density_strength_exhibits_hierarchical_depth_structure": c2b,
        "C3_nearby_experts_are_capability_specialized": c3,
        "C_radius_expert_identity_density_changes_with_scale": c_radius,
        "evidence_basis": "D_map exploratory evidence only -- no held-out D_confirm evaluation has been run for any of these claims.",
        "supporting_counts": {
            "n_capabilities_with_stable_depth_dominance": n_stable_capability_depth,
            "n_significant_depth_density_contrasts_vision_plus_language": total_significant_depth_contrasts,
            "non_monotonic_trajectory_fraction": non_monotonic_fraction,
        },
    }


# =================================================================================================
# Section 15: next-stage recommendation (descriptive, does NOT implement anything)
# =================================================================================================


def compute_next_stage_recommendation(
    depth_selectivity: Dict[str, Any], depth_contrast_significance: Dict[str, Any], language_depth_answer: Dict[str, Any],
) -> Dict[str, Any]:
    n_stable_capability_depth = sum(
        1 for cap, info in depth_selectivity["vision_depth_selectivity"]["direction_A_capability_to_anatomy"].items() if info["dominance_stable_across_at_least_2_radii"]
    ) + sum(
        1 for cap, info in depth_selectivity["language_depth_selectivity"]["direction_A_capability_to_anatomy"].items() if info["dominance_stable_across_at_least_2_radii"]
    )
    total_significant = depth_contrast_significance["vision_depth"]["density_ge_0.02_diff"] + depth_contrast_significance["language_depth"]["density_ge_0.02_diff"]
    language_answer_is_sharp = language_depth_answer["answer"] in ("A", "B", "C")

    exceptional = n_stable_capability_depth >= 4 and total_significant >= 6 and language_answer_is_sharp
    recommendation = "attention_vs_mlp_drilldown" if exceptional else "geometry_low_dimensional_structure"
    return {
        "next_stage_recommendation": recommendation,
        "rationale": (
            f"n_capabilities_with_stable_depth_dominance={n_stable_capability_depth}, "
            f"n_significant_depth_density_contrasts={total_significant}, "
            f"language_depth_answer={language_depth_answer['answer']} ({language_depth_answer['answer_meaning']}). "
            + (
                "This crosses the frozen exceptional-evidence bar (>=4 stable capability-depth "
                "dominances, >=6 significant depth contrasts, and a sharp single-depth-band "
                "language answer), so attention-vs-MLP decomposition is justified BEFORE geometry."
                if exceptional else
                "This does not cross the frozen exceptional-evidence bar for jumping ahead of the "
                "roadmap's default next step -- geometry / low-dimensional useful perturbation "
                "structure remains the recommended next stage."
            )
        ),
        "frozen_default": "geometry_low_dimensional_structure",
        "exceptional_evidence_bar_met": exceptional,
    }


# =================================================================================================
# Section 12: Stage 8 -> Stage 9 story
# =================================================================================================


def compute_stage8_stage9_story(
    depth_selectivity: Dict[str, Any], language_depth_answer: Dict[str, Any], vision_depth_summary: Dict[str, Any],
    depth_contrast_significance: Dict[str, Any], enrichment: Dict[str, Any], trajectories: Dict[str, Any],
) -> Dict[str, Any]:
    total_significant = depth_contrast_significance["vision_depth"]["density_ge_0.02_diff"] + depth_contrast_significance["language_depth"]["density_ge_0.02_diff"]
    a_sharpened = total_significant > 0
    b_resolved = language_depth_answer["answer"] in ("A", "B", "C")
    n_vision_stable = sum(1 for cap, info in vision_depth_summary.items() if info["stable_leading_depth_across_at_least_2_radii"] is not None)
    c_separated_by_depth = n_vision_stable > 0
    enrichment_summary = enrichment["summary"]
    d_more_localized = enrichment_summary["n_cells_density_enriched_ratio_gt_1"] > enrichment_summary["n_cells_density_diluted_ratio_lt_1"]
    non_monotonic_fraction = trajectories.get("non_monotonic_fraction")
    e_radius_reorganizes = non_monotonic_fraction is not None and non_monotonic_fraction > 0.0

    return {
        "A_did_stage9_sharpen_stage8_localization": {"answer": a_sharpened, "evidence": f"{total_significant} BH-significant depth-density contrasts (vision+language combined)."},
        "B_did_spatial_reasoning_language_signal_resolve_to_a_depth": {"answer": b_resolved, "resolved_to": language_depth_answer["answer"], "meaning": language_depth_answer["answer_meaning"]},
        "C_did_vision_capabilities_separate_by_depth": {"answer": c_separated_by_depth, "n_capabilities_with_stable_depth_leader": n_vision_stable},
        "D_are_experts_more_localized_at_depth_than_at_l1": {
            "answer": d_more_localized,
            "n_cells_enriched_vs_diluted": f"{enrichment_summary['n_cells_density_enriched_ratio_gt_1']} enriched vs {enrichment_summary['n_cells_density_diluted_ratio_lt_1']} diluted",
        },
        "E_does_radius_still_reorganize_expert_identity_after_depth_conditioning": {"answer": e_radius_reorganizes, "non_monotonic_trajectory_fraction": non_monotonic_fraction},
    }


# =================================================================================================
# CSV exports
# =================================================================================================


def write_depth_atlas_csv(depth_atlas: Dict[str, Any], path: Path) -> None:
    s8a.write_atlas_csv(depth_atlas, path)


def write_depth_contrasts_csv(depth_contrasts: Dict[str, Any], path: Path) -> None:
    header = ["parent", "capability", "radius", "region_a", "region_b", "mean_delta_diff", "mean_delta_diff_bh_q",
              "density_ge_0.02_diff", "density_ge_0.02_diff_bh_q", "positive_thicket_mass_diff", "positive_thicket_mass_diff_bh_q"]
    rows = []
    for parent_key, contrasts in (("vision", depth_contrasts["vision_depth_contrasts"]), ("language", depth_contrasts["language_depth_contrasts"])):
        for cap, radius_map in contrasts.items():
            for radius_key, pair_map in radius_map.items():
                for cell in pair_map.values():
                    rows.append([parent_key, cap, cell["radius"], cell["region_a"], cell["region_b"], cell["mean_delta_diff"],
                                 cell.get("mean_delta_diff_bh_q"), cell["density_ge_0.02_diff"], cell.get("density_ge_0.02_diff_bh_q"),
                                 cell["positive_thicket_mass_diff"], cell.get("positive_thicket_mass_diff_bh_q")])
    s8a._write_csv(path, header, rows)


def write_enrichment_csv(enrichment: Dict[str, Any], path: Path) -> None:
    header = ["capability", "child_region", "parent_region", "radius", "child_density_ge_0.02", "parent_density_ge_0.02",
              "density_ratio_child_over_parent", "child_positive_thicket_mass", "parent_positive_thicket_mass", "mass_ratio_child_over_parent"]
    rows = []
    for cap, region_map in enrichment["cells"].items():
        for child, radius_map in region_map.items():
            for cell in radius_map.values():
                rows.append([cap, child, cell["parent_region"], cell["radius"], cell["child_density_ge_0.02"], cell["parent_density_ge_0.02"],
                             cell["density_ratio_child_over_parent"], cell["child_positive_thicket_mass"], cell["parent_positive_thicket_mass"],
                             cell["mass_ratio_child_over_parent"]])
    s8a._write_csv(path, header, rows)


# =================================================================================================
# Markdown report
# =================================================================================================


def build_markdown_report(
    integrity: Dict[str, Any], baseline_check: Dict[str, Any], language_depth_answer: Dict[str, Any],
    story: Dict[str, Any], claim_gate: Dict[str, Any], next_stage: Dict[str, Any], numerical_audit: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Stage 9: hierarchical anatomical localization -- analysis")
    lines.append("")
    lines.append(f"Integrity gate: **{'PASS' if integrity['all_checks_pass'] else 'FAIL'}**. Model revision: `{integrity['model_revision']}`.")
    lines.append(f"Baselines match Stage-8 authoritative values: **{baseline_check['all_match_stage8_authoritative']}**.")
    lines.append("")
    lines.append("## Hero question: spatial_reasoning language depth")
    lines.append("")
    lines.append(f"Answer: **{language_depth_answer['answer']}** ({language_depth_answer['answer_meaning']}).")
    lines.append("")
    lines.append("## Stage 8 -> Stage 9 story")
    lines.append("")
    for key, val in story.items():
        lines.append(f"- **{key}**: {val['answer']}")
    lines.append("")
    lines.append("## Paper claim gate")
    lines.append("")
    for key in ("C1_nearby_visual_specialists_exist", "C2a_expert_density_strength_depends_on_coarse_anatomy",
                "C2b_expert_density_strength_exhibits_hierarchical_depth_structure", "C3_nearby_experts_are_capability_specialized",
                "C_radius_expert_identity_density_changes_with_scale"):
        lines.append(f"- **{key}**: {claim_gate[key]}")
    lines.append("")
    lines.append("## Next-stage recommendation")
    lines.append("")
    lines.append(f"**{next_stage['next_stage_recommendation']}** -- {next_stage['rationale']}")
    lines.append("")
    lines.append("## Numerical patch audit")
    lines.append("")
    lines.append(numerical_audit["limitation_note"])
    lines.append(f"strict={numerical_audit['strict_count']}, quantization_limited={numerical_audit['quantization_limited_count']}, "
                  f"admissibility_violations={numerical_audit['n_admissibility_violations']}.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--stage8-analysis-dir", default=str(DEFAULT_STAGE8_ANALYSIS_DIR))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    records = load_all(results_dir)
    checkpoint = json.loads((results_dir / "checkpoint_manifest.json").read_text())
    run_manifest = json.loads((results_dir / "run_manifest.json").read_text())
    baseline_scores = json.loads((results_dir / "baseline_scores.json").read_text())

    integrity = run_stage9_integrity_gate(records, checkpoint, run_manifest)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_json(analysis_dir / "integrity_report.json", integrity)
    ensure_stage9_analysis_integrity(integrity)
    print(f"Integrity gate PASSED ({sum(1 for v in integrity.values() if isinstance(v, bool))} checks).")

    baseline_table = compute_baseline_table(records, baseline_scores)
    baseline_check = ensure_baselines_match_stage8_authoritative(baseline_table)
    _write_json(analysis_dir / "baseline_table.json", {"baseline_table": baseline_table, "stage8_authoritative_check": baseline_check})

    depth_atlas = compute_depth_atlas(records)
    _write_json(analysis_dir / "depth_atlas_cell_statistics.json", depth_atlas)
    write_depth_atlas_csv(depth_atlas, analysis_dir / "depth_atlas_cell_statistics.csv")

    depth_contrasts = compute_depth_contrasts(records)
    _write_json(analysis_dir / "depth_contrasts.json", depth_contrasts)
    write_depth_contrasts_csv(depth_contrasts, analysis_dir / "depth_contrasts.csv")
    depth_contrast_significance = summarize_depth_contrast_significance(depth_contrasts)

    depth_selectivity = compute_depth_selectivity(records)
    _write_json(analysis_dir / "depth_selectivity.json", depth_selectivity)

    language_depth_answer = classify_language_depth_answer(depth_atlas, depth_contrasts["language_depth_contrasts"])
    vision_depth_summary = compute_vision_depth_summary(depth_atlas)

    stage8_atlas_path = Path(args.stage8_analysis_dir) / "atlas_cell_statistics.json"
    stage8_atlas = json.loads(stage8_atlas_path.read_text()) if stage8_atlas_path.exists() else {}
    enrichment = compute_parent_child_enrichment(depth_atlas, stage8_atlas)
    _write_json(analysis_dir / "parent_child_enrichment.json", enrichment)
    write_enrichment_csv(enrichment, analysis_dir / "parent_child_enrichment.csv")

    specialization = compute_specialization_by_depth_radius(records)
    _write_json(analysis_dir / "specialization_by_depth_radius.json", specialization)

    trajectories = compute_depth_radius_trajectories(records)
    _write_json(analysis_dir / "radius_trajectories.json", trajectories)

    numerical_audit = compute_numerical_patch_audit(records)
    _write_json(analysis_dir / "numerical_patch_audit.json", numerical_audit)

    bridge = {
        "language_depth_answer": language_depth_answer, "vision_depth_summary": vision_depth_summary,
        "depth_contrast_significance": depth_contrast_significance,
    }
    _write_json(analysis_dir / "stage8_stage9_bridge.json", bridge)

    story = compute_stage8_stage9_story(depth_selectivity, language_depth_answer, vision_depth_summary, depth_contrast_significance, enrichment, trajectories)
    claim_gate = compute_paper_claim_gate(depth_atlas, depth_selectivity, depth_contrast_significance, enrichment, language_depth_answer, specialization, trajectories)
    _write_json(analysis_dir / "paper_claim_gate.json", claim_gate)

    next_stage = compute_next_stage_recommendation(depth_selectivity, depth_contrast_significance, language_depth_answer)
    _write_json(analysis_dir / "next_stage_recommendation.json", next_stage)

    figure_data = build_figure_data(depth_atlas, stage8_atlas, enrichment, specialization, trajectories)
    _write_json(analysis_dir / "figure_data.json", figure_data)

    report = build_markdown_report(integrity, baseline_check, language_depth_answer, story, claim_gate, next_stage, numerical_audit)
    (analysis_dir / "stage9_analysis.md").write_text(report)

    print(f"Wrote analysis outputs to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
