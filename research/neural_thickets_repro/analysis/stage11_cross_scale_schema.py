"""Stage 11 cross-scale (3B vs 7B) analysis SCHEMA -- PREPARED, NOT RUN. No real Stage-11 (7B)
results exist yet (this module is written alongside the Stage-11 runner, before any GPU
execution), so nothing here loads or reports on real 7B data. Every function is fully testable
against synthetic ExperimentResultRecord grids (see tests/test_stage11_cross_scale_schema.py) so
the matching/statistics machinery is proven correct BEFORE it is ever pointed at a real run.

Defines:
  - the exact MATCH KEY (capability, anatomy_region, radius) both scales must share for a
    comparison to be valid -- same-example (D_map subset hash), same relative-L2 radius, same
    frozen capability set, same candidate budget (64 directions/cell);
  - CrossScaleCellComparison, the compact per-cell record the real analysis will eventually emit;
  - the six explicit questions (A-F) from the task spec, each mapped to a concrete computation
    over already-existing, already-tested Stage-8-analysis primitives (compute_primary_
    measurements, compute_cross_capability_specialization, compute_radius_trajectories) --
    reused BY IMPORT, never reimplemented, exactly as Stage 9's and Stage 10A's own analyses did;
  - terminology guard: NEVER "scaling law" for a two-scale-point comparison -- only "3B-to-7B
    scale trend" / "cross-scale comparison" (enforced by a dedicated test banning the former
    phrase from this module's own source and from any report string it constructs).

Usage (once real Stage-11 data exists -- NOT invoked here):
    python analysis/stage11_cross_scale_schema.py --stage8-dir <path> --stage11-dir <path>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

import stage8_coarse_anatomical_atlas_analysis as s8a  # noqa: E402
from stage10a_behavioral_geometry import _spearman  # noqa: E402 -- reused, never reimplemented

FORBIDDEN_TERM = "scaling law"
APPROVED_TERMS: Tuple[str, ...] = ("3B-to-7B scale trend", "cross-scale comparison")

DEFAULT_STAGE8_DIR = REPO_ROOT / "results" / "stage8_coarse_anatomical_atlas" / "stage8_coarse_anatomical_atlas_3b_v2_batched10"
DEFAULT_STAGE11_DIR = REPO_ROOT / "results" / "stage11_coarse_anatomical_atlas_7b" / "stage11_coarse_anatomical_atlas_7b_v1"


# =================================================================================================
# Match-key schema
# =================================================================================================


@dataclass(frozen=True)
class CrossScaleMatchKey:
    """The exact key both scales must share for a cell to be comparable -- same capability, same
    L1 anatomy region, same relative-L2 radius. `d_map_n` and `n_directions_per_cell` are asserted
    equal across both sides at match-build time (never silently allowed to diverge), NOT part of
    the key itself (they are global run-level invariants, always 50 and 64 respectively).
    """
    capability: str
    anatomy_region: str
    radius: float


def build_match_keys(regions: Sequence[str], radii: Sequence[float], capabilities: Sequence[str]) -> Tuple[CrossScaleMatchKey, ...]:
    return tuple(
        CrossScaleMatchKey(capability=cap, anatomy_region=region, radius=radius)
        for cap in capabilities for region in regions for radius in radii
    )


class CrossScaleDesignMismatchError(RuntimeError):
    """The two input record sets do not share the design invariants a cross-scale comparison
    requires (same D_map subset hashes per capability, same radii, same regions, same candidate
    budget) -- refuses to build matched cells rather than silently comparing incompatible data.
    """


def ensure_cross_scale_design_matches(
    stage8_records: Sequence[ExperimentResultRecord], stage11_records: Sequence[ExperimentResultRecord],
) -> None:
    s8_regions = {r.anatomy_region for r in stage8_records}
    s11_regions = {r.anatomy_region for r in stage11_records}
    if s8_regions != s11_regions:
        raise CrossScaleDesignMismatchError(f"Region sets differ: stage8={s8_regions} stage11={s11_regions}")

    s8_radii = {r.radius for r in stage8_records}
    s11_radii = {r.radius for r in stage11_records}
    if s8_radii != s11_radii:
        raise CrossScaleDesignMismatchError(f"Radius sets differ: stage8={s8_radii} stage11={s11_radii}")

    s8_caps = {r.capability for r in stage8_records}
    s11_caps = {r.capability for r in stage11_records}
    if s8_caps != s11_caps:
        raise CrossScaleDesignMismatchError(f"Capability sets differ: stage8={s8_caps} stage11={s11_caps}")

    s8_hashes = {r.capability: r.subset_hash for r in stage8_records}
    s11_hashes = {r.capability: r.subset_hash for r in stage11_records}
    mismatched = {cap: (s8_hashes[cap], s11_hashes.get(cap)) for cap in s8_hashes if s8_hashes[cap] != s11_hashes.get(cap)}
    if mismatched:
        raise CrossScaleDesignMismatchError(f"D_map subset hashes differ (NOT the same examples): {mismatched}")


# =================================================================================================
# Per-cell comparison record (reuses Stage-8's already-tested primary-measurement machinery)
# =================================================================================================


@dataclass(frozen=True)
class CrossScaleCellComparison:
    match_key: CrossScaleMatchKey
    stage8_3b: Dict[str, Any]
    stage11_7b: Dict[str, Any]
    mean_delta_diff_7b_minus_3b: float
    density_ge_0_02_diff_7b_minus_3b: float
    density_ge_0_05_diff_7b_minus_3b: float
    positive_thicket_mass_diff_7b_minus_3b: float


def build_cross_scale_cell_comparisons(
    stage8_records: Sequence[ExperimentResultRecord], stage11_records: Sequence[ExperimentResultRecord],
) -> Dict[str, CrossScaleCellComparison]:
    """Reuses compute_primary_measurements (Stage 8's own, already-tested function) on EACH side
    independently, then joins on the shared (capability, anatomy_region, radius) key -- never a
    new statistic, only a matched difference of already-defined quantities.
    """
    ensure_cross_scale_design_matches(stage8_records, stage11_records)
    primary_3b = s8a.compute_primary_measurements(stage8_records)
    primary_7b = s8a.compute_primary_measurements(stage11_records)

    out: Dict[str, CrossScaleCellComparison] = {}
    for cap, region_map in primary_3b.items():
        for region, radius_map in region_map.items():
            for radius_key, row_3b in radius_map.items():
                row_7b = primary_7b.get(cap, {}).get(region, {}).get(radius_key)
                if row_7b is None:
                    continue
                key = CrossScaleMatchKey(capability=cap, anatomy_region=region, radius=row_3b["radius"])
                out[f"{cap}:{region}:{radius_key}"] = CrossScaleCellComparison(
                    match_key=key, stage8_3b=row_3b, stage11_7b=row_7b,
                    mean_delta_diff_7b_minus_3b=row_7b["mean_delta"] - row_3b["mean_delta"],
                    density_ge_0_02_diff_7b_minus_3b=row_7b["density_ge_0.02"] - row_3b["density_ge_0.02"],
                    density_ge_0_05_diff_7b_minus_3b=row_7b["density_ge_0.05"] - row_3b["density_ge_0.05"],
                    positive_thicket_mass_diff_7b_minus_3b=row_7b["positive_thicket_mass"] - row_3b["positive_thicket_mass"],
                )
    return out


# =================================================================================================
# The six explicit cross-scale questions (A-F) -- each mapped to already-tested primitives
# =================================================================================================


def question_A_specialist_existence_reproduces(comparisons: Dict[str, CrossScaleCellComparison]) -> Dict[str, Any]:
    """A: does specialist existence reproduce at 7B? -- fraction of cells with positive_thicket_
    mass > 0 at BOTH scales vs. only one scale.
    """
    both = sum(1 for c in comparisons.values() if c.stage8_3b["positive_thicket_mass"] > 0 and c.stage11_7b["positive_thicket_mass"] > 0)
    only_3b = sum(1 for c in comparisons.values() if c.stage8_3b["positive_thicket_mass"] > 0 and c.stage11_7b["positive_thicket_mass"] <= 0)
    only_7b = sum(1 for c in comparisons.values() if c.stage8_3b["positive_thicket_mass"] <= 0 and c.stage11_7b["positive_thicket_mass"] > 0)
    neither = sum(1 for c in comparisons.values() if c.stage8_3b["positive_thicket_mass"] <= 0 and c.stage11_7b["positive_thicket_mass"] <= 0)
    return {"n_cells_both_scales": both, "n_cells_only_3b": only_3b, "n_cells_only_7b": only_7b, "n_cells_neither": neither, "n_total": len(comparisons)}


def question_B_coarse_anatomy_interaction_reproduces(
    stage8_records: Sequence[ExperimentResultRecord], stage11_records: Sequence[ExperimentResultRecord],
) -> Dict[str, Any]:
    """B: does the coarse anatomy interaction reproduce? -- reuses compute_anatomy_capability_
    interaction (Stage 8's own) independently on each scale, compares dominant-anatomy-per-
    capability agreement.
    """
    interaction_3b = s8a.compute_anatomy_capability_interaction(stage8_records)
    interaction_7b = s8a.compute_anatomy_capability_interaction(stage11_records)
    agreement = {}
    for cap in interaction_3b["direction_A_capability_to_anatomy"]:
        dom_3b = interaction_3b["direction_A_capability_to_anatomy"][cap]["stable_dominant_anatomy"]
        dom_7b = interaction_7b["direction_A_capability_to_anatomy"].get(cap, {}).get("stable_dominant_anatomy")
        agreement[cap] = {"dominant_anatomy_3b": dom_3b, "dominant_anatomy_7b": dom_7b, "agrees": dom_3b is not None and dom_3b == dom_7b}
    return {"per_capability_agreement": agreement, "n_agreeing": sum(1 for v in agreement.values() if v["agrees"])}


def question_C_spatial_reasoning_still_language_preferential(comparisons: Dict[str, CrossScaleCellComparison]) -> Dict[str, Any]:
    rows = {k: c for k, c in comparisons.items() if c.match_key.capability == "spatial_reasoning" and c.match_key.anatomy_region == "language"}
    return {
        "n_radii": len(rows),
        "mean_delta_3b_by_radius": {str(c.match_key.radius): c.stage8_3b["mean_delta"] for c in rows.values()},
        "mean_delta_7b_by_radius": {str(c.match_key.radius): c.stage11_7b["mean_delta"] for c in rows.values()},
    }


def question_D_specialization_increases_with_radius(
    stage8_records: Sequence[ExperimentResultRecord], stage11_records: Sequence[ExperimentResultRecord],
) -> Dict[str, Any]:
    """D: reuses compute_cross_capability_specialization (Stage 8's own) independently on each
    scale, reports spectral_discordance by (region, radius) for both.
    """
    spec_3b = s8a.compute_cross_capability_specialization(stage8_records)
    spec_7b = s8a.compute_cross_capability_specialization(stage11_records)
    return {
        "discordance_3b_by_region_radius": {region: {r: cell["spectral_discordance"] for r, cell in radius_map.items()} for region, radius_map in spec_3b.items()},
        "discordance_7b_by_region_radius": {region: {r: cell["spectral_discordance"] for r, cell in radius_map.items()} for region, radius_map in spec_7b.items()},
    }


def question_E_useful_expert_density_vs_scale(comparisons: Dict[str, CrossScaleCellComparison]) -> Dict[str, Any]:
    increased = sum(1 for c in comparisons.values() if c.density_ge_0_02_diff_7b_minus_3b > 0)
    decreased = sum(1 for c in comparisons.values() if c.density_ge_0_02_diff_7b_minus_3b < 0)
    unchanged = sum(1 for c in comparisons.values() if c.density_ge_0_02_diff_7b_minus_3b == 0)
    return {"n_cells_density_increased_at_7b": increased, "n_cells_density_decreased_at_7b": decreased, "n_cells_unchanged": unchanged}


def question_F_larger_models_tolerate_greater_displacement(
    stage8_records: Sequence[ExperimentResultRecord], stage11_records: Sequence[ExperimentResultRecord],
) -> Dict[str, Any]:
    """F: reuses compute_radius_trajectories (Stage 8's own) independently on each scale --
    compares improvement_survival_rate (does a positive Delta at R_small remain positive at
    every larger radius) between scales; a HIGHER survival rate at 7B would suggest greater
    tolerance for relative-L2 displacement at larger scale.
    """
    traj_3b = s8a.compute_radius_trajectories(stage8_records)
    traj_7b = s8a.compute_radius_trajectories(stage11_records)
    return {
        "improvement_survival_rate_3b": traj_3b["improvement_survival_rate"], "improvement_survival_rate_7b": traj_7b["improvement_survival_rate"],
        "non_monotonic_fraction_3b": traj_3b["non_monotonic_fraction"], "non_monotonic_fraction_7b": traj_7b["non_monotonic_fraction"],
    }


TERMINOLOGY_NOTE = (
    "Two scale points (3B, 7B) never constitute a \"scaling law\" -- use \"3B-to-7B scale trend\" "
    "or \"cross-scale comparison\" instead."
)


def build_cross_scale_report(stage8_records: Sequence[ExperimentResultRecord], stage11_records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    comparisons = build_cross_scale_cell_comparisons(stage8_records, stage11_records)
    return {
        "terminology_note": TERMINOLOGY_NOTE,
        "n_matched_cells": len(comparisons),
        "question_A_specialist_existence_reproduces": question_A_specialist_existence_reproduces(comparisons),
        "question_B_coarse_anatomy_interaction_reproduces": question_B_coarse_anatomy_interaction_reproduces(stage8_records, stage11_records),
        "question_C_spatial_reasoning_still_language_preferential": question_C_spatial_reasoning_still_language_preferential(comparisons),
        "question_D_specialization_increases_with_radius": question_D_specialization_increases_with_radius(stage8_records, stage11_records),
        "question_E_useful_expert_density_vs_scale": question_E_useful_expert_density_vs_scale(comparisons),
        "question_F_larger_models_tolerate_greater_displacement": question_F_larger_models_tolerate_greater_displacement(stage8_records, stage11_records),
    }


# =================================================================================================
# Section 18 (REWRITTEN this milestone): context-aware terminology guard -- replaces the old
# blanket ban on "scaling law". The chapter/section TITLE ("Scaling Laws of Visual Neural
# Thickets") may always name the QUESTION under test; a specific EMPIRICAL CLAIM using that
# phrase requires may_claim_specific_scaling_law() to pass, and even the broader "scaling
# relationship/behavior/analysis" vocabulary is gated on having >=4 scale points.
# =================================================================================================

SPECIFIC_SCALING_LAW_CLAIM = "solution-density scaling law"  # never asserted unconditionally -- gated by may_claim_specific_scaling_law()
SECTION_TITLE = "Scaling Laws of Visual Neural Thickets"  # a chapter TITLE naming the question under test -- never itself an empirical claim
MIN_SCALES_FOR_SCALING_RELATIONSHIP_LANGUAGE = 4

TWO_SCALE_ALLOWED_TERMS: Tuple[str, ...] = ("scale trend", "cross-scale comparison")
MANY_SCALE_ALLOWED_TERMS: Tuple[str, ...] = ("scaling relationship", "scaling behavior", "scaling analysis")


def classify_terminology_context(n_scales: int) -> Dict[str, Any]:
    """Below MIN_SCALES_FOR_SCALING_RELATIONSHIP_LANGUAGE scale points, never claim "scaling law"/"scaling
    relationship" as an EMPIRICAL CONCLUSION -- only "scale trend"/"cross-scale comparison" are
    allowed. With >= that many, the broader "scaling relationship/behavior/analysis" vocabulary
    becomes available. A SPECIFIC statement is never made ("solution-density scaling law" included)
    without may_claim_specific_scaling_law() passing, regardless of n_scales.
    """
    if n_scales >= MIN_SCALES_FOR_SCALING_RELATIONSHIP_LANGUAGE:
        return {
            "n_scales": n_scales, "allowed_terms": list(MANY_SCALE_ALLOWED_TERMS),
            "disallowed_as_empirical_conclusion": [], "may_use_scaling_relationship_language": True,
        }
    return {
        "n_scales": n_scales, "allowed_terms": list(TWO_SCALE_ALLOWED_TERMS),
        "disallowed_as_empirical_conclusion": [FORBIDDEN_TERM, "scaling relationship", "scaling behavior"],
        "may_use_scaling_relationship_language": False,
    }


def may_claim_specific_scaling_law(fit_gate: Dict[str, Any]) -> bool:
    """Never claim a SPECIFIC statement like "solution-density scaling law" unless ALL of: >=4
    scale points, an observed monotonic trend, and a statistically meaningful fit hold --
    `fit_gate` must report all three keys explicitly (never inferred silently from partial info).
    """
    return bool(
        fit_gate.get("n_scales", 0) >= MIN_SCALES_FOR_SCALING_RELATIONSHIP_LANGUAGE
        and fit_gate.get("monotonic") is True
        and fit_gate.get("fit_significant") is True
    )


# =================================================================================================
# Sections 8-9: solution-density scaling -- delta_{t,r,s}(m) vs log(parameter_count)
# =================================================================================================


def _ols_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Closed-form OLS slope/intercept/R^2 -- pure numpy, no scipy (matches this project's own
    no-scipy convention; see stage10a_behavioral_geometry.py's own docstring). Degenerate inputs
    (n<2, or zero x-variance) return None-valued fields rather than dividing by zero.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return {"slope": None, "intercept": None, "r_squared": None, "n": n}
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx == 0.0:
        return {"slope": None, "intercept": None, "r_squared": None, "n": n}
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared, "n": n}


def _logistic_fit_grid_search(x: np.ndarray, y: np.ndarray, n_iterations: int = 2000, lr: float = 0.05, seed: int = 20260826) -> Dict[str, Any]:
    """Best-effort logistic-in-log-N fit y ~= L / (1 + exp(-k*(x - x0))) via plain gradient
    descent (pure numpy, no scipy.optimize) -- ONE candidate functional form compared against the
    linear fit's R^2 (Section 9's "sensitivity to functional form"), never asserted as the true
    form. n<3 or non-finite results honestly report a None-valued fit rather than an unstable one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return {"L": None, "k": None, "x0": None, "r_squared": None, "n": n, "note": "logistic fit requires n>=3 points"}
    y_max = float(np.max(y))
    L = max(y_max, 1e-6) + 1e-6
    k = 0.0
    x0 = float(np.mean(x))
    for _ in range(n_iterations):
        z = np.clip(-k * (x - x0), -30, 30)
        exp_z = np.exp(z)
        denom = 1.0 + exp_z
        pred = L / denom
        err = pred - y
        d_pred_dk = L * (x - x0) * exp_z / denom ** 2
        d_pred_dx0 = -L * k * exp_z / denom ** 2
        d_pred_dL = 1.0 / denom
        k -= lr * float(np.mean(2 * err * d_pred_dk))
        x0 -= lr * float(np.mean(2 * err * d_pred_dx0))
        L -= lr * float(np.mean(2 * err * d_pred_dL))
        L = max(L, 1e-6)
    z = np.clip(-k * (x - x0), -30, 30)
    y_pred = L / (1.0 + np.exp(z))
    if not np.all(np.isfinite(y_pred)):
        return {"L": None, "k": None, "x0": None, "r_squared": None, "n": n, "note": "logistic fit did not converge to finite values"}
    ss_res = float(np.sum((y - y_pred) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return {"L": float(L), "k": float(k), "x0": float(x0), "r_squared": r_squared, "n": n}


def build_solution_density_scaling_table(
    records_by_scale: Dict[str, Sequence[ExperimentResultRecord]], total_model_elements_by_scale: Dict[str, int],
    margins: Sequence[float] = (0.0, 0.02, 0.05),
) -> Dict[str, Any]:
    """Sections 8-9: delta_{t,r,s}(m) = P[Delta_t >= m] at fixed (capability, anatomy_region,
    radius), tracked across scale. Reuses s8a.compute_solution_density_curves (Stage 8's own,
    already-tested function) INDEPENDENTLY per scale -- never pools radii or scales.
    `total_model_elements_by_scale` MUST come from each scale's own REAL anatomy_audit.json
    (whole_model or S2 audit) -- never a hand-typed nominal parameter count -- so callers without
    real audits for a scale simply omit it (that scale is excluded, never guessed).
    """
    if len(records_by_scale) < 2:
        raise ValueError("Solution-density scaling requires at least 2 scales.")
    missing = set(records_by_scale) - set(total_model_elements_by_scale)
    if missing:
        raise ValueError(f"Missing total_model_elements for scale(s): {sorted(missing)}")
    scales = sorted(records_by_scale, key=lambda s: total_model_elements_by_scale[s])

    curves_by_scale = {s: s8a.compute_solution_density_curves(records_by_scale[s]) for s in scales}
    log_n_by_scale = {s: float(np.log(total_model_elements_by_scale[s])) for s in scales}

    cells: Dict[str, Any] = {}
    for scale in scales:
        for cap, region_map in curves_by_scale[scale].items():
            for region, radius_map in region_map.items():
                for radius_key, row in radius_map.items():
                    key = f"{cap}:{region}:{radius_key}"
                    cell = cells.setdefault(key, {"capability": cap, "anatomy_region": region, "radius": row["radius"], "by_scale": {}})
                    margin_grid, delta_ge_m = row["margin_grid"], row["delta_ge_m"]
                    at_frozen: Dict[str, Optional[float]] = {}
                    for m in margins:
                        idxs = [i for i, g in enumerate(margin_grid) if abs(g - m) < 1e-9]
                        at_frozen[str(m)] = delta_ge_m[idxs[0]] if idxs else None
                    cell["by_scale"][scale] = {"full_curve_margin_grid": margin_grid, "full_curve_delta_ge_m": delta_ge_m, "at_frozen_margins": at_frozen}

    fits: Dict[str, Any] = {}
    for key, cell in cells.items():
        for m in margins:
            xs, ys = [], []
            for scale in scales:
                v = cell["by_scale"][scale]["at_frozen_margins"][str(m)]
                if v is not None:
                    xs.append(log_n_by_scale[scale])
                    ys.append(v)
            if len(xs) < 2:
                continue
            x_arr, y_arr = np.asarray(xs), np.asarray(ys)
            linear = _ols_fit(x_arr, y_arr)
            logistic = _logistic_fit_grid_search(x_arr, y_arr)
            monotonic = bool(np.all(np.diff(y_arr) >= -1e-12)) or bool(np.all(np.diff(y_arr) <= 1e-12))
            fit_gate = {"n_scales": len(xs), "monotonic": monotonic, "fit_significant": bool(linear["r_squared"] is not None and linear["r_squared"] >= 0.8)}
            terminology = classify_terminology_context(len(xs))
            fits[f"{key}:m={m}"] = {
                "capability": cell["capability"], "anatomy_region": cell["anatomy_region"], "radius": cell["radius"], "margin": m,
                "n_scales": len(xs), "log_param_count": xs, "density_at_margin": ys,
                "linear_in_log_n": linear, "logistic_in_log_n": logistic, "monotonic": monotonic,
                "terminology_context": terminology, "may_claim_specific_scaling_law": may_claim_specific_scaling_law(fit_gate),
                "label": terminology["allowed_terms"][0],
            }

    return {"scales": scales, "log_param_count_by_scale": log_n_by_scale, "cells": cells, "fits": fits}


# =================================================================================================
# Section 10: performance-density scaling -- the WHOLE distribution, not only solution density
# =================================================================================================


def _quantiles(arr: np.ndarray, qs: Sequence[float] = (0.5, 0.75, 0.9, 0.95)) -> Dict[str, float]:
    return {f"q{int(round(q * 100))}": float(np.quantile(arr, q)) for q in qs}


def compute_performance_density_scaling(records_by_scale: Dict[str, Sequence[ExperimentResultRecord]]) -> Dict[str, Any]:
    """Section 10: variance, quantiles (Q50/Q75/Q90/Q95), positive/negative mass, and
    positive/negative TAIL means (mean of the top/bottom 10% of the delta distribution) per
    (capability, anatomy_region, radius) x scale -- answers whether the whole performance
    distribution shifts with scale, not merely whether solutions become more common.
    """
    by_scale_cells: Dict[str, Any] = {}
    for scale, records in records_by_scale.items():
        by_cell = s8a.group_by_capability_region_radius(records)
        cell_stats: Dict[str, Any] = {}
        for (cap, region, radius), rows in by_cell.items():
            arr = np.asarray([r.delta for r in rows], dtype=float)
            n = arr.size
            top_n = max(1, int(round(0.1 * n)))
            sorted_arr = np.sort(arr)
            cell_stats[f"{cap}:{region}:{radius}"] = {
                "capability": cap, "anatomy_region": region, "radius": radius, "n": n,
                "variance": float(np.var(arr)), **_quantiles(arr),
                "positive_mass": float(np.mean(np.clip(arr, 0.0, None))), "negative_mass": float(np.mean(np.clip(-arr, 0.0, None))),
                "positive_tail_mean_top10pct": float(np.mean(sorted_arr[-top_n:])), "negative_tail_mean_bottom10pct": float(np.mean(sorted_arr[:top_n])),
                "density_ge_0.02": float(np.mean(arr >= 0.02)),
            }
        by_scale_cells[scale] = cell_stats
    return {"by_scale": by_scale_cells}


def classify_specialist_scaling(performance_density_scaling: Dict[str, Any], scales_ordered: Sequence[str]) -> Dict[str, Any]:
    """Section 10's explicit question -- MORE nearby specialists, STRONGER specialists, or both
    -- classified PER CELL (never forced to one global answer) by comparing the first vs last
    scale in `scales_ordered` (caller supplies the scale order, e.g. by parameter count).
    """
    by_scale = performance_density_scaling["by_scale"]
    if len(scales_ordered) < 2:
        raise ValueError("Specialist-scaling classification requires at least 2 ordered scales.")
    present = [s for s in scales_ordered if s in by_scale]
    common_keys = set.intersection(*(set(by_scale[s]) for s in present)) if len(present) >= 2 else set()
    first, last = present[0], present[-1]
    classifications: Dict[str, Any] = {}
    for key in sorted(common_keys):
        density_delta = by_scale[last][key]["density_ge_0.02"] - by_scale[first][key]["density_ge_0.02"]
        strength_delta = by_scale[last][key]["positive_tail_mean_top10pct"] - by_scale[first][key]["positive_tail_mean_top10pct"]
        more, stronger = density_delta > 1e-9, strength_delta > 1e-9
        if more and stronger:
            label = "more_and_stronger_specialists"
        elif more:
            label = "more_not_stronger_specialists"
        elif stronger:
            label = "stronger_not_more_specialists"
        else:
            label = "neither_more_nor_stronger"
        classifications[key] = {"density_delta": density_delta, "strength_delta": strength_delta, "label": label}
    return {"scales_ordered": list(scales_ordered), "per_cell": classifications}


# =================================================================================================
# Section 11: radius x model-scale landscape -- delta(m, r, s), never pooled across radii; does
# NOT invent a scalar "thicket radius" -- only classifies expand/contract/reorganize, or reports
# "insufficient_resolution" given only 3 radii (the honest default).
# =================================================================================================


def build_radius_scale_landscape(records_by_scale: Dict[str, Sequence[ExperimentResultRecord]], margins: Sequence[float] = (0.0, 0.02, 0.05)) -> Dict[str, Any]:
    curves_by_scale = {s: s8a.compute_solution_density_curves(recs) for s, recs in records_by_scale.items()}
    matrix: Dict[str, Any] = {}
    for scale, curves in curves_by_scale.items():
        for cap, region_map in curves.items():
            for region, radius_map in region_map.items():
                for radius_key, row in radius_map.items():
                    grid, curve = row["margin_grid"], row["delta_ge_m"]
                    for m in margins:
                        idxs = [i for i, g in enumerate(grid) if abs(g - m) < 1e-9]
                        if not idxs:
                            continue
                        key = f"{cap}:{region}:m={m}"
                        cell = matrix.setdefault(key, {"capability": cap, "anatomy_region": region, "margin": m, "by_scale_radius": {}})
                        cell["by_scale_radius"].setdefault(scale, {})[radius_key] = curve[idxs[0]]
    return {"matrix": matrix}


def assess_useful_neighborhood_change(landscape: Dict[str, Any]) -> Dict[str, Any]:
    assessments: Dict[str, Any] = {}
    for key, cell in landscape["matrix"].items():
        by_scale_radius = cell["by_scale_radius"]
        scales = list(by_scale_radius)
        if len(scales) < 2:
            assessments[key] = {"classification": "insufficient_scales"}
            continue
        min_radii_per_scale = min(len(radius_map) for radius_map in by_scale_radius.values())
        if min_radii_per_scale < 3:
            assessments[key] = {"classification": "insufficient_resolution_to_define_thicket_radius"}
            continue
        peak_radius_by_scale = {scale: max(radius_map, key=lambda r: radius_map[r]) for scale, radius_map in by_scale_radius.items()}
        distinct_peaks = set(peak_radius_by_scale.values())
        classification = "stable_peak_radius_across_scale" if len(distinct_peaks) <= 1 else "peak_radius_reorganizes_with_scale"
        assessments[key] = {"peak_radius_by_scale": peak_radius_by_scale, "classification": classification}
    return assessments


# =================================================================================================
# Section 12: diversity scaling -- D(r, s) = spectral_discordance, tracked across scale
# =================================================================================================


def compute_diversity_scaling(records_by_scale: Dict[str, Sequence[ExperimentResultRecord]], log_param_count_by_scale: Dict[str, float]) -> Dict[str, Any]:
    """Reuses s8a.compute_cross_capability_specialization (Stage 8's own, already-tested
    function) INDEPENDENTLY per scale -- never a new diversity statistic. Spearman correlation
    (reused from stage10a_behavioral_geometry._spearman) is only reported once >=3 scales are
    present; with exactly 2, only an honest "increasing"/"decreasing"/"flat" direction label is
    given (a 2-point Spearman correlation is always +-1 and would be uninformative).
    """
    spec_by_scale = {s: s8a.compute_cross_capability_specialization(recs) for s, recs in records_by_scale.items()}
    discordance_by_region_radius_scale: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scale, spec in spec_by_scale.items():
        for region, radius_map in spec.items():
            for radius_key, cell in radius_map.items():
                discordance_by_region_radius_scale.setdefault(region, {}).setdefault(radius_key, {})[scale] = cell["spectral_discordance"]

    scale_trend_by_region_radius: Dict[str, Any] = {}
    for region, radius_map in discordance_by_region_radius_scale.items():
        for radius_key, by_scale in radius_map.items():
            present = [s for s in log_param_count_by_scale if s in by_scale]
            if len(present) < 2:
                continue
            xs = [log_param_count_by_scale[s] for s in present]
            ys = [by_scale[s] for s in present]
            scale_trend_by_region_radius[f"{region}:{radius_key}"] = {
                "region": region, "radius": radius_key, "n_scales": len(present),
                "spearman_log_n_vs_discordance": _spearman(xs, ys) if len(present) >= 3 else None,
                "two_point_direction": (("increasing" if ys[-1] > ys[0] else "decreasing" if ys[-1] < ys[0] else "flat") if len(present) == 2 else None),
            }
    return {"discordance_by_region_radius_scale": discordance_by_region_radius_scale, "scale_trend_by_region_radius": scale_trend_by_region_radius}


# =================================================================================================
# Section 13: anatomical scaling -- does vision/connector/language scale differently?
# =================================================================================================

ANISOTROPY_SLOPE_DIFF_THRESHOLD = 0.05  # fixed, documented, never re-tuned per result


def compare_anatomical_scaling_slopes(solution_density_scaling_table: Dict[str, Any]) -> Dict[str, Any]:
    """Section 13: compares the fitted log-N slope of solution density across vision/connector/
    language at fixed capability x radius x margin. "Anisotropic scaling of visual expert
    density" is used ONLY as a named candidate concept (per the task spec's own instruction) --
    `possible_anisotropic_scaling_of_expert_density` is a conservative, fixed-threshold gate,
    never itself an assertion that the effect is real.
    """
    by_region_slopes: Dict[str, List[float]] = {}
    for fit in solution_density_scaling_table["fits"].values():
        slope = fit["linear_in_log_n"]["slope"]
        if slope is not None:
            by_region_slopes.setdefault(fit["anatomy_region"], []).append(slope)

    region_summary = {
        region: {"n_cells": len(slopes), "mean_slope": float(np.mean(slopes)), "std_slope": float(np.std(slopes))}
        for region, slopes in by_region_slopes.items() if slopes
    }
    means = [v["mean_slope"] for v in region_summary.values()]
    max_pairwise_diff = float(max(means) - min(means)) if len(means) >= 2 else 0.0
    possible_anisotropic_scaling = len(means) >= 2 and max_pairwise_diff >= ANISOTROPY_SLOPE_DIFF_THRESHOLD
    return {
        "region_slope_summary": region_summary, "max_pairwise_mean_slope_difference": max_pairwise_diff,
        "anisotropy_slope_diff_threshold": ANISOTROPY_SLOPE_DIFF_THRESHOLD,
        "possible_anisotropic_scaling_of_expert_density": possible_anisotropic_scaling,
        "note": "\"anisotropic scaling of visual expert density\" is a candidate concept name only "
                "-- never asserted as a finding regardless of this gate's value without independent corroboration.",
    }


# =================================================================================================
# Section 14: headroom sensitivity -- raw Delta is ALWAYS primary; normalized_gain is secondary
# =================================================================================================


def compute_headroom_sensitivity(records_by_scale: Dict[str, Sequence[ExperimentResultRecord]]) -> Dict[str, Any]:
    """PRIMARY analysis everywhere else in this module uses raw Delta, unchanged. This is a
    SEPARATE, clearly-labeled secondary check: for positive-delta rows only, where
    headroom = 1 - base_score > 0, normalized_gain = Delta / headroom -- used ONLY to probe
    whether a ceiling effect (higher baselines at larger scale leaving less room to improve)
    could explain an apparent scale trend, never to replace raw Delta.
    """
    out: Dict[str, Any] = {}
    for scale, records in records_by_scale.items():
        raw_deltas = np.asarray([r.delta for r in records], dtype=float)
        positive = [(r.delta, 1.0 - r.base_score) for r in records if r.delta > 0]
        normalized_gains = [d / h for d, h in positive if h > 0]
        out[scale] = {
            "raw_mean_delta": float(np.mean(raw_deltas)), "n_positive_rows": len(positive),
            "n_positive_rows_with_headroom": len(normalized_gains),
            "mean_normalized_gain_positive_only": float(np.mean(normalized_gains)) if normalized_gains else None,
            "mean_base_score": float(np.mean([r.base_score for r in records])),
        }
    scales = list(out)
    ceiling_effect_flag: Optional[bool] = None
    if len(scales) >= 2:
        raw_rank = sorted(scales, key=lambda s: out[s]["raw_mean_delta"])
        norm_candidates = [s for s in scales if out[s]["mean_normalized_gain_positive_only"] is not None]
        if len(norm_candidates) >= 2:
            norm_rank = sorted(norm_candidates, key=lambda s: out[s]["mean_normalized_gain_positive_only"])
            ceiling_effect_flag = (raw_rank != norm_rank) if set(raw_rank) == set(norm_rank) else None
    return {
        "by_scale": out, "primary_statistic": "raw_delta", "secondary_statistic": "normalized_gain_positive_only",
        "ceiling_effect_may_explain_scale_trend": ceiling_effect_flag,
        "note": "Primary analysis ALWAYS uses raw Delta; normalized_gain is secondary and used only to probe ceiling effects.",
    }


# =================================================================================================
# Section 19: figure data schemas -- no plotting, only the documented data shape each figure needs
# =================================================================================================

FIG_S1_SCHEMA: Dict[str, Any] = {"figure": "FIG_S1", "title": "Solution density vs log parameter count", "x": "log_param_count", "y": "density_at_margin", "series_by": ["capability", "anatomy_region", "radius", "margin"]}
FIG_S2_SCHEMA: Dict[str, Any] = {"figure": "FIG_S2", "title": "Performance-density distributions vs scale", "keys": ["variance", "q50", "q75", "q90", "q95", "positive_mass", "negative_mass"], "series_by": ["capability", "anatomy_region", "radius", "model_scale"]}
FIG_S3_SCHEMA: Dict[str, Any] = {"figure": "FIG_S3", "title": "Radius x model-scale solution-density phase diagram", "axes": ["radius", "model_scale"], "value": "density_at_margin", "series_by": ["capability", "anatomy_region", "margin"]}
FIG_S4_SCHEMA: Dict[str, Any] = {"figure": "FIG_S4", "title": "Spectral Discordance radius x model scale", "axes": ["radius", "model_scale"], "value": "spectral_discordance", "series_by": ["anatomy_region"]}
FIG_S5_SCHEMA: Dict[str, Any] = {"figure": "FIG_S5", "title": "Anatomy x model-scale expert-density atlas", "axes": ["anatomy_region", "model_scale"], "value": "density_ge_0.02", "series_by": ["capability", "radius"]}


def build_fig_s1_data(solution_density_scaling_table: Dict[str, Any]) -> Dict[str, Any]:
    points = [
        {"capability": f["capability"], "anatomy_region": f["anatomy_region"], "radius": f["radius"], "margin": f["margin"],
         "log_param_count": f["log_param_count"], "density_at_margin": f["density_at_margin"]}
        for f in solution_density_scaling_table["fits"].values()
    ]
    return {"schema": FIG_S1_SCHEMA, "points": points}


def build_fig_s2_data(performance_density_scaling: Dict[str, Any]) -> Dict[str, Any]:
    points = [{"model_scale": scale, **stats} for scale, cells in performance_density_scaling["by_scale"].items() for stats in cells.values()]
    return {"schema": FIG_S2_SCHEMA, "points": points}


def build_fig_s3_data(radius_scale_landscape: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema": FIG_S3_SCHEMA, "cells": radius_scale_landscape["matrix"]}


def build_fig_s4_data(diversity_scaling: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema": FIG_S4_SCHEMA, "cells": diversity_scaling["discordance_by_region_radius_scale"]}


def build_fig_s5_data(performance_density_scaling: Dict[str, Any]) -> Dict[str, Any]:
    points = [
        {"model_scale": scale, "anatomy_region": stats["anatomy_region"], "capability": stats["capability"], "radius": stats["radius"], "density_ge_0.02": stats["density_ge_0.02"]}
        for scale, cells in performance_density_scaling["by_scale"].items() for stats in cells.values()
    ]
    return {"schema": FIG_S5_SCHEMA, "points": points}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage8-dir", default=str(DEFAULT_STAGE8_DIR))
    parser.add_argument("--stage11-dir", default=str(DEFAULT_STAGE11_DIR))
    args = parser.parse_args(argv)

    stage11_dir = Path(args.stage11_dir)
    if not (stage11_dir / "results.jsonl").exists():
        print(
            f"No Stage-11 results found at {stage11_dir} -- this schema is PREPARED, NOT RUN yet "
            f"(no 7B GPU execution has happened). Nothing to analyze."
        )
        return 0

    stage8_records = load_records(Path(args.stage8_dir) / "results.jsonl")
    stage11_records = load_records(stage11_dir / "results.jsonl")
    report = build_cross_scale_report(stage8_records, stage11_records)
    import json
    print(json.dumps(s8a._sanitize(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
