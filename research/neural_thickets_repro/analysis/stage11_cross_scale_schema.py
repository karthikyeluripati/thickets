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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ANALYSIS_ROOT = Path(__file__).resolve().parent
for p in (SRC_ROOT, ANALYSIS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neural_thickets_repro.run_global_visual_thicket_pilot import load_records  # noqa: E402
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

import stage8_coarse_anatomical_atlas_analysis as s8a  # noqa: E402

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
