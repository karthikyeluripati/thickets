"""Stage 11 Track S1: WHOLE-MODEL Neural-Thickets-style scaling. A single L1 region --
WHOLE_MODEL_REGION_LABEL ("whole_model", scaling_common's name for the anatomy atlas's own
already-built, already-validated Level-0 "full_model" region, i.e. the union of every trainable
parameter) -- x 3 frozen common relative-L2 radii x 64 direction families x 6 frozen capabilities,
at any RUNNABLE scale (currently 3B, 7B; see scaling_common.RUNNABLE_SCALES).

This is the PRIMARY Neural-Thickets-style scaling experiment (task spec Section 2): it estimates
p(Delta | capability, radius, model_scale) and delta(m | capability, radius, model_scale) WITHOUT
any anatomical resolution -- Track S2 (run_stage11_coarse_anatomical_atlas_7b.py's existing
vision/connector/language design, generalized via the same scaling_common primitives in the
unified dispatcher run_stage11_visual_thicket_scaling.py) is the separate, additional,
anatomically-resolved track.

WHY THE HISTORICAL 3B "GLOBAL" RUN IS NOT REUSED HERE: see
scaling_common.WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE -- that run's global_gaussian_upstream
perturbations skip visual.* parameters, so it is NOT a valid whole-model anchor. A genuine 3B
whole-model backfill (this module, invoked with --scale 3B) is therefore required (task spec
Section 16), using the exact same anatomical_relative_l2 semantics as every other Stage-8/9/11
run, so vision IS included in both theta_0's L2 norm and epsilon's support.

FROZEN FULL CONFIG (per scale)
=================================================================================================
model:        resolved via ScalingModelSpec + resolve_immutable_model_revision (never invented)
region:       ("whole_model",) -- 1 region, covering 100% of the model's trainable parameters
radii:        STAGE8_RADII, by identity -- NOT recalibrated for any scale (deliberate; a shift in
              the useful/destructive regime at the SAME fractional displacement is itself a result)
capabilities: STAGE8_CAPABILITIES, by identity
directions:   STAGE8_N_DIRECTIONS_PER_CELL (64) per radius, independent per-scale seed namespace
              (scaling_common.build_scaling_direction_seed_bank -- includes scale_label), same
              direction reused across the 3 radii
data:         the SAME D_map N=50 example manifests as Stage 8 (STAGE8_BASE_SEED reused, exact
              subset-hash-equality gate against STAGE8_AUTHORITATIVE_SUBSET_HASHES)
Total:        1 region x 3 radii x 64 directions = 192 unique perturbations
              192 x 6 capabilities = 1152 perturbation x capability result rows
              1152 x 50 = 57,600 perturbed model-example evaluations

SMOKE MODE: 1 region x 3 radii x 1 direction family x 6 capabilities x D_map N=5 = 3 unique
perturbations, 18 result rows, 90 perturbed model-example evaluations.

Reuses, by import: everything run_stage11_coarse_anatomical_atlas_7b.py already reuses from
Stage 8/9 (engine config, v3 quantization-aware solver, baseline repeatability preflight,
Stage-9's bounded-memory candidate lifecycle) -- see that module's own docstring for the full
derivation of each. This module additionally reuses scaling_common.py for every
scale/track-generic piece (model-spec registry, anatomy audit, direction-seed-bank, region-name
reporting) rather than re-deriving any of it a second time.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from .run_global_visual_thicket_pilot import (
    RESTORATION_MODE,
    CapabilityContext,
    RayEngineLLMAdapter,
    RestorationFailedError,
    append_candidate_rows,
    detect_vllm_engine_mode,
    format_base_snapshot_confirmation,
    format_runtime_compatibility_diagnostic,
    get_vllm_version,
    launch_stage6_engine,
    load_completed_perturbation_rows,
    load_records,
    reset_to_base_weights_via_rpc,
    store_base_weights_via_rpc,
    verify_exact_fixed_base_restoration_via_rpc,
)
from .run_stage7b_anatomical_calibration import (
    ENABLE_PREFIX_CACHING,
    MULTIMODAL_CACHE_POLICY,
    PERTURBATION_MODE,
    REALIZED_RADIUS_TOLERANCE,
    RADIUS_REALIZATION_METHOD,
    build_stage7b_engine_config,
    ensure_encoder_cache_reset_available,
    ensure_stage7b_encoder_cache_reset_mechanism_exposed,
)
from .run_stage8_coarse_anatomical_atlas import (
    STAGE8_BASE_SEED,
    BaselineNondeterminismError,
    DatasetRoleViolationError,
    build_d_map_capability_contexts,
    ensure_baseline_repeatability,
    run_baseline_repeatability_preflight_rpc,
)
from .run_stage9_hierarchical_anatomical_atlas import (
    STAGE8_AUTHORITATIVE_SUBSET_HASHES,
    _is_ray_unrecoverable_error,
    _write_candidate_telemetry_line,
)
from .scaling_common import (
    MODEL_FAMILY,
    STAGE8_CAPABILITIES,
    STAGE8_D_MAP_N,
    STAGE8_GENERATE_BATCH_SIZE,
    STAGE8_N_DIRECTIONS_PER_CELL,
    STAGE8_RADII,
    STAGE8_SMOKE_D_MAP_N,
    STAGE8_SMOKE_N_DIRECTIONS,
    WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE,
    WHOLE_MODEL_REGION_LABEL,
    ModelRevisionResolutionError,
    ScalingModelSpec,
    build_scaling_direction_seed_bank,
    compute_anatomy_audit_hash,
    compute_direction_seed_bank_hash,
    ensure_scale_runnable,
    ensure_scaling_anatomy_audit_passes,
    ensure_whole_model_covers_100_percent,
    get_scaling_model_spec,
    report_region_param_names_for_scaling,
    report_scaling_anatomy_audit,
    resolve_immutable_model_revision,
)
from .scoped_anatomical_perturbation import (
    QUANTIZATION_AWARE_METHOD_V3,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
)
from .stage11_32b_readiness import V3_SOLVER_DISTRIBUTED_EXTENSION_NOTE, run_32b_readiness_preflight_and_report
from .thicket.perturbation import PERTURBATION_MODES, PerturbationManifest
from .thicket.schema import ExperimentResultRecord
from .vlm_adapter import reset_vllm_encoder_cache_full

assert PERTURBATION_MODE in PERTURBATION_MODES
assert RADIUS_REALIZATION_METHOD == QUANTIZATION_AWARE_METHOD_V3

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

EXPERIMENT_ID = "stage11_whole_model_scaling"
TRACK = "whole_model"
WHOLE_MODEL_REGIONS: Tuple[str, ...] = (WHOLE_MODEL_REGION_LABEL,)
DATASET_ROLE = "map"
_ALLOWED_D_MAP_SIZES: Tuple[int, ...] = (STAGE8_D_MAP_N, STAGE8_SMOKE_D_MAP_N)


def _base_seed_for_scale(scale_label: str) -> int:
    """A distinct STAGE11_BASE_SEED-shaped constant per scale, deterministically derived from a
    single frozen root rather than hand-typed once per scale (avoids ever accidentally reusing
    the exact same integer for two different scales by copy-paste).
    """
    from .thicket.seeds import derive_seed

    return derive_seed(20260904, "stage11_whole_model_base_seed", scale_label) % (2 ** 31)


def compute_whole_model_run_signature(scale_label: str, radii: Sequence[float], n_directions: int, d_map_n: int, generation_batch_size: int) -> str:
    is_frozen_full = (tuple(radii) == STAGE8_RADII and n_directions == STAGE8_N_DIRECTIONS_PER_CELL and d_map_n == STAGE8_D_MAP_N)
    scale_lower = scale_label.lower()
    if is_frozen_full:
        if generation_batch_size == STAGE8_GENERATE_BATCH_SIZE:
            return f"stage11_{scale_lower}_whole_model_v1"
        return f"stage11_{scale_lower}_whole_model_batched{generation_batch_size}"
    radius_label = "-".join(f"{r:.6f}".replace(".", "") for r in radii)
    return f"stage11_smoke_{scale_lower}_whole_model_r{radius_label}_n{d_map_n}_dir{n_directions}"


@dataclass(frozen=True)
class WholeModelPlan:
    scale_label: str
    model_name: str
    model_revision: str
    model_family: str
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_directions_per_cell: int
    d_map_n: int
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: bool
    run_signature: str
    output_dir: Path
    generation_batch_size: int = STAGE8_GENERATE_BATCH_SIZE

    @property
    def region_labels(self) -> Tuple[str, ...]:
        return WHOLE_MODEL_REGIONS

    @property
    def total_unique_perturbations(self) -> int:
        return len(self.region_labels) * len(self.radii) * self.n_directions_per_cell

    @property
    def total_perturbation_capability_evaluations(self) -> int:
        return self.total_unique_perturbations * len(self.capabilities)

    @property
    def total_perturbed_model_example_evaluations(self) -> int:
        return self.total_perturbation_capability_evaluations * self.d_map_n

    @property
    def is_smoke(self) -> bool:
        return not (self.radii == STAGE8_RADII and self.n_directions_per_cell == STAGE8_N_DIRECTIONS_PER_CELL and self.d_map_n == STAGE8_D_MAP_N)


def build_whole_model_plan(
    *, spec: ScalingModelSpec, model_revision: str, output_root: "str | Path",
    radii: Sequence[float] = STAGE8_RADII, n_directions_per_cell: int = STAGE8_N_DIRECTIONS_PER_CELL,
    d_map_n: int = STAGE8_D_MAP_N, generation_batch_size: int = STAGE8_GENERATE_BATCH_SIZE,
) -> WholeModelPlan:
    if not radii:
        raise ValueError("Whole-model track requires at least one common radius.")
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Whole-model D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    if generation_batch_size <= 0:
        raise ValueError(f"generation_batch_size must be positive, got {generation_batch_size}")

    run_signature = compute_whole_model_run_signature(spec.scale_label, radii, n_directions_per_cell, d_map_n, generation_batch_size)
    return WholeModelPlan(
        scale_label=spec.scale_label, model_name=spec.model_name, model_revision=model_revision, model_family=spec.model_family,
        radii=tuple(radii), capabilities=STAGE8_CAPABILITIES, n_directions_per_cell=n_directions_per_cell, d_map_n=d_map_n,
        radius_realization_method=RADIUS_REALIZATION_METHOD, multimodal_cache_policy=MULTIMODAL_CACHE_POLICY,
        enable_prefix_caching=ENABLE_PREFIX_CACHING, run_signature=run_signature, output_dir=Path(output_root) / run_signature,
        generation_batch_size=generation_batch_size,
    )


def build_whole_model_smoke_plan(*, spec: ScalingModelSpec, model_revision: str, output_root: "str | Path") -> WholeModelPlan:
    """1 region x 3 radii x 1 direction family x 6 capabilities x D_map N=5 = 3 perturbations,
    18 rows, 90 perturbed model-example evaluations.
    """
    return build_whole_model_plan(
        spec=spec, model_revision=model_revision, output_root=output_root,
        radii=STAGE8_RADII, n_directions_per_cell=STAGE8_SMOKE_N_DIRECTIONS, d_map_n=STAGE8_SMOKE_D_MAP_N,
    )


@dataclass(frozen=True)
class WholeModelDirectionAssignment:
    manifest: PerturbationManifest
    direction_index: int
    direction_seed: int

    @property
    def direction_family_id(self) -> str:
        return f"{WHOLE_MODEL_REGION_LABEL}:{self.direction_index}"


def build_whole_model_population(
    plan: WholeModelPlan, seed_bank: Dict[str, Tuple[int, ...]], mask_hash: str,
) -> Dict[float, Tuple[WholeModelDirectionAssignment, ...]]:
    seeds = seed_bank.get(WHOLE_MODEL_REGION_LABEL)
    if seeds is None or len(seeds) != plan.n_directions_per_cell:
        raise ValueError(f"Direction seed bank for {WHOLE_MODEL_REGION_LABEL!r} must have exactly {plan.n_directions_per_cell} seeds.")

    population_by_radius: Dict[float, Tuple[WholeModelDirectionAssignment, ...]] = {}
    for radius in plan.radii:
        assignments = tuple(
            WholeModelDirectionAssignment(
                manifest=PerturbationManifest(
                    seed=seed, perturbation_mode=PERTURBATION_MODE, anatomy_region=WHOLE_MODEL_REGION_LABEL, radius=radius, sigma=None,
                    model_family=plan.model_family, model_scale=plan.scale_label, model_revision=plan.model_revision,
                    parameter_mask_hash=mask_hash,
                ),
                direction_index=i, direction_seed=seed,
            )
            for i, seed in enumerate(seeds)
        )
        population_by_radius[radius] = assignments
    return population_by_radius


class WholeModelDirectionSeedReuseViolationError(RuntimeError):
    pass


def validate_whole_model_direction_seed_reuse(plan: WholeModelPlan, population_by_radius: Dict[float, Tuple[WholeModelDirectionAssignment, ...]]) -> None:
    all_ids = [a.manifest.perturbation_id for assignments in population_by_radius.values() for a in assignments]
    if len(all_ids) != len(set(all_ids)):
        raise WholeModelDirectionSeedReuseViolationError(f"Duplicate perturbation_id(s) ({len(all_ids)} total, {len(set(all_ids))} unique).")
    expected_total = len(plan.radii) * plan.n_directions_per_cell
    if len(all_ids) != expected_total:
        raise WholeModelDirectionSeedReuseViolationError(f"Whole-model population has {len(all_ids)} perturbations, expected {expected_total}.")

    seed_counts: Dict[int, int] = {}
    for assignments in population_by_radius.values():
        for a in assignments:
            seed_counts[a.direction_seed] = seed_counts.get(a.direction_seed, 0) + 1
    if len(seed_counts) != plan.n_directions_per_cell:
        raise WholeModelDirectionSeedReuseViolationError(f"{len(seed_counts)} distinct direction seeds, expected {plan.n_directions_per_cell}.")
    wrong = {seed: n for seed, n in seed_counts.items() if n != len(plan.radii)}
    if wrong:
        raise WholeModelDirectionSeedReuseViolationError(f"Seed(s) not reused exactly {len(plan.radii)}x (once per radius): {wrong}")


class Stage11SubsetHashMismatchError(RuntimeError):
    pass


def run_subset_hash_check(capability_contexts: Dict[str, CapabilityContext]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for cap, ctx in capability_contexts.items():
        expected = STAGE8_AUTHORITATIVE_SUBSET_HASHES.get(cap)
        report[cap] = {"expected_stage8_subset_hash": expected, "live_subset_hash": ctx.subset_hash, "matches": ctx.subset_hash == expected}
    report["all_match"] = all(v["matches"] for v in report.values())
    return report


def ensure_subset_hashes_match_stage8(report: Dict[str, Any]) -> None:
    """FULL MODE ONLY hard stop -- UNCHANGED, kept strict. Comparing an N=5 smoke subset hash
    against these N=50 authoritative hashes is architecturally meaningless (they are DIFFERENT
    sample sizes by construction) -- see SUBSET_GATE_MODE_SMOKE below for the mode-appropriate
    smoke check instead. Never call this function against a smoke (N=5) subset.
    """
    if not report.get("all_match"):
        mismatches = {cap: v for cap, v in report.items() if cap != "all_match" and not v["matches"]}
        raise Stage11SubsetHashMismatchError(f"Whole-model D_map subset hashes do not match Stage-8's authoritative 3B manifests: {mismatches}")


# =================================================================================================
# Subset gate -- MODE-AWARE (this repair pass: ports Stage 9's own validated smoke/full
# baseline-gate fix to Stage 11's whole-model subset-hash gate).
#
# ROOT CAUSE (live Stage-11 3B whole_model --smoke run): main() unconditionally ran
# run_subset_hash_check/ensure_subset_hashes_match_stage8 -- the FULL-mode check that compares the
# live D_map subset hash against STAGE8_AUTHORITATIVE_SUBSET_HASHES -- regardless of --smoke. The
# authoritative hashes are keyed to Stage 8's own frozen D_map N=50 manifests; a smoke run's D_map
# is N=5 BY DESIGN (STAGE8_SMOKE_D_MAP_N). An N=5 subset can never equal an N=50 subset hash, so
# every one of the 6 capabilities mismatched -- a spurious, guaranteed failure caused by applying
# the wrong gate to the wrong sample size, not a real subset-construction or dataset problem. This
# is the exact bug class Stage 9's own run_stage9_hierarchical_anatomical_atlas.py already hit and
# fixed (see that module's "Baseline gate -- MODE-AWARE" section) -- ported here rather than
# reinvented.
#
# FIX: the subset gate is now explicitly mode-dispatched (build_subset_gate_report /
# ensure_subset_gate_passes), never silently skipped in either mode:
#   - FULL (D_map N=50): UNCHANGED exact-equality check against STAGE8_AUTHORITATIVE_SUBSET_
#     HASHES via run_subset_hash_check/ensure_subset_hashes_match_stage8 -- still a HARD STOP on
#     any mismatch, still comparing every one of the 6 capabilities.
#   - SMOKE (D_map N=5): NEVER compared to the N=50 hashes at all -- instead, the D_map is built
#     TWICE, independently (a second, distinct subset_ids_dir so no on-disk cache can substitute
#     for genuine re-derivation from the same STAGE8_BASE_SEED), and the two passes' subset hashes
#     (and example counts) must be IDENTICAL -- proving the smoke subset's own construction is
#     deterministic, which is the property that actually matters at N=5.
# =================================================================================================

SUBSET_GATE_MODE_FULL = "stage8_full_n50_exact_equality"
SUBSET_GATE_MODE_SMOKE = "smoke_n5_deterministic_repeatability"


class Stage11SmokeSubsetNondeterminismError(RuntimeError):
    """SMOKE MODE ONLY: the D_map N=5 subset was NOT reconstructed identically across two
    independent builds (or some capability did not have exactly the expected N=5 examples) --
    hard stop. Never compared to the N=50 Stage-8 authoritative hashes; that comparison is
    architecturally meaningless at N=5 (different sample size by construction).
    """


def run_smoke_subset_determinism_check(
    pass_a: Dict[str, CapabilityContext], pass_b: Dict[str, CapabilityContext], d_map_n: int,
) -> Dict[str, Any]:
    """SMOKE MODE ONLY: `pass_a` and `pass_b` must be built from two INDEPENDENT
    build_d_map_capability_contexts() calls (distinct subset_ids_dir, same STAGE8_BASE_SEED) so a
    persisted-ids-file cache can never substitute for genuine re-derivation. Reports, per
    capability: both passes' subset hashes, whether they match, and whether both passes actually
    have exactly `d_map_n` examples -- never a comparison to Stage-8's N=50 authoritative hashes.
    """
    report: Dict[str, Any] = {}
    for capability, ctx_a in pass_a.items():
        ctx_b = pass_b.get(capability)
        n_a, n_b = len(ctx_a.examples), (len(ctx_b.examples) if ctx_b is not None else None)
        matches = ctx_b is not None and ctx_a.subset_hash == ctx_b.subset_hash
        n_matches_expected = n_a == d_map_n and n_b == d_map_n
        report[capability] = {
            "pass_a_subset_hash": ctx_a.subset_hash, "pass_b_subset_hash": ctx_b.subset_hash if ctx_b is not None else None,
            "matches": matches, "n_examples_pass_a": n_a, "n_examples_pass_b": n_b, "n_examples_expected": d_map_n,
            "n_matches_expected": n_matches_expected,
        }
    per_capability = list(report.values())
    report["all_deterministic"] = all(v["matches"] for v in per_capability)
    report["all_n_matches_expected"] = all(v["n_matches_expected"] for v in per_capability)
    return report


def ensure_smoke_subset_determinism(report: Dict[str, Any]) -> None:
    if not report.get("all_deterministic") or not report.get("all_n_matches_expected"):
        bad = {c: v for c, v in report.items() if isinstance(v, dict) and (not v["matches"] or not v["n_matches_expected"])}
        raise Stage11SmokeSubsetNondeterminismError(
            f"Stage-11 SMOKE D_map subset construction is not deterministic and/or does not have "
            f"the expected N for {len(bad)} capability(ies): {bad}. Smoke mode NEVER compares "
            f"subset hashes against the N=50 Stage-8 authoritative manifests."
        )


def build_subset_gate_report(
    *, is_smoke: bool, d_map_n: int,
    full_subset_hash_report: Optional[Dict[str, Any]] = None, smoke_determinism_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mode-explicit dispatch -- persists `subset_gate_mode` (plus `d_map_n` and the per-capability
    subset hashes, nested under either `subset_hash_equality` or `smoke_determinism`) so a reader
    of subset_gate.json always knows which gate ran and at which D_map size, never has to infer it.
    """
    if is_smoke:
        if smoke_determinism_report is None:
            raise ValueError("smoke_determinism_report is required when is_smoke=True")
        return {"subset_gate_mode": SUBSET_GATE_MODE_SMOKE, "d_map_n": d_map_n, "smoke_determinism": smoke_determinism_report}
    if full_subset_hash_report is None:
        raise ValueError("full_subset_hash_report is required when is_smoke=False")
    return {"subset_gate_mode": SUBSET_GATE_MODE_FULL, "d_map_n": d_map_n, "subset_hash_equality": full_subset_hash_report}


def ensure_subset_gate_passes(gate_report: Dict[str, Any]) -> None:
    """Single dispatch point -- never silently skips validation: an unrecognized subset_gate_mode
    is itself a hard failure, not a silent no-op.
    """
    mode = gate_report.get("subset_gate_mode")
    if mode == SUBSET_GATE_MODE_SMOKE:
        ensure_smoke_subset_determinism(gate_report["smoke_determinism"])
    elif mode == SUBSET_GATE_MODE_FULL:
        ensure_subset_hashes_match_stage8(gate_report["subset_hash_equality"])
    else:
        raise ValueError(f"Unknown subset_gate_mode {mode!r} -- refusing to silently skip subset validation.")


class IncompatibleWholeModelCheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class WholeModelCheckpointManifest:
    experiment_id: str
    run_signature: str
    scale_label: str
    track: str
    restoration_mode: str
    perturbation_mode: str
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: bool
    generation_batch_size: int
    model_revision: str
    dataset_role: str
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_directions_per_cell: int
    d_map_n: int
    subset_hashes: Dict[str, str]
    whole_model_mask_hash: str
    direction_seed_bank_hash: str
    anatomy_audit_hash: str
    expected_unique_perturbations: int
    expected_result_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "run_signature": self.run_signature, "scale_label": self.scale_label, "track": self.track,
            "restoration_mode": self.restoration_mode, "perturbation_mode": self.perturbation_mode, "perturbation_semantics": self.perturbation_mode,
            "radius_realization_method": self.radius_realization_method, "multimodal_cache_policy": self.multimodal_cache_policy,
            "enable_prefix_caching": self.enable_prefix_caching, "generation_batch_size": self.generation_batch_size,
            "model_revision": self.model_revision, "dataset_role": self.dataset_role,
            "radii": list(self.radii), "capabilities": list(self.capabilities),
            "n_directions_per_cell": self.n_directions_per_cell, "d_map_n": self.d_map_n,
            "subset_hashes": dict(sorted(self.subset_hashes.items())), "whole_model_mask_hash": self.whole_model_mask_hash,
            "direction_seed_bank_hash": self.direction_seed_bank_hash, "anatomy_audit_hash": self.anatomy_audit_hash,
            "expected_unique_perturbations": self.expected_unique_perturbations, "expected_result_rows": self.expected_result_rows,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WholeModelCheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], run_signature=d["run_signature"], scale_label=d["scale_label"], track=d["track"],
            restoration_mode=d["restoration_mode"], perturbation_mode=d["perturbation_mode"], radius_realization_method=d["radius_realization_method"],
            multimodal_cache_policy=d["multimodal_cache_policy"], enable_prefix_caching=d["enable_prefix_caching"],
            generation_batch_size=d["generation_batch_size"], model_revision=d["model_revision"], dataset_role=d["dataset_role"],
            radii=tuple(d["radii"]), capabilities=tuple(d["capabilities"]), n_directions_per_cell=d["n_directions_per_cell"], d_map_n=d["d_map_n"],
            subset_hashes=dict(d["subset_hashes"]), whole_model_mask_hash=d["whole_model_mask_hash"],
            direction_seed_bank_hash=d["direction_seed_bank_hash"], anatomy_audit_hash=d["anatomy_audit_hash"],
            expected_unique_perturbations=d["expected_unique_perturbations"], expected_result_rows=d["expected_result_rows"],
        )


def build_whole_model_checkpoint_manifest(
    plan: WholeModelPlan, capability_contexts: Dict[str, CapabilityContext], mask_hash: str,
    seed_bank: Dict[str, Tuple[int, ...]], anatomy_audit: Dict[str, Any],
) -> WholeModelCheckpointManifest:
    return WholeModelCheckpointManifest(
        experiment_id=EXPERIMENT_ID, run_signature=plan.run_signature, scale_label=plan.scale_label, track=TRACK,
        restoration_mode=RESTORATION_MODE, perturbation_mode=PERTURBATION_MODE, radius_realization_method=plan.radius_realization_method,
        multimodal_cache_policy=plan.multimodal_cache_policy, enable_prefix_caching=plan.enable_prefix_caching,
        generation_batch_size=plan.generation_batch_size, model_revision=plan.model_revision, dataset_role=DATASET_ROLE,
        radii=plan.radii, capabilities=plan.capabilities, n_directions_per_cell=plan.n_directions_per_cell, d_map_n=plan.d_map_n,
        subset_hashes={c: ctx.subset_hash for c, ctx in capability_contexts.items()}, whole_model_mask_hash=mask_hash,
        direction_seed_bank_hash=compute_direction_seed_bank_hash(seed_bank), anatomy_audit_hash=compute_anatomy_audit_hash(anatomy_audit),
        expected_unique_perturbations=plan.total_unique_perturbations, expected_result_rows=plan.total_perturbation_capability_evaluations,
    )


def ensure_whole_model_checkpoint_manifest(path: Path, current: WholeModelCheckpointManifest) -> WholeModelCheckpointManifest:
    if path.exists():
        existing = WholeModelCheckpointManifest.from_dict(json.loads(path.read_text()))
        if existing != current:
            raise IncompatibleWholeModelCheckpointError(
                f"Existing checkpoint at {path} is incompatible with this run -- refusing to resume: "
                f"existing={existing.to_dict()} current={current.to_dict()}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current.to_dict(), indent=2))
    return current


def build_whole_model_run_manifest_summary(checkpoint: WholeModelCheckpointManifest, records: List[ExperimentResultRecord]) -> Dict[str, Any]:
    actual_unique_perturbations = len({r.perturbation_id for r in records})
    actual_result_rows = len(records)
    run_complete = (actual_unique_perturbations == checkpoint.expected_unique_perturbations and actual_result_rows == checkpoint.expected_result_rows)
    return {**checkpoint.to_dict(), "actual_unique_perturbations": actual_unique_perturbations, "actual_result_rows": actual_result_rows, "run_complete": run_complete}


def write_whole_model_run_manifest(output_dir: Path) -> Dict[str, Any]:
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_manifest.json found at {checkpoint_path} -- this run never started.")
    checkpoint = WholeModelCheckpointManifest.from_dict(json.loads(checkpoint_path.read_text()))
    records = load_records(output_dir / "results.jsonl")
    manifest = build_whole_model_run_manifest_summary(checkpoint, records)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list of per-worker results.")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; whole-model track is TP=1-only.")
    return results[0]


def _collective_rpc_single_worker(engine: Any, method, args: tuple = (), *, label: str, ray_get: Optional[Callable] = None) -> Any:
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def evaluate_one_whole_model_candidate_rpc(
    engine: Any, assignment: WholeModelDirectionAssignment, region_param_names: Sequence[str],
    capability_contexts: Dict[str, CapabilityContext], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable] = None, generation_batch_size: int = STAGE8_GENERATE_BATCH_SIZE,
    rss_checkpoint: Optional[Callable[[str], None]] = None,
) -> List[ExperimentResultRecord]:
    """Byte-identical lifecycle to run_stage11_coarse_anatomical_atlas_7b.
    evaluate_one_stage11_candidate_rpc, generalized to the single whole_model region (Stage 9's
    bounded-memory pattern, applied from the start).
    """
    manifest = assignment.manifest
    if manifest.perturbation_mode != PERTURBATION_MODE:
        raise ValueError(f"Whole-model track only evaluates {PERTURBATION_MODE!r} manifests, got {manifest.perturbation_mode!r}")

    from .mem_telemetry import release_transient_memory

    def _checkpoint(label: str) -> None:
        if rss_checkpoint is not None:
            rss_checkpoint(label)

    _checkpoint("before_candidate")

    llm_adapter = RayEngineLLMAdapter(engine, ray_get=ray_get)
    records: List[ExperimentResultRecord] = []
    try:
        apply_result = _collective_rpc_single_worker(
            engine, scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
            args=(manifest.seed, manifest.radius, manifest.anatomy_region, tuple(region_param_names)),
            label="scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3", ray_get=ray_get,
        )
        realized_r = apply_result["realized_relative_l2"]
        acceptance_mode = apply_result["radius_acceptance_mode"]
        if acceptance_mode == "strict":
            if abs(realized_r - manifest.radius) > REALIZED_RADIUS_TOLERANCE:
                raise RuntimeError(
                    f"Perturbation {manifest.perturbation_id!r} (whole_model, requested radius={manifest.radius}): "
                    f"strict-mode realized relative-L2 {realized_r} still differs by more than {REALIZED_RADIUS_TOLERANCE}."
                )
        elif acceptance_mode == "quantization_limited":
            if apply_result["relative_radius_error"] > QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE:
                raise RuntimeError(
                    f"Perturbation {manifest.perturbation_id!r} (whole_model, requested radius={manifest.radius}): "
                    f"quantization-limited relative error {apply_result['relative_radius_error']} still exceeds "
                    f"{QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE}."
                )
        else:
            raise RuntimeError(f"Unknown radius_acceptance_mode {acceptance_mode!r} for perturbation {manifest.perturbation_id!r}.")

        reset_vllm_encoder_cache_full(engine)
        _checkpoint("after_perturbation_applied")

        for capability, ctx in capability_contexts.items():
            if ctx.partition.manifest_hash != ctx.subset_hash:
                raise DatasetRoleViolationError(f"CapabilityContext for {capability!r} has an inconsistent subset_hash.")
            result = run_benchmark(
                ctx.benchmark, ctx.examples, llm_adapter, tokenizer, sampling_params,
                max_requests_per_generate=generation_batch_size,
            )
            perturbed_score = result.aggregate_metrics["primary_metric"]
            base_score = ctx.base_score
            per_example_hash = result.generation_hash()
            parser_failure_rate = result.aggregate_metrics.get("parser_failure_rate")
            records.append(ExperimentResultRecord(
                experiment_id=EXPERIMENT_ID, perturbation_id=manifest.perturbation_id,
                model_family=manifest.model_family, model_scale=manifest.model_scale, model_revision=manifest.model_revision,
                perturbation_mode=manifest.perturbation_mode, anatomy_region=manifest.anatomy_region,
                radius=manifest.radius, sigma=manifest.sigma, seed=manifest.seed, parameter_mask_hash=manifest.parameter_mask_hash,
                capability=capability, dataset_role=DATASET_ROLE, subset_hash=ctx.subset_hash,
                base_score=base_score, perturbed_score=perturbed_score, delta=perturbed_score - base_score,
                parser_failure_rate=parser_failure_rate,
                per_example_result_path=None, per_example_result_hash=per_example_hash,
                runtime_metadata={
                    "restoration_mode": RESTORATION_MODE, "perturbation_semantics": PERTURBATION_MODE,
                    "radius_realization_method": apply_result["radius_realization_method"],
                    "radius_acceptance_mode": apply_result["radius_acceptance_mode"],
                    "quantization_limited": apply_result["quantization_limited"],
                    "requested_relative_l2": manifest.radius, "designed_relative_l2": apply_result["designed_relative_l2"],
                    "realized_relative_l2": realized_r, "realized_abs_error": apply_result["realized_abs_error"],
                    "absolute_radius_error": apply_result["absolute_radius_error"],
                    "relative_radius_error": apply_result["relative_radius_error"],
                    "quantization_plateau": apply_result["quantization_plateau"],
                    "region_param_count": apply_result["region_param_count"],
                    "theta_region_l2_norm": apply_result["theta_l2_norm"], "epsilon_region_l2_norm": apply_result["realized_epsilon_l2_norm"],
                    "multimodal_cache_policy": MULTIMODAL_CACHE_POLICY,
                    "cache_reset_before_evaluation": True, "cache_reset_after_restoration": False,
                    "direction_family_id": assignment.direction_family_id, "direction_seed": assignment.direction_seed,
                    "direction_index": assignment.direction_index, "region": WHOLE_MODEL_REGION_LABEL,
                    "generation_batch_size": generation_batch_size, "model_scale": manifest.model_scale, "track": TRACK,
                },
            ))
            del result
            release_transient_memory()
            _checkpoint(f"after_capability_{capability}")
    except Exception as exc:
        if _is_ray_unrecoverable_error(exc):
            raise
        reset_to_base_weights_via_rpc(engine, ray_get=ray_get)
        raise

    reset_to_base_weights_via_rpc(engine, ray_get=ray_get)
    verification = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Exact fixed-base restoration failed after whole-model candidate {manifest.perturbation_id!r} "
            f"(radius={manifest.radius}, seed={manifest.seed}): max_abs_drift={verification['max_abs_drift']}"
        )

    reset_vllm_encoder_cache_full(engine)
    for record in records:
        record.runtime_metadata["cache_reset_after_restoration"] = True

    return records


def run_whole_model_rpc(
    plan: WholeModelPlan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, seed_bank: Dict[str, Tuple[int, ...]], region_param_names: Sequence[str],
    mask_hash: str, anatomy_audit: Dict[str, Any], *, run_benchmark: Callable, ray_get: Optional[Callable] = None,
) -> int:
    from .mem_telemetry import release_transient_memory, rss_mb

    population_by_radius = build_whole_model_population(plan, seed_bank, mask_hash)
    validate_whole_model_direction_seed_reuse(plan, population_by_radius)

    current_checkpoint = build_whole_model_checkpoint_manifest(plan, capability_contexts, mask_hash, seed_bank, anatomy_audit)
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    ensure_whole_model_checkpoint_manifest(checkpoint_path, current_checkpoint)

    results_path = plan.output_dir / "results.jsonl"
    telemetry_path = plan.output_dir / "candidate_memory_telemetry.jsonl"
    completed_ids = set(load_completed_perturbation_rows(results_path, plan.capabilities).keys())

    newly_completed_rows = 0
    perturbation_index = 0
    previous_candidate_rss_mb: Optional[float] = None

    for radius, assignments in population_by_radius.items():
        for assignment in assignments:
            if assignment.manifest.perturbation_id in completed_ids:
                continue

            rss_start_mb = rss_mb()
            capability_rss_mb: List[float] = []
            rss_after_perturbation_mb: Optional[float] = None

            def _rss_checkpoint(label: str, _cap_rss=capability_rss_mb) -> None:
                nonlocal rss_after_perturbation_mb
                if label == "after_perturbation_applied":
                    rss_after_perturbation_mb = rss_mb()
                elif label.startswith("after_capability_"):
                    _cap_rss.append(rss_mb())

            records = evaluate_one_whole_model_candidate_rpc(
                engine, assignment, region_param_names, capability_contexts, tokenizer, sampling_params,
                run_benchmark=run_benchmark, ray_get=ray_get, generation_batch_size=plan.generation_batch_size,
                rss_checkpoint=_rss_checkpoint,
            )
            append_candidate_rows(results_path, records)
            rss_after_checkpoint_mb = rss_mb()

            newly_completed_rows += len(records)
            del records
            release_transient_memory()
            rss_after_cleanup_mb = rss_mb()

            high_water_mb = max(
                v for v in (rss_start_mb, rss_after_perturbation_mb, *capability_rss_mb, rss_after_checkpoint_mb, rss_after_cleanup_mb)
                if v is not None
            )
            _write_candidate_telemetry_line(telemetry_path, {
                "perturbation_index": perturbation_index, "perturbation_id": assignment.manifest.perturbation_id,
                "region": WHOLE_MODEL_REGION_LABEL, "radius": radius, "direction_index": assignment.direction_index,
                "rss_start_mb": rss_start_mb, "rss_after_perturbation_mb": rss_after_perturbation_mb,
                "rss_after_capability_mb": capability_rss_mb, "rss_after_checkpoint_mb": rss_after_checkpoint_mb,
                "rss_after_cleanup_mb": rss_after_cleanup_mb,
                "delta_from_previous_candidate_mb": ((rss_after_cleanup_mb - previous_candidate_rss_mb) if previous_candidate_rss_mb is not None else 0.0),
                "high_water_mb": high_water_mb,
            })
            previous_candidate_rss_mb = rss_after_cleanup_mb
            perturbation_index += 1

    return newly_completed_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale", required=True, choices=("3B", "7B", "32B"),
        help="3B/7B are runnable unconditionally. 32B is runnable ONLY via --smoke, and ONLY after "
             "the Section-14 readiness gates pass -- see stage11_32b_readiness.py. 72B is not a "
             "valid choice here at all (hard-disabled, not merely gated).",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="32B ONLY -- defaults to 4 (task spec Section 3). Ignored for 3B/7B, which remain TP=1 exactly as before this option existed.")
    parser.add_argument("--model-revision-ref", default=None, help="Overrides the registry's default revision_ref (\"main\") if given.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "stage11_whole_model_scaling"))
    parser.add_argument("--smoke", action="store_true", help="1 region x 3 radii x 1 direction family x 6 capabilities x 5 D_map examples = 3 perturbations, 18 rows, 90 evaluations.")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU, no Hub call")
    args = parser.parse_args(argv)

    if args.scale == "32B":
        # NEVER calls ensure_scale_runnable (which would unconditionally raise for 32B, exactly
        # as it still does for any scale outside RUNNABLE_SCALES). Only the ONE structural,
        # zero-evidence-needed requirement is enforced here (--smoke present -- this module is
        # whole_model-only already, so the anatomy-track prohibition is automatically satisfied).
        # The FULL gate requirement (ensure_32b_smoke_permitted, needing REAL evidence) is
        # evaluated further below, only after the readiness pre-flight has actually produced a
        # gate report -- evaluating it here, before any evidence exists, would just be a
        # guaranteed-always-raise no-op that never lets the pre-flight report get written.
        if not args.smoke:
            print("32B is runnable ONLY via --smoke -- refusing to run a full 32B whole-model sweep.", file=sys.stderr)
            return 1
    else:
        ensure_scale_runnable(args.scale)
    spec = get_scaling_model_spec(args.scale)
    if args.model_revision_ref is not None:
        spec = ScalingModelSpec(scale_label=spec.scale_label, model_name=spec.model_name, revision_ref=args.model_revision_ref, model_family=spec.model_family)

    if args.dry_run:
        placeholder_revision = "UNRESOLVED-dry-run-only"
        plan = (build_whole_model_smoke_plan if args.smoke else build_whole_model_plan)(spec=spec, model_revision=placeholder_revision, output_root=args.output_root)
        print(f"Whole-model plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}, scale={plan.scale_label}): region={WHOLE_MODEL_REGION_LABEL} radii={plan.radii}")
        print(f"capabilities={plan.capabilities}")
        print(f"n_directions_per_cell={plan.n_directions_per_cell} d_map_n={plan.d_map_n} total_unique_perturbations={plan.total_unique_perturbations}")
        print(f"total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations}")
        print(f"total_perturbed_model_example_evaluations={plan.total_perturbed_model_example_evaluations}")
        print(f"output_dir={plan.output_dir}")
        print(f"historical_disqualification_note={WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE!r}")
        print("(dry-run: model revision NOT resolved, no Hub/GPU call made)")
        return 0

    try:
        assert_feasible(
            "Stage 11 whole-model scaling", [check_cuda(), check_module("vllm"), check_module("ray"), check_module("huggingface_hub"), check_disk(REPO_ROOT, 50.0)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    resolution = resolve_immutable_model_revision(spec.model_name, spec.revision_ref)
    print(f"Resolved model revision: {resolution}")

    plan = (build_whole_model_smoke_plan if args.smoke else build_whole_model_plan)(spec=spec, model_revision=resolution["resolved_revision"], output_root=args.output_root)
    print(f"Whole-model plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}, scale={plan.scale_label}): region={WHOLE_MODEL_REGION_LABEL} radii={plan.radii}")
    print(f"total_unique_perturbations={plan.total_unique_perturbations} total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations} total_perturbed_model_example_evaluations={plan.total_perturbed_model_example_evaluations}")

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "model_revision_resolution.json").write_text(json.dumps(resolution, indent=2))

    if plan.scale_label == "32B":
        # EARLY, DEDICATED 32B branch -- diverges completely from every line of 3B/7B code below
        # (engine_config, launch_stage6_engine, store_base_weights_via_rpc, run_whole_model_rpc,
        # ...none of which are TP-aware or cpu_base_weights-aware) rather than threading 32B
        # conditionals through that existing, already-validated 3B/7B path. 3B/7B reaching this
        # point never take this branch (plan.scale_label is "3B"/"7B" there) and fall straight
        # through to the unchanged code that follows, byte-identical to before this milestone.
        tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 4
        preflight = run_32b_readiness_preflight_and_report(resolved_revision=resolution["resolved_revision"], tensor_parallel_size=tp_size, output_dir=plan.output_dir)
        print(f"32B readiness gate report written to {preflight['report_path']}")
        print(f"32B gate results: {preflight['gate_results']}")
        if not preflight["smoke_permitted"]:
            print("32B whole-model smoke BLOCKED -- not all G1-G8 gates report PASS.", file=sys.stderr)
            print(V3_SOLVER_DISTRIBUTED_EXTENSION_NOTE, file=sys.stderr)
            print(
                "The distributed v3 solver now EXISTS and is proven correct on CPU (world_size=1 "
                "equivalence + simulated 2-rank equivalence, tests/test_thicket_distributed_v3_solver.py) "
                "-- G4/G5 report READY_FOR_LIVE_VERIFICATION, not PASS, because CPU tests alone can never "
                "satisfy a gate that requires live TP hardware evidence (task spec Section 11). "
                "No engine was launched, no GPU memory was touched, no scientific row can exist for this attempt.",
                file=sys.stderr,
            )
            return 0
        # Unreachable today (smoke_permitted requires literal PASS on every gate, and G4/G5 can
        # only ever be PASS given a real `live_test_passed` result this module never fabricates)
        # -- kept as the single documented continuation point for the live-GPU integration step
        # that follows once real TP hardware evidence exists: launch a TP={tp_size} engine with
        # build_32b_engine_config(tensor_parallel_size=tp_size), dispatch store_base_weights_cpu_
        # rpc/collective_rpc_all_workers/scoped_apply_anatomical_perturbation_bf16_quantization_
        # aware_v3_distributed in place of the legacy single-worker calls below, and run the
        # (now distributed-aware) candidate loop.
        raise RuntimeError("Unreachable: 32B gate evaluation must have already returned above.")

    engine_config = build_stage7b_engine_config()
    assert engine_config["enable_prefix_caching"] is False, "Whole-model track must never run with prefix caching enabled."
    print(format_runtime_compatibility_diagnostic(
        {"model_name": plan.model_name, "requested_revision": plan.model_revision, "resolved_snapshot_path": "<resolved below>"},
        worker_extension_cls="utils.worker_extn.WorkerExtension", vllm_version=get_vllm_version(),
        engine_mode=detect_vllm_engine_mode(), engine_config=engine_config,
    ))

    from .benchmarks.runner import run_benchmark
    from .config import load_capability_benchmark_config
    from .run_capability_benchmark_gate import load_adapter
    from .thicket.data_roles import write_data_role_manifest
    from .vlm_adapter import bootstrap_ray, resolve_model_snapshot, verify_workers_can_import_external_root

    resolved_snapshot_path = resolve_model_snapshot(plan.model_name, plan.model_revision)

    subset_ids_dir = plan.output_dir / "d_map_subsets"
    capability_contexts = build_d_map_capability_contexts(
        STAGE8_BASE_SEED, subset_ids_dir, plan.d_map_n,
        load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
    )
    for capability, ctx in capability_contexts.items():
        write_data_role_manifest(ctx.partition, plan.output_dir / "data_roles" / f"{capability}_d_map.json")

    # MODE-AWARE subset gate (this repair pass -- see this section's own module-level comment
    # above SUBSET_GATE_MODE_FULL for the full root-cause writeup): FULL mode keeps the exact
    # Stage-8 N=50 subset-hash equality check UNCHANGED; SMOKE mode (D_map N=5) never compares its
    # subset hash numerically to the N=50 Stage-8 manifests at all -- it independently rebuilds
    # the D_map a second time and requires both builds' subset hashes (and example counts) to
    # match instead.
    if plan.is_smoke:
        print(f"Running Stage-11 SMOKE subset gate (D_map N={plan.d_map_n} deterministic reconstruction -- never compared to Stage-8's N=50 authoritative hashes)...")
        second_subset_ids_dir = plan.output_dir / "d_map_subsets_smoke_pass_b"
        capability_contexts_pass_b = build_d_map_capability_contexts(
            STAGE8_BASE_SEED, second_subset_ids_dir, plan.d_map_n,
            load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
        )
        smoke_determinism_report = run_smoke_subset_determinism_check(capability_contexts, capability_contexts_pass_b, plan.d_map_n)
        del capability_contexts_pass_b
        subset_gate_report = build_subset_gate_report(is_smoke=True, d_map_n=plan.d_map_n, smoke_determinism_report=smoke_determinism_report)
        (plan.output_dir / "subset_gate.json").write_text(json.dumps(subset_gate_report, indent=2))
        ensure_subset_gate_passes(subset_gate_report)
        print(f"Confirmed: Stage-11 smoke D_map N={plan.d_map_n} subset construction is deterministic across two independent builds for all 6 capabilities "
              f"(never compared to Stage-8's N=50 authoritative hashes).")
    else:
        full_subset_hash_report = run_subset_hash_check(capability_contexts)
        subset_gate_report = build_subset_gate_report(is_smoke=False, d_map_n=plan.d_map_n, full_subset_hash_report=full_subset_hash_report)
        (plan.output_dir / "subset_gate.json").write_text(json.dumps(subset_gate_report, indent=2))
        ensure_subset_gate_passes(subset_gate_report)
        print("Confirmed: live whole-model D_map subset hashes exactly match Stage-8's authoritative 3B manifests for all 6 capabilities.")

    seed_bank = build_scaling_direction_seed_bank(_base_seed_for_scale(plan.scale_label), plan.scale_label, WHOLE_MODEL_REGIONS, plan.n_directions_per_cell)
    (plan.output_dir / "direction_family_manifest.json").write_text(json.dumps(
        {"regions": list(WHOLE_MODEL_REGIONS), "n_directions_per_cell": plan.n_directions_per_cell,
         "seed_bank": {r: list(s) for r, s in seed_bank.items()}, "direction_seed_bank_hash": compute_direction_seed_bank_hash(seed_bank)}, indent=2,
    ))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolved_snapshot_path)

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    import ray

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)
        ensure_stage7b_encoder_cache_reset_mechanism_exposed(EXTERNAL_ROOT)

        engines, pgs = launch_stage6_engine(
            resolved_snapshot_path, precision=engine_config["precision"], gpu_memory_utilization=engine_config["gpu_memory_utilization"],
            max_model_len=engine_config["max_model_len"], tensor_parallel_size=engine_config["tensor_parallel_size"],
            enable_prefix_caching=engine_config["enable_prefix_caching"],
        )
        engine = engines[0]

        store_base_weights_via_rpc(engine)
        print(format_base_snapshot_confirmation(engine_config["gpu_memory_utilization"], engine_config["base_snapshot_mode"]))

        ensure_encoder_cache_reset_available(engine)

        anatomy_audit = _collective_rpc_single_worker(engine, report_scaling_anatomy_audit, args=(WHOLE_MODEL_REGIONS, MODEL_FAMILY), label="report_scaling_anatomy_audit")
        (plan.output_dir / "anatomy_audit.json").write_text(json.dumps(anatomy_audit, indent=2))
        ensure_scaling_anatomy_audit_passes(anatomy_audit, WHOLE_MODEL_REGIONS)
        ensure_whole_model_covers_100_percent(anatomy_audit)
        print(f"Confirmed: live {plan.scale_label} whole_model audit passed (100% of {anatomy_audit['total_model_elements']} elements, mask_hash={anatomy_audit['regions']['whole_model']['mask_hash'][:12]}...).")

        region_info = _collective_rpc_single_worker(engine, report_region_param_names_for_scaling, args=(WHOLE_MODEL_REGIONS, MODEL_FAMILY), label="report_region_param_names_for_scaling")
        region_param_names = tuple(region_info[WHOLE_MODEL_REGION_LABEL]["param_names"])
        mask_hash = anatomy_audit["regions"][WHOLE_MODEL_REGION_LABEL]["mask_hash"]
        if region_info[WHOLE_MODEL_REGION_LABEL]["mask_hash"] != mask_hash:
            raise RuntimeError("Mask-hash mismatch between report_region_param_names_for_scaling and report_scaling_anatomy_audit for whole_model.")

        llm_adapter = RayEngineLLMAdapter(engine)
        from .run_global_visual_thicket_pilot import load_or_compute_baseline_scores
        baseline_path = plan.output_dir / "baseline_scores.json"
        load_or_compute_baseline_scores(baseline_path, capability_contexts, plan.model_revision, plan.run_signature, llm_adapter, tokenizer, sampling_params)

        print(f"Running baseline repeatability preflight for {plan.scale_label} whole_model before any perturbation...")
        preflight_report = run_baseline_repeatability_preflight_rpc(
            engine, capability_contexts, tokenizer, sampling_params, run_benchmark=run_benchmark, generation_batch_size=plan.generation_batch_size,
        )
        (plan.output_dir / "baseline_repeatability_preflight.json").write_text(json.dumps(preflight_report, indent=2))
        ensure_baseline_repeatability(preflight_report)
        print(f"Baseline repeatability preflight PASSED for all {len(preflight_report)} capabilities.")

        newly_written_rows = run_whole_model_rpc(
            plan, capability_contexts, engine, tokenizer, sampling_params, seed_bank, region_param_names, mask_hash, anatomy_audit, run_benchmark=run_benchmark,
        )
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    manifest = write_whole_model_run_manifest(plan.output_dir)
    print(f"Wrote {newly_written_rows} NEW result rows this run to {plan.output_dir / 'results.jsonl'}")
    print(f"Run manifest: {json.dumps(manifest, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
