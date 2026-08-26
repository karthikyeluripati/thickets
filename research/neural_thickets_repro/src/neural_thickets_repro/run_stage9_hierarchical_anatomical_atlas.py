"""Stage 9: hierarchical (L2 depth-band) anatomical localization. Drills down into the two L1
parents Stage 8 found reproducible anatomical signal in -- vision, language -- at the SAME
frozen common radii, SAME six capabilities, SAME execution stack Stage 8 already proved
memory-bounded and scientifically valid. multimodal_connector_or_merger is deliberately NOT
drilled down (Stage 8 found no stable capability-selective dominance there, and it is
architecturally far smaller than vision/language); it remains a coarse L1 reference only.

Central question: do the coarse L1 regions hide sharper, DEPTH-SPECIFIC expert populations?

=================================================================================================
REUSE, BY IDENTITY, FROM STAGE 7B/8 (this module changes NONE of these):
=================================================================================================
- perturbation_mode=anatomical_relative_l2, radius realization
  scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 (v3 quantization-aware
  acceptance, unchanged).
- multimodal_cache_policy=full_encoder_reset_vllm011_verified_v2 / enable_prefix_caching=False.
- launch_stage6_engine / build_stage7b_engine_config (bf16, max_model_len=4096,
  gpu_memory_utilization=0.60, TP=1).
- STAGE8_GENERATE_BATCH_SIZE=10 bounded-generation path (benchmarks.runner.run_benchmark's
  max_requests_per_generate) for BOTH the baseline check and every candidate's 6 capability
  evaluations -- this is what made Stage 8's own N=50 evaluation driver-RSS-safe; Stage 9 is the
  SAME per-capability N=50 evaluation shape, 12x more candidates, so the identical bound applies.
- STAGE8_RADII (the 3 frozen common radii), STAGE8_CAPABILITIES (the 6 frozen capabilities),
  STAGE8_D_MAP_N (50) -- imported BY IDENTITY, never retyped.
- fixed-base restoration discipline (store_base_weights once; reset_to_base_weights_via_rpc +
  verify_exact_fixed_base_restoration_via_rpc every candidate).

=================================================================================================
WHAT IS NEW IN STAGE 9
=================================================================================================
1. SIX CHILD REGIONS (thicket.anatomy_stage9.build_stage9_hierarchical_partition), replacing
   Stage 8's three L1 anatomies as the perturbation target: vision_early, vision_mid,
   vision_late, language_early, language_mid, language_late. Built ONCE per candidate, live,
   from the model's own real named_parameters() (report_stage9_child_param_names, the RPC
   analogue of Stage 7B/8's report_region_param_names) -- never a hardcoded tensor list.
   Language's two L1-parent tensors NOT covered by the numbered-layer depth bands (the input
   token embedding, the final pre-head normalization) are resolved by
   anatomy_stage9's deterministic architectural-position rule: input-side -> language_early,
   output-side -> language_late. Vision's own three depth bands were live-audited (this repair
   pass) to already exactly partition the vision L1 parent with ZERO uncovered tensor -- see
   anatomy_stage9's own module docstring for the confirmed evidence (REPRO_SPEC.md's live
   tensor-name inventory, cross-checked against thicket.anatomy.validate_atlas's own
   `uncovered_by_parent` report on a fixture mirroring that exact tensor structure).

2. DIRECTION-FAMILY SEED BANK PER CHILD REGION (not per L1 region): 64 seeds for EACH of the 6
   child regions, derived as a function of (Stage-9 base seed, child region name, direction
   index) ONLY -- never radius -- via thicket.seeds.derive_seed, reused across all 3 radii
   within that SAME child region (identical mechanism to Stage 8's own per-L1-region seed bank,
   see build_stage8_direction_seed_bank's own docstring for the underlying guarantee:
   apply_anatomical_relative_l2's noise generation is a pure function of seed alone, never of
   r). Population: 6 child regions x 3 radii x 64 directions = 1152 unique perturbations x 6
   capabilities = 6912 rows. Across DIFFERENT child regions (including vision_early vs
   language_early, or vision_early vs vision_mid), the same numeric seed integer MAY
   coincidentally recur but must NEVER be interpreted as the same geometric direction --
   disjoint parameter subspaces by construction.

3. STAGE-8-BASELINE-EQUALITY HARD GATE (run_stage9_baseline_equality_check /
   ensure_stage9_baseline_matches_stage8): before ANY Stage-9 perturbation, computes theta_0
   baseline under the identical execution policy and requires an EXACT match (never
   approximate, never averaged) against the authoritative Stage-8 baseline for every one of the
   6 capabilities (visual_grounding=0.880, counting=0.680, spatial_reasoning=0.700,
   ocr_text_recognition_grounded=0.938, relational_reasoning=0.540,
   fine_grained_recognition=0.420). Any mismatch is a HARD STOP
   (Stage9BaselineMismatchError) before any of the 1152 perturbations are evaluated.

4. Stage9CheckpointManifest additionally records the Stage-8 parent run's own identity/hash
   (stage8_parent_run_signature, stage8_parent_direction_seed_bank_hash) purely as PROVENANCE
   (never re-validated against a live Stage-8 checkpoint file -- Stage 8 is already complete and
   archived) -- so a Stage-9 result can always be traced back to which authoritative Stage-8 run
   motivated it -- plus a `partition_audit_hash` (sha256 of the canonical JSON of both parents'
   Stage9PartitionAudit) so a resume against a differently-computed partition is rejected exactly
   like a model-revision or capability-hash mismatch.

=================================================================================================
FROZEN FULL Stage-9 CONFIG
=================================================================================================
model:          Qwen/Qwen2.5-VL-3B-Instruct @ 66285546d2b821cf421d4f5eb2576359d3770cd3, bf16
child regions:  vision_early, vision_mid, vision_late, language_early, language_mid,
                language_late (STAGE9_CHILD_REGIONS)
radii:          STAGE8_RADII, unchanged, imported by identity (never recalibrated per depth band)
capabilities:   STAGE8_CAPABILITIES, unchanged, imported by identity
directions:     STAGE9_N_DIRECTIONS_PER_CELL (64) per (child region, radius) cell, same
                direction seed reused across the 3 radii within one child region
data:           STAGE8_D_MAP_N (50) -- Stage 8's own frozen D_map manifests/subset hashes reused
                exactly, never resampled
Total:          6 child regions x 3 radii x 64 directions = 1152 unique perturbations
                1152 x 6 capabilities = 6912 perturbation x capability result rows
                6912 x 50 = 345,600 perturbed model-example evaluations
No D_confirm/D_select/D_test is ever constructed or referenced anywhere in this module.

=================================================================================================
SMOKE MODE -- EXECUTION SIZE ONLY, same scientific definitions
=================================================================================================
--smoke overrides ONLY population size: all 6 STAGE9_CHILD_REGIONS, all 3 STAGE8_RADII, but only
STAGE9_SMOKE_N_DIRECTIONS (1) direction family, all 6 frozen capabilities, STAGE9_SMOKE_D_MAP_N
(5) D_map examples/capability. Expected: 6 x 3 x 1 = 18 unique perturbations, 18 x 6 = 108
result rows, 108 x 5 = 540 perturbed model-example evaluations.

=================================================================================================
DRIVER-RSS OOM DURING FULL CANDIDATE EVALUATION -- root-caused and fixed (this repair pass)
=================================================================================================
The live full run (1152 candidates -- exactly 2x Stage 8's 576) OOM'd mid-sweep at driver RSS
~101 GB (node ~116.42 GB, 95% threshold), NOT during the baseline preflight (already fixed for
Stage 8/reused here) but during candidate evaluation itself
(run_stage9_rpc -> evaluate_one_stage9_candidate_rpc -> run_benchmark -> ray.get). Differential
audit against Stage 8's identical-shaped code (run_stage8_rpc/evaluate_one_stage8_candidate_rpc)
found the ONE structural difference that scales with candidate count and is retained for the
ENTIRE run: run_stage9_rpc (like the pre-fix run_stage8_rpc) accumulated EVERY candidate's full
ExperimentResultRecord objects -- both already-checkpointed rows (reloaded whole on every
resume, growing with resume depth) and newly-evaluated ones -- into one `all_records` list held
for the whole 1152-candidate sweep, then returned. Fixed: run_stage9_rpc now retains only the
SET of already-completed perturbation IDs (never their row contents) and returns an int count
of newly-written rows -- results.jsonl itself (already re-read independently by
write_stage9_run_manifest) is the sole source of truth for totals, never re-accumulated in
memory. evaluate_one_stage9_candidate_rpc additionally `del`s each capability's RunResult and
calls mem_telemetry.release_transient_memory() immediately after its needed values are
extracted (fixes ownership/retention FIRST, gc/malloc_trim second, never gc alone). New
per-candidate RSS telemetry (candidate_memory_telemetry.jsonl, kept fully separate from the
scientific results.jsonl schema) records rss_start/after_perturbation/after_each_capability/
after_checkpoint/after_cleanup/delta_from_previous_candidate/high_water per candidate, so a
future run's own telemetry can directly show whether driver RSS still rises monotonically.
AUDITED, confirmed NOT a contributing issue (documented rather than re-investigated blind):
direction tensors are never pre-materialized on the driver (only integer seeds live in
build_stage9_direction_seed_bank; the actual Gaussian noise tensors are generated INSIDE the
GPU worker, per RPC call, by apply_anatomical_relative_l2's own _generate_noise); D_map
capability_contexts and the six child-region partition/mask reports are each built EXACTLY ONCE
in main(), before the candidate loop, never rebuilt per candidate. FAILURE PATH (Section 8 of
the spec): the live crash log showed a primary Ray OOM followed by a confusing SECONDARY
RayActorError from evaluate_one_stage9_candidate_rpc's own `finally`-block restoration attempt
against an already-dead engine. Fixed: `_is_ray_unrecoverable_error` detects this class of
exception and skips ALL further RPCs (no restoration, no cache reset) when the engine is no
longer usable, letting the ORIGINAL exception propagate cleanly -- a candidate that fails this
way has zero rows appended, which already satisfies "an interrupted candidate is never
checkpointed as complete" without a masking secondary crash. Any OTHER (recoverable-engine)
exception still gets the existing best-effort restoration attempt, unchanged.
EXECUTION-ONLY: Stage9CheckpointManifest/compute_stage9_run_signature are UNCHANGED by this
pass -- the fixed runner resumes the SAME stage9_hierarchical_anatomical_atlas_3b_v1 scientific
run (already-checkpointed candidates are skipped exactly as before), never a new run identity.

=================================================================================================
DO NOT DO YET
=================================================================================================
Attention-vs-MLP, individual heads, low-rank/SVD geometry, 7B/72B or frontier models,
post-training, routing, distillation, D_confirm-based final claims.
"""
from __future__ import annotations

import argparse
import hashlib
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
    build_d_map_context,
    detect_vllm_engine_mode,
    format_base_snapshot_confirmation,
    format_runtime_compatibility_diagnostic,
    get_vllm_version,
    launch_stage6_engine,
    load_completed_perturbation_rows,
    load_records,
    reset_to_base_weights_via_rpc,
    resolve_and_report_model_snapshot,
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
    STAGE8_CAPABILITIES,
    STAGE8_D_MAP_N,
    STAGE8_RADII,
    ensure_baseline_repeatability,
    run_baseline_repeatability_preflight_rpc,
    _FULL_RUN_SIGNATURE_V2_BATCHED as STAGE8_PARENT_RUN_SIGNATURE,
)
from .scoped_anatomical_perturbation import (
    QUANTIZATION_AWARE_METHOD_V3,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    CorrectionOutOfRegionDriftError,
    QuantizationToleranceExceededError,
    RadiusCorrectionFailedError,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
)
from .thicket.anatomy_stage9 import (
    STAGE9_CHILD_REGIONS,
    STAGE9_DRILLDOWN_PARENTS,
    Stage9PartitionAudit,
    build_stage9_hierarchical_partition,
    ensure_stage9_partition_valid,
)
from .thicket.perturbation import PERTURBATION_MODES, PerturbationManifest
from .thicket.schema import ExperimentResultRecord
from .thicket.seeds import derive_seed
from .vlm_adapter import ensure_full_encoder_cache_reset_exposed, reset_vllm_encoder_cache_full

assert PERTURBATION_MODE in PERTURBATION_MODES
assert RADIUS_REALIZATION_METHOD == QUANTIZATION_AWARE_METHOD_V3

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

EXPERIMENT_ID = "stage9_hierarchical_anatomical_atlas"
STAGE9_BASE_SEED = 20260901  # distinct from Stage 6 (20260823) / Stage 7B (20260824) / Stage 8 (20260825)

STAGE9_RADII: Tuple[float, ...] = STAGE8_RADII  # unchanged, imported by identity -- never recalibrated per depth band
STAGE9_CAPABILITIES: Tuple[str, ...] = STAGE8_CAPABILITIES  # unchanged, imported by identity
STAGE9_N_DIRECTIONS_PER_CELL = 64
STAGE9_D_MAP_N = STAGE8_D_MAP_N  # 50, Stage 8's own frozen D_map size -- reuses Stage 8's manifests exactly
STAGE9_GENERATE_BATCH_SIZE = 10  # same bounded-generation path Stage 8 proved memory-safe at N=50
DATASET_ROLE = "map"

STAGE9_SMOKE_N_DIRECTIONS = 1
STAGE9_SMOKE_D_MAP_N = 5

_ALLOWED_D_MAP_SIZES: Tuple[int, ...] = (STAGE9_D_MAP_N, STAGE9_SMOKE_D_MAP_N)

_FULL_RUN_SIGNATURE = "stage9_hierarchical_anatomical_atlas_3b_v1"

# Authoritative Stage-8 baseline, per capability -- the exact hard-gate values from the
# completed, authoritative stage8_coarse_anatomical_atlas_3b_v2_batched10 run. Never averaged,
# never silently replaced; any live Stage-9 mismatch is a hard stop (see
# ensure_stage9_baseline_matches_stage8).
STAGE8_AUTHORITATIVE_BASELINE: Dict[str, float] = {
    "visual_grounding": 0.880,
    "counting": 0.680,
    "spatial_reasoning": 0.700,
    "ocr_text_recognition_grounded": 0.938,
    "relational_reasoning": 0.540,
    "fine_grained_recognition": 0.420,
}

# Authoritative Stage-8 D_map N=50 subset hashes, per capability -- from the completed
# stage8_coarse_anatomical_atlas_3b_v2_batched10 run's own checkpoint/run manifest. FULL-mode
# only (see ensure_stage9_subset_hashes_match_stage8): Stage 9's full run must reuse Stage 8's
# EXACT frozen D_map manifests, never a re-shuffled/re-sampled subset. Never checked in smoke
# mode -- an N=5 subset's manifest hash legitimately differs from the N=50 hash by construction
# (different sample count), never a defect.
STAGE8_AUTHORITATIVE_SUBSET_HASHES: Dict[str, str] = {
    "visual_grounding": "22b70bcf4d278dbce11f8b8be793f318937a5e1d17a2bf855a4b395b56dd4ac5",
    "counting": "0c3c6c75944ecdd00ab2c4d406cf5da449d375976a377e9d49b7305d4a98e33b",
    "spatial_reasoning": "e371fc3dff529b61766e87558e3bed68de51082ee57d8d6859a9a13a331801a4",
    "ocr_text_recognition_grounded": "3117bd779516420d50c44c2155e1a4ccc589c2fd67237f6b3245a00a00802ecb",
    "relational_reasoning": "cf507f2d18ab763415153222513f373839813c6427bab1a2e62bdcaa27e42917",
    "fine_grained_recognition": "7f710b0c998e93188806fccadab6a0263fee2405da976a16b792381720674f99",
}


class DatasetRoleViolationError(RuntimeError):
    """Something other than the 'map' dataset role was requested, or an unrecognized D_map
    size was requested -- Stage 9 must never construct or reference D_confirm/D_select/D_test.
    """


class Stage9BaselineMismatchError(RuntimeError):
    """FULL MODE ONLY: the live theta_0 baseline computed under Stage 9's own execution policy
    (D_map N=50, Stage 8's own frozen manifests) does not exactly match the authoritative
    Stage-8 baseline for at least one capability. Hard stop BEFORE any of the 1152 Stage-9
    perturbations are evaluated -- never averaged, never silently replaced. Never raised in
    smoke mode (D_map N=5) -- comparing an N=5 live score against an N=50 baseline is not a
    meaningful equality check at all (this repair pass fixes exactly that prior bug); see
    Stage9SmokeBaselineNondeterminismError for smoke mode's own, sample-size-appropriate gate.
    """


class Stage9SubsetHashMismatchError(RuntimeError):
    """FULL MODE ONLY: the live D_map N=50 subset hash for at least one capability does not
    match the authoritative Stage-8 N=50 subset hash -- Stage 9's full run must reuse Stage 8's
    exact frozen D_map manifests, never a re-shuffled/re-sampled subset. Never checked in smoke
    mode (see STAGE8_AUTHORITATIVE_SUBSET_HASHES's own docstring).
    """


def compute_stage9_run_signature(
    child_regions: Sequence[str], radii: Sequence[float], n_directions: int, d_map_n: int,
) -> str:
    """`_FULL_RUN_SIGNATURE` iff child_regions/radii/n_directions/d_map_n exactly match the
    frozen full config; otherwise a deterministic "stage9_smoke_..." descriptive string built
    from the ACTUAL values -- so a failed/partial smoke run can never be resumed as (or mistaken
    for) the full run, and vice versa.
    """
    if (
        tuple(sorted(child_regions)) == tuple(sorted(STAGE9_CHILD_REGIONS)) and tuple(radii) == STAGE9_RADII
        and n_directions == STAGE9_N_DIRECTIONS_PER_CELL and d_map_n == STAGE9_D_MAP_N
    ):
        return _FULL_RUN_SIGNATURE
    region_label = "-".join(sorted(child_regions))
    radius_label = "-".join(f"{r:.6f}".replace(".", "") for r in radii)
    return f"stage9_smoke_{region_label}_r{radius_label}_n{d_map_n}_dir{n_directions}"


# =============================================================================================
# Plan (pure arithmetic, no I/O, no GPU)
# =============================================================================================


@dataclass(frozen=True)
class Stage9Plan:
    model_name: str
    model_revision: str
    model_family: str
    model_scale: str
    child_regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_directions_per_cell: int
    d_map_n: int
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: bool
    generation_batch_size: int
    run_signature: str
    output_dir: Path

    @property
    def total_unique_perturbations(self) -> int:
        return len(self.child_regions) * len(self.radii) * self.n_directions_per_cell

    @property
    def total_perturbation_capability_evaluations(self) -> int:
        return self.total_unique_perturbations * len(self.capabilities)

    @property
    def total_perturbed_model_example_evaluations(self) -> int:
        return self.total_perturbation_capability_evaluations * self.d_map_n

    @property
    def is_smoke(self) -> bool:
        return not (
            tuple(sorted(self.child_regions)) == tuple(sorted(STAGE9_CHILD_REGIONS)) and self.radii == STAGE9_RADII
            and self.n_directions_per_cell == STAGE9_N_DIRECTIONS_PER_CELL and self.d_map_n == STAGE9_D_MAP_N
        )


def build_stage9_plan(
    *, model_name: str, model_revision: str, output_root: "str | Path",
    child_regions: Sequence[str] = STAGE9_CHILD_REGIONS, radii: Sequence[float] = STAGE9_RADII,
    n_directions_per_cell: int = STAGE9_N_DIRECTIONS_PER_CELL, d_map_n: int = STAGE9_D_MAP_N,
    model_family: str = "qwen2_5_vl", model_scale: str = "3B",
    generation_batch_size: int = STAGE9_GENERATE_BATCH_SIZE,
) -> Stage9Plan:
    if not child_regions:
        raise ValueError("Stage 9 requires at least one child region.")
    if not radii:
        raise ValueError("Stage 9 requires at least one radius.")
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 9 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    if generation_batch_size <= 0:
        raise ValueError(f"generation_batch_size must be positive, got {generation_batch_size}")

    run_signature = compute_stage9_run_signature(child_regions, radii, n_directions_per_cell, d_map_n)
    return Stage9Plan(
        model_name=model_name, model_revision=model_revision, model_family=model_family, model_scale=model_scale,
        child_regions=tuple(child_regions), radii=tuple(radii), capabilities=STAGE9_CAPABILITIES,
        n_directions_per_cell=n_directions_per_cell, d_map_n=d_map_n,
        radius_realization_method=RADIUS_REALIZATION_METHOD, multimodal_cache_policy=MULTIMODAL_CACHE_POLICY,
        enable_prefix_caching=ENABLE_PREFIX_CACHING, generation_batch_size=generation_batch_size,
        run_signature=run_signature, output_dir=Path(output_root) / run_signature,
    )


def build_stage9_smoke_plan(*, model_name: str, model_revision: str, output_root: "str | Path") -> Stage9Plan:
    """6 child regions x 3 radii x 1 direction family x 6 capabilities x D_map N=5 = 18
    perturbations, 108 rows, 540 perturbed model-example evaluations.
    """
    return build_stage9_plan(
        model_name=model_name, model_revision=model_revision, output_root=output_root,
        child_regions=STAGE9_CHILD_REGIONS, radii=STAGE9_RADII,
        n_directions_per_cell=STAGE9_SMOKE_N_DIRECTIONS, d_map_n=STAGE9_SMOKE_D_MAP_N,
    )


# =============================================================================================
# Direction-family seed bank (per child region) + population
# =============================================================================================


def build_stage9_direction_seed_bank(base_seed: int, child_regions: Sequence[str], n_directions: int) -> Dict[str, Tuple[int, ...]]:
    """Exactly `n_directions` seeds PER CHILD REGION, derived as a function of (child region
    name, direction index) ONLY -- never radius -- so the identical seed is reused across all 3
    radii within one child region. Reuses thicket.seeds.derive_seed with a Stage-9-specific
    namespace ("stage9_direction_family") -- an independent stream from Stage 8's own
    "stage8_direction_family" namespace even where a child region name happens to coincide with
    an L1 region name.
    """
    return {
        region: tuple(derive_seed(base_seed, "stage9_direction_family", region, str(i)) for i in range(n_directions))
        for region in child_regions
    }


def compute_direction_seed_bank_hash(seed_bank: Dict[str, Tuple[int, ...]]) -> str:
    canonical = json.dumps({region: list(seeds) for region, seeds in sorted(seed_bank.items())}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_partition_audit_hash(audits: Dict[str, Stage9PartitionAudit]) -> str:
    canonical = json.dumps(
        {
            parent: {
                "child_band_names": audit.child_band_names,
                "uncovered_tensors": list(audit.uncovered_tensors),
                "uncovered_tensor_assignment": audit.uncovered_tensor_assignment,
                "union_equals_parent": audit.union_equals_parent,
                "children_pairwise_disjoint": audit.children_pairwise_disjoint,
            }
            for parent, audit in sorted(audits.items())
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Stage9DirectionAssignment:
    manifest: PerturbationManifest
    child_region: str
    direction_index: int
    direction_seed: int

    @property
    def direction_family_id(self) -> str:
        return f"{self.child_region}:{self.direction_index}"


def build_stage9_population(
    plan: Stage9Plan, seed_bank: Dict[str, Tuple[int, ...]], child_mask_hash_by_region: Dict[str, str],
) -> Dict[Tuple[str, float], Tuple[Stage9DirectionAssignment, ...]]:
    missing_regions = set(plan.child_regions) - set(child_mask_hash_by_region)
    if missing_regions:
        raise ValueError(f"Missing parameter_mask_hash for child region(s): {sorted(missing_regions)}")
    missing_bank_regions = set(plan.child_regions) - set(seed_bank)
    if missing_bank_regions:
        raise ValueError(f"Missing direction seed bank for child region(s): {sorted(missing_bank_regions)}")
    for region in plan.child_regions:
        if len(seed_bank[region]) != plan.n_directions_per_cell:
            raise ValueError(f"Direction seed bank for {region!r} has {len(seed_bank[region])} seeds, expected {plan.n_directions_per_cell}.")

    population_by_cell: Dict[Tuple[str, float], Tuple[Stage9DirectionAssignment, ...]] = {}
    for region in plan.child_regions:
        mask_hash = child_mask_hash_by_region[region]
        seeds = seed_bank[region]
        for radius in plan.radii:
            assignments = tuple(
                Stage9DirectionAssignment(
                    manifest=PerturbationManifest(
                        seed=seed, perturbation_mode=PERTURBATION_MODE, anatomy_region=region, radius=radius, sigma=None,
                        model_family=plan.model_family, model_scale=plan.model_scale, model_revision=plan.model_revision,
                        parameter_mask_hash=mask_hash,
                    ),
                    child_region=region, direction_index=i, direction_seed=seed,
                )
                for i, seed in enumerate(seeds)
            )
            population_by_cell[(region, radius)] = assignments
    return population_by_cell


class DirectionSeedReuseViolationError(RuntimeError):
    """The population does not satisfy Stage 9's per-child-region direction-family invariant."""


def validate_stage9_direction_seed_reuse(
    plan: Stage9Plan, population_by_cell: Dict[Tuple[str, float], Tuple[Stage9DirectionAssignment, ...]],
) -> None:
    all_ids = [a.manifest.perturbation_id for cell in population_by_cell.values() for a in cell]
    if len(all_ids) != len(set(all_ids)):
        raise DirectionSeedReuseViolationError(f"Duplicate perturbation_id(s) ({len(all_ids)} total, {len(set(all_ids))} unique).")
    expected_total = len(plan.child_regions) * len(plan.radii) * plan.n_directions_per_cell
    if len(all_ids) != expected_total:
        raise DirectionSeedReuseViolationError(f"Stage-9 population has {len(all_ids)} perturbations, expected {expected_total}.")

    by_region_seed: Dict[str, Dict[int, int]] = {}
    for (region, _radius), assignments in population_by_cell.items():
        counts = by_region_seed.setdefault(region, {})
        for a in assignments:
            counts[a.direction_seed] = counts.get(a.direction_seed, 0) + 1
    for region in plan.child_regions:
        counts = by_region_seed.get(region, {})
        if len(counts) != plan.n_directions_per_cell:
            raise DirectionSeedReuseViolationError(f"Region {region!r} has {len(counts)} distinct direction seeds, expected {plan.n_directions_per_cell}.")
        wrong = {seed: n for seed, n in counts.items() if n != len(plan.radii)}
        if wrong:
            raise DirectionSeedReuseViolationError(f"Region {region!r}: seed(s) not reused exactly {len(plan.radii)}x: {wrong}")


# =============================================================================================
# Checkpoint identity
# =============================================================================================


class IncompatibleStage9CheckpointError(RuntimeError):
    """An existing checkpoint_manifest.json in this output directory does not match the
    current run's identity -- refuses to resume a differently-configured partial run.
    """


@dataclass(frozen=True)
class Stage9CheckpointManifest:
    experiment_id: str
    run_signature: str
    restoration_mode: str
    perturbation_mode: str
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: bool
    generation_batch_size: int
    model_revision: str
    dataset_role: str
    child_regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_directions_per_cell: int
    d_map_n: int
    subset_hashes: Dict[str, str]
    child_mask_hashes: Dict[str, str]
    direction_seed_bank_hash: str
    partition_audit_hash: str
    stage8_parent_run_signature: str
    expected_unique_perturbations: int
    expected_result_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "run_signature": self.run_signature,
            "restoration_mode": self.restoration_mode, "perturbation_mode": self.perturbation_mode,
            "perturbation_semantics": self.perturbation_mode, "radius_realization_method": self.radius_realization_method,
            "multimodal_cache_policy": self.multimodal_cache_policy, "enable_prefix_caching": self.enable_prefix_caching,
            "generation_batch_size": self.generation_batch_size,
            "model_revision": self.model_revision, "dataset_role": self.dataset_role,
            "child_regions": list(self.child_regions), "radii": list(self.radii), "capabilities": list(self.capabilities),
            "n_directions_per_cell": self.n_directions_per_cell, "d_map_n": self.d_map_n,
            "subset_hashes": dict(sorted(self.subset_hashes.items())),
            "child_mask_hashes": dict(sorted(self.child_mask_hashes.items())),
            "direction_seed_bank_hash": self.direction_seed_bank_hash,
            "partition_audit_hash": self.partition_audit_hash,
            "stage8_parent_run_signature": self.stage8_parent_run_signature,
            "expected_unique_perturbations": self.expected_unique_perturbations,
            "expected_result_rows": self.expected_result_rows,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage9CheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], run_signature=d["run_signature"], restoration_mode=d["restoration_mode"],
            perturbation_mode=d["perturbation_mode"], radius_realization_method=d["radius_realization_method"],
            multimodal_cache_policy=d["multimodal_cache_policy"], enable_prefix_caching=d["enable_prefix_caching"],
            generation_batch_size=d["generation_batch_size"],
            model_revision=d["model_revision"], dataset_role=d["dataset_role"],
            child_regions=tuple(d["child_regions"]), radii=tuple(d["radii"]), capabilities=tuple(d["capabilities"]),
            n_directions_per_cell=d["n_directions_per_cell"], d_map_n=d["d_map_n"],
            subset_hashes=dict(d["subset_hashes"]), child_mask_hashes=dict(d["child_mask_hashes"]),
            direction_seed_bank_hash=d["direction_seed_bank_hash"], partition_audit_hash=d["partition_audit_hash"],
            stage8_parent_run_signature=d["stage8_parent_run_signature"],
            expected_unique_perturbations=d["expected_unique_perturbations"], expected_result_rows=d["expected_result_rows"],
        )


def build_stage9_checkpoint_manifest(
    plan: Stage9Plan, capability_contexts: Dict[str, CapabilityContext], child_mask_hash_by_region: Dict[str, str],
    seed_bank: Dict[str, Tuple[int, ...]], audits: Dict[str, Stage9PartitionAudit],
) -> Stage9CheckpointManifest:
    if plan.d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 9 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {plan.d_map_n}")
    missing_regions = set(plan.child_regions) - set(child_mask_hash_by_region)
    if missing_regions:
        raise ValueError(f"Missing child_mask_hashes for region(s): {sorted(missing_regions)}")
    return Stage9CheckpointManifest(
        experiment_id=EXPERIMENT_ID, run_signature=plan.run_signature, restoration_mode=RESTORATION_MODE,
        perturbation_mode=PERTURBATION_MODE, radius_realization_method=plan.radius_realization_method,
        multimodal_cache_policy=plan.multimodal_cache_policy, enable_prefix_caching=plan.enable_prefix_caching,
        generation_batch_size=plan.generation_batch_size,
        model_revision=plan.model_revision, dataset_role=DATASET_ROLE,
        child_regions=plan.child_regions, radii=plan.radii, capabilities=plan.capabilities,
        n_directions_per_cell=plan.n_directions_per_cell, d_map_n=plan.d_map_n,
        subset_hashes={c: ctx.subset_hash for c, ctx in capability_contexts.items()},
        child_mask_hashes={r: child_mask_hash_by_region[r] for r in plan.child_regions},
        direction_seed_bank_hash=compute_direction_seed_bank_hash(seed_bank),
        partition_audit_hash=compute_partition_audit_hash(audits),
        stage8_parent_run_signature=STAGE8_PARENT_RUN_SIGNATURE,
        expected_unique_perturbations=plan.total_unique_perturbations,
        expected_result_rows=plan.total_perturbation_capability_evaluations,
    )


def ensure_stage9_checkpoint_manifest(path: Path, current: Stage9CheckpointManifest) -> Stage9CheckpointManifest:
    if path.exists():
        existing = Stage9CheckpointManifest.from_dict(json.loads(path.read_text()))
        if existing != current:
            raise IncompatibleStage9CheckpointError(
                f"Existing checkpoint at {path} is incompatible with this run -- refusing to "
                f"resume: existing={existing.to_dict()} current={current.to_dict()}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current.to_dict(), indent=2))
    return current


def build_stage9_run_manifest_summary(checkpoint: Stage9CheckpointManifest, records: List[ExperimentResultRecord]) -> Dict[str, Any]:
    actual_unique_perturbations = len({r.perturbation_id for r in records})
    actual_result_rows = len(records)
    run_complete = (
        actual_unique_perturbations == checkpoint.expected_unique_perturbations
        and actual_result_rows == checkpoint.expected_result_rows
    )
    return {
        "experiment_id": checkpoint.experiment_id, "run_signature": checkpoint.run_signature,
        "restoration_mode": checkpoint.restoration_mode, "perturbation_mode": checkpoint.perturbation_mode,
        "perturbation_semantics": checkpoint.perturbation_mode, "radius_realization_method": checkpoint.radius_realization_method,
        "multimodal_cache_policy": checkpoint.multimodal_cache_policy, "enable_prefix_caching": checkpoint.enable_prefix_caching,
        "generation_batch_size": checkpoint.generation_batch_size, "model_revision": checkpoint.model_revision,
        "child_regions": list(checkpoint.child_regions), "radii": list(checkpoint.radii), "capabilities": list(checkpoint.capabilities),
        "n_directions_per_cell": checkpoint.n_directions_per_cell, "d_map_n": checkpoint.d_map_n,
        "subset_hashes": dict(sorted(checkpoint.subset_hashes.items())),
        "child_mask_hashes": dict(sorted(checkpoint.child_mask_hashes.items())),
        "direction_seed_bank_hash": checkpoint.direction_seed_bank_hash, "partition_audit_hash": checkpoint.partition_audit_hash,
        "stage8_parent_run_signature": checkpoint.stage8_parent_run_signature,
        "expected_unique_perturbations": checkpoint.expected_unique_perturbations, "actual_unique_perturbations": actual_unique_perturbations,
        "expected_result_rows": checkpoint.expected_result_rows, "actual_result_rows": actual_result_rows,
        "run_complete": run_complete,
    }


NUMERICAL_SOLVER_PATCH_REASON = "post-v3-failure deterministic bracket expansion"


def _git_commit_sha() -> str:
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]  # git walks up to find .git regardless -- matches run_capability_benchmark_gate.py's own convention
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def write_numerical_solver_patch_provenance(output_dir: Path) -> Dict[str, Any]:
    """Provenance sidecar for the scoped_anatomical_perturbation.py bracket-expansion repair
    pass (see that module's "POST-V3-FAILURE DETERMINISTIC BRACKET EXPANSION" docstring
    section) -- written fresh at the start of every Stage-9 run so a resumed run's output
    directory always records that the patched solver was in effect, without asserting anything
    about how many pre-patch candidates exist locally (this function never inspects
    results.jsonl for that count -- see module docstring: "completed pre-patch candidates count
    is not assumed locally").
    """
    from .scoped_anatomical_perturbation import (
        MAX_BRACKET_EXPANSION_STEPS,
        QUANTIZATION_AWARE_METHOD_V3,
        QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
        RADIUS_REALIZATION_TOLERANCE,
        MAX_RADIUS_SOLVER_ITERATIONS,
    )

    record = {
        "patch_commit_sha": _git_commit_sha(),
        "patch_commit_sha_note": (
            "the commit that introduces/updates this file records the actual SHA -- see `git log "
            "--follow -- src/neural_thickets_repro/scoped_anatomical_perturbation.py` if this run's "
            "SHA above predates the patch landing"
        ),
        "reason": NUMERICAL_SOLVER_PATCH_REASON,
        "radius_realization_method": QUANTIZATION_AWARE_METHOD_V3,  # unchanged by this patch -- no v4, no new method name
        "original_first_n_attempts_search_unchanged": True,
        "original_max_solver_iterations": MAX_RADIUS_SOLVER_ITERATIONS,
        "original_acceptance_thresholds_unchanged": True,
        "strict_tolerance": RADIUS_REALIZATION_TOLERANCE,
        "quantization_plateau_relative_tolerance": QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
        "max_bracket_expansion_steps": MAX_BRACKET_EXPANSION_STEPS,
        "activates_only_after_legacy_v3_failure": True,
        "completed_pre_patch_candidates_count_is_not_assumed_locally": True,
    }
    (output_dir / "numerical_solver_patch_provenance.json").write_text(json.dumps(record, indent=2))
    return record


def write_stage9_run_manifest(output_dir: Path) -> Dict[str, Any]:
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_manifest.json found at {checkpoint_path} -- this run never started (or output_dir is wrong).")
    checkpoint = Stage9CheckpointManifest.from_dict(json.loads(checkpoint_path.read_text()))
    records = load_records(output_dir / "results.jsonl")
    manifest = build_stage9_run_manifest_summary(checkpoint, records)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_d_map_capability_contexts(
    base_seed: int, subset_ids_dir: "str | Path", d_map_n: int, *, load_capability_benchmark_config: Callable, load_adapter: Callable,
) -> Dict[str, CapabilityContext]:
    """Reuses Stage 8's own frozen D_map subset-selection seed/rule -- `base_seed` MUST be
    Stage 8's own STAGE8_BASE_SEED (passed explicitly by main(), never re-derived here) so the
    persisted subset IDs are byte-identical to Stage 8's, satisfying "reuse Stage-8 subset
    hashes exactly, do NOT resample examples." Reusing a DIFFERENT base_seed would silently
    resample -- build_or_load_subset's own dataset-drift guard would only catch this if the
    persisted ids_path already existed at a stage9-local location; passing Stage 8's own
    base_seed makes this correct by construction rather than by a defensive check.
    """
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 9 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    from .run_global_visual_thicket_pilot import CAPABILITY_CONFIG_FILES

    contexts: Dict[str, CapabilityContext] = {}
    for capability in STAGE9_CAPABILITIES:
        cfg_path = REPO_ROOT / "configs" / "benchmarks" / CAPABILITY_CONFIG_FILES[capability]
        cfg = load_capability_benchmark_config(cfg_path)
        benchmark = load_adapter(cfg.dataset.adapter)
        ctx = build_d_map_context(benchmark, cfg, capability, d_map_n, base_seed, subset_ids_dir)
        contexts[capability] = ctx
    return contexts


# =============================================================================================
# Worker-RPC transport
# =============================================================================================


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list of per-worker results. Got: {results!r}")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; Stage 9 is TP=1-only and expects exactly 1.")
    return results[0]


def _collective_rpc_single_worker(engine: Any, method, args: tuple = (), *, label: str, ray_get: Optional[Callable] = None) -> Any:
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def report_stage9_child_param_names(worker_self, child_regions: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Runs entirely inside the worker process: builds the Stage-9 hierarchical partition from
    the model's REAL named_parameters(), hard-verifies it (ensure_stage9_partition_valid), and
    returns, for each requested child region, its exact parameter-name list and stable
    mask_hash -- never touches or perturbs any weight. Also returns the partition audits
    (JSON-serializable dict form) for persistence.
    """
    model = worker_self.model_runner.model
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names, model_family="qwen2_5_vl")
    ensure_stage9_partition_valid(children, audits)
    audits_dict = {
        parent: {
            "parent": audit.parent, "child_band_names": audit.child_band_names, "uncovered_tensors": list(audit.uncovered_tensors),
            "uncovered_tensor_assignment": audit.uncovered_tensor_assignment,
            "union_equals_parent": audit.union_equals_parent, "children_pairwise_disjoint": audit.children_pairwise_disjoint,
        }
        for parent, audit in audits.items()
    }
    return {
        "regions": {region: {"param_names": list(children[region].param_names), "mask_hash": children[region].mask_hash} for region in child_regions},
        "audits": audits_dict,
    }


# =============================================================================================
# Baseline gate -- MODE-AWARE (this repair pass: the smoke-mode N=5-vs-N=50 gate bug fix)
# =============================================================================================
#
# ROOT CAUSE (live Stage-9 --smoke run): main() unconditionally ran the FULL-mode exact-equality
# check (run_stage9_baseline_equality_check/ensure_stage9_baseline_matches_stage8) regardless of
# --smoke, comparing a D_map N=5 live score against the authoritative D_map N=50 Stage-8
# baseline. Exact numerical equality between two DIFFERENT sample sizes is not a meaningful
# check at all -- it was guaranteed to spuriously fail (or, worse, spuriously pass by
# coincidence) regardless of whether Stage 9's execution policy was actually correct. This is a
# gate-selection bug, not a scientific-design or execution-policy problem: nothing about the
# frozen 6 capabilities/3 radii/64 directions/child masks/v3 realization/cache policy/prefix
# caching/generation batch size/model revision/Stage-8 baseline VALUES changes here.
#
# FIX: the baseline gate is now explicitly mode-dispatched (build_stage9_baseline_gate_report /
# ensure_stage9_baseline_gate_passes), never silently skipped in either mode:
#   - FULL (D_map N=50): UNCHANGED exact-equality check against STAGE8_AUTHORITATIVE_BASELINE,
#     PLUS (new) an exact match of the live D_map subset hashes against
#     STAGE8_AUTHORITATIVE_SUBSET_HASHES -- Stage 9's full run must reuse Stage 8's own frozen
#     N=50 manifests, never a re-shuffled/re-sampled subset. Either mismatch remains a HARD STOP.
#   - SMOKE (D_map N=5): a sample-size-appropriate REPEATABILITY gate instead -- reuses Stage
#     8's own already-proven run_baseline_repeatability_preflight_rpc/ensure_baseline_
#     repeatability BY IDENTITY (two full theta_0 resets + full encoder-cache resets +
#     bounded-generation evaluation passes, exact score/generation_hash/parsed_prediction_hash
#     equality required between the two passes) -- never compares the resulting N=5 score
#     numerically to the N=50 Stage-8 baseline at all, architecturally: the function's own
#     signature never receives STAGE8_AUTHORITATIVE_BASELINE as an input.
# =============================================================================================


def run_stage9_baseline_equality_check(baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
    """FULL MODE ONLY. Exact-equality check against the authoritative Stage-8 N=50 baseline --
    UNCHANGED from before this repair pass. Never call this against a smoke (N=5) baseline.
    """
    report: Dict[str, Any] = {}
    for capability, expected in STAGE8_AUTHORITATIVE_BASELINE.items():
        live_score = baseline_scores.get("capabilities", {}).get(capability, {}).get("score")
        report[capability] = {"expected_stage8_baseline": expected, "live_stage9_baseline": live_score, "matches": live_score == expected}
    report["all_match"] = all(row["matches"] for row in report.values() if isinstance(row, dict))
    return report


def ensure_stage9_baseline_matches_stage8(report: Dict[str, Any]) -> None:
    """FULL MODE ONLY hard stop -- UNCHANGED from before this repair pass."""
    mismatched = {cap: row for cap, row in report.items() if isinstance(row, dict) and not row["matches"]}
    if mismatched:
        raise Stage9BaselineMismatchError(
            f"Stage-9 live baseline does not match the authoritative Stage-8 baseline for "
            f"{len(mismatched)} capability(ies): {mismatched}. Refusing to start the Stage-9 "
            f"hierarchical sweep -- never averaged, never silently replaced."
        )


def run_stage9_subset_hash_check(capability_contexts: Dict[str, CapabilityContext]) -> Dict[str, Any]:
    """FULL MODE ONLY: confirms the live D_map N=50 subset hashes exactly match Stage 8's own
    frozen manifests (Section 1 of the spec: "require the Stage-9 full D_map manifests/subset
    hashes to be the same authoritative Stage-8 N=50 manifests").
    """
    report: Dict[str, Any] = {}
    for capability, expected in STAGE8_AUTHORITATIVE_SUBSET_HASHES.items():
        ctx = capability_contexts.get(capability)
        live_hash = ctx.subset_hash if ctx is not None else None
        report[capability] = {"expected_stage8_subset_hash": expected, "live_subset_hash": live_hash, "matches": live_hash == expected}
    report["all_match"] = all(row["matches"] for row in report.values() if isinstance(row, dict))
    return report


def ensure_stage9_subset_hashes_match_stage8(report: Dict[str, Any]) -> None:
    """FULL MODE ONLY hard stop."""
    mismatched = {cap: row for cap, row in report.items() if isinstance(row, dict) and not row["matches"]}
    if mismatched:
        raise Stage9SubsetHashMismatchError(
            f"Stage-9 live D_map subset hash does not match the authoritative Stage-8 N=50 "
            f"subset hash for {len(mismatched)} capability(ies): {mismatched}. Stage 9's full "
            f"run must reuse Stage 8's exact frozen D_map manifests -- refusing to proceed with "
            f"a re-shuffled/re-sampled subset."
        )


BASELINE_GATE_MODE_FULL = "stage8_full_exact_equality"
BASELINE_GATE_MODE_SMOKE = "smoke_n5_repeatability"


def build_stage9_baseline_gate_report(
    *, is_smoke: bool, d_map_n: int,
    full_equality_report: Optional[Dict[str, Any]] = None, full_subset_hash_report: Optional[Dict[str, Any]] = None,
    smoke_repeatability_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mode-explicit dispatch (Section 4 of the spec) -- persists `baseline_gate_mode` so a
    reader of stage9_baseline_gate.json always knows which gate ran, never has to infer it.
    """
    if is_smoke:
        if smoke_repeatability_report is None:
            raise ValueError("smoke_repeatability_report is required when is_smoke=True")
        return {"baseline_gate_mode": BASELINE_GATE_MODE_SMOKE, "d_map_n": d_map_n, "capability_repeatability": smoke_repeatability_report}
    if full_equality_report is None or full_subset_hash_report is None:
        raise ValueError("full_equality_report and full_subset_hash_report are both required when is_smoke=False")
    return {
        "baseline_gate_mode": BASELINE_GATE_MODE_FULL, "d_map_n": d_map_n,
        "baseline_equality": full_equality_report, "subset_hash_equality": full_subset_hash_report,
    }


def ensure_stage9_baseline_gate_passes(gate_report: Dict[str, Any]) -> None:
    """Single dispatch point -- never silently skips validation: an unrecognized
    baseline_gate_mode is itself a hard failure, not a silent no-op.
    """
    mode = gate_report.get("baseline_gate_mode")
    if mode == BASELINE_GATE_MODE_SMOKE:
        ensure_baseline_repeatability(gate_report["capability_repeatability"])
    elif mode == BASELINE_GATE_MODE_FULL:
        ensure_stage9_baseline_matches_stage8(gate_report["baseline_equality"])
        ensure_stage9_subset_hashes_match_stage8(gate_report["subset_hash_equality"])
    else:
        raise ValueError(f"Unknown baseline_gate_mode {mode!r} -- refusing to silently skip baseline validation.")


# =============================================================================================
# Per-candidate lifecycle
# =============================================================================================


def _is_ray_unrecoverable_error(exc: BaseException) -> bool:
    """True if `exc` indicates the Ray actor/engine is no longer usable (actor death, node-
    level OOM kill) -- in which case ANY further RPC (restoration, cache reset) would itself
    raise a confusing SECONDARY exception against a dead actor, masking the real root cause
    (this is exactly the live failure pattern this repair pass fixes: a primary Ray OOM
    followed by a second RayActorError from reset_to_base_weights_via_rpc). Lazily imports ray
    (never at module scope, matching this project's existing convention) so this stays
    importable/testable without ray installed -- returns False, never raises, when ray isn't
    available or the exception type can't be resolved.
    """
    try:
        import ray
    except ImportError:
        return False
    candidate_types = tuple(
        t for t in (
            getattr(ray.exceptions, "RayActorError", None),
            getattr(ray.exceptions, "OutOfMemoryError", None),
            getattr(ray.exceptions, "ActorDiedError", None),
            getattr(ray.exceptions, "RayTaskError", None),
        ) if t is not None
    )
    return bool(candidate_types) and isinstance(exc, candidate_types)


def evaluate_one_stage9_candidate_rpc(
    engine: Any, assignment: Stage9DirectionAssignment, child_region_param_names: Sequence[str],
    capability_contexts: Dict[str, CapabilityContext], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable] = None, generation_batch_size: int = STAGE9_GENERATE_BATCH_SIZE,
    rss_checkpoint: Optional[Callable[[str], None]] = None,
) -> List[ExperimentResultRecord]:
    """Identical lifecycle to Stage 8's evaluate_one_stage8_candidate_rpc (same v3
    quantization-aware radius acceptance, same twice-per-candidate cache reset, same bounded
    generation path), generalized to perturb a Stage-9 CHILD region instead of an L1 anatomy,
    and to persist direction-family identity keyed by child region.

    THIS REPAIR PASS (driver-RSS OOM during full candidate evaluation, ~101 GB, 1152 candidates
    -- exactly 2x Stage 8's 576): each capability's RunResult is now explicitly `del`eted and
    mem_telemetry.release_transient_memory() called immediately after the values needed for the
    result row are extracted (Section 6 of the spec) -- fixes ownership/retention FIRST, gc/
    malloc_trim second, never gc alone. `rss_checkpoint` (default None, zero overhead for every
    caller that doesn't pass it) fires at "before_candidate", "after_perturbation_applied", and
    "after_capability_<name>" for each of the six capabilities (Section 3's points A/B/C) --
    D/E/F/G are the caller's (run_stage9_rpc's) own responsibility, since checkpoint-append and
    candidate-to-candidate cleanup happen outside this function's scope.

    FAILURE PATH (Section 8): if an exception during perturb/evaluate indicates the Ray engine
    itself is no longer usable (_is_ray_unrecoverable_error), NO further RPC is attempted
    against the dead actor -- the exception propagates immediately, with zero rows appended
    (an interrupted candidate is never checkpointed as complete, by construction). Any OTHER
    exception (e.g. a radius-correction failure against a still-live engine) still gets the
    existing best-effort restoration attempt before propagating, exactly as before this pass --
    restoration is never skipped during a live, recoverable engine.
    """
    manifest = assignment.manifest
    if manifest.perturbation_mode != PERTURBATION_MODE:
        raise ValueError(f"Stage 9 only evaluates {PERTURBATION_MODE!r} manifests, got {manifest.perturbation_mode!r}")

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
            args=(manifest.seed, manifest.radius, manifest.anatomy_region, tuple(child_region_param_names)),
            label="scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3", ray_get=ray_get,
        )
        realized_r = apply_result["realized_relative_l2"]
        acceptance_mode = apply_result["radius_acceptance_mode"]
        if acceptance_mode == "strict":
            if abs(realized_r - manifest.radius) > REALIZED_RADIUS_TOLERANCE:
                raise RealizedRadiusMismatchError(
                    f"Perturbation {manifest.perturbation_id!r} (child_region={manifest.anatomy_region!r}, "
                    f"requested radius={manifest.radius}): strict-mode realized relative-L2 {realized_r} "
                    f"still differs by more than {REALIZED_RADIUS_TOLERANCE}."
                )
        elif acceptance_mode == "quantization_limited":
            if apply_result["relative_radius_error"] > QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE:
                raise RealizedRadiusMismatchError(
                    f"Perturbation {manifest.perturbation_id!r} (child_region={manifest.anatomy_region!r}, "
                    f"requested radius={manifest.radius}): quantization-limited relative error "
                    f"{apply_result['relative_radius_error']} still exceeds {QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE}."
                )
        else:
            raise RealizedRadiusMismatchError(f"Unknown radius_acceptance_mode {acceptance_mode!r} for perturbation {manifest.perturbation_id!r}.")

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
                    "cache_reset_before_evaluation": True,
                    "cache_reset_after_restoration": False,  # flipped True below, only on success
                    "child_region": assignment.child_region, "direction_family_id": assignment.direction_family_id,
                    "direction_seed": assignment.direction_seed, "direction_index": assignment.direction_index,
                    "generation_batch_size": generation_batch_size,
                },
            ))
            del result
            release_transient_memory()
            _checkpoint(f"after_capability_{capability}")
    except Exception as exc:
        if _is_ray_unrecoverable_error(exc):
            # Engine/actors are dead -- do NOT issue further RPC calls against them (this is
            # exactly the live secondary-crash pattern this repair pass fixes). Zero rows have
            # been appended to results.jsonl for this candidate, so it is already, correctly,
            # incomplete -- propagate the ORIGINAL exception cleanly, never masked by a second one.
            raise
        # Any other (recoverable-engine) failure: still attempt best-effort restoration before
        # propagating, exactly as before this repair pass -- restoration is never skipped while
        # the engine itself is still usable.
        reset_to_base_weights_via_rpc(engine, ray_get=ray_get)
        raise

    reset_to_base_weights_via_rpc(engine, ray_get=ray_get)
    verification = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Exact fixed-base restoration failed after Stage-9 candidate {manifest.perturbation_id!r} "
            f"(child_region={manifest.anatomy_region!r}, radius={manifest.radius}, seed={manifest.seed}): "
            f"max_abs_drift={verification['max_abs_drift']}"
        )

    reset_vllm_encoder_cache_full(engine)
    for record in records:
        record.runtime_metadata["cache_reset_after_restoration"] = True

    return records


class RealizedRadiusMismatchError(RuntimeError):
    """Defensive, mode-aware re-check on an already-accepted radius-realization result --
    should never actually fire (scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3
    itself already guarantees this bound or raises), kept as a second, independent layer.
    """


def _write_candidate_telemetry_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def run_stage9_rpc(
    plan: Stage9Plan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, seed_bank: Dict[str, Tuple[int, ...]], child_region_param_names_by_region: Dict[str, Sequence[str]],
    child_mask_hash_by_region: Dict[str, str], audits: Dict[str, Stage9PartitionAudit], *, run_benchmark: Callable, ray_get: Optional[Callable] = None,
) -> int:
    """A candidate is checkpointed ONLY after its full apply -> evaluate(all 6 capabilities) ->
    reset -> verify cycle has already succeeded inside evaluate_one_stage9_candidate_rpc.

    THIS REPAIR PASS (driver-RSS OOM during full candidate evaluation): NO LONGER accumulates
    every candidate's ExperimentResultRecord objects in memory for the whole 1152-candidate
    sweep (Section 2 of the spec's "preferred design") -- neither the already-checkpointed rows
    (previously re-loaded into a full in-memory list every resume, growing with resume depth)
    nor the newly-evaluated ones. results.jsonl is already the durable, resumable source of
    truth (write_stage9_run_manifest re-reads the authoritative total from it independently);
    this function now returns only the COUNT of rows newly appended this call. Per-candidate RSS
    telemetry (Section 3) is written to its OWN file (candidate_memory_telemetry.jsonl),
    NEVER embedded into the scientific results.jsonl rows -- see the module-level docstring
    section on driver-RSS OOM root-causing for the full context.
    """
    from .mem_telemetry import release_transient_memory, rss_mb

    population_by_cell = build_stage9_population(plan, seed_bank, child_mask_hash_by_region)
    validate_stage9_direction_seed_reuse(plan, population_by_cell)

    current_checkpoint = build_stage9_checkpoint_manifest(plan, capability_contexts, child_mask_hash_by_region, seed_bank, audits)
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    ensure_stage9_checkpoint_manifest(checkpoint_path, current_checkpoint)

    results_path = plan.output_dir / "results.jsonl"
    telemetry_path = plan.output_dir / "candidate_memory_telemetry.jsonl"
    # Only the SET of already-completed perturbation IDs is retained going forward -- never
    # their full row contents (the previous version's `all_records.extend(rows) for rows in
    # completed.values()` retained every already-checkpointed candidate's records too, growing
    # with resume depth on top of the newly-evaluated ones).
    completed_ids = set(load_completed_perturbation_rows(results_path, plan.capabilities).keys())

    newly_completed_rows = 0
    perturbation_index = 0
    previous_candidate_rss_mb: Optional[float] = None

    for (region, radius), assignments in population_by_cell.items():
        region_param_names = child_region_param_names_by_region[region]
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

            records = evaluate_one_stage9_candidate_rpc(
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
                "child_region": region, "radius": radius, "direction_index": assignment.direction_index,
                "rss_start_mb": rss_start_mb, "rss_after_perturbation_mb": rss_after_perturbation_mb,
                "rss_after_capability_mb": capability_rss_mb, "rss_after_checkpoint_mb": rss_after_checkpoint_mb,
                "rss_after_cleanup_mb": rss_after_cleanup_mb,
                "delta_from_previous_candidate_mb": (
                    (rss_after_cleanup_mb - previous_candidate_rss_mb) if previous_candidate_rss_mb is not None else 0.0
                ),
                "high_water_mb": high_water_mb,
            })
            previous_candidate_rss_mb = rss_after_cleanup_mb
            perturbation_index += 1

    return newly_completed_rows


# =============================================================================================
# CLI entry point
# =============================================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "stage9_hierarchical_anatomical_atlas"))
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"tiny live GPU smoke: 6 child regions x 3 radii x 1 direction family x 6 capabilities x "
             f"{STAGE9_SMOKE_D_MAP_N} D_map examples/capability = 18 perturbations, 108 rows, 540 "
             f"perturbed model-example evaluations -- execution size only, same scientific protocol.",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU")
    args = parser.parse_args(argv)

    from .config import load_config

    cfg = load_config(args.config)

    if args.smoke:
        plan = build_stage9_smoke_plan(model_name=cfg.model.name, model_revision=cfg.model.revision, output_root=args.output_root)
    else:
        plan = build_stage9_plan(model_name=cfg.model.name, model_revision=cfg.model.revision, output_root=args.output_root)

    print(f"Stage 9 plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}): child_regions={plan.child_regions} radii={plan.radii}")
    print(f"capabilities={plan.capabilities}")
    print(f"n_directions_per_cell={plan.n_directions_per_cell} d_map_n={plan.d_map_n} total_unique_perturbations={plan.total_unique_perturbations}")
    print(f"total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations}")
    print(f"total_perturbed_model_example_evaluations={plan.total_perturbed_model_example_evaluations}")
    print(f"radius_realization_method={plan.radius_realization_method}")
    print(f"multimodal_cache_policy={plan.multimodal_cache_policy}")
    print(f"enable_prefix_caching={plan.enable_prefix_caching}")
    print(f"generation_batch_size={plan.generation_batch_size}")
    print(f"stage8_parent_run_signature={STAGE8_PARENT_RUN_SIGNATURE}")
    print(f"output_dir={plan.output_dir}")
    baseline_gate_description = (
        f"SMOKE baseline gate: D_map N={plan.d_map_n} two-pass theta_0 repeatability (score/"
        f"generation_hash/parsed_prediction_hash), never compared to the Stage-8 N=50 baseline"
        if plan.is_smoke else
        "FULL baseline gate: exact equality + subset-hash match against the authoritative Stage-8 N=50 baseline"
    )
    print(f"baseline_gate_mode={'smoke_n5_repeatability' if plan.is_smoke else 'stage8_full_exact_equality'} ({baseline_gate_description})")
    print(
        "Lifecycle: mode-aware baseline gate (see above) -> for each candidate: "
        "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 (child region) -> full "
        "encoder-cache reset -> evaluate all 6 capabilities (bounded generation microbatches) -> "
        "reset_to_base_weights -> verify exact restoration -> full encoder-cache reset again -> "
        "checkpoint candidate rows."
    )

    if args.dry_run:
        return 0

    try:
        assert_feasible(
            f"Stage 9 hierarchical anatomical atlas ({plan.run_signature})",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision")
    model_resolution = resolve_and_report_model_snapshot(plan.model_name, plan.model_revision)
    engine_config = build_stage7b_engine_config()
    assert engine_config["enable_prefix_caching"] is False, "Stage 9 must never run with prefix caching enabled."
    print(format_runtime_compatibility_diagnostic(
        model_resolution, worker_extension_cls="utils.worker_extn.WorkerExtension",
        vllm_version=get_vllm_version(), engine_mode=detect_vllm_engine_mode(), engine_config=engine_config,
    ))

    from .benchmarks.runner import run_benchmark
    from .config import load_capability_benchmark_config
    from .run_capability_benchmark_gate import load_adapter
    from .run_stage8_coarse_anatomical_atlas import STAGE8_BASE_SEED
    from .thicket.data_roles import write_data_role_manifest
    from .vlm_adapter import bootstrap_ray, verify_workers_can_import_external_root

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    write_numerical_solver_patch_provenance(plan.output_dir)
    subset_ids_dir = plan.output_dir / "d_map_subsets"
    capability_contexts = build_d_map_capability_contexts(
        STAGE8_BASE_SEED, subset_ids_dir, plan.d_map_n,  # Stage 8's OWN base seed -- reuses Stage 8's subset hashes exactly, never resamples
        load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
    )
    for capability, ctx in capability_contexts.items():
        write_data_role_manifest(ctx.partition, plan.output_dir / "data_roles" / f"{capability}_d_map.json")

    seed_bank = build_stage9_direction_seed_bank(STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    (plan.output_dir / "direction_family_manifest.json").write_text(json.dumps(
        {"child_regions": list(plan.child_regions), "n_directions_per_cell": plan.n_directions_per_cell,
         "seed_bank": {r: list(s) for r, s in seed_bank.items()},
         "direction_seed_bank_hash": compute_direction_seed_bank_hash(seed_bank)}, indent=2,
    ))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_resolution["resolved_snapshot_path"])

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
            model_resolution["resolved_snapshot_path"], precision=engine_config["precision"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"], max_model_len=engine_config["max_model_len"],
            tensor_parallel_size=engine_config["tensor_parallel_size"], enable_prefix_caching=engine_config["enable_prefix_caching"],
        )
        engine = engines[0]

        store_base_weights_via_rpc(engine)
        print(format_base_snapshot_confirmation(engine_config["gpu_memory_utilization"], engine_config["base_snapshot_mode"]))

        ensure_encoder_cache_reset_available(engine)
        print(f"Confirmed working multimodal-encoder-cache reset (multimodal_cache_policy={plan.multimodal_cache_policy!r}).")

        child_info = _collective_rpc_single_worker(engine, report_stage9_child_param_names, args=(plan.child_regions,), label="report_stage9_child_param_names")
        child_region_param_names_by_region = {r: tuple(info["param_names"]) for r, info in child_info["regions"].items()}
        child_mask_hash_by_region = {r: info["mask_hash"] for r, info in child_info["regions"].items()}
        (plan.output_dir / "partition_audit.json").write_text(json.dumps(child_info["audits"], indent=2))
        for parent, audit in child_info["audits"].items():
            if not audit["union_equals_parent"] or not audit["children_pairwise_disjoint"]:
                raise RuntimeError(f"Live Stage-9 partition audit failed for parent {parent!r}: {audit}")
        print("Confirmed: live Stage-9 hierarchical partition audit passed (union==parent, pairwise disjoint) for vision and language.")

        llm_adapter = RayEngineLLMAdapter(engine)
        from .run_global_visual_thicket_pilot import load_or_compute_baseline_scores
        baseline_path = plan.output_dir / "baseline_scores.json"
        load_or_compute_baseline_scores(baseline_path, capability_contexts, plan.model_revision, plan.run_signature, llm_adapter, tokenizer, sampling_params)
        baseline_scores = json.loads(baseline_path.read_text())

        # MODE-AWARE baseline gate (this repair pass -- see this section's own module-level
        # docstring above run_stage9_baseline_equality_check for the full root-cause writeup):
        # FULL mode keeps the exact Stage-8 N=50 equality + subset-hash checks UNCHANGED; SMOKE
        # mode (D_map N=5) never compares its score numerically to the N=50 Stage-8 baseline at
        # all -- it runs a sample-size-appropriate two-pass repeatability check instead, reusing
        # Stage 8's own already-proven run_baseline_repeatability_preflight_rpc BY IDENTITY.
        if plan.is_smoke:
            print(f"Running Stage-9 SMOKE baseline gate (D_map N={plan.d_map_n} two-pass repeatability -- never compared to the Stage-8 N=50 baseline)...")
            smoke_repeatability_report = run_baseline_repeatability_preflight_rpc(
                engine, capability_contexts, tokenizer, sampling_params, run_benchmark=run_benchmark,
                generation_batch_size=plan.generation_batch_size,
            )
            gate_report = build_stage9_baseline_gate_report(
                is_smoke=True, d_map_n=plan.d_map_n, smoke_repeatability_report=smoke_repeatability_report,
            )
            (plan.output_dir / "stage9_baseline_gate.json").write_text(json.dumps(gate_report, indent=2))
            ensure_stage9_baseline_gate_passes(gate_report)
            print(f"Confirmed: Stage-9 smoke D_map N={plan.d_map_n} baseline is exactly repeatable "
                  f"(score/generation_hash/parsed_prediction_hash) across two theta_0 passes for all 6 capabilities.")
        else:
            print("Running Stage-9 FULL baseline gate (exact equality + subset-hash match against the authoritative Stage-8 N=50 baseline)...")
            full_equality_report = run_stage9_baseline_equality_check(baseline_scores)
            full_subset_hash_report = run_stage9_subset_hash_check(capability_contexts)
            gate_report = build_stage9_baseline_gate_report(
                is_smoke=False, d_map_n=plan.d_map_n,
                full_equality_report=full_equality_report, full_subset_hash_report=full_subset_hash_report,
            )
            (plan.output_dir / "stage9_baseline_gate.json").write_text(json.dumps(gate_report, indent=2))
            ensure_stage9_baseline_gate_passes(gate_report)
            print("Confirmed: live Stage-9 baseline and D_map subset hashes match the authoritative Stage-8 N=50 manifests for all 6 capabilities.")

        newly_written_rows = run_stage9_rpc(
            plan, capability_contexts, engine, tokenizer, sampling_params, seed_bank,
            child_region_param_names_by_region, child_mask_hash_by_region,
            {k: Stage9PartitionAudit(**v) for k, v in child_info["audits"].items()},
            run_benchmark=run_benchmark,
        )
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    # The authoritative row/perturbation totals are always read back from results.jsonl itself
    # (write_stage9_run_manifest), never accumulated in memory for the whole sweep -- see
    # run_stage9_rpc's own docstring for why (this repair pass's driver-RSS OOM fix).
    manifest = write_stage9_run_manifest(plan.output_dir)
    print(f"Wrote {newly_written_rows} NEW result rows this run to {plan.output_dir / 'results.jsonl'}")
    print(f"Run manifest: {json.dumps(manifest, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
