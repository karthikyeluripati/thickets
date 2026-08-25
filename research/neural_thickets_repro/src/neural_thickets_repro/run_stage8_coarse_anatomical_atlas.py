"""Stage 8: paper-scale coarse anatomical atlas. L1 anatomy (vision / multimodal_connector_or_
merger / language) x 3 frozen common radii x 6 frozen visual capabilities, at the common radii
already calibrated and approved by Stage 7B (`stage8_radius_final_recommendation.json`,
proceed_to_stage8=true) and confirmed by the Stage-6 cache-safe reproduction
(`stage6_global_gaussian_upstream_cache_safe_v2`). This module answers ONE question: is
P(improvement | capability, anatomy, radius) measurably non-uniform? It explicitly does NOT
select a "best" region/radius, does NOT claim "experts live in region X" (that requires D_confirm,
not built here), and does NOT drill into L2 depth bands / attention-vs-MLP / individual heads --
see the module-level "DO NOT DO YET" list at the bottom of this docstring.

=================================================================================================
REUSE, BY IDENTITY, FROM STAGE 7B (this repair pass changes NONE of these -- see
run_stage7b_anatomical_calibration.py's own docstring for the full derivation of each):
=================================================================================================
- perturbation_mode=anatomical_relative_l2, dispatched via
  scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 -- the EXACT same
  quantization-aware v3 two-tier (strict <=1e-6, else quantization-limited <=0.1% relative,
  only under a PROVEN plateau) radius-acceptance rule. NO numerical tolerance is touched here.
- multimodal_cache_policy=full_encoder_reset_vllm011_verified_v2 / enable_prefix_caching=False --
  the identical verified cache-safe lifecycle (reset BEFORE capability evaluation and AGAIN after
  restoration is verified, unconditionally, never relying on candidate ordering).
- launch_stage6_engine / build_stage7b_engine_config (bfloat16, max_model_len=4096,
  gpu_memory_utilization=0.60, TP=1, enable_prefix_caching=False).
- fixed-base restoration discipline (store_base_weights once; reset_to_base_weights_via_rpc +
  verify_exact_fixed_base_restoration_via_rpc every candidate).
- FULL_CALIBRATION_REGIONS (imported BY IDENTITY, so region NAMES/masks are byte-identical to
  Stage 7A/7B -- "use the exact existing Stage-7A mask definitions and hashes" is satisfied by
  construction: region masks are derived live from the model's own named_parameters() via
  thicket.anatomy.build_anatomy_atlas, the SAME call Stage 7B makes, never a hardcoded hash).

=================================================================================================
WHAT IS NEW IN STAGE 8 (this repair pass)
=================================================================================================
1. THREE FROZEN COMMON RADII (not Stage 7B's six) -- STAGE8_RADII below is asserted, at import
   time, to equal exactly (FULL_CALIBRATION_RADII[0], FULL_CALIBRATION_RADII[1],
   FULL_CALIBRATION_RADII[3]) from run_stage7b_anatomical_calibration -- i.e. literally the same
   three float values recommended in stage8_radius_final_recommendation.json (R_small, R_mid,
   R_transition), never retyped as independent literals that could silently drift.

2. SIX FROZEN CAPABILITIES (visual_grounding, counting, spatial_reasoning,
   ocr_text_recognition_grounded, relational_reasoning, fine_grained_recognition) -- the 3 new
   entries (counting/relational_reasoning/fine_grained_recognition) were added, purely
   additively, to run_global_visual_thicket_pilot.CAPABILITY_CONFIG_FILES this same repair pass;
   every existing Stage 6/7B caller still only ever looks up its own frozen 3-key subset.

3. DIRECTION-FAMILY REUSE ACROSS RADII (the core new scientific-design requirement): Stage 7B's
   own generate_perturbation_population() derives its per-perturbation seed as a function of
   (mode, region, RADIUS, sigma, i) -- i.e. each (region, radius) cell gets an INDEPENDENTLY
   sampled direction, by design (Stage 7B never needed radius trajectories). Stage 8 instead
   needs the SAME fixed Gaussian direction, scaled to three different radii, so per-direction
   radius trajectories (Delta(R_small) -> Delta(R_mid) -> Delta(R_transition)) are meaningful.
   build_stage8_direction_seed_bank() derives exactly STAGE8_N_DIRECTIONS_PER_CELL seeds PER
   REGION as a function of (region, i) only -- deliberately NOT a function of radius -- via
   thicket.seeds.derive_seed(base_seed, "stage8_direction_family", region, str(i)).
   build_stage8_population() then constructs PerturbationManifest objects DIRECTLY (bypassing
   generate_perturbation_population entirely) reusing that SAME seed value for direction i across
   all 3 STAGE8_RADII within one region -- apply_anatomical_relative_l2's own noise generation
   (_generate_noise(p, seed), a pure function of seed alone, never of r) guarantees this produces
   the IDENTICAL normalized direction, only rescaled to the requested radius; each of the 3
   resulting PerturbationManifests still gets a DISTINCT perturbation_id (compute_perturbation_id
   hashes radius too), so this is 64 directions x 3 radii = 192 unique perturbations PER REGION,
   576 total -- never 64 total. Because seeds are (deliberately) reused 3x within a region, this
   population must NOT be validated with thicket.perturbation.validate_unique_worker_seeds (which
   assumes global seed uniqueness and would reject the reuse Stage 8 requires by design) --
   validate_stage8_direction_seed_reuse() below checks the CORRECT invariant instead: unique
   perturbation_ids (576), and per region, exactly STAGE8_N_DIRECTIONS_PER_CELL distinct seeds
   each appearing exactly len(STAGE8_RADII) times.
   Across REGIONS, the same numeric seed integer MAY coincidentally recur (region name is folded
   into the derive_seed() call, so in practice it will not, but nothing enforces this) -- per
   explicit instruction, region-a-seed-i and region-b-seed-i must NEVER be interpreted as "the
   same geometric direction": they live in disjoint parameter subspaces by construction (a
   region's noise is only ever sampled over that region's own parameter names). `region` is
   always persisted alongside `direction_seed`/`direction_index` for exactly this reason.

4. BASELINE REPEATABILITY PREFLIGHT (entirely new, Part 4 of the task spec): BEFORE any of the
   576 perturbations are evaluated, run_baseline_repeatability_preflight_rpc() evaluates each of
   the 6 capabilities' D_map subset TWICE against theta_0 (reset_to_base_weights_via_rpc +
   reset_vllm_encoder_cache_full before EACH pass, enable_prefix_caching already False for the
   whole run) and compares primary_metric score, benchmarks.runner.RunResult.generation_hash()
   (the per-example-result hash), and parsed_prediction_hash() across the two passes. ANY
   capability whose two passes disagree on ANY of the three raises
   BaselineNondeterminismError BEFORE the main population loop starts -- this is a hard
   pre-experiment gate, never a soft warning, and never silently averaged. This is Stage 8's own
   concern; it does not touch or resume Stage 6/7B's own baseline_scores.json files.

5. SIX-CAPABILITY CHECKPOINT IDENTITY (Stage8CheckpointManifest): mirrors
   Stage7bCheckpointManifest's discipline exactly, plus a `direction_seed_bank_hash` field (a
   sha256 over the canonical JSON of the FULL {region: [seed, ...]} bank) so a resume attempt
   against a DIFFERENT seed bank (e.g. a different base_seed) is rejected exactly like a
   model-revision or capability-hash mismatch would be -- "hard fail if ... seed bank [differs]"
   from the task spec, satisfied by including the bank's own hash in the compared identity.

=================================================================================================
FROZEN FULL Stage-8 CONFIG (do not re-derive; matches Section 1/2/3/5/6/9 of the approved spec)
=================================================================================================
model:        Qwen/Qwen2.5-VL-3B-Instruct @ 66285546d2b821cf421d4f5eb2576359d3770cd3, bf16
regions:      vision, multimodal_connector_or_merger, language (STAGE8_REGIONS ==
              run_stage7b_anatomical_calibration.FULL_CALIBRATION_REGIONS, by identity)
radii:        STAGE8_RADII == (FULL_CALIBRATION_RADII[0], [1], [3]) == the 3 radii FROZEN by
              stage8_radius_final_recommendation.json (R_small, R_mid, R_transition); the two
              calibration-scale-destructive radii (0.1784941427189971, 0.3569882854379942) are
              NEVER included here.
capabilities: visual_grounding, counting, spatial_reasoning, ocr_text_recognition_grounded,
              relational_reasoning, fine_grained_recognition (STAGE8_CAPABILITIES, fixed order)
directions:   STAGE8_N_DIRECTIONS_PER_CELL (64) per (region, radius) cell, SAME direction seed
              reused across the 3 radii within one region (see point 3 above)
data:         a deterministic STAGE8_D_MAP_N (50)-example D_map subset per capability
Total:        3 regions x 3 radii x 64 directions = 576 unique perturbations
              576 x 6 capabilities = 3456 perturbation x capability result rows
              3456 x 50 = 172,800 perturbed model-example evaluations
No D_confirm/D_select/D_test is ever constructed or referenced anywhere in this module.

=================================================================================================
SMOKE MODE -- EXECUTION SIZE ONLY, same scientific definitions
=================================================================================================
--smoke overrides ONLY population size: all 3 STAGE8_REGIONS, all 3 STAGE8_RADII, but only
STAGE8_SMOKE_N_DIRECTIONS (1) direction family, all 6 frozen capabilities, STAGE8_SMOKE_D_MAP_N
(5) D_map examples/capability. Expected: 3 regions x 3 radii x 1 direction = 9 unique
perturbations, 9 x 6 = 54 result rows, 54 x 5 = 270 perturbed model-example evaluations. This
validates the FULL structural path (direction-family reuse across radii, baseline-repeatability
preflight, cache-safe lifecycle, checkpoint/resume) end-to-end on real (tiny) GPU work before any
commitment to the 576-candidate run; it does NOT require behavioral improvement as a pass
criterion.

=================================================================================================
DO NOT DO YET (explicitly out of scope for this module, per the approved Stage-8 spec)
=================================================================================================
L2 early/mid/late language drill-down, attention-vs-MLP, individual heads, low-rank geometry,
7B/72B or frontier models, post-training, routing, distillation, synthetic emergence, and any
D_confirm-based final "experts live in X" claim. Stage 8 is ONLY the paper-scale L1
anatomy x capability x radius atlas.
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
    FULL_CALIBRATION_REGIONS,
    MULTIMODAL_CACHE_POLICY,
    PERTURBATION_MODE,
    REALIZED_RADIUS_TOLERANCE,
    RADIUS_REALIZATION_METHOD,
    EncoderCacheResetUnavailableError,
    RealizedRadiusMismatchError,
    build_stage7b_engine_config,
    ensure_encoder_cache_reset_available,
    ensure_stage7b_encoder_cache_reset_mechanism_exposed,
    report_region_param_names,
)
from .run_stage7b_anatomical_calibration import FULL_CALIBRATION_RADII as _STAGE7B_FULL_CALIBRATION_RADII
from .scoped_anatomical_perturbation import (
    QUANTIZATION_AWARE_METHOD_V3,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    CorrectionOutOfRegionDriftError,
    QuantizationToleranceExceededError,
    RadiusCorrectionFailedError,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
)
from .thicket.perturbation import PERTURBATION_MODES, PerturbationManifest
from .thicket.schema import ExperimentResultRecord
from .thicket.seeds import derive_seed
from .vlm_adapter import ensure_full_encoder_cache_reset_exposed, reset_vllm_encoder_cache_full

assert PERTURBATION_MODE in PERTURBATION_MODES
assert RADIUS_REALIZATION_METHOD == QUANTIZATION_AWARE_METHOD_V3  # never a different/looser method

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

EXPERIMENT_ID = "stage8_coarse_anatomical_atlas"
STAGE8_BASE_SEED = 20260825  # distinct from Stage 6's 20260823 and Stage 7B's 20260824

# --- FROZEN full-Stage-8 config -------------------------------------------------------------
STAGE8_REGIONS: Tuple[str, ...] = FULL_CALIBRATION_REGIONS  # byte-identical Stage-7A region set
STAGE8_RADII: Tuple[float, ...] = (
    _STAGE7B_FULL_CALIBRATION_RADII[0], _STAGE7B_FULL_CALIBRATION_RADII[1], _STAGE7B_FULL_CALIBRATION_RADII[3],
)
assert STAGE8_RADII == (0.0035698828543799426, 0.017849414271899712, 0.07139765708759885), (
    "STAGE8_RADII drifted from the frozen stage8_radius_final_recommendation.json common radii."
)
_STAGE8_EXCLUDED_DESTRUCTIVE_RADII: Tuple[float, ...] = (0.1784941427189971, 0.3569882854379942)
assert not (set(STAGE8_RADII) & set(_STAGE8_EXCLUDED_DESTRUCTIVE_RADII))

STAGE8_CAPABILITIES: Tuple[str, ...] = (
    "visual_grounding", "counting", "spatial_reasoning",
    "ocr_text_recognition_grounded", "relational_reasoning", "fine_grained_recognition",
)
STAGE8_N_DIRECTIONS_PER_CELL = 64
STAGE8_D_MAP_N = 50
DATASET_ROLE = "map"  # the ONLY role this module ever constructs or references

# --- SMOKE mode: EXECUTION SIZE ONLY, same scientific rules ----------------------------------
STAGE8_SMOKE_N_DIRECTIONS = 1
STAGE8_SMOKE_D_MAP_N = 5

_ALLOWED_D_MAP_SIZES: Tuple[int, ...] = (STAGE8_D_MAP_N, STAGE8_SMOKE_D_MAP_N)

# --- Bounded generation (this repair pass -- driver-RSS OOM audit) ---------------------------
# The full v1 attempt (stage8_coarse_anatomical_atlas_3b_v1) OOM'd during the N=50
# baseline-repeatability preflight: node RAM ~110.67/116.42 GB, DRIVER Python process RSS
# ~100.86 GB, while the RayWorkerWrapper/RandOptNcclLLM actors themselves were only ~2.18/1.88
# GB -- i.e. primarily driver/host RAM growth, not GPU VRAM. Code-inspection root cause (see
# this module's own docstring "DRIVER MEMORY AUDIT" section below): every Stage-8 capability
# adapter's load_examples() (benchmarks/adapters/*.py) decodes a PIL.Image for EVERY row of the
# ENTIRE underlying dataset/split BEFORE subset selection reduces it to D_map N=50 --
# HuggingFaceM4/the_cauldron's "tallyqa" config alone is ~98.7k rows. build_d_map_capability_
# contexts() calls this once per capability (6x), sequentially, before the baseline preflight
# even starts -- each call transiently materializes a full dataset's worth of decoded images,
# and glibc's allocator does not necessarily return that freed memory to the OS between calls,
# so driver RSS can ratchet upward across the 6 sequential builds even though each transient
# list is logically freed the moment it goes out of scope. STAGE8_GENERATE_BATCH_SIZE bounds
# the SEPARATE (and also contributing) per-capability llm.generate() driver-side request/output
# materialization; benchmarks/runner.run_benchmark's own default (None) is untouched, so
# Stage 6/7B's byte-identical single-call behavior is never affected.
STAGE8_GENERATE_BATCH_SIZE = 10

_FULL_RUN_SIGNATURE_V1_UNBATCHED = "stage8_coarse_anatomical_atlas_3b_v1"  # the OOM'd attempt's identity -- kept as permanent, never-resumed-into provenance
_FULL_RUN_SIGNATURE_V2_BATCHED = "stage8_coarse_anatomical_atlas_3b_v2_batched10"  # valid only when generation_batch_size == STAGE8_GENERATE_BATCH_SIZE


class DatasetRoleViolationError(RuntimeError):
    """Something other than the 'map' dataset role was requested, or an unrecognized D_map
    size was requested -- Stage 8 must never construct or reference D_confirm/D_select/D_test.
    """


class BaselineNondeterminismError(RuntimeError):
    """The baseline repeatability preflight (Section 4 of the spec) found at least one
    capability whose TWO independent theta_0 evaluations of the identical D_map subset
    disagreed on score, generation_hash, or parsed_prediction_hash. Hard stop BEFORE any of the
    576 Stage-8 perturbations are evaluated -- Stage 8 never silently averages baseline
    stochasticity, and never proceeds with a capability whose own zero-perturbation measurement
    is not reproducible.
    """


def compute_stage8_run_signature(
    regions: Sequence[str], radii: Sequence[float], n_directions: int, d_map_n: int,
    generation_batch_size: Optional[int] = None,
) -> str:
    """Batching is an EXECUTION/reproducibility parameter (this repair pass), folded into the
    signature exactly like radius_realization_method/multimodal_cache_policy are for Stage 7B --
    so a checkpoint written under a different generation_batch_size (including the OOM'd v1
    attempt's implicit "unbatched", generation_batch_size=None) can never be silently resumed
    under a different one.

    For the frozen full scientific config (regions/radii/n_directions/d_map_n exactly matching
    STAGE8_REGIONS/STAGE8_RADII/STAGE8_N_DIRECTIONS_PER_CELL/STAGE8_D_MAP_N):
        generation_batch_size=None                        -> _FULL_RUN_SIGNATURE_V1_UNBATCHED
        generation_batch_size==STAGE8_GENERATE_BATCH_SIZE  -> _FULL_RUN_SIGNATURE_V2_BATCHED
        any OTHER explicit batch size                      -> its own disjoint "..._batched{N}" label
    Otherwise (any smoke/diagnostic-size config): a deterministic "stage8_smoke_..." descriptive
    string built from the ACTUAL values, batch-size-suffixed when set -- so a failed/partial
    smoke run can never be resumed as (or mistaken for) the full run, and vice versa.
    """
    is_frozen_full_scientific_config = (
        tuple(regions) == STAGE8_REGIONS and tuple(radii) == STAGE8_RADII
        and n_directions == STAGE8_N_DIRECTIONS_PER_CELL and d_map_n == STAGE8_D_MAP_N
    )
    if is_frozen_full_scientific_config:
        if generation_batch_size is None:
            return _FULL_RUN_SIGNATURE_V1_UNBATCHED
        if generation_batch_size == STAGE8_GENERATE_BATCH_SIZE:
            return _FULL_RUN_SIGNATURE_V2_BATCHED
        return f"stage8_coarse_anatomical_atlas_3b_batched{generation_batch_size}"
    region_label = "-".join(regions)
    radius_label = "-".join(f"{r:.6f}".replace(".", "") for r in radii)
    batch_label = f"_batched{generation_batch_size}" if generation_batch_size is not None else ""
    return f"stage8_smoke_{region_label}_r{radius_label}_n{d_map_n}_dir{n_directions}{batch_label}"


# =============================================================================================
# Plan (pure arithmetic, no I/O, no GPU)
# =============================================================================================


@dataclass(frozen=True)
class Stage8Plan:
    model_name: str
    model_revision: str
    model_family: str
    model_scale: str
    regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_directions_per_cell: int
    d_map_n: int
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: bool
    run_signature: str
    output_dir: Path
    # Execution-only (this repair pass, driver-RSS OOM audit): None reproduces the OOM'd v1
    # attempt's exact unbatched llm.generate() call shape; STAGE8_GENERATE_BATCH_SIZE (10) is
    # what every NEW real Stage-8 execution (main()'s default, --baseline-memory-smoke) uses.
    # Never a scientific parameter -- see compute_stage8_run_signature's own docstring.
    generation_batch_size: Optional[int] = None

    @property
    def total_unique_perturbations(self) -> int:
        return len(self.regions) * len(self.radii) * self.n_directions_per_cell

    @property
    def total_perturbation_capability_evaluations(self) -> int:
        return self.total_unique_perturbations * len(self.capabilities)

    @property
    def total_perturbed_model_example_evaluations(self) -> int:
        return self.total_perturbation_capability_evaluations * self.d_map_n

    @property
    def is_smoke(self) -> bool:
        return not (
            self.regions == STAGE8_REGIONS and self.radii == STAGE8_RADII
            and self.n_directions_per_cell == STAGE8_N_DIRECTIONS_PER_CELL and self.d_map_n == STAGE8_D_MAP_N
        )


def build_stage8_plan(
    *, model_name: str, model_revision: str, output_root: "str | Path",
    regions: Sequence[str] = STAGE8_REGIONS, radii: Sequence[float] = STAGE8_RADII,
    n_directions_per_cell: int = STAGE8_N_DIRECTIONS_PER_CELL, d_map_n: int = STAGE8_D_MAP_N,
    model_family: str = "qwen2_5_vl", model_scale: str = "3B",
    generation_batch_size: Optional[int] = None,
) -> Stage8Plan:
    """`generation_batch_size` defaults to None (preserves every existing caller/test's prior
    behavior -- the frozen full config then resolves to the OOM'd v1 identity). Real Stage-8
    executions (main()'s default CLI path, --baseline-memory-smoke) pass
    generation_batch_size=STAGE8_GENERATE_BATCH_SIZE explicitly to get the v2_batched10 identity.
    """
    if not regions:
        raise ValueError("Stage 8 requires at least one anatomy region.")
    if not radii:
        raise ValueError("Stage 8 requires at least one common radius.")
    if set(radii) & set(_STAGE8_EXCLUDED_DESTRUCTIVE_RADII):
        raise ValueError(f"Stage 8 must never include the calibration-scale-destructive radii {_STAGE8_EXCLUDED_DESTRUCTIVE_RADII}.")
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 8 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    if generation_batch_size is not None and generation_batch_size <= 0:
        raise ValueError(f"generation_batch_size must be positive or None, got {generation_batch_size}")

    run_signature = compute_stage8_run_signature(regions, radii, n_directions_per_cell, d_map_n, generation_batch_size)
    return Stage8Plan(
        model_name=model_name, model_revision=model_revision, model_family=model_family, model_scale=model_scale,
        regions=tuple(regions), radii=tuple(radii), capabilities=STAGE8_CAPABILITIES,
        n_directions_per_cell=n_directions_per_cell, d_map_n=d_map_n,
        radius_realization_method=RADIUS_REALIZATION_METHOD, multimodal_cache_policy=MULTIMODAL_CACHE_POLICY,
        enable_prefix_caching=ENABLE_PREFIX_CACHING, run_signature=run_signature, output_dir=Path(output_root) / run_signature,
        generation_batch_size=generation_batch_size,
    )


def build_stage8_smoke_plan(
    *, model_name: str, model_revision: str, output_root: "str | Path", generation_batch_size: Optional[int] = None,
) -> Stage8Plan:
    """3 regions x 3 radii x 1 direction family x 6 capabilities x D_map N=5 = 9 perturbations,
    54 rows, 270 perturbed model-example evaluations.
    """
    return build_stage8_plan(
        model_name=model_name, model_revision=model_revision, output_root=output_root,
        regions=STAGE8_REGIONS, radii=STAGE8_RADII, n_directions_per_cell=STAGE8_SMOKE_N_DIRECTIONS, d_map_n=STAGE8_SMOKE_D_MAP_N,
        generation_batch_size=generation_batch_size,
    )


# =============================================================================================
# Direction-family seed bank + population (the core Stage-8-specific design)
# =============================================================================================


def build_stage8_direction_seed_bank(base_seed: int, regions: Sequence[str], n_directions: int) -> Dict[str, Tuple[int, ...]]:
    """Exactly `n_directions` seeds PER region, derived as a function of (region, direction
    index) ONLY -- deliberately NEVER a function of radius -- so the identical seed value is
    reused for direction i across every radius within region `region` (see module docstring
    point 3). Calling this twice with identical arguments always returns the IDENTICAL bank.
    """
    return {
        region: tuple(derive_seed(base_seed, "stage8_direction_family", region, str(i)) for i in range(n_directions))
        for region in regions
    }


def compute_direction_seed_bank_hash(seed_bank: Dict[str, Tuple[int, ...]]) -> str:
    canonical = json.dumps({region: list(seeds) for region, seeds in sorted(seed_bank.items())}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Stage8DirectionAssignment:
    """One (region, direction_index) pair's manifest at one radius -- pairs a PerturbationManifest
    with the direction-family metadata that isn't itself part of PerturbationManifest's frozen
    schema (direction_index; direction_seed and region ARE already on the manifest as .seed/
    .anatomy_region, repeated here only for a single self-contained persisted identity).
    """
    manifest: PerturbationManifest
    region: str
    direction_index: int
    direction_seed: int

    @property
    def direction_family_id(self) -> str:
        return f"{self.region}:{self.direction_index}"


def build_stage8_population(
    plan: Stage8Plan, seed_bank: Dict[str, Tuple[int, ...]], parameter_mask_hash_by_region: Dict[str, str],
) -> Dict[Tuple[str, float], Tuple[Stage8DirectionAssignment, ...]]:
    missing_regions = set(plan.regions) - set(parameter_mask_hash_by_region)
    if missing_regions:
        raise ValueError(f"Missing parameter_mask_hash for region(s): {sorted(missing_regions)}")
    missing_bank_regions = set(plan.regions) - set(seed_bank)
    if missing_bank_regions:
        raise ValueError(f"Missing direction seed bank for region(s): {sorted(missing_bank_regions)}")
    for region in plan.regions:
        if len(seed_bank[region]) != plan.n_directions_per_cell:
            raise ValueError(
                f"Direction seed bank for region {region!r} has {len(seed_bank[region])} seeds, "
                f"expected {plan.n_directions_per_cell}."
            )

    population_by_cell: Dict[Tuple[str, float], Tuple[Stage8DirectionAssignment, ...]] = {}
    for region in plan.regions:
        mask_hash = parameter_mask_hash_by_region[region]
        seeds = seed_bank[region]
        for radius in plan.radii:
            assignments = tuple(
                Stage8DirectionAssignment(
                    manifest=PerturbationManifest(
                        seed=seed, perturbation_mode=PERTURBATION_MODE, anatomy_region=region, radius=radius, sigma=None,
                        model_family=plan.model_family, model_scale=plan.model_scale, model_revision=plan.model_revision,
                        parameter_mask_hash=mask_hash,
                    ),
                    region=region, direction_index=i, direction_seed=seed,
                )
                for i, seed in enumerate(seeds)
            )
            population_by_cell[(region, radius)] = assignments
    return population_by_cell


class DirectionSeedReuseViolationError(RuntimeError):
    """The population does not satisfy Stage 8's direction-family invariant (each region's
    `n_directions_per_cell` seeds each reused EXACTLY once per radius, never fewer/more, never
    an accidental collision with a DIFFERENT direction index within the same region) -- or the
    resulting perturbation_ids are not all unique. Hard-fails rather than silently proceeding
    with a population that would make radius-trajectory analysis (Section 13) meaningless.
    """


def validate_stage8_direction_seed_reuse(
    plan: Stage8Plan, population_by_cell: Dict[Tuple[str, float], Tuple[Stage8DirectionAssignment, ...]],
) -> None:
    all_ids = [a.manifest.perturbation_id for cell in population_by_cell.values() for a in cell]
    if len(all_ids) != len(set(all_ids)):
        raise DirectionSeedReuseViolationError(f"Duplicate perturbation_id(s) in the Stage-8 population ({len(all_ids)} total, {len(set(all_ids))} unique).")
    expected_total = len(plan.regions) * len(plan.radii) * plan.n_directions_per_cell
    if len(all_ids) != expected_total:
        raise DirectionSeedReuseViolationError(f"Stage-8 population has {len(all_ids)} perturbations, expected {expected_total}.")

    by_region_seed: Dict[str, Dict[int, int]] = {}
    for (region, _radius), assignments in population_by_cell.items():
        counts = by_region_seed.setdefault(region, {})
        for a in assignments:
            counts[a.direction_seed] = counts.get(a.direction_seed, 0) + 1
    for region in plan.regions:
        counts = by_region_seed.get(region, {})
        if len(counts) != plan.n_directions_per_cell:
            raise DirectionSeedReuseViolationError(f"Region {region!r} has {len(counts)} distinct direction seeds, expected {plan.n_directions_per_cell}.")
        wrong = {seed: n for seed, n in counts.items() if n != len(plan.radii)}
        if wrong:
            raise DirectionSeedReuseViolationError(f"Region {region!r}: seed(s) not reused exactly {len(plan.radii)}x (once per radius): {wrong}")


# =============================================================================================
# Checkpoint identity
# =============================================================================================


class IncompatibleStage8CheckpointError(RuntimeError):
    """An existing checkpoint_manifest.json in this output directory does not match the
    current run's identity -- refuses to resume a differently-configured partial run.
    """


@dataclass(frozen=True)
class Stage8CheckpointManifest:
    experiment_id: str
    run_signature: str
    restoration_mode: str
    perturbation_mode: str
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: bool
    model_revision: str
    dataset_role: str
    regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_directions_per_cell: int
    d_map_n: int
    subset_hashes: Dict[str, str]
    region_mask_hashes: Dict[str, str]
    direction_seed_bank_hash: str
    expected_unique_perturbations: int
    expected_result_rows: int
    # Execution-only identity field (this repair pass) -- None for a checkpoint written before
    # this field existed (or the OOM'd v1 unbatched attempt); a resume against a DIFFERENT
    # batch size is rejected by ordinary dataclass equality just like any other mismatch.
    generation_batch_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "run_signature": self.run_signature,
            "restoration_mode": self.restoration_mode, "perturbation_mode": self.perturbation_mode,
            "perturbation_semantics": self.perturbation_mode, "radius_realization_method": self.radius_realization_method,
            "multimodal_cache_policy": self.multimodal_cache_policy, "enable_prefix_caching": self.enable_prefix_caching,
            "model_revision": self.model_revision, "dataset_role": self.dataset_role,
            "regions": list(self.regions), "radii": list(self.radii), "capabilities": list(self.capabilities),
            "n_directions_per_cell": self.n_directions_per_cell, "d_map_n": self.d_map_n,
            "subset_hashes": dict(sorted(self.subset_hashes.items())),
            "region_mask_hashes": dict(sorted(self.region_mask_hashes.items())),
            "direction_seed_bank_hash": self.direction_seed_bank_hash,
            "expected_unique_perturbations": self.expected_unique_perturbations,
            "expected_result_rows": self.expected_result_rows,
            "generation_batch_size": self.generation_batch_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage8CheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], run_signature=d["run_signature"], restoration_mode=d["restoration_mode"],
            perturbation_mode=d["perturbation_mode"], radius_realization_method=d["radius_realization_method"],
            multimodal_cache_policy=d["multimodal_cache_policy"], enable_prefix_caching=d["enable_prefix_caching"],
            model_revision=d["model_revision"], dataset_role=d["dataset_role"],
            regions=tuple(d["regions"]), radii=tuple(d["radii"]), capabilities=tuple(d["capabilities"]),
            n_directions_per_cell=d["n_directions_per_cell"], d_map_n=d["d_map_n"],
            subset_hashes=dict(d["subset_hashes"]), region_mask_hashes=dict(d["region_mask_hashes"]),
            direction_seed_bank_hash=d["direction_seed_bank_hash"],
            expected_unique_perturbations=d["expected_unique_perturbations"], expected_result_rows=d["expected_result_rows"],
            generation_batch_size=d.get("generation_batch_size"),
        )


def build_stage8_checkpoint_manifest(
    plan: Stage8Plan, capability_contexts: Dict[str, CapabilityContext], region_mask_hashes: Dict[str, str],
    seed_bank: Dict[str, Tuple[int, ...]],
) -> Stage8CheckpointManifest:
    if plan.d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 8 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {plan.d_map_n}")
    missing_regions = set(plan.regions) - set(region_mask_hashes)
    if missing_regions:
        raise ValueError(f"Missing region_mask_hashes for region(s): {sorted(missing_regions)}")
    return Stage8CheckpointManifest(
        experiment_id=EXPERIMENT_ID, run_signature=plan.run_signature, restoration_mode=RESTORATION_MODE,
        perturbation_mode=PERTURBATION_MODE, radius_realization_method=plan.radius_realization_method,
        multimodal_cache_policy=plan.multimodal_cache_policy, enable_prefix_caching=plan.enable_prefix_caching,
        model_revision=plan.model_revision, dataset_role=DATASET_ROLE,
        regions=plan.regions, radii=plan.radii, capabilities=plan.capabilities,
        n_directions_per_cell=plan.n_directions_per_cell, d_map_n=plan.d_map_n,
        subset_hashes={c: ctx.subset_hash for c, ctx in capability_contexts.items()},
        region_mask_hashes={r: region_mask_hashes[r] for r in plan.regions},
        direction_seed_bank_hash=compute_direction_seed_bank_hash(seed_bank),
        expected_unique_perturbations=plan.total_unique_perturbations,
        expected_result_rows=plan.total_perturbation_capability_evaluations,
        generation_batch_size=plan.generation_batch_size,
    )


def ensure_stage8_checkpoint_manifest(path: Path, current: Stage8CheckpointManifest) -> Stage8CheckpointManifest:
    if path.exists():
        existing = Stage8CheckpointManifest.from_dict(json.loads(path.read_text()))
        if existing != current:
            raise IncompatibleStage8CheckpointError(
                f"Existing checkpoint at {path} is incompatible with this run -- refusing to "
                f"resume: existing={existing.to_dict()} current={current.to_dict()}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current.to_dict(), indent=2))
    return current


def build_stage8_run_manifest_summary(checkpoint: Stage8CheckpointManifest, records: List[ExperimentResultRecord]) -> Dict[str, Any]:
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
        "model_revision": checkpoint.model_revision,
        "regions": list(checkpoint.regions), "radii": list(checkpoint.radii),
        "capabilities": list(checkpoint.capabilities), "n_directions_per_cell": checkpoint.n_directions_per_cell,
        "d_map_n": checkpoint.d_map_n, "subset_hashes": dict(sorted(checkpoint.subset_hashes.items())),
        "region_mask_hashes": dict(sorted(checkpoint.region_mask_hashes.items())),
        "direction_seed_bank_hash": checkpoint.direction_seed_bank_hash,
        "expected_unique_perturbations": checkpoint.expected_unique_perturbations,
        "actual_unique_perturbations": actual_unique_perturbations,
        "expected_result_rows": checkpoint.expected_result_rows, "actual_result_rows": actual_result_rows,
        "generation_batch_size": checkpoint.generation_batch_size,
        "run_complete": run_complete,
    }


def write_stage8_run_manifest(output_dir: Path) -> Dict[str, Any]:
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_manifest.json found at {checkpoint_path} -- this run never started (or output_dir is wrong).")
    checkpoint = Stage8CheckpointManifest.from_dict(json.loads(checkpoint_path.read_text()))
    records = load_records(output_dir / "results.jsonl")
    manifest = build_stage8_run_manifest_summary(checkpoint, records)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_d_map_capability_contexts(
    base_seed: int, subset_ids_dir: "str | Path", d_map_n: int, *, load_capability_benchmark_config: Callable, load_adapter: Callable,
    rss_checkpoint: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, CapabilityContext]:
    """DRIVER MEMORY AUDIT (this repair pass -- see STAGE8_GENERATE_BATCH_SIZE's own comment for
    the full failure context): every Stage-8 adapter's load_examples() (confirmed by direct
    source read of counting_tallyqa.py, fine_grained_recognition_cub.py,
    ocr_text_recognition_textvqa.py, visual_grounding_refcoco.py, and the shared GQA
    _gqa_filtered_base.py used by spatial_reasoning/relational_reasoning) iterates the ENTIRE
    underlying dataset/split and attaches a decoded PIL.Image to an Example for EVERY row --
    subset_selection.build_or_load_subset only reduces this to d_map_n AFTER the full candidate
    pool already exists in memory. For HuggingFaceM4/the_cauldron's "tallyqa" config alone that
    is ~98.7k rows/images, transiently materialized in full before being cut down to N=50. This
    is called once per capability, sequentially, for all 6 STAGE8_CAPABILITIES, so a 6x
    repeated multi-GB transient spike is architecturally built into this exact loop.
    `CapabilityContext.examples` itself (the value actually kept resident for the rest of the
    run) is NOT the problem -- it holds only the persisted d_map_n=50 subset, i.e. <=300 decoded
    images total across all 6 capabilities, never the 6 full corpora simultaneously; refactoring
    every adapter's load_examples() to defer image decoding until after subset selection would
    fix the transient spike at its source but touches the frozen Gate-2 adapter contracts this
    stage was told to reuse unchanged -- NOT done in this pass, flagged instead as the
    identified root cause pending explicit sign-off. What IS done here, additively and without
    touching any adapter: mem_telemetry.release_transient_memory() (gc.collect() +
    glibc malloc_trim(0)) after each capability's context is built, so the freed-but-not-yet-
    returned-to-the-OS memory from one capability's transient full-dataset materialization
    cannot ratchet driver RSS upward into the next capability's own load.
    """
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 8 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    from .mem_telemetry import release_transient_memory, rss_mb
    from .run_global_visual_thicket_pilot import CAPABILITY_CONFIG_FILES

    def _checkpoint(label: str) -> None:
        if rss_checkpoint is not None:
            rss_checkpoint(label, rss_mb())

    contexts: Dict[str, CapabilityContext] = {}
    for capability in STAGE8_CAPABILITIES:  # fixed order
        cfg_path = REPO_ROOT / "configs" / "benchmarks" / CAPABILITY_CONFIG_FILES[capability]
        cfg = load_capability_benchmark_config(cfg_path)
        benchmark = load_adapter(cfg.dataset.adapter)
        ctx = build_d_map_context(benchmark, cfg, capability, d_map_n, base_seed, subset_ids_dir)
        contexts[capability] = ctx
        release_transient_memory()
        _checkpoint(f"after_{capability}_context_built")
    _checkpoint("after_all_capability_contexts_constructed")
    return contexts


# =============================================================================================
# Worker-RPC transport (same TP=1 list-unwrap convention as every other GPU script here)
# =============================================================================================


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list of per-worker results. Got: {results!r}")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; Stage 8 is TP=1-only and expects exactly 1.")
    return results[0]


def _collective_rpc_single_worker(engine: Any, method, args: tuple = (), *, label: str, ray_get: Optional[Callable] = None) -> Any:
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


# =============================================================================================
# Part 4: baseline repeatability preflight
# =============================================================================================


def _run_one_baseline_pass(
    engine: Any, llm_adapter: Any, ctx: CapabilityContext, tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable], generation_batch_size: Optional[int],
    capability: str, pass_label: str,
) -> Dict[str, Any]:
    """One pass (A or B) of the baseline-repeatability preflight for one capability, with full
    RSS telemetry (Section 2 of the spec): rss_before / rss_after_requests / rss_after_generate
    / rss_after_scoring / rss_after_cleanup / peak_delta, all in MB. Retains ONLY the compact
    (score, generation_hash, parsed_prediction_hash) triple after this returns -- the RunResult
    itself (per-example raw generations/parsed predictions/scores) is explicitly `del`eted and
    mem_telemetry.release_transient_memory() is called before returning (Section 5 of the spec:
    do not retain full baseline outputs).
    """
    from .mem_telemetry import release_transient_memory, rss_mb

    telemetry: Dict[str, float] = {"rss_before_mb": rss_mb()}

    def _rss_checkpoint(label: str) -> None:
        telemetry[f"rss_{label}_mb"] = rss_mb()

    reset_to_base_weights_via_rpc(engine, ray_get=ray_get)
    reset_vllm_encoder_cache_full(engine)

    result = run_benchmark(
        ctx.benchmark, ctx.examples, llm_adapter, tokenizer, sampling_params,
        max_requests_per_generate=generation_batch_size, rss_checkpoint=_rss_checkpoint,
    )
    score = result.aggregate_metrics["primary_metric"]
    generation_hash = result.generation_hash()
    parsed_prediction_hash = result.parsed_prediction_hash()
    telemetry.setdefault("rss_after_scoring_mb", rss_mb())

    del result
    release_transient_memory()
    telemetry["rss_after_cleanup_mb"] = rss_mb()
    rss_values = [v for k, v in telemetry.items() if k.startswith("rss_")]
    telemetry["peak_delta_mb"] = max(rss_values) - telemetry["rss_before_mb"]

    return {
        "capability": capability, "pass": pass_label, "n_examples": len(ctx.examples),
        "score": score, "generation_hash": generation_hash, "parsed_prediction_hash": parsed_prediction_hash,
        "memory_telemetry": telemetry,
    }


def run_baseline_repeatability_preflight_rpc(
    engine: Any, capability_contexts: Dict[str, CapabilityContext], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable] = None, generation_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """For each of the 6 capabilities, in fixed order: reset theta_0, full verified encoder-
    cache reset, evaluate D_map N=50 (via the SAME bounded-generation path candidate evaluation
    uses -- generation_batch_size, default None preserves the original unbatched call shape),
    record score/generation_hash/parsed_prediction_hash -- TWICE -- and compare. Never averages
    across the two passes; every field is compared for EXACT equality. Returns a per-capability
    report (including each pass's own RSS telemetry, Section 2 of the spec); raises nothing
    itself (see ensure_baseline_repeatability below for the hard-stop gate) so a caller can
    persist the full report before deciding whether to stop.
    """
    llm_adapter = RayEngineLLMAdapter(engine, ray_get=ray_get)
    report: Dict[str, Any] = {}
    for capability, ctx in capability_contexts.items():
        pass_a = _run_one_baseline_pass(
            engine, llm_adapter, ctx, tokenizer, sampling_params, run_benchmark=run_benchmark, ray_get=ray_get,
            generation_batch_size=generation_batch_size, capability=capability, pass_label="A",
        )
        pass_b = _run_one_baseline_pass(
            engine, llm_adapter, ctx, tokenizer, sampling_params, run_benchmark=run_benchmark, ray_get=ray_get,
            generation_batch_size=generation_batch_size, capability=capability, pass_label="B",
        )

        score_match = pass_a["score"] == pass_b["score"]
        generation_hash_match = pass_a["generation_hash"] == pass_b["generation_hash"]
        parsed_prediction_hash_match = pass_a["parsed_prediction_hash"] == pass_b["parsed_prediction_hash"]
        report[capability] = {
            "score_a": pass_a["score"], "score_b": pass_b["score"], "score_match": score_match,
            "generation_hash_a": pass_a["generation_hash"], "generation_hash_b": pass_b["generation_hash"],
            "generation_hash_match": generation_hash_match,
            "parsed_prediction_hash_a": pass_a["parsed_prediction_hash"], "parsed_prediction_hash_b": pass_b["parsed_prediction_hash"],
            "parsed_prediction_hash_match": parsed_prediction_hash_match,
            "deterministic": score_match and generation_hash_match and parsed_prediction_hash_match,
            "memory_telemetry": {"pass_a": pass_a["memory_telemetry"], "pass_b": pass_b["memory_telemetry"]},
            "n_examples": pass_a["n_examples"],
        }
    return report


def ensure_baseline_repeatability(report: Dict[str, Any]) -> None:
    """HARD STOP before Stage-8 perturbations if any capability's preflight was not
    deterministic -- names exactly which capability/check failed, never silently proceeds.
    """
    failed = {cap: r for cap, r in report.items() if not r["deterministic"]}
    if failed:
        details = ", ".join(
            f"{cap} (score_match={r['score_match']}, generation_hash_match={r['generation_hash_match']}, "
            f"parsed_prediction_hash_match={r['parsed_prediction_hash_match']})"
            for cap, r in failed.items()
        )
        raise BaselineNondeterminismError(
            f"Baseline repeatability preflight failed for {len(failed)} capability(ies): {details}. "
            f"Refusing to start the Stage-8 576-perturbation sweep -- this is especially significant "
            f"for fine_grained_recognition, which prior work suggested may have inference variability."
        )


# =============================================================================================
# Per-candidate lifecycle
# =============================================================================================


def evaluate_one_stage8_candidate_rpc(
    engine: Any, assignment: Stage8DirectionAssignment, region_param_names: Sequence[str],
    capability_contexts: Dict[str, CapabilityContext], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable] = None, generation_batch_size: Optional[int] = None,
) -> List[ExperimentResultRecord]:
    """Identical lifecycle to run_stage7b_anatomical_calibration.evaluate_one_calibration_
    candidate_rpc (same v3 quantization-aware radius acceptance, same twice-per-candidate cache
    reset -- once before capability evaluation, once after restoration is verified), generalized
    to loop over all 6 STAGE8_CAPABILITIES and to persist direction-family identity
    (direction_family_id/direction_seed/direction_index/region) in runtime_metadata so Section
    13's radius trajectories can group rows by direction family across the 3 radii.

    `generation_batch_size` (this repair pass, driver-RSS OOM audit): threaded straight through
    to run_benchmark's own `max_requests_per_generate` for EVERY one of the 6 capabilities' N=50
    evaluations -- the same bounded-generation path the baseline preflight uses (Section 7 of
    the spec: "the same N=50 issue will affect all 576 candidates if only the baseline preflight
    is fixed"). Weights are NOT reset and the encoder cache is NOT reset between microbatches
    within one capability's own evaluation (only once before/after the whole candidate, as
    before) -- the accepted perturbation's weights are unchanged across microbatches, so cached
    encoder entries remain valid.
    """
    manifest = assignment.manifest
    if manifest.perturbation_mode != PERTURBATION_MODE:
        raise ValueError(f"Stage 8 only evaluates {PERTURBATION_MODE!r} manifests, got {manifest.perturbation_mode!r}")

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
                raise RealizedRadiusMismatchError(
                    f"Perturbation {manifest.perturbation_id!r} (region={manifest.anatomy_region!r}, "
                    f"requested radius={manifest.radius}): strict-mode realized relative-L2 {realized_r} "
                    f"still differs by more than {REALIZED_RADIUS_TOLERANCE}."
                )
        elif acceptance_mode == "quantization_limited":
            if apply_result["relative_radius_error"] > QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE:
                raise RealizedRadiusMismatchError(
                    f"Perturbation {manifest.perturbation_id!r} (region={manifest.anatomy_region!r}, "
                    f"requested radius={manifest.radius}): quantization-limited relative error "
                    f"{apply_result['relative_radius_error']} still exceeds {QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE}."
                )
        else:
            raise RealizedRadiusMismatchError(f"Unknown radius_acceptance_mode {acceptance_mode!r} for perturbation {manifest.perturbation_id!r}.")

        reset_vllm_encoder_cache_full(engine)

        for capability, ctx in capability_contexts.items():
            if ctx.partition.manifest_hash != ctx.subset_hash:
                raise DatasetRoleViolationError(f"CapabilityContext for {capability!r} has an inconsistent subset_hash.")
            result = run_benchmark(
                ctx.benchmark, ctx.examples, llm_adapter, tokenizer, sampling_params,
                max_requests_per_generate=generation_batch_size,
            )
            perturbed_score = result.aggregate_metrics["primary_metric"]
            base_score = ctx.base_score
            records.append(ExperimentResultRecord(
                experiment_id=EXPERIMENT_ID, perturbation_id=manifest.perturbation_id,
                model_family=manifest.model_family, model_scale=manifest.model_scale, model_revision=manifest.model_revision,
                perturbation_mode=manifest.perturbation_mode, anatomy_region=manifest.anatomy_region,
                radius=manifest.radius, sigma=manifest.sigma, seed=manifest.seed, parameter_mask_hash=manifest.parameter_mask_hash,
                capability=capability, dataset_role=DATASET_ROLE, subset_hash=ctx.subset_hash,
                base_score=base_score, perturbed_score=perturbed_score, delta=perturbed_score - base_score,
                parser_failure_rate=result.aggregate_metrics.get("parser_failure_rate"),
                per_example_result_path=None, per_example_result_hash=result.generation_hash(),
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
                    # Direction-family identity (this repair pass, Section 6/13 of the spec) --
                    # the SAME direction_seed recurs across all 3 radii within `region`; never
                    # interpret the same numeric seed across DIFFERENT regions as the same
                    # geometric direction (disjoint parameter subspaces).
                    "direction_family_id": assignment.direction_family_id,
                    "direction_seed": assignment.direction_seed,
                    "direction_index": assignment.direction_index,
                    "region": assignment.region,
                    "generation_batch_size": generation_batch_size,
                },
            ))
    finally:
        reset_to_base_weights_via_rpc(engine, ray_get=ray_get)

    verification = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Exact fixed-base restoration failed after Stage-8 candidate {manifest.perturbation_id!r} "
            f"(region={manifest.anatomy_region!r}, radius={manifest.radius}, seed={manifest.seed}): "
            f"max_abs_drift={verification['max_abs_drift']}"
        )

    reset_vllm_encoder_cache_full(engine)
    for record in records:
        record.runtime_metadata["cache_reset_after_restoration"] = True

    return records


def run_stage8_rpc(
    plan: Stage8Plan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, seed_bank: Dict[str, Tuple[int, ...]], region_param_names_by_region: Dict[str, Sequence[str]],
    parameter_mask_hash_by_region: Dict[str, str], *, run_benchmark: Callable, ray_get: Optional[Callable] = None,
) -> List[ExperimentResultRecord]:
    """A candidate is checkpointed (append_candidate_rows) ONLY after its full apply ->
    evaluate(all 6 capabilities) -> reset -> verify cycle has already succeeded inside
    evaluate_one_stage8_candidate_rpc -- a row on disk is itself proof restoration passed.
    Never depends on candidate ordering (see evaluate_one_stage8_candidate_rpc's own docstring).
    """
    population_by_cell = build_stage8_population(plan, seed_bank, parameter_mask_hash_by_region)
    validate_stage8_direction_seed_reuse(plan, population_by_cell)

    current_checkpoint = build_stage8_checkpoint_manifest(plan, capability_contexts, parameter_mask_hash_by_region, seed_bank)
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    ensure_stage8_checkpoint_manifest(checkpoint_path, current_checkpoint)

    results_path = plan.output_dir / "results.jsonl"
    completed = load_completed_perturbation_rows(results_path, plan.capabilities)

    all_records: List[ExperimentResultRecord] = []
    for rows in completed.values():
        all_records.extend(rows)

    for (region, _radius), assignments in population_by_cell.items():
        region_param_names = region_param_names_by_region[region]
        for assignment in assignments:
            if assignment.manifest.perturbation_id in completed:
                continue
            records = evaluate_one_stage8_candidate_rpc(
                engine, assignment, region_param_names, capability_contexts, tokenizer, sampling_params,
                run_benchmark=run_benchmark, ray_get=ray_get, generation_batch_size=plan.generation_batch_size,
            )
            append_candidate_rows(results_path, records)
            all_records.extend(records)
    return all_records


# =============================================================================================
# CLI entry point
# =============================================================================================


def _extract_all_rss_values(preflight_report: Dict[str, Any]) -> List[float]:
    """Flattens every rss_*_mb value out of the nested per-capability/per-pass memory_telemetry
    dicts in a baseline-repeatability preflight report -- used to compute the overall peak RSS
    observed during the preflight (Section 9 of the spec).
    """
    values: List[float] = []
    for cap_report in preflight_report.values():
        for pass_telemetry in cap_report.get("memory_telemetry", {}).values():
            for key, value in pass_telemetry.items():
                if key.startswith("rss_") and key.endswith("_mb"):
                    values.append(value)
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "stage8_coarse_anatomical_atlas"))
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"tiny live GPU smoke: 3 regions x 3 radii x 1 direction family x 6 capabilities x "
             f"{STAGE8_SMOKE_D_MAP_N} D_map examples/capability = 9 perturbations, 54 rows, 270 "
             f"perturbed model-example evaluations -- execution size only, same scientific protocol.",
    )
    parser.add_argument(
        "--baseline-memory-smoke", action="store_true",
        help="Driver-RSS diagnostic (this repair pass, root-causing the v1 full-run OOM): launches "
             "the exact Stage-8 engine, all 6 capabilities, D_map N=50, runs the two-pass baseline"
             "-repeatability preflight under bounded generation, records RSS telemetry throughout, "
             "and evaluates ZERO perturbations (6 capabilities x 50 examples x 2 passes = 600 "
             "baseline model-example evaluations total).",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU")
    parser.add_argument(
        "--generation-batch-size", type=int, default=STAGE8_GENERATE_BATCH_SIZE,
        help=f"Bounded llm.generate() microbatch size (default {STAGE8_GENERATE_BATCH_SIZE}) -- "
             f"execution/memory-bounding only, never a scientific parameter. Pass 0 or a negative "
             f"value's caller should use None instead; this flag has no way to request the OOM'd "
             f"v1 unbatched behavior on purpose.",
    )
    args = parser.parse_args(argv)

    if args.smoke and args.baseline_memory_smoke:
        print("--smoke and --baseline-memory-smoke are mutually exclusive.", file=sys.stderr)
        return 1
    if args.generation_batch_size <= 0:
        print(f"--generation-batch-size must be positive, got {args.generation_batch_size}.", file=sys.stderr)
        return 1

    from .config import load_config

    cfg = load_config(args.config)

    if args.smoke:
        plan = build_stage8_smoke_plan(
            model_name=cfg.model.name, model_revision=cfg.model.revision, output_root=args.output_root,
            generation_batch_size=args.generation_batch_size,
        )
    else:
        # Both the real full run AND --baseline-memory-smoke use the frozen full scientific
        # config (all 6 capabilities, D_map N=50) -- baseline-memory-smoke differs only in that
        # it never calls run_stage8_rpc (see below), never touching the perturbation population.
        plan = build_stage8_plan(
            model_name=cfg.model.name, model_revision=cfg.model.revision, output_root=args.output_root,
            generation_batch_size=args.generation_batch_size,
        )

    print(f"Stage 8 plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}): regions={plan.regions} radii={plan.radii}")
    print(f"capabilities={plan.capabilities}")
    print(f"n_directions_per_cell={plan.n_directions_per_cell} d_map_n={plan.d_map_n} total_unique_perturbations={plan.total_unique_perturbations}")
    print(f"total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations}")
    print(f"total_perturbed_model_example_evaluations={plan.total_perturbed_model_example_evaluations}")
    print(f"radius_realization_method={plan.radius_realization_method}")
    print(f"multimodal_cache_policy={plan.multimodal_cache_policy}")
    print(f"enable_prefix_caching={plan.enable_prefix_caching}")
    print(f"generation_batch_size={plan.generation_batch_size}")
    print(f"output_dir={plan.output_dir}")
    if args.baseline_memory_smoke:
        print(
            "--baseline-memory-smoke: all 6 capabilities x D_map N=50 x 2 passes = 600 baseline "
            "model-example evaluations; ZERO perturbations will be evaluated."
        )
    else:
        print(
            "Lifecycle: baseline repeatability preflight (each of 6 capabilities evaluated TWICE "
            "against theta_0, hard stop on any disagreement) -> for each candidate: "
            "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 -> full encoder-cache "
            "reset -> evaluate all 6 capabilities (bounded generation microbatches) -> "
            "reset_to_base_weights -> verify exact restoration -> full encoder-cache reset again "
            "-> checkpoint candidate rows."
        )

    if args.dry_run:
        return 0

    try:
        assert_feasible(
            f"Stage 8 coarse anatomical atlas ({plan.run_signature})",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision")
    model_resolution = resolve_and_report_model_snapshot(plan.model_name, plan.model_revision)
    engine_config = build_stage7b_engine_config()  # bf16 / max_model_len=4096 / gpu_mem=0.60 / TP=1 / enable_prefix_caching=False, reused BY IDENTITY
    assert engine_config["enable_prefix_caching"] is False, "Stage 8 must never run with prefix caching enabled."
    print(format_runtime_compatibility_diagnostic(
        model_resolution, worker_extension_cls="utils.worker_extn.WorkerExtension",
        vllm_version=get_vllm_version(), engine_mode=detect_vllm_engine_mode(), engine_config=engine_config,
    ))

    from .benchmarks.runner import run_benchmark
    from .config import load_capability_benchmark_config
    from .run_capability_benchmark_gate import load_adapter
    from .thicket.data_roles import write_data_role_manifest
    from .vlm_adapter import bootstrap_ray, verify_workers_can_import_external_root

    from .mem_telemetry import rss_mb

    start_rss_mb = rss_mb()

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    subset_ids_dir = plan.output_dir / "d_map_subsets"
    context_rss_log: Dict[str, float] = {}
    capability_contexts = build_d_map_capability_contexts(
        STAGE8_BASE_SEED, subset_ids_dir, plan.d_map_n,
        load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
        rss_checkpoint=lambda label, value: context_rss_log.__setitem__(label, value),
    )
    for capability, ctx in capability_contexts.items():
        write_data_role_manifest(ctx.partition, plan.output_dir / "data_roles" / f"{capability}_d_map.json")

    seed_bank = None
    if not args.baseline_memory_smoke:
        # --baseline-memory-smoke never touches the perturbation population -- no direction
        # seed bank is needed or persisted for that mode.
        seed_bank = build_stage8_direction_seed_bank(STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
        (plan.output_dir / "direction_family_manifest.json").write_text(json.dumps(
            {"regions": list(plan.regions), "n_directions_per_cell": plan.n_directions_per_cell,
             "seed_bank": {r: list(s) for r, s in seed_bank.items()},
             "direction_seed_bank_hash": compute_direction_seed_bank_hash(seed_bank)}, indent=2,
        ))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_resolution["resolved_snapshot_path"])

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore  # upstream, unmodified -- teardown only

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

        region_info = _collective_rpc_single_worker(engine, report_region_param_names, args=(plan.regions,), label="report_region_param_names")
        region_param_names_by_region = {r: tuple(info["param_names"]) for r, info in region_info.items()}
        parameter_mask_hash_by_region = {r: info["mask_hash"] for r, info in region_info.items()}

        llm_adapter = RayEngineLLMAdapter(engine)
        from .run_global_visual_thicket_pilot import load_or_compute_baseline_scores
        baseline_path = plan.output_dir / "baseline_scores.json"
        load_or_compute_baseline_scores(baseline_path, capability_contexts, plan.model_revision, plan.run_signature, llm_adapter, tokenizer, sampling_params)

        print("Running baseline repeatability preflight (Section 4) before any Stage-8 perturbation...")
        preflight_report = run_baseline_repeatability_preflight_rpc(
            engine, capability_contexts, tokenizer, sampling_params, run_benchmark=run_benchmark,
            generation_batch_size=plan.generation_batch_size,
        )
        (plan.output_dir / "baseline_repeatability_preflight.json").write_text(json.dumps(preflight_report, indent=2))

        all_rss_values = [start_rss_mb, *context_rss_log.values(), *_extract_all_rss_values(preflight_report), rss_mb()]
        peak_rss_mb = max(all_rss_values)
        final_rss_mb_before_perturbations = rss_mb()
        memory_diagnostic = {
            "start_rss_mb": start_rss_mb, "capability_context_rss_mb": context_rss_log,
            "peak_rss_mb": peak_rss_mb, "final_rss_mb": final_rss_mb_before_perturbations,
            "peak_minus_start_mb": peak_rss_mb - start_rss_mb,
            "generation_batch_size": plan.generation_batch_size,
        }
        (plan.output_dir / "baseline_memory_diagnostic.json").write_text(json.dumps(memory_diagnostic, indent=2))
        print(
            f"Memory diagnostic: start={start_rss_mb:.1f}MB peak={peak_rss_mb:.1f}MB "
            f"final={final_rss_mb_before_perturbations:.1f}MB peak_minus_start={peak_rss_mb - start_rss_mb:.1f}MB"
        )

        ensure_baseline_repeatability(preflight_report)
        print(f"Baseline repeatability preflight PASSED for all {len(preflight_report)} capabilities.")

        if args.baseline_memory_smoke:
            print("--baseline-memory-smoke complete -- zero perturbations evaluated.")
            records = None
        else:
            records = run_stage8_rpc(
                plan, capability_contexts, engine, tokenizer, sampling_params, seed_bank,
                region_param_names_by_region, parameter_mask_hash_by_region, run_benchmark=run_benchmark,
            )
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    if args.baseline_memory_smoke:
        return 0

    manifest = write_stage8_run_manifest(plan.output_dir)
    print(f"Wrote {len(records)} result rows to {plan.output_dir / 'results.jsonl'}")
    print(f"Run manifest: {json.dumps(manifest, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
