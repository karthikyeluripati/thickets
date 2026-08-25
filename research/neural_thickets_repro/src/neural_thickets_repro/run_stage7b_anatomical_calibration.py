"""Stage 7B: anatomical-region calibration sweep. Identifies qualitative behavioral regimes
(near-base / active-non-collapsed / destructive) per (anatomy region, radius) cell -- it is
explicitly NOT a hyperparameter search (section 7): no function in this module selects a "best"
radius by capability score, and none ever will (see `test_no_best_radius_selection_logic_exists`
in the accompanying test file, which asserts this mechanically against the module's own public
names).

perturbation_mode = anatomical_relative_l2 (thicket.perturbation.apply_anatomical_relative_l2,
the EXACT-rescale mode -- see that module's docstring for why this is scientifically distinct
from scoped_perturbation.py's expectation-only relative_l2 scale mode). Dispatched via
scoped_anatomical_perturbation.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3
(this repair pass -- see QUANTIZATION-AWARE ACCEPTANCE v3 section below).

=================================================================================================
BF16 REALIZED-RADIUS CORRECTION -- v1 (proven root cause, not assumed): a real RunPod smoke
candidate (region=vision, r=0.035698828543799424) hard-failed with realized
r=0.03569534313727009 (abs error 3.485e-06 against a 1e-6 tolerance). Root cause, proven by
instrumentation (tests/test_scoped_anatomical_perturbation.py): `apply_anatomical_relative_l2`'s
own `realized_epsilon_l2_norm` was computed from the additive `delta` tensor BEFORE the in-place
`p.add_()` -- on bf16 weights, that add rounds AGAIN, so the TRUE weight-space displacement
(theta_after - theta_before, measured in fp32) is measurably different from `delta`'s own norm.
`thicket.perturbation.apply_anatomical_relative_l2` measures and returns both the DESIGNED
(pre-add) and the TRUE REALIZED (post-add) values; v1's `scoped_apply_anatomical_perturbation_
bf16_corrected` iteratively rescaled the SAME fixed seeded Gaussian direction via PROPORTIONAL
correction only.

BF16 BRACKETED SOLVER -- v2 (this repair pass): v1's proportional-only correction was then run
for real on a full-calibration attempt and OSCILLATED without converging at the smallest frozen
radius (region=vision, r=0.0035698828543799426): 5 attempts alternated overshoot/undershoot
(closest was attempt 4, abs error 1.37e-6, still > tolerance) and exhausted its budget. Root
cause: near this radius/region, the map from scalar magnitude to TRUE bf16-realized relative-L2
is a piecewise-constant "staircase" (bf16 rounding discreteness dominates at very small deltas),
on which linear extrapolation can legally overshoot and oscillate forever. v2's `scoped_apply_
anatomical_perturbation_bf16_bracketed` replaces linear extrapolation, once an
overshoot/undershoot pair is observed, with deterministic BISECTION inside a maintained scalar
bracket (never resampling the direction, only the scalar magnitude) -- see scoped_anatomical_
perturbation.py's own docstring for the full derivation, and `solve_bf16_radius` there for the
pure, directly-unit-tested control-flow logic (including a replay of the exact live oscillating
sequence above). Hard-fails (RadiusCorrectionFailedError) with `quantization_plateau=True` and
the nearest achievable realized values on both sides of the target when no bf16-representable
point is actually within tolerance -- never silently accepted with a looser bound.

`radius_realization_method` ("fixed_direction_bf16_bracketed_v2") was part of both the
run_signature and the checkpoint/run manifest identity, so a checkpoint written under a
DIFFERENT method (v1's "fixed_direction_bf16_corrected_v1", or any earlier one-shot method)
could never be silently resumed under it -- the v1 full-run attempt that produced the
oscillating failure was left on disk, untouched, as provenance.

QUANTIZATION-AWARE ACCEPTANCE -- v3 (this repair pass): v2's bracketed solver was then run for
real on the Stage-7B three-region smallest-radius numerical smoke and produced DECISIVE evidence
that strict 1e-6 is physically unattainable for one specific cell: vision and language both
converged strictly (abs errors 9.91e-7 and 3.46e-7), but the connector region proved a genuine
`quantization_plateau` -- the solver observed an EXACT repeated realized value during bisection,
with the target provably bracketed between two attainable bf16 states whose nearest RELATIVE
error is ~3.52e-4 (0.0352%), comfortably inside a 0.1% admissibility bound. v3 does NOT change
the solver (`solve_bf16_radius` is reused UNCHANGED -- same bisection, same plateau detection);
it adds a strictly narrower two-tier ACCEPTANCE rule on top, implemented in
`scoped_anatomical_perturbation.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3`
(see that module's own docstring for the full derivation):
  - STRICT: `abs(realized - requested) <= 1e-6` -- accepted exactly as v2 did.
  - QUANTIZATION-LIMITED FALLBACK: only when the solver has PROVEN `quantization_plateau=True`
    (never merely "ran out of attempts"), AND the requested radius is bracketed by two
    attainable bf16 states, AND the nearer one's RELATIVE error (a fraction of the requested
    radius -- scale-appropriate across the six-decade-wide frozen radius grid, never an absolute
    number) is `<= 1e-3` (0.1%) -- a NUMERICAL ADMISSIBILITY bound, never an experimental
    hyperparameter and never chosen from any capability/task performance signal. The ACTUAL
    realized radius (never the nominal requested value) is always what gets persisted.
  - Otherwise: HARD FAIL (`RadiusCorrectionFailedError` / `QuantizationToleranceExceededError`).

`radius_realization_method` ("fixed_direction_bf16_quantization_aware_v3") is part of both the
run_signature (EVERY signature is method-suffixed, this repair pass -- including the full
-calibration identity itself, e.g. `full_fixed_direction_bf16_quantization_aware_v3`, so it is
never confused with a bare "full" from an earlier method) and the checkpoint/run manifest
identity, so a checkpoint written under v1 or v2 can never be silently resumed under v3 -- their
artifacts are left on disk, untouched, as provenance; v3 always starts as a scientifically fresh
run under its own run_signature/output_dir.

CACHE LIFECYCLE FIX -- multimodal_cache_policy=full_reset_on_weight_change_v1 (this repair pass):
the v3 full run (144 perturbations/432 rows, commit 0307f99) was analyzed and found to have every
vision- and multimodal_connector_or_merger-region row with delta EXACTLY 0.0 and
per_example_result_hash collapsed to a SINGLE value across all 6 radii x 8 seeds, despite a real,
confirmed nonzero weight displacement (epsilon_region_l2_norm > 0 per candidate) -- see
analysis/stage7b_anatomical_calibration_analysis.py's compute_data_integrity_report for the exact
detection logic and evidence (288 of 432 rows affected). Root cause, confirmed by source
inspection: this module launches its engine via run_global_visual_thicket_pilot.
launch_stage6_engine()/build_stage6_engine_config() -- the exact path GATE2_CACHE_SAFETY_
REVIEW.md analyzed and declared safe ONLY because "the visual encoder is never perturbed" under
Stage 6; Stage 7B perturbs both vision and connector regions, violating that precondition, and
never called vlm_adapter.reset_vllm_encoder_cache_full() anywhere. Fix reuses the EXISTING
mechanism from vlm_adapter.py BY IDENTITY (ensure_full_encoder_cache_reset_exposed /
reset_vllm_encoder_cache_full) -- no new cache-clearing implementation. `radius_realization_
method` is intentionally left UNCHANGED (this is a cache-lifecycle correction, not a
radius-solving change); `multimodal_cache_policy` is tracked as an independent, orthogonal
identity field folded into the run_signature/checkpoint/run-manifest/candidate-runtime-metadata
(EVERY signature, including the full identity, is now BOTH method- and cache-policy-suffixed,
e.g. `full_fixed_direction_bf16_quantization_aware_v3_cache_reset_v1`), so the old no-cache-reset
v3 run (the one analyzed at commit 0307f99, whose vision/connector rows are invalid) can never be
silently resumed into, or confused with, the corrected run -- it is left on disk, untouched, as
provenance. The reset is called TWICE per candidate (see evaluate_one_calibration_candidate_rpc's
own docstring): once immediately after the accepted perturbation and before ANY capability is
evaluated, and once again immediately after the post-candidate fixed-base restoration is
verified -- required for EVERY region type, including language, since the immediately preceding
candidate may have perturbed vision/connector and left candidate-specific embeddings cached; the
lifecycle never relies on candidate ordering. At startup, ensure_stage7b_encoder_cache_reset_
mechanism_exposed() (pre-launch) and ensure_encoder_cache_reset_available() (post-launch, against
the live engine) both HARD FAIL BEFORE EXPERIMENT if the reset mechanism is not reachable/working
-- Stage 7B never silently proceeds without a proven-working cache-invalidation path.
=================================================================================================

=================================================================================================
FROZEN FULL-CALIBRATION PAPER CONFIG (Stage 7A live evidence, model
Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3 -- do not re-derive)
=================================================================================================
regions:      vision, multimodal_connector_or_merger, language (thicket.anatomy L1 regions --
              the SAME common radius grid across all three, never a per-region grid)
radii:        FULL_CALIBRATION_RADII below -- the exact six sigma-to-relative-L2 anchors Stage
              7A measured against the live model (vision params=631,975,680 norm=609.1628871207417;
              connector params=36,708,608 norm=162.99748421626117; language params=3,085,938,688
              norm=1556.1078070923304). Stage 7A's own empirical sigma=.001 raw-Gaussian check
              (analytical r=0.035698828543799424, realized r=0.035691540989192055, abs diff
              7.29e-06, restoration_exact=true) used RAW sigma, not the exact-rescale
              anatomical_relative_l2 mode this module uses -- that tiny gap is expected and is
              NOT the invariant this module enforces; see REALIZED_RADIUS_TOLERANCE below.
capabilities: visual_grounding, ocr_text_recognition_grounded, spatial_reasoning
population:   FULL_CALIBRATION_N_PER_CELL (8) perturbations per (region, radius) cell
data:         a deterministic FULL_CALIBRATION_D_MAP_N (20)-example D_map subset per capability
Total:        3 regions x 6 radii x 8 seeds = 144 unique anatomical perturbations
              144 x 3 capabilities = 432 perturbation x capability result rows
No D_confirm/select/test is ever constructed or referenced anywhere in this module
(data_roles.partition_data_roles is only ever called with sizes={"map": n}, exactly like Stage
6's own build_d_map_context).

=================================================================================================
SMOKE MODE (this repair pass) -- EXECUTION SIZE ONLY, same scientific definitions
=================================================================================================
`--smoke` overrides ONLY region/radius/population-size/D_map-size to SMOKE_REGION ("vision"),
SMOKE_RADIUS (exactly FULL_CALIBRATION_RADII[2] == 0.035698828543799424), SMOKE_N_PER_CELL (1),
SMOKE_D_MAP_N (5) -- all 3 frozen capabilities are still evaluated. Expected: 1 unique
perturbation, 3 result rows, 15 perturbed model-example evaluations, plus 3 baseline
evaluations (5 examples x 3 capabilities). Smoke output is written under its own
`run_signature` (never "full" -- see `compute_stage7b_run_signature`), so it can never collide
with, resume as, or be mistaken for full-calibration output.

=================================================================================================
ENGINE (this repair pass -- regression fix, see diagnostics/anatomy_inventory_gpu.py's own
docstring for the identical RunPod KV-cache-OOM this avoids): reuses
run_global_visual_thicket_pilot.launch_stage6_engine (max_model_len=4096,
gpu_memory_utilization=0.60, tensor_parallel_size=1, bfloat16, enforce_eager=True,
worker_extension_cls=utils.worker_extn.WorkerExtension, immutable resolved snapshot) --
external/RandOpt's own launch_engines() (no max_model_len parameter, defaults to the real
model's 128000-token context) is NEVER called. `store_base_weights` is called EXACTLY ONCE,
explicitly, immediately after engine launch -- for both smoke and full runs.
=================================================================================================

Lifecycle per candidate (fixed-base, same restoration discipline as Stage 6):
    [inside scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3, per solver trial:]
        reset_to_base_weights() -- from the frozen theta_0 snapshot, every trial
        -> apply the SAME fixed seeded direction at the trial's scalar magnitude
        -> verify outside-region parameters are EXACTLY unchanged (max_abs_drift == 0.0)
        -> measure the TRUE (post-BF16-add) realized relative-L2; accept strictly, or
           bisect/proportion-correct and retry (see solve_bf16_radius)
    [if strict convergence fails but a plateau is PROVEN and the nearest attainable bf16 state's
     RELATIVE error is <= 0.1%: reset -> reapply the selected scalar -> verify EXACT reproduction
     -> verify outside-region invariance again -- only then is the fallback state accepted]
    -> (accepted, strict or quantization-limited) evaluate all 3 capabilities' D_map subsets, in
       fixed order -- the accepted state's weights are exactly what remains loaded, never
       reconstructed afterward
    -> reset_to_base_weights()
    -> verify EXACT fixed-base restoration (reused from run_global_visual_thicket_pilot)
    -> append candidate rows (checkpointed -- same append-only, resume-safe discipline as Stage 6)

Hard-fails (never silently continues, never loosened for smoke) on:
    - the solver failing to reach REALIZED_RADIUS_TOLERANCE within MAX_RADIUS_SOLVER_ITERATIONS
      trials (RadiusCorrectionFailedError, evidence includes quantization_plateau and the
      nearest achievable realized values on both sides of the target when detected)
    - any out-of-region parameter drift on ANY solver trial, not just the accepted one
      (CorrectionOutOfRegionDriftError)
    - a failed exact fixed-base restoration verification (RestorationFailedError)

=================================================================================================
KNOWN GAP / TODO (Stage 7A finding -- NOT a blocker for this L1 calibration; recorded here for
the later L2 hierarchical language drill-down, per instruction, without modifying L1 anatomy):
=================================================================================================
Stage 7A's live inventory showed `language_early`/`language_middle`/`language_late` (the L2
depth-band partition of the `language` L1 region) do NOT cover two `language` parameters:
    language_model.model.embed_tokens.weight
    language_model.model.norm.weight
This is thicket.anatomy.validate_atlas's own already-reported, non-fatal
`uncovered_by_parent["language"]` finding (embeddings/final-norm sit outside any numbered
decoder layer, by construction of the depth-band partition rule) -- it does not affect L1
region membership or this module's L1-only calibration in any way. Before any later L2
hierarchical language drill-down assigns radii/experts to language_early/middle/late
specifically, these two tensors' treatment (e.g. an explicit fourth "language_embeddings_and_
norm" band, or folding them into language_early) must be decided and documented -- NOT done
here, and L1 anatomy (thicket/anatomy.py) is NOT modified by this task.
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
    CAPABILITY_CONFIG_FILES,
    PILOT_CAPABILITIES,
    RESTORATION_MODE,
    CapabilityContext,
    RayEngineLLMAdapter,
    RestorationFailedError,
    append_candidate_rows,
    build_d_map_context,
    build_stage6_engine_config,
    detect_vllm_engine_mode,
    format_base_snapshot_confirmation,
    format_runtime_compatibility_diagnostic,
    get_vllm_version,
    launch_stage6_engine,
    load_completed_perturbation_rows,
    load_or_compute_baseline_scores,
    load_records,
    reset_to_base_weights_via_rpc,
    resolve_and_report_model_snapshot,
    store_base_weights_via_rpc,
    verify_exact_fixed_base_restoration_via_rpc,
)
from .scoped_anatomical_perturbation import (
    QUANTIZATION_AWARE_METHOD_V3,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    CorrectionOutOfRegionDriftError,
    QuantizationToleranceExceededError,
    RadiusCorrectionFailedError,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
)
from .thicket.anatomy import build_anatomy_atlas
from .thicket.perturbation import PERTURBATION_MODES, PerturbationManifest, generate_perturbation_population, validate_unique_worker_seeds
from .thicket.schema import ExperimentResultRecord
from .vlm_adapter import ensure_full_encoder_cache_reset_exposed, reset_vllm_encoder_cache_full

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

EXPERIMENT_ID = "stage7b_anatomical_calibration"
PERTURBATION_MODE = "anatomical_relative_l2"
assert PERTURBATION_MODE in PERTURBATION_MODES

STAGE7B_BASE_SEED = 20260824  # distinct from Stage 6's own base_seed (20260823); deterministic, fixed

# --- FROZEN full-calibration paper config -- Stage 7A live evidence, never re-derived here ----
FULL_CALIBRATION_REGIONS: Tuple[str, ...] = ("vision", "multimodal_connector_or_merger", "language")
FULL_CALIBRATION_RADII: Tuple[float, ...] = (
    0.0035698828543799426, 0.017849414271899712, 0.035698828543799424,
    0.07139765708759885, 0.1784941427189971, 0.3569882854379942,
)
CALIBRATION_CAPABILITIES: Tuple[str, ...] = PILOT_CAPABILITIES  # identical for smoke and full -- never overridden
FULL_CALIBRATION_N_PER_CELL = 8
FULL_CALIBRATION_D_MAP_N = 20
DATASET_ROLE = "map"  # the ONLY role this module ever constructs or references

# --- SMOKE mode: EXECUTION SIZE ONLY, same regions/radii vocabulary and same scientific rules -
SMOKE_REGION = "vision"
SMOKE_RADIUS = FULL_CALIBRATION_RADII[2]  # exactly 0.035698828543799424, the frozen value the task specified
SMOKE_N_PER_CELL = 1
SMOKE_D_MAP_N = 5

_ALLOWED_D_MAP_SIZES: Tuple[int, ...] = (FULL_CALIBRATION_D_MAP_N, SMOKE_D_MAP_N)

# The realization method this run uses -- reused BY IDENTITY from scoped_anatomical_
# perturbation.py, never redefined here. Folded into the run_signature (see
# compute_stage7b_run_signature -- EVERY signature, including the full-calibration identity, is
# method-suffixed, this repair pass) and the checkpoint/run manifest identity, so a checkpoint
# written under a different method (v1's proportional-only corrector, v2's bracketed-but-strict
# -only solver, or any earlier one-shot method) is never silently resumed under this one (v3,
# the quantization-aware two-tier acceptance rule).
RADIUS_REALIZATION_METHOD = QUANTIZATION_AWARE_METHOD_V3

# Tolerance for the exact-rescale realized-vs-requested relative-L2 check -- matches the
# tolerance already established by tests/test_thicket_perturbation.py's
# test_anatomical_relative_l2_hits_requested_ratio_exactly for the same underlying primitive,
# and scoped_anatomical_perturbation.RADIUS_REALIZATION_TOLERANCE. NEVER loosened for smoke.
REALIZED_RADIUS_TOLERANCE = 1e-6

# --- CACHE LIFECYCLE FIX (v1, prior repair pass) -------------------------------------------------
# Root cause (Stage 7B v3 full-run analysis, commit 0307f99): every vision- and connector-region
# row had delta EXACTLY 0.0 and a per_example_result_hash collapsed to a single value across all
# 6 radii x 8 seeds, despite a real, confirmed nonzero weight displacement -- vLLM's cached
# multimodal-encoder output for the fixed image inputs was never invalidated after an
# anatomical perturbation, so vision/connector-perturbed generation silently reused the BASE
# model's cached image embeddings. GATE2_CACHE_SAFETY_REVIEW.md's "safe by construction" analysis
# for launch_stage6_engine() depended entirely on "the visual encoder is never perturbed", which
# Stage 7B violates. v1's fix wired vlm_adapter.py's reset_vllm_encoder_cache_full into the
# candidate lifecycle -- correct in SHAPE, but its own actor-side implementation called
# `self.llm_engine.reset_encoder_cache()` directly, which does not exist on the pinned vLLM
# 0.11.0 runtime (confirmed live: `AttributeError: 'LLMEngine' object has no attribute
# 'reset_encoder_cache'`, commit 74f273b's cache-safety smoke). `radius_realization_method`
# (fixed_direction_bf16_quantization_aware_v3) remains INTENTIONALLY unchanged across both cache
# -policy versions -- these are cache-lifecycle corrections, never radius-solving changes.
#
# --- CACHE LIFECYCLE FIX v2: pinned-vLLM-0.11.0 VERIFIED reset (this repair pass) -----------------
# Reuses vlm_adapter.py's reset_vllm_encoder_cache_full/ensure_full_encoder_cache_reset_exposed
# BY IDENTITY, unchanged at this call site -- the version-aware, verified 4-layer reset (native
# reset_encoder_cache() where available, else an explicit pinned-v0.11.0 reproduction covering
# frontend/processor cache, engine MM receiver cache, the scheduler's own EncoderCacheManager
# bookkeeping, AND every GPU worker's physical encoder_cache tensors, each layer independently
# verified before/after -- never merely "no exception raised") now lives entirely inside
# vlm_adapter.py itself (see that module's own docstring, divergence #9). Tracked as its own
# cache-policy VERSION (never the same string as v1) precisely because the cache SEMANTICS
# changed again -- a checkpoint/run written under v1 (which called an API that does not exist on
# this runtime and would have hard-failed immediately, never producing any real rows) can never
# be silently resumed under v2.
MULTIMODAL_CACHE_POLICY = "full_encoder_reset_vllm011_verified_v2"

# Deterministic, explicit short label for the run_signature -- NEVER auto-abbreviated (a missing
# entry hard-fails via _format_cache_policy_for_signature rather than guessing a truncation).
# v1's entry is kept (never removed) purely so a v1-identified checkpoint dict, if one ever
# existed, still resolves to a disjoint signature rather than a KeyError.
_CACHE_POLICY_SIGNATURE_LABELS: Dict[str, str] = {
    "full_reset_on_weight_change_v1": "cache_reset_v1",
    MULTIMODAL_CACHE_POLICY: "cache_reset_v011_verified_v2",
}

_UNKNOWN_LEGACY_CACHE_POLICY = "unknown_pre_cache_reset_legacy"  # sentinel for a checkpoint written before this field existed at all -- NEVER equal to any real MULTIMODAL_CACHE_POLICY value

# --- PREFIX-CACHE SAFETY (this repair pass) -------------------------------------------------
# Live Stage-7B logs showed enable_prefix_caching=True (vLLM's own default, never overridden by
# the shared launch_stage6_engine() call this module used) -- unsafe across Stage 7B's repeated
# weight-mutation candidate loop, since decoder KV prefixes may have been computed under a
# PREVIOUS candidate's now-stale weights. Frozen to False for every Stage 7B run (never a
# per-run knob); disabling entirely is preferred over resetting it candidate-by-candidate.
# Stage 6 is UNAFFECTED: launch_stage6_engine()'s own enable_prefix_caching parameter is
# additive/opt-in (None by default, omitted from engine_kwargs entirely), so Stage 6's frozen
# behavior is untouched -- see that function's own docstring.
ENABLE_PREFIX_CACHING = False


def _format_cache_policy_for_signature(cache_policy: str) -> str:
    try:
        return _CACHE_POLICY_SIGNATURE_LABELS[cache_policy]
    except KeyError:
        raise ValueError(
            f"No run_signature label registered for multimodal_cache_policy={cache_policy!r} -- "
            f"add one to _CACHE_POLICY_SIGNATURE_LABELS explicitly rather than guessing an "
            f"abbreviation."
        )


class EncoderCacheResetUnavailableError(RuntimeError):
    """The full multimodal-encoder-cache reset mechanism (vlm_adapter.
    ensure_full_encoder_cache_reset_exposed / reset_vllm_encoder_cache_full) is not reachable --
    either the monkey-patch that exposes it on the Ray actor class failed before engine launch,
    or a live, post-launch verification call against the running engine failed. Hard-fails
    BEFORE any candidate work begins: Stage 7B perturbs vision/connector regions, and an
    unreachable cache reset would silently reproduce the exact stale-encoder-cache artifact
    MULTIMODAL_CACHE_POLICY exists to prevent.
    """


def ensure_stage7b_encoder_cache_reset_mechanism_exposed(external_root: "str | Path" = EXTERNAL_ROOT) -> None:
    """MUST be called BEFORE the engine is launched (mirrors vlm_adapter.
    ensure_full_encoder_cache_reset_exposed's own requirement -- it monkey-patches RandOptNcclLLM
    so Ray's actor-method registry includes reset_encoder_cache_full when launch_engines() wraps
    it). Wraps any failure in EncoderCacheResetUnavailableError for a single, uniform "cache
    reset unreachable, hard fail before experiment" error type across both the pre-launch and
    post-launch (see ensure_encoder_cache_reset_available below) checks.
    """
    try:
        ensure_full_encoder_cache_reset_exposed(external_root)
    except Exception as exc:  # noqa: BLE001
        raise EncoderCacheResetUnavailableError(
            f"Stage 7B requires vlm_adapter.ensure_full_encoder_cache_reset_exposed() to succeed "
            f"BEFORE the engine is launched -- failed with {type(exc).__name__}: {exc}. Refusing "
            f"to start Stage 7B (multimodal_cache_policy={MULTIMODAL_CACHE_POLICY!r}) without a "
            f"reachable cache-invalidation path."
        ) from exc


def ensure_encoder_cache_reset_available(engine: Any) -> None:
    """Proves the full cache-reset mechanism actually WORKS end-to-end against the LIVE, already
    -launched engine (not merely that the attribute was exposed pre-launch) -- catches a
    real-world mismatch (e.g. an installed vLLM version whose LLMEngine no longer exposes
    reset_encoder_cache()) that a static attribute check alone would miss. Called once, right
    after store_base_weights_via_rpc, BEFORE any candidate evaluation begins.
    """
    try:
        reset_vllm_encoder_cache_full(engine)
    except Exception as exc:  # noqa: BLE001
        raise EncoderCacheResetUnavailableError(
            f"Stage 7B requires a working full multimodal-encoder-cache reset "
            f"(multimodal_cache_policy={MULTIMODAL_CACHE_POLICY!r}) -- reset_vllm_encoder_cache_full "
            f"failed against the live engine ({type(exc).__name__}: {exc}). Refusing to start "
            f"Stage 7B candidate evaluation without a proven-working cache-invalidation path, "
            f"since vision/connector perturbations would otherwise silently reuse stale cached "
            f"multimodal-encoder output -- the exact bug this policy exists to prevent."
        ) from exc


class RealizedRadiusMismatchError(RuntimeError):
    """Defensive, mode-aware re-check (no extra RPC round trip) on the ALREADY accepted result:
    for radius_acceptance_mode="strict", the realized relative-L2 did not match the requested
    radius within REALIZED_RADIUS_TOLERANCE; for "quantization_limited", the RELATIVE error did
    not stay within QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE. Should never actually fire in
    practice, since scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 itself
    already guarantees one of these two bounds (or raises RadiusCorrectionFailedError /
    QuantizationToleranceExceededError) before returning; kept as a second, independent layer
    rather than trusting a single check.
    """


class DatasetRoleViolationError(RuntimeError):
    """Something other than the 'map' dataset role was requested, or an unrecognized D_map
    size was requested -- this module must never construct or reference D_confirm/D_select/
    D_test, and must never silently accept an ad-hoc dataset size beyond the two frozen ones
    (full=20, smoke=5).
    """


def _format_radius_for_signature(radius: float) -> str:
    """Deterministic, collision-resistant short label for a radius value, used only to build a
    human-readable run_signature -- never used for any numeric comparison.
    """
    return f"{radius:.6f}".replace(".", "")


def compute_stage7b_run_signature(
    regions: Sequence[str], radii: Sequence[float], n_per_cell: int, d_map_n: int,
    radius_realization_method: str = RADIUS_REALIZATION_METHOD,
    multimodal_cache_policy: str = MULTIMODAL_CACHE_POLICY,
) -> str:
    """`f"full_{radius_realization_method}_{cache_label}"` iff regions/radii/n_per_cell/d_map_n
    exactly match the frozen full-calibration identity (both the radius-realization METHOD and
    the multimodal-CACHE-POLICY are suffixed onto the signature -- no bare "full" literal exists
    anymore, exactly so a v3+cache-reset-v1 run's output directory is
    `full_fixed_direction_bf16_quantization_aware_v3_cache_reset_v1` and can never be confused
    with, or resumed from, an older method's OR an older/absent cache-policy's own full-run
    directory -- in particular the v3-no-cache-reset run analyzed at commit 0307f99, whose
    vision/connector rows are scientifically invalid, lives under a permanently different
    `full_fixed_direction_bf16_quantization_aware_v3` directory and is never touched by this
    signature); otherwise a deterministic "smoke_..." descriptive string built from the ACTUAL
    values (never a fixed literal string), also suffixed with both labels. This is what makes it
    impossible for a failed/partial smoke run -- OR a run made under an OLDER realization method
    OR an OLDER/absent cache policy -- to ever be resumed as (or mistaken for) this run: they
    always live under different `output_dir`s.
    """
    cache_label = _format_cache_policy_for_signature(multimodal_cache_policy)
    if (
        tuple(regions) == FULL_CALIBRATION_REGIONS and tuple(radii) == FULL_CALIBRATION_RADII
        and n_per_cell == FULL_CALIBRATION_N_PER_CELL and d_map_n == FULL_CALIBRATION_D_MAP_N
    ):
        return f"full_{radius_realization_method}_{cache_label}"
    region_label = "-".join(regions)
    radius_label = "-".join(_format_radius_for_signature(r) for r in radii)
    return f"smoke_{region_label}_r{radius_label}_n{d_map_n}_p{n_per_cell}_{radius_realization_method}_{cache_label}"


# =============================================================================================
# Plan + population (pure arithmetic, no I/O, no GPU)
# =============================================================================================


@dataclass(frozen=True)
class Stage7bPlan:
    model_name: str
    model_revision: str
    model_family: str
    model_scale: str
    regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_per_cell: int
    d_map_n: int
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: bool
    run_signature: str
    output_dir: Path

    @property
    def total_unique_perturbations(self) -> int:
        return len(self.regions) * len(self.radii) * self.n_per_cell

    @property
    def total_perturbation_capability_evaluations(self) -> int:
        return self.total_unique_perturbations * len(self.capabilities)

    @property
    def is_smoke(self) -> bool:
        """Checked against the actual FIELDS, not the run_signature string (which is always
        method-suffixed and therefore never a fixed literal to compare against) -- true iff any
        of regions/radii/n_per_cell/d_map_n deviates from the frozen full-calibration identity.
        """
        return not (
            self.regions == FULL_CALIBRATION_REGIONS and self.radii == FULL_CALIBRATION_RADII
            and self.n_per_cell == FULL_CALIBRATION_N_PER_CELL and self.d_map_n == FULL_CALIBRATION_D_MAP_N
        )


def build_stage7b_plan(
    *, model_name: str, model_revision: str, output_root: "str | Path",
    regions: Sequence[str] = FULL_CALIBRATION_REGIONS, radii: Sequence[float] = FULL_CALIBRATION_RADII,
    n_per_cell: int = FULL_CALIBRATION_N_PER_CELL, d_map_n: int = FULL_CALIBRATION_D_MAP_N,
    model_family: str = "qwen2_5_vl", model_scale: str = "3B",
    radius_realization_method: str = RADIUS_REALIZATION_METHOD,
    multimodal_cache_policy: str = MULTIMODAL_CACHE_POLICY,
    enable_prefix_caching: bool = ENABLE_PREFIX_CACHING,
) -> Stage7bPlan:
    """Defaults to the FROZEN full-calibration identity exactly; pass explicit `regions`/
    `radii`/`n_per_cell`/`d_map_n` (see `build_smoke_stage7b_plan`/
    `build_cache_safety_smoke_stage7b_plan`) to override execution size only -- never used to
    invent new scientific definitions (radii/regions callers pass here must still come from this
    module's own frozen constants). `radius_realization_method`/`multimodal_cache_policy` are
    not knobs real callers override (there is only ONE of each implemented) -- exposed as
    parameters purely so tests can construct a plan under a hypothetical different/old
    method/policy to prove it never collides with this one's run_signature or checkpoint
    identity. `enable_prefix_caching` is likewise frozen to False for every real Stage 7B run
    (see ENABLE_PREFIX_CACHING's own comment) -- exposed only for the same test-isolation reason.
    """
    if not regions:
        raise ValueError("Stage 7B requires at least one anatomy region.")
    if not radii:
        raise ValueError("Stage 7B requires at least one calibration radius.")
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 7B D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")

    run_signature = compute_stage7b_run_signature(regions, radii, n_per_cell, d_map_n, radius_realization_method, multimodal_cache_policy)
    return Stage7bPlan(
        model_name=model_name, model_revision=model_revision, model_family=model_family, model_scale=model_scale,
        regions=tuple(regions), radii=tuple(radii), capabilities=CALIBRATION_CAPABILITIES,
        n_per_cell=n_per_cell, d_map_n=d_map_n, radius_realization_method=radius_realization_method,
        multimodal_cache_policy=multimodal_cache_policy, enable_prefix_caching=enable_prefix_caching,
        run_signature=run_signature, output_dir=Path(output_root) / run_signature,
    )


def build_smoke_stage7b_plan(*, model_name: str, model_revision: str, output_root: "str | Path") -> Stage7bPlan:
    """The one frozen EXECUTION-size smoke configuration this task defines: region=vision,
    radius=SMOKE_RADIUS (exactly FULL_CALIBRATION_RADII[2]), 1 perturbation, 5 D_map examples/
    capability, all 3 frozen capabilities. Execution size only -- same scientific protocol.
    See `build_cache_safety_smoke_stage7b_plan` for the SEPARATE, 3-region cache-lifecycle smoke.
    """
    return build_stage7b_plan(
        model_name=model_name, model_revision=model_revision, output_root=output_root,
        regions=(SMOKE_REGION,), radii=(SMOKE_RADIUS,), n_per_cell=SMOKE_N_PER_CELL, d_map_n=SMOKE_D_MAP_N,
    )


def build_cache_safety_smoke_stage7b_plan(*, model_name: str, model_revision: str, output_root: "str | Path") -> Stage7bPlan:
    """CACHE-SAFETY SMOKE (this repair pass): validates the corrected cache-invalidation
    lifecycle across ALL THREE region types in one small GPU-cheap run, before committing to
    another full 144-candidate/432-row run. All 3 frozen regions x radius=SMOKE_RADIUS (exactly
    FULL_CALIBRATION_RADII[2]) x 1 perturbation/region x all 3 frozen capabilities x SMOKE_D_MAP_N
    (5) examples/capability = 3 unique perturbations, 9 result rows, 45 perturbed
    model-example evaluations. This is EXECUTION/INSTRUMENTATION validation only -- it does NOT
    require Delta != 0 as a correctness invariant (a valid perturbation can legitimately leave
    N=5 predictions unchanged); the pass condition is that the cache-reset lifecycle actually ran
    (see runtime_metadata's cache_reset_before_evaluation/cache_reset_after_restoration fields on
    every persisted row), not any particular behavioral outcome. Structurally distinct
    run_signature/output_dir from both the single-region execution smoke
    (`build_smoke_stage7b_plan`) and the full calibration run, since it spans all 3 regions but
    n_per_cell=1/d_map_n=5 (never matches the frozen full-calibration identity).
    """
    return build_stage7b_plan(
        model_name=model_name, model_revision=model_revision, output_root=output_root,
        regions=FULL_CALIBRATION_REGIONS, radii=(SMOKE_RADIUS,), n_per_cell=1, d_map_n=SMOKE_D_MAP_N,
    )


def build_stage7b_population(
    plan: Stage7bPlan, base_seed: int, parameter_mask_hash_by_region: Dict[str, str],
) -> Dict[Tuple[str, float], Tuple[PerturbationManifest, ...]]:
    """One population of `plan.n_per_cell` PerturbationManifests per (region, radius) cell.
    `parameter_mask_hash_by_region` must have an entry for every region in `plan.regions` (the
    region's own AnatomyRegion.mask_hash, from the live model's atlas) -- a perturbation's
    identity is tied to the exact parameter set it can touch, not just its region NAME.
    Validates worker-seed uniqueness across the ENTIRE combined population (every cell
    together), same discipline as Stage 6's build_stage6_perturbation_population.
    """
    missing_regions = set(plan.regions) - set(parameter_mask_hash_by_region)
    if missing_regions:
        raise ValueError(f"Missing parameter_mask_hash for region(s): {sorted(missing_regions)}")

    population_by_cell: Dict[Tuple[str, float], Tuple[PerturbationManifest, ...]] = {}
    all_manifests: List[PerturbationManifest] = []
    for region in plan.regions:
        mask_hash = parameter_mask_hash_by_region[region]
        for radius in plan.radii:
            population = generate_perturbation_population(
                mode=PERTURBATION_MODE, n=plan.n_per_cell, base_seed=base_seed,
                model_family=plan.model_family, model_scale=plan.model_scale, model_revision=plan.model_revision,
                parameter_mask_hash=mask_hash, anatomy_region=region, radius=radius, sigma=None,
            )
            population_by_cell[(region, radius)] = population
            all_manifests.extend(population)

    validate_unique_worker_seeds(all_manifests)
    return population_by_cell


# =============================================================================================
# Checkpoint identity (mirrors run_global_visual_thicket_pilot.CheckpointManifest's discipline)
# =============================================================================================


class IncompatibleCalibrationCheckpointError(RuntimeError):
    """An existing checkpoint_manifest.json in this output directory does not match the
    current run's identity -- hard-fails rather than silently resuming a differently
    -configured partial run. Since smoke and full always compute DIFFERENT `run_signature`s
    (and therefore different output_dir paths), this can only fire for a genuine
    re-configuration of the SAME run_signature's directory, never a smoke-vs-full mixup.
    """


_UNKNOWN_LEGACY_REALIZATION_METHOD = "unknown_pre_correction_legacy"  # sentinel for a checkpoint written before this field existed -- NEVER equal to RADIUS_REALIZATION_METHOD, so from_dict never silently treats an old checkpoint as compatible


@dataclass(frozen=True)
class Stage7bCheckpointManifest:
    experiment_id: str
    run_signature: str
    restoration_mode: str
    perturbation_mode: str
    radius_realization_method: str
    multimodal_cache_policy: str
    enable_prefix_caching: Optional[bool]
    model_revision: str
    dataset_role: str
    regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_per_cell: int
    d_map_n: int
    subset_hashes: Dict[str, str]
    region_mask_hashes: Dict[str, str]
    expected_unique_perturbations: int
    expected_result_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "run_signature": self.run_signature,
            "restoration_mode": self.restoration_mode, "perturbation_mode": self.perturbation_mode,
            "perturbation_semantics": self.perturbation_mode, "radius_realization_method": self.radius_realization_method,
            "multimodal_cache_policy": self.multimodal_cache_policy, "enable_prefix_caching": self.enable_prefix_caching,
            "model_revision": self.model_revision, "dataset_role": self.dataset_role, "regions": list(self.regions),
            "radii": list(self.radii), "capabilities": list(self.capabilities), "n_per_cell": self.n_per_cell, "d_map_n": self.d_map_n,
            "subset_hashes": dict(sorted(self.subset_hashes.items())),
            "region_mask_hashes": dict(sorted(self.region_mask_hashes.items())),
            "expected_unique_perturbations": self.expected_unique_perturbations,
            "expected_result_rows": self.expected_result_rows,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage7bCheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], run_signature=d["run_signature"], restoration_mode=d["restoration_mode"],
            perturbation_mode=d["perturbation_mode"],
            radius_realization_method=d.get("radius_realization_method", _UNKNOWN_LEGACY_REALIZATION_METHOD),
            multimodal_cache_policy=d.get("multimodal_cache_policy", _UNKNOWN_LEGACY_CACHE_POLICY),
            enable_prefix_caching=d.get("enable_prefix_caching"),  # None (distinct from True/False) for a checkpoint written before this field existed
            model_revision=d["model_revision"], dataset_role=d["dataset_role"],
            regions=tuple(d["regions"]), radii=tuple(d["radii"]), capabilities=tuple(d["capabilities"]),
            n_per_cell=d["n_per_cell"], d_map_n=d["d_map_n"], subset_hashes=dict(d["subset_hashes"]),
            region_mask_hashes=dict(d.get("region_mask_hashes", {})),
            expected_unique_perturbations=d["expected_unique_perturbations"], expected_result_rows=d["expected_result_rows"],
        )


def build_stage7b_checkpoint_manifest(
    plan: Stage7bPlan, capability_contexts: Dict[str, CapabilityContext], region_mask_hashes: Dict[str, str],
) -> Stage7bCheckpointManifest:
    if plan.d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 7B D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {plan.d_map_n}")
    missing_regions = set(plan.regions) - set(region_mask_hashes)
    if missing_regions:
        raise ValueError(f"Missing region_mask_hashes for region(s): {sorted(missing_regions)}")
    return Stage7bCheckpointManifest(
        experiment_id=EXPERIMENT_ID, run_signature=plan.run_signature, restoration_mode=RESTORATION_MODE,
        perturbation_mode=PERTURBATION_MODE, radius_realization_method=plan.radius_realization_method,
        multimodal_cache_policy=plan.multimodal_cache_policy, enable_prefix_caching=plan.enable_prefix_caching,
        model_revision=plan.model_revision, dataset_role=DATASET_ROLE,
        regions=plan.regions, radii=plan.radii, capabilities=plan.capabilities, n_per_cell=plan.n_per_cell, d_map_n=plan.d_map_n,
        subset_hashes={c: ctx.subset_hash for c, ctx in capability_contexts.items()},
        region_mask_hashes={r: region_mask_hashes[r] for r in plan.regions},
        expected_unique_perturbations=plan.total_unique_perturbations,
        expected_result_rows=plan.total_perturbation_capability_evaluations,
    )


def ensure_stage7b_checkpoint_manifest(path: Path, current: Stage7bCheckpointManifest) -> Stage7bCheckpointManifest:
    if path.exists():
        existing = Stage7bCheckpointManifest.from_dict(json.loads(path.read_text()))
        if existing != current:
            raise IncompatibleCalibrationCheckpointError(
                f"Existing checkpoint at {path} is incompatible with this run -- refusing to "
                f"resume: existing={existing.to_dict()} current={current.to_dict()}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current.to_dict(), indent=2))
    return current


def build_stage7b_run_manifest_summary(checkpoint: Stage7bCheckpointManifest, records: List[ExperimentResultRecord]) -> Dict[str, Any]:
    """The full-run-safety accounting (section 5): expected vs actual candidate/row counts,
    every identity field a resume/audit needs, and `run_complete` -- the SAME
    accounting-before-trusting-a-run discipline as Stage 6's build_run_manifest_summary.
    """
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
        "regions": list(checkpoint.regions), "radii": list(checkpoint.radii), "n_per_cell": checkpoint.n_per_cell,
        "d_map_n": checkpoint.d_map_n, "subset_hashes": dict(sorted(checkpoint.subset_hashes.items())),
        "region_mask_hashes": dict(sorted(checkpoint.region_mask_hashes.items())),
        "expected_unique_perturbations": checkpoint.expected_unique_perturbations,
        "actual_unique_perturbations": actual_unique_perturbations,
        "expected_result_rows": checkpoint.expected_result_rows, "actual_result_rows": actual_result_rows,
        "run_complete": run_complete,
    }


def write_stage7b_run_manifest(output_dir: Path) -> Dict[str, Any]:
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_manifest.json found at {checkpoint_path} -- this run never started (or output_dir is wrong).")
    checkpoint = Stage7bCheckpointManifest.from_dict(json.loads(checkpoint_path.read_text()))
    records = load_records(output_dir / "results.jsonl")
    manifest = build_stage7b_run_manifest_summary(checkpoint, records)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_d_map_capability_contexts(
    base_seed: int, subset_ids_dir: "str | Path", d_map_n: int, *, load_capability_benchmark_config: Callable, load_adapter: Callable,
) -> Dict[str, CapabilityContext]:
    """Builds the fixed 3-capability D_map context set at the given size (FULL_CALIBRATION_D_MAP_N
    for a full run, SMOKE_D_MAP_N for smoke) -- the ONLY dataset role this module ever touches.
    `load_capability_benchmark_config`/`load_adapter` are injected (rather than imported at
    module top) purely so this stays importable/testable without pulling in the full
    benchmark-gate config machinery at import time; real callers pass
    `.config.load_capability_benchmark_config` / `.run_capability_benchmark_gate.load_adapter`.
    """
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 7B D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    contexts: Dict[str, CapabilityContext] = {}
    for capability in CALIBRATION_CAPABILITIES:  # fixed order, matches Stage 6's own required order
        cfg_path = REPO_ROOT / "configs" / "benchmarks" / CAPABILITY_CONFIG_FILES[capability]
        cfg = load_capability_benchmark_config(cfg_path)
        benchmark = load_adapter(cfg.dataset.adapter)
        ctx = build_d_map_context(benchmark, cfg, capability, d_map_n, base_seed, subset_ids_dir)
        contexts[capability] = ctx
    return contexts


# =============================================================================================
# Worker-RPC transport (same TP=1 list-unwrap convention as every other GPU script here)
# =============================================================================================


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list of per-worker results. Got: {results!r}")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; Stage 7B is TP=1-only and expects exactly 1.")
    return results[0]


def _collective_rpc_single_worker(engine: Any, method, args: tuple = (), *, label: str, ray_get: Optional[Callable] = None) -> Any:
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def report_region_param_names(worker_self, regions: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Runs entirely inside the worker process: builds the atlas from the model's REAL
    named_parameters() and returns, for each requested L1 region, its exact parameter-name list
    and stable mask_hash -- never touches or perturbs any weight. L1 anatomy (thicket/anatomy.py)
    itself is unmodified; this is purely a read-only report over its output.
    """
    model = worker_self.model_runner.model
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names, model_family="qwen2_5_vl")
    return {region: {"param_names": list(atlas.region(region).param_names), "mask_hash": atlas.region(region).mask_hash} for region in regions}


# =============================================================================================
# Per-candidate lifecycle
# =============================================================================================


def evaluate_one_calibration_candidate_rpc(
    engine: Any, manifest: PerturbationManifest, region_param_names: Sequence[str],
    capability_contexts: Dict[str, CapabilityContext], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable] = None,
) -> List[ExperimentResultRecord]:
    """`run_benchmark` is injected (real callers pass benchmarks.runner.run_benchmark) so this
    function stays testable against a fake engine + fake benchmark runner with zero GPU/ray.

    Dispatches scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 -- the ENTIRE
    reset -> apply -> measure-TRUE-realized-radius -> accept(strict)-or-bisect/proportion-retry
    -or-accept(quantization-limited fallback, PROVEN plateau only)-or-hard-fail sequence happens
    inside THAT single RPC call, entirely before this function ever reaches the
    capability-evaluation loop below (no capability evaluation occurs before radius acceptance),
    and the accepted state's weights remain loaded exactly as-is (see that function's own
    docstring -- including, for the fallback path, its own explicit reset/reapply/verify
    -reproduction sequence). It raises RadiusCorrectionFailedError / QuantizationToleranceExceededError
    / CorrectionOutOfRegionDriftError on any unrecovered violation -- this function's own
    RealizedRadiusMismatchError check afterward is a defensive, no-extra-RPC re-verification of
    the already-accepted result, MODE-AWARE (strict re-checks the absolute 1e-6 bound;
    quantization_limited re-checks the relative 0.1% bound -- an absolute-1e-6 re-check would
    incorrectly reject every legitimate quantization-limited acceptance, which by definition can
    exceed 1e-6 absolute error), not the primary enforcement.

    CACHE LIFECYCLE (this repair pass, multimodal_cache_policy=full_reset_on_weight_change_v1 --
    see the module-level constant's own docstring for the confirmed root cause this fixes):
    reset_vllm_encoder_cache_full(engine) is called TWICE per candidate -- once immediately after
    the accepted perturbation and BEFORE any capability is evaluated (so vLLM's cached
    multimodal-encoder output for the fixed image inputs can never be reused from the stale,
    pre-perturbation weight state), and once again immediately after the post-candidate
    fixed-base restoration is verified, BEFORE returning (so the candidate immediately following
    -- of ANY region type, including language -- never inherits THIS candidate's now-stale
    cached embeddings either; required unconditionally, never relying on candidate ordering).
    Reuses vlm_adapter.reset_vllm_encoder_cache_full BY IDENTITY (imported into this module's
    namespace, so tests can monkeypatch it directly) -- no new cache-clearing implementation.
    """
    if manifest.perturbation_mode != PERTURBATION_MODE:
        raise ValueError(f"Stage 7B only evaluates {PERTURBATION_MODE!r} manifests, got {manifest.perturbation_mode!r}")

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
                    f"requested radius={manifest.radius}): strict-mode realized relative-L2 "
                    f"{realized_r} still differs by more than {REALIZED_RADIUS_TOLERANCE} -- this "
                    f"should be unreachable, since scoped_apply_anatomical_perturbation_bf16_"
                    f"quantization_aware_v3 itself guarantees this bound or raises RadiusCorrectionFailedError."
                )
        elif acceptance_mode == "quantization_limited":
            if apply_result["relative_radius_error"] > QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE:
                raise RealizedRadiusMismatchError(
                    f"Perturbation {manifest.perturbation_id!r} (region={manifest.anatomy_region!r}, "
                    f"requested radius={manifest.radius}): quantization-limited relative error "
                    f"{apply_result['relative_radius_error']} still exceeds "
                    f"{QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE} -- this should be unreachable, since "
                    f"scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 itself "
                    f"guarantees this bound or raises QuantizationToleranceExceededError."
                )
        else:
            raise RealizedRadiusMismatchError(f"Unknown radius_acceptance_mode {acceptance_mode!r} for perturbation {manifest.perturbation_id!r}.")

        # Invalidate the multimodal-encoder cache BEFORE evaluating under the accepted
        # perturbed weights -- required for EVERY region type (including language), since the
        # immediately preceding candidate may have perturbed vision/connector and left
        # candidate-specific embeddings cached; never relying on candidate ordering.
        reset_vllm_encoder_cache_full(engine)

        for capability, ctx in capability_contexts.items():
            if ctx.partition.manifest_hash != ctx.subset_hash:
                raise DatasetRoleViolationError(f"CapabilityContext for {capability!r} has an inconsistent subset_hash.")
            result = run_benchmark(ctx.benchmark, ctx.examples, llm_adapter, tokenizer, sampling_params)
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
                    "designed_abs_error": apply_result["designed_abs_error"],
                    "realized_relative_l2": realized_r,  # the ACTUAL realized value -- never the nominal requested value
                    "realized_abs_error": apply_result["realized_abs_error"],
                    "absolute_radius_error": apply_result["absolute_radius_error"],
                    "relative_radius_error": apply_result["relative_radius_error"],
                    "initial_realized_relative_l2": apply_result["initial_realized_relative_l2"],
                    "final_realized_relative_l2": apply_result["final_realized_relative_l2"],
                    "final_absolute_radius_error": apply_result["final_absolute_radius_error"],
                    "final_scale": apply_result["final_scale"],
                    "accepted_scalar": apply_result["accepted_scalar"],
                    "correction_iterations": apply_result["correction_iterations"],
                    "solver_iterations": apply_result["solver_iterations"],
                    "quantization_plateau": apply_result["quantization_plateau"],
                    "nearest_realized_below": apply_result["nearest_realized_below"],
                    "nearest_realized_above": apply_result["nearest_realized_above"],
                    "attainable_gap": apply_result["attainable_gap"],
                    "strict_tolerance": apply_result["strict_tolerance"],
                    "quantization_plateau_relative_tolerance": apply_result["quantization_plateau_relative_tolerance"],
                    "direction_seed": apply_result["direction_seed"],
                    "region_param_count": apply_result["region_param_count"],
                    "theta_region_l2_norm": apply_result["theta_l2_norm"], "epsilon_region_l2_norm": apply_result["realized_epsilon_l2_norm"],
                    "multimodal_cache_policy": MULTIMODAL_CACHE_POLICY,
                    "cache_reset_before_evaluation": True,
                    # Set False here, flipped to True below ONLY after the post-restoration
                    # reset has actually succeeded -- if it fails, this function raises before
                    # `return records`, so these records are never persisted (run_stage7b_rpc
                    # only checkpoints rows after this function returns successfully).
                    "cache_reset_after_restoration": False,
                },
            ))
    finally:
        reset_to_base_weights_via_rpc(engine, ray_get=ray_get)

    verification = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Exact fixed-base restoration failed after calibration candidate {manifest.perturbation_id!r} "
            f"(region={manifest.anatomy_region!r}, radius={manifest.radius}, seed={manifest.seed}): "
            f"max_abs_drift={verification['max_abs_drift']}"
        )

    # Invalidate the multimodal-encoder cache AGAIN, AFTER restoration is verified and BEFORE
    # returning -- so the NEXT candidate (of any region type) never inherits this candidate's
    # cached embeddings either. Only once this has actually succeeded do the already-built
    # records get marked cache_reset_after_restoration=True.
    reset_vllm_encoder_cache_full(engine)
    for record in records:
        record.runtime_metadata["cache_reset_after_restoration"] = True

    return records


def run_stage7b_rpc(
    plan: Stage7bPlan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, base_seed: int, region_param_names_by_region: Dict[str, Sequence[str]],
    parameter_mask_hash_by_region: Dict[str, str], *, run_benchmark: Callable, ray_get: Optional[Callable] = None,
) -> List[ExperimentResultRecord]:
    """Builds the full population for `plan` (full or smoke -- identical code path, only the
    plan's own sizes differ), validates/creates the checkpoint manifest (hard-fails on an
    incompatible existing one -- see IncompatibleCalibrationCheckpointError), skips any
    perturbation already completely persisted, and evaluates + durably persists every
    remaining one. A candidate is appended to results.jsonl (`append_candidate_rows`, reused
    from Stage 6 unmodified) ONLY after its full apply -> evaluate -> reset -> verify cycle has
    already succeeded inside `evaluate_one_calibration_candidate_rpc` -- so a row on disk is
    itself proof restoration passed for that candidate.
    """
    population_by_cell = build_stage7b_population(plan, base_seed, parameter_mask_hash_by_region)

    current_checkpoint = build_stage7b_checkpoint_manifest(plan, capability_contexts, parameter_mask_hash_by_region)
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    ensure_stage7b_checkpoint_manifest(checkpoint_path, current_checkpoint)

    results_path = plan.output_dir / "results.jsonl"
    completed = load_completed_perturbation_rows(results_path, plan.capabilities)

    all_records: List[ExperimentResultRecord] = []
    for rows in completed.values():
        all_records.extend(rows)

    for (region, _radius), manifests in population_by_cell.items():
        region_param_names = region_param_names_by_region[region]
        for manifest in manifests:
            if manifest.perturbation_id in completed:
                continue
            records = evaluate_one_calibration_candidate_rpc(
                engine, manifest, region_param_names, capability_contexts, tokenizer, sampling_params,
                run_benchmark=run_benchmark, ray_get=ray_get,
            )
            append_candidate_rows(results_path, records)
            all_records.extend(records)
    return all_records


# =============================================================================================
# Stage-7B-specific engine config (prefix-cache safety, this repair pass)
# =============================================================================================


def build_stage7b_engine_config() -> Dict[str, Any]:
    """Reuses run_global_visual_thicket_pilot.build_stage6_engine_config()'s own dict BY
    IDENTITY (never duplicated) plus ONE additive override: enable_prefix_caching=False. Live
    Stage-7B logs showed enable_prefix_caching=True (vLLM's own default, since
    launch_stage6_engine() was never told otherwise) -- unsafe across Stage 7B's repeated
    weight-mutation candidate loop, since decoder KV prefixes may have been computed under a
    PREVIOUS candidate's now-stale weights (a hazard Stage 6 does not share). The shared
    Stage-6 launcher/config itself is NOT modified to change its own default -- see
    launch_stage6_engine's own docstring for why passing this value explicitly here leaves
    every existing Stage 6 caller byte-identical to before.
    """
    config = dict(build_stage6_engine_config())
    config["enable_prefix_caching"] = ENABLE_PREFIX_CACHING
    return config


# =============================================================================================
# CLI entry point
# =============================================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "stage7b_anatomical_calibration"))
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny live GPU smoke: region=vision, radius=%s, 1 perturbation, %d D_map examples/capability -- execution size only, same scientific protocol as the frozen full calibration" % (SMOKE_RADIUS, SMOKE_D_MAP_N),
    )
    parser.add_argument(
        "--cache-safety-smoke", action="store_true",
        help="tiny live GPU smoke validating the corrected multimodal-encoder-cache-reset lifecycle: all 3 regions x radius=%s x 1 perturbation/region x %d D_map examples/capability -- instrumentation/lifecycle validation only, NOT a behavioral (Delta != 0) check" % (SMOKE_RADIUS, SMOKE_D_MAP_N),
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU")
    args = parser.parse_args(argv)

    if args.smoke and args.cache_safety_smoke:
        print("--smoke and --cache-safety-smoke are mutually exclusive.", file=sys.stderr)
        return 1

    from .config import load_config

    cfg = load_config(args.config)

    if args.smoke:
        plan = build_smoke_stage7b_plan(model_name=cfg.model.name, model_revision=cfg.model.revision, output_root=args.output_root)
    elif args.cache_safety_smoke:
        plan = build_cache_safety_smoke_stage7b_plan(model_name=cfg.model.name, model_revision=cfg.model.revision, output_root=args.output_root)
    else:
        plan = build_stage7b_plan(model_name=cfg.model.name, model_revision=cfg.model.revision, output_root=args.output_root)

    print(f"Stage 7B plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}): regions={plan.regions} radii={plan.radii} capabilities={plan.capabilities}")
    print(f"n_per_cell={plan.n_per_cell} d_map_n={plan.d_map_n} total_unique_perturbations={plan.total_unique_perturbations}")
    print(f"total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations}")
    print(f"radius_realization_method={plan.radius_realization_method}")
    print(f"multimodal_cache_policy={plan.multimodal_cache_policy}")
    print(f"enable_prefix_caching={plan.enable_prefix_caching}")
    print(f"output_dir={plan.output_dir}")
    print(
        "Lifecycle: scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 (per trial: "
        "reset_to_base_weights -> apply SAME fixed seeded direction at trial scale -> verify "
        "zero out-of-region drift -> measure TRUE post-BF16-add realized radius -> accept "
        "strictly (<=1e-6), or proportion/bisect-and-retry up to 20 trials; if a quantization "
        "plateau is PROVEN and the nearest attainable state's relative error is <=0.1%, reset -> "
        "reapply that scalar -> verify exact reproduction -> accept as quantization_limited; "
        "otherwise hard-fail) -> FULLY INVALIDATE the multimodal-encoder cache "
        "(reset_vllm_encoder_cache_full) -> evaluate visual_grounding/ocr_text_recognition_grounded/"
        "spatial_reasoning D_map -> reset_to_base_weights -> verify exact restoration -> FULLY "
        "INVALIDATE the multimodal-encoder cache again -> checkpoint candidate rows."
    )

    if args.dry_run:
        return 0

    try:
        assert_feasible(
            f"Stage 7B anatomical calibration ({plan.run_signature})",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision")
    model_resolution = resolve_and_report_model_snapshot(plan.model_name, plan.model_revision)
    # Stage-7B-specific: identical to Stage 6's frozen (max_model_len=4096, gpu_memory_
    # utilization=0.60, tensor_parallel_size=1, bfloat16) EXCEPT enable_prefix_caching=False --
    # see build_stage7b_engine_config's own docstring. Never build_stage6_engine_config()
    # directly here -- that would silently re-enable prefix caching (vLLM's own default).
    engine_config = build_stage7b_engine_config()
    assert engine_config["enable_prefix_caching"] is False, "Stage 7B must never run with prefix caching enabled -- see build_stage7b_engine_config."
    print(format_runtime_compatibility_diagnostic(
        model_resolution, worker_extension_cls="utils.worker_extn.WorkerExtension",
        vllm_version=get_vllm_version(), engine_mode=detect_vllm_engine_mode(), engine_config=engine_config,
    ))
    print(f"Stage 7B engine override: enable_prefix_caching={engine_config['enable_prefix_caching']} (Stage 6's own launcher/default is unaffected -- see launch_stage6_engine's own docstring).")

    from .benchmarks.runner import run_benchmark
    from .config import load_capability_benchmark_config
    from .run_capability_benchmark_gate import load_adapter
    from .thicket.data_roles import write_data_role_manifest
    from .vlm_adapter import bootstrap_ray, verify_workers_can_import_external_root

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    subset_ids_dir = plan.output_dir / "d_map_subsets"
    capability_contexts = build_d_map_capability_contexts(
        STAGE7B_BASE_SEED, subset_ids_dir, plan.d_map_n,
        load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
    )
    for capability, ctx in capability_contexts.items():
        write_data_role_manifest(ctx.partition, plan.output_dir / "data_roles" / f"{capability}_d_map.json")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_resolution["resolved_snapshot_path"])

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore  # upstream, unmodified -- teardown only; launch uses OUR OWN launch_stage6_engine, never upstream launch_engines

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    import ray

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        # HARD FAIL BEFORE EXPERIMENT if the full multimodal-encoder-cache reset mechanism
        # cannot be exposed -- MUST run before launch_stage6_engine (it monkey-patches
        # RandOptNcclLLM so Ray's actor-method registry includes reset_encoder_cache_full).
        ensure_stage7b_encoder_cache_reset_mechanism_exposed(EXTERNAL_ROOT)

        engines, pgs = launch_stage6_engine(
            model_resolution["resolved_snapshot_path"], precision=engine_config["precision"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"], max_model_len=engine_config["max_model_len"],
            tensor_parallel_size=engine_config["tensor_parallel_size"],
            enable_prefix_caching=engine_config["enable_prefix_caching"],
        )
        engine = engines[0]

        store_base_weights_via_rpc(engine)
        print(format_base_snapshot_confirmation(engine_config["gpu_memory_utilization"], engine_config["base_snapshot_mode"]))

        # HARD FAIL BEFORE EXPERIMENT if the cache reset mechanism doesn't actually WORK
        # end-to-end against the LIVE engine (not merely that it was exposed pre-launch) --
        # proven before any candidate evaluation begins.
        ensure_encoder_cache_reset_available(engine)
        print(f"Confirmed working multimodal-encoder-cache reset (multimodal_cache_policy={plan.multimodal_cache_policy!r}).")

        region_info = _collective_rpc_single_worker(engine, report_region_param_names, args=(plan.regions,), label="report_region_param_names")
        region_param_names_by_region = {r: tuple(info["param_names"]) for r, info in region_info.items()}
        parameter_mask_hash_by_region = {r: info["mask_hash"] for r, info in region_info.items()}

        llm_adapter = RayEngineLLMAdapter(engine)
        baseline_path = plan.output_dir / "baseline_scores.json"
        load_or_compute_baseline_scores(baseline_path, capability_contexts, plan.model_revision, plan.run_signature, llm_adapter, tokenizer, sampling_params)

        records = run_stage7b_rpc(
            plan, capability_contexts, engine, tokenizer, sampling_params, STAGE7B_BASE_SEED,
            region_param_names_by_region, parameter_mask_hash_by_region, run_benchmark=run_benchmark,
        )
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    manifest = write_stage7b_run_manifest(plan.output_dir)
    print(f"Wrote {len(records)} result rows to {plan.output_dir / 'results.jsonl'}")
    print(f"Run manifest: {json.dumps(manifest, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
