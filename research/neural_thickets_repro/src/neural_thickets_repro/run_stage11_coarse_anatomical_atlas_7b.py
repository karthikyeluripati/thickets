"""Stage 11: 7B cross-scale coarse-anatomy replication -- a DIRECT replication of Stage 8's
paper-scale L1 anatomy (vision / multimodal_connector_or_merger / language) x 3 frozen common
relative-L2 radii x 6 frozen capabilities atlas, at Qwen/Qwen2.5-VL-7B-Instruct instead of the 3B
model. Answers: does the coarse neural-thicket anatomy discovered at 3B reproduce at 7B?

=================================================================================================
REUSE, BY IMPORT, FROM STAGE 8/9 (this module changes NONE of these -- see those modules' own
docstrings for the full derivation of each):
=================================================================================================
- STAGE8_REGIONS/STAGE8_RADII/STAGE8_CAPABILITIES/STAGE8_N_DIRECTIONS_PER_CELL/STAGE8_D_MAP_N/
  STAGE8_GENERATE_BATCH_SIZE -- the frozen scientific design is byte-identical to Stage 8's; only
  the MODEL changes. Radii are NOT recalibrated for 7B (Section 4 of the task spec) -- they stay
  the exact same three relative-L2 fractions, deliberately, so a shift in the useful/destructive
  regime at the SAME fractional displacement is itself a result, never optimized away.
- build_stage7b_engine_config() (bf16 / max_model_len=4096 / gpu_memory_utilization=0.60 / TP=1 /
  enable_prefix_caching=False) -- retained UNCHANGED; if 7B cannot load under this configuration
  the run must STOP (main() does not silently change it -- see the --dry-run / live-load path).
- scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 (the v3 two-tier radius
  acceptance rule, INCLUDING the post-legacy-v3 bracket-expansion fix) -- identical dispatch,
  because it is already model-size-agnostic (a pure function of the region's own live parameter
  tensors, whatever their shapes turn out to be at 7B).
- run_baseline_repeatability_preflight_rpc / ensure_baseline_repeatability /
  BaselineNondeterminismError (Stage 8's OWN two-pass theta_0 repeatability gate) -- reused
  UNCHANGED: Section 8 of the task spec explicitly does NOT want a cross-scale score-equality
  gate (7B baselines are NOT expected to equal 3B scores) -- it wants exactly the repeatability
  check Stage 8 already built for this same reason.
- build_d_map_capability_contexts, called with Stage 8's OWN STAGE8_BASE_SEED (never a new one)
  -- guarantees the IDENTICAL D_map example IDs/subset hashes as the real 3B Stage-8 run, which
  is what makes a same-example cross-scale comparison meaningful. This is a DIFFERENT seed
  concern from the direction-family seed bank below (D_map example selection vs. perturbation
  direction sampling are independent RNG streams by design, exactly as they always have been).
- _is_ray_unrecoverable_error / _write_candidate_telemetry_line (Stage 9's already-generic,
  model-agnostic memory-lifecycle helpers) -- reused BY IMPORT rather than re-implemented a third
  time.
- STAGE8_AUTHORITATIVE_SUBSET_HASHES (defined once, in run_stage9_hierarchical_anatomical_atlas.py)
  -- reused BY IMPORT for the hard subset-hash-equality gate below, never re-typed as a third
  literal copy of the same 6 hash strings.

=================================================================================================
WHAT IS NEW IN STAGE 11
=================================================================================================
1. MODEL: Qwen/Qwen2.5-VL-7B-Instruct. The exact immutable HF revision is resolved LIVE via
   resolve_immutable_model_revision() (huggingface_hub's model_info API) before any snapshot
   download -- never a hand-typed/invented commit hash. Hard-fails (ModelRevisionResolutionError)
   if the Hub cannot return a genuine 40-hex-character commit SHA. The resolution result
   (requested ref, resolved SHA, method) is persisted to model_revision_resolution.json.

2. ANATOMY AUDIT: report_stage11_anatomy_audit() builds the L1 partition from the REAL 7B model's
   live named_parameters() (thicket.anatomy.build_anatomy_atlas -- unmodified, purely a function
   of parameter-name strings, so it needs no 7B-specific code at all) and additionally computes,
   per region, from the REAL live tensors (never assumed/copied from 3B): tensor count, total
   element count, fp32 L2 norm, mask_hash, and percentage of total model elements -- plus the
   full-model-level "no uncovered parameters" / "pairwise disjoint" checks. main() hard-fails
   before touching any perturbation if this audit does not pass.

3. INDEPENDENT 7B DIRECTION-SEED NAMESPACE: build_stage11_direction_seed_bank() derives seeds via
   derive_seed(base_seed, "stage11_direction_family", region, str(i)) -- a namespace string
   DISTINCT from Stage 8's "stage8_direction_family", using its OWN STAGE11_BASE_SEED. Even
   though the two models' parameter tensors have different shapes (so numerically reusing a seed
   would not even produce a comparable direction), the namespace is kept independent by
   construction so 3B seed-i and 7B seed-i are never even superficially conflated. Seed INDEX
   correspondence (i=0..63) is retained only for bookkeeping/matched-index reporting in the
   (not-yet-run) cross-scale analysis -- never a claim of geometric pairing.

4. STAGE11CHECKPOINTMANIFEST: mirrors Stage8CheckpointManifest's identity discipline, plus an
   `anatomy_audit_hash` field (sha256 over the canonical JSON of the live 7B audit report) so a
   resume attempt against a DIFFERENT (accidentally re-derived) 7B anatomy is rejected exactly
   like a model-revision mismatch would be, and a `stage8_parent_run_signature` field explicitly
   linking to "stage8_coarse_anatomical_atlas_3b_v2_batched10" for provenance.

5. MEMORY LIFECYCLE: evaluate_one_stage11_candidate_rpc / run_stage11_rpc port Stage 9's ALREADY
   -FIXED (post-driver-RSS-OOM) pattern -- no all-record accumulator, per-capability RunResult
   `del` + release_transient_memory(), candidate-level RSS telemetry in its own
   candidate_memory_telemetry.jsonl (never embedded in scientific rows), and the
   _is_ray_unrecoverable_error-gated failure path -- rather than Stage 8's original run_stage8_rpc
   (which still retains the full-sweep accumulator; see run_stage8_coarse_anatomical_atlas.py's
   own module docstring). This is a DELIBERATE choice per the task's explicit instruction to reuse
   "the proven Stage-8/9 bounded-memory lifecycle" and never regress to an in-memory accumulator.

=================================================================================================
FROZEN FULL Stage-11 CONFIG
=================================================================================================
model:        Qwen/Qwen2.5-VL-7B-Instruct @ <resolved live, never invented>, bf16
regions:      vision, multimodal_connector_or_merger, language (STAGE11_REGIONS ==
              STAGE8_REGIONS, by identity -- masks are re-derived LIVE from the real 7B model,
              never assumed to transfer numerically from 3B)
radii:        STAGE11_RADII == STAGE8_RADII, by identity -- NOT recalibrated for 7B (deliberate)
capabilities: STAGE11_CAPABILITIES == STAGE8_CAPABILITIES, by identity
directions:   STAGE11_N_DIRECTIONS_PER_CELL (64) per (region, radius) cell, independent 7B seed
              namespace, same direction reused across the 3 radii within one region
data:         the SAME D_map N=50 example manifests as Stage 8 (STAGE8_BASE_SEED reused)
Total:        3 regions x 3 radii x 64 directions = 576 unique perturbations
              576 x 6 capabilities = 3456 perturbation x capability result rows
              3456 x 50 = 172,800 perturbed model-example evaluations
Matches Stage 8's candidate budget exactly. No D_confirm/D_select/D_test is ever constructed.

=================================================================================================
SMOKE MODE -- EXECUTION SIZE ONLY, same scientific definitions
=================================================================================================
--smoke: all 3 STAGE11_REGIONS, all 3 STAGE11_RADII, 1 direction family, all 6 capabilities,
D_map N=5. Expected: 3 x 3 x 1 = 9 unique perturbations, 9 x 6 = 54 result rows, 54 x 5 = 270
perturbed model-example evaluations. Smoke baseline gate is the SAME two-pass N=5 repeatability
check (never a numerical comparison against 3B or against the 7B full-run N=50 baseline).

=================================================================================================
DO NOT DO YET (explicitly out of scope for this module)
=================================================================================================
7B depth Stage 9, 72B, attention-vs-MLP, parameter-space Stage 10B, post-training, routing,
distillation, D_confirm. The cross-scale ANALYSIS schema (Section 12 of the task spec) is
prepared in analysis/stage11_cross_scale_schema.py but is NOT run against real data here -- no
7B results exist yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    EncoderCacheResetUnavailableError,
    RealizedRadiusMismatchError,
    build_stage7b_engine_config,
    ensure_encoder_cache_reset_available,
    ensure_stage7b_encoder_cache_reset_mechanism_exposed,
    report_region_param_names,
)
from .run_stage8_coarse_anatomical_atlas import (
    STAGE8_BASE_SEED,
    STAGE8_CAPABILITIES,
    STAGE8_D_MAP_N,
    STAGE8_GENERATE_BATCH_SIZE,
    STAGE8_N_DIRECTIONS_PER_CELL,
    STAGE8_RADII,
    STAGE8_REGIONS,
    STAGE8_SMOKE_D_MAP_N,
    STAGE8_SMOKE_N_DIRECTIONS,
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
from .scoped_anatomical_perturbation import (
    QUANTIZATION_AWARE_METHOD_V3,
    QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
)
from .thicket.anatomy import AnatomyValidationError, build_anatomy_atlas, validate_atlas
from .thicket.perturbation import PERTURBATION_MODES, PerturbationManifest
from .thicket.schema import ExperimentResultRecord
from .thicket.seeds import derive_seed
from .vlm_adapter import reset_vllm_encoder_cache_full

assert PERTURBATION_MODE in PERTURBATION_MODES
assert RADIUS_REALIZATION_METHOD == QUANTIZATION_AWARE_METHOD_V3  # never a different/looser method

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

EXPERIMENT_ID = "stage11_coarse_anatomical_atlas_7b"
MODEL_NAME_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_FAMILY_7B = "qwen2_5_vl"
MODEL_SCALE_7B = "7B"
STAGE11_BASE_SEED = 20260904  # distinct from Stage 6 (20260823) / Stage 7B (20260824) / Stage 8 (20260825) / Stage 9 (20260901)
STAGE8_PARENT_RUN_SIGNATURE = "stage8_coarse_anatomical_atlas_3b_v2_batched10"

# --- FROZEN full-Stage-11 scientific config -- byte-identical to Stage 8's, by import ----------
STAGE11_REGIONS: Tuple[str, ...] = STAGE8_REGIONS
STAGE11_RADII: Tuple[float, ...] = STAGE8_RADII
STAGE11_CAPABILITIES: Tuple[str, ...] = STAGE8_CAPABILITIES
STAGE11_N_DIRECTIONS_PER_CELL: int = STAGE8_N_DIRECTIONS_PER_CELL
STAGE11_D_MAP_N: int = STAGE8_D_MAP_N
STAGE11_GENERATE_BATCH_SIZE: int = STAGE8_GENERATE_BATCH_SIZE
DATASET_ROLE = "map"

STAGE11_SMOKE_N_DIRECTIONS = STAGE8_SMOKE_N_DIRECTIONS
STAGE11_SMOKE_D_MAP_N = STAGE8_SMOKE_D_MAP_N
_ALLOWED_D_MAP_SIZES: Tuple[int, ...] = (STAGE11_D_MAP_N, STAGE11_SMOKE_D_MAP_N)

_FULL_RUN_SIGNATURE = "stage11_coarse_anatomical_atlas_7b_v1"


# =================================================================================================
# Section 1: model resolution -- resolve the exact immutable HF revision, never invent one
# =================================================================================================


class ModelRevisionResolutionError(RuntimeError):
    """Could not establish a genuine, immutable (full 40-hex-character commit SHA) HuggingFace
    revision for the requested model -- hard stop. Stage 11 never proceeds with a mutable ref
    (e.g. "main") passed straight through to snapshot_download, and never fabricates a SHA.
    """


_FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _looks_like_full_git_sha(value: str) -> bool:
    return bool(_FULL_GIT_SHA_RE.match(value))


def resolve_immutable_model_revision(model_name: str, revision_ref: str) -> Dict[str, Any]:
    """If `revision_ref` is already a full 40-hex-character commit SHA, it is used as-is
    (already pinned -- no live Hub call needed, and idempotent on a re-run). Otherwise queries
    the HF Hub LIVE (huggingface_hub.HfApi().model_info) to resolve the ref (e.g. "main", a
    branch, a tag) to the immutable commit SHA it currently points to, and hard-fails
    (ModelRevisionResolutionError) if the Hub call itself fails OR the returned `.sha` is not a
    genuine full commit SHA -- this is the ONLY place a revision is ever accepted from, never a
    hand-typed literal in this module's source.
    """
    if _looks_like_full_git_sha(revision_ref):
        return {"model_name": model_name, "requested_ref": revision_ref, "resolved_revision": revision_ref, "resolution_method": "already_pinned"}

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id=model_name, revision=revision_ref)
        resolved_sha = getattr(info, "sha", None)
    except Exception as exc:  # noqa: BLE001 -- any Hub/network failure is a hard stop, never silently retried with a guess
        raise ModelRevisionResolutionError(
            f"Could not resolve an immutable revision for {model_name!r} at ref {revision_ref!r}: {exc}"
        ) from exc

    if not resolved_sha or not _looks_like_full_git_sha(resolved_sha):
        raise ModelRevisionResolutionError(
            f"HuggingFace Hub did not return a genuine full commit SHA for {model_name!r} at ref "
            f"{revision_ref!r} (got {resolved_sha!r}) -- refusing to proceed with an unpinned or "
            f"malformed revision."
        )
    return {"model_name": model_name, "requested_ref": revision_ref, "resolved_revision": resolved_sha, "resolution_method": "resolved_via_hf_api"}


# =================================================================================================
# Section 2: hard 7B anatomy audit (tensor counts, element counts, norms, percentages, hashes)
# =================================================================================================


def report_stage11_anatomy_audit(worker_self, regions: Sequence[str]) -> Dict[str, Any]:
    """Runs entirely inside the worker process against the REAL 7B model's live
    named_parameters() -- NEVER assumes 3B tensor counts/hashes/norms transfer. Builds the L1
    atlas the SAME way Stage 8 does (thicket.anatomy.build_anatomy_atlas, a pure function of
    parameter-NAME strings, needing no 7B-specific code), then additionally measures, from the
    REAL live tensors: per-region tensor count, total element count, fp32 L2 norm, and percentage
    of the full model's total element count -- plus the full-model-level completeness/
    disjointness audit (validate_atlas, converted to plain booleans here rather than letting its
    exception cross the RPC boundary uncontrolled).
    """
    import torch

    model = worker_self.model_runner.model
    named = list(model.named_parameters())
    names = [n for n, _ in named]
    tensor_by_name = dict(named)

    atlas = build_anatomy_atlas(names, model_family=MODEL_FAMILY_7B)

    union_equals_full_model = True
    pairwise_disjoint = True
    uncovered_full_model: Tuple[str, ...] = ()
    try:
        report = validate_atlas(atlas)
        uncovered_full_model = report.uncovered_by_parent.get("full_model", ())
        union_equals_full_model = len(uncovered_full_model) == 0
    except AnatomyValidationError as exc:
        pairwise_disjoint = "overlap" not in str(exc).lower()
        union_equals_full_model = False

    total_elements = int(sum(t.numel() for t in tensor_by_name.values()))

    region_reports: Dict[str, Any] = {}
    for region in regions:
        atlas_region = atlas.region(region)
        region_tensors = [tensor_by_name[n] for n in atlas_region.param_names]
        n_elements = int(sum(t.numel() for t in region_tensors))
        l2_norm_sq = 0.0
        for t in region_tensors:
            l2_norm_sq += float(t.detach().float().pow(2).sum().item())
        region_reports[region] = {
            "region": region, "n_tensors": len(atlas_region.param_names), "n_elements": n_elements,
            "l2_norm": l2_norm_sq ** 0.5, "mask_hash": atlas_region.mask_hash,
            "percentage_of_total_elements": (100.0 * n_elements / total_elements) if total_elements else 0.0,
        }

    return {
        "regions": region_reports, "total_model_elements": total_elements,
        "union_equals_full_model": union_equals_full_model, "pairwise_disjoint": pairwise_disjoint,
        "uncovered_by_full_model": list(uncovered_full_model),
    }


def compute_anatomy_audit_hash(audit: Dict[str, Any]) -> str:
    """Stable hash over the audit's REGION mask_hashes/element counts (never the raw per-tensor
    norms, which could carry harmless floating-point summation-order noise across identical
    reruns) -- used as a Stage11CheckpointManifest identity field so a resume against a
    DIFFERENTLY-DERIVED 7B anatomy is rejected exactly like a model-revision mismatch would be.
    """
    canonical = json.dumps({
        region: {"mask_hash": r["mask_hash"], "n_tensors": r["n_tensors"], "n_elements": r["n_elements"]}
        for region, r in sorted(audit["regions"].items())
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_stage11_anatomy_audit_passes(audit: Dict[str, Any], regions: Sequence[str]) -> None:
    if not audit["union_equals_full_model"]:
        raise RuntimeError(f"Stage-11 live 7B anatomy audit FAILED: uncovered full-model parameters: {audit['uncovered_by_full_model']}")
    if not audit["pairwise_disjoint"]:
        raise RuntimeError("Stage-11 live 7B anatomy audit FAILED: L1 regions are not pairwise disjoint.")
    missing = set(regions) - set(audit["regions"])
    if missing:
        raise RuntimeError(f"Stage-11 live 7B anatomy audit is missing region(s): {sorted(missing)}")
    empty = [r for r, info in audit["regions"].items() if info["n_tensors"] == 0]
    if empty:
        raise RuntimeError(f"Stage-11 live 7B anatomy audit found EMPTY region(s): {empty}")


# =================================================================================================
# Plan (pure arithmetic, no I/O, no GPU)
# =================================================================================================


def compute_stage11_run_signature(
    regions: Sequence[str], radii: Sequence[float], n_directions: int, d_map_n: int, generation_batch_size: Optional[int],
) -> str:
    is_frozen_full_scientific_config = (
        tuple(regions) == STAGE11_REGIONS and tuple(radii) == STAGE11_RADII
        and n_directions == STAGE11_N_DIRECTIONS_PER_CELL and d_map_n == STAGE11_D_MAP_N
    )
    if is_frozen_full_scientific_config:
        if generation_batch_size == STAGE11_GENERATE_BATCH_SIZE:
            return _FULL_RUN_SIGNATURE
        return f"stage11_coarse_anatomical_atlas_7b_batched{generation_batch_size}"
    region_label = "-".join(regions)
    radius_label = "-".join(f"{r:.6f}".replace(".", "") for r in radii)
    batch_label = f"_batched{generation_batch_size}" if generation_batch_size is not None else ""
    return f"stage11_smoke_{region_label}_r{radius_label}_n{d_map_n}_dir{n_directions}{batch_label}"


@dataclass(frozen=True)
class Stage11Plan:
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
    generation_batch_size: int = STAGE11_GENERATE_BATCH_SIZE

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
            self.regions == STAGE11_REGIONS and self.radii == STAGE11_RADII
            and self.n_directions_per_cell == STAGE11_N_DIRECTIONS_PER_CELL and self.d_map_n == STAGE11_D_MAP_N
        )


def build_stage11_plan(
    *, model_name: str, model_revision: str, output_root: "str | Path",
    regions: Sequence[str] = STAGE11_REGIONS, radii: Sequence[float] = STAGE11_RADII,
    n_directions_per_cell: int = STAGE11_N_DIRECTIONS_PER_CELL, d_map_n: int = STAGE11_D_MAP_N,
    generation_batch_size: int = STAGE11_GENERATE_BATCH_SIZE,
) -> Stage11Plan:
    if not regions:
        raise ValueError("Stage 11 requires at least one anatomy region.")
    if not radii:
        raise ValueError("Stage 11 requires at least one common radius.")
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 11 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    if generation_batch_size <= 0:
        raise ValueError(f"generation_batch_size must be positive, got {generation_batch_size}")

    run_signature = compute_stage11_run_signature(regions, radii, n_directions_per_cell, d_map_n, generation_batch_size)
    return Stage11Plan(
        model_name=model_name, model_revision=model_revision, model_family=MODEL_FAMILY_7B, model_scale=MODEL_SCALE_7B,
        regions=tuple(regions), radii=tuple(radii), capabilities=STAGE11_CAPABILITIES,
        n_directions_per_cell=n_directions_per_cell, d_map_n=d_map_n,
        radius_realization_method=RADIUS_REALIZATION_METHOD, multimodal_cache_policy=MULTIMODAL_CACHE_POLICY,
        enable_prefix_caching=ENABLE_PREFIX_CACHING, run_signature=run_signature, output_dir=Path(output_root) / run_signature,
        generation_batch_size=generation_batch_size,
    )


def build_stage11_smoke_plan(*, model_name: str, model_revision: str, output_root: "str | Path") -> Stage11Plan:
    """3 regions x 3 radii x 1 direction family x 6 capabilities x D_map N=5 = 9 perturbations,
    54 rows, 270 perturbed model-example evaluations.
    """
    return build_stage11_plan(
        model_name=model_name, model_revision=model_revision, output_root=output_root,
        regions=STAGE11_REGIONS, radii=STAGE11_RADII, n_directions_per_cell=STAGE11_SMOKE_N_DIRECTIONS, d_map_n=STAGE11_SMOKE_D_MAP_N,
    )


# =================================================================================================
# Direction-family seed bank + population -- independent 7B namespace
# =================================================================================================


def build_stage11_direction_seed_bank(base_seed: int, regions: Sequence[str], n_directions: int) -> Dict[str, Tuple[int, ...]]:
    """A namespace string ("stage11_direction_family") DISTINCT from Stage 8's own
    ("stage8_direction_family") -- 3B seed-i and 7B seed-i are never geometrically paired, even
    incidentally; seed INDEX correspondence (i) is retained only for bookkeeping.
    """
    return {
        region: tuple(derive_seed(base_seed, "stage11_direction_family", region, str(i)) for i in range(n_directions))
        for region in regions
    }


def compute_direction_seed_bank_hash(seed_bank: Dict[str, Tuple[int, ...]]) -> str:
    canonical = json.dumps({region: list(seeds) for region, seeds in sorted(seed_bank.items())}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Stage11DirectionAssignment:
    manifest: PerturbationManifest
    region: str
    direction_index: int
    direction_seed: int

    @property
    def direction_family_id(self) -> str:
        return f"{self.region}:{self.direction_index}"


def build_stage11_population(
    plan: Stage11Plan, seed_bank: Dict[str, Tuple[int, ...]], parameter_mask_hash_by_region: Dict[str, str],
) -> Dict[Tuple[str, float], Tuple[Stage11DirectionAssignment, ...]]:
    missing_regions = set(plan.regions) - set(parameter_mask_hash_by_region)
    if missing_regions:
        raise ValueError(f"Missing parameter_mask_hash for region(s): {sorted(missing_regions)}")
    missing_bank_regions = set(plan.regions) - set(seed_bank)
    if missing_bank_regions:
        raise ValueError(f"Missing direction seed bank for region(s): {sorted(missing_bank_regions)}")
    for region in plan.regions:
        if len(seed_bank[region]) != plan.n_directions_per_cell:
            raise ValueError(f"Direction seed bank for region {region!r} has {len(seed_bank[region])} seeds, expected {plan.n_directions_per_cell}.")

    population_by_cell: Dict[Tuple[str, float], Tuple[Stage11DirectionAssignment, ...]] = {}
    for region in plan.regions:
        mask_hash = parameter_mask_hash_by_region[region]
        seeds = seed_bank[region]
        for radius in plan.radii:
            assignments = tuple(
                Stage11DirectionAssignment(
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
    """The population does not satisfy Stage 11's direction-family invariant (mirrors Stage 8's
    own validate_stage8_direction_seed_reuse exactly, over the independent 7B seed bank).
    """


def validate_stage11_direction_seed_reuse(
    plan: Stage11Plan, population_by_cell: Dict[Tuple[str, float], Tuple[Stage11DirectionAssignment, ...]],
) -> None:
    all_ids = [a.manifest.perturbation_id for cell in population_by_cell.values() for a in cell]
    if len(all_ids) != len(set(all_ids)):
        raise DirectionSeedReuseViolationError(f"Duplicate perturbation_id(s) in the Stage-11 population ({len(all_ids)} total, {len(set(all_ids))} unique).")
    expected_total = len(plan.regions) * len(plan.radii) * plan.n_directions_per_cell
    if len(all_ids) != expected_total:
        raise DirectionSeedReuseViolationError(f"Stage-11 population has {len(all_ids)} perturbations, expected {expected_total}.")

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


# =================================================================================================
# Subset-hash equality gate against the authoritative Stage-8 (3B) D_map manifests
# =================================================================================================


class Stage11SubsetHashMismatchError(RuntimeError):
    """Stage 11's live D_map subset hashes do not exactly match Stage 8's authoritative 3B
    manifests -- the cross-scale comparison would silently be over DIFFERENT examples. Hard stop.
    """


def run_stage11_subset_hash_check(capability_contexts: Dict[str, CapabilityContext]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for cap, ctx in capability_contexts.items():
        expected = STAGE8_AUTHORITATIVE_SUBSET_HASHES.get(cap)
        report[cap] = {"expected_stage8_subset_hash": expected, "live_subset_hash": ctx.subset_hash, "matches": ctx.subset_hash == expected}
    report["all_match"] = all(v["matches"] for v in report.values())
    return report


def ensure_stage11_subset_hashes_match_stage8(report: Dict[str, Any]) -> None:
    if not report.get("all_match"):
        mismatches = {cap: v for cap, v in report.items() if cap != "all_match" and not v["matches"]}
        raise Stage11SubsetHashMismatchError(f"Stage-11 D_map subset hashes do not match Stage-8's authoritative 3B manifests: {mismatches}")


# =================================================================================================
# Checkpoint identity
# =================================================================================================


class IncompatibleStage11CheckpointError(RuntimeError):
    """An existing checkpoint_manifest.json in this output directory does not match the
    current run's identity -- refuses to resume a differently-configured partial run.
    """


@dataclass(frozen=True)
class Stage11CheckpointManifest:
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
    regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_directions_per_cell: int
    d_map_n: int
    subset_hashes: Dict[str, str]
    region_mask_hashes: Dict[str, str]
    direction_seed_bank_hash: str
    anatomy_audit_hash: str
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
            "regions": list(self.regions), "radii": list(self.radii), "capabilities": list(self.capabilities),
            "n_directions_per_cell": self.n_directions_per_cell, "d_map_n": self.d_map_n,
            "subset_hashes": dict(sorted(self.subset_hashes.items())),
            "region_mask_hashes": dict(sorted(self.region_mask_hashes.items())),
            "direction_seed_bank_hash": self.direction_seed_bank_hash, "anatomy_audit_hash": self.anatomy_audit_hash,
            "stage8_parent_run_signature": self.stage8_parent_run_signature,
            "expected_unique_perturbations": self.expected_unique_perturbations,
            "expected_result_rows": self.expected_result_rows,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage11CheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], run_signature=d["run_signature"], restoration_mode=d["restoration_mode"],
            perturbation_mode=d["perturbation_mode"], radius_realization_method=d["radius_realization_method"],
            multimodal_cache_policy=d["multimodal_cache_policy"], enable_prefix_caching=d["enable_prefix_caching"],
            generation_batch_size=d["generation_batch_size"], model_revision=d["model_revision"], dataset_role=d["dataset_role"],
            regions=tuple(d["regions"]), radii=tuple(d["radii"]), capabilities=tuple(d["capabilities"]),
            n_directions_per_cell=d["n_directions_per_cell"], d_map_n=d["d_map_n"],
            subset_hashes=dict(d["subset_hashes"]), region_mask_hashes=dict(d["region_mask_hashes"]),
            direction_seed_bank_hash=d["direction_seed_bank_hash"], anatomy_audit_hash=d["anatomy_audit_hash"],
            stage8_parent_run_signature=d["stage8_parent_run_signature"],
            expected_unique_perturbations=d["expected_unique_perturbations"], expected_result_rows=d["expected_result_rows"],
        )


def build_stage11_checkpoint_manifest(
    plan: Stage11Plan, capability_contexts: Dict[str, CapabilityContext], region_mask_hashes: Dict[str, str],
    seed_bank: Dict[str, Tuple[int, ...]], anatomy_audit: Dict[str, Any],
) -> Stage11CheckpointManifest:
    if plan.d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 11 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {plan.d_map_n}")
    missing_regions = set(plan.regions) - set(region_mask_hashes)
    if missing_regions:
        raise ValueError(f"Missing region_mask_hashes for region(s): {sorted(missing_regions)}")
    return Stage11CheckpointManifest(
        experiment_id=EXPERIMENT_ID, run_signature=plan.run_signature, restoration_mode=RESTORATION_MODE,
        perturbation_mode=PERTURBATION_MODE, radius_realization_method=plan.radius_realization_method,
        multimodal_cache_policy=plan.multimodal_cache_policy, enable_prefix_caching=plan.enable_prefix_caching,
        generation_batch_size=plan.generation_batch_size, model_revision=plan.model_revision, dataset_role=DATASET_ROLE,
        regions=plan.regions, radii=plan.radii, capabilities=plan.capabilities,
        n_directions_per_cell=plan.n_directions_per_cell, d_map_n=plan.d_map_n,
        subset_hashes={c: ctx.subset_hash for c, ctx in capability_contexts.items()},
        region_mask_hashes={r: region_mask_hashes[r] for r in plan.regions},
        direction_seed_bank_hash=compute_direction_seed_bank_hash(seed_bank),
        anatomy_audit_hash=compute_anatomy_audit_hash(anatomy_audit),
        stage8_parent_run_signature=STAGE8_PARENT_RUN_SIGNATURE,
        expected_unique_perturbations=plan.total_unique_perturbations,
        expected_result_rows=plan.total_perturbation_capability_evaluations,
    )


def ensure_stage11_checkpoint_manifest(path: Path, current: Stage11CheckpointManifest) -> Stage11CheckpointManifest:
    if path.exists():
        existing = Stage11CheckpointManifest.from_dict(json.loads(path.read_text()))
        if existing != current:
            raise IncompatibleStage11CheckpointError(
                f"Existing checkpoint at {path} is incompatible with this run -- refusing to resume: "
                f"existing={existing.to_dict()} current={current.to_dict()}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current.to_dict(), indent=2))
    return current


def build_stage11_run_manifest_summary(checkpoint: Stage11CheckpointManifest, records: List[ExperimentResultRecord]) -> Dict[str, Any]:
    actual_unique_perturbations = len({r.perturbation_id for r in records})
    actual_result_rows = len(records)
    run_complete = (
        actual_unique_perturbations == checkpoint.expected_unique_perturbations
        and actual_result_rows == checkpoint.expected_result_rows
    )
    return {
        **checkpoint.to_dict(),
        "actual_unique_perturbations": actual_unique_perturbations, "actual_result_rows": actual_result_rows,
        "run_complete": run_complete,
    }


def write_stage11_run_manifest(output_dir: Path) -> Dict[str, Any]:
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_manifest.json found at {checkpoint_path} -- this run never started (or output_dir is wrong).")
    checkpoint = Stage11CheckpointManifest.from_dict(json.loads(checkpoint_path.read_text()))
    records = load_records(output_dir / "results.jsonl")
    manifest = build_stage11_run_manifest_summary(checkpoint, records)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# =================================================================================================
# Worker-RPC transport (same TP=1 list-unwrap convention as every other GPU script here)
# =================================================================================================


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list of per-worker results. Got: {results!r}")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; Stage 11 is TP=1-only and expects exactly 1.")
    return results[0]


def _collective_rpc_single_worker(engine: Any, method, args: tuple = (), *, label: str, ray_get: Optional[Callable] = None) -> Any:
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


# =================================================================================================
# Per-candidate lifecycle -- bounded-memory pattern ported from Stage 9's (already-fixed) design
# =================================================================================================


def evaluate_one_stage11_candidate_rpc(
    engine: Any, assignment: Stage11DirectionAssignment, region_param_names: Sequence[str],
    capability_contexts: Dict[str, CapabilityContext], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable] = None, generation_batch_size: int = STAGE11_GENERATE_BATCH_SIZE,
    rss_checkpoint: Optional[Callable[[str], None]] = None,
) -> List[ExperimentResultRecord]:
    """Identical lifecycle to Stage 8's evaluate_one_stage8_candidate_rpc (same v3 quantization
    -aware radius acceptance, same twice-per-candidate cache reset, same bounded generation
    path), but with Stage 9's bounded-memory candidate-lifecycle fix applied from the start:
    per-capability RunResult is `del`eted and release_transient_memory() called immediately after
    the values needed for the result row are extracted, and the failure path skips further RPCs
    against an already-dead Ray engine (_is_ray_unrecoverable_error, imported from Stage 9).
    """
    manifest = assignment.manifest
    if manifest.perturbation_mode != PERTURBATION_MODE:
        raise ValueError(f"Stage 11 only evaluates {PERTURBATION_MODE!r} manifests, got {manifest.perturbation_mode!r}")

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
                    "direction_family_id": assignment.direction_family_id, "direction_seed": assignment.direction_seed,
                    "direction_index": assignment.direction_index, "region": assignment.region,
                    "generation_batch_size": generation_batch_size, "model_scale": MODEL_SCALE_7B,
                    "stage8_parent_run_signature": STAGE8_PARENT_RUN_SIGNATURE,
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
            f"Exact fixed-base restoration failed after Stage-11 candidate {manifest.perturbation_id!r} "
            f"(region={manifest.anatomy_region!r}, radius={manifest.radius}, seed={manifest.seed}): "
            f"max_abs_drift={verification['max_abs_drift']}"
        )

    reset_vllm_encoder_cache_full(engine)
    for record in records:
        record.runtime_metadata["cache_reset_after_restoration"] = True

    return records


def run_stage11_rpc(
    plan: Stage11Plan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, seed_bank: Dict[str, Tuple[int, ...]], region_param_names_by_region: Dict[str, Sequence[str]],
    parameter_mask_hash_by_region: Dict[str, str], anatomy_audit: Dict[str, Any], *, run_benchmark: Callable, ray_get: Optional[Callable] = None,
) -> int:
    """Bounded-memory candidate loop (Stage 9's fixed pattern, applied from the start): retains
    only the SET of already-completed perturbation IDs, never their full row contents; returns
    only the COUNT of rows newly appended this call. Per-candidate RSS telemetry is written to
    its own candidate_memory_telemetry.jsonl, never embedded in the scientific rows.
    """
    from .mem_telemetry import release_transient_memory, rss_mb

    population_by_cell = build_stage11_population(plan, seed_bank, parameter_mask_hash_by_region)
    validate_stage11_direction_seed_reuse(plan, population_by_cell)

    current_checkpoint = build_stage11_checkpoint_manifest(plan, capability_contexts, parameter_mask_hash_by_region, seed_bank, anatomy_audit)
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    ensure_stage11_checkpoint_manifest(checkpoint_path, current_checkpoint)

    results_path = plan.output_dir / "results.jsonl"
    telemetry_path = plan.output_dir / "candidate_memory_telemetry.jsonl"
    completed_ids = set(load_completed_perturbation_rows(results_path, plan.capabilities).keys())

    newly_completed_rows = 0
    perturbation_index = 0
    previous_candidate_rss_mb: Optional[float] = None

    for (region, radius), assignments in population_by_cell.items():
        region_param_names = region_param_names_by_region[region]
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

            records = evaluate_one_stage11_candidate_rpc(
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
                "region": region, "radius": radius, "direction_index": assignment.direction_index,
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


# =================================================================================================
# CLI entry point
# =================================================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME_7B)
    parser.add_argument("--model-revision-ref", default="main", help="A git ref (branch/tag/SHA) to resolve to an immutable commit SHA before execution -- NEVER trusted as already pinned unless it is already a full 40-hex-character SHA.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "stage11_coarse_anatomical_atlas_7b"))
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"tiny live GPU smoke: 3 regions x 3 radii x 1 direction family x 6 capabilities x "
             f"{STAGE11_SMOKE_D_MAP_N} D_map examples/capability = 9 perturbations, 54 rows, 270 "
             f"perturbed model-example evaluations -- execution size only, same scientific protocol.",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU, no Hub call")
    args = parser.parse_args(argv)

    if args.dry_run:
        # A placeholder revision label is used ONLY for the dry-run print path (no Hub call, no
        # GPU) -- main()'s real (non-dry-run) path below ALWAYS resolves a genuine revision
        # before this plan's model_revision is ever persisted to a checkpoint or used to load a
        # model.
        placeholder_revision = "UNRESOLVED-dry-run-only"
        if args.smoke:
            plan = build_stage11_smoke_plan(model_name=args.model_name, model_revision=placeholder_revision, output_root=args.output_root)
        else:
            plan = build_stage11_plan(model_name=args.model_name, model_revision=placeholder_revision, output_root=args.output_root)
        print(f"Stage 11 plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}): regions={plan.regions} radii={plan.radii}")
        print(f"capabilities={plan.capabilities}")
        print(f"n_directions_per_cell={plan.n_directions_per_cell} d_map_n={plan.d_map_n} total_unique_perturbations={plan.total_unique_perturbations}")
        print(f"total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations}")
        print(f"total_perturbed_model_example_evaluations={plan.total_perturbed_model_example_evaluations}")
        print(f"stage8_parent_run_signature={STAGE8_PARENT_RUN_SIGNATURE}")
        print(f"output_dir={plan.output_dir}")
        print("(dry-run: model revision NOT resolved, no Hub/GPU call made)")
        return 0

    try:
        assert_feasible(
            f"Stage 11 coarse anatomical atlas 7B replication",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_module("huggingface_hub"), check_disk(REPO_ROOT, 50.0)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    resolution = resolve_immutable_model_revision(args.model_name, args.model_revision_ref)
    print(f"Resolved model revision: {resolution}")

    if args.smoke:
        plan = build_stage11_smoke_plan(model_name=args.model_name, model_revision=resolution["resolved_revision"], output_root=args.output_root)
    else:
        plan = build_stage11_plan(model_name=args.model_name, model_revision=resolution["resolved_revision"], output_root=args.output_root)

    print(f"Stage 11 plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}): regions={plan.regions} radii={plan.radii}")
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

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "model_revision_resolution.json").write_text(json.dumps(resolution, indent=2))

    engine_config = build_stage7b_engine_config()  # bf16 / max_model_len=4096 / gpu_mem=0.60 / TP=1 / enable_prefix_caching=False -- retained UNCHANGED for 7B; STOP (do not proceed) if load fails under this config
    assert engine_config["enable_prefix_caching"] is False, "Stage 11 must never run with prefix caching enabled."
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
        STAGE8_BASE_SEED, subset_ids_dir, plan.d_map_n,  # Stage 8's OWN base seed -- reuses Stage 8's D_map example manifests/subset hashes exactly
        load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
    )
    for capability, ctx in capability_contexts.items():
        write_data_role_manifest(ctx.partition, plan.output_dir / "data_roles" / f"{capability}_d_map.json")

    subset_hash_report = run_stage11_subset_hash_check(capability_contexts)
    (plan.output_dir / "stage11_subset_hash_check.json").write_text(json.dumps(subset_hash_report, indent=2))
    ensure_stage11_subset_hashes_match_stage8(subset_hash_report)
    print("Confirmed: live Stage-11 D_map subset hashes exactly match Stage-8's authoritative 3B manifests for all 6 capabilities.")

    seed_bank = build_stage11_direction_seed_bank(STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    (plan.output_dir / "direction_family_manifest.json").write_text(json.dumps(
        {"regions": list(plan.regions), "n_directions_per_cell": plan.n_directions_per_cell,
         "seed_bank": {r: list(s) for r, s in seed_bank.items()},
         "direction_seed_bank_hash": compute_direction_seed_bank_hash(seed_bank)}, indent=2,
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
            resolved_snapshot_path, precision=engine_config["precision"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"], max_model_len=engine_config["max_model_len"],
            tensor_parallel_size=engine_config["tensor_parallel_size"], enable_prefix_caching=engine_config["enable_prefix_caching"],
        )
        engine = engines[0]

        store_base_weights_via_rpc(engine)
        print(format_base_snapshot_confirmation(engine_config["gpu_memory_utilization"], engine_config["base_snapshot_mode"]))

        ensure_encoder_cache_reset_available(engine)
        print(f"Confirmed working multimodal-encoder-cache reset (multimodal_cache_policy={plan.multimodal_cache_policy!r}).")

        anatomy_audit = _collective_rpc_single_worker(engine, report_stage11_anatomy_audit, args=(plan.regions,), label="report_stage11_anatomy_audit")
        (plan.output_dir / "anatomy_audit.json").write_text(json.dumps(anatomy_audit, indent=2))
        ensure_stage11_anatomy_audit_passes(anatomy_audit, plan.regions)
        print(f"Confirmed: live 7B anatomy audit passed (union==full_model, pairwise disjoint) for {plan.regions}.")
        for region, info in anatomy_audit["regions"].items():
            print(f"  {region}: n_tensors={info['n_tensors']} n_elements={info['n_elements']} "
                  f"pct_of_total={info['percentage_of_total_elements']:.3f}% l2_norm={info['l2_norm']:.3f} mask_hash={info['mask_hash'][:12]}...")

        # Re-derives the exact param-name TUPLE for perturbation dispatch from the SAME live
        # atlas the audit already built (report_stage11_anatomy_audit only returns summary
        # counts/norms/hashes, never the full name list) -- reuses Stage 8/7B's own
        # report_region_param_names RPC BY IDENTITY (already a pure function of the live
        # model's named_parameters(), needing no 7B-specific code).
        region_info = _collective_rpc_single_worker(engine, report_region_param_names, args=(plan.regions,), label="report_region_param_names")
        region_param_names_by_region = {r: tuple(info["param_names"]) for r, info in region_info.items()}
        parameter_mask_hash_by_region = {r: anatomy_audit["regions"][r]["mask_hash"] for r in plan.regions}
        for region in plan.regions:
            if region_info[region]["mask_hash"] != parameter_mask_hash_by_region[region]:
                raise RuntimeError(
                    f"Mask-hash mismatch between report_region_param_names and report_stage11_anatomy_audit "
                    f"for region {region!r} -- the two RPCs disagree about live 7B partition membership."
                )

        llm_adapter = RayEngineLLMAdapter(engine)
        from .run_global_visual_thicket_pilot import load_or_compute_baseline_scores
        baseline_path = plan.output_dir / "baseline_scores.json"
        load_or_compute_baseline_scores(baseline_path, capability_contexts, plan.model_revision, plan.run_signature, llm_adapter, tokenizer, sampling_params)

        print("Running baseline repeatability preflight (7B baselines are NOT expected to equal 3B) before any Stage-11 perturbation...")
        preflight_report = run_baseline_repeatability_preflight_rpc(
            engine, capability_contexts, tokenizer, sampling_params, run_benchmark=run_benchmark,
            generation_batch_size=plan.generation_batch_size,
        )
        (plan.output_dir / "baseline_repeatability_preflight.json").write_text(json.dumps(preflight_report, indent=2))
        ensure_baseline_repeatability(preflight_report)
        print(f"Baseline repeatability preflight PASSED for all {len(preflight_report)} capabilities.")

        newly_written_rows = run_stage11_rpc(
            plan, capability_contexts, engine, tokenizer, sampling_params, seed_bank,
            region_param_names_by_region, parameter_mask_hash_by_region, anatomy_audit, run_benchmark=run_benchmark,
        )
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    manifest = write_stage11_run_manifest(plan.output_dir)
    print(f"Wrote {newly_written_rows} NEW result rows this run to {plan.output_dir / 'results.jsonl'}")
    print(f"Run manifest: {json.dumps(manifest, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
