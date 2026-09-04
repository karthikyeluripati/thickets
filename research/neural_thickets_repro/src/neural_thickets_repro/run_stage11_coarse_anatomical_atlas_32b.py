"""Stage 11 Track S2: 32B coarse-anatomy sweep -- vision / multimodal_connector_or_merger /
language x 3 frozen common relative-L2 radii x 6 frozen capabilities, at Qwen/Qwen2.5-VL-32B-
Instruct under TP=4. This is the SAME L1 anatomy design run_stage11_coarse_anatomical_atlas_7b.py
already executes at 7B (which is TP=1-only by construction) -- this is a NARROW, separate module
that reuses that module's scientific structure (plan construction, region x radius x direction
ordering, region-namespaced direction seed families, checkpoint identity, results.jsonl semantics,
completion manifest, subset gate, baseline repeatability, candidate lifecycle) BY IMPORT wherever
it is already scale/TP-agnostic, and replaces ONLY the execution/RPC layer with the already-proven
TP=4 32B distributed primitives run_stage11_whole_model_scaling.py's own 32B branch established
(collective_rpc_all_workers, the distributed anatomy audit, the distributed v3 solver, the
distributed reset/restore/verify primitives). The working 7B anatomy runner is NEVER modified by
this module -- 7B stays TP=1, unchanged, exactly as it was before this module existed.

WHY A NEW MODULE RATHER THAN GENERALIZING THE 7B ONE: run_stage11_coarse_anatomical_atlas_7b.py's
own `_collective_rpc_single_worker` hard-asserts exactly 1 per-worker RPC result ("Stage 11 is
TP=1-only") and its candidate evaluator dispatches the NON-distributed v3 solver primitive --
both structurally incompatible with TP=4. Refactoring that module to branch on scale (the pattern
run_stage11_whole_model_scaling.py itself uses for its own 3B/7B/32B `is_32b` branches) was
explicitly ruled out for THIS module by the task spec ("Do NOT refactor the working 7B anatomy
runner") -- so this module exists standalone instead, reusing every scale/TP-agnostic piece of the
7B module's own design BY IMPORT (never re-derived, never copy-pasted with silent drift risk).

FROZEN FULL S2 32B CONFIG -- byte-identical to Stage 8/Stage 11-7B's own frozen design, only the
model/TP/execution layer differs:
    model:        Qwen/Qwen2.5-VL-32B-Instruct, TP=4, BF16, max_model_len=4096,
                  gpu_memory_utilization=0.60, enforce_eager, prefix caching disabled,
                  cpu_base_weights CPU-resident base snapshot (never the legacy GPU-doubling mode)
    regions:      vision, multimodal_connector_or_merger, language (STAGE8_REGIONS, by identity)
    radii:        STAGE8_RADII, by identity -- NOT recalibrated for 32B (deliberate)
    capabilities: STAGE8_CAPABILITIES, by identity
    directions:   64 per (region, radius) cell, independent 32B seed namespace
                  (scaling_common.build_scaling_direction_seed_bank, scale_label="32B"), same
                  direction reused across the 3 radii within one region
    data:         the SAME D_map N=50 example manifests as Stage 8 (STAGE8_BASE_SEED reused,
                  exact subset-hash-equality gate against STAGE8_AUTHORITATIVE_SUBSET_HASHES)
Total:  3 regions x 3 radii x 64 directions = 576 unique perturbations
        576 x 6 capabilities = 3456 perturbation x capability result rows
        3456 x 50 = 172,800 perturbed model-example evaluations

SMOKE: 3 regions x 3 radii x 1 direction x 6 capabilities x D_map N=5 = 9 perturbations, 54 rows,
270 perturbed model-example evaluations.

LIVE S2 READINESS -- see stage11_32b_s2_live_evidence.py: BOTH smoke and full 32B S2 require a
canonical, strictly identity-bound S2 live-readiness artifact proving ALL THREE regions
(vision/multimodal_connector_or_merger/language) individually pass the strict distributed-v3
solver + rank-consensus + exact-restoration checks, gathered in ONE live TP=4 engine session
(diagnostics/stage11_32b_s2_live_v3_solver_probe.py) -- never inherited from S1's own (single-
region) solver-probe evidence. G1/G2/G3/G6/G7/G8 (TP/hardware/snapshot/test facts, not region-
scoped) are reused from S1's own validated base G1-G8 artifact UNCHANGED.

DO NOT DO YET: S1 results are never touched by this module (separate output_dir, separate
checkpoint identity, separate readiness artifact). 72B remains hard-disabled (this module is
32B-only; 72B is not a valid --scale choice here at all). No depth/head/MLP experiments, no new
scientific axis.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Live-discovered NVLink-less/cross-NUMA topology fix (2026-09-04, real 4xL40S TP=4 pod) --
# IDENTICAL to the fix already required by diagnostics/stage11_32b_live_v3_solver_probe.py and
# diagnostics/stage11_32b_s2_live_v3_solver_probe.py. Without it, this module's own engine
# launch + first collective RPC hung indefinitely (observed live: 1h10m, 100% GPU utilization
# on all 4 ranks, near-zero GPU memory resident, zero forward progress) inside NCCL collective
# init -- confirmed root cause by comparing against the two probe scripts above, which set this
# and do NOT hang. setdefault (not a hard override) so an operator's own explicit NCCL_P2P_DISABLE
# is never silently clobbered. This is infrastructure/topology config only -- touches no
# scientific parameter (radius, seed, capability, threshold, or scoring rule).
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

from .env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from .run_global_visual_thicket_pilot import (
    RESTORATION_MODE,
    CapabilityContext,
    RayEngineLLMAdapter,
    RestorationFailedError,
    append_candidate_rows,
    format_runtime_compatibility_diagnostic,
    detect_vllm_engine_mode,
    get_vllm_version,
    launch_stage6_engine,
    load_completed_perturbation_rows,
    load_records,
    load_or_compute_baseline_scores,
)
from .run_stage7b_anatomical_calibration import (
    ENABLE_PREFIX_CACHING,
    MULTIMODAL_CACHE_POLICY,
    PERTURBATION_MODE,
    REALIZED_RADIUS_TOLERANCE,
    EncoderCacheResetUnavailableError,
    RealizedRadiusMismatchError,
    ensure_encoder_cache_reset_available,
    ensure_stage7b_encoder_cache_reset_mechanism_exposed,
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
from .run_stage11_coarse_anatomical_atlas_7b import (
    STAGE8_PARENT_RUN_SIGNATURE,
    DirectionSeedReuseViolationError,
    Stage11DirectionAssignment,
    build_stage11_population,
    validate_stage11_direction_seed_reuse,
)
from .run_stage11_whole_model_scaling import (
    SUBSET_GATE_MODE_FULL,
    SUBSET_GATE_MODE_SMOKE,
    Stage11SmokeSubsetNondeterminismError,
    Stage11SubsetHashMismatchError,
    build_subset_gate_report,
    ensure_smoke_subset_determinism,
    ensure_subset_gate_passes,
    ensure_subset_hashes_match_stage8,
    reset_to_base_weights_distributed_rpc,
    restore_and_verify_distributed_rpc,
    run_smoke_subset_determinism_check,
    run_subset_hash_check,
)
from .scaling_common import (
    MODEL_FAMILY,
    ScalingModelSpec,
    build_scaling_direction_seed_bank,
    compute_anatomy_audit_hash,
    compute_direction_seed_bank_hash,
    ensure_scaling_anatomy_audit_passes,
    get_scaling_model_spec,
    report_region_param_names_for_scaling,
    resolve_immutable_model_revision,
)
from .scoped_anatomical_perturbation import QUANTIZATION_PLATEAU_RELATIVE_TOLERANCE
from .stage11_32b_readiness import build_32b_engine_config
from .thicket.perturbation import PERTURBATION_MODES
from .thicket.schema import ExperimentResultRecord
from .vlm_adapter import reset_vllm_encoder_cache_full

assert PERTURBATION_MODE in PERTURBATION_MODES

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

EXPERIMENT_ID = "stage11_coarse_anatomical_atlas_32b"
MODEL_SCALE_32B = "32B"
TRACK = "anatomy"
DATASET_ROLE = "map"

# --- FROZEN full-S2-32B config -- byte-identical to Stage 8/11-7B's own, by import ------------
STAGE11_32B_REGIONS: Tuple[str, ...] = STAGE8_REGIONS
STAGE11_32B_RADII: Tuple[float, ...] = STAGE8_RADII
STAGE11_32B_CAPABILITIES: Tuple[str, ...] = STAGE8_CAPABILITIES
STAGE11_32B_N_DIRECTIONS_PER_CELL: int = STAGE8_N_DIRECTIONS_PER_CELL
STAGE11_32B_D_MAP_N: int = STAGE8_D_MAP_N
STAGE11_32B_GENERATE_BATCH_SIZE: int = STAGE8_GENERATE_BATCH_SIZE
STAGE11_32B_SMOKE_N_DIRECTIONS = STAGE8_SMOKE_N_DIRECTIONS
STAGE11_32B_SMOKE_D_MAP_N = STAGE8_SMOKE_D_MAP_N
_ALLOWED_D_MAP_SIZES: Tuple[int, ...] = (STAGE11_32B_D_MAP_N, STAGE11_32B_SMOKE_D_MAP_N)

_FULL_RUN_SIGNATURE = "stage11_32b_anatomy_v1"


def _base_seed_for_32b_anatomy() -> int:
    """A distinct, deterministically-derived STAGE11_BASE_SEED-shaped constant for 32B S2 -- NOT
    the 7B module's own literal STAGE11_BASE_SEED (20260904), and NOT whole_model's own per-scale
    derivation either (that one folds only scale_label into the direction-seed-BANK namespace,
    not this base-seed value) -- derived from one frozen root so 32B S2 can never accidentally
    reuse 7B anatomy's exact seed bank.
    """
    from .thicket.seeds import derive_seed

    return derive_seed(20260904, "stage11_32b_anatomy_base_seed") % (2 ** 31)


STAGE11_32B_BASE_SEED = _base_seed_for_32b_anatomy()


# =================================================================================================
# Plan (pure arithmetic, no I/O, no GPU)
# =================================================================================================


def compute_stage11_32b_run_signature(
    regions: Sequence[str], radii: Sequence[float], n_directions: int, d_map_n: int, generation_batch_size: int,
) -> str:
    is_frozen_full_scientific_config = (
        tuple(regions) == STAGE11_32B_REGIONS and tuple(radii) == STAGE11_32B_RADII
        and n_directions == STAGE11_32B_N_DIRECTIONS_PER_CELL and d_map_n == STAGE11_32B_D_MAP_N
    )
    if is_frozen_full_scientific_config:
        if generation_batch_size == STAGE11_32B_GENERATE_BATCH_SIZE:
            return _FULL_RUN_SIGNATURE
        return f"stage11_32b_anatomy_batched{generation_batch_size}"
    region_label = "-".join(regions)
    radius_label = "-".join(f"{r:.6f}".replace(".", "") for r in radii)
    return f"stage11_smoke_32b_anatomy_{region_label}_r{radius_label}_n{d_map_n}_dir{n_directions}"


@dataclass(frozen=True)
class Stage11_32BPlan:
    model_name: str
    model_revision: str
    model_family: str
    model_scale: str  # ALWAYS "32B" -- named `model_scale` (not `scale_label`) so this plan is
                       # structurally interchangeable with run_stage11_coarse_anatomical_atlas_7b.
                       # Stage11Plan for build_stage11_population/validate_stage11_direction_seed_
                       # reuse, reused BY IMPORT unmodified from that module.
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
    tensor_parallel_size: int = 4
    generation_batch_size: int = STAGE11_32B_GENERATE_BATCH_SIZE

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
            self.regions == STAGE11_32B_REGIONS and self.radii == STAGE11_32B_RADII
            and self.n_directions_per_cell == STAGE11_32B_N_DIRECTIONS_PER_CELL and self.d_map_n == STAGE11_32B_D_MAP_N
        )


def build_stage11_32b_plan(
    *, model_name: str, model_revision: str, output_root: "str | Path",
    regions: Sequence[str] = STAGE11_32B_REGIONS, radii: Sequence[float] = STAGE11_32B_RADII,
    n_directions_per_cell: int = STAGE11_32B_N_DIRECTIONS_PER_CELL, d_map_n: int = STAGE11_32B_D_MAP_N,
    generation_batch_size: int = STAGE11_32B_GENERATE_BATCH_SIZE, tensor_parallel_size: int = 4,
) -> Stage11_32BPlan:
    if not regions:
        raise ValueError("Stage 11 32B S2 requires at least one anatomy region.")
    if not radii:
        raise ValueError("Stage 11 32B S2 requires at least one common radius.")
    if d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 11 32B S2 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {d_map_n}")
    if generation_batch_size <= 0:
        raise ValueError(f"generation_batch_size must be positive, got {generation_batch_size}")
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")

    run_signature = compute_stage11_32b_run_signature(regions, radii, n_directions_per_cell, d_map_n, generation_batch_size)
    return Stage11_32BPlan(
        model_name=model_name, model_revision=model_revision, model_family=MODEL_FAMILY, model_scale=MODEL_SCALE_32B,
        regions=tuple(regions), radii=tuple(radii), capabilities=STAGE11_32B_CAPABILITIES,
        n_directions_per_cell=n_directions_per_cell, d_map_n=d_map_n,
        radius_realization_method="fixed_direction_bf16_quantization_aware_v3_distributed",
        multimodal_cache_policy=MULTIMODAL_CACHE_POLICY, enable_prefix_caching=ENABLE_PREFIX_CACHING,
        run_signature=run_signature, output_dir=Path(output_root) / run_signature,
        tensor_parallel_size=tensor_parallel_size, generation_batch_size=generation_batch_size,
    )


def build_stage11_32b_smoke_plan(*, model_name: str, model_revision: str, output_root: "str | Path", tensor_parallel_size: int = 4) -> Stage11_32BPlan:
    """3 regions x 3 radii x 1 direction family x 6 capabilities x D_map N=5 = 9 perturbations,
    54 rows, 270 perturbed model-example evaluations.
    """
    return build_stage11_32b_plan(
        model_name=model_name, model_revision=model_revision, output_root=output_root,
        regions=STAGE11_32B_REGIONS, radii=STAGE11_32B_RADII, n_directions_per_cell=STAGE11_32B_SMOKE_N_DIRECTIONS,
        d_map_n=STAGE11_32B_SMOKE_D_MAP_N, tensor_parallel_size=tensor_parallel_size,
    )


# =================================================================================================
# Checkpoint identity
# =================================================================================================


class IncompatibleStage11_32BCheckpointError(RuntimeError):
    """An existing checkpoint_manifest.json in this output directory does not match the current
    run's identity -- refuses to resume a differently-configured partial run. Fails closed on any
    mismatch of model revision, region hashes, radii, subset hashes, direction bank, tensor-
    parallel size, or any other scientific-contract field below.
    """


@dataclass(frozen=True)
class Stage11_32BCheckpointManifest:
    experiment_id: str
    run_signature: str
    scale_label: str
    track: str
    tensor_parallel_size: int
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
            "experiment_id": self.experiment_id, "run_signature": self.run_signature, "scale_label": self.scale_label, "track": self.track,
            "tensor_parallel_size": self.tensor_parallel_size,
            "restoration_mode": self.restoration_mode, "perturbation_mode": self.perturbation_mode, "perturbation_semantics": self.perturbation_mode,
            "radius_realization_method": self.radius_realization_method, "multimodal_cache_policy": self.multimodal_cache_policy,
            "enable_prefix_caching": self.enable_prefix_caching, "generation_batch_size": self.generation_batch_size,
            "model_revision": self.model_revision, "dataset_role": self.dataset_role,
            "regions": list(self.regions), "radii": list(self.radii), "capabilities": list(self.capabilities),
            "n_directions_per_cell": self.n_directions_per_cell, "d_map_n": self.d_map_n,
            "subset_hashes": dict(sorted(self.subset_hashes.items())), "region_mask_hashes": dict(sorted(self.region_mask_hashes.items())),
            "direction_seed_bank_hash": self.direction_seed_bank_hash, "anatomy_audit_hash": self.anatomy_audit_hash,
            "stage8_parent_run_signature": self.stage8_parent_run_signature,
            "expected_unique_perturbations": self.expected_unique_perturbations, "expected_result_rows": self.expected_result_rows,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage11_32BCheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], run_signature=d["run_signature"], scale_label=d["scale_label"], track=d["track"],
            tensor_parallel_size=d["tensor_parallel_size"], restoration_mode=d["restoration_mode"], perturbation_mode=d["perturbation_mode"],
            radius_realization_method=d["radius_realization_method"], multimodal_cache_policy=d["multimodal_cache_policy"],
            enable_prefix_caching=d["enable_prefix_caching"], generation_batch_size=d["generation_batch_size"],
            model_revision=d["model_revision"], dataset_role=d["dataset_role"],
            regions=tuple(d["regions"]), radii=tuple(d["radii"]), capabilities=tuple(d["capabilities"]),
            n_directions_per_cell=d["n_directions_per_cell"], d_map_n=d["d_map_n"],
            subset_hashes=dict(d["subset_hashes"]), region_mask_hashes=dict(d["region_mask_hashes"]),
            direction_seed_bank_hash=d["direction_seed_bank_hash"], anatomy_audit_hash=d["anatomy_audit_hash"],
            stage8_parent_run_signature=d["stage8_parent_run_signature"],
            expected_unique_perturbations=d["expected_unique_perturbations"], expected_result_rows=d["expected_result_rows"],
        )


def build_stage11_32b_checkpoint_manifest(
    plan: Stage11_32BPlan, capability_contexts: Dict[str, CapabilityContext], region_mask_hashes: Dict[str, str],
    seed_bank: Dict[str, Tuple[int, ...]], anatomy_audit: Dict[str, Any],
) -> Stage11_32BCheckpointManifest:
    if plan.d_map_n not in _ALLOWED_D_MAP_SIZES:
        raise DatasetRoleViolationError(f"Stage 11 32B S2 D_map size must be one of {_ALLOWED_D_MAP_SIZES}, got {plan.d_map_n}")
    missing_regions = set(plan.regions) - set(region_mask_hashes)
    if missing_regions:
        raise ValueError(f"Missing region_mask_hashes for region(s): {sorted(missing_regions)}")
    return Stage11_32BCheckpointManifest(
        experiment_id=EXPERIMENT_ID, run_signature=plan.run_signature, scale_label=plan.model_scale, track=TRACK,
        tensor_parallel_size=plan.tensor_parallel_size, restoration_mode=RESTORATION_MODE, perturbation_mode=PERTURBATION_MODE,
        radius_realization_method=plan.radius_realization_method, multimodal_cache_policy=plan.multimodal_cache_policy,
        enable_prefix_caching=plan.enable_prefix_caching, generation_batch_size=plan.generation_batch_size,
        model_revision=plan.model_revision, dataset_role=DATASET_ROLE,
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


def ensure_stage11_32b_checkpoint_manifest(path: Path, current: Stage11_32BCheckpointManifest) -> Stage11_32BCheckpointManifest:
    if path.exists():
        existing = Stage11_32BCheckpointManifest.from_dict(json.loads(path.read_text()))
        if existing != current:
            raise IncompatibleStage11_32BCheckpointError(
                f"Existing checkpoint at {path} is incompatible with this run -- refusing to resume: "
                f"existing={existing.to_dict()} current={current.to_dict()}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current.to_dict(), indent=2))
    return current


def build_stage11_32b_run_manifest_summary(checkpoint: Stage11_32BCheckpointManifest, records: List[ExperimentResultRecord]) -> Dict[str, Any]:
    actual_unique_perturbations = len({r.perturbation_id for r in records})
    actual_result_rows = len(records)
    run_complete = (actual_unique_perturbations == checkpoint.expected_unique_perturbations and actual_result_rows == checkpoint.expected_result_rows)
    return {**checkpoint.to_dict(), "actual_unique_perturbations": actual_unique_perturbations, "actual_result_rows": actual_result_rows, "run_complete": run_complete}


def write_stage11_32b_run_manifest(output_dir: Path) -> Dict[str, Any]:
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_manifest.json found at {checkpoint_path} -- this run never started (or output_dir is wrong).")
    checkpoint = Stage11_32BCheckpointManifest.from_dict(json.loads(checkpoint_path.read_text()))
    records = load_records(output_dir / "results.jsonl")
    manifest = build_stage11_32b_run_manifest_summary(checkpoint, records)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# =================================================================================================
# 32B S2 live readiness gate -- see stage11_32b_s2_live_evidence.py's own module docstring for the
# full rationale (why S1's single-region solver evidence cannot authorize S2 on its own).
# =================================================================================================


def run_32b_s2_readiness_preflight_and_report(
    *, resolved_revision: str, tensor_parallel_size: int, output_dir: Any,
    base_live_evidence_dir: Optional[Any] = None, s2_live_evidence_dir: Optional[Any] = None,
    current_gpu_uuids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """The function main() calls for 32B S2, BEFORE any engine launch. Mirrors stage11_32b_
    readiness.run_32b_readiness_preflight_and_report's own shape exactly, but consumes S2's own
    multi-region canonical evidence (stage11_32b_s2_live_evidence.load_and_validate_canonical_s2_
    live_evidence) instead of S1's single-region one. UNLIKE S1's preflight, this NEVER falls back
    to a CPU-design-proof READY_FOR_LIVE_VERIFICATION for G4/G5 when no valid evidence is found --
    S2 has no CPU-only proof story that covers all three regions' real shard shapes; G4/G5 stay
    NOT_YET_VERIFIED until a genuine multi-region live probe actually exists.
    """
    from .stage11_32b_live_evidence import DEFAULT_LIVE_READINESS_EVIDENCE_DIR, LiveEvidenceIdentityRequirement, query_live_gpu_uuids
    from .stage11_32b_readiness import GATE_IDS, GATE_NOT_YET_VERIFIED, GATE_PASS, Stage32BSmokeNotPermittedError, ensure_32b_smoke_permitted
    from .stage11_32b_s2_live_evidence import DEFAULT_S2_LIVE_READINESS_EVIDENCE_DIR, load_and_validate_canonical_s2_live_evidence

    requirement = LiveEvidenceIdentityRequirement(resolved_revision=resolved_revision, tensor_parallel_size=tensor_parallel_size)
    evidence = load_and_validate_canonical_s2_live_evidence(
        requirement,
        base_evidence_dir=base_live_evidence_dir if base_live_evidence_dir is not None else DEFAULT_LIVE_READINESS_EVIDENCE_DIR,
        s2_evidence_dir=s2_live_evidence_dir if s2_live_evidence_dir is not None else DEFAULT_S2_LIVE_READINESS_EVIDENCE_DIR,
        current_gpu_uuids=current_gpu_uuids if current_gpu_uuids is not None else query_live_gpu_uuids(),
    )

    if evidence["ok"]:
        gate_results = evidence["gate_results"]
    else:
        gate_results = {g: GATE_NOT_YET_VERIFIED for g in GATE_IDS}

    all_gates_pass = bool(gate_results) and all(v == GATE_PASS for v in gate_results.values())
    report_dict = {
        "scale_label": MODEL_SCALE_32B, "track": TRACK, "resolved_revision": resolved_revision,
        "intended_tp_size": tensor_parallel_size, "regions": list(STAGE11_32B_REGIONS),
        "gate_results": gate_results, "all_gates_pass": all_gates_pass,
        "live_evidence_found": evidence["found"], "live_evidence_ok": evidence["ok"], "live_evidence_reasons": evidence["reasons"],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage11_32b_s2_readiness_gate_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))

    try:
        ensure_32b_smoke_permitted(gate_results)
        smoke_permitted = True
    except Stage32BSmokeNotPermittedError:
        smoke_permitted = False

    return {"report_path": report_path, "smoke_permitted": smoke_permitted, "gate_results": gate_results, "live_evidence": evidence}


# =================================================================================================
# Per-candidate lifecycle -- TP=4-aware analog of run_stage11_coarse_anatomical_atlas_7b.evaluate_
# one_stage11_candidate_rpc, generalizing run_stage11_whole_model_scaling.evaluate_one_whole_model_
# candidate_distributed_rpc's own TP=4 dispatch from ONE fixed region (whole_model) to the
# assignment's OWN region (vision / multimodal_connector_or_merger / language) -- byte-identical
# transactional ordering and ExperimentResultRecord schema otherwise.
# =================================================================================================


def evaluate_one_stage11_32b_candidate_distributed_rpc(
    engine: Any, assignment: Stage11DirectionAssignment, region_param_names: Sequence[str],
    capability_contexts: Dict[str, CapabilityContext], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, ray_get: Optional[Callable] = None, generation_batch_size: int = STAGE11_32B_GENERATE_BATCH_SIZE,
    rss_checkpoint: Optional[Callable[[str], None]] = None, tensor_parallel_size: int = 4,
) -> List[ExperimentResultRecord]:
    manifest = assignment.manifest
    if manifest.perturbation_mode != PERTURBATION_MODE:
        raise ValueError(f"Stage 11 32B S2 only evaluates {PERTURBATION_MODE!r} manifests, got {manifest.perturbation_mode!r}")

    from .mem_telemetry import release_transient_memory
    from .thicket.distributed_perturbation import collective_rpc_all_workers
    from .thicket.distributed_v3_solver import (
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed,
        verify_solver_rank_consensus,
    )
    from .thicket.vllm_shard_mapping import build_shard_specs_for_region

    def _checkpoint(label: str) -> None:
        if rss_checkpoint is not None:
            rss_checkpoint(label)

    _checkpoint("before_candidate")

    def _apply_distributed_v3_rpc(worker_self):
        shard_specs = build_shard_specs_for_region(worker_self.model_runner.model, region_param_names)
        result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed(
            worker_self, manifest.seed, manifest.radius, manifest.anatomy_region, region_param_names, shard_specs,
        )
        result["rank"] = getattr(worker_self, "rank", None)
        return result

    llm_adapter = RayEngineLLMAdapter(engine, ray_get=ray_get)
    records: List[ExperimentResultRecord] = []
    try:
        per_rank_results = collective_rpc_all_workers(
            engine, _apply_distributed_v3_rpc, label="distributed_v3_solver_candidate", expected_world_size=tensor_parallel_size, ray_get=ray_get,
        )
        verify_solver_rank_consensus(per_rank_results)  # hard-fail on any rank disagreement, never averaged
        apply_result = per_rank_results[0]

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
                    "cache_reset_before_evaluation": True, "cache_reset_after_restoration": False,
                    "direction_family_id": assignment.direction_family_id, "direction_seed": assignment.direction_seed,
                    "direction_index": assignment.direction_index, "region": assignment.region,
                    "generation_batch_size": generation_batch_size, "model_scale": MODEL_SCALE_32B, "track": TRACK,
                    "tensor_parallel_size": tensor_parallel_size, "distributed_rank_consensus_verified": True,
                    "stage8_parent_run_signature": STAGE8_PARENT_RUN_SIGNATURE,
                },
            ))
            del result
            release_transient_memory()
            _checkpoint(f"after_capability_{capability}")
    except Exception as exc:
        if _is_ray_unrecoverable_error(exc):
            raise
        restore_and_verify_distributed_rpc(engine, tensor_parallel_size=tensor_parallel_size, ray_get=ray_get)
        raise

    verification = restore_and_verify_distributed_rpc(engine, tensor_parallel_size=tensor_parallel_size, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Exact fixed-base restoration failed after Stage-11 32B S2 candidate {manifest.perturbation_id!r} "
            f"(region={manifest.anatomy_region!r}, radius={manifest.radius}, seed={manifest.seed}): "
            f"global_max_abs_drift={verification['global_max_abs_drift']}"
        )

    reset_vllm_encoder_cache_full(engine)
    for record in records:
        record.runtime_metadata["cache_reset_after_restoration"] = True

    return records


def run_stage11_32b_rpc(
    plan: Stage11_32BPlan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, seed_bank: Dict[str, Tuple[int, ...]], region_param_names_by_region: Dict[str, Sequence[str]],
    parameter_mask_hash_by_region: Dict[str, str], anatomy_audit: Dict[str, Any], *, run_benchmark: Callable,
    ray_get: Optional[Callable] = None, evaluate_one_candidate: Optional[Callable] = None,
) -> int:
    """Bounded-memory candidate loop -- byte-identical structure to run_stage11_coarse_anatomical_
    atlas_7b.run_stage11_rpc, with the TP=4 distributed evaluator substituted via the SAME late-
    binding injection pattern established project-wide (`evaluate_one_candidate=None` resolves to
    evaluate_one_stage11_32b_candidate_distributed_rpc AT CALL TIME).

    CHECKPOINT/RESUME: `append_candidate_rows` is called ONLY after a candidate's full apply ->
    evaluate (all `len(plan.capabilities)`) -> restore -> verify cycle has ALREADY succeeded (see
    evaluate_one_stage11_32b_candidate_distributed_rpc above) -- a candidate that crashes mid-way
    writes ZERO rows for that perturbation_id (its in-memory `records` list is simply lost), so a
    resumed run's `load_completed_perturbation_rows` (exact-capability-set-complete rows only)
    never sees a partial group, `completed_ids` naturally excludes it, and the loop below re-
    attempts it FROM SCRATCH -- never appending a second, duplicate, or partial row for an already
    -complete perturbation_id, and never skipping an incomplete one.
    """
    from .mem_telemetry import release_transient_memory, rss_mb

    if evaluate_one_candidate is None:
        evaluate_one_candidate = evaluate_one_stage11_32b_candidate_distributed_rpc

    population_by_cell = build_stage11_population(plan, seed_bank, parameter_mask_hash_by_region)
    validate_stage11_direction_seed_reuse(plan, population_by_cell)

    current_checkpoint = build_stage11_32b_checkpoint_manifest(plan, capability_contexts, parameter_mask_hash_by_region, seed_bank, anatomy_audit)
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    ensure_stage11_32b_checkpoint_manifest(checkpoint_path, current_checkpoint)

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

            records = evaluate_one_candidate(
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
                "delta_from_previous_candidate_mb": ((rss_after_cleanup_mb - previous_candidate_rss_mb) if previous_candidate_rss_mb is not None else 0.0),
                "high_water_mb": high_water_mb,
            })
            previous_candidate_rss_mb = rss_after_cleanup_mb
            perturbation_index += 1

    return newly_completed_rows


# =================================================================================================
# CLI entry point
# =================================================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Defaults to 4 (task spec Section 3).")
    parser.add_argument("--model-revision-ref", default=None, help="Overrides the registry's default revision_ref (\"main\") if given.")
    parser.add_argument(
        "--base-live-evidence-dir", default=None,
        help="Overrides the fixed S1 G1-G8 evidence location (stage11_32b_live_evidence."
             "DEFAULT_LIVE_READINESS_EVIDENCE_DIR) this module reuses for G1/G2/G3/G6/G7/G8. Omit for normal use.",
    )
    parser.add_argument(
        "--s2-live-evidence-dir", default=None,
        help="Overrides the fixed S2 multi-region solver-probe evidence location (stage11_32b_s2_"
             "live_evidence.DEFAULT_S2_LIVE_READINESS_EVIDENCE_DIR). Omit for normal use.",
    )
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "stage11_coarse_anatomical_atlas_32b"))
    parser.add_argument("--smoke", action="store_true", help="3 regions x 3 radii x 1 direction family x 6 capabilities x 5 D_map examples = 9 perturbations, 54 rows, 270 evaluations.")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU, no Hub call")
    args = parser.parse_args(argv)

    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 4
    spec = get_scaling_model_spec("32B")
    if args.model_revision_ref is not None:
        spec = ScalingModelSpec(scale_label=spec.scale_label, model_name=spec.model_name, revision_ref=args.model_revision_ref, model_family=spec.model_family)

    if args.dry_run:
        placeholder_revision = "UNRESOLVED-dry-run-only"
        plan = (build_stage11_32b_smoke_plan if args.smoke else build_stage11_32b_plan)(
            model_name=spec.model_name, model_revision=placeholder_revision, output_root=args.output_root, tensor_parallel_size=tp_size,
        )
        print(f"Stage 11 32B S2 plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}): regions={plan.regions} radii={plan.radii}")
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
            "Stage 11 32B S2 coarse anatomical atlas", [check_cuda(), check_module("vllm"), check_module("ray"), check_module("huggingface_hub"), check_disk(REPO_ROOT, 50.0)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    resolution = resolve_immutable_model_revision(spec.model_name, spec.revision_ref)
    print(f"Resolved model revision: {resolution}")

    plan = (build_stage11_32b_smoke_plan if args.smoke else build_stage11_32b_plan)(
        model_name=spec.model_name, model_revision=resolution["resolved_revision"], output_root=args.output_root, tensor_parallel_size=tp_size,
    )
    print(f"Stage 11 32B S2 plan (run_signature={plan.run_signature}, is_smoke={plan.is_smoke}): regions={plan.regions} radii={plan.radii}")
    print(f"total_unique_perturbations={plan.total_unique_perturbations} total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations} total_perturbed_model_example_evaluations={plan.total_perturbed_model_example_evaluations}")

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "model_revision_resolution.json").write_text(json.dumps(resolution, indent=2))

    # 32B S2 READINESS GATE -- must pass before ANY of the shared lifecycle below runs. Never
    # falls back to CPU-only READY_FOR_LIVE_VERIFICATION for G4/G5 (see run_32b_s2_readiness_
    # preflight_and_report's own docstring) -- only a genuine, all-three-regions-PASS multi-region
    # live probe authorizes continuation, for BOTH --smoke and full (never distinguished by this
    # gate, exactly like the whole_model 32B branch's own readiness gate).
    preflight = run_32b_s2_readiness_preflight_and_report(
        resolved_revision=resolution["resolved_revision"], tensor_parallel_size=tp_size, output_dir=plan.output_dir,
        base_live_evidence_dir=args.base_live_evidence_dir, s2_live_evidence_dir=args.s2_live_evidence_dir,
    )
    print(f"32B S2 readiness gate report written to {preflight['report_path']}")
    print(f"32B S2 gate results: {preflight['gate_results']}")
    live_evidence = preflight["live_evidence"]
    print(f"32B S2 live-evidence lookup: found={live_evidence['found']} ok={live_evidence['ok']} reasons={live_evidence['reasons']}")
    if not preflight["smoke_permitted"]:
        print("32B S2 (anatomy) BLOCKED -- not all G1-G8 gates report PASS.", file=sys.stderr)
        if live_evidence["found"] and not live_evidence["ok"]:
            print(
                "A live S2 readiness evidence artifact WAS found but failed identity-binding/validation "
                "against THIS invocation -- it will never silently authorize a mismatched run. See the "
                "reasons printed above.", file=sys.stderr,
            )
        else:
            print(
                "No live S2 readiness evidence was found for this model/revision/TP configuration -- run "
                "diagnostics/stage11_32b_s2_live_v3_solver_probe.py on real TP hardware first (all three "
                "regions -- vision, multimodal_connector_or_merger, language -- must PASS in ONE artifact). "
                "No engine was launched, no GPU memory was touched, no scientific row can exist for this attempt.",
                file=sys.stderr,
            )
        return 0
    print("32B S2 readiness gates ALL PASS (validated live evidence) -- continuing into the scientific candidate lifecycle.")

    engine_config = build_32b_engine_config(tensor_parallel_size=tp_size)
    assert engine_config["enable_prefix_caching"] is False, "Stage 11 32B S2 must never run with prefix caching enabled."
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

    # MODE-AWARE subset gate -- reuses run_stage11_whole_model_scaling.py's own already-validated
    # smoke/full split BY IMPORT (generic over Dict[str, CapabilityContext] + d_map_n, never
    # whole_model- or 7B-specific), so this exact policy can never drift independently a third time.
    if plan.is_smoke:
        print(f"Running Stage-11 32B S2 SMOKE subset gate (D_map N={plan.d_map_n} deterministic reconstruction -- never compared to Stage-8's N=50 authoritative hashes)...")
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
        print("Confirmed: Stage-11 32B S2 smoke D_map subset construction is deterministic across two independent builds for all 6 capabilities.")
    else:
        full_subset_hash_report = run_subset_hash_check(capability_contexts)
        subset_gate_report = build_subset_gate_report(is_smoke=False, d_map_n=plan.d_map_n, full_subset_hash_report=full_subset_hash_report)
        (plan.output_dir / "subset_gate.json").write_text(json.dumps(subset_gate_report, indent=2))
        ensure_subset_gate_passes(subset_gate_report)  # HARD STOP before model execution if the authoritative Stage8/Stage11 N=50 subset hashes differ
        print("Confirmed: live Stage-11 32B S2 D_map subset hashes exactly match Stage-8's authoritative 3B manifests for all 6 capabilities.")

    seed_bank = build_scaling_direction_seed_bank(STAGE11_32B_BASE_SEED, MODEL_SCALE_32B, plan.regions, plan.n_directions_per_cell)
    (plan.output_dir / "direction_family_manifest.json").write_text(json.dumps(
        {"regions": list(plan.regions), "n_directions_per_cell": plan.n_directions_per_cell,
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

        from .thicket.cpu_base_snapshot import store_base_weights_cpu_rpc
        from .thicket.distributed_perturbation import collective_rpc_all_workers

        collective_rpc_all_workers(engine, store_base_weights_cpu_rpc, label="store_base_weights_cpu", expected_world_size=tp_size)
        print(f"Confirmed working CPU base snapshot (base_snapshot_mode={engine_config['base_snapshot_mode']!r}).")

        ensure_encoder_cache_reset_available(engine)
        print(f"Confirmed working multimodal-encoder-cache reset (multimodal_cache_policy={MULTIMODAL_CACHE_POLICY!r}).")

        # ---------------------------------------------------------------------------------------
        # ANATOMY AUDIT (task spec): globally inventory vision/connector/language on TP=4, require
        # rank-consensus, pairwise disjointness, and union == the complete perturbable model
        # parameter set -- reuses thicket.distributed_anatomy_audit BY IMPORT (already generic
        # over region_labels, already TP-rank-consensus-checked; built for whole_model, correct
        # for anatomy without modification per that module's own docstring).
        # ---------------------------------------------------------------------------------------
        from .thicket.distributed_anatomy_audit import report_global_anatomy_audit_rpc, verify_anatomy_audit_rank_consensus

        per_rank_anatomy_audit = collective_rpc_all_workers(
            engine, report_global_anatomy_audit_rpc, args=(plan.regions, MODEL_FAMILY), label="report_global_anatomy_audit", expected_world_size=tp_size,
        )
        verify_anatomy_audit_rank_consensus(per_rank_anatomy_audit)  # hard-fails on any rank disagreement
        anatomy_audit = per_rank_anatomy_audit[0]
        (plan.output_dir / "anatomy_audit.json").write_text(json.dumps(anatomy_audit, indent=2))
        ensure_scaling_anatomy_audit_passes(anatomy_audit, plan.regions)  # union==full_model, pairwise disjoint, no empty region
        print(f"Confirmed: live 32B anatomy audit passed (union==full_model, pairwise disjoint, rank consensus) for {plan.regions}.")
        for region, info in anatomy_audit["regions"].items():
            print(f"  {region}: n_tensors={info['n_tensors']} n_elements={info['n_elements']} pct_of_total={info['percentage_of_total_elements']:.3f}% mask_hash={info['mask_hash'][:12]}...")

        per_rank_region_info = collective_rpc_all_workers(
            engine, report_region_param_names_for_scaling, args=(plan.regions, MODEL_FAMILY), label="report_region_param_names_for_scaling", expected_world_size=tp_size,
        )
        mismatched_ranks = [r for r in per_rank_region_info if r != per_rank_region_info[0]]
        if mismatched_ranks:
            raise RuntimeError("Ranks disagree on report_region_param_names_for_scaling for 32B S2 regions -- this is purely name-based and rank-independent, so any disagreement indicates a real bug.")
        region_info = per_rank_region_info[0]
        region_param_names_by_region = {r: tuple(region_info[r]["param_names"]) for r in plan.regions}
        parameter_mask_hash_by_region = {r: anatomy_audit["regions"][r]["mask_hash"] for r in plan.regions}
        for region in plan.regions:
            if region_info[region]["mask_hash"] != parameter_mask_hash_by_region[region]:
                raise RuntimeError(f"Mask-hash mismatch between report_region_param_names_for_scaling and report_global_anatomy_audit for region {region!r}.")

        llm_adapter = RayEngineLLMAdapter(engine)
        baseline_path = plan.output_dir / "baseline_scores.json"
        load_or_compute_baseline_scores(baseline_path, capability_contexts, plan.model_revision, plan.run_signature, llm_adapter, tokenizer, sampling_params)

        print("Running baseline repeatability preflight (32B baselines are NOT expected to equal 3B/7B/32B-whole_model) before any Stage-11 32B S2 perturbation...")
        preflight_report = run_baseline_repeatability_preflight_rpc(
            engine, capability_contexts, tokenizer, sampling_params, run_benchmark=run_benchmark, generation_batch_size=plan.generation_batch_size,
            reset_fn=functools.partial(reset_to_base_weights_distributed_rpc, tensor_parallel_size=tp_size),
        )
        (plan.output_dir / "baseline_repeatability_preflight.json").write_text(json.dumps(preflight_report, indent=2))
        ensure_baseline_repeatability(preflight_report)
        print(f"Baseline repeatability preflight PASSED for all {len(preflight_report)} capabilities.")

        newly_written_rows = run_stage11_32b_rpc(
            plan, capability_contexts, engine, tokenizer, sampling_params, seed_bank,
            region_param_names_by_region, parameter_mask_hash_by_region, anatomy_audit, run_benchmark=run_benchmark,
            evaluate_one_candidate=functools.partial(evaluate_one_stage11_32b_candidate_distributed_rpc, tensor_parallel_size=tp_size),
        )
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    manifest = write_stage11_32b_run_manifest(plan.output_dir)
    print(f"Wrote {newly_written_rows} NEW result rows this run to {plan.output_dir / 'results.jsonl'}")
    print(f"Run manifest: {json.dumps(manifest, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
