"""Stage 6: 3B Global Visual-Thicket Pilot -- the FIRST REAL PAPER EXPERIMENT.

Answers RQ1 ("do pretrained VLMs contain dense and diverse nearby experts across distinct
visual capabilities?") and produces the raw material for Figure 2 (performance-density
distributions, solution-density curves, radius dependence, best-of-N behavior) and a first
look at Figure 4's Spectral Discordance -- NOT the anatomy experiment (that's Stage 7+): every
perturbation here is a `global_gaussian_upstream` perturbation (spec section C1 of
VISUAL_THICKET_EXPERIMENT_SPEC.md), never `anatomical_relative_l2`.

UPSTREAM RECONCILIATION (Stage 6): the pinned upstream commit
(external/setup_external_repo.py, `536df0a308f3990b6270c991fbb96bd0b779a58e`) was cloned and
read directly this stage (never vendored -- see external/EXTERNAL_COMMIT.txt's existing
"read, described, and invoked externally, never transcribed" discipline):
  - `randopt.py`'s CLI default `--sigma_values 0.0001,0.0005,0.001,0.002,0.005,0.01` is the
    EXACT upstream sigma grid (UPSTREAM_SIGMA_GRID below) -- already recorded as
    `sigma_default` in REPRO_SPEC.md's "Sigma -- resolution plan" section, now directly
    re-confirmed against the actual pinned source rather than only the file's own memory.
  - `utils/worker_extn.py`'s `WorkerExtension.perturb_self_weights`/`restore_self_weights`
    confirm `perturb_cpu.py`'s existing reimplementation exactly: a fresh
    `torch.Generator(device=p.device).manual_seed(int(seed))` per named parameter, every
    parameter visited but only ones NOT prefixed `visual.`/`model.visual.` actually modified
    (`_should_perturb`, unless `PERTURB_VISUAL=1`), restore via regenerating the identical
    noise and subtracting rather than a stored-copy restore. `thicket.perturbation`'s
    `global_gaussian_upstream` mode already wraps `perturb_cpu.perturb`/`restore` unchanged
    (a CPU-testable reference implementation of the exact same math) -- nothing about THAT
    needed fixing.
  - `randopt.py:run_sampling` draws `population_size` UNIQUE seeds without replacement, then
    draws ONE sigma per candidate independently WITH replacement from the sigma list via
    `rng.choice(sigma_list, size=population_size)` -- i.e. upstream does NOT evaluate a fixed
    count per sigma bucket. This pilot deliberately DEPARTS from that one detail (evaluating a
    fixed `perturbations_per_sigma` count per sigma, per this stage's own task spec) to get a
    clean per-sigma breakdown for Figure 2's radius-dependence panel; every other mechanic
    (the sigma VALUES themselves, the per-tensor-reseed noise, the perturb-then-restore
    lifecycle) is unchanged and reused verbatim.
  - Spectral Discordance: grepped the full pinned checkout (`spectral|discordance|spearman`,
    case-insensitive) -- ZERO matches anywhere in upstream source. There is no upstream
    implementation to reconcile against. Stage 5's implementation (ported from the published
    paper's Definition 2.2) therefore remains the sole authoritative definition; nothing
    needed changing (see VISUAL_THICKET_EXPERIMENT_SPEC.md's updated H2 status).

RUNPOD SMOKE FAILURE + FIX (this repair pass, commit 630bc34 -> here): the first smoke crashed
with `AttributeError: 'LLMEngine' object has no attribute 'model_executor'` at
`llm.llm_engine.model_executor.driver_worker.model_runner.model` -- a FRONTEND/driver-process
attribute path that simply does not exist under vLLM V1 (confirmed against the pinned upstream
source: `utils/worker_extn.py`'s real model access, `self.model_runner.model`, only ever runs
INSIDE the worker process, reached exclusively via `LLM.collective_rpc(method, args)` --
`core/engine.py:launch_engines` sets `worker_extension_cls="utils.worker_extn.WorkerExtension"`
on every engine it creates and dispatches every weight-touching operation through
`engine.collective_rpc.remote(...)`, NEVER through a frontend attribute). ROOT CAUSE: the
original `main()` tried to reach the model directly from the driver process, instead of
dispatching through the collective_rpc mechanism this project's OWN existing GQA RandOpt
integration (`run_randopt_image_aware.py`, `run_scoped_randopt.py`) already uses and has
already validated on GPU. FIX (this pass): removed ALL frontend model access; every
weight-touching operation (perturb, restore, mask-hash/inventory, restoration verification) now
dispatches via `collective_rpc` -- perturb/restore reuse upstream's OWN
`perturb_self_weights`/`restore_self_weights` methods (string dispatch, unmodified upstream
code); mask-hash/inventory and restoration verification (which upstream does not provide) are
NEW plain Python functions in `thicket/worker_rpc.py`, dispatched as CALLABLES via
`collective_rpc(callable, args)` -- the IDENTICAL pattern already established by
`scoped_perturbation.scoped_apply_perturbation` before this stage. This needed no
`worker_extension_cls` subclass and zero changes to `external/RandOpt/core/engine.py`.

MODEL REVISION PINNING FIX (this repair pass): the RunPod log showed `revision=None` reaching
vLLM despite the pilot config declaring an exact pinned revision -- `main()` previously passed
`plan.model_name` straight to the engine launcher without ever resolving the pin. Fixed by
calling `vlm_adapter.resolve_model_snapshot(model_name, revision)` (the SAME function
`run_capability_benchmark_gate.run_one_capability` already uses for this exact purpose) BEFORE
constructing any engine, and passing the resulting immutable local snapshot PATH (not the bare
HF repo name) to `launch_engines`. All three of {model_name, requested_revision,
resolved_snapshot_path} are persisted to `model_resolution.json`.

EXECUTION MODEL (matches upstream's own `run_sampling` sequence, confirmed by reading it, and
this project's own already-validated Ray/collective_rpc integration): ONE vLLM engine (Ray
actor, TP=1) launched once for the whole pilot; per perturbation: perturb via collective_rpc
-> generate on every capability's fixed D_map subset, in a fixed order (via
`engine.generate.remote`, wrapped by `RayEngineLLMAdapter` so the existing, UNMODIFIED
`benchmarks/runner.run_benchmark` free function needs no changes at all) -> restore via
collective_rpc -> verify restoration via collective_rpc against a fingerprint captured once
before the sweep -> next perturbation. No per-perturbation or per-capability engine reload.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .benchmarks.runner import RunResult, run_benchmark
from .benchmarks.subset_selection import build_or_load_subset
from .thicket import diversity as thicket_diversity
from .thicket import metrics as thicket_metrics
from .thicket.data_roles import DataRolePartition, partition_data_roles, write_data_role_manifest
from .thicket.perturbation import (
    NUMPY_SEED_DOMAIN,
    PerturbationManifest,
    generate_perturbation_population,
    validate_unique_worker_seeds,
)
from .thicket.schema import ExperimentResultRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

# Confirmed directly against the pinned upstream commit this stage -- see module docstring.
UPSTREAM_SIGMA_GRID: Tuple[float, ...] = (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01)

# Exactly these three -- never add a fourth in this pilot (spec section 3).
PILOT_CAPABILITIES: Tuple[str, ...] = ("visual_grounding", "ocr_text_recognition_grounded", "spatial_reasoning")

CAPABILITY_CONFIG_FILES: Dict[str, str] = {
    "visual_grounding": "visual_grounding.yaml",
    "ocr_text_recognition_grounded": "ocr_text_recognition_grounded.yaml",
    "spatial_reasoning": "spatial_reasoning.yaml",
}

DEFAULT_PERTURBATIONS_PER_SIGMA = 64
DEFAULT_SUBSET_SIZE = 50
DEFAULT_RESTORATION_ATOL = 1e-4
DEFAULT_RESTORATION_RTOL = 1e-3

# Stage-6-specific engine construction (this repair pass): Qwen2.5-VL-3B-Instruct's own
# default max_model_len (128000) drove a real RunPod smoke KV-cache OOM (required 4.39 GiB,
# available 3.01 GiB at gpu_memory_utilization=0.75 on an L40S) -- this pilot only ever
# generates short, fixed-length responses (max_tokens=256), so 4096 (already sufficient for
# this project's prior Qwen2.5-VL benchmark runs) is set explicitly rather than raising
# gpu_memory_utilization to accommodate an irrelevant 128k context.
STAGE6_MAX_MODEL_LEN = 4096
STAGE6_GPU_MEMORY_UTILIZATION = 0.75

SOLUTION_DENSITY_MARGINS: Tuple[float, ...] = (0.0, 0.02, 0.05)

# The upstream WorkerExtension call pair this pilot dispatches for perturb/restore -- the
# "regenerate identical noise and subtract" convention (vs. the separate store_base_weights/
# apply_perturbation/reset_to_base_weights ensemble methods upstream also exposes), matching
# EXACTLY what `thicket.perturbation`'s `global_gaussian_upstream` mode already wraps for its
# CPU-testable reference implementation, and what `run_randopt_image_aware.py` already calls
# this "released_compat" (see its own RESTORATION_MODES) -- named here for documentation only,
# not a mode switch (this pilot only ever uses this one).
WORKER_PERTURB_METHOD = "perturb_self_weights"
WORKER_RESTORE_METHOD = "restore_self_weights"


class PilotConfigError(ValueError):
    """The loaded pilot YAML/overrides violate a Stage-6 scientific-integrity constraint
    (wrong capability set, an invented sigma grid, etc.) -- never silently accepted.
    """


class RestorationFailedError(RuntimeError):
    """A perturbation's restore step did not return model parameters to within tolerance of
    the stored base fingerprint -- the experiment must abort here (spec section 7), never
    silently continue with possibly-accumulated drift.
    """


class CollectiveRpcResultError(RuntimeError):
    """`collective_rpc` returned something other than the documented list-of-per-worker
    -results contract, or an unexpected number of results for this TP=1-only pilot -- never
    silently unwrapped/guessed.
    """


# =============================================================================================
# Pilot plan: pure arithmetic, no I/O, no GPU -- everything printed by --dry-run.
# =============================================================================================


@dataclass(frozen=True)
class PilotPlan:
    model_name: str
    model_revision: str
    model_family: str
    model_scale: str
    capabilities: Tuple[str, ...]
    sigma_grid: Tuple[float, ...]
    perturbations_per_sigma: int
    examples_per_capability: int
    output_dir: Path
    restoration_atol: float
    restoration_rtol: float

    @property
    def total_unique_perturbations(self) -> int:
        return len(self.sigma_grid) * self.perturbations_per_sigma

    @property
    def total_perturbation_capability_evaluations(self) -> int:
        return self.total_unique_perturbations * len(self.capabilities)

    @property
    def baseline_evaluations(self) -> int:
        return len(self.capabilities) * self.examples_per_capability

    @property
    def total_model_example_evaluations(self) -> int:
        return self.total_perturbation_capability_evaluations * self.examples_per_capability + self.baseline_evaluations


def load_pilot_config(path: "str | Path") -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_pilot_plan(
    raw_config: dict, *, perturbations_per_sigma: Optional[int] = None, subset_size: Optional[int] = None,
    output_dir: Optional[str] = None, restoration_atol: Optional[float] = None, restoration_rtol: Optional[float] = None,
) -> PilotPlan:
    """Validates the two Stage-6 scientific-integrity invariants that must never silently
    drift (spec sections 3 and 5): the capability set is EXACTLY `PILOT_CAPABILITIES`, and the
    sigma grid is EXACTLY `UPSTREAM_SIGMA_GRID` (order-independent) -- never a config typo
    substituting a fourth capability or an invented sigma value.
    """
    model = raw_config["model"]
    pilot = raw_config["pilot"]

    capabilities = tuple(pilot["capabilities"])
    if set(capabilities) != set(PILOT_CAPABILITIES):
        raise PilotConfigError(f"Stage-6 pilot capabilities must be exactly {PILOT_CAPABILITIES}, got {capabilities}")

    sigma_grid = tuple(float(s) for s in pilot["sigma_grid"])
    if set(sigma_grid) != set(UPSTREAM_SIGMA_GRID):
        raise PilotConfigError(f"Stage-6 pilot sigma grid must be exactly the upstream grid {UPSTREAM_SIGMA_GRID}, got {sigma_grid}")

    return PilotPlan(
        model_name=model["name"], model_revision=model["revision"], model_family=model.get("family", "qwen2_5_vl"),
        model_scale=model.get("scale", "3B"), capabilities=capabilities, sigma_grid=sigma_grid,
        perturbations_per_sigma=perturbations_per_sigma if perturbations_per_sigma is not None else pilot["perturbations_per_sigma"],
        examples_per_capability=subset_size if subset_size is not None else pilot["examples_per_capability"],
        output_dir=Path(output_dir) if output_dir is not None else REPO_ROOT / raw_config["outputs"]["root"],
        restoration_atol=restoration_atol if restoration_atol is not None else pilot.get("restoration_atol", DEFAULT_RESTORATION_ATOL),
        restoration_rtol=restoration_rtol if restoration_rtol is not None else pilot.get("restoration_rtol", DEFAULT_RESTORATION_RTOL),
    )


def format_pilot_plan(plan: PilotPlan) -> str:
    lines = [
        "=== Stage 6: 3B Global Visual-Thicket Pilot -- plan (printed before any GPU execution) ===",
        f"model_name: {plan.model_name}",
        f"model_revision: {plan.model_revision}",
        f"capabilities ({len(plan.capabilities)}): {list(plan.capabilities)}",
        f"sigma_grid ({len(plan.sigma_grid)}): {list(plan.sigma_grid)}",
        f"perturbations_per_sigma: {plan.perturbations_per_sigma}",
        f"total_unique_perturbations: {plan.total_unique_perturbations}",
        f"examples_per_capability (D_map): {plan.examples_per_capability}",
        f"total_perturbation_x_capability_evaluations: {plan.total_perturbation_capability_evaluations}",
        f"baseline_evaluations: {plan.baseline_evaluations}",
        f"total_model_example_evaluations: {plan.total_model_example_evaluations}",
        "expected_model_loading_strategy: ONE vLLM engine (Ray actor, TP=1, "
        "worker_extension_cls=utils.worker_extn.WorkerExtension, launched by OUR OWN "
        "launch_stage6_engine() -- max_model_len=4096, no store_base_weights call) loaded "
        "once for the whole pilot; per perturbation: perturb_self_weights via collective_rpc "
        "-> evaluate visual_grounding, ocr_text_recognition_grounded, spatial_reasoning "
        "D_map subsets in that fixed order (engine.generate.remote) -> restore_self_weights "
        "via collective_rpc -> verify_restoration_rpc via collective_rpc against a "
        "fingerprint captured once before the sweep -> next perturbation. No "
        "per-perturbation or per-capability engine reload. Zero frontend "
        "llm_engine.model_executor access anywhere in this path.",
        f"output_dir: {plan.output_dir}",
    ]
    return "\n".join(lines)


# =============================================================================================
# D_map construction (spec section 4) -- reuses existing subset_selection + data_roles as-is.
# =============================================================================================


@dataclass
class CapabilityContext:
    capability: str
    benchmark: Any
    examples: List[Any]
    partition: DataRolePartition
    subset_hash: str
    base_result: Optional[RunResult] = None
    base_score: Optional[float] = None


def build_d_map_context(benchmark: Any, cfg: Any, capability: str, n: int, seed: int, subset_ids_dir: "str | Path") -> CapabilityContext:
    """Builds (and persists) the fixed D_map subset for one capability. The SAME `subset_ids_
    dir`/`n`/`seed` always reproduce the identical subset (via `build_or_load_subset`'s own
    persistence -- a re-run replays the persisted IDs rather than re-shuffling), and the
    partition's `manifest_hash` (from `thicket.data_roles`) is what every result row's
    `subset_hash` field references.
    """
    all_examples = benchmark.load_examples(cfg)
    ids_path = Path(subset_ids_dir) / f"{capability}_d_map_{n}.json"
    subset = build_or_load_subset(all_examples, n, benchmark.subset_selection_rule(), seed, ids_path)
    ids = [e.example_id for e in subset]
    partition = partition_data_roles(ids, sizes={"map": len(ids)}, seed=seed)
    return CapabilityContext(capability=capability, benchmark=benchmark, examples=subset, partition=partition, subset_hash=partition.manifest_hash)


# =============================================================================================
# Worker-RPC transport (Task 1/2 of this repair pass) -- ZERO frontend model_executor access.
# =============================================================================================


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    """Same TP=1 list-unwrap validation this project already established
    (run_scoped_randopt.py/diagnostics/gate2_restoration_ab.py) -- duplicated rather than
    cross-imported from another top-level script, consistent with this project's convention.
    vLLM's collective_rpc returns a LIST of per-worker results even under TP=1; never indexed
    into as a bare value.
    """
    if not isinstance(results, list):
        raise CollectiveRpcResultError(
            f"collective_rpc({label!r}) returned {type(results).__name__}, expected vLLM's "
            f"own list-of-per-worker-results contract. Got: {results!r}"
        )
    if len(results) != 1:
        raise CollectiveRpcResultError(
            f"collective_rpc({label!r}) returned {len(results)} per-worker results; this "
            f"pilot is TP=1-only (launch_engines(..., tensor_parallel_size=1, ...)) and "
            f"expects exactly 1."
        )
    return results[0]


def _collective_rpc_single_worker(engine: Any, method: "str | Callable", args: tuple = (), *, label: str, ray_get: Optional[Callable] = None) -> Any:
    """Dispatches `method` (a STRING naming an existing worker-extension method, e.g. upstream's
    own `perturb_self_weights`, or a plain CALLABLE defined in our own package, e.g.
    `thicket.worker_rpc.verify_restoration_rpc` -- the exact same Callable-dispatch convention
    already established by `scoped_perturbation.scoped_apply_perturbation`) via
    `engine.collective_rpc.remote(method, args=args)`, then `ray.get`s and unwraps the
    single-worker result. `ray_get` is injectable purely for CPU testing (a fake engine +
    identity function) -- real callers never pass it, and get the real `ray.get`.
    """
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def perturb_via_rpc(engine: Any, seed: int, sigma: float, *, ray_get: Optional[Callable] = None) -> None:
    """Dispatches upstream's OWN, unmodified `perturb_self_weights(seed, sigma, negate=False)`
    -- string method dispatch, no reimplementation.
    """
    _collective_rpc_single_worker(engine, WORKER_PERTURB_METHOD, args=(seed, sigma, False), label=WORKER_PERTURB_METHOD, ray_get=ray_get)


def restore_via_rpc(engine: Any, seed: int, sigma: float, *, ray_get: Optional[Callable] = None) -> None:
    """Dispatches upstream's OWN, unmodified `restore_self_weights(seed, sigma, negate=False)`
    -- MUST reuse the identical (seed, sigma) as the matching perturb_via_rpc call.
    """
    _collective_rpc_single_worker(engine, WORKER_RESTORE_METHOD, args=(seed, sigma, False), label=WORKER_RESTORE_METHOD, ray_get=ray_get)


def compute_mask_info_via_rpc(engine: Any, *, ray_get: Optional[Callable] = None) -> Dict:
    """Dispatches `thicket.worker_rpc.compute_perturbable_mask_info_rpc` as a Callable (spec
    Task 2 item 3) -- computed and hashed entirely inside the worker.
    """
    from .thicket.worker_rpc import compute_perturbable_mask_info_rpc

    return _collective_rpc_single_worker(engine, compute_perturbable_mask_info_rpc, args=(), label="compute_perturbable_mask_info_rpc", ray_get=ray_get)


def compute_restoration_fingerprint_via_rpc(engine: Any, *, ray_get: Optional[Callable] = None) -> Dict[str, float]:
    """Dispatches `thicket.worker_rpc.compute_restoration_fingerprint_rpc` as a Callable --
    per-tensor L2 norms, computed inside the worker, no second model copy."""
    from .thicket.worker_rpc import compute_restoration_fingerprint_rpc

    return _collective_rpc_single_worker(engine, compute_restoration_fingerprint_rpc, args=(), label="compute_restoration_fingerprint_rpc", ray_get=ray_get)


def verify_restoration_via_rpc(engine: Any, base_fingerprint: Dict[str, float], atol: float, rtol: float, *, ray_get: Optional[Callable] = None) -> Dict:
    """Dispatches `thicket.worker_rpc.verify_restoration_rpc` as a Callable (spec Task 2 item
    4) -- checks the restoration invariant (see that module's docstring) entirely inside the
    worker and returns only a small diagnostic dict, never full tensors.
    """
    from .thicket.worker_rpc import verify_restoration_rpc

    return _collective_rpc_single_worker(engine, verify_restoration_rpc, args=(base_fingerprint, atol, rtol), label="verify_restoration_rpc", ray_get=ray_get)


class RayEngineLLMAdapter:
    """Thin synchronous adapter so the UNMODIFIED `benchmarks.runner.run_benchmark` (which
    calls `llm.generate(requests, sampling_params, use_tqdm=...)` and expects the return value
    directly, never a Ray ObjectRef) can be pointed at a Ray-actor-wrapped vLLM engine -- the
    only way collective_rpc-manipulable weights are reachable under vLLM V1 (see module
    docstring). `run_benchmark` itself needed ZERO changes for this repair pass.
    """

    def __init__(self, engine: Any, ray_get: Optional[Callable] = None):
        self._engine = engine
        if ray_get is None:
            import ray
            ray_get = ray.get
        self._ray_get = ray_get

    def generate(self, requests, sampling_params, use_tqdm: bool = True):
        return self._ray_get(self._engine.generate.remote(requests, sampling_params, use_tqdm=use_tqdm))


# =============================================================================================
# Per-perturbation evaluation lifecycle (spec sections 5-7) -- RPC-only, no frontend model access.
# =============================================================================================


def evaluate_one_perturbation_rpc(
    engine: Any, manifest: PerturbationManifest, capability_contexts: Dict[str, CapabilityContext],
    tokenizer: Any, sampling_params: Any, base_fingerprint: Dict[str, float], restoration_atol: float, restoration_rtol: float,
    *, ray_get: Optional[Callable] = None,
) -> List[ExperimentResultRecord]:
    """Applies `manifest`'s perturbation ONCE (via collective_rpc), evaluates every
    capability's D_map subset (in `capability_contexts`'s own iteration order -- callers must
    pass an order-preserving dict built in the required visual_grounding -> OCR ->
    spatial_reasoning order), restores (via collective_rpc), then verifies restoration (via
    collective_rpc) -- raising RestorationFailedError aborts the whole experiment rather than
    silently proceeding to the next perturbation with possibly-drifted base weights.
    """
    llm_adapter = RayEngineLLMAdapter(engine, ray_get=ray_get)
    perturb_via_rpc(engine, manifest.seed, manifest.sigma, ray_get=ray_get)
    records: List[ExperimentResultRecord] = []
    try:
        for capability, ctx in capability_contexts.items():
            result = run_benchmark(ctx.benchmark, ctx.examples, llm_adapter, tokenizer, sampling_params)
            perturbed_score = result.aggregate_metrics["primary_metric"]
            base_score = ctx.base_score
            records.append(ExperimentResultRecord(
                experiment_id="visual_thicket_global_3b_pilot", perturbation_id=manifest.perturbation_id,
                model_family=manifest.model_family, model_scale=manifest.model_scale, model_revision=manifest.model_revision,
                perturbation_mode=manifest.perturbation_mode, anatomy_region=manifest.anatomy_region,
                radius=manifest.radius, sigma=manifest.sigma, seed=manifest.seed, parameter_mask_hash=manifest.parameter_mask_hash,
                capability=capability, dataset_role="map", subset_hash=ctx.subset_hash,
                base_score=base_score, perturbed_score=perturbed_score, delta=perturbed_score - base_score,
                parser_failure_rate=result.aggregate_metrics.get("parser_failure_rate"),
                per_example_result_path=None, per_example_result_hash=result.generation_hash(),
                runtime_metadata={},
            ))
    finally:
        restore_via_rpc(engine, manifest.seed, manifest.sigma, ray_get=ray_get)

    verification = verify_restoration_via_rpc(engine, base_fingerprint, restoration_atol, restoration_rtol, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Restoration check failed after perturbation {manifest.perturbation_id!r} "
            f"(seed={manifest.seed}, sigma={manifest.sigma}): {verification['n_failing']}/"
            f"{verification['n_checked']} tensor(s) exceeded tolerance "
            f"(atol={restoration_atol}, rtol={restoration_rtol}); worst offenders: "
            f"{verification['worst_offenders']}"
        )
    return records


def build_stage6_perturbation_population(plan: PilotPlan, base_seed: int, parameter_mask_hash: str) -> Tuple[PerturbationManifest, ...]:
    """Builds the FULL Stage-6 population across every sigma bucket (spec section 5: a fixed
    `perturbations_per_sigma` count per sigma, `global_gaussian_upstream` mode only).

    SEED-DOMAIN FIX (this repair pass): the pinned upstream WorkerExtension._set_seed calls
    `np.random.seed(seed)`, which hard-requires `0 <= seed <= 2**32 - 1` -- `thicket.seeds.
    derive_seed`'s own 63-bit output falls outside that domain, which is exactly what crashed
    a real RunPod smoke at the first `perturb_self_weights(seed, sigma)` call. Every manifest's
    worker seed is therefore reduced into that domain (`seed % NUMPY_SEED_DOMAIN`) AT
    GENERATION TIME, via `generate_perturbation_population`'s `seed_modulus` parameter -- the
    value stored on `PerturbationManifest.seed` (and used to compute `perturbation_id`) IS the
    exact uint32 value later passed to `perturb_self_weights`/`restore_self_weights`, never a
    larger value silently truncated only at RPC time.

    The ENTIRE combined population (every sigma bucket together, not just one at a time) is
    then validated for worker-seed uniqueness (`validate_unique_worker_seeds`) BEFORE any
    perturbation is ever applied -- a collision anywhere in the pilot hard-fails
    (DuplicateWorkerSeedError) rather than being silently resolved in a way that would change
    reproducibility.
    """
    all_manifests: List[PerturbationManifest] = []
    for sigma in plan.sigma_grid:
        population = generate_perturbation_population(
            mode="global_gaussian_upstream", n=plan.perturbations_per_sigma, base_seed=base_seed,
            model_family=plan.model_family, model_scale=plan.model_scale, model_revision=plan.model_revision,
            parameter_mask_hash=parameter_mask_hash, anatomy_region=None, radius=None, sigma=sigma,
            seed_modulus=NUMPY_SEED_DOMAIN,
        )
        all_manifests.extend(population)
    validate_unique_worker_seeds(all_manifests)
    return tuple(all_manifests)


def run_pilot_rpc(
    plan: PilotPlan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, base_seed: int, parameter_mask_hash: str, *, ray_get: Optional[Callable] = None,
) -> List[ExperimentResultRecord]:
    """Loops perturbation -> capability, exactly the nesting spec section 6 requires (sigma is
    the outer grouping `build_stage6_perturbation_population` adds on top, for a clean
    per-sigma bucket count -- see module docstring's upstream-reconciliation note). The
    restoration fingerprint is captured ONCE via RPC before the whole sweep.
    """
    base_fingerprint = compute_restoration_fingerprint_via_rpc(engine, ray_get=ray_get)
    population = build_stage6_perturbation_population(plan, base_seed, parameter_mask_hash)
    all_records: List[ExperimentResultRecord] = []
    for manifest in population:
        all_records.extend(evaluate_one_perturbation_rpc(
            engine, manifest, capability_contexts, tokenizer, sampling_params,
            base_fingerprint, plan.restoration_atol, plan.restoration_rtol, ray_get=ray_get,
        ))
    return all_records


# =============================================================================================
# Model-revision resolution (Task 4) + runtime compatibility diagnostic (Task 5)
# =============================================================================================


def resolve_and_report_model_snapshot(model_name: str, revision: str) -> Dict[str, str]:
    """Resolves `model_name@revision` to an immutable local HF snapshot path BEFORE
    constructing vLLM -- the SAME `vlm_adapter.resolve_model_snapshot` function
    `run_capability_benchmark_gate.run_one_capability` already uses for this exact purpose
    (never relies on the HF repo's current HEAD, unlike passing the bare repo name straight to
    vLLM, which is what produced `revision=None` in the RunPod log this pass fixes). Returns
    all three of {model_name, requested_revision, resolved_snapshot_path} for persistence.
    """
    from .vlm_adapter import resolve_model_snapshot

    resolved_snapshot_path = resolve_model_snapshot(model_name, revision)
    return {"model_name": model_name, "requested_revision": revision, "resolved_snapshot_path": resolved_snapshot_path}


def get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return "not installed"


def detect_vllm_engine_mode() -> str:
    """Best-effort V1/V0 report: vLLM >=0.8 defaults to the V1 engine unless VLLM_USE_V1=0 is
    explicitly set -- reports the env var directly (the actual signal this project's own
    RunPod failure hinged on) rather than guessing at internal engine class names that vary
    across versions.
    """
    import os

    if os.environ.get("VLLM_USE_V1") == "0":
        return "V0 (VLLM_USE_V1=0 explicitly set)"
    return "V1 (default; VLLM_USE_V1 not set to 0)"


def build_stage6_engine_config() -> Dict[str, Any]:
    """The exact Stage-6-specific engine construction parameters (this repair pass) --
    persisted and printed as runtime metadata so a real run's actual max_model_len/
    gpu_memory_utilization is always auditable, not merely assumed from this module's source.
    """
    return {
        "max_model_len": STAGE6_MAX_MODEL_LEN,
        "gpu_memory_utilization": STAGE6_GPU_MEMORY_UTILIZATION,
        "tensor_parallel_size": 1,
        "precision": "bfloat16",
        "store_base_weights_called": False,
    }


def format_runtime_compatibility_diagnostic(model_resolution: Dict[str, str], worker_extension_cls: str, vllm_version: str, engine_mode: str, engine_config: Dict[str, Any]) -> str:
    lines = [
        "=== Stage 6: runtime compatibility diagnostic (printed before execution) ===",
        f"vllm_version: {vllm_version}",
        f"engine_mode: {engine_mode}",
        f"worker_extension_cls: {worker_extension_cls}",
        f"model_name: {model_resolution['model_name']}",
        f"requested_revision: {model_resolution['requested_revision']}",
        f"resolved_snapshot_path: {model_resolution['resolved_snapshot_path']}",
        f"max_model_len: {engine_config['max_model_len']}",
        f"gpu_memory_utilization: {engine_config['gpu_memory_utilization']}",
        f"store_base_weights_called: {engine_config['store_base_weights_called']}",
    ]
    return "\n".join(lines)


# =============================================================================================
# Figure-2 metrics + diversity (spec sections 9-10)
# =============================================================================================


def compute_figure2_summary(records: List[ExperimentResultRecord]) -> Dict[str, Dict[str, Dict]]:
    """Groups records by (capability, sigma) and computes every Figure-2 statistic. These are
    REPORTING margins (0.00, 0.02, 0.05), never expert-confirmation thresholds -- no top-expert
    selection happens anywhere in this module.
    """
    groups: Dict[Tuple[str, float], List[float]] = {}
    for r in records:
        groups.setdefault((r.capability, r.sigma), []).append(r.delta)

    summary: Dict[str, Dict[str, Dict]] = {}
    for (capability, sigma), deltas in groups.items():
        mean, std = thicket_metrics.mean_std(deltas)
        q = thicket_metrics.quantiles(deltas, qs=(0.5, 0.75, 0.9, 0.95, 0.99))
        summary.setdefault(capability, {})[str(sigma)] = {
            "sigma": sigma, "n": len(deltas), "mean": mean, "std": std,
            "median": q[0.5], "q75": q[0.75], "q90": q[0.9], "q95": q[0.95], "q99": q[0.99],
            "probability_of_improvement": thicket_metrics.probability_of_improvement(deltas),
            "probability_of_degradation": thicket_metrics.probability_of_degradation(deltas),
            "solution_density": thicket_metrics.solution_density(deltas, margins=SOLUTION_DENSITY_MARGINS),
            "positive_thicket_mass": thicket_metrics.positive_thicket_mass(deltas),
            "best_of_n_expected": thicket_metrics.best_of_n_expected(deltas).tolist(),
        }
    return summary


def build_delta_matrix(records: List[ExperimentResultRecord]) -> Tuple[Tuple[str, ...], Tuple[str, ...], np.ndarray]:
    """Pools EVERY sigma together (each (seed, sigma) pair is its own distinct perturbation_id
    across the whole grid) -- this is the perturbation x capability matrix diversity metrics
    operate on. Raises if any (perturbation, capability) cell is missing -- every perturbation
    must have been evaluated on every capability (spec section 5's core requirement).
    """
    perturbation_ids = tuple(sorted({r.perturbation_id for r in records}))
    capabilities = tuple(sorted({r.capability for r in records}))
    row_index = {pid: i for i, pid in enumerate(perturbation_ids)}
    col_index = {c: j for j, c in enumerate(capabilities)}

    matrix = np.full((len(perturbation_ids), len(capabilities)), np.nan)
    for r in records:
        matrix[row_index[r.perturbation_id], col_index[r.capability]] = r.delta

    if np.isnan(matrix).any():
        n_missing = int(np.isnan(matrix).sum())
        raise ValueError(f"Delta matrix has {n_missing} missing (perturbation, capability) entries -- every perturbation must be evaluated on every capability.")
    return perturbation_ids, capabilities, matrix


def compute_diversity_summary(records: List[ExperimentResultRecord]) -> Dict[str, Any]:
    perturbation_ids, capabilities, matrix = build_delta_matrix(records)
    return {
        "perturbation_ids": list(perturbation_ids),
        "capabilities": list(capabilities),
        "task_rank_correlation_matrix": thicket_diversity.task_rank_correlation_matrix(matrix).tolist(),
        "spectral_discordance": thicket_diversity.spectral_discordance(matrix),
    }


# =============================================================================================
# Stage-6-specific engine launcher (this repair pass) -- NOT external/RandOpt/core/engine.py's
# launch_engines(), which upstream unconditionally follows with collective_rpc(
# "store_base_weights") and has no max_model_len parameter at all.
# =============================================================================================


def launch_stage6_engine(
    model_path: str, *, precision: str = "bfloat16", gpu_memory_utilization: float = STAGE6_GPU_MEMORY_UTILIZATION,
    max_model_len: int = STAGE6_MAX_MODEL_LEN, tensor_parallel_size: int = 1,
) -> Tuple[list, list]:
    """Stage-6-specific single-engine Ray/vLLM launcher -- an INDEPENDENT, from-scratch
    function in OUR OWN package (external/RandOpt is not modified or subclassed on disk in
    any way); reuses upstream's `RandOptNcclLLM` class directly (imported, never copied) and
    the identical `worker_extension_cls="utils.worker_extn.WorkerExtension"` string, so every
    existing collective_rpc perturb/restore/mask/verify call in this module keeps working
    completely unchanged.

    Deliberately does NOT call `external/RandOpt/core/engine.py:launch_engines()`, for two
    reasons, neither a change to upstream's actual RandOpt search semantics:
      1. `launch_engines()` accepts no `max_model_len` -- Qwen2.5-VL-3B-Instruct's own default
         (128000) drove the real RunPod smoke's KV-cache OOM (required 4.39 GiB, available
         3.01 GiB at gpu_memory_utilization=0.75 on an L40S). `max_model_len=4096` (already
         sufficient for this project's prior Qwen2.5-VL benchmark runs) is passed explicitly
         instead of raising gpu_memory_utilization to accommodate an irrelevant 128k context.
      2. `launch_engines()` unconditionally calls `collective_rpc("store_base_weights")` after
         creating every engine -- upstream's own "Ensemble methods" (store_base_weights/
         apply_perturbation/reset_to_base_weights) clone every model parameter a SECOND time
         onto the GPU. Stage 6's lifecycle (perturb_self_weights -> evaluate 3 capabilities ->
         restore_self_weights -> verify_restoration_rpc) never reads `self._base_weights` --
         only the ensemble methods this pilot never calls do -- so that second ~3B-parameter
         GPU copy is pure waste here, and this function never issues that RPC call.

    Mirrors the REST of `launch_engines()`'s single-engine (TP=1) setup: one GPU-only
    placement group, `RandOptNcclLLM` as a Ray actor with `distributed_executor_backend=
    "ray"`, `enforce_eager=True`, `limit_mm_per_prompt={"image": 1}`. Returns `([engine], [pg])`
    -- the identical list-shaped return `launch_engines()` gives -- so upstream's own
    unmodified `cleanup_engines([engine], [pg])` still works for teardown, and the rest of
    this module needs no further changes.
    """
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if str(EXTERNAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import RandOptNcclLLM  # type: ignore  # upstream, unmodified -- reused, never copied

    pg_bundles = [{"GPU": 1, "CPU": 0} for _ in range(tensor_parallel_size)]
    pg = placement_group(pg_bundles, lifetime="detached")
    ray.get(pg.ready(), timeout=120)

    strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0,
    )
    engine_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="ray",
        worker_extension_cls="utils.worker_extn.WorkerExtension",
        dtype=precision,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 1},
    )
    engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(RandOptNcclLLM).remote(**engine_kwargs)

    # Deliberately NOT calling collective_rpc("store_base_weights") here -- see docstring above.
    return [engine], [pg]


# =============================================================================================
# CLI
# =============================================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "visual_thicket_global_3b_pilot.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--perturbations-per-sigma", type=int, default=None, help="smoke override, e.g. 2 -- does not modify the paper pilot config")
    parser.add_argument("--subset-size", type=int, default=None, help="smoke override, e.g. 5 -- does not modify the paper pilot config")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU")
    args = parser.parse_args(argv)

    raw_config = load_pilot_config(args.config)
    plan = build_pilot_plan(
        raw_config, perturbations_per_sigma=args.perturbations_per_sigma, subset_size=args.subset_size, output_dir=args.output_dir,
    )
    print(format_pilot_plan(plan))

    if args.dry_run:
        return 0

    # --- Real GPU execution path: lazy-imports vllm/ray/transformers, not exercised by CPU
    # tests, not run in this Stage-6 repair-pass preparation session (see module docstring).
    model_resolution = resolve_and_report_model_snapshot(plan.model_name, plan.model_revision)
    engine_config = build_stage6_engine_config()
    print(format_runtime_compatibility_diagnostic(
        model_resolution, worker_extension_cls="utils.worker_extn.WorkerExtension",
        vllm_version=get_vllm_version(), engine_mode=detect_vllm_engine_mode(), engine_config=engine_config,
    ))

    from .config import load_capability_benchmark_config
    from .run_capability_benchmark_gate import load_adapter
    from .vlm_adapter import bootstrap_ray, verify_workers_can_import_external_root

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "model_resolution.json").write_text(json.dumps(model_resolution, indent=2))
    (plan.output_dir / "engine_config.json").write_text(json.dumps(engine_config, indent=2))
    subset_ids_dir = plan.output_dir / "d_map_subsets"
    base_seed = raw_config["pilot"].get("base_seed", 20260823)

    capability_contexts: Dict[str, CapabilityContext] = {}
    for capability in PILOT_CAPABILITIES:  # fixed order, matches the required execution order
        cfg_path = REPO_ROOT / "configs" / "benchmarks" / CAPABILITY_CONFIG_FILES[capability]
        cfg = load_capability_benchmark_config(cfg_path)
        benchmark = load_adapter(cfg.dataset.adapter)
        ctx = build_d_map_context(benchmark, cfg, capability, plan.examples_per_capability, base_seed, subset_ids_dir)
        write_data_role_manifest(ctx.partition, plan.output_dir / "data_roles" / f"{capability}_d_map.json")
        capability_contexts[capability] = ctx

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_resolution["resolved_snapshot_path"])

    if str(EXTERNAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore  # upstream, unmodified -- teardown only; launch uses OUR OWN launch_stage6_engine (see its docstring for why)

    bootstrap_ray(EXTERNAL_ROOT)

    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)
        engines, pgs = launch_stage6_engine(
            model_resolution["resolved_snapshot_path"], precision=engine_config["precision"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"], max_model_len=engine_config["max_model_len"],
            tensor_parallel_size=engine_config["tensor_parallel_size"],
        )
        engine = engines[0]

        parameter_mask_hash = compute_mask_info_via_rpc(engine)["mask_hash"]

        llm_adapter = RayEngineLLMAdapter(engine)
        for capability, ctx in capability_contexts.items():
            base_result = run_benchmark(ctx.benchmark, ctx.examples, llm_adapter, tokenizer, sampling_params)
            ctx.base_result = base_result
            ctx.base_score = base_result.aggregate_metrics["primary_metric"]

        records = run_pilot_rpc(plan, capability_contexts, engine, tokenizer, sampling_params, base_seed, parameter_mask_hash)
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)

    results_path = plan.output_dir / "results.jsonl"
    with results_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    figure2_summary = compute_figure2_summary(records)
    (plan.output_dir / "figure2_summary.json").write_text(json.dumps(figure2_summary, indent=2))

    diversity_summary = compute_diversity_summary(records)
    (plan.output_dir / "diversity_summary.json").write_text(json.dumps(diversity_summary, indent=2))

    print(f"Wrote {len(records)} result rows to {results_path}")
    print(f"Wrote {plan.output_dir}/figure2_summary.json and {plan.output_dir}/diversity_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
