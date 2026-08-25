"""Stage 6: 3B Global Visual-Thicket Pilot -- the FIRST REAL PAPER EXPERIMENT.

Answers RQ1 ("do pretrained VLMs contain dense and diverse nearby experts across distinct
visual capabilities?") and produces the raw material for Figure 2 (performance-density
distributions, solution-density curves, radius dependence, best-of-N behavior) and a first
look at Figure 4's Spectral Discordance -- NOT the anatomy experiment (that's Stage 7+): every
perturbation here is a `global_gaussian_upstream` PERTURBATION (spec section C1 of
VISUAL_THICKET_EXPERIMENT_SPEC.md), never `anatomical_relative_l2`.

    perturbation_semantics = global_gaussian_upstream   (unchanged since Stage 6 began)
    restoration_mode       = fixed_base                 (THIS repair pass -- see below)

These are two SEPARATE things and must never be described as one "identical to released
RandOpt" lifecycle: the Gaussian perturbation itself (`perturb_self_weights`, upstream's own
per-tensor-reseed math) is unchanged; only HOW the model is returned to theta_0 between
candidates has changed.

FIXED-BASE RESTORATION FIX (this repair pass -- root cause + fix): a real 384-candidate RunPod
run reached sigma=0.01 and aborted with `RestorationFailedError` at perturbation_id
`5a417b7937eca5ad522e9c6b` (seed=1480723517, sigma=0.01) -- 1/254 perturbable tensors exceeded
tolerance, worst offender `language_model.model.layers.5.self_attn.o_proj.weight` with a norm
discrepancy of 0.047271728515625. Diagnosis: the restoration mechanism at that time was
upstream's `perturb_self_weights`/`restore_self_weights` pair -- native-BF16 add-then-
regenerate-and-subtract -- which is NOT exactly invertible after BF16 rounding; over a
384-candidate sweep this can (and, empirically, did) accumulate past any reasonable tolerance.
This exact problem was ALREADY solved earlier in this repository for scoped RandOpt
(`scoped_perturbation.py`, `diagnostics/scope_isolation_gpu_check.py`) via upstream's OWN
`store_base_weights()`/`reset_to_base_weights()` pair -- a direct GPU-resident tensor COPY from
a frozen snapshot, never an add-then-subtract. Stage 6 now reuses that exact existing design
(no new restoration mechanism invented): `store_base_weights()` is called ONCE, immediately
after the engine initializes; every candidate does `reset_to_base_weights()` ->
`perturb_self_weights(seed, sigma)` -> evaluate -> `reset_to_base_weights()` -> verify EXACT
restoration. `restore_self_weights` is NEVER called by Stage 6 anymore (it remains available,
unmodified, for historical released-RandOpt reproduction elsewhere in this project, e.g.
`run_randopt_image_aware.py`'s `released_compat` restoration mode).

Because `reset_to_base_weights` is a direct copy (not an inverse of an additive operation),
restoration can and must now be held to an EXACT standard -- see `thicket/worker_rpc.py`'s
`verify_exact_fixed_base_restoration_rpc`, which reuses `diagnostics/perturb_restore_drift.py`
's already-unit-tested `measure_drift` (the SAME exact-equality check already established and
GPU-validated by `diagnostics/scope_isolation_gpu_check.py`'s Test A-G).

This restoration-mode change does NOT touch the perturbation distribution, the 384 uint32
seeds, the perturbation IDs, the sigma grid, the per-sigma population count, or the capability
order -- `PerturbationManifest` describes epsilon_i and is completely independent of how the
model is returned to theta_0 between candidates.

UPSTREAM RECONCILIATION (Stage 6, earlier repair passes): the pinned upstream commit
(external/setup_external_repo.py, `536df0a308f3990b6270c991fbb96bd0b779a58e`) was cloned and
read directly (never vendored -- see external/EXTERNAL_COMMIT.txt's existing "read, described,
and invoked externally, never transcribed" discipline):
  - `randopt.py`'s CLI default `--sigma_values 0.0001,0.0005,0.001,0.002,0.005,0.01` is the
    EXACT upstream sigma grid (UPSTREAM_SIGMA_GRID below).
  - `utils/worker_extn.py`'s `perturb_self_weights`/`store_base_weights`/`reset_to_base_
    weights` are used unmodified, string-dispatched via `collective_rpc`.
  - Spectral Discordance: zero matches anywhere in upstream source -- Stage 5's paper-
    Definition-2.2 port remains the sole authoritative definition.

RUNPOD FIXES BAKED IN FROM EARLIER REPAIR PASSES (still true, not re-explained at length here):
  - Zero frontend `llm_engine.model_executor` access anywhere -- every weight-touching
    operation dispatches via `collective_rpc`.
  - Model revision is resolved to an immutable local HF snapshot path
    (`vlm_adapter.resolve_model_snapshot`) BEFORE constructing any engine.
  - `max_model_len=4096` (Qwen2.5-VL-3B-Instruct's own 128000 default caused a real KV-cache
    OOM on an L40S at gpu_memory_utilization=0.75); `gpu_memory_utilization` is now 0.60 (this
    repair pass -- `store_base_weights` creates a second GPU-resident weight copy, needing more
    headroom than the perturb/restore-only design this value was originally tuned for).
  - Every derived perturbation seed is folded into numpy's `[0, 2**32)` domain
    (`NUMPY_SEED_DOMAIN`) at population-generation time, with a hard uniqueness check across
    the full population.

STALE-OUTPUT SAFETY + CHECKPOINT/RESUME (this repair pass): a smoke run and the real full run
previously shared an ambiguous output identity (a 12-perturbation smoke's `results.jsonl`
could be mistaken for the 1152-row full run's output after the full run crashed mid-sweep).
Fixed via a deterministic `run_signature` ("full" iff both `perturbations_per_sigma` and
`examples_per_capability` exactly match the paper config; otherwise `smoke_p{P}_n{N}") that is
ALWAYS appended to `output_dir`, so a full run and any override run can never collide on disk.
Every run also gets a `checkpoint_manifest.json` recording its full identity (restoration_mode,
perturbation_semantics, model_revision, subset hashes, expected counts) -- an existing,
INCOMPATIBLE checkpoint (e.g. one written under the old subtractive-restoration design) hard
-fails rather than silently resuming. Candidate rows are appended to `results.jsonl` ONLY after
that candidate's full apply -> evaluate -> reset -> verify cycle has already succeeded, so an
interrupted run can safely resume from exactly where it left off -- never re-doing completed
GPU work, never fabricating results for a candidate that never finished. `write_paper_summary`
refuses to generate `figure2_summary.json`/`diversity_summary.json` for a run whose actual
perturbation/row counts fall short of the checkpoint's own recorded expectations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from .vlm_adapter import ensure_full_encoder_cache_reset_exposed, reset_vllm_encoder_cache_full

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

# Confirmed directly against the pinned upstream commit -- see module docstring.
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

# Stage-6-specific engine construction. max_model_len: Qwen2.5-VL-3B-Instruct's own default
# (128000) drove a real RunPod KV-cache OOM (required 4.39 GiB, available 3.01 GiB at
# gpu_memory_utilization=0.75 on an L40S) -- this pilot only ever generates short, fixed-length
# responses (max_tokens=256), so 4096 (already sufficient for this project's prior Qwen2.5-VL
# benchmark runs) is set explicitly rather than raising gpu_memory_utilization to accommodate
# an irrelevant 128k context. gpu_memory_utilization: lowered from 0.75 to 0.60 (this repair
# pass) -- store_base_weights() now creates a second GPU-resident copy of every perturbable
# parameter (the frozen theta_0 snapshot), so the engine needs more headroom than the original
# perturb/restore-only (no second copy) design assumed. max_model_len is NOT changed.
STAGE6_MAX_MODEL_LEN = 4096
STAGE6_GPU_MEMORY_UTILIZATION = 0.60

SOLUTION_DENSITY_MARGINS: Tuple[float, ...] = (0.0, 0.02, 0.05)

# Frozen scientific-interpretation labels (this repair pass) -- persisted/printed everywhere a
# run's identity matters (engine config, checkpoint manifest, run manifest) so the lifecycle is
# never silently described as identical to released RandOpt.
PERTURBATION_SEMANTICS = "global_gaussian_upstream"
RESTORATION_MODE = "fixed_base"
BASE_SNAPSHOT_MODE = "store_base_weights"

# Upstream WorkerExtension method names this pilot dispatches via collective_rpc.
WORKER_PERTURB_METHOD = "perturb_self_weights"
WORKER_STORE_BASE_METHOD = "store_base_weights"
WORKER_RESET_TO_BASE_METHOD = "reset_to_base_weights"
# Historical released-compatible restoration -- NEVER called by Stage 6's fixed_base lifecycle
# (see module docstring); kept only so the underlying upstream call shape stays documented and
# testable in isolation, exactly as `run_randopt_image_aware.py`'s own `released_compat` mode
# already uses it elsewhere in this project.
WORKER_RESTORE_METHOD = "restore_self_weights"


class PilotConfigError(ValueError):
    """The loaded pilot YAML/overrides violate a Stage-6 scientific-integrity constraint
    (wrong capability set, an invented sigma grid, etc.) -- never silently accepted.
    """


class RestorationFailedError(RuntimeError):
    """A perturbation's fixed-base reset did not return every perturbable parameter to an
    EXACT match with the frozen base snapshot -- the experiment must abort here (spec section
    4), never silently continue with possibly-accumulated drift.
    """


class CollectiveRpcResultError(RuntimeError):
    """`collective_rpc` returned something other than the documented list-of-per-worker
    -results contract, or an unexpected number of results for this TP=1-only pilot -- never
    silently unwrapped/guessed.
    """


class IncompatibleCheckpointError(RuntimeError):
    """An existing checkpoint_manifest.json / baseline_scores.json in this output directory
    does not match the current run's identity (restoration_mode, perturbation_semantics,
    model_revision, subset hashes, run_signature, or expected counts) -- e.g. it was written
    by a run using the old subtractive restoration. Hard-fails rather than silently resuming
    or partially trusting incompatible prior results.
    """


class IncompleteRunError(RuntimeError):
    """The run's own accounting shows fewer actual unique perturbations or result rows than
    the checkpoint's own recorded expectations -- refuses to (re)generate
    figure2_summary.json/diversity_summary.json (spec section 6's stale-output-safety fix): a
    partial run's output must never be mistaken for the finished paper summary.
    """


# =============================================================================================
# Pilot plan: pure arithmetic, no I/O, no GPU -- everything printed by --dry-run.
# =============================================================================================


def compute_run_signature(
    perturbations_per_sigma: int, examples_per_capability: int,
    paper_perturbations_per_sigma: int, paper_examples_per_capability: int,
) -> str:
    """"full" iff BOTH values exactly match the paper pilot config's own (un-overridden)
    values; otherwise a deterministic "smoke_p{P}_n{N}" identity. This is what guarantees a
    full run and any override/smoke run can never write into the same output directory,
    however `--output-dir` was specified (the identity is always appended on top of it).
    """
    if perturbations_per_sigma == paper_perturbations_per_sigma and examples_per_capability == paper_examples_per_capability:
        return "full"
    return f"smoke_p{perturbations_per_sigma}_n{examples_per_capability}"


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
    run_signature: str
    # Cache-lifecycle identity (this repair pass, Stage-7B-precedent audit of Stage 6's own
    # prefix-KV-cache exposure -- see compute_cache_safe_run_signature's own docstring for the
    # full mechanism). None for every HISTORICAL Stage 6 plan (build_pilot_plan never sets
    # these) -- defaulted so every existing PilotPlan(...) construction site stays unaffected.
    multimodal_cache_policy: Optional[str] = None
    enable_prefix_caching: Optional[bool] = None

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
    output_dir: Optional[str] = None,
) -> PilotPlan:
    """Validates the two Stage-6 scientific-integrity invariants that must never silently
    drift (spec sections 3 and 5): the capability set is EXACTLY `PILOT_CAPABILITIES`, and the
    sigma grid is EXACTLY `UPSTREAM_SIGMA_GRID` (order-independent) -- never a config typo
    substituting a fourth capability or an invented sigma value. `output_dir` always has the
    run's `run_signature` appended, so smoke/override runs and the real full run never share
    an ambiguous output identity (see module docstring).
    """
    model = raw_config["model"]
    pilot = raw_config["pilot"]

    capabilities = tuple(pilot["capabilities"])
    if set(capabilities) != set(PILOT_CAPABILITIES):
        raise PilotConfigError(f"Stage-6 pilot capabilities must be exactly {PILOT_CAPABILITIES}, got {capabilities}")

    sigma_grid = tuple(float(s) for s in pilot["sigma_grid"])
    if set(sigma_grid) != set(UPSTREAM_SIGMA_GRID):
        raise PilotConfigError(f"Stage-6 pilot sigma grid must be exactly the upstream grid {UPSTREAM_SIGMA_GRID}, got {sigma_grid}")

    paper_perturbations_per_sigma = pilot["perturbations_per_sigma"]
    paper_examples_per_capability = pilot["examples_per_capability"]
    actual_perturbations_per_sigma = perturbations_per_sigma if perturbations_per_sigma is not None else paper_perturbations_per_sigma
    actual_examples_per_capability = subset_size if subset_size is not None else paper_examples_per_capability

    run_signature = compute_run_signature(
        actual_perturbations_per_sigma, actual_examples_per_capability, paper_perturbations_per_sigma, paper_examples_per_capability,
    )
    base_output_root = Path(output_dir) if output_dir is not None else REPO_ROOT / raw_config["outputs"]["root"]

    return PilotPlan(
        model_name=model["name"], model_revision=model["revision"], model_family=model.get("family", "qwen2_5_vl"),
        model_scale=model.get("scale", "3B"), capabilities=capabilities, sigma_grid=sigma_grid,
        perturbations_per_sigma=actual_perturbations_per_sigma, examples_per_capability=actual_examples_per_capability,
        output_dir=base_output_root / run_signature, run_signature=run_signature,
    )


# =============================================================================================
# CACHE-SAFE STAGE-6 REPRODUCTION (this repair pass -- prefix-KV-cache audit)
# =============================================================================================
# Historical Stage 6 launched via launch_stage6_engine() without overriding enable_prefix_
# caching -- vLLM's own default on the pinned build (confirmed True via Stage 7B's own live
# log, same shared launcher/pinned environment) therefore applied. Source-traced (not
# documentation-inferred): capability_contexts (and therefore every ctx.examples -- the same
# D_map prompts/images) is built EXACTLY ONCE in main(), before the candidate loop, and passed
# BY REFERENCE into every evaluate_one_perturbation_rpc call across the whole population (see
# run_pilot_rpc's `for manifest in population: ... evaluate_one_perturbation_rpc(engine,
# manifest, capability_contexts, ...)`) -- so identical prompt prefixes ARE reused across every
# perturbation candidate, while language weights change between them (reset_to_base_weights ->
# perturb_self_weights each iteration). reset_prefix_cache() is never called anywhere in this
# project's actual code (grepped, zero hits outside comments/docstrings). Nothing in the traced
# call path invalidates vLLM's own internal prefix-cache bookkeeping when weights are mutated
# via collective_rpc -- architecturally identical to the multimodal-encoder-cache bug (vLLM's
# caching layers have no hook for "weights changed underneath me"). This does not PROVE the
# historical Stage 6 result is wrong (no A/B "did output change" check like Gate 2's own has
# ever been run for this specific risk) -- stage6_cache_safety_status = "cache_suspect" (the
# conservative default the source does not disprove), never "safe" or "invalid" outright.
#
# DOES NOT change perturbation_semantics=global_gaussian_upstream, the region (already proven
# to equal language), the frozen UPSTREAM_SIGMA_GRID, DEFAULT_PERTURBATIONS_PER_SIGMA=64,
# DEFAULT_SUBSET_SIZE=50, PILOT_CAPABILITIES, or fixed-base restoration -- ONLY the execution
# -level cache lifecycle. Historical Stage 6 output (results/visual_thicket_global_3b_pilot/
# full/) is untouched, preserved as provenance, marked cache_suspect (not invalid) until this
# reproduction's own results are available; its checkpoint can never resume into this one (see
# compute_cache_safe_run_signature below -- always structurally disjoint from bare "full").

STAGE6_CACHE_SAFE_MULTIMODAL_CACHE_POLICY = "full_encoder_reset_vllm011_verified_v2"
STAGE6_CACHE_SAFE_ENABLE_PREFIX_CACHING = False
STAGE6_CACHE_SAFE_RUN_LABEL = "stage6_global_gaussian_upstream_cache_safe_v2"

# PART 6's dedicated cache-safety smoke -- a genuinely REDUCED sigma SET (2 of the 6 frozen
# sigmas), which build_pilot_plan's own sigma-grid validation (must equal UPSTREAM_SIGMA_GRID)
# does not support -- build_cache_safe_smoke_pilot_plan() below constructs its PilotPlan
# directly rather than going through build_pilot_plan, exactly mirroring Stage 7B's own
# SMOKE_REGION/SMOKE_RADIUS frozen-constant pattern (execution size only, same protocol).
STAGE6_CACHE_SAFE_SMOKE_SIGMA_GRID: Tuple[float, ...] = (0.0005, 0.001)
STAGE6_CACHE_SAFE_SMOKE_PERTURBATIONS_PER_SIGMA = 2
STAGE6_CACHE_SAFE_SMOKE_EXAMPLES_PER_CAPABILITY = 5


def compute_cache_safe_run_signature(
    perturbations_per_sigma: int, examples_per_capability: int, sigma_grid: Sequence[float],
    *, paper_perturbations_per_sigma: int = DEFAULT_PERTURBATIONS_PER_SIGMA,
    paper_examples_per_capability: int = DEFAULT_SUBSET_SIZE, paper_sigma_grid: Sequence[float] = UPSTREAM_SIGMA_GRID,
) -> str:
    """`STAGE6_CACHE_SAFE_RUN_LABEL` ("stage6_global_gaussian_upstream_cache_safe_v2") iff
    sigma_grid/perturbations_per_sigma/examples_per_capability exactly match the frozen paper
    values; otherwise a deterministic smoke-shaped variant built from the ACTUAL values.
    Structurally disjoint from `compute_run_signature`'s own "full"/"smoke_..." identities by
    construction (different prefix entirely) -- the historical run's checkpoint can therefore
    never be silently resumed into a cache-safe run's output_dir, and vice versa.
    """
    if (
        tuple(sorted(sigma_grid)) == tuple(sorted(paper_sigma_grid))
        and perturbations_per_sigma == paper_perturbations_per_sigma and examples_per_capability == paper_examples_per_capability
    ):
        return STAGE6_CACHE_SAFE_RUN_LABEL
    sigma_label = "-".join(str(s) for s in sorted(sigma_grid))
    return f"{STAGE6_CACHE_SAFE_RUN_LABEL}_smoke_sigma{sigma_label}_p{perturbations_per_sigma}_n{examples_per_capability}"


def build_stage6_cache_safe_engine_config() -> Dict[str, Any]:
    """Reuses build_stage6_engine_config()'s own dict BY IDENTITY (never duplicated) plus ONE
    additive override: enable_prefix_caching=False -- see this section's own module-level
    docstring for why. Mirrors run_stage7b_anatomical_calibration.build_stage7b_engine_config
    exactly.
    """
    config = dict(build_stage6_engine_config())
    config["enable_prefix_caching"] = STAGE6_CACHE_SAFE_ENABLE_PREFIX_CACHING
    return config


def build_cache_safe_pilot_plan(
    raw_config: dict, *, perturbations_per_sigma: Optional[int] = None, subset_size: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> PilotPlan:
    """The FULL cache-safe reproduction -- reuses build_pilot_plan()'s OWN validation
    (capabilities must be exactly PILOT_CAPABILITIES, sigma_grid must be exactly
    UPSTREAM_SIGMA_GRID) and base construction UNCHANGED, never duplicated or loosened, then
    layers the cache-safe identity on top via `dataclasses.replace` -- never mutates the base
    plan, never touches build_pilot_plan's own historical-default code path.
    """
    base_plan = build_pilot_plan(raw_config, perturbations_per_sigma=perturbations_per_sigma, subset_size=subset_size, output_dir=output_dir)
    run_signature = compute_cache_safe_run_signature(base_plan.perturbations_per_sigma, base_plan.examples_per_capability, base_plan.sigma_grid)
    base_output_root = Path(output_dir) if output_dir is not None else REPO_ROOT / raw_config["outputs"]["root"]
    return replace(
        base_plan, run_signature=run_signature, output_dir=base_output_root / run_signature,
        multimodal_cache_policy=STAGE6_CACHE_SAFE_MULTIMODAL_CACHE_POLICY, enable_prefix_caching=STAGE6_CACHE_SAFE_ENABLE_PREFIX_CACHING,
    )


def build_cache_safe_smoke_pilot_plan(raw_config: dict) -> PilotPlan:
    """PART 6's dedicated cache-safety smoke: sigma in {0.0005, 0.001} (2 of the 6 frozen
    sigmas), 2 directions/sigma, D_map N=5, all 3 frozen capabilities -- 4 unique perturbations,
    12 result rows, 60 perturbed model-example evaluations. Does NOT call build_pilot_plan
    (whose sigma-grid validation requires the full 6-sigma set) -- constructs PilotPlan
    directly. Execution/instrumentation validation only (verifies perturbation still works,
    fixed-base restoration, prefix caching is False, full cache-reset lifecycle, output/
    checkpoint identity) -- Delta improvement is never a pass criterion for this smoke.
    """
    model = raw_config["model"]
    run_signature = compute_cache_safe_run_signature(
        STAGE6_CACHE_SAFE_SMOKE_PERTURBATIONS_PER_SIGMA, STAGE6_CACHE_SAFE_SMOKE_EXAMPLES_PER_CAPABILITY, STAGE6_CACHE_SAFE_SMOKE_SIGMA_GRID,
    )
    output_root = REPO_ROOT / raw_config["outputs"]["root"]
    return PilotPlan(
        model_name=model["name"], model_revision=model["revision"], model_family=model.get("family", "qwen2_5_vl"),
        model_scale=model.get("scale", "3B"), capabilities=PILOT_CAPABILITIES, sigma_grid=STAGE6_CACHE_SAFE_SMOKE_SIGMA_GRID,
        perturbations_per_sigma=STAGE6_CACHE_SAFE_SMOKE_PERTURBATIONS_PER_SIGMA,
        examples_per_capability=STAGE6_CACHE_SAFE_SMOKE_EXAMPLES_PER_CAPABILITY,
        output_dir=output_root / run_signature, run_signature=run_signature,
        multimodal_cache_policy=STAGE6_CACHE_SAFE_MULTIMODAL_CACHE_POLICY, enable_prefix_caching=STAGE6_CACHE_SAFE_ENABLE_PREFIX_CACHING,
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
        f"run_signature: {plan.run_signature}",
        f"perturbation_semantics: {PERTURBATION_SEMANTICS}",
        f"restoration_mode: {RESTORATION_MODE}",
        "expected_model_loading_strategy: ONE vLLM engine (Ray actor, TP=1, "
        "worker_extension_cls=utils.worker_extn.WorkerExtension, launched by OUR OWN "
        "launch_stage6_engine() -- max_model_len=4096, gpu_memory_utilization=0.60) loaded "
        "once for the whole pilot; store_base_weights called EXACTLY ONCE immediately after "
        "launch to freeze theta_0; per perturbation: reset_to_base_weights -> "
        "perturb_self_weights(seed, sigma) -> evaluate visual_grounding, "
        "ocr_text_recognition_grounded, spatial_reasoning D_map subsets in that fixed order "
        "(engine.generate.remote) -> reset_to_base_weights -> verify EXACT restoration (zero "
        "changed perturbable tensors) via collective_rpc -- restore_self_weights is NEVER "
        "called. No per-perturbation or per-capability engine reload. Zero frontend "
        "llm_engine.model_executor access anywhere in this path.",
        f"output_dir: {plan.output_dir}",
    ]
    if plan.multimodal_cache_policy is not None:
        lines.append(f"multimodal_cache_policy: {plan.multimodal_cache_policy}")
    if plan.enable_prefix_caching is not None:
        lines.append(f"enable_prefix_caching: {plan.enable_prefix_caching}")
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
# Worker-RPC transport -- ZERO frontend model_executor access.
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
    own `perturb_self_weights`/`store_base_weights`/`reset_to_base_weights`, or a plain
    CALLABLE defined in our own package -- the exact same Callable-dispatch convention already
    established by `scoped_perturbation.scoped_apply_perturbation`) via `engine.collective_rpc
    .remote(method, args=args)`, then `ray.get`s and unwraps the single-worker result.
    `ray_get` is injectable purely for CPU testing (a fake engine + identity function) -- real
    callers never pass it, and get the real `ray.get`.
    """
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def perturb_via_rpc(engine: Any, seed: int, sigma: float, *, ray_get: Optional[Callable] = None) -> None:
    """Dispatches upstream's OWN, unmodified `perturb_self_weights(seed, sigma, negate=False)`
    -- string method dispatch, no reimplementation. MUST be called immediately after
    `reset_to_base_weights_via_rpc` so it perturbs from the exact frozen base, not whatever the
    current (possibly already-perturbed) state happens to be.
    """
    _collective_rpc_single_worker(engine, WORKER_PERTURB_METHOD, args=(seed, sigma, False), label=WORKER_PERTURB_METHOD, ray_get=ray_get)


def restore_via_rpc(engine: Any, seed: int, sigma: float, *, ray_get: Optional[Callable] = None) -> None:
    """Dispatches upstream's OWN, unmodified `restore_self_weights(seed, sigma, negate=False)`
    -- the HISTORICAL released-compatible restoration mechanism. NEVER called by Stage 6's
    fixed_base lifecycle (see module docstring) -- kept only so the underlying upstream call
    shape stays documented and independently testable.
    """
    _collective_rpc_single_worker(engine, WORKER_RESTORE_METHOD, args=(seed, sigma, False), label=WORKER_RESTORE_METHOD, ray_get=ray_get)


def store_base_weights_via_rpc(engine: Any, *, ray_get: Optional[Callable] = None) -> None:
    """Dispatches upstream's OWN, unmodified `store_base_weights()` -- clones every current
    parameter as the frozen theta_0 snapshot. Called EXACTLY ONCE per Stage-6 run, immediately
    after the engine initializes, before any perturbation or baseline evaluation.
    """
    _collective_rpc_single_worker(engine, WORKER_STORE_BASE_METHOD, args=(), label=WORKER_STORE_BASE_METHOD, ray_get=ray_get)


def reset_to_base_weights_via_rpc(engine: Any, *, ray_get: Optional[Callable] = None) -> None:
    """Dispatches upstream's OWN, unmodified `reset_to_base_weights()` -- a direct tensor COPY
    from the frozen theta_0 snapshot (`store_base_weights_via_rpc` must have already been
    called on this engine). Called TWICE per candidate: once before perturbing (defensive --
    guarantees perturbation is always applied from the exact stored base regardless of
    whatever the current state is) and once after evaluating (the actual restoration step).
    """
    _collective_rpc_single_worker(engine, WORKER_RESET_TO_BASE_METHOD, args=(), label=WORKER_RESET_TO_BASE_METHOD, ray_get=ray_get)


def compute_mask_info_via_rpc(engine: Any, *, ray_get: Optional[Callable] = None) -> Dict:
    """Dispatches `thicket.worker_rpc.compute_perturbable_mask_info_rpc` as a Callable --
    computed and hashed entirely inside the worker.
    """
    from .thicket.worker_rpc import compute_perturbable_mask_info_rpc

    return _collective_rpc_single_worker(engine, compute_perturbable_mask_info_rpc, args=(), label="compute_perturbable_mask_info_rpc", ray_get=ray_get)


def verify_exact_fixed_base_restoration_via_rpc(engine: Any, *, ray_get: Optional[Callable] = None) -> Dict:
    """Dispatches `thicket.worker_rpc.verify_exact_fixed_base_restoration_rpc` as a Callable --
    checks the EXACT fixed-base restoration invariant (see that module's docstring) entirely
    inside the worker and returns only a small diagnostic dict, never full tensors.
    """
    from .thicket.worker_rpc import verify_exact_fixed_base_restoration_rpc

    return _collective_rpc_single_worker(engine, verify_exact_fixed_base_restoration_rpc, args=(), label="verify_exact_fixed_base_restoration_rpc", ray_get=ray_get)


class RayEngineLLMAdapter:
    """Thin synchronous adapter so the UNMODIFIED `benchmarks.runner.run_benchmark` (which
    calls `llm.generate(requests, sampling_params, use_tqdm=...)` and expects the return value
    directly, never a Ray ObjectRef) can be pointed at a Ray-actor-wrapped vLLM engine -- the
    only way collective_rpc-manipulable weights are reachable under vLLM V1 (see module
    docstring). `run_benchmark` itself needed ZERO changes.
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
# Per-perturbation evaluation lifecycle (fixed-base restoration, this repair pass)
# =============================================================================================


def evaluate_one_perturbation_rpc(
    engine: Any, manifest: PerturbationManifest, capability_contexts: Dict[str, CapabilityContext],
    tokenizer: Any, sampling_params: Any, *, ray_get: Optional[Callable] = None,
) -> List[ExperimentResultRecord]:
    """Fixed-base lifecycle for candidate i (this repair pass -- replaces the subtractive
    `restore_self_weights` restoration used through commit f71f608):

        reset_to_base_weights()                 -- guarantee current == theta_0
        perturb_self_weights(seed_i, sigma)      -- apply epsilon_i from theta_0
        evaluate every capability's D_map subset (in `capability_contexts`'s own iteration
            order -- callers must pass an order-preserving dict built in the required
            visual_grounding -> OCR -> spatial_reasoning order)
        reset_to_base_weights()                  -- restore, via direct copy, back to theta_0
        verify EXACT restoration (zero changed perturbable tensors)

    `restore_self_weights` is NEVER called here. Raising RestorationFailedError aborts the
    whole experiment rather than silently proceeding to the next candidate with possibly
    -drifted base weights.

    CACHE LIFECYCLE (this repair pass, prefix-KV-cache audit -- see the CACHE-SAFE STAGE-6
    REPRODUCTION section above for the full mechanism/evidence): reset_vllm_encoder_cache_full
    is called TWICE per candidate -- once immediately after the accepted perturbation and
    BEFORE any capability is evaluated, and once again immediately after the post-candidate
    fixed-base restoration is verified -- UNCONDITIONALLY, for every Stage 6 execution
    (historical-shaped or cache-safe), mirroring run_stage7b_anatomical_calibration.
    evaluate_one_calibration_candidate_rpc exactly. This changes NO perturbation math or
    scientific semantics -- it is purely an execution-safety addition (mathematically
    unnecessary for a language-only perturbation with respect to the MULTIMODAL encoder cache
    specifically, but removes cache ambiguity entirely per the task's own instruction, and
    additionally forces a fresh generation state that cannot straddle a prefix-cache boundary
    either way).
    """
    llm_adapter = RayEngineLLMAdapter(engine, ray_get=ray_get)
    reset_to_base_weights_via_rpc(engine, ray_get=ray_get)
    perturb_via_rpc(engine, manifest.seed, manifest.sigma, ray_get=ray_get)
    reset_vllm_encoder_cache_full(engine)
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
                runtime_metadata={"restoration_mode": RESTORATION_MODE, "perturbation_semantics": PERTURBATION_SEMANTICS},
            ))
    finally:
        reset_to_base_weights_via_rpc(engine, ray_get=ray_get)

    verification = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Exact fixed-base restoration failed after perturbation {manifest.perturbation_id!r} "
            f"(seed={manifest.seed}, sigma={manifest.sigma}): max_abs_drift="
            f"{verification['max_abs_drift']}, fraction_elements_differing="
            f"{verification['fraction_elements_differing']}"
        )

    reset_vllm_encoder_cache_full(engine)
    return records


def build_stage6_perturbation_population(plan: PilotPlan, base_seed: int, parameter_mask_hash: str) -> Tuple[PerturbationManifest, ...]:
    """Builds the FULL Stage-6 population across every sigma bucket (spec section 5: a fixed
    `perturbations_per_sigma` count per sigma, `global_gaussian_upstream` mode only) --
    UNCHANGED by the fixed-base restoration fix (this function describes epsilon_i, never how
    the model is returned to theta_0 between candidates).

    SEED-DOMAIN FIX (earlier repair pass): the pinned upstream WorkerExtension._set_seed calls
    `np.random.seed(seed)`, which hard-requires `0 <= seed <= 2**32 - 1` -- `thicket.seeds.
    derive_seed`'s own 63-bit output falls outside that domain. Every manifest's worker seed is
    therefore reduced into that domain (`seed % NUMPY_SEED_DOMAIN`) AT GENERATION TIME, via
    `generate_perturbation_population`'s `seed_modulus` parameter -- the value stored on
    `PerturbationManifest.seed` (and used to compute `perturbation_id`) IS the exact uint32
    value later passed to `perturb_self_weights`, never a larger value silently truncated only
    at RPC time.

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


# =============================================================================================
# Checkpoint / resume (spec section 7) -- durable per-candidate persistence.
# =============================================================================================


@dataclass(frozen=True)
class CheckpointManifest:
    """A run's full identity, persisted to `checkpoint_manifest.json` on first write and
    validated (never silently overwritten) on every subsequent resume attempt. Equality is
    exact dataclass field equality -- ANY difference (including, critically,
    `restoration_mode`) makes an existing checkpoint incompatible.
    """
    experiment_id: str
    run_signature: str
    restoration_mode: str
    perturbation_semantics: str
    model_revision: str
    subset_hashes: Dict[str, str]
    subset_size: int
    perturbations_per_sigma: int
    expected_unique_perturbations: int
    expected_result_rows: int
    # Cache-lifecycle identity (this repair pass) -- see PilotPlan's own field docstring. None
    # for every historical checkpoint (predates this field entirely); defaulted so every
    # existing CheckpointManifest(...) construction site and legacy on-disk dict stays valid.
    multimodal_cache_policy: Optional[str] = None
    enable_prefix_caching: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "run_signature": self.run_signature,
            "restoration_mode": self.restoration_mode, "perturbation_semantics": self.perturbation_semantics,
            "model_revision": self.model_revision, "subset_hashes": dict(sorted(self.subset_hashes.items())),
            "subset_size": self.subset_size, "perturbations_per_sigma": self.perturbations_per_sigma,
            "expected_unique_perturbations": self.expected_unique_perturbations, "expected_result_rows": self.expected_result_rows,
            "multimodal_cache_policy": self.multimodal_cache_policy, "enable_prefix_caching": self.enable_prefix_caching,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], run_signature=d["run_signature"], restoration_mode=d["restoration_mode"],
            perturbation_semantics=d["perturbation_semantics"], model_revision=d["model_revision"],
            subset_hashes=dict(d["subset_hashes"]), subset_size=d["subset_size"], perturbations_per_sigma=d["perturbations_per_sigma"],
            expected_unique_perturbations=d["expected_unique_perturbations"], expected_result_rows=d["expected_result_rows"],
            multimodal_cache_policy=d.get("multimodal_cache_policy"), enable_prefix_caching=d.get("enable_prefix_caching"),
        )


def build_stage6_checkpoint_manifest(plan: PilotPlan, capability_contexts: Dict[str, CapabilityContext]) -> CheckpointManifest:
    return CheckpointManifest(
        experiment_id="visual_thicket_global_3b_pilot", run_signature=plan.run_signature, restoration_mode=RESTORATION_MODE,
        perturbation_semantics=PERTURBATION_SEMANTICS, model_revision=plan.model_revision,
        subset_hashes={c: ctx.subset_hash for c, ctx in capability_contexts.items()}, subset_size=plan.examples_per_capability,
        perturbations_per_sigma=plan.perturbations_per_sigma, expected_unique_perturbations=plan.total_unique_perturbations,
        expected_result_rows=plan.total_perturbation_capability_evaluations,
        multimodal_cache_policy=plan.multimodal_cache_policy, enable_prefix_caching=plan.enable_prefix_caching,
    )


def ensure_checkpoint_manifest(path: Path, current: CheckpointManifest) -> CheckpointManifest:
    """Writes `current` if no checkpoint exists yet; otherwise hard-fails
    (IncompatibleCheckpointError) unless the existing checkpoint is field-for-field identical.
    This is the mechanism that makes it impossible to silently resume a differently
    -configured partial run (e.g. one that used subtractive restoration) into this run.
    """
    if path.exists():
        existing = CheckpointManifest.from_dict(json.loads(path.read_text()))
        if existing != current:
            raise IncompatibleCheckpointError(
                f"Existing checkpoint at {path} is incompatible with this run -- refusing to "
                f"resume: existing={existing.to_dict()} current={current.to_dict()}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current.to_dict(), indent=2))
    return current


def load_records(results_path: Path) -> List[ExperimentResultRecord]:
    if not results_path.exists():
        return []
    records = []
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ExperimentResultRecord.from_dict(json.loads(line)))
    return records


def load_completed_perturbation_rows(results_path: Path, expected_capabilities: Sequence[str]) -> Dict[str, List[ExperimentResultRecord]]:
    """Groups already-persisted rows by perturbation_id, returning only those with a COMPLETE
    set (exactly one row per expected capability). An incomplete group is excluded and will be
    re-run from scratch -- resuming NEVER trusts a partial candidate.
    """
    rows_by_pid: Dict[str, List[ExperimentResultRecord]] = {}
    for record in load_records(results_path):
        rows_by_pid.setdefault(record.perturbation_id, []).append(record)
    expected = set(expected_capabilities)
    return {pid: rows for pid, rows in rows_by_pid.items() if {r.capability for r in rows} == expected and len(rows) == len(expected)}


def append_candidate_rows(results_path: Path, records: List[ExperimentResultRecord]) -> None:
    """Durable per-candidate persistence: called ONLY after a candidate's entire apply ->
    evaluate -> reset -> verify cycle has already succeeded (see evaluate_one_perturbation_rpc)
    -- a row appearing in this file is therefore proof restoration passed for that candidate,
    never written speculatively beforehand. flush+fsync so a crash immediately after this call
    returns still leaves the just-completed candidate durably on disk.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_or_compute_baseline_scores(
    baseline_path: Path, capability_contexts: Dict[str, CapabilityContext], model_revision: str, run_signature: str,
    llm_adapter: Any, tokenizer: Any, sampling_params: Any,
) -> None:
    """Populates `ctx.base_score` (from exact theta_0, evaluated immediately after
    `store_base_weights`) for every capability. Reused from `baseline_path` only if the
    persisted model_revision/run_signature/every capability's subset_hash all match this run's
    identity (spec section 8) -- any mismatch hard-fails (IncompatibleCheckpointError) rather
    than silently reusing a stale baseline; a missing file means compute fresh and persist.
    """
    if baseline_path.exists():
        persisted = json.loads(baseline_path.read_text())
        compatible = (
            persisted.get("model_revision") == model_revision and persisted.get("run_signature") == run_signature
            and all(
                capability in persisted.get("capabilities", {}) and persisted["capabilities"][capability]["subset_hash"] == ctx.subset_hash
                for capability, ctx in capability_contexts.items()
            )
        )
        if not compatible:
            raise IncompatibleCheckpointError(
                f"Existing baseline_scores.json at {baseline_path} does not match this run's "
                f"identity -- refusing to reuse stale baseline scores."
            )
        for capability, ctx in capability_contexts.items():
            ctx.base_score = persisted["capabilities"][capability]["score"]
        return

    persisted = {"model_revision": model_revision, "run_signature": run_signature, "capabilities": {}}
    for capability, ctx in capability_contexts.items():
        base_result = run_benchmark(ctx.benchmark, ctx.examples, llm_adapter, tokenizer, sampling_params)
        ctx.base_result = base_result
        ctx.base_score = base_result.aggregate_metrics["primary_metric"]
        persisted["capabilities"][capability] = {"score": ctx.base_score, "subset_hash": ctx.subset_hash}
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(persisted, indent=2))


def run_pilot_rpc(
    plan: PilotPlan, capability_contexts: Dict[str, CapabilityContext], engine: Any, tokenizer: Any,
    sampling_params: Any, base_seed: int, parameter_mask_hash: str, *, ray_get: Optional[Callable] = None,
) -> List[ExperimentResultRecord]:
    """Builds the full population, validates/creates the checkpoint manifest (hard-fails on an
    incompatible existing one), skips any perturbation already completely persisted, and
    evaluates + durably persists every remaining one -- exactly the checkpoint/resume contract
    of spec section 7.
    """
    population = build_stage6_perturbation_population(plan, base_seed, parameter_mask_hash)

    current_checkpoint = build_stage6_checkpoint_manifest(plan, capability_contexts)
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    ensure_checkpoint_manifest(checkpoint_path, current_checkpoint)

    results_path = plan.output_dir / "results.jsonl"
    completed = load_completed_perturbation_rows(results_path, plan.capabilities)

    all_records: List[ExperimentResultRecord] = []
    for rows in completed.values():
        all_records.extend(rows)

    for manifest in population:
        if manifest.perturbation_id in completed:
            continue
        records = evaluate_one_perturbation_rpc(engine, manifest, capability_contexts, tokenizer, sampling_params, ray_get=ray_get)
        append_candidate_rows(results_path, records)
        all_records.extend(records)
    return all_records


# =============================================================================================
# Model-revision resolution + runtime compatibility diagnostic
# =============================================================================================


def resolve_and_report_model_snapshot(model_name: str, revision: str) -> Dict[str, str]:
    """Resolves `model_name@revision` to an immutable local HF snapshot path BEFORE
    constructing vLLM -- the SAME `vlm_adapter.resolve_model_snapshot` function
    `run_capability_benchmark_gate.run_one_capability` already uses for this exact purpose.
    Returns all three of {model_name, requested_revision, resolved_snapshot_path}.
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
    explicitly set -- reports the env var directly rather than guessing at internal engine
    class names that vary across versions.
    """
    if os.environ.get("VLLM_USE_V1") == "0":
        return "V0 (VLLM_USE_V1=0 explicitly set)"
    return "V1 (default; VLLM_USE_V1 not set to 0)"


def build_stage6_engine_config() -> Dict[str, Any]:
    """The exact Stage-6-specific engine construction + scientific-interpretation parameters --
    persisted and printed as runtime metadata so a real run's actual configuration is always
    auditable, not merely assumed from this module's source.
    """
    return {
        "max_model_len": STAGE6_MAX_MODEL_LEN,
        "gpu_memory_utilization": STAGE6_GPU_MEMORY_UTILIZATION,
        "tensor_parallel_size": 1,
        "precision": "bfloat16",
        "restoration_mode": RESTORATION_MODE,
        "perturbation_semantics": PERTURBATION_SEMANTICS,
        "base_snapshot_mode": BASE_SNAPSHOT_MODE,
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
        f"restoration_mode: {engine_config['restoration_mode']}",
        f"perturbation_semantics: {engine_config['perturbation_semantics']}",
        f"base_snapshot_mode: {engine_config['base_snapshot_mode']}",
    ]
    return "\n".join(lines)


def format_base_snapshot_confirmation(gpu_memory_utilization: float, base_snapshot_mode: str) -> str:
    """Printed once, immediately after `store_base_weights_via_rpc` actually succeeds --
    `base_snapshot_stored: True` is a genuine runtime observation (unlike the pre-execution
    diagnostic above, which is printed before the engine even exists).
    """
    return (
        "=== Stage 6: base snapshot stored (theta_0 frozen) ===\n"
        f"gpu_memory_utilization: {gpu_memory_utilization}\n"
        f"base_snapshot_mode: {base_snapshot_mode}\n"
        "base_snapshot_stored: True"
    )


# =============================================================================================
# Figure-2 metrics + diversity
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


def build_run_manifest_summary(checkpoint: CheckpointManifest, records: List[ExperimentResultRecord]) -> Dict[str, Any]:
    actual_unique_perturbations = len({r.perturbation_id for r in records})
    actual_result_rows = len(records)
    run_complete = (
        actual_unique_perturbations == checkpoint.expected_unique_perturbations
        and actual_result_rows == checkpoint.expected_result_rows
    )
    return {
        "subset_size": checkpoint.subset_size, "perturbations_per_sigma": checkpoint.perturbations_per_sigma,
        "expected_unique_perturbations": checkpoint.expected_unique_perturbations, "actual_unique_perturbations": actual_unique_perturbations,
        "expected_result_rows": checkpoint.expected_result_rows, "actual_result_rows": actual_result_rows,
        "restoration_mode": checkpoint.restoration_mode, "run_complete": run_complete,
    }


def write_paper_summary(output_dir: Path) -> Dict[str, Any]:
    """(Re)builds figure2_summary.json/diversity_summary.json from `output_dir/results.jsonl`,
    gated by `output_dir/checkpoint_manifest.json`'s own recorded expectations -- refuses
    (IncompleteRunError) if the actual counts fall short. ALWAYS writes run_manifest.json first
    (the accounting itself, including for an incomplete run, so any shortfall stays visible on
    disk even when the paper summary is refused). Safe to call standalone (`--summarize-only`)
    against a possibly-interrupted run's output directory without accidentally treating
    partial results as the finished paper summary (spec section 6).
    """
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_manifest.json found at {checkpoint_path} -- this run never started (or output_dir is wrong).")
    checkpoint = CheckpointManifest.from_dict(json.loads(checkpoint_path.read_text()))
    records = load_records(output_dir / "results.jsonl")

    manifest = build_run_manifest_summary(checkpoint, records)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not manifest["run_complete"]:
        raise IncompleteRunError(f"Refusing to generate the paper summary for an incomplete run at {output_dir}: {manifest}")

    figure2_summary = compute_figure2_summary(records)
    (output_dir / "figure2_summary.json").write_text(json.dumps(figure2_summary, indent=2))
    diversity_summary = compute_diversity_summary(records)
    (output_dir / "diversity_summary.json").write_text(json.dumps(diversity_summary, indent=2))
    return manifest


# =============================================================================================
# Stage-6-specific engine launcher -- NOT external/RandOpt/core/engine.py's launch_engines(),
# which has no max_model_len parameter and unconditionally calls store_base_weights itself
# (Stage 6 needs precise, controlled ownership of exactly when/how many times that happens).
# =============================================================================================


def launch_stage6_engine(
    model_path: str, *, precision: str = "bfloat16", gpu_memory_utilization: float = STAGE6_GPU_MEMORY_UTILIZATION,
    max_model_len: int = STAGE6_MAX_MODEL_LEN, tensor_parallel_size: int = 1,
    enable_prefix_caching: Optional[bool] = None,
) -> Tuple[list, list]:
    """Stage-6-specific single-engine Ray/vLLM launcher -- an INDEPENDENT, from-scratch
    function in OUR OWN package (external/RandOpt is not modified or subclassed on disk in
    any way); reuses upstream's `RandOptNcclLLM` class directly (imported, never copied) and
    the identical `worker_extension_cls="utils.worker_extn.WorkerExtension"` string, so every
    existing collective_rpc perturb/reset/mask/verify call in this module keeps working
    completely unchanged.

    Deliberately does NOT call `external/RandOpt/core/engine.py:launch_engines()`:
      1. `launch_engines()` accepts no `max_model_len` -- see STAGE6_MAX_MODEL_LEN's own
         comment for the real KV-cache OOM this avoids.
      2. `launch_engines()` unconditionally calls `collective_rpc("store_base_weights")`
         itself, immediately on creation, for every engine -- Stage 6 instead calls
         `store_base_weights_via_rpc` explicitly, exactly once, from `main()`, so the timing
         and count of that call is deliberate and auditable rather than an implicit side
         effect of engine construction.

    Mirrors the REST of `launch_engines()`'s single-engine (TP=1) setup: one GPU-only
    placement group, `RandOptNcclLLM` as a Ray actor with `distributed_executor_backend=
    "ray"`, `enforce_eager=True`, `limit_mm_per_prompt={"image": 1}`. Returns `([engine], [pg])`
    -- the identical list-shaped return `launch_engines()` gives -- so upstream's own
    unmodified `cleanup_engines([engine], [pg])` still works for teardown.

    `enable_prefix_caching` (this repair pass, Stage-7B cache-safety fix): ADDITIVE, opt-in
    override -- `None` (the default, and the only value Stage 6 itself ever passes) omits the
    key from `engine_kwargs` entirely, leaving vLLM's own default exactly as before; this
    function's behavior for every EXISTING (Stage 6) caller is therefore byte-identical to
    before this parameter existed. Stage 7B (run_stage7b_anatomical_calibration.py's
    build_stage7b_engine_config) explicitly passes `enable_prefix_caching=False`: decoder KV
    prefixes may have been computed under a PREVIOUS candidate's now-stale weights, which is
    unsafe across Stage 7B's weight-mutation candidate loop (a hazard Stage 6 does not share,
    since Stage 6 never mutates weights mid-run in the same repeated apply/evaluate/restore
    cycle) -- disabling it entirely is preferred over resetting it candidate-by-candidate.
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
    if enable_prefix_caching is not None:
        engine_kwargs["enable_prefix_caching"] = enable_prefix_caching
    engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(RandOptNcclLLM).remote(**engine_kwargs)

    # Deliberately NOT calling collective_rpc("store_base_weights") here -- main() does it
    # explicitly, exactly once, via store_base_weights_via_rpc -- see docstring above.
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
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="skip GPU execution entirely -- (re)build figure2/diversity summaries from an "
             "existing output_dir's results.jsonl + checkpoint_manifest.json (refuses if incomplete)",
    )
    parser.add_argument(
        "--cache-safe", action="store_true",
        help="the FULL prefix-KV-cache-safe reproduction (this repair pass): SAME frozen "
             "scientific config (perturbation_semantics=global_gaussian_upstream, sigma_grid, "
             "n_per_sigma=64, capabilities, D_map N=50, fixed-base restoration) as historical "
             "Stage 6, under a structurally disjoint run identity (stage6_global_gaussian_"
             "upstream_cache_safe_v2) with enable_prefix_caching=False and the same verified "
             "multimodal_cache_policy=full_encoder_reset_vllm011_verified_v2 lifecycle Stage 7B "
             "uses -- never overwrites or resumes the historical (cache_suspect) run.",
    )
    parser.add_argument(
        "--cache-safe-smoke", action="store_true",
        help="tiny live GPU smoke validating the cache-safe Stage-6 lifecycle: sigma in "
             "{0.0005, 0.001} x 2 directions/sigma x 3 capabilities x D_map N=5 -- 4 "
             "perturbations, 12 rows, 60 perturbed model-example evaluations. Instrumentation/"
             "lifecycle validation only, NOT a behavioral (Delta != 0) check.",
    )
    args = parser.parse_args(argv)

    if sum([args.cache_safe, args.cache_safe_smoke]) > 1:
        print("--cache-safe and --cache-safe-smoke are mutually exclusive.", file=sys.stderr)
        return 1

    raw_config = load_pilot_config(args.config)
    if args.cache_safe:
        plan = build_cache_safe_pilot_plan(
            raw_config, perturbations_per_sigma=args.perturbations_per_sigma, subset_size=args.subset_size, output_dir=args.output_dir,
        )
    elif args.cache_safe_smoke:
        plan = build_cache_safe_smoke_pilot_plan(raw_config)
    else:
        plan = build_pilot_plan(
            raw_config, perturbations_per_sigma=args.perturbations_per_sigma, subset_size=args.subset_size, output_dir=args.output_dir,
        )
    print(format_pilot_plan(plan))

    if args.dry_run:
        return 0

    if args.summarize_only:
        manifest = write_paper_summary(plan.output_dir)
        print(json.dumps(manifest, indent=2))
        return 0

    # --- Real GPU execution path: lazy-imports vllm/ray/transformers, not exercised by CPU
    # tests, not run in this Stage-6 repair-pass preparation session (see module docstring).
    model_resolution = resolve_and_report_model_snapshot(plan.model_name, plan.model_revision)
    # cache-safe modes use build_stage6_cache_safe_engine_config() (adds enable_prefix_caching=
    # False on top of the SAME frozen Stage-6 config) -- historical default stays byte-identical.
    engine_config = build_stage6_cache_safe_engine_config() if (args.cache_safe or args.cache_safe_smoke) else build_stage6_engine_config()
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
    from core.engine import cleanup_engines  # type: ignore  # upstream, unmodified -- teardown only; launch uses OUR OWN launch_stage6_engine

    bootstrap_ray(EXTERNAL_ROOT)

    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        # HARD FAIL BEFORE EXPERIMENT if the full multimodal-encoder-cache reset mechanism
        # cannot be exposed -- MUST run before launch_stage6_engine (see vlm_adapter.py's own
        # ensure_full_encoder_cache_reset_exposed docstring). Unconditional: reset_vllm_
        # encoder_cache_full is now called every candidate regardless of run mode.
        ensure_full_encoder_cache_reset_exposed(EXTERNAL_ROOT)

        engines, pgs = launch_stage6_engine(
            model_resolution["resolved_snapshot_path"], precision=engine_config["precision"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"], max_model_len=engine_config["max_model_len"],
            tensor_parallel_size=engine_config["tensor_parallel_size"],
            enable_prefix_caching=engine_config.get("enable_prefix_caching"),
        )
        engine = engines[0]

        store_base_weights_via_rpc(engine)
        print(format_base_snapshot_confirmation(engine_config["gpu_memory_utilization"], engine_config["base_snapshot_mode"]))

        # HARD FAIL BEFORE EXPERIMENT if the cache reset mechanism doesn't actually WORK
        # end-to-end against the LIVE engine (not merely that it was exposed pre-launch).
        try:
            reset_vllm_encoder_cache_full(engine)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Stage 6 requires a working full multimodal-encoder-cache reset -- verification "
                f"failed against the live engine ({type(exc).__name__}: {exc}). Refusing to start "
                f"candidate evaluation without a proven-working cache-invalidation path."
            ) from exc
        print("Confirmed working multimodal-encoder-cache reset.")

        parameter_mask_hash = compute_mask_info_via_rpc(engine)["mask_hash"]

        llm_adapter = RayEngineLLMAdapter(engine)
        baseline_path = plan.output_dir / "baseline_scores.json"
        load_or_compute_baseline_scores(baseline_path, capability_contexts, plan.model_revision, plan.run_signature, llm_adapter, tokenizer, sampling_params)

        records = run_pilot_rpc(plan, capability_contexts, engine, tokenizer, sampling_params, base_seed, parameter_mask_hash)
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)

    manifest = write_paper_summary(plan.output_dir)
    print(f"Wrote {len(records)} result rows to {plan.output_dir / 'results.jsonl'}")
    print(f"Run manifest: {json.dumps(manifest, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
