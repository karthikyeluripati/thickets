"""Stage 6: 3B Global Visual-Thicket Pilot -- the FIRST REAL PAPER EXPERIMENT.

Answers RQ1 ("do pretrained VLMs contain dense and diverse nearby experts across distinct
visual capabilities?") and produces the raw material for Figure 2 (performance-density
distributions, solution-density curves, radius dependence, best-of-N behavior) and a first
look at Figure 4's Spectral Discordance -- NOT the anatomy experiment (that's Stage 7+): every
perturbation here is a `global_gaussian_upstream` perturbation (spec section C1 of
VISUAL_THICKET_EXPERIMENT_SPEC.md), never `anatomical_relative_l2`.

UPSTREAM RECONCILIATION (this stage): the pinned upstream commit
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
    noise and subtracting rather than a stored-copy restore. This stage's `global_gaussian_
    upstream` perturbation mode (`thicket.perturbation`) already wraps `perturb_cpu.perturb`/
    `restore` unchanged -- nothing needed fixing here.
  - `randopt.py:run_sampling` draws `population_size` UNIQUE seeds without replacement, then
    draws ONE sigma per candidate independently WITH replacement from the sigma list via
    `rng.choice(sigma_list, size=population_size)` -- i.e. upstream does NOT evaluate a fixed
    count per sigma bucket. This pilot deliberately DEPARTS from that one detail (evaluating a
    fixed `perturbations_per_sigma` count per sigma, per this stage's own task spec) to get a
    clean per-sigma breakdown for Figure 2's radius-dependence panel; every other mechanic
    (the sigma VALUES themselves, the per-tensor-reseed noise, the perturb-then-restore
    lifecycle) is unchanged and reused verbatim via `thicket.perturbation`.
  - Spectral Discordance: grepped the full pinned checkout (`spectral|discordance|spearman`,
    case-insensitive) -- ZERO matches anywhere in upstream source. There is no upstream
    implementation to reconcile against. Stage 5's implementation (ported from the published
    paper's Definition 2.2) therefore remains the sole authoritative definition; nothing
    needed changing (see VISUAL_THICKET_EXPERIMENT_SPEC.md's updated H2 status).

EXECUTION MODEL (matches upstream's own `run_sampling` sequence, confirmed by reading it):
one model loaded once; per perturbation: apply (in-place) -> generate on every capability's
fixed D_map subset, in a fixed order -> restore (regenerate identical noise, subtract) ->
verify restoration against a stored base snapshot -> next perturbation. No per-candidate model
reload -- this project's existing `benchmarks/runner.run_benchmark` free function (already
designed for exactly this reuse, see its own docstring) is called once per capability per
perturbation, unmodified.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .benchmarks.runner import RunResult, run_benchmark
from .benchmarks.subset_selection import build_or_load_subset
from .perturb_cpu import should_perturb
from .thicket import diversity as thicket_diversity
from .thicket import metrics as thicket_metrics
from .thicket.anatomy import compute_mask_hash
from .thicket.data_roles import DataRolePartition, partition_data_roles, write_data_role_manifest
from .thicket.perturbation import (
    PerturbationManifest,
    apply_global_gaussian_upstream,
    generate_perturbation_population,
    undo_global_gaussian_upstream,
)
from .thicket.schema import ExperimentResultRecord

REPO_ROOT = Path(__file__).resolve().parents[2]

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
DEFAULT_RESTORATION_ATOL = 1e-3

SOLUTION_DENSITY_MARGINS: Tuple[float, ...] = (0.0, 0.02, 0.05)


class PilotConfigError(ValueError):
    """The loaded pilot YAML/overrides violate a Stage-6 scientific-integrity constraint
    (wrong capability set, an invented sigma grid, etc.) -- never silently accepted.
    """


class RestorationFailedError(RuntimeError):
    """A perturbation's restore step did not return model parameters to within tolerance of
    the stored base snapshot -- the experiment must abort here (spec section 7), never
    silently continue with possibly-accumulated drift.
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
    output_dir: Optional[str] = None, restoration_atol: Optional[float] = None,
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
        "expected_model_loading_strategy: ONE vLLM engine loaded once for the whole pilot; per "
        "perturbation: apply_global_gaussian_upstream (in-place) -> evaluate visual_grounding, "
        "ocr_text_recognition_grounded, spatial_reasoning D_map subsets in that fixed order -> "
        "undo_global_gaussian_upstream (in-place, regenerate+subtract identical noise) -> "
        "verify_restoration against the stored base snapshot -> next perturbation. No "
        "per-perturbation or per-capability engine reload.",
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
# Restoration safety (spec section 7)
# =============================================================================================


def snapshot_state(model) -> Dict[str, Any]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def verify_restoration(model, base_snapshot: Dict[str, Any], atol: float = DEFAULT_RESTORATION_ATOL) -> Dict[str, float]:
    """Compares every current parameter against the stored base snapshot; raises
    RestorationFailedError (never a warning) if any tensor's max absolute difference exceeds
    `atol`. Returns the full per-tensor max-abs-diff map on success, for logging.
    """
    max_diffs: Dict[str, float] = {}
    failures: Dict[str, float] = {}
    current = dict(model.named_parameters())
    for name, base_tensor in base_snapshot.items():
        diff = (current[name].detach().float() - base_tensor.float()).abs().max().item()
        max_diffs[name] = diff
        if diff > atol:
            failures[name] = diff
    if failures:
        worst = dict(sorted(failures.items(), key=lambda kv: -kv[1])[:5])
        raise RestorationFailedError(
            f"Restoration check failed for {len(failures)}/{len(base_snapshot)} parameter(s) "
            f"(max abs diff exceeds atol={atol}): worst offenders {worst}"
        )
    return max_diffs


# =============================================================================================
# Per-perturbation evaluation lifecycle (spec sections 5-7)
# =============================================================================================


def evaluate_one_perturbation(
    model: Any, manifest: PerturbationManifest, capability_contexts: Dict[str, CapabilityContext],
    llm: Any, tokenizer: Any, sampling_params: Any, base_snapshot: Dict[str, Any], restoration_atol: float,
) -> List[ExperimentResultRecord]:
    """Applies `manifest`'s perturbation ONCE, evaluates every capability's D_map subset (in
    `capability_contexts`'s own iteration order -- callers must pass an order-preserving dict
    built in the required visual_grounding -> OCR -> spatial_reasoning order), restores, then
    verifies restoration -- raising RestorationFailedError aborts the whole experiment rather
    than silently proceeding to the next perturbation with possibly-drifted base weights.
    """
    apply_record = apply_global_gaussian_upstream(model, seed=manifest.seed, sigma=manifest.sigma)
    records: List[ExperimentResultRecord] = []
    try:
        for capability, ctx in capability_contexts.items():
            result = run_benchmark(ctx.benchmark, ctx.examples, llm, tokenizer, sampling_params)
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
        undo_global_gaussian_upstream(model, apply_record)
    verify_restoration(model, base_snapshot, atol=restoration_atol)
    return records


def compute_global_gaussian_mask_hash(model: Any) -> str:
    """The 'mask' for `global_gaussian_upstream` is every named parameter NOT skipped by
    `should_perturb` (visual-encoder prefixes excluded) -- hashed the identical way
    `thicket.anatomy` hashes every other region, for a consistent `parameter_mask_hash`
    convention across both perturbation modes.
    """
    names = [name for name, _ in model.named_parameters() if should_perturb(name)]
    return compute_mask_hash(names)


def run_pilot(
    plan: PilotPlan, capability_contexts: Dict[str, CapabilityContext], model: Any, llm: Any, tokenizer: Any,
    sampling_params: Any, base_seed: int, parameter_mask_hash: str,
) -> List[ExperimentResultRecord]:
    """Loops sigma -> perturbation -> capability, exactly the nesting spec section 6 requires
    at the perturbation/capability level (sigma is the outer grouping this pilot adds on top,
    for a clean per-sigma bucket count -- see module docstring's upstream-reconciliation note).
    """
    base_snapshot = snapshot_state(model)
    all_records: List[ExperimentResultRecord] = []
    for sigma in plan.sigma_grid:
        population = generate_perturbation_population(
            mode="global_gaussian_upstream", n=plan.perturbations_per_sigma, base_seed=base_seed,
            model_family=plan.model_family, model_scale=plan.model_scale, model_revision=plan.model_revision,
            parameter_mask_hash=parameter_mask_hash, anatomy_region=None, radius=None, sigma=sigma,
        )
        for manifest in population:
            all_records.extend(evaluate_one_perturbation(
                model, manifest, capability_contexts, llm, tokenizer, sampling_params, base_snapshot, plan.restoration_atol,
            ))
    return all_records


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

    # --- Real GPU execution path: lazy-imports vllm/transformers, not exercised by CPU tests,
    # not run in this Stage-6 preparation session (see module docstring / task instructions).
    from .config import load_capability_benchmark_config
    from .run_capability_benchmark_gate import build_llm_and_tokenizer, load_adapter

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    subset_ids_dir = plan.output_dir / "d_map_subsets"
    base_seed = raw_config["pilot"].get("base_seed", 20260823)

    capability_contexts: Dict[str, CapabilityContext] = {}
    for capability in PILOT_CAPABILITIES:  # fixed order, matches the required execution order
        cfg_path = REPO_ROOT / "configs" / "benchmarks" / CAPABILITY_CONFIG_FILES[capability]
        cfg = load_capability_benchmark_config(cfg_path)
        benchmark = load_adapter(cfg.dataset.adapter)
        seed = base_seed  # same fixed seed reused for every capability's own D_map construction
        ctx = build_d_map_context(benchmark, cfg, capability, plan.examples_per_capability, seed, subset_ids_dir)
        write_data_role_manifest(ctx.partition, plan.output_dir / "data_roles" / f"{capability}_d_map.json")
        capability_contexts[capability] = ctx

    llm, tokenizer = build_llm_and_tokenizer(plan.model_name, precision="bfloat16")
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    for capability, ctx in capability_contexts.items():
        base_result = run_benchmark(ctx.benchmark, ctx.examples, llm, tokenizer, sampling_params)
        ctx.base_result = base_result
        ctx.base_score = base_result.aggregate_metrics["primary_metric"]

    parameter_mask_hash = compute_global_gaussian_mask_hash(llm.llm_engine.model_executor.driver_worker.model_runner.model)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    records = run_pilot(plan, capability_contexts, model, llm, tokenizer, sampling_params, base_seed, parameter_mask_hash)

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
