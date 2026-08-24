"""Stage 7B: anatomical-region calibration sweep -- PREPARED, NOT YET RUN (Stage 7A task
section 6). Identifies qualitative behavioral regimes (near-base / active-non-collapsed /
destructive) per (anatomy region, radius) cell -- it is explicitly NOT a hyperparameter search
(section 7): no function in this module selects a "best" radius by capability score, and none
ever will (see `test_no_best_radius_selection_logic_exists` in the accompanying test file,
which asserts this mechanically against the module's own public names).

perturbation_mode = anatomical_relative_l2 (thicket.perturbation.apply_anatomical_relative_l2,
the EXACT-rescale mode -- see that module's docstring for why this is scientifically distinct
from scoped_perturbation.py's expectation-only relative_l2 scale mode). Dispatched via
scoped_anatomical_perturbation.scoped_apply_anatomical_perturbation.

regions:      vision, multimodal_connector_or_merger, language (thicket.anatomy L1 regions --
              the SAME common radius grid across all three, never a per-region grid)
capabilities: visual_grounding, ocr_text_recognition_grounded, spatial_reasoning (PILOT_CAPABILITIES,
              reused from run_global_visual_thicket_pilot -- same fixed evaluation order)
population:   CALIBRATION_N_PER_CELL (8) perturbations per (region, radius) cell
data:         a deterministic CALIBRATION_D_MAP_N (20) -example D_map subset per capability,
              built via run_global_visual_thicket_pilot.build_d_map_context (the SAME D_map
              machinery Stage 6 already uses) -- D_confirm/select/test are never constructed or
              referenced anywhere in this module (data_roles.partition_data_roles is only ever
              called with sizes={"map": n}, exactly like Stage 6's own build_d_map_context).

Lifecycle per candidate (fixed-base, same restoration discipline as Stage 6):
    reset_to_base_weights() [defensive, inside scoped_apply_anatomical_perturbation]
    -> apply region-only exact-relative-L2 perturbation
    -> verify realized_relative_l2 matches requested radius within REALIZED_RADIUS_TOLERANCE
    -> verify outside-region parameters are EXACTLY unchanged (max_abs_drift == 0.0)
    -> evaluate all 3 capabilities' D_map(20) subsets, in fixed order
    -> reset_to_base_weights()
    -> verify EXACT fixed-base restoration (reused from run_global_visual_thicket_pilot)
    -> append candidate rows (checkpointed -- same append-only, resume-safe discipline as Stage 6)

Hard-fails (never silently continues) on:
    - abs(realized_relative_l2 - requested_radius) > REALIZED_RADIUS_TOLERANCE
    - any out-of-region parameter drift (max_abs_drift != 0.0)
    - a failed exact fixed-base restoration verification
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .run_global_visual_thicket_pilot import (
    CAPABILITY_CONFIG_FILES,
    PILOT_CAPABILITIES,
    RESTORATION_MODE,
    CapabilityContext,
    RayEngineLLMAdapter,
    RestorationFailedError,
    build_d_map_context,
    reset_to_base_weights_via_rpc,
    verify_exact_fixed_base_restoration_via_rpc,
)
from .scoped_anatomical_perturbation import diag_full_model_drift, diag_region_drift, diag_snapshot_base, scoped_apply_anatomical_perturbation
from .thicket.perturbation import PERTURBATION_MODES, PerturbationManifest, generate_perturbation_population, validate_unique_worker_seeds
from .thicket.schema import ExperimentResultRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

EXPERIMENT_ID = "stage7b_anatomical_calibration"
PERTURBATION_MODE = "anatomical_relative_l2"
assert PERTURBATION_MODE in PERTURBATION_MODES

# Common across all three regions -- Stage 7A section 5/6 explicitly forbids a per-region or
# per-capability radius grid at this stage.
CALIBRATION_REGIONS: Tuple[str, ...] = ("vision", "multimodal_connector_or_merger", "language")
CALIBRATION_CAPABILITIES: Tuple[str, ...] = PILOT_CAPABILITIES
CALIBRATION_N_PER_CELL = 8
CALIBRATION_D_MAP_N = 20
DATASET_ROLE = "map"  # the ONLY role this module ever constructs or references

# Tolerance for the exact-rescale realized-vs-requested relative-L2 check -- matches the
# tolerance already established by tests/test_thicket_perturbation.py's
# test_anatomical_relative_l2_hits_requested_ratio_exactly for the same underlying primitive.
REALIZED_RADIUS_TOLERANCE = 1e-6


class RealizedRadiusMismatchError(RuntimeError):
    """The realized relative-L2 norm did not match the requested radius within
    REALIZED_RADIUS_TOLERANCE -- hard-fails rather than silently recording the requested value.
    """


class OutOfRegionDriftError(RuntimeError):
    """A parameter outside the perturbed anatomy region changed -- hard-fails rather than
    silently recording a candidate whose perturbation leaked outside its declared scope.
    """


class DatasetRoleViolationError(RuntimeError):
    """Something other than the 'map' dataset role was requested -- this module must never
    construct or reference D_confirm/D_select/D_test.
    """


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
    output_dir: Path

    @property
    def total_unique_perturbations(self) -> int:
        return len(self.regions) * len(self.radii) * self.n_per_cell

    @property
    def total_perturbation_capability_evaluations(self) -> int:
        return self.total_unique_perturbations * len(self.capabilities)


def build_stage7b_plan(
    *, model_name: str, model_revision: str, radii: Sequence[float], output_dir: "str | Path",
    model_family: str = "qwen2_5_vl", model_scale: str = "3B",
) -> Stage7bPlan:
    """`radii` is expected to come from Stage 7A's own `stage7_calibration_plan.json`
    (`proposed_initial_calibration_radii.radii`) -- this function does not derive or filter
    radii itself, it only assembles the fixed regions/capabilities/counts around whatever
    radii the caller already mechanically derived.
    """
    if not radii:
        raise ValueError("Stage 7B requires at least one calibration radius.")
    return Stage7bPlan(
        model_name=model_name, model_revision=model_revision, model_family=model_family, model_scale=model_scale,
        regions=CALIBRATION_REGIONS, radii=tuple(radii), capabilities=CALIBRATION_CAPABILITIES,
        n_per_cell=CALIBRATION_N_PER_CELL, d_map_n=CALIBRATION_D_MAP_N, output_dir=Path(output_dir),
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
    -configured partial run.
    """


@dataclass(frozen=True)
class Stage7bCheckpointManifest:
    experiment_id: str
    restoration_mode: str
    perturbation_mode: str
    model_revision: str
    dataset_role: str
    regions: Tuple[str, ...]
    radii: Tuple[float, ...]
    capabilities: Tuple[str, ...]
    n_per_cell: int
    d_map_n: int
    subset_hashes: Dict[str, str]
    expected_unique_perturbations: int
    expected_result_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "restoration_mode": self.restoration_mode,
            "perturbation_mode": self.perturbation_mode, "model_revision": self.model_revision,
            "dataset_role": self.dataset_role, "regions": list(self.regions), "radii": list(self.radii),
            "capabilities": list(self.capabilities), "n_per_cell": self.n_per_cell, "d_map_n": self.d_map_n,
            "subset_hashes": dict(sorted(self.subset_hashes.items())),
            "expected_unique_perturbations": self.expected_unique_perturbations,
            "expected_result_rows": self.expected_result_rows,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage7bCheckpointManifest":
        return cls(
            experiment_id=d["experiment_id"], restoration_mode=d["restoration_mode"], perturbation_mode=d["perturbation_mode"],
            model_revision=d["model_revision"], dataset_role=d["dataset_role"], regions=tuple(d["regions"]),
            radii=tuple(d["radii"]), capabilities=tuple(d["capabilities"]), n_per_cell=d["n_per_cell"], d_map_n=d["d_map_n"],
            subset_hashes=dict(d["subset_hashes"]), expected_unique_perturbations=d["expected_unique_perturbations"],
            expected_result_rows=d["expected_result_rows"],
        )


def build_stage7b_checkpoint_manifest(plan: Stage7bPlan, capability_contexts: Dict[str, CapabilityContext]) -> Stage7bCheckpointManifest:
    if plan.d_map_n != CALIBRATION_D_MAP_N:
        raise DatasetRoleViolationError(f"Stage 7B D_map size is fixed at {CALIBRATION_D_MAP_N}, got {plan.d_map_n}")
    return Stage7bCheckpointManifest(
        experiment_id=EXPERIMENT_ID, restoration_mode=RESTORATION_MODE, perturbation_mode=PERTURBATION_MODE,
        model_revision=plan.model_revision, dataset_role=DATASET_ROLE, regions=plan.regions, radii=plan.radii,
        capabilities=plan.capabilities, n_per_cell=plan.n_per_cell, d_map_n=plan.d_map_n,
        subset_hashes={c: ctx.subset_hash for c, ctx in capability_contexts.items()},
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


def build_d_map_capability_contexts(base_seed: int, subset_ids_dir: "str | Path", *, load_capability_benchmark_config: Callable, load_adapter: Callable) -> Dict[str, CapabilityContext]:
    """Builds the fixed 3-capability, CALIBRATION_D_MAP_N-example D_map context set -- the ONLY
    dataset role this module ever touches. `load_capability_benchmark_config`/`load_adapter`
    are injected (rather than imported at module top) purely so this stays importable/testable
    without pulling in the full benchmark-gate config machinery at import time; real callers
    pass `.config.load_capability_benchmark_config` / `.run_capability_benchmark_gate.load_adapter`.
    """
    contexts: Dict[str, CapabilityContext] = {}
    for capability in CALIBRATION_CAPABILITIES:  # fixed order, matches Stage 6's own required order
        cfg_path = REPO_ROOT / "configs" / "benchmarks" / CAPABILITY_CONFIG_FILES[capability]
        cfg = load_capability_benchmark_config(cfg_path)
        benchmark = load_adapter(cfg.dataset.adapter)
        ctx = build_d_map_context(benchmark, cfg, capability, CALIBRATION_D_MAP_N, base_seed, subset_ids_dir)
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
    """
    if manifest.perturbation_mode != PERTURBATION_MODE:
        raise ValueError(f"Stage 7B only evaluates {PERTURBATION_MODE!r} manifests, got {manifest.perturbation_mode!r}")

    _collective_rpc_single_worker(engine, diag_snapshot_base, args=(), label="diag_snapshot_base", ray_get=ray_get)

    llm_adapter = RayEngineLLMAdapter(engine, ray_get=ray_get)
    records: List[ExperimentResultRecord] = []
    try:
        apply_result = _collective_rpc_single_worker(
            engine, scoped_apply_anatomical_perturbation,
            args=(manifest.seed, manifest.radius, manifest.anatomy_region, tuple(region_param_names)),
            label="scoped_apply_anatomical_perturbation", ray_get=ray_get,
        )
        realized_r = apply_result["realized_relative_l2"]
        if abs(realized_r - manifest.radius) > REALIZED_RADIUS_TOLERANCE:
            raise RealizedRadiusMismatchError(
                f"Perturbation {manifest.perturbation_id!r} (region={manifest.anatomy_region!r}, "
                f"requested radius={manifest.radius}): realized relative-L2 {realized_r} differs "
                f"by more than {REALIZED_RADIUS_TOLERANCE}."
            )

        drift = _collective_rpc_single_worker(engine, diag_region_drift, args=(tuple(region_param_names),), label="diag_region_drift", ray_get=ray_get)
        if drift["out_of_region"]["max_abs_drift"] != 0.0:
            raise OutOfRegionDriftError(
                f"Perturbation {manifest.perturbation_id!r} (region={manifest.anatomy_region!r}) changed "
                f"{drift['out_of_region']['fraction_elements_differing']:.2e} fraction of out-of-region "
                f"elements (max_abs_drift={drift['out_of_region']['max_abs_drift']}) -- refusing to record."
            )

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
                    "restoration_mode": RESTORATION_MODE, "requested_relative_l2": manifest.radius,
                    "realized_relative_l2": realized_r, "region_param_count": apply_result["region_param_count"],
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
    return records


# =============================================================================================
# CLI entry point -- NOT executed against GPU by this task (section 6: "prepare, do not run")
# =============================================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-plan", required=True, help="path to Stage 7A's stage7_calibration_plan.json (source of proposed_initial_calibration_radii)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "stage7b_anatomical_calibration"))
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit -- no model load, no GPU")
    args = parser.parse_args(argv)

    calibration_plan_data = json.loads(Path(args.calibration_plan).read_text())
    radii = calibration_plan_data["proposed_initial_calibration_radii"]["radii"]
    model_name = calibration_plan_data["model_name"]
    model_revision = calibration_plan_data["model_revision"]

    plan = build_stage7b_plan(model_name=model_name, model_revision=model_revision, radii=radii, output_dir=args.output_dir)
    print(f"Stage 7B plan: regions={plan.regions} radii={plan.radii} capabilities={plan.capabilities}")
    print(f"n_per_cell={plan.n_per_cell} d_map_n={plan.d_map_n} total_unique_perturbations={plan.total_unique_perturbations}")
    print(f"total_perturbation_x_capability_evaluations={plan.total_perturbation_capability_evaluations}")
    print(
        "Lifecycle: reset_to_base_weights -> scoped_apply_anatomical_perturbation (exact "
        "relative-L2 rescale) -> verify realized radius within tolerance -> verify zero "
        "out-of-region drift -> evaluate visual_grounding/ocr_text_recognition_grounded/"
        "spatial_reasoning D_map(20) -> reset_to_base_weights -> verify exact restoration."
    )
    print("THIS SCRIPT DOES NOT LAUNCH A GPU ENGINE YET -- Stage 7A explicitly prepares, but does not run, Stage 7B.")

    if args.dry_run:
        return 0

    raise NotImplementedError(
        "Stage 7B GPU execution is intentionally not wired up in this task -- Stage 7A section "
        "6 requires the calibration runner to be PREPARED, not run. Re-run with --dry-run, or "
        "extend this main() only once Stage 7B is explicitly authorized."
    )


if __name__ == "__main__":
    raise SystemExit(main())
