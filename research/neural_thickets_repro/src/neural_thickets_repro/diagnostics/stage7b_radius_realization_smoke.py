"""Stage 7B targeted numerical smoke: tests ONLY radius realization + restoration, at the
HARDEST calibration case -- the smallest frozen radius (FULL_CALIBRATION_RADII[0] ==
0.0035698828543799426) -- across all three L1 anatomy regions (vision,
multimodal_connector_or_merger, language), one deterministic perturbation each.

Motivation: the connector region (36,708,608 parameters -- roughly 17x smaller than vision's
631,975,680 and 84x smaller than language's 3,085,938,688) has a DIFFERENT bf16 quantization
floor than vision/language -- a real run of THIS script's own earlier (v2, bracketed-but-
strict-only) version found vision and language converge strictly (abs errors 9.91e-7 and
3.46e-7) but connector proves a genuine quantization_plateau (nearest attainable bf16 states
0.0035686268537125777 / 0.0035711468798464217, relative error ~=3.52e-4, i.e. 0.0352%) -- this
script (now dispatching v3, the quantization-aware two-tier acceptance rule) exists to surface
exactly this kind of per-region numerical floor cheaply, before committing to a 144-candidate
full run.

Does NOT evaluate any capability/dataset -- purely weight-space: apply (bf16 quantization-aware
solver, v3), verify outside-region invariance, reset, verify exact restoration. Reuses
run_stage7b_anatomical_calibration.py's own frozen constants (FULL_CALIBRATION_REGIONS,
FULL_CALIBRATION_RADII, STAGE7B_BASE_SEED) and scoped_anatomical_perturbation.py's
scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 directly -- never reimplements
the solver or the acceptance rule.

The per-region seed is derived via the EXACT SAME generate_perturbation_population call the real
Stage-7B full run uses for (region, smallest_radius, candidate index 0) -- so this script
literally previews that specific real candidate's radius-realization behavior, not a
freestanding synthetic case.

Output is written under `results/stage7b_radius_realization_smoke/`, deliberately SEPARATE from
`results/stage7b_anatomical_calibration/` (Stage 7B's own full/smoke output root) -- this is a
pre-flight numerical diagnostic, never mistaken for calibration candidate rows.

Usage (pod, GPU required):
    python -m neural_thickets_repro.diagnostics.stage7b_radius_realization_smoke \
        --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from ..env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from ..run_stage7b_anatomical_calibration import (
    FULL_CALIBRATION_RADII,
    FULL_CALIBRATION_REGIONS,
    PERTURBATION_MODE,
    RADIUS_REALIZATION_METHOD,
    STAGE7B_BASE_SEED,
    RestorationFailedError,
    build_stage6_engine_config,
    detect_vllm_engine_mode,
    format_runtime_compatibility_diagnostic,
    get_vllm_version,
    launch_stage6_engine,
    report_region_param_names,
    reset_to_base_weights_via_rpc,
    resolve_and_report_model_snapshot,
    store_base_weights_via_rpc,
    verify_exact_fixed_base_restoration_via_rpc,
)
from ..scoped_anatomical_perturbation import (
    CorrectionOutOfRegionDriftError,
    QuantizationToleranceExceededError,
    RadiusCorrectionFailedError,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
)
from ..thicket.perturbation import generate_perturbation_population
from ..vlm_adapter import bootstrap_ray, verify_workers_can_import_external_root

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

SMALLEST_CALIBRATION_RADIUS = FULL_CALIBRATION_RADII[0]  # exactly 0.0035698828543799426 -- the hardest case


def derive_smallest_radius_seed(region: str, *, model_family: str, model_scale: str, model_revision: str) -> int:
    """The EXACT seed the real Stage-7B full run would use for (region, SMALLEST_CALIBRATION_
    RADIUS, candidate index 0) -- derived via the SAME generate_perturbation_population call,
    never a freestanding/independently-chosen seed. Seed derivation does not depend on
    parameter_mask_hash (only perturbation_id does), so a placeholder mask hash here is safe --
    the returned `.seed` is identical to what a real run's own population would assign.
    """
    population = generate_perturbation_population(
        mode=PERTURBATION_MODE, n=1, base_seed=STAGE7B_BASE_SEED, model_family=model_family,
        model_scale=model_scale, model_revision=model_revision, parameter_mask_hash="placeholder_not_seed_relevant",
        anatomy_region=region, radius=SMALLEST_CALIBRATION_RADIUS, sigma=None,
    )
    return population[0].seed


def report_post_solve_outside_region_drift(worker_self, region_param_names: Sequence[str]) -> Dict[str, Any]:
    """Runs entirely inside the worker process, AFTER a solve has accepted (weights still
    loaded) -- re-measures outside-region drift directly against `_base_weights` (no separate
    ad-hoc snapshot mechanism needed) so the report can state a concrete changed-tensor count
    and max drift for the ACCEPTED candidate, not just "the solver's own internal check passed".
    """
    from ..diagnostics.perturb_restore_drift import measure_drift

    if not hasattr(worker_self, "_base_weights"):
        raise RuntimeError("report_post_solve_outside_region_drift requires store_base_weights() to have already been called.")
    model = worker_self.model_runner.model
    base_state = worker_self._base_weights
    region_names_set = set(region_param_names)

    outside = measure_drift(model, base_state, param_filter=lambda n: n not in region_names_set)
    total_params = sum(1 for _ in model.named_parameters())
    changed_tensor_count = 0
    for name, p in model.named_parameters():
        if name in region_names_set:
            continue
        if not torch.equal(p.detach(), base_state[name]):
            changed_tensor_count += 1
    return {
        "outside_region_max_abs_drift": outside["max_abs_drift"],
        "outside_region_changed_tensor_count": changed_tensor_count,
        "outside_region_total_tensor_count": total_params - len(region_names_set),
    }


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list of per-worker results. Got: {results!r}")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; this diagnostic is TP=1-only and expects exactly 1.")
    return results[0]


def _rpc(engine, method, args=(), *, label: str, ray_get=None):
    if ray_get is None:
        import ray

        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def run_one_region_smallest_radius_smoke(
    engine, region: str, region_param_names: Sequence[str], seed: int, *, ray_get=None,
) -> Dict[str, Any]:
    """The per-region lifecycle (Section 3 of the task): apply (bf16 quantization-aware v3
    solver + two-tier acceptance) -> verify outside-region invariance for the ACCEPTED candidate
    -> reset -> verify exact restoration. No dataset/capability evaluation anywhere. Returns
    exactly the fields the task requires, plus the acceptance-mode fields v3 adds.
    """
    solved = False
    radius_acceptance_mode = None
    quantization_limited = None
    plateau = False
    solver_iterations = None
    final_realized = None
    absolute_error = None
    relative_error = None
    nearest_realized_below = None
    nearest_realized_above = None
    error_message = None
    outside_drift = {"outside_region_max_abs_drift": None, "outside_region_changed_tensor_count": None, "outside_region_total_tensor_count": None}

    try:
        apply_result = _rpc(
            engine, scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
            args=(seed, SMALLEST_CALIBRATION_RADIUS, region, tuple(region_param_names)),
            label="scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3", ray_get=ray_get,
        )
        solved = True
        radius_acceptance_mode = apply_result["radius_acceptance_mode"]
        quantization_limited = apply_result["quantization_limited"]
        solver_iterations = apply_result["solver_iterations"]
        final_realized = apply_result["realized_relative_l2"]
        absolute_error = apply_result["absolute_radius_error"]
        relative_error = apply_result["relative_radius_error"]
        plateau = apply_result["quantization_plateau"]
        nearest_realized_below = apply_result["nearest_realized_below"]
        nearest_realized_above = apply_result["nearest_realized_above"]
        outside_drift = _rpc(engine, report_post_solve_outside_region_drift, args=(tuple(region_param_names),), label="report_post_solve_outside_region_drift", ray_get=ray_get)
    except QuantizationToleranceExceededError as e:
        plateau = True
        error_message = str(e)
    except (RadiusCorrectionFailedError, CorrectionOutOfRegionDriftError) as e:
        error_message = str(e)

    reset_to_base_weights_via_rpc(engine, ray_get=ray_get)
    verification = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Radius-realization smoke: exact fixed-base restoration failed after region "
            f"{region!r} (seed={seed}): max_abs_drift={verification['max_abs_drift']}"
        )

    return {
        "region": region,
        "requested_radius": SMALLEST_CALIBRATION_RADIUS,
        "seed": seed,
        "radius_realization_method": RADIUS_REALIZATION_METHOD,
        "solved": solved,
        "radius_acceptance_mode": radius_acceptance_mode,
        "quantization_limited": quantization_limited,
        "final_realized_radius": final_realized,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "solver_iterations": solver_iterations,
        "quantization_plateau": plateau,
        "nearest_realized_below": nearest_realized_below,
        "nearest_realized_above": nearest_realized_above,
        "outside_region_changed_tensor_count": outside_drift["outside_region_changed_tensor_count"],
        "outside_region_max_abs_drift": outside_drift["outside_region_max_abs_drift"],
        "restoration_exact": bool(verification["ok"]),
        "error": error_message,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "stage7b_radius_realization_smoke" / "report.json"))
    args = parser.parse_args(argv)

    from ..config import load_config

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "Stage 7B targeted radius-realization numerical smoke",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision")
    model_resolution = resolve_and_report_model_snapshot(cfg.model.name, cfg.model.revision)
    engine_config = build_stage6_engine_config()
    print(format_runtime_compatibility_diagnostic(
        model_resolution, worker_extension_cls="utils.worker_extn.WorkerExtension",
        vllm_version=get_vllm_version(), engine_mode=detect_vllm_engine_mode(), engine_config=engine_config,
    ))
    print(f"Smallest calibration radius under test: {SMALLEST_CALIBRATION_RADIUS}")
    print(f"Regions under test: {FULL_CALIBRATION_REGIONS} (3 perturbations total, no dataset evaluation)")

    import ray

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore  # upstream, unmodified -- teardown only; launch uses OUR OWN launch_stage6_engine

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)
        engines, pgs = launch_stage6_engine(
            model_resolution["resolved_snapshot_path"], precision=engine_config["precision"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"], max_model_len=engine_config["max_model_len"],
            tensor_parallel_size=engine_config["tensor_parallel_size"],
        )
        engine = engines[0]

        store_base_weights_via_rpc(engine)

        region_info = _rpc(engine, report_region_param_names, args=(FULL_CALIBRATION_REGIONS,), label="report_region_param_names")

        results = []
        for region in FULL_CALIBRATION_REGIONS:
            region_param_names = tuple(region_info[region]["param_names"])
            seed = derive_smallest_radius_seed(region, model_family="qwen2_5_vl", model_scale="3B", model_revision=cfg.model.revision)
            print(f"\n=== region={region} (seed={seed}) ===")
            result = run_one_region_smallest_radius_smoke(engine, region, region_param_names, seed)
            print(json.dumps(result, indent=2))
            results.append(result)
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    report = {
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "smallest_calibration_radius": SMALLEST_CALIBRATION_RADIUS,
        "radius_realization_method": RADIUS_REALIZATION_METHOD,
        "regions": results,
        "note": (
            "Purely a weight-space radius-realization + restoration diagnostic -- no dataset "
            "or capability evaluation. Kept separate from results/stage7b_anatomical_"
            "calibration/ (Stage 7B's own full/smoke output)."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    overall_ok = all(r["solved"] and r["restoration_exact"] for r in results)
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
