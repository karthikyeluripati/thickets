"""Stage 7A: live anatomy inventory + upstream-scope inventory + sigma-to-relative-L2 radius
anchoring, against the REAL loaded Qwen2.5-VL-3B-Instruct (pinned revision) -- Sections 1-5 of
the Stage-7A task. Does NOT run any anatomical perturbation sweep (that is Stage 7B, a
separate, not-yet-executed module: run_stage7b_anatomical_calibration.py) and does NOT touch
the vision encoder except in the optional, single-perturbation empirical sanity check
(Section 4), which perturbs only the Stage-6 upstream scope (language, i.e. `full_lm`) and
resets exactly before returning.

REGRESSION FIX (this repair pass): the first version of this script launched the engine via
`external/RandOpt/core/engine.py`'s upstream `launch_engines()`, which accepts no
`max_model_len` and defaults to the real model's own 128000 -- on a real RunPod L40S at
`gpu_memory_utilization=0.75` this required 4.39 GiB of KV cache against 3.01 GiB available,
and vLLM aborted during engine initialization before Stage 7A ever ran. This is the EXACT same
class of problem Stage 6 already solved (see run_global_visual_thicket_pilot.py's own
`launch_stage6_engine` docstring) -- Stage 7A now REUSES that exact, already-battle-tested
Stage-6 launcher and its engine-config helpers directly (import, not reimplementation):
`launch_stage6_engine`, `build_stage6_engine_config` (max_model_len=4096,
gpu_memory_utilization=0.60, tensor_parallel_size=1, precision=bfloat16 -- all Stage-6-frozen,
none of them re-derived here), `resolve_and_report_model_snapshot`,
`format_runtime_compatibility_diagnostic`, `format_base_snapshot_confirmation`,
`store_base_weights_via_rpc`, `reset_to_base_weights_via_rpc`,
`verify_exact_fixed_base_restoration_via_rpc`, `RestorationFailedError`. `external/RandOpt` is
untouched by this fix -- only `cleanup_engines` (teardown, unconditionally safe for any engine
shaped like `launch_engines()`'s own return value, per `launch_stage6_engine`'s own docstring)
is still imported from it; `launch_engines` itself is never called or imported here anymore.

BASE-SNAPSHOT LIFECYCLE (this repair pass): unlike upstream's `launch_engines()`, which
unconditionally calls `store_base_weights()` as a hidden side effect of engine construction,
`launch_stage6_engine` does NOT call it at all -- the caller must do so explicitly. Section 1/2
anatomy/upstream-scope inventory needs no base snapshot (it only reads `named_parameters()`,
never perturbs). `store_base_weights_via_rpc` (a second, GPU-resident full-model weight copy)
is therefore called ONLY when Section 4's empirical check is not skipped -- see
`maybe_run_empirical_check` below.

Reuses, never reimplements:
  - thicket.anatomy.build_anatomy_atlas / validate_atlas (frozen L1/L2 region discovery)
  - thicket.anatomy_inventory.build_full_anatomy_inventory (per-region numeric report)
  - thicket.upstream_scope.compute_upstream_scope_inventory (Section 2)
  - thicket.radius_mapping.build_sigma_relative_l2_mapping / select_common_calibration_radii
    (Section 3/5) and FROZEN_STAGE6_SIGMAS (imported from run_global_visual_thicket_pilot's own
    UPSTREAM_SIGMA_GRID -- never a second, independently-typed copy of that grid)
  - scopes.build_scope_manifest / scoped_perturbation.scoped_apply_perturbation (Section 4's
    empirical check dispatches the EXISTING "full_lm" raw_sigma scope path unchanged -- "full_lm"
    and anatomy.py's "language" region are the SAME parameter set, confirmed by this script's
    own measured equality check, not merely assumed)
  - run_global_visual_thicket_pilot.launch_stage6_engine and its engine-config/diagnostic/
    base-snapshot/restoration helpers (this repair pass -- see above)

No dataset is loaded or evaluated anywhere in this module -- Sections 1-5 are pure weight
inspection plus (optionally) one single-perturbation weight-norm measurement.

Usage (pod, GPU required):
    python -m neural_thickets_repro.diagnostics.anatomy_inventory_gpu \
        --config configs/gqa_repro.yaml \
        --out results/stage7_anatomy_calibration/stage7_calibration_plan.json

Writes two files next to `--out`: the full calibration-plan JSON (`--out` itself) and a
standalone `upstream_scope_inventory.json` (Section 2's own persisted-evidence requirement).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import load_config
from ..env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from ..run_global_visual_thicket_pilot import (
    RestorationFailedError,
    build_stage6_engine_config,
    format_base_snapshot_confirmation,
    format_runtime_compatibility_diagnostic,
    detect_vllm_engine_mode,
    get_vllm_version,
    launch_stage6_engine,
    reset_to_base_weights_via_rpc,
    resolve_and_report_model_snapshot,
    store_base_weights_via_rpc,
    verify_exact_fixed_base_restoration_via_rpc,
)
from ..scoped_perturbation import scoped_apply_perturbation
from ..thicket.anatomy import build_anatomy_atlas, validate_atlas
from ..thicket.anatomy_inventory import build_full_anatomy_inventory
from ..thicket.radius_mapping import ANCHOR_LABEL, FROZEN_STAGE6_SIGMAS, build_sigma_relative_l2_mapping, select_common_calibration_radii
from ..thicket.upstream_scope import compute_upstream_scope_inventory
from ..vlm_adapter import bootstrap_ray, verify_workers_can_import_external_root

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

# Fixed, clearly-not-a-real-candidate seed/sigma for the optional empirical sanity check --
# same naming convention as scope_isolation_gpu_check.py's TEST_SEED/TEST_SIGMA. Frozen: this
# diagnostic checks ONLY sigma=0.001 with ONE seed, never a population or a sigma sweep.
EMPIRICAL_CHECK_SEED = 999_999_998
EMPIRICAL_CHECK_SIGMA = 0.001
EMPIRICAL_CHECK_SCOPE = "full_lm"


def _report_anatomy_and_upstream_scope(worker_self) -> Dict[str, Any]:
    """Runs entirely inside the worker process. Builds the atlas from the model's REAL
    named_parameters(), validates it (hard-fails on unexpectedly empty/overlapping regions --
    never silently accepted), and computes the full Section 1 + Section 2 numeric report. Reads
    weights only -- never perturbs, never requires a stored base snapshot.
    """
    model = worker_self.model_runner.model
    named_parameters = dict(model.named_parameters())
    names = list(named_parameters.keys())

    atlas = build_anatomy_atlas(names, model_family="qwen2_5_vl")
    validate_atlas(atlas)  # raises AnatomyValidationError on empty region / sibling overlap

    anatomy_report = build_full_anatomy_inventory(
        atlas, named_parameters, model_family="qwen2_5_vl", model_revision="<filled by caller>",
    )
    upstream_report = compute_upstream_scope_inventory(atlas, named_parameters)

    return {"anatomy_inventory": anatomy_report, "upstream_scope_inventory": upstream_report}


def _validate_collective_rpc_results(results: Any, *, label: str) -> Any:
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list of per-worker results. Got: {results!r}")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; this diagnostic is TP=1-only and expects exactly 1.")
    return results[0]


def _rpc(engine, method, args=(), *, label: str, ray_get: Optional[Any] = None):
    if ray_get is None:
        import ray

        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def _run_empirical_norm_sanity_check(engine, *, ray_get: Optional[Any] = None) -> Dict[str, Any]:
    """Section 4: apply exactly ONE raw-sigma perturbation (sigma=0.001, one fixed seed) to the
    Stage-6 upstream scope (scopes.py's "full_lm" -- confirmed elsewhere in this report to
    equal anatomy.py's "language" L1 region), measure the realized ||epsilon||_2 / ||theta||_2
    against the analytical sigma*sqrt(d)/||theta|| approximation, then reset exactly to base and
    VERIFY exact restoration (reused `verify_exact_fixed_base_restoration_via_rpc` -- the same
    check Stage 6 runs after every candidate). Requires `store_base_weights_via_rpc` to have
    already been called on this engine. Never evaluates any dataset and never runs a
    population. `ray_get` is injectable purely for CPU testing (a fake engine + identity
    function) -- real callers never pass it.
    """
    print(f"\nEmpirical norm sanity check: seed={EMPIRICAL_CHECK_SEED}, sigma={EMPIRICAL_CHECK_SIGMA}, scope={EMPIRICAL_CHECK_SCOPE} (raw_sigma)...")
    result = _rpc(
        engine, scoped_apply_perturbation,
        args=(EMPIRICAL_CHECK_SEED, EMPIRICAL_CHECK_SIGMA, EMPIRICAL_CHECK_SCOPE, "raw_sigma"),
        label="scoped_apply_perturbation", ray_get=ray_get,
    )
    realized_l2 = result["actual_perturbation_l2"]
    theta_l2 = result["scope_base_l2_norm"]
    d = result["scope_total_element_count"]
    realized_relative_l2 = realized_l2 / theta_l2 if theta_l2 > 0 else 0.0
    analytical_relative_l2 = (EMPIRICAL_CHECK_SIGMA * (d ** 0.5)) / theta_l2 if theta_l2 > 0 else 0.0

    print("  Resetting to base...")
    reset_to_base_weights_via_rpc(engine, ray_get=ray_get)

    verification = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=ray_get)
    if not verification["ok"]:
        raise RestorationFailedError(
            f"Exact fixed-base restoration failed after Stage 7A's empirical norm sanity check "
            f"(seed={EMPIRICAL_CHECK_SEED}, sigma={EMPIRICAL_CHECK_SIGMA}): "
            f"max_abs_drift={verification['max_abs_drift']}"
        )

    return {
        "seed": EMPIRICAL_CHECK_SEED,
        "sigma": EMPIRICAL_CHECK_SIGMA,
        "scope": EMPIRICAL_CHECK_SCOPE,
        "d": d,
        "theta_l2_norm": theta_l2,
        "realized_epsilon_l2_norm": realized_l2,
        "realized_relative_l2": realized_relative_l2,
        "analytical_expected_relative_l2": analytical_relative_l2,
        "absolute_difference": abs(realized_relative_l2 - analytical_relative_l2),
        "restoration_verified_exact": True,
        "note": "Single-seed empirical measurement -- verifies the sigma*sqrt(d) approximation is accurate at model scale, not a claim about any other sigma.",
    }


def maybe_run_empirical_check(
    engine, *, skip_empirical_check: bool, gpu_memory_utilization: float, base_snapshot_mode: str, ray_get: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Section 4's gate: `store_base_weights_via_rpc` (a second, GPU-resident full-model weight
    copy) is called EXACTLY ONCE, and ONLY when the empirical check is not skipped -- inventory
    /norm measurement alone (Sections 1-3, 5) never needs it. Returns None, without storing any
    base snapshot at all, when `skip_empirical_check` is True.
    """
    if skip_empirical_check:
        return None
    store_base_weights_via_rpc(engine, ray_get=ray_get)
    print(format_base_snapshot_confirmation(gpu_memory_utilization, base_snapshot_mode))
    return _run_empirical_norm_sanity_check(engine, ray_get=ray_get)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "stage7_anatomy_calibration" / "stage7_calibration_plan.json"))
    parser.add_argument("--skip-empirical-check", action="store_true", help="skip Section 4's single-perturbation GPU sanity check (and never stores a base snapshot)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "Stage 7A anatomy inventory + radius anchoring",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision")

    model_resolution = resolve_and_report_model_snapshot(cfg.model.name, cfg.model.revision)
    engine_config = build_stage6_engine_config()  # frozen: max_model_len=4096, gpu_memory_utilization=0.60, tensor_parallel_size=1, precision=bfloat16
    print(format_runtime_compatibility_diagnostic(
        model_resolution, worker_extension_cls="utils.worker_extn.WorkerExtension",
        vllm_version=get_vllm_version(), engine_mode=detect_vllm_engine_mode(), engine_config=engine_config,
    ))

    import ray

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore  # upstream, unmodified -- teardown only; launch uses OUR OWN launch_stage6_engine, never upstream launch_engines

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines = None
    pgs = None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)
        engines, pgs = launch_stage6_engine(
            model_resolution["resolved_snapshot_path"], precision=engine_config["precision"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"], max_model_len=engine_config["max_model_len"],
            tensor_parallel_size=engine_config["tensor_parallel_size"],
        )
        engine = engines[0]
        try:
            print("Building live anatomy atlas + upstream-scope inventory (Sections 1-2)...")
            combined = _rpc(engine, _report_anatomy_and_upstream_scope, args=(), label="_report_anatomy_and_upstream_scope")
            anatomy_inventory = combined["anatomy_inventory"]
            anatomy_inventory["model_revision"] = cfg.model.revision
            upstream_scope_inventory = combined["upstream_scope_inventory"]

            empirical_check = maybe_run_empirical_check(
                engine, skip_empirical_check=args.skip_empirical_check,
                gpu_memory_utilization=engine_config["gpu_memory_utilization"],
                base_snapshot_mode=engine_config["base_snapshot_mode"],
            )
        finally:
            cleanup_engines(engines, pgs)
    finally:
        if engines is None and ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    upstream_scope_scalar = upstream_scope_inventory["upstream_perturbed_scope"]
    language_region = anatomy_inventory["regions"]["language"]

    sigma_mapping_upstream = build_sigma_relative_l2_mapping(
        FROZEN_STAGE6_SIGMAS, upstream_scope_scalar["parameter_count"], upstream_scope_scalar["l2_norm"],
        scope_label="upstream_perturbed_scope",
    )
    sigma_mapping_language = build_sigma_relative_l2_mapping(
        FROZEN_STAGE6_SIGMAS, language_region["parameter_count"], language_region["l2_norm"],
        scope_label="language_l1_region",
    )

    proposed_radii = select_common_calibration_radii([row["r_hat"] for row in sigma_mapping_upstream])

    calibration_plan = {
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "engine_config": engine_config,
        "anatomy_inventory": anatomy_inventory,
        "upstream_scope_inventory": upstream_scope_inventory,
        "sigma_to_relative_l2_mapping": {
            "upstream_perturbed_scope": sigma_mapping_upstream,
            "language_l1_region": sigma_mapping_language,
            "note": (
                f"Every row is labeled {ANCHOR_LABEL!r} -- an analytical EXPECTED-NORM anchor "
                "(sigma * sqrt(d) / ||theta||_2), not a measured realized radius and not a "
                "final anatomical hyperparameter."
            ),
        },
        "empirical_norm_sanity_check": empirical_check,
        "proposed_initial_calibration_radii": {
            "regions": ["vision", "multimodal_connector_or_merger", "language"],
            "radii": proposed_radii,
            "derivation": (
                "Mechanically derived from the six translated Stage-6 sigma scales "
                "(upstream_perturbed_scope mapping), deduplicated -- never optimized against "
                "any capability score, and never a per-region or per-capability grid."
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration_plan, indent=2))
    print(f"Wrote {out_path}")

    upstream_scope_path = out_path.parent / "upstream_scope_inventory.json"
    upstream_scope_path.write_text(json.dumps(upstream_scope_inventory, indent=2))
    print(f"Wrote {upstream_scope_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
