"""LIVE 32B S2 (coarse anatomy) distributed v3 SOLVER readiness probe -- the S2-specific analog
of stage11_32b_live_v3_solver_probe.py (which proves G4/G5 for exactly ONE parameter subset --
multimodal_connector_or_merger). S2 perturbs THREE disjoint L1 regions (vision,
multimodal_connector_or_merger, language), each with different shard shapes/sizes -- a proof
gathered against one subset does not, on its own, establish the solver converges correctly on the
others. This script runs the REAL iterative bracket/bisection/plateau/quantization-limited solver
(scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed) against ALL THREE
frozen S2 regions, IN ONE live TP=4 engine session (the 32B model is loaded exactly once, never
per-region), and persists a single canonical multi-region artifact.

GPU required (4x L40S, TP=4). Writes ZERO scientific result rows -- readiness artifact only
(stage11_32b_s2_live_v3_solver_probe_report.json). Does NOT run the scientific 32B S2 smoke, S2
full, S1 whole_model, or 72B.

ACCEPTANCE CRITERION per region (identical to the S1 probe, never loosened): the solver enforces
its own scientific bar internally (radius_acceptance_mode="strict" within 1e-6, or explicitly
"quantization_limited" within 1e-3, itself re-verified bit-exactly) -- this script's job is to run
that unchanged solver against each region's REAL live sharded parameters and confirm every rank's
FULL bracket trajectory is IDENTICAL, then confirm exact restoration before moving to the next
region.

Usage (on the pod):
    python -m neural_thickets_repro.diagnostics.stage11_32b_s2_live_v3_solver_probe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # same live-discovered NVLink-less/cross-NUMA topology fix as the S1 G1-G8/solver probes

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

TP_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.60
MAX_MODEL_LEN = 4096

# Frozen mid radius -- IDENTICAL to the S1 solver probe's own PROBE_RADIUS, never invented,
# applied to every one of the three regions ("the same frozen diagnostic radius/seed policy").
PROBE_RADIUS = 0.017849414271899712
# A region-specific deterministic seed NAMESPACE (never the raw literal PROBE_SEED reused
# numerically) -- derived from one frozen root so each region's probe trial is independently
# reproducible and clearly attributable in the persisted artifact, while staying fully
# deterministic given the frozen root alone.
_S2_PROBE_SEED_ROOT = 9202608291  # the S1 probe's own PROBE_SEED, reused as the ROOT (not directly as a trial seed)


def _region_probe_seed(region: str) -> int:
    from neural_thickets_repro.thicket.seeds import derive_seed

    return derive_seed(_S2_PROBE_SEED_ROOT, "s2_live_v3_solver_probe", region)


def _dispatch(engine, fn, *, args: tuple = (), kwargs: Dict[str, Any] = None, label: str, ray_get) -> List[Any]:
    from neural_thickets_repro.thicket.distributed_perturbation import _validate_collective_rpc_results_multi_worker

    results = ray_get(engine.collective_rpc.remote(fn, args=args, kwargs=kwargs or {}))
    return _validate_collective_rpc_results_multi_worker(results, label=label, expected_world_size=TP_SIZE)


def _run_distributed_v3_solver_rpc(worker_self, seed: int, r: float, region_name: str, region_param_names: List[str]) -> Dict[str, Any]:
    """Worker-side RPC: the REAL Stage-11 candidate-lifecycle distributed v3 solver, dispatched
    identically to every TP rank, generalized (relative to the S1 probe's own hardcoded-region
    version) to accept ANY region name/param-name list -- reused for all three S2 regions in turn
    within the same engine session.
    """
    from neural_thickets_repro.thicket.distributed_v3_solver import (
        scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed,
    )
    from neural_thickets_repro.thicket.vllm_shard_mapping import build_shard_specs_for_region

    if not hasattr(worker_self, "_base_weights_cpu"):
        raise RuntimeError("_run_distributed_v3_solver_rpc: no _base_weights_cpu on this worker -- store_base_weights_cpu_rpc must be called first.")
    model = worker_self.model_runner.model
    shard_specs = build_shard_specs_for_region(model, region_param_names)
    result = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed(
        worker_self, seed, r, region_name, region_param_names, shard_specs,
    )
    result["rank"] = getattr(worker_self, "rank", None)
    return result


def _verify_full_bracket_trajectory_consensus(per_rank_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Byte-identical logic to the S1 probe's own helper (kept as its own copy here, not a shared
    import, so this module never depends on the S1 diagnostics script existing/being importable
    independently -- both are standalone pod-run diagnostics entry points).
    """
    if not per_rank_results:
        raise ValueError("_verify_full_bracket_trajectory_consensus requires at least one per-rank result.")
    first_attempts = per_rank_results[0].get("attempts")
    mismatched_ranks = [r.get("rank") for r in per_rank_results if r.get("attempts") != first_attempts]
    return {"ok": not mismatched_ranks, "n_ranks": len(per_rank_results), "mismatched_ranks": mismatched_ranks, "n_attempts": len(first_attempts) if first_attempts else 0}


def _probe_one_region(engine, ray_get, region: str, region_param_names: List[str]) -> Dict[str, Any]:
    """Runs the full solver-probe-then-restore sequence for ONE region against the ALREADY
    -RUNNING engine (no reload) -- returns the per-region artifact dict. Every region starts from
    a clean base (the previous region's own restore+verify step, or the initial canonical
    snapshot for the first region), so regions never interfere with each other's trial state.
    """
    from neural_thickets_repro.stage11_32b_readiness import GATE_FAIL, GATE_PASS
    from neural_thickets_repro.thicket.cpu_base_snapshot import reset_to_base_weights_cpu_rpc, verify_exact_fixed_base_restoration_cpu_rpc
    from neural_thickets_repro.thicket.distributed_perturbation import aggregate_distributed_restoration_verification
    from neural_thickets_repro.thicket.distributed_v3_solver import SolverRankConsensusError, verify_solver_rank_consensus

    seed = _region_probe_seed(region)
    info: Dict[str, Any] = {"region": region, "region_param_count": len(region_param_names), "probe_seed": seed, "probe_radius": PROBE_RADIUS}

    print(f"--- Probing region {region!r} ({len(region_param_names)} parameters), seed={seed} r={PROBE_RADIUS} ---")
    t0 = time.time()
    try:
        solver_results = _dispatch(engine, _run_distributed_v3_solver_rpc, args=(seed, PROBE_RADIUS, region, region_param_names), label=f"distributed_v3_solver_{region}", ray_get=ray_get)
        solver_error = None
    except Exception as exc:  # noqa: BLE001 -- a solver failure is the exact "STOP and report" case for this region
        solver_results = None
        solver_error = f"{type(exc).__name__}: {exc}"
        print(f"SOLVER FAILED for region {region!r}: {solver_error}", file=sys.stderr)
    info["solver_seconds"] = time.time() - t0
    info["solver_error"] = solver_error

    if solver_results is not None:
        print("Per-rank solver results:")
        for r in solver_results:
            print(f"  rank={r['rank']} accepted_scalar={r['accepted_scalar']} mode={r['radius_acceptance_mode']} "
                  f"realized={r['realized_relative_l2']} iterations={r['solver_iterations']} plateau={r['quantization_plateau']}")
        info["per_rank_solver_results"] = [{k: v for k, v in r.items() if k != "attempts"} for r in solver_results]

        try:
            core_consensus = verify_solver_rank_consensus(solver_results)
            core_consensus_ok, core_consensus_error = True, None
        except SolverRankConsensusError as exc:
            core_consensus, core_consensus_ok, core_consensus_error = None, False, str(exc)
        trajectory_consensus = _verify_full_bracket_trajectory_consensus(solver_results)
        rank_consensus_ok = core_consensus_ok and trajectory_consensus["ok"]
        info["rank_consensus"] = {"core_fields": core_consensus, "core_fields_ok": core_consensus_ok, "core_fields_error": core_consensus_error, "full_bracket_trajectory": trajectory_consensus}
        print(f"Region {region!r} rank consensus: core_fields={'OK' if core_consensus_ok else 'FAIL'} full_bracket_trajectory={'OK' if trajectory_consensus['ok'] else 'FAIL'}")

        accepted_mode = solver_results[0]["radius_acceptance_mode"]
        acceptance_valid = accepted_mode in ("strict", "quantization_limited")
        info["acceptance_mode"], info["acceptance_valid"] = accepted_mode, acceptance_valid
        info["accepted_scalar"] = solver_results[0]["accepted_scalar"]
        info["realized_relative_l2"] = solver_results[0]["realized_relative_l2"]
        info["solver_iterations"] = solver_results[0]["solver_iterations"]
        info["shard_identity_hash"] = solver_results[0].get("parameter_mask_hash") or solver_results[0].get("region_param_count")
    else:
        rank_consensus_ok, acceptance_valid = False, False
        info["rank_consensus"], info["acceptance_mode"], info["acceptance_valid"] = None, None, False

    # Restore + verify exact restoration for THIS region before moving to the next -- required
    # regardless of solver outcome (never leaves a failed region's partial state behind for the
    # next region's evaluate() calls to trip over).
    _dispatch(engine, reset_to_base_weights_cpu_rpc, label=f"reset_after_{region}_probe", ray_get=ray_get)
    restoration_raw = _dispatch(engine, verify_exact_fixed_base_restoration_cpu_rpc, kwargs={}, label=f"verify_restoration_{region}", ray_get=ray_get)
    restoration_aggregate = aggregate_distributed_restoration_verification(restoration_raw)
    info["restoration"] = restoration_aggregate
    print(f"Region {region!r} restoration: global_max_abs_drift={restoration_aggregate['global_max_abs_drift']} "
          f"any_rank_has_differing_elements={restoration_aggregate['any_rank_has_differing_elements']} ok={restoration_aggregate['ok']}")

    solver_ok = solver_results is not None and rank_consensus_ok and acceptance_valid
    region_g4_g5_pass = solver_ok and restoration_aggregate["ok"]
    info["g4_g5_final"] = {"G4": GATE_PASS if region_g4_g5_pass else GATE_FAIL, "G5": GATE_PASS if region_g4_g5_pass else GATE_FAIL}
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "stage11_32b_s2_live_readiness"))
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from neural_thickets_repro.stage11_32b_s2_live_evidence import S2_REGIONS

    report: Dict[str, Any] = {"probe": "distributed_v3_solver_g4_g5_strict_s2_multi_region", "regions_probed": list(S2_REGIONS), "tensor_parallel_size": TP_SIZE}

    import subprocess as sp

    commit = sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    print(f"repo commit: {commit}")
    report["repo_commit"] = commit

    from neural_thickets_repro.stage11_32b_live_evidence import query_live_gpu_uuids

    report["gpu_uuids"] = query_live_gpu_uuids()

    from neural_thickets_repro.env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module

    try:
        assert_feasible("32B S2 live v3 solver probe", [check_cuda(), check_module("vllm"), check_module("ray"), check_module("huggingface_hub"), check_disk(REPO_ROOT, 10.0)])
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    from neural_thickets_repro.scaling_common import get_scaling_model_spec, resolve_immutable_model_revision

    spec = get_scaling_model_spec("32B")
    resolution = resolve_immutable_model_revision(spec.model_name, spec.revision_ref)
    print(f"Resolved model revision: {resolution}")
    report["resolved_revision"] = resolution
    if resolution["resolved_revision"] != "7cfb30d71a1f4f49a57592323337a4a4727301da":
        print(f"WARNING: resolved revision {resolution['resolved_revision']} does not match the frozen 7cfb30d71a1f4f49a57592323337a4a4727301da.", file=sys.stderr)

    from neural_thickets_repro.run_global_visual_thicket_pilot import launch_stage6_engine
    from neural_thickets_repro.vlm_adapter import bootstrap_ray, resolve_model_snapshot, verify_workers_can_import_external_root

    resolved_snapshot_path = resolve_model_snapshot(spec.model_name, resolution["resolved_revision"])
    print(f"Resolved snapshot path: {resolved_snapshot_path}")

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)
    import ray

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        t0 = time.time()
        # ONE engine launch for the ENTIRE probe -- the 32B model is loaded exactly once here,
        # never per-region, per the explicit "do not reload the model once per region" requirement.
        engines, pgs = launch_stage6_engine(
            resolved_snapshot_path, precision="bfloat16", gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN, tensor_parallel_size=TP_SIZE, enable_prefix_caching=False,
        )
        engine = engines[0]

        from neural_thickets_repro.thicket.cpu_base_snapshot import store_base_weights_cpu_rpc

        _dispatch(engine, store_base_weights_cpu_rpc, kwargs={"pin_memory": True}, label="store_base_weights_cpu", ray_get=ray.get)
        engine_ready_seconds = time.time() - t0
        print(f"Engine ready + canonical CPU base snapshot stored on all ranks in {engine_ready_seconds:.1f}s")
        report["engine_ready_seconds"] = engine_ready_seconds

        from neural_thickets_repro.scaling_common import MODEL_FAMILY, report_region_param_names_for_scaling

        per_rank_region_info = _dispatch(engine, report_region_param_names_for_scaling, args=(tuple(S2_REGIONS), MODEL_FAMILY), label="report_region_param_names_for_scaling", ray_get=ray.get)
        mismatched_ranks = [r for r in per_rank_region_info if r != per_rank_region_info[0]]
        if mismatched_ranks:
            raise RuntimeError("Ranks disagree on report_region_param_names_for_scaling for S2 regions -- this is purely name-based and rank-independent, so any disagreement indicates a real bug.")
        region_info = per_rank_region_info[0]

        regions_report: Dict[str, Any] = {}
        for region in S2_REGIONS:
            param_names = region_info[region]["param_names"]
            regions_report[region] = _probe_one_region(engine, ray.get, region, param_names)
            regions_report[region]["shard_identity_hash"] = region_info[region]["mask_hash"]

        report["regions"] = regions_report
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    all_regions_pass = all(regions_report[r]["g4_g5_final"]["G4"] == "PASS" and regions_report[r]["g4_g5_final"]["G5"] == "PASS" for r in S2_REGIONS)
    report["all_regions_pass"] = all_regions_pass
    report["scientific_rows_written"] = 0  # this probe writes no candidate/result rows anywhere -- structurally true by construction

    report_path = output_dir / "stage11_32b_s2_live_v3_solver_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {report_path}")
    for region in S2_REGIONS:
        print(f"  {region}: {regions_report[region]['g4_g5_final']}")
    print(f"all_regions_pass: {all_regions_pass}")
    print("\nDO NOT RUN THE 32B S2 SCIENTIFIC SMOKE FROM THIS SCRIPT. This is a readiness probe only.")
    return 0 if all_regions_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
