"""LIVE 32B distributed v3 SOLVER readiness probe -- Stage-11 32B scaling, STRICT G4/G5
verification via the REAL iterative bracket/bisection/plateau/quantization-limited solver
(scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed), not the raw
one-shot apply_anatomical_relative_l2_distributed primitive the earlier G1-G8 readiness run used
to move G4/G5 off NOT_YET_VERIFIED. That earlier probe proved the distributed relative-L2
PRIMITIVE and real NCCL collectives work correctly; it explicitly does NOT exercise the actual
Stage-11 candidate-lifecycle function, so a corrected, ungrounded-tolerance-free PASS for G4/G5
requires running THIS solver, live, once.

GPU required (4x L40S, TP=4). Writes ZERO scientific result rows -- this is a readiness artifact
only (`stage11_32b_live_v3_solver_probe_report.json`), never a candidate row in any results
table. Does NOT run the scientific 32B smoke, 32B full, 32B anatomy, or 72B.

ACCEPTANCE CRITERION (never the earlier probe's 10% readiness sanity bound): the solver itself
already enforces the correct scientific bar internally -- it raises RadiusCorrectionFailedError/
QuantizationToleranceExceededError unless it converges `radius_acceptance_mode="strict"` (within
strict_tolerance=1e-6) or explicitly accepts `radius_acceptance_mode="quantization_limited"`
(within quantization_plateau_relative_tolerance=1e-3, itself already re-verified bit-exactly
inside the solver's own quantization-limited branch). This script's job is to run that unchanged
solver on real live TP=4 sharded 32B parameters and confirm every rank's trajectory is IDENTICAL
-- not to invent a separate, looser acceptance rule.

Usage (on the pod):
    python -m neural_thickets_repro.diagnostics.stage11_32b_live_v3_solver_probe
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # same live-discovered NVLink-less/cross-NUMA topology fix as the G1-G8 probe

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

TP_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.60
MAX_MODEL_LEN = 4096

# Fixed, clearly-labeled probe seed -- distinct from every seed used by the earlier G1-G8 probe
# or any real candidate stream.
PROBE_SEED = 9202608291
# Frozen mid radius (STAGE8_RADII[1]), never invented -- explicitly requested this task.
PROBE_RADIUS = 0.017849414271899712


def _connector_region_param_names_rpc(worker_self) -> List[str]:
    return sorted(n for n, _ in worker_self.model_runner.model.named_parameters() if "visual.merger." in n)


def _dispatch(engine, fn, *, args: tuple = (), kwargs: Dict[str, Any] = None, label: str, ray_get) -> List[Any]:
    from neural_thickets_repro.thicket.distributed_perturbation import _validate_collective_rpc_results_multi_worker

    results = ray_get(engine.collective_rpc.remote(fn, args=args, kwargs=kwargs or {}))
    return _validate_collective_rpc_results_multi_worker(results, label=label, expected_world_size=TP_SIZE)


def _run_distributed_v3_solver_rpc(worker_self, seed: int, r: float, region_param_names: List[str]) -> Dict[str, Any]:
    """Worker-side RPC: the REAL Stage-11 candidate-lifecycle distributed v3 solver, dispatched
    identically to every TP rank. Requires `worker_self._base_weights_cpu` (canonical CPU base
    snapshot) to already be stored -- the caller stores it explicitly, once, before this dispatch
    (never an implicit side effect here).
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
        worker_self, seed, r, "g4_g5_v3_solver_readiness_probe", region_param_names, shard_specs,
    )
    result["rank"] = getattr(worker_self, "rank", None)
    return result


def _verify_full_bracket_trajectory_consensus(per_rank_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stronger than distributed_v3_solver.verify_solver_rank_consensus's own 5-field check
    (accepted_scalar/radius_acceptance_mode/realized_relative_l2/solver_iterations/
    quantization_plateau) -- this task explicitly asks for the full BRACKET TRAJECTORY (every
    solver attempt's trial scalar and realized/designed radius, not just the final accepted
    values) to agree across ranks. `attempts` is the solver's own full per-iteration history
    (_build_quantization_aware_result's own field) -- compared here for exact equality, never
    approximately.
    """
    if not per_rank_results:
        raise ValueError("_verify_full_bracket_trajectory_consensus requires at least one per-rank result.")
    first_attempts = per_rank_results[0].get("attempts")
    mismatched_ranks = [r.get("rank") for r in per_rank_results if r.get("attempts") != first_attempts]
    return {"ok": not mismatched_ranks, "n_ranks": len(per_rank_results), "mismatched_ranks": mismatched_ranks, "n_attempts": len(first_attempts) if first_attempts else 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "stage11_32b_live_readiness"))
    parser.add_argument(
        "--existing-report", default=str(REPO_ROOT / "results" / "stage11_32b_live_readiness" / "stage11_32b_live_readiness_report.json"),
        help="The prior G1-G8 report (G1/G2/G3/G6/G7/G8 already PASS there) -- this probe's real G4/G5 result is merged into a copy of it for a combined smoke_permitted verdict; the ORIGINAL file is never overwritten by this script.",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"probe": "distributed_v3_solver_g4_g5_strict", "probe_seed": PROBE_SEED, "probe_radius": PROBE_RADIUS}

    import subprocess as sp

    commit = sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    print(f"repo commit: {commit}")
    report["repo_commit"] = commit

    from neural_thickets_repro.stage11_32b_live_evidence import query_live_gpu_uuids

    report["gpu_uuids"] = query_live_gpu_uuids()  # optional hardware-fingerprint binding evidence -- see that module's own docstring for the "if available" framing

    from neural_thickets_repro.env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module

    try:
        assert_feasible("32B live v3 solver probe", [check_cuda(), check_module("vllm"), check_module("ray"), check_module("huggingface_hub"), check_disk(REPO_ROOT, 10.0)])
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    from neural_thickets_repro.scaling_common import get_scaling_model_spec, resolve_immutable_model_revision
    from neural_thickets_repro.stage11_32b_readiness import (
        GATE_FAIL, GATE_PASS, Stage32BSmokeNotPermittedError, ensure_32b_smoke_permitted, g6_exact_restoration,
    )

    spec = get_scaling_model_spec("32B")
    resolution = resolve_immutable_model_revision(spec.model_name, spec.revision_ref)
    print(f"Resolved model revision: {resolution}")
    report["resolved_revision"] = resolution
    if resolution["resolved_revision"] != "7cfb30d71a1f4f49a57592323337a4a4727301da":
        print(f"WARNING: resolved revision {resolution['resolved_revision']} does not match the frozen 7cfb30d71a1f4f49a57592323337a4a4727301da used by the earlier G1-G8 run.", file=sys.stderr)

    from neural_thickets_repro.run_global_visual_thicket_pilot import launch_stage6_engine
    from neural_thickets_repro.vlm_adapter import bootstrap_ray, resolve_model_snapshot, verify_workers_can_import_external_root

    resolved_snapshot_path = resolve_model_snapshot(spec.model_name, resolution["resolved_revision"])
    print(f"Resolved snapshot path: {resolved_snapshot_path}")

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)
    import ray

    engines, pgs = None, None
    gate_results: Dict[str, str] = {}
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        t0 = time.time()
        engines, pgs = launch_stage6_engine(
            resolved_snapshot_path, precision="bfloat16", gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN, tensor_parallel_size=TP_SIZE, enable_prefix_caching=False,
        )
        engine = engines[0]

        # ---------------------------------------------------------------------------------------
        # Canonical CPU base snapshot -- explicit, once, before any solver trial (never an
        # implicit side effect). The first collective_rpc also forces the async Ray actor
        # __init__ (real weight load) to complete before we measure anything timing-sensitive.
        # ---------------------------------------------------------------------------------------
        from neural_thickets_repro.thicket.cpu_base_snapshot import (
            reset_to_base_weights_cpu_rpc, store_base_weights_cpu_rpc, verify_exact_fixed_base_restoration_cpu_rpc,
        )
        from neural_thickets_repro.thicket.distributed_perturbation import aggregate_distributed_restoration_verification

        _dispatch(engine, store_base_weights_cpu_rpc, kwargs={"pin_memory": True}, label="store_base_weights_cpu", ray_get=ray.get)
        engine_ready_seconds = time.time() - t0
        print(f"Engine ready + canonical CPU base snapshot stored on all ranks in {engine_ready_seconds:.1f}s")
        report["engine_ready_seconds"] = engine_ready_seconds

        region_param_names_per_rank = _dispatch(engine, _connector_region_param_names_rpc, label="region_param_names", ray_get=ray.get)
        region_param_names = region_param_names_per_rank[0]
        assert all(names == region_param_names for names in region_param_names_per_rank), "ranks disagree on connector region parameter names"
        print(f"Probe region: multimodal_connector_or_merger ({len(region_param_names)} parameters)")
        report["region"] = "multimodal_connector_or_merger"
        report["region_param_count"] = len(region_param_names)

        # ---------------------------------------------------------------------------------------
        # THE REAL SOLVER -- run once, dispatched identically to all 4 ranks.
        # ---------------------------------------------------------------------------------------
        print(f"Running the REAL distributed v3 solver: seed={PROBE_SEED} r={PROBE_RADIUS} ...")
        t_solver = time.time()
        try:
            solver_results = _dispatch(
                engine, _run_distributed_v3_solver_rpc, args=(PROBE_SEED, PROBE_RADIUS, region_param_names),
                label="distributed_v3_solver", ray_get=ray.get,
            )
            solver_error = None
        except Exception as exc:  # noqa: BLE001 -- a solver failure (RadiusCorrectionFailedError, QuantizationToleranceExceededError, consensus errors surfaced by vLLM's own RPC error propagation) is the exact "if it fails: STOP and report exact failure" case
            solver_results = None
            solver_error = f"{type(exc).__name__}: {exc}"
            print(f"SOLVER FAILED: {solver_error}", file=sys.stderr)
        solver_seconds = time.time() - t_solver
        report["solver_seconds"] = solver_seconds
        report["solver_error"] = solver_error

        if solver_results is not None:
            from neural_thickets_repro.thicket.distributed_v3_solver import verify_solver_rank_consensus, SolverRankConsensusError

            print("Per-rank solver results:")
            for r in solver_results:
                print(f"  rank={r['rank']} accepted_scalar={r['accepted_scalar']} mode={r['radius_acceptance_mode']} "
                      f"realized={r['realized_relative_l2']} iterations={r['solver_iterations']} plateau={r['quantization_plateau']}")
            report["per_rank_solver_results"] = [
                {k: v for k, v in r.items() if k != "attempts"} for r in solver_results  # attempts persisted separately below (can be large)
            ]
            (output_dir / "v3_solver_probe_full_attempts.json").write_text(json.dumps([r.get("attempts") for r in solver_results], indent=2, default=str))

            try:
                core_consensus = verify_solver_rank_consensus(solver_results)
                core_consensus_ok = True
                core_consensus_error = None
            except SolverRankConsensusError as exc:
                core_consensus = None
                core_consensus_ok = False
                core_consensus_error = str(exc)

            trajectory_consensus = _verify_full_bracket_trajectory_consensus(solver_results)
            rank_consensus_ok = core_consensus_ok and trajectory_consensus["ok"]
            report["rank_consensus"] = {"core_fields": core_consensus, "core_fields_ok": core_consensus_ok, "core_fields_error": core_consensus_error, "full_bracket_trajectory": trajectory_consensus}
            print(f"Rank consensus (core fields): {'OK' if core_consensus_ok else 'FAIL -- ' + str(core_consensus_error)}")
            print(f"Rank consensus (full bracket trajectory, {trajectory_consensus['n_attempts']} attempts): {'OK' if trajectory_consensus['ok'] else 'FAIL -- mismatched ranks: ' + str(trajectory_consensus['mismatched_ranks'])}")

            accepted_mode = solver_results[0]["radius_acceptance_mode"]
            acceptance_valid = accepted_mode in ("strict", "quantization_limited")
            print(f"Acceptance mode: {accepted_mode!r} (valid={acceptance_valid}) -- enforced by the solver's OWN strict_tolerance=1e-6 / quantization_plateau_relative_tolerance=1e-3, no looser bound applied here.")
            report["acceptance_mode"], report["acceptance_valid"] = accepted_mode, acceptance_valid
        else:
            rank_consensus_ok, acceptance_valid = False, False
            report["rank_consensus"], report["acceptance_mode"], report["acceptance_valid"] = None, None, False

        # ---------------------------------------------------------------------------------------
        # Restore + verify exact global restoration -- required regardless of solver outcome.
        # ---------------------------------------------------------------------------------------
        _dispatch(engine, reset_to_base_weights_cpu_rpc, label="reset_after_solver_probe", ray_get=ray.get)
        restoration_raw = _dispatch(engine, verify_exact_fixed_base_restoration_cpu_rpc, kwargs={}, label="verify_restoration", ray_get=ray.get)
        restoration_aggregate = aggregate_distributed_restoration_verification(restoration_raw)
        report["restoration"] = restoration_aggregate
        print(f"Restoration: global_max_abs_drift={restoration_aggregate['global_max_abs_drift']} any_rank_has_differing_elements={restoration_aggregate['any_rank_has_differing_elements']} ok={restoration_aggregate['ok']}")

        solver_ok = solver_results is not None and rank_consensus_ok and acceptance_valid
        g4_g5_pass = solver_ok and restoration_aggregate["ok"]
        gate_results["G4"] = GATE_PASS if g4_g5_pass else GATE_FAIL
        gate_results["G5"] = GATE_PASS if g4_g5_pass else GATE_FAIL
        report["g4_g5_final"] = gate_results

    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    # -------------------------------------------------------------------------------------------
    # Merge into a COPY of the prior G1-G8 report for a combined smoke_permitted verdict -- the
    # original file (G1/G2/G3/G6/G7/G8 already real-PASS there) is never overwritten in place.
    # -------------------------------------------------------------------------------------------
    existing_path = Path(args.existing_report)
    combined_gate_results: Dict[str, str] = {}
    if existing_path.exists():
        prior = json.loads(existing_path.read_text())
        combined_gate_results.update(prior.get("gate_results", {}))
    combined_gate_results.update(gate_results)
    report["combined_gate_results"] = combined_gate_results
    try:
        ensure_32b_smoke_permitted(combined_gate_results)
        smoke_permitted = True
    except Stage32BSmokeNotPermittedError as exc:
        smoke_permitted = False
        report["smoke_not_permitted_reason"] = str(exc)
    report["smoke_permitted"] = smoke_permitted
    report["scientific_rows_written"] = 0  # this probe writes no candidate/result rows anywhere -- structurally true by construction (no results.jsonl/append_candidate_rows call exists in this file)

    report_path = output_dir / "stage11_32b_live_v3_solver_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {report_path}")
    print(f"G4/G5 (this probe): {gate_results}")
    print(f"Combined G1-G8: {combined_gate_results}")
    print(f"smoke_permitted: {smoke_permitted}")
    print("\nDO NOT RUN THE 32B SCIENTIFIC SMOKE FROM THIS SCRIPT. This is a readiness probe only.")
    return 0 if smoke_permitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
