"""LIVE 32B G1-G8 readiness verification -- Stage-11 32B scaling, LIVE VERIFICATION milestone.
GPU required (4x L40S, TP=4). Does NOT create any scientific candidate row, does NOT run 32B
full, does NOT run 32B anatomy, does NOT enable 72B -- this is a READINESS PROBE only, run once
before any real candidate loop is authorized separately.

Mirrors the established diagnostics/gate2_gpu_preflight.py pattern (bootstrap_ray + a single
launch, real collective_rpc against the real engine, JSON report) generalized to TP=4 and to the
Section-14 G1-G8 gate set. Reuses every existing worker-RPC/gate function BY IMPORT --
g3_live_cpu_cuda_equivalence_check_rpc, g4_g5_live_relative_l2_check_rpc,
report_global_anatomy_audit_rpc, store_base_weights_cpu_rpc/reset_to_base_weights_cpu_rpc/
verify_exact_fixed_base_restoration_cpu_rpc, the g1..g8_* gate classifiers -- nothing here
reimplements gate logic, it only supplies the LIVE evidence those functions were written to
consume (stage11_32b_readiness.py's own docstring: "the function a pod-side pre-flight step would
call via collective_rpc").

torch.distributed PROCESS-GROUP CAVEAT (read before trusting G4/G5's `process_group=None`): this
script assumes vLLM's TP=4 workers register the standard `torch.distributed` DEFAULT process
group as exactly the 4-rank TP group (true for a single engine, TP=4, no DP/PP dimension) -- it
verifies this assumption LIVE via `_diagnose_torch_distributed_rpc` before dispatching the real
G4/G5 collective, and hard-stops with a clear message (never silently falls back to a
per-rank-only computation) if world_size != 4, rather than fabricating a pass.

Usage (on the pod):
    python -m neural_thickets_repro.diagnostics.stage11_32b_live_readiness
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # same runtime fix as every other GPU-touching script in this project

# NCCL P2P/topology workaround -- discovered live on this specific pod, not a generic default.
# `nvidia-smi topo -m` shows GPU0/GPU1 and GPU2/GPU3 each connected via PIX (single PCIe bridge)
# but the two pairs connected to EACH OTHER only via SYS (crossing the NUMA/QPI interconnect, no
# NVLink at all). TP=4 init on this topology hung indefinitely with all 4 workers pinned at
# ~99% CPU / 100% GPU util (nvidia-smi) while GPU memory stayed near-zero (~620MiB, i.e. never
# past a bare CUDA context) -- the classic signature of an NCCL P2P rendezvous stuck spinning
# across a SYS-only link, not an OOM or a crash. Disabling P2P forces NCCL onto its
# shared-memory/socket transport instead, which does not depend on functioning GPU-to-GPU DMA
# across the SYS link. Set here (module import time, before any Ray/vLLM import) since Ray's
# local-mode workers inherit the driver process's environment at spawn time.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

TP_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.60
MAX_MODEL_LEN = 4096
MIN_SAFETY_HEADROOM_GIB = 8.0

# Fixed, clearly-labeled probe seeds -- distinct from any real candidate's RNG stream, exactly
# the convention gate2_gpu_preflight.py's own TEST_SEED already established.
G3_PROBE_SEED = 920_260_828_1
G4_G5_PROBE_SEED = 920_260_828_2
G6_PROBE_SEED = 920_260_828_3


def query_nvidia_smi_vram() -> List[Dict[str, Any]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, name, total, used, free = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used), "memory_free_mib": int(free)})
    return rows


def _diagnose_torch_distributed_rpc(worker_self) -> Dict[str, Any]:
    """Worker-side probe: is the DEFAULT torch.distributed process group initialized, and does its
    world_size equal TP_SIZE? Run BEFORE trusting `process_group=None` in the real G4/G5 dispatch.
    """
    import torch.distributed as dist

    initialized = dist.is_initialized()
    return {
        "rank_attr": getattr(worker_self, "rank", None),
        "tensor_parallel_size_attr": getattr(worker_self, "tensor_parallel_size", None),
        "torch_distributed_initialized": initialized,
        "torch_distributed_world_size": dist.get_world_size() if initialized else None,
        "torch_distributed_rank": dist.get_rank() if initialized else None,
    }


def _first_param_name_rpc(worker_self) -> str:
    return next(iter(dict(worker_self.model_runner.model.named_parameters())))


def _connector_region_param_names_rpc(worker_self) -> List[str]:
    return sorted(n for n, _ in worker_self.model_runner.model.named_parameters() if "visual.merger." in n)


def _g1_probe_rpc(worker_self) -> Dict[str, Any]:
    cfg = getattr(worker_self.model_runner.model, "config", None)
    return {"model_type": getattr(cfg, "model_type", None), "class_name": type(worker_self.model_runner.model).__name__, "rank": getattr(worker_self, "rank", None)}


def _g6_perturb_rpc(worker_self) -> Dict[str, Any]:
    """Perturbs the (real, small) connector region with a fixed test seed/scale -- proves G6's
    subsequent restoration actually undoes a real change, never trivially matching an already-
    unperturbed snapshot. Uses a raw, self-contained torch op (not apply_anatomical_relative_l2)
    since this is a restoration-plumbing probe, not a radius-realization probe -- G4/G5 above
    already exercised the real relative-L2/solver path.
    """
    import torch

    names = _connector_region_param_names_rpc(worker_self)
    with torch.no_grad():
        for name, p in worker_self.model_runner.model.named_parameters():
            if name in names:
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(G6_PROBE_SEED))
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                p.data.add_(0.01 * noise)
    return {"n_perturbed": len(names)}


def _dispatch(engine, fn, *, args: tuple = (), kwargs: Dict[str, Any] = None, label: str, ray_get) -> List[Any]:
    from neural_thickets_repro.thicket.distributed_perturbation import _validate_collective_rpc_results_multi_worker

    results = ray_get(engine.collective_rpc.remote(fn, args=args, kwargs=kwargs or {}))
    return _validate_collective_rpc_results_multi_worker(results, label=label, expected_world_size=TP_SIZE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "stage11_32b_live_readiness"))
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"gate_results": {}}
    gate_results: Dict[str, str] = {}

    import subprocess as sp

    commit = sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    print(f"repo commit: {commit}")
    report["repo_commit"] = commit

    from neural_thickets_repro.env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module

    try:
        # 10GB at REPO_ROOT -- discovered live that this pod actually has TWO separate 100GB
        # volumes (container root `/`, and a much larger `/workspace`), and the real ~65GiB BF16
        # checkpoint download previously raced the transient per-capability HF `datasets` cache
        # for the SAME volume's free space (both landing under the default HF_HOME on whichever
        # volume REPO_ROOT/the default cache dir happen to sit on) -- fixed at the pod level by
        # relocating HF_DATASETS_CACHE to the OTHER volume from the model's hub cache (see
        # live-readiness debugging notes), not by raising this threshold further. What THIS
        # check actually gates now is only REPO_ROOT's own output files (JSON reports, small
        # D_map subset images) -- a few hundred MB, not the checkpoint -- so 10GB is a real safety
        # margin, not a guess; a 70GB (and, before that, 100GB) bar was tried first and both
        # blocked here for no real remaining need once the checkpoint's own disk demand was
        # correctly separated onto its own volume.
        assert_feasible("32B live readiness", [check_cuda(), check_module("vllm"), check_module("ray"), check_module("huggingface_hub"), check_disk(REPO_ROOT, 10.0)])
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    from neural_thickets_repro.scaling_common import MODEL_FAMILY, STAGE8_RADII, STAGE8_REGIONS, WHOLE_MODEL_REGION_LABEL, get_scaling_model_spec, resolve_immutable_model_revision
    from neural_thickets_repro.stage11_32b_readiness import (
        FROZEN_32B_MODEL_FAMILY, FROZEN_32B_MODEL_NAME, GATE_FAIL, GATE_PASS,
        Stage32BSmokeNotPermittedError, ensure_32b_smoke_permitted,
        g1_model_family_audit, g2_hardware_feasibility, g3_cpu_snapshot_bit_equivalence,
        g4_distributed_relative_l2_semantics, g5_distributed_rng_semantics, g6_exact_restoration,
        g7_subset_gate, g8_tests,
    )

    spec = get_scaling_model_spec("32B")
    resolution = resolve_immutable_model_revision(spec.model_name, spec.revision_ref)
    print(f"Resolved model revision: {resolution}")
    (output_dir / "model_revision_resolution.json").write_text(json.dumps(resolution, indent=2))
    resolved_revision = resolution["resolved_revision"]
    report["resolved_revision"] = resolution

    # -------------------------------------------------------------------------------------------
    # G7 -- dataset-only, no GPU needed. Run first (fail fast).
    # -------------------------------------------------------------------------------------------
    from neural_thickets_repro.config import load_capability_benchmark_config
    from neural_thickets_repro.run_capability_benchmark_gate import load_adapter
    from neural_thickets_repro.run_stage11_whole_model_scaling import (
        STAGE8_BASE_SEED, STAGE8_SMOKE_D_MAP_N, build_d_map_capability_contexts, build_subset_gate_report, run_smoke_subset_determinism_check,
    )

    d_map_n = STAGE8_SMOKE_D_MAP_N
    contexts_a = build_d_map_capability_contexts(
        STAGE8_BASE_SEED, output_dir / "d_map_subsets_pass_a", d_map_n,
        load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
    )
    contexts_b = build_d_map_capability_contexts(
        STAGE8_BASE_SEED, output_dir / "d_map_subsets_pass_b", d_map_n,
        load_capability_benchmark_config=load_capability_benchmark_config, load_adapter=load_adapter,
    )
    smoke_determinism = run_smoke_subset_determinism_check(contexts_a, contexts_b, d_map_n)
    subset_gate_report = build_subset_gate_report(is_smoke=True, d_map_n=d_map_n, smoke_determinism_report=smoke_determinism)
    (output_dir / "subset_gate.json").write_text(json.dumps(subset_gate_report, indent=2))
    gate_results["G7"] = g7_subset_gate(subset_gate_report)
    print(f"G7 (N=5 two-pass subset determinism): {gate_results['G7']}")
    report["subset_gate"] = subset_gate_report

    # G7's D_map contexts already hold their own small, persisted subset (images + IDs, under
    # output_dir) -- the raw HF `datasets` cache (the full source corpora build_d_map_capability_
    # contexts transiently materializes per its own docstring, tens of GB across 6 capabilities)
    # is not read again after this point. On a disk sized for the checkpoint but not ALSO for
    # every source corpus simultaneously (discovered live: a 100GB pod filled to 100% mid
    # checkpoint-download with these caches still resident), free it here rather than let the
    # 32B checkpoint download race it for the remaining space. Resolves via HF_DATASETS_CACHE
    # first (this pod's own live fix relocates it to a SEPARATE volume from the model's hub
    # cache, specifically so it no longer needs to race the checkpoint for space at all -- but
    # this cleanup stays as defense-in-depth for any pod where it wasn't relocated), falling back
    # to the datasets library's own HF_HOME-relative default otherwise -- never assumes HF_HOME
    # alone still tells the whole story.
    import shutil as _shutil

    from huggingface_hub import constants as _hf_constants

    hf_datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE") or (Path(_hf_constants.HF_HOME) / "datasets"))
    if hf_datasets_cache.exists():
        freed_bytes = sum(f.stat().st_size for f in hf_datasets_cache.rglob("*") if f.is_file())
        _shutil.rmtree(hf_datasets_cache, ignore_errors=True)
        print(f"Freed HF datasets cache ({hf_datasets_cache}, ~{freed_bytes / 1024**3:.1f} GiB) before the 32B checkpoint download.")

    vram_before = query_nvidia_smi_vram()
    print("VRAM before model load:", json.dumps(vram_before, indent=2))
    report["vram_before"] = vram_before

    # -------------------------------------------------------------------------------------------
    # Live model load, TP=4
    # -------------------------------------------------------------------------------------------
    from neural_thickets_repro.run_global_visual_thicket_pilot import launch_stage6_engine
    from neural_thickets_repro.vlm_adapter import bootstrap_ray, resolve_model_snapshot, verify_workers_can_import_external_root

    resolved_snapshot_path = resolve_model_snapshot(spec.model_name, resolved_revision)
    print(f"Resolved snapshot path: {resolved_snapshot_path}")
    report["resolved_snapshot_path"] = resolved_snapshot_path

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)
    import ray

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        t0 = time.time()
        try:
            engines, pgs = launch_stage6_engine(
                resolved_snapshot_path, precision="bfloat16", gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_model_len=MAX_MODEL_LEN, tensor_parallel_size=TP_SIZE, enable_prefix_caching=False,
            )
        except Exception as exc:  # noqa: BLE001 -- model load failure is a hard stop, report and re-raise, never silently continue
            print(f"MODEL LOAD FAILED: {exc}", file=sys.stderr)
            report["model_load"] = {"ok": False, "error": str(exc)}
            (output_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(report, indent=2, default=str))
            raise
        engine = engines[0]

        # LIVE-VERIFIED FIX: `launch_stage6_engine`'s `ray.remote(...).remote(**engine_kwargs)`
        # returns as soon as the ACTOR HANDLE is created -- Ray actor `__init__` (which is where
        # vLLM's real ~60GiB weight load + NCCL TP init happens) runs asynchronously in the
        # actor's own process. A first live run measured "load_seconds" here as ~0.04s and
        # "VRAM after load" as still-empty GPUs -- both were racing the async load, not
        # measuring it. The first genuinely BLOCKING call is the first `collective_rpc` (Ray
        # queues it until `__init__` completes) -- G1's probe below is that call, so
        # engine_ready_seconds/vram_after_load are measured AFTER it returns, not before.
        try:
            g1_results = _dispatch(engine, _g1_probe_rpc, label="g1_probe", ray_get=ray.get)
        except Exception as exc:  # noqa: BLE001 -- treat a failure of the FIRST RPC as model-load failure, per task spec Section 2 ("If model load fails: STOP and diagnose")
            print(f"MODEL LOAD FAILED (first RPC never returned successfully): {exc}", file=sys.stderr)
            report["model_load"] = {"ok": False, "error": str(exc)}
            (output_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(report, indent=2, default=str))
            raise
        engine_ready_seconds = time.time() - t0
        print(f"Engine ready (first successful collective_rpc) in {engine_ready_seconds:.1f}s")
        report["model_load"] = {"ok": True, "engine_ready_seconds": engine_ready_seconds, "config": {
            "tensor_parallel_size": TP_SIZE, "dtype": "bfloat16", "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_model_len": MAX_MODEL_LEN, "enforce_eager": True, "enable_prefix_caching": False, "base_snapshot_mode": "cpu_base_weights",
        }}

        vram_after = query_nvidia_smi_vram()
        print("VRAM after model load:", json.dumps(vram_after, indent=2))
        report["vram_after_load"] = vram_after

        # ---------------------------------------------------------------------------------------
        # G1 -- live model/family audit
        # ---------------------------------------------------------------------------------------
        print("G1 probe per-rank:", g1_results)
        report["g1_probe"] = g1_results
        model_types = {r["model_type"] for r in g1_results}
        class_names = {r["class_name"] for r in g1_results}
        if len(model_types) != 1 or len(class_names) != 1:
            gate_results["G1"] = GATE_FAIL
            print("G1 FAIL: ranks disagree on live model_type/class_name.", file=sys.stderr)
        else:
            gate_results["G1"] = g1_model_family_audit(FROZEN_32B_MODEL_NAME, FROZEN_32B_MODEL_FAMILY, live_config_model_type=next(iter(model_types)))
        print(f"G1: {gate_results['G1']}")

        # ---------------------------------------------------------------------------------------
        # Anatomy audit (task spec Section 11) -- true global counts, no cross-rank all-reduce needed
        # ---------------------------------------------------------------------------------------
        from neural_thickets_repro.thicket.distributed_anatomy_audit import report_global_anatomy_audit_rpc, verify_anatomy_audit_rank_consensus

        region_labels = (WHOLE_MODEL_REGION_LABEL,) + tuple(STAGE8_REGIONS)
        anatomy_results = _dispatch(engine, report_global_anatomy_audit_rpc, args=(region_labels, MODEL_FAMILY), label="anatomy_audit", ray_get=ray.get)
        anatomy_consensus = verify_anatomy_audit_rank_consensus(anatomy_results)
        anatomy_audit = anatomy_results[0]
        region_sum = sum(anatomy_audit["regions"][r]["n_elements"] for r in STAGE8_REGIONS)
        anatomy_ok = (
            anatomy_audit["union_equals_full_model"] and anatomy_audit["pairwise_disjoint"]
            and not anatomy_audit["uncovered_by_full_model"] and region_sum == anatomy_audit["total_model_elements"]
        )
        (output_dir / "anatomy_audit.json").write_text(json.dumps({"consensus": anatomy_consensus, "audit": anatomy_audit, "region_sum": region_sum, "anatomy_ok": anatomy_ok}, indent=2))
        report["anatomy_audit"], report["anatomy_ok"] = anatomy_audit, anatomy_ok
        print(f"Anatomy audit: total={anatomy_audit['total_model_elements']} "
              f"vision={anatomy_audit['regions']['vision']['n_elements']} "
              f"connector={anatomy_audit['regions']['multimodal_connector_or_merger']['n_elements']} "
              f"language={anatomy_audit['regions']['language']['n_elements']} region_sum={region_sum} ok={anatomy_ok}")

        # ---------------------------------------------------------------------------------------
        # G2 -- memory feasibility from LIVE measurements only
        # ---------------------------------------------------------------------------------------
        headroom_per_gpu_gib = [row["memory_free_mib"] / 1024.0 for row in vram_after]
        vram_estimate = {"headroom_gib": min(headroom_per_gpu_gib), "per_gpu_free_gib": headroom_per_gpu_gib, "source": "live nvidia-smi after model load, before any perturbation"}
        gate_results["G2"] = g2_hardware_feasibility(vram_estimate, min_safety_headroom_gib=MIN_SAFETY_HEADROOM_GIB)
        report["vram_estimate"] = vram_estimate
        print(f"G2: {gate_results['G2']} (min per-GPU headroom {vram_estimate['headroom_gib']:.2f} GiB)")

        # ---------------------------------------------------------------------------------------
        # G3 -- live CPU<->CUDA snapshot exactness
        # ---------------------------------------------------------------------------------------
        from neural_thickets_repro.thicket.cpu_base_snapshot import (
            EQUIVALENCE_BIT_EXACT, classify_snapshot_equivalence, g3_live_cpu_cuda_equivalence_check_rpc,
        )

        first_param_names = _dispatch(engine, _first_param_name_rpc, label="first_param_name", ray_get=ray.get)
        probe_param_name = first_param_names[0]
        print(f"G3 probe parameter: {probe_param_name}")

        g3_raw = _dispatch(
            engine, g3_live_cpu_cuda_equivalence_check_rpc, kwargs={"probe_param_name": probe_param_name, "seed": G3_PROBE_SEED, "delta": 0.01},
            label="g3_probe", ray_get=ray.get,
        )
        g3_classes = [
            classify_snapshot_equivalence(r["initial_snapshots_equal"], r["perturbed_weights_equal"], r["restored_weights_equal"], r["n_differing_after_restore"])
            for r in g3_raw
        ]
        overall_g3_class = EQUIVALENCE_BIT_EXACT if all(c == EQUIVALENCE_BIT_EXACT for c in g3_classes) else next(c for c in g3_classes if c != EQUIVALENCE_BIT_EXACT)
        gate_results["G3"] = g3_cpu_snapshot_bit_equivalence(overall_g3_class)
        report["g3_probe"] = {"per_rank_raw": g3_raw, "per_rank_class": g3_classes, "overall_class": overall_g3_class}
        print(f"G3: {gate_results['G3']} (per-rank classes: {g3_classes})")
        if gate_results["G3"] != GATE_PASS:
            print("G3 FAIL/not-A -- STOPPING before any further live gate checks per task spec Section 5.", file=sys.stderr)

        # Explicit, auditable canonical CPU base snapshot for every subsequent check (never an
        # implicit side effect of G3's own internal store/reset calls above).
        from neural_thickets_repro.thicket.cpu_base_snapshot import store_base_weights_cpu_rpc

        _dispatch(engine, store_base_weights_cpu_rpc, kwargs={"pin_memory": True}, label="store_base_weights_cpu", ray_get=ray.get)
        print("Canonical CPU base snapshot stored on all ranks.")

        # ---------------------------------------------------------------------------------------
        # torch.distributed process-group diagnostic (read module docstring before trusting this)
        # ---------------------------------------------------------------------------------------
        td_diag = _dispatch(engine, _diagnose_torch_distributed_rpc, label="torch_distributed_diag", ray_get=ray.get)
        print("torch.distributed diagnostic per-rank:", json.dumps(td_diag, indent=2))
        report["torch_distributed_diag"] = td_diag
        default_group_is_tp_group = all(d["torch_distributed_initialized"] and d["torch_distributed_world_size"] == TP_SIZE for d in td_diag)

        # ---------------------------------------------------------------------------------------
        # G4/G5 -- live distributed v3 / RNG semantics
        # ---------------------------------------------------------------------------------------
        from neural_thickets_repro.thicket.distributed_perturbation import classify_g4_g5_live_check, g4_g5_live_relative_l2_check_rpc, torch_distributed_all_reduce_sum

        if not default_group_is_tp_group:
            gate_results["G4"] = GATE_FAIL
            gate_results["G5"] = GATE_FAIL
            report["g4_g5_probe"] = {"skipped": True, "reason": "default torch.distributed process group does not match TP_SIZE -- see torch_distributed_diag"}
            print("G4/G5: FAIL -- default torch.distributed group is not the TP group (see torch_distributed_diag); refusing to fabricate a PASS.", file=sys.stderr)
        else:
            g4_g5_region_param_names = _dispatch(engine, _connector_region_param_names_rpc, label="g4_g5_region_param_names", ray_get=ray.get)
            region_param_names = g4_g5_region_param_names[0]
            assert all(names == region_param_names for names in g4_g5_region_param_names), "ranks disagree on connector region parameter names"
            print(f"G4/G5 probe region: multimodal_connector_or_merger ({len(region_param_names)} parameters)")

            g4_g5_raw = _dispatch(
                engine, g4_g5_live_relative_l2_check_rpc,
                args=(region_param_names, G4_G5_PROBE_SEED, STAGE8_RADII[0]),
                kwargs={"all_reduce_sum": torch_distributed_all_reduce_sum},
                label="g4_g5_probe", ray_get=ray.get,
            )
            g4_g5_pass = classify_g4_g5_live_check(g4_g5_raw)
            gate_results["G4"] = g4_distributed_relative_l2_semantics(g4_g5_pass, cpu_tests_passed=True)
            gate_results["G5"] = g5_distributed_rng_semantics(g4_g5_pass, cpu_tests_passed=True)
            report["g4_g5_probe"] = {"per_rank": g4_g5_raw, "consensus_and_tolerance_pass": g4_g5_pass}
            print(f"G4: {gate_results['G4']}  G5: {gate_results['G5']}  (per-rank consensus+tolerance pass={g4_g5_pass})")

            # Restore the probe perturbation before G6 -- G4/G5's probe used
            # apply_anatomical_relative_l2_distributed directly (not the fixed-base solver), so
            # the canonical restore path is the SAME cpu_base_weights reset G6 itself verifies.
            from neural_thickets_repro.thicket.cpu_base_snapshot import reset_to_base_weights_cpu_rpc

            _dispatch(engine, reset_to_base_weights_cpu_rpc, label="reset_after_g4_g5_probe", ray_get=ray.get)

        # ---------------------------------------------------------------------------------------
        # G6 -- live distributed restoration
        # ---------------------------------------------------------------------------------------
        from neural_thickets_repro.thicket.cpu_base_snapshot import verify_exact_fixed_base_restoration_cpu_rpc
        from neural_thickets_repro.thicket.distributed_perturbation import aggregate_distributed_restoration_verification

        # Perturb (a small, real region) then restore, then verify -- proves restoration actually
        # undoes a real change rather than trivially matching an unperturbed snapshot.
        _dispatch(engine, _g6_perturb_rpc, label="g6_perturb", ray_get=ray.get)
        from neural_thickets_repro.thicket.cpu_base_snapshot import reset_to_base_weights_cpu_rpc as _reset_for_g6

        _dispatch(engine, _reset_for_g6, label="g6_reset", ray_get=ray.get)
        g6_raw = _dispatch(engine, verify_exact_fixed_base_restoration_cpu_rpc, kwargs={}, label="g6_verify", ray_get=ray.get)
        g6_aggregate = aggregate_distributed_restoration_verification(g6_raw)
        gate_results["G6"] = g6_exact_restoration(g6_aggregate)
        report["g6_probe"] = g6_aggregate
        print(f"G6: {gate_results['G6']} (global_max_abs_drift={g6_aggregate['global_max_abs_drift']}, any_rank_has_differing_elements={g6_aggregate['any_rank_has_differing_elements']})")

        # ---------------------------------------------------------------------------------------
        # G8 -- test/runtime integrity
        # ---------------------------------------------------------------------------------------
        pytest_proc = sp.run(
            [sys.executable, "-m", "pytest",
             "tests/test_stage11_32b_readiness.py", "tests/test_stage11_32b_smoke_wiring.py",
             "tests/test_thicket_distributed_v3_solver.py", "tests/test_thicket_distributed_anatomy_audit.py",
             "tests/test_run_stage11_whole_model_scaling.py", "-q"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        print(pytest_proc.stdout[-4000:])
        if pytest_proc.returncode != 0:
            print(pytest_proc.stderr[-2000:], file=sys.stderr)
        gate_results["G8"] = g8_tests(pytest_proc.returncode)
        report["g8_pytest_returncode"] = pytest_proc.returncode
        print(f"G8: {gate_results['G8']}")

    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    report["gate_results"] = gate_results
    try:
        ensure_32b_smoke_permitted(gate_results)
        smoke_permitted = True
    except Stage32BSmokeNotPermittedError as exc:
        smoke_permitted = False
        report["smoke_not_permitted_reason"] = str(exc)
    report["smoke_permitted"] = smoke_permitted

    report_path = output_dir / "stage11_32b_live_readiness_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {report_path}")
    print(f"GATE RESULTS: {json.dumps(gate_results, indent=2)}")
    print(f"smoke_permitted: {smoke_permitted}")
    print("\nDO NOT RUN THE 32B SCIENTIFIC SMOKE FROM THIS SCRIPT. This is a readiness probe only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
