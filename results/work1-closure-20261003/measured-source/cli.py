"""Command-line pilot with strict reports and explicit evidence limitations."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from .benchmark import audit_trace, environment, run_trace
from .candidate import CandidateSpec, RNG_SCHEMES
from .execution import STRATEGIES, WeightState
from .workloads import HFWorkload, SyntheticWorkload
from .upstream import UpstreamWeightState, WORKER_BLOB


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload + "\n")
    os.replace(temporary, path)


def revision_info() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain")}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=["synthetic", "hf"], default="synthetic")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--strategy", choices=[*STRATEGIES, "both"], default="both")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma", type=float, default=0.001)
    p.add_argument("--rng", choices=RNG_SCHEMES, default=RNG_SCHEMES[0])
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--upstream-root", help="local pinned RandOpt checkout; original weight ops only")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--data", default="examples/arithmetic_smoke.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--fixed-length", action="store_true")
    p.add_argument("--no-empty-cache", action="store_true", help="explicit cache-policy ablation")
    p.add_argument("--no-cuda-events", action="store_true", help="wall-only instrumentation control")
    p.add_argument("--trace", action="store_true", help="diagnostic run, not throughput evidence")
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if min(args.candidates, args.repeats, args.threads) < 1 or args.warmup < 0:
        p.error("candidates/repeats/threads must be positive; warmup must be nonnegative")
    if args.seed < 0 or args.seed + args.candidates >= 2**31:
        p.error("candidate seeds must be in [0, 2**31)")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        p.error("only CPU and single-device CUDA are implemented")
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is unavailable; no silent CPU fallback")
    if args.workload == "hf" and not args.model:
        p.error("--model is required for an HF workload")
    if args.upstream_root and (args.strategy != "legacy-add-subtract" or args.no_empty_cache
                               or args.rng != "randopt-per-tensor-v1"):
        p.error("--upstream-root requires --strategy legacy-add-subtract and original RNG/cache settings")
    # Validate recipe before doing expensive loading.
    CandidateSpec("validation", args.seed, args.sigma, args.rng)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "thicket-profile-v0.1", "status": "running",
                "configuration": vars(args), "source": revision_info(),
                "remote_base_commit": "deb414253139cc2559d19cdfe7e6b4786e7c40db",
                "upstream_reference_commit": "4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca",
                "scope": "single-device PyTorch reference; NOT vLLM/Ray end-to-end",
                "claim": "baseline characterization only; no optimized kernel implemented"}
    write_json(root / "manifest.json", manifest)
    try:
        torch.set_num_threads(args.threads)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = getattr(torch, args.dtype)
        if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 is not supported by this CUDA device")
        workload = (SyntheticWorkload(device, dtype, args.width, args.depth, args.batch)
                    if args.workload == "synthetic" else
                    HFWorkload(args.model, args.revision, args.data, device, dtype,
                               args.max_new_tokens, args.fixed_length))
        manifest.update({"environment": environment(device), "workload": workload.metadata})
        write_json(root / "manifest.json", manifest)
        summaries = {}
        strategies = STRATEGIES if args.strategy == "both" else [args.strategy]
        for strategy in strategies:
            if args.upstream_root:
                state = UpstreamWeightState(workload.model, strategy,
                    upstream_root=args.upstream_root, detail=args.trace)
            else:
                state = WeightState(workload.model, strategy,
                    cleanup_cache=not args.no_empty_cache, detail=args.trace)
            candidates = [CandidateSpec(state.base_id, args.seed + i, args.sigma, args.rng)
                          for i in range(args.candidates)]
            trace_path = root / "candidates.json"
            specs = [asdict(c) | {"candidate_id": c.candidate_id} for c in candidates]
            if trace_path.exists():
                if json.loads(trace_path.read_text()) != specs:
                    raise RuntimeError("candidate identity changed between strategies")
            else:
                write_json(trace_path, specs)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            if args.trace:
                with torch.profiler.profile(activities=activities, record_shapes=True,
                                            profile_memory=True) as prof:
                    report = run_trace(state, candidates, workload.infer, workload.score,
                                       repeats=args.repeats, warmup=args.warmup,
                                       cuda_events=not args.no_cuda_events)
                prof.export_chrome_trace(str(root / f"{strategy}.trace.json"))
            else:
                report = run_trace(state, candidates, workload.infer, workload.score,
                                   repeats=args.repeats, warmup=args.warmup,
                                   cuda_events=not args.no_cuda_events)
            # Disable profiler labels for the separate correctness audit.
            state.detail = False
            report["weight_operator_implementation"] = (
                {"kind": "original-pinned-RandOpt-worker", "blob": WORKER_BLOB}
                if args.upstream_root else {"kind": "PyTorch-reference-reimplementation"})
            report["correctness"] = audit_trace(state, candidates, workload.infer, workload.score)
            report["exact_semantics_gate"] = all(report["correctness"][k] for k in
                ("all_outputs_exact", "all_rewards_exact", "all_resets_exact"))
            write_json(root / f"{strategy}.json", report)
            summaries[strategy] = report["summary"] | {
                "exact_semantics_gate": report["exact_semantics_gate"],
                "gpu_performance_evidence": report["gpu_performance_evidence"]}
            del state
        write_json(root / "summary.json", summaries)
        manifest["status"] = "complete"
        manifest["note"] = "Completion is not a claim of equivalence or speedup; inspect correctness gates."
        write_json(root / "manifest.json", manifest)
        print(json.dumps(summaries, indent=2, allow_nan=False))
        return 0
    except BaseException as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(root / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    sys.exit(main())
