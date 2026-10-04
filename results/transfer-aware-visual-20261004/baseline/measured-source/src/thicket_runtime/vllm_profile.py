"""Work 1: pinned RandOpt launcher/worker, Ray RPC, and actual vLLM generation.

One engine / one GPU only. This freezes candidate selection and does not run the
paper's training, ensemble selection, or distillation experiment.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .cli import revision_info, write_json
from .remote_trace import audit_remote_trace, bind_recipes, run_remote_trace
from .upstream import verify_upstream
from . import vllm_audit

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


class RayExecutor:
    def __init__(self, engine, prompts, answers, sampling):
        import ray
        self.ray, self.engine = ray, engine
        self.prompts, self.answers, self.sampling = prompts, answers, sampling

    def rpc(self, method, *args):
        result = self.ray.get(self.engine.collective_rpc.remote(method, args=args), timeout=300)
        if len(result) != 1:
            raise RuntimeError("this profiler requires exactly one worker")
        return result[0]

    def rebase(self):
        return self.rpc("reset_to_base_weights")

    def apply(self, candidate, strategy):
        if strategy == "legacy-add-subtract":
            return self.rpc("perturb_self_weights", candidate["seed"], candidate["sigma"], False)
        return self.rpc("apply_perturbation", candidate["seed"], candidate["sigma"])

    def restore(self, candidate, strategy):
        if strategy == "legacy-add-subtract":
            return self.rpc("restore_self_weights", candidate["seed"], candidate["sigma"], False)
        return self.rebase()

    def infer(self):
        result = self.ray.get(self.engine.generate.remote(self.prompts, self.sampling,
                                                         use_tqdm=False), timeout=300)
        if len(result) != len(self.answers) or any(len(r.outputs) != 1 for r in result):
            raise RuntimeError("generation request/output cardinality mismatch")
        return [{"text": r.outputs[0].text, "token_ids": list(r.outputs[0].token_ids),
                 "finish_reason": r.outputs[0].finish_reason,
                 "stop_reason": r.outputs[0].stop_reason} for r in result]

    def score(self, outputs):
        if len(outputs) != len(self.answers):
            raise ValueError("score cardinality mismatch")
        correct = 0
        for output, expected in zip(outputs, self.answers):
            values = re.findall(r"[-+]?\d+", output["text"])
            correct += bool(values and values[-1] == expected)
        return correct / len(self.answers)

    def memory(self, reset=False):
        return self.rpc(vllm_audit.memory, reset)

    def drift(self):
        return self.rpc(vllm_audit.drift)

    def fingerprint(self):
        return self.rpc(vllm_audit.fingerprint)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--recipes", default="results/llm-bounded-20261003/fixed-32-b4/candidates.json")
    p.add_argument("--data", default="examples/arithmetic_smoke.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--fixed-length", action="store_true")
    p.add_argument("--stop", action="append", default=[])
    p.add_argument("--strategy-order", choices=["legacy-first", "snapshot-first"], default="legacy-first")
    p.add_argument("--limit-candidates", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if min(args.max_new_tokens, args.limit_candidates, args.repeats) < 1 or args.warmup < 0:
        p.error("positive token/candidate/repeat bounds and nonnegative warmup required")
    if args.fixed_length and args.stop:
        p.error("fixed work must not use early stop strings")
    if not 0 < args.gpu_memory_utilization < 1:
        p.error("GPU memory utilization must be in (0, 1)")
    upstream = verify_upstream(args.upstream_root)
    raw = Path(args.data).read_bytes()
    data = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not data or any(not isinstance(r.get("prompt"), str) or
                       not isinstance(r.get("answer"), (str, int)) for r in data):
        p.error("data requires nonempty prompt/answer records")
    source_recipes = json.loads(Path(args.recipes).read_text())[:args.limit_candidates]
    if len(source_recipes) != args.limit_candidates:
        p.error("not enough source candidate recipes")
    bind_recipes(source_recipes, "preflight")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "thicket-vllm-ray-v1", "status": "running",
                "configuration": vars(args), "command": [sys.executable, *sys.argv],
                "source": revision_info(), "upstream": upstream,
                "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in Path(__file__).parent.glob("*.py")},
                "model": MODEL, "model_revision": REVISION,
                "data_sha256": hashlib.sha256(raw).hexdigest(), "examples": len(data),
                "scope": "actual pinned RandOpt launcher/worker; one Ray/vLLM engine; frozen candidate trace",
                "search_policy": "replay seeds 42-45 and sigma 0.001; no selection/adaptation/update",
                "scorer": "last-signed-integer-exact-match-v1",
                "scorer_scope": "same HF pilot scorer, not the full upstream dataset handler",
                "cache_isolation": "prefix caching disabled; generate completes all requests before weights change",
                "uncertainty": "within-process repeats descriptive; no speedup inference",
                "versions": {n: importlib.metadata.version(n) for n in
                             ("torch", "vllm", "ray", "transformers", "numpy")}}
    write_json(root / "manifest.json", manifest)
    (root / "inputs.jsonl").write_bytes(raw)
    write_json(root / "source-candidates.json", source_recipes)
    subprocess.run(["nvidia-smi", "-q"], stdout=(root / "nvidia-smi-before.txt").open("w"), check=True)
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(root / "pip-freeze.txt").open("w"), check=True)
    engines, pgs = [], []
    try:
        import ray
        from huggingface_hub import snapshot_download
        from vllm import SamplingParams
        upstream_root = str(Path(args.upstream_root).resolve())
        sys.path.insert(0, upstream_root)
        from core.engine import launch_engines, cleanup_engines
        model_path = snapshot_download(MODEL, revision=REVISION,
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        if Path(model_path).name != REVISION:
            raise ValueError("resolved model snapshot does not match pinned revision")
        # Start our own bounded local Ray cluster; never attach to an existing job.
        worker_path = os.pathsep.join([upstream_root, str(Path(__file__).resolve().parents[1])])
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False,
                 object_store_memory=512 * 1024**2,
                 runtime_env={"env_vars": {"PYTHONPATH": worker_path,
                                          "OMP_NUM_THREADS": "1",
                                          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}})
        engines, pgs = launch_engines(1, model_path, precision="bfloat16",
                                     tensor_parallel_size=1, enable_prefix_caching=False,
                                     gpu_memory_utilization=args.gpu_memory_utilization)
        generation = {"temperature": 0.0, "seed": 0, "max_tokens": args.max_new_tokens,
                      "min_tokens": args.max_new_tokens if args.fixed_length else 0,
                      "ignore_eos": args.fixed_length, "stop": args.stop or None}
        api = RayExecutor(engines[0], [r["prompt"] for r in data],
                          [str(r["answer"]).strip() for r in data], SamplingParams(**generation))
        state = api.rpc(vllm_audit.initialize)
        write_json(root / "noise-layout-probe.json", api.rpc(vllm_audit.noise_layout_probe))
        candidates = bind_recipes(source_recipes, state["base_id"])
        write_json(root / "native-state.json", state)
        write_json(root / "candidates.json", candidates)
        manifest.update({"generation": generation, "native_base_id": state["base_id"],
                         "hf_base_id": source_recipes[0]["base_id"],
                         "cross_backend_candidate_identity_equal": state["base_id"] == source_recipes[0]["base_id"],
                         "identity_note": "native layout/base ID is authoritative; HF IDs are provenance only",
                         "engine": {"tensor_parallel_size": 1, "distributed_executor_backend": "ray",
                                    "enforce_eager": True, "enable_prefix_caching": False,
                                    "gpu_memory_utilization": args.gpu_memory_utilization},
                         "environment": state["environment"]})
        write_json(root / "manifest.json", manifest)
        strategies = ["legacy-add-subtract", "snapshot-copy"]
        if args.strategy_order == "snapshot-first":
            strategies.reverse()
        summaries = {}
        for strategy in strategies:
            print("TIMING", strategy, flush=True)
            report = run_remote_trace(api, candidates, strategy, repeats=args.repeats, warmup=args.warmup)
            # Preserve timing data even if a later audit fails.
            write_json(root / (strategy + ".timings.json"), report)
            print("AUDIT", strategy, flush=True)
            report["correctness"] = audit_remote_trace(api, candidates, strategy)
            report["exact_semantics_gate"] = report["correctness"]["exact_semantics_gate"]
            report["weight_operator_implementation"] = upstream
            write_json(root / (strategy + ".json"), report)
            summaries[strategy] = report["summary"] | {"exact_semantics_gate": report["exact_semantics_gate"]}
        write_json(root / "summary.json", summaries)
        manifest["status"] = "complete"
        print(json.dumps(summaries, indent=2), flush=True)
        return 0
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(root / "manifest.json", manifest)
        if engines:
            cleanup_engines(engines, pgs)
        elif "ray" in locals():
            ray.shutdown()
        subprocess.run(["nvidia-smi", "-q"], stdout=(root / "nvidia-smi-after.txt").open("w"), check=False)


if __name__ == "__main__":
    raise SystemExit(main())
