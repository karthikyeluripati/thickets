"""Phases 1-3 only: independent native RandOpt candidates and offline simulation.

No proposal/search algorithm, speculative verifier, or optimized kernel lives in
this path. A passing gate is a prerequisite for a separate phase-4 implementation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .agreement import BLOCK_SIZES, compare_tokens, feasibility_gate, simulate_prefix, summarize_records
from .candidate import CandidateSpec
from .cli import revision_info, write_json
from .upstream import verify_upstream
from .vllm_profile import MODEL, REVISION, RayExecutor
from . import vllm_audit

GSM_BLOBS = {
    "utils/reward_score/gsm8k.py": "85783e641d512fb95510aad3182217ed605725ba",
    "data_handlers/gsm8k.py": "03050c28caf7b2d1b035dd84948b52bcc2898bbe",
    "baselines/examples/data_preprocess/gsm8k.py": "1656cdbc896a8f14fc7e09705d36335f52165533",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reward(upstream):
    for name, expected in GSM_BLOBS.items():
        raw = (Path(upstream) / name).read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != expected:
            raise ValueError(f"GSM8K upstream source mismatch: {name}")
    spec = importlib.util.spec_from_file_location("_pinned_gsm_reward", Path(upstream) / "utils/reward_score/gsm8k.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def reward(text, answer):
        strict = module.compute_score(text, answer, method="strict")
        return strict if strict else module.compute_score(text, answer, method="flexible")
    return reward


def natural_reward(text, answer):
    values = re.findall(r"[-+]?\d+", text)
    return float(bool(values and values[-1] == answer))


class TraceExecutor(RayExecutor):
    def infer(self):
        result = self.ray.get(self.engine.generate.remote(self.prompts, self.sampling, use_tqdm=False), timeout=300)
        if len(result) != len(self.prompts) or any(len(r.outputs) != 1 for r in result):
            raise RuntimeError("generation cardinality mismatch")
        return [{"text": r.outputs[0].text, "token_ids": list(r.outputs[0].token_ids),
                 "prompt_token_ids": list(r.prompt_token_ids),
                 "finish_reason": r.outputs[0].finish_reason,
                 "stop_reason": r.outputs[0].stop_reason} for r in result]


def make_pairs(base, outputs, data):
    if len(base) != len(outputs) or len(outputs) != len(data):
        raise ValueError("trace cardinality mismatch")
    pairs = []
    for b, c, item in zip(base, outputs, data):
        if b["prompt_token_ids"] != c["prompt_token_ids"]:
            raise ValueError("base/candidate prompt tokenization mismatch")
        metrics = compare_tokens(b["token_ids"], c["token_ids"])
        metrics.update(prompt_id=item["id"], candidate_output=c,
                       candidate_reward=c["reward"], base_reward=b["reward"],
                       candidate_capped=c["finish_reason"] == "length",
                       base_capped=b["finish_reason"] == "length",
                       uncensored_complete_equality=(metrics["complete_observed_token_equality"]
                            and b["finish_reason"] == c["finish_reason"] == "stop"
                            and b["stop_reason"] == c["stop_reason"]),
                       simulation={str(k): simulate_prefix(b["token_ids"], c["token_ids"], k)
                                   for k in BLOCK_SIZES})
        pairs.append(metrics)
    return pairs


def generate_workloads(api, workloads, SamplingParams):
    outputs, timings = {}, {}
    for spec, data, scorer in workloads:
        api.prompts = [r["prompt"] for r in data]
        api.answers = [str(r["answer"]) for r in data]
        api.sampling = SamplingParams(temperature=0, seed=0, max_tokens=spec["max_tokens"],
                                      stop=spec["stop"] or None)
        start = time.perf_counter()
        rows = api.infer()
        elapsed = time.perf_counter() - start
        outputs[spec["name"]] = rows
        timings[spec["name"]] = {"inference_seconds": elapsed}
    return outputs, timings


def score_workloads(outputs, workloads, timings):
    for spec, data, scorer in workloads:
        start = time.perf_counter()
        for output, item in zip(outputs[spec["name"]], data):
            output["reward"] = scorer(output["text"], str(item["answer"]))
        timings[spec["name"]]["score_seconds"] = time.perf_counter() - start


def independent_candidate(api, candidate, workloads, SamplingParams):
    """Audits and identity hashing are deliberately outside lifecycle timings."""
    api.rebase()
    if not api.drift()["exact_base"]:
        raise RuntimeError("candidate did not start from exact base")
    api.memory(reset=True)
    start = time.perf_counter()
    api.apply(candidate, "snapshot-copy")
    apply_seconds = time.perf_counter() - start
    try:
        candidate_state_id = api.fingerprint()
        outputs, timings = generate_workloads(api, workloads, SamplingParams)
    finally:
        start = time.perf_counter()
        api.restore(candidate, "snapshot-copy")
        restore_seconds = time.perf_counter() - start
    memory = api.memory()
    score_workloads(outputs, workloads, timings)
    audit = api.drift()
    return {"candidate": candidate, "candidate_state_id": candidate_state_id,
            "outputs": outputs, "descriptive_costs": {"apply_seconds": apply_seconds,
             "restore_seconds": restore_seconds, "workloads": timings},
            "memory": memory, "restoration": audit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--protocol", default="experiments/shared_speculative_protocol.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(Path(args.protocol).read_text())
    if protocol["model"] != MODEL or protocol["model_revision"] != REVISION:
        raise ValueError("this experiment requires the Work 1 pinned model")
    upstream = verify_upstream(args.upstream_root)
    reward = load_reward(args.upstream_root)
    workloads = []
    for spec in protocol["workloads"]:
        data = [json.loads(line) for line in Path(spec["path"]).read_text(encoding="utf-8").splitlines()]
        if len(data) != spec["count"] or len({r["id"] for r in data}) != len(data):
            raise ValueError("frozen prompt count/identity mismatch")
        workloads.append((spec, data, reward if spec["scorer"].startswith("pinned-randopt") else natural_reward))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    (root / "candidates").mkdir()
    (root / "measured-source").mkdir()
    repo = Path(__file__).resolve().parents[2]
    sources = list((repo / "src/thicket_runtime").glob("*.py")) + [Path(args.protocol)]
    sources += [repo / "scripts/freeze_gsm8k.py", repo / "scripts/run_shared_feasibility.sh"]
    for f in sources:
        relative = f.resolve().relative_to(repo)
        target = root / "measured-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
    for spec, _, _ in workloads:
        shutil.copyfile(spec["path"], root / (spec["name"] + ".inputs.jsonl"))
    for f in (repo / "examples").glob("*.provenance.json"):
        shutil.copyfile(f, root / f.name)
    write_json(root / "protocol.json", protocol)
    manifest = {"schema": "shared-speculative-feasibility-v1", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "command": [sys.executable, *sys.argv], "source": revision_info(),
                "protocol_sha256": sha256(args.protocol), "upstream": upstream,
                "gsm8k_git_blobs": GSM_BLOBS, "model": MODEL, "model_revision": REVISION,
                "input_sha256": {s["name"]: sha256(s["path"]) for s, _, _ in workloads},
                "source_sha256": {str(f.resolve().relative_to(repo)): sha256(f) for f in sources},
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "vllm", "ray", "transformers", "numpy")},
                "cache_isolation": "prefix caching disabled; requests fully complete before each mutation",
                "engine": {"enforce_eager": True, "dtype": "bfloat16", "TP": 1, "backend": "ray"},
                "measurement_scope": "independent greedy outputs and offline round simulation; no verifier or speedup measurement"}
    write_json(root / "manifest.json", manifest)
    for command, name in ((["nvidia-smi", "-q"], "nvidia-smi-before.txt"),
                          ([sys.executable, "-m", "pip", "freeze"], "pip-freeze.txt")):
        with (root / name).open("w") as stream:
            subprocess.run(command, stdout=stream, check=True)
    engines, pgs, records = [], [], []
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
            raise ValueError("resolved model revision mismatch")
        worker_path = os.pathsep.join([upstream_root, str(repo / "src")])
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=512 * 1024**2,
                 runtime_env={"env_vars": {"PYTHONPATH": worker_path, "OMP_NUM_THREADS": "1",
                                          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}})
        engines, pgs = launch_engines(1, model_path, precision="bfloat16", tensor_parallel_size=1,
                                     enable_prefix_caching=False,
                                     gpu_memory_utilization=protocol["gpu_memory_utilization"])
        api = TraceExecutor(engines[0], [], [], None)
        state = api.rpc(vllm_audit.initialize)
        write_json(root / "native-state.json", state)
        manifest["native_base_id"] = state["base_id"]
        manifest["environment"] = state["environment"]
        candidates = []
        for seed in range(protocol["seed_start"], protocol["seed_start"] + protocol["seeds_per_sigma"]):
            for sigma in protocol["sigmas"]:
                c = CandidateSpec(state["base_id"], seed, sigma)
                candidates.append(asdict(c) | {"candidate_id": c.candidate_id})
        write_json(root / "planned-candidates.json", candidates)
        write_json(root / "manifest.json", manifest)
        base, base_times = generate_workloads(api, workloads, SamplingParams)
        score_workloads(base, workloads, base_times)
        write_json(root / "base.json", {"outputs": base, "descriptive_costs": base_times})
        zero = CandidateSpec(state["base_id"], protocol["seed_start"], 0.)
        control = independent_candidate(api, asdict(zero) | {"candidate_id": zero.candidate_id}, workloads, SamplingParams)
        write_json(root / "zero-sigma-control.json", control)
        controls = {"zero_sigma_output_equal": control["outputs"] == base,
                    "zero_sigma_exact_restore": control["restoration"]["exact_base"],
                    "zero_sigma_state_equal": control["candidate_state_id"] == state["base_id"]}
        write_json(root / "controls.json", controls)
        if not all(controls.values()):
            raise RuntimeError("zero-sigma determinism control failed")
        started, slowest_seed = time.monotonic(), 0.
        group_size = len(protocol["sigmas"])
        manifest["sweep_stop_reason"] = "planned_count"
        for offset in range(0, len(candidates), group_size):
            if offset and time.monotonic() - started + 1.5 * slowest_seed > protocol["sweep_budget_seconds"]:
                manifest["sweep_stop_reason"] = "predeclared_time_budget"
                break
            seed_start = time.monotonic()
            for candidate in candidates[offset:offset + group_size]:
                raw = independent_candidate(api, candidate, workloads, SamplingParams)
                record = {key: value for key, value in raw.items() if key != "outputs"}
                record["workloads"] = {s["name"]: make_pairs(base[s["name"]], raw["outputs"][s["name"]], data)
                                       for s, data, _ in workloads}
                path = root / "candidates" / (candidate["candidate_id"] + ".json")
                write_json(path, record)
                if not raw["restoration"]["exact_base"]:
                    raise RuntimeError("snapshot restoration audit failed")
                records.append(record)
                print(json.dumps({"completed": len(records), "seed": candidate["seed"], "sigma": candidate["sigma"],
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
            slowest_seed = max(slowest_seed, time.monotonic() - seed_start)
        manifest["sweep_seconds"] = time.monotonic() - started
        manifest["completed_candidates"] = len(records)
        manifest["completed_seeds_per_sigma"] = len(records) // group_size
        api.rebase()
        final_base, final_times = generate_workloads(api, workloads, SamplingParams)
        score_workloads(final_base, workloads, final_times)
        write_json(root / "base-repeat.json", {"outputs": final_base, "descriptive_costs": final_times})
        repeat = independent_candidate(api, candidates[0], workloads, SamplingParams)
        write_json(root / "candidate-repeat.json", repeat)
        original_outputs = {name: [r["candidate_output"] for r in rows] for name, rows in records[0]["workloads"].items()}
        controls.update(base_repeat_output_equal=base == final_base,
                        candidate_repeat_output_equal=repeat["outputs"] == original_outputs,
                        candidate_repeat_state_equal=repeat["candidate_state_id"] == records[0]["candidate_state_id"],
                        candidate_repeat_exact_restore=repeat["restoration"]["exact_base"],
                        final_base_hash_equal=api.fingerprint() == state["base_id"],
                        all_candidate_restores_exact=all(r["restoration"]["exact_base"] for r in records))
        write_json(root / "controls.json", controls)
        summary = summarize_records(records)
        write_json(root / "summary.json", summary)
        gate = feasibility_gate(summary, protocol, controls_passed=all(controls.values()), complete=True)
        write_json(root / "gate.json", gate)
        manifest["status"] = "complete" if all(controls.values()) else "invalid_controls"
        print(json.dumps({"gate": gate, "completed": len(records)}), flush=True)
        return 0 if all(controls.values()) else 2
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "manifest.json", manifest)
        if engines:
            cleanup_engines(engines, pgs)
        elif "ray" in locals():
            ray.shutdown()
        with (root / "nvidia-smi-after.txt").open("w") as stream:
            subprocess.run(["nvidia-smi", "-q"], stdout=stream, check=False)
        write_json(root / "sha256.json", {str(f.relative_to(root)): sha256(f) for f in sorted(root.rglob("*"))
                                          if f.is_file() and f.name != "sha256.json"})


if __name__ == "__main__":
    raise SystemExit(main())
