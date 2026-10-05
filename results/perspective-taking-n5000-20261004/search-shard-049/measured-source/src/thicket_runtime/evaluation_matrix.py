"""Exhaustive, snapshot-anchored matrix collection. No partial evaluation policy."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .candidate import CandidateSpec
from .cli import revision_info, write_json
from .shared_speculative import (TraceExecutor, generate_workloads, independent_candidate,
                                 load_reward, score_workloads, sha256, GSM_BLOBS)
from .upstream import verify_upstream
from .vllm_profile import MODEL, REVISION
from . import vllm_audit


def write_gzip(path, value):
    if path.exists():
        raise FileExistsError(path)
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(data, mtime=0))


def reward_and_extractor(root):
    reward = load_reward(root)  # validates scorer, handler and preprocessor blobs
    spec = importlib.util.spec_from_file_location("_matrix_gsm8k", Path(root) / "utils/reward_score/gsm8k.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    def extract(text):
        answer = mod.extract_solution(text, method="strict")
        if answer is None:
            answer = mod.extract_solution(text, method="flexible")
        return answer if answer is not None else ""
    return reward, extract


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--protocol", default="experiments/adaptive_evaluation_protocol.json")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    protocol = json.loads(Path(args.protocol).read_text())
    if (protocol["model"], protocol["model_revision"]) != (MODEL, REVISION):
        raise ValueError("model identity mismatch")
    upstream = verify_upstream(args.upstream_root)
    reward, extract = reward_and_extractor(args.upstream_root)
    if sha256(protocol["provenance_path"]) != protocol["provenance_sha256"]:
        raise ValueError("frozen provenance checksum mismatch")
    provenance = json.loads(Path(protocol["provenance_path"]).read_text())
    data = []
    for key, name in (("selection_path", "selection200"), ("heldout_path", "heldout40")):
        if sha256(protocol[key]) != provenance["sources"][name]["jsonl_sha256"]:
            raise ValueError("frozen input checksum mismatch")
        data.extend(json.loads(line) for line in Path(protocol[key]).read_text(encoding="utf-8").splitlines())
    if len(data) != 240 or len({r["id"] for r in data}) != 240:
        raise ValueError("expected 200 selection and 40 test prompts")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    populations = protocol["populations"]
    for pop in populations:
        (root / pop).mkdir()
    for f in [*Path(__file__).parent.glob("*.py"), Path(args.protocol),
              repo / "scripts/freeze_adaptive_gsm8k.py", repo / "scripts/freeze_gsm8k.py",
              repo / "scripts/run_evaluation_matrix.sh"]:
        target = root / "measured-source" / f.resolve().relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
    for key in ("selection_path", "heldout_path", "provenance_path"):
        shutil.copyfile(protocol[key], root / Path(protocol[key]).name)
    write_json(root / "protocol.json", protocol)
    manifest = {"schema": "exhaustive-candidate-matrix-v1", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), "source": revision_info(),
                "command": [sys.executable, *sys.argv], "upstream": upstream, "gsm_blobs": GSM_BLOBS,
                "model": MODEL, "model_revision": REVISION, "protocol_sha256": sha256(args.protocol),
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "vllm", "ray", "transformers", "numpy")},
                "cache_isolation": "prefix cache disabled; all requests drained before weight change",
                "execution": "original RandOpt snapshot apply/reset, native packed tensors, eager Ray/vLLM, BF16 TP1",
                "collection": "every collected candidate sees every prompt; no partial/adaptive evaluation",
                "raw_encoding": "gzip of compact UTF-8 JSON, mtime=0; generated tokens/text retained exactly",
                "selection_budget_scope": "200 selection prompts only; 40 test prompts are separate final ensemble evaluation"}
    write_json(root / "manifest.json", manifest)
    for command, name in ((["nvidia-smi", "-q"], "nvidia-smi-before.txt"),
                          ([sys.executable, "-m", "pip", "freeze"], "pip-freeze.txt")):
        with (root / name).open("w") as stream:
            subprocess.run(command, stdout=stream, check=True)
    engines, pgs, matrices = [], [], {name: [] for name in populations}
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
            raise ValueError("model snapshot mismatch")
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
        if state["base_id"] != protocol["expected_native_base_id"]:
            raise ValueError("native base differs from pinned reference")
        manifest["native_base_id"] = state["base_id"]
        manifest["environment"] = state["environment"]
        workloads = [({"name": "matrix240", "max_tokens": protocol["max_tokens"], "stop": []}, data, reward)]
        planned = []
        for offset in range(protocol["target_seeds_per_population"]):
            for pop, first in populations.items():
                for sigma in protocol["sigmas"]:
                    c = CandidateSpec(state["base_id"], first + offset, sigma)
                    planned.append({"population": pop, **asdict(c), "candidate_id": c.candidate_id})
        write_json(root / "planned-candidates.json", planned)
        base, timings = generate_workloads(api, workloads, SamplingParams)
        score_workloads(base, workloads, timings)
        base_rows = base["matrix240"]
        write_gzip(root / "base.json.gz", {"outputs": base_rows, "costs": timings})
        zero = CandidateSpec(state["base_id"], 10000, 0.)
        control = independent_candidate(api, asdict(zero) | {"candidate_id": zero.candidate_id}, workloads, SamplingParams)
        write_gzip(root / "zero-control.json.gz", control)
        controls = {"zero_outputs_equal": control["outputs"]["matrix240"] == base_rows,
                    "zero_state_equal": control["candidate_state_id"] == state["base_id"],
                    "zero_restore_exact": control["restoration"]["exact_base"]}
        write_json(root / "controls.json", controls)
        if not all(controls.values()):
            raise RuntimeError("initial correctness controls failed")
        started, slowest, block_size = time.monotonic(), 0., len(populations) * len(protocol["sigmas"])
        first_raw = None
        manifest["stop_reason"] = "planned_count"
        for offset in range(0, len(planned), block_size):
            if offset and time.monotonic() - started + 1.5 * slowest > protocol["sweep_budget_seconds"]:
                manifest["stop_reason"] = "predeclared_time_budget"
                break
            block_started = time.monotonic()
            for c in planned[offset:offset + block_size]:
                raw = independent_candidate(api, c, workloads, SamplingParams)
                if first_raw is None:
                    first_raw = raw
                rows = raw.pop("outputs")["matrix240"]
                if any(r["prompt_token_ids"] != b["prompt_token_ids"] for r, b in zip(rows, base_rows)):
                    raise ValueError("candidate prompt tokenization changed")
                # Input tokens are stored once with base; IDs bind each output.
                compact = [{k: v for k, v in r.items() if k != "prompt_token_ids"} |
                           {"prompt_id": item["id"], "voting_answer": extract(r["text"])}
                           for r, item in zip(rows, data)]
                raw["outputs"] = compact
                write_gzip(root / c["population"] / (c["candidate_id"] + ".json.gz"), raw)
                if not raw["restoration"]["exact_base"]:
                    raise RuntimeError("restoration audit failed")
                matrices[c["population"]].append({"candidate": c, "state_id": raw["candidate_state_id"],
                    "rewards": [r["reward"] for r in rows], "tokens": [len(r["token_ids"]) for r in rows],
                    "voting_answers": [r["voting_answer"] for r in compact],
                    "capped": [r["finish_reason"] == "length" for r in rows]})
                print(json.dumps({"completed": sum(map(len, matrices.values())), "population": c["population"],
                                  "seed": c["seed"], "sigma": c["sigma"],
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
            slowest = max(slowest, time.monotonic() - block_started)
        manifest["sweep_seconds"] = time.monotonic() - started
        manifest["candidates_per_population"] = {k: len(v) for k, v in matrices.items()}
        manifest["completed_candidates"] = sum(map(len, matrices.values()))
        for pop, rows in matrices.items():
            write_json(root / ("matrix-" + pop + ".json"), {"population": pop, "base_id": state["base_id"],
                "selection_count": protocol["selection_prompts"], "prompt_ids": [r["id"] for r in data],
                "answers": [str(r["answer"]) for r in data], "rows": rows})
        api.rebase()
        end_base, times = generate_workloads(api, workloads, SamplingParams)
        score_workloads(end_base, workloads, times)
        write_gzip(root / "base-repeat.json.gz", {"outputs": end_base["matrix240"], "costs": times})
        repeat = independent_candidate(api, planned[0], workloads, SamplingParams)
        write_gzip(root / "candidate-repeat.json.gz", repeat)
        controls.update(base_repeat_equal=base_rows == end_base["matrix240"],
                        candidate_repeat_equal=[{k: v for k, v in r.items() if k != "prompt_token_ids"}
                                                for r in repeat["outputs"]["matrix240"]]
                            == [{k: v for k, v in r.items() if k not in ("prompt_id", "voting_answer")}
                                for r in first_raw["outputs"]],
                        candidate_state_repeat_equal=repeat["candidate_state_id"] == first_raw["candidate_state_id"],
                        repeat_restore_exact=repeat["restoration"]["exact_base"],
                        final_base_equal=api.fingerprint() == state["base_id"], all_collected_resets_exact=True)
        write_json(root / "controls.json", controls)
        enough = min(map(len, matrices.values())) >= 3 * protocol["minimum_seeds_per_population"]
        manifest["status"] = "complete" if enough and all(controls.values()) else "insufficient_or_invalid"
        print(json.dumps({"status": manifest["status"], "candidates": manifest["completed_candidates"]}), flush=True)
        return 0 if manifest["status"] == "complete" else 2
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
        write_json(root / "sha256.json", {f.relative_to(root).as_posix(): sha256(f)
                                          for f in sorted(root.rglob("*")) if f.is_file() and f.name != "sha256.json"})


if __name__ == "__main__":
    raise SystemExit(main())
