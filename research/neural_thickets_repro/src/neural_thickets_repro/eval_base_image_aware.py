"""Gate 1 FULL baseline, image-aware adapter. GPU required.

Runs after CONFIRMED image-blindness (GATE1_DIAGNOSIS.md: 100-example paired audit --
text-only 15%/22% [head/march-era scoring] vs identical-prompt-with-image 59%/59%; 45
wrong->correct flips vs 1 reverse; 20/20 question<->image pairs verified against source).

This is the minimal, documented deviation from the released code for the BASELINE case:
same data (all 12,578 testdev-balanced examples), same prompt text/chat-template, same
decoding (temperature=0, max_tokens=256, seed from config), same answer extraction/scoring
(both current-HEAD and paper-era pre-cb108d44d6 scoring, both reported) -- the ONLY change
is that the image is actually attached to the vLLM request via multi_modal_data
(vlm_adapter.generate_with_images), instead of being silently dropped.

Does NOT touch core/engine.py, utils/worker_extn.py, or any perturbation/selection/voting
code. Does NOT run RandOpt. Gate 2 stays blocked regardless of what this baseline reports --
this script cannot start it (no such code path exists here).

Usage:
    python -m neural_thickets_repro.eval_base_image_aware --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

from .config import load_config
from .env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from .vlm_adapter import generate_with_images, load_gqa_handler, resolve_model_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

PUBLISHED_BASE_ACCURACY = 0.566


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def _package_versions() -> Dict[str, str]:
    versions = {}
    for mod_name in ("torch", "transformers", "vllm", "ray", "numpy"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    return versions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--data-dir", default=str(EXTERNAL_ROOT / "data" / "gqa"))
    parser.add_argument("--max-examples", type=int, default=None, help="override for a quick partial run; default = all")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "base_image_aware"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "eval_base_image_aware",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision", "dataset.selection_split", "dataset.test_split")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    handler = load_gqa_handler(EXTERNAL_ROOT)
    testdev_path = str(Path(args.data_dir) / "testdev.parquet")
    task_datas = handler.load_data(testdev_path, split="test", max_samples=args.max_examples)

    n_with_images = sum(1 for d in task_datas if "image_path" in d)
    n_skipped_no_image = len(task_datas) - n_with_images
    if n_skipped_no_image:
        print(
            f"WARNING: only {n_with_images}/{len(task_datas)} examples have image_path -- "
            f"the rest have no local image and cannot be evaluated image-aware.",
            file=sys.stderr,
        )
    task_datas = [d for d in task_datas if "image_path" in d]
    print(f"Evaluating {len(task_datas)} examples (image-aware).")

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype=cfg.model.precision,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0, seed=cfg.reproducibility.global_seed, max_tokens=cfg.evaluation.max_tokens
    )

    start = time.time()
    responses = generate_with_images(llm, sampling_params, task_datas, tokenizer)
    elapsed = time.time() - start
    print(f"Generation done in {elapsed:.1f}s ({elapsed/len(task_datas):.3f}s/example).")

    predictions_path = out_dir / "predictions.jsonl"
    correct_head = correct_march = 0
    with predictions_path.open("w") as f:
        for d, resp in zip(task_datas, responses):
            extracted = handler.extract_answer_for_voting(resp) or ""
            is_correct_head = bool(extracted) and handler.is_voted_answer_correct(extracted, d["ground_truth"])
            is_correct_march = handler.compute_reward(resp, d["ground_truth"]) > 0
            correct_head += int(is_correct_head)
            correct_march += int(is_correct_march)
            f.write(json.dumps({
                "example_id": str(d["question_id"]),
                "image_id": Path(d["image_path"]).stem,
                "question": d["messages"][0]["content"][1]["text"],
                "reference_answer": d["ground_truth"]["answer"],
                "raw_prediction": resp,
                "normalized_prediction": extracted,
                "correct_head_scoring": bool(is_correct_head),
                "correct_march_scoring": bool(is_correct_march),
            }) + "\n")

    n = len(task_datas)
    accuracy_head = correct_head / n
    accuracy_march = correct_march / n

    metrics = {
        "n": n,
        "accuracy_head_scoring": accuracy_head,
        "accuracy_march_scoring": accuracy_march,
        "published_base_accuracy": PUBLISHED_BASE_ACCURACY,
        "diff_head_vs_published_pp": (accuracy_head - PUBLISHED_BASE_ACCURACY) * 100,
        "diff_march_vs_published_pp": (accuracy_march - PUBLISHED_BASE_ACCURACY) * 100,
        "gate1_threshold_pp": {"proceed_at_most": cfg.gates.baseline_tolerance_pp.proceed_at_most,
                                "investigate_at_most": cfg.gates.baseline_tolerance_pp.investigate_at_most},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    run_metadata = {
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "model_snapshot_path": model_path,
        "dataset_name": "lmms-lab-encoder/GQA",
        "dataset_revision": cfg.dataset.revision,
        "test_split": cfg.dataset.test_split,
        "our_repo_git_commit": _git_commit(),
        "external_randopt_commit": "536df0a308f3990b6270c991fbb96bd0b779a58e",
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "seed": cfg.reproducibility.global_seed,
        "decoding": {"temperature": 0.0, "max_tokens": cfg.evaluation.max_tokens},
        "prompt_template": "Look at the image and answer the question.\\n\\nQuestion: {question}\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{{}}.",
        "image_handling": "vLLM multi_modal_data (the confirmed fix -- see vlm_adapter.py divergence #4)",
        "n_examples_evaluated": n,
        "n_examples_skipped_no_image": n_skipped_no_image,
        "generation_elapsed_seconds": elapsed,
        "command": " ".join(sys.argv),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    print(f"\n=== Gate 1 FULL baseline (image-aware, n={n}) ===")
    print(f"  current-HEAD scoring:  {accuracy_head:.4f}  ({correct_head}/{n})")
    print(f"  march/paper-era scoring: {accuracy_march:.4f}  ({correct_march}/{n})")
    print(f"  published:             {PUBLISHED_BASE_ACCURACY:.4f}")
    print(f"  diff (march vs published): {metrics['diff_march_vs_published_pp']:+.2f} pp")
    print(f"\nWrote {predictions_path}")
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
