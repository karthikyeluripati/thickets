"""Capability Benchmark Gate CLI orchestrator -- base-model (ZERO perturbation) evaluation
for exactly one capability per invocation. Sibling of eval_base_image_aware.py, generalized
across the benchmarks/ package (a single local vllm.LLM, no Ray actors, no perturbation --
identical infra shape to Gate 1) instead of being GQA-specific.

Runs no RandOpt, applies no perturbation. GPU required for anything beyond --dry-run.

Usage:
    python -m neural_thickets_repro.run_capability_benchmark_gate \
        --config configs/benchmarks/counting.yaml --repeat --image-sanity

    # cheap N=5 smoke test (still calls the model, just on 5 examples) before a full run:
    python -m neural_thickets_repro.run_capability_benchmark_gate \
        --config configs/benchmarks/counting.yaml --subset-size 5

    # even cheaper: no GPU/model call at all, just data loading + integrity:
    python -m neural_thickets_repro.run_capability_benchmark_gate \
        --config configs/benchmarks/counting.yaml --dry-run
"""
from __future__ import annotations

import os

# Must execute before torch/vllm can be imported anywhere in this process -- same runtime
# compatibility fix as eval_base_image_aware.py (fork + CUDA-already-touched-in-parent =
# "Cannot re-initialize CUDA in forked subprocess"); not a reproduction-behavior change.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from .benchmarks.base import CapabilityBenchmark
from .benchmarks.card import BenchmarkCardData, write_card
from .benchmarks.image_sanity import run_image_sanity_check
from .benchmarks.integrity import validate_examples
from .benchmarks.runner import run_benchmark, write_predictions_jsonl
from .benchmarks.subset_selection import build_or_load_subset
from .config import load_capability_benchmark_config
from .env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from .vlm_adapter import resolve_model_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_spawn_configured() -> None:
    value = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if value != "spawn":
        raise RuntimeError(
            f"VLLM_WORKER_MULTIPROC_METHOD must be 'spawn' before vLLM initializes, got "
            f"{value!r}. This should have been forced at module import time."
        )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def load_adapter(dotted_path: str) -> CapabilityBenchmark:
    """dotted_path: "package.module.ClassName" -- e.g. cfg.dataset.adapter from the
    capability's YAML. Instantiated with no constructor args, per every adapter's own
    no-arg-default convention (real dependencies like a GQAHandler resolve lazily on first
    use, see _gqa_filtered_base.py).
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls()


def build_output_dir(base_output_dir: Path, capability: str) -> Path:
    return base_output_dir / capability


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeat", action="store_true", help="run a second identical pass to check repeatability; omitting caps Status below PASS")
    parser.add_argument("--image-sanity", action="store_true", help="run the correct/shuffled/text-only image-dependence sanity check")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "benchmark_gate"))
    parser.add_argument("--subset-size", type=int, default=None, help="override the config's dataset.subset_size, e.g. for a cheap N=5 smoke test")
    parser.add_argument("--dry-run", action="store_true", help="load examples / build the subset / run integrity checks only -- no model call, no GPU needed")
    args = parser.parse_args(argv)

    cfg = load_capability_benchmark_config(args.config)
    cfg.require_resolved("model.revision", "dataset.split")

    subset_size = args.subset_size if args.subset_size is not None else cfg.dataset.subset_size
    out_dir = build_output_dir(Path(args.output_dir), cfg.dataset.capability)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(cfg.dataset.adapter)
    if adapter.capability != cfg.dataset.capability:
        raise ValueError(
            f"Config dataset.capability={cfg.dataset.capability!r} does not match the "
            f"adapter's own capability={adapter.capability!r} ({cfg.dataset.adapter})."
        )

    print(f"=== Loading examples for capability={adapter.capability} (source={adapter.dataset_source()}) ===")
    all_examples = adapter.load_examples(cfg)
    print(f"  loaded {len(all_examples)} candidate examples")

    subset_ids_path = REPO_ROOT / "artifacts" / "benchmark_subsets" / f"{adapter.name}_{subset_size}.json"
    subset = build_or_load_subset(
        all_examples, subset_size, adapter.subset_selection_rule(),
        adapter.subset_selection_seed(cfg.reproducibility.global_seed), subset_ids_path,
    )
    print(f"  selected fixed subset of {len(subset)} examples (rule={adapter.subset_selection_rule()!r}), IDs at {subset_ids_path}")

    integrity = validate_examples(subset, n_requested=subset_size)
    print(
        f"  integrity: {integrity.n_valid_images}/{integrity.n_loaded} valid images, "
        f"{integrity.n_duplicate_ids} duplicate IDs, {integrity.n_missing_targets} missing targets"
    )
    (out_dir / "integrity_report.json").write_text(json.dumps(integrity.to_dict(), indent=2))
    if not integrity.passed:
        print("STOP: integrity check failed -- see counts above. Not proceeding to inference.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("--dry-run: stopping before model load/generation.")
        return 0

    try:
        assert_feasible(
            "run_capability_benchmark_gate",
            [check_cuda(), check_module("vllm"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    _assert_spawn_configured()  # must pass before the vLLM import immediately below
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path, dtype=cfg.model.precision, enforce_eager=True,
        gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, disable_log_stats=True,
    )
    sampling_params = SamplingParams(temperature=0.0, seed=cfg.reproducibility.global_seed, max_tokens=cfg.generation.max_tokens)

    print(f"\n=== Base run (N={len(subset)}) ===")
    base_result = run_benchmark(adapter, subset, llm, tokenizer, sampling_params)
    write_predictions_jsonl(base_result, subset, out_dir / "predictions.jsonl")
    (out_dir / "metrics.json").write_text(json.dumps(base_result.aggregate_metrics, indent=2))
    print(f"  primary_metric={base_result.aggregate_metrics['primary_metric']:.4f}  parser_failure_rate={base_result.aggregate_metrics['parser_failure_rate']:.4f}")

    repeatability_status = "NOT_RUN"
    repeat_metrics = None
    generation_hash_match = None
    parsed_prediction_hash_match = None
    if args.repeat:
        print(f"\n=== Repeat run (N={len(subset)}) ===")
        repeat_result = run_benchmark(adapter, subset, llm, tokenizer, sampling_params)
        repeat_metrics = repeat_result.aggregate_metrics
        generation_hash_match = base_result.generation_hash() == repeat_result.generation_hash()
        parsed_prediction_hash_match = base_result.parsed_prediction_hash() == repeat_result.parsed_prediction_hash()
        metrics_match = base_result.aggregate_metrics["primary_metric"] == repeat_result.aggregate_metrics["primary_metric"]
        # A raw-wording change with an identical parsed/scored answer is NOT itself a
        # repeatability failure -- both facts are recorded separately (see docstring/card.py),
        # never collapsed into one pass/fail bit that would hide the distinction.
        repeatability_status = "PASS" if (parsed_prediction_hash_match and metrics_match) else "FAIL"
        (out_dir / "repeatability.json").write_text(json.dumps({
            "repeatability_status": repeatability_status,
            "raw_generation_hash_match": generation_hash_match,
            "parsed_prediction_hash_match": parsed_prediction_hash_match,
            "metric_match": metrics_match,
            "base_primary_metric": base_result.aggregate_metrics["primary_metric"],
            "repeat_primary_metric": repeat_result.aggregate_metrics["primary_metric"],
            "absolute_repeat_difference": abs(base_result.aggregate_metrics["primary_metric"] - repeat_result.aggregate_metrics["primary_metric"]),
        }, indent=2))
        print(f"  repeatability={repeatability_status}  raw_generation_hash_match={generation_hash_match}  parsed_prediction_hash_match={parsed_prediction_hash_match}")

    image_sanity_result = None
    if args.image_sanity:
        sanity_n = min(cfg.gates.image_sanity_subset_size, len(subset))
        print(f"\n=== Image-dependence sanity check (n={sanity_n}) ===")
        image_sanity_result = run_image_sanity_check(
            adapter, subset[:sanity_n], llm, tokenizer, sampling_params, seed=cfg.reproducibility.global_seed,
        )
        (out_dir / "image_sanity.json").write_text(json.dumps(image_sanity_result.to_dict(), indent=2))
        print(
            f"  correct={image_sanity_result.correct_image_primary_metric:.4f}  "
            f"shuffled={image_sanity_result.shuffled_image_primary_metric:.4f}  "
            f"text_only={image_sanity_result.text_only_primary_metric}"
        )

    non_generic_metric_keys = [k for k in base_result.aggregate_metrics if k not in ("primary_metric", "parser_failure_rate")]
    card = BenchmarkCardData(
        dataset=adapter.name, capability=adapter.capability, dataset_revision=adapter.dataset_revision(),
        split=cfg.dataset.split, subset_size=len(subset),
        subset_seed=adapter.subset_selection_seed(cfg.reproducibility.global_seed), subset_ids_path=str(subset_ids_path),
        integrity=integrity,
        prompt_template=str(adapter.build_prompt(subset[0])) if subset else "N/A",
        generation_config={"decoding": cfg.generation.decoding, "max_tokens": cfg.generation.max_tokens, "seed": cfg.reproducibility.global_seed},
        prediction_parser=f"{type(adapter).__module__}.{type(adapter).__name__}.parse_prediction",
        metric_description=f"primary_metric={base_result.aggregate_metrics['primary_metric']!r}; also reports: {', '.join(non_generic_metric_keys)}",
        base_metrics=base_result.aggregate_metrics, repeat_metrics=repeat_metrics,
        repeatability_status=repeatability_status, generation_hash_match=generation_hash_match,
        parsed_prediction_hash_match=parsed_prediction_hash_match, image_sanity=image_sanity_result,
        known_caveats=adapter.known_caveats(),
    )
    status, reasons = write_card(card, cfg.gates, out_dir)

    run_metadata = {
        "capability": adapter.capability, "dataset": adapter.name, "our_repo_git_commit": _git_commit(),
        "model_name": cfg.model.name, "model_revision": cfg.model.revision, "model_snapshot_path": model_path,
        "python_version": platform.python_version(), "platform": platform.platform(), "command": " ".join(sys.argv),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    print(f"\n=== {adapter.capability} gate result: {status} ===")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
