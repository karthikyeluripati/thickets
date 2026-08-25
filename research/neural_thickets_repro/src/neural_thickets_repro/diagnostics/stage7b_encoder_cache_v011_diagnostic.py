"""Stage 7B multimodal-encoder-cache v0.11.0 compatibility diagnostic -- GPU required.

This is the correct smoke to run BEFORE another Stage-7B cache-safety smoke
(run_stage7b_anatomical_calibration.py --cache-safety-smoke): it validates the version-aware
full encoder-cache reset (vlm_adapter.reset_vllm_encoder_cache_full -- see that module's own
docstring, divergence #9, for the full mechanism) directly, in isolation, with NO weight
perturbation anywhere in this script -- so a failure here can only mean the cache-reset
compatibility layer itself is wrong for this exact pinned runtime, never a radius/anatomy bug.

Root cause this exists to catch before it's ever trusted inside the real candidate loop: the
FIRST implementation of the actor-side reset called `self.llm_engine.reset_encoder_cache()`
directly, which does not exist on the pinned vLLM 0.11.0 runtime (confirmed live:
`AttributeError: 'LLMEngine' object has no attribute 'reset_encoder_cache'`, commit 74f273b's
cache-safety smoke). The corrected compatibility path's exact internal attribute names
(`EncoderCacheManager.cached`/`freeable`/`cache_size`/`num_free_slots`, the
`engine.engine_core.engine_core.scheduler` layout) are this project's best-faith reconstruction,
UNRESOLVED against actually running on the pinned pod (see vlm_adapter.py's own docstring) --
this diagnostic is exactly the tool that confirms or corrects them on real hardware.

Steps (all against ONE fixed real image example, at theta_0, no weight perturbation):
  1.  launch the Stage-7B engine (run_stage7b_anatomical_calibration.build_stage7b_engine_config
      -- enable_prefix_caching=False -- via launch_stage6_engine, same as the real run)
  2.  ensure_full_encoder_cache_reset_exposed() (must run before launch, done in step 1's setup)
  3.  reset_vllm_encoder_cache_full(engine) ONCE, immediately after launch -- this both proves
      the compatibility path is reachable at all AND reports vllm.__version__ + whether a
      native reset_encoder_cache() exists + which compatibility path was actually taken
      (engine_side_report["path"]) -- printed here (steps 2-4 of the task spec)
  4.  report_vllm_encoder_cache_state(engine) -- the INITIAL (post-reset, expected empty)
      scheduler/worker encoder-cache state (step 5 of the task spec)
  5.  generate ONCE for one real image example, under theta_0, no perturbation (step 6)
  6.  report_vllm_encoder_cache_state(engine) again -- prove a real cache entry now exists
      (step 7) -- hard-fails (reports FAIL, not a silent pass) if it does NOT, since that would
      mean this diagnostic's own premise (vLLM actually caches encoder output) is wrong
  7.  reset_vllm_encoder_cache_full(engine) -- the real, full, verified reset (step 8)
  8.  report_vllm_encoder_cache_state(engine) again -- prove scheduler logical cache is fresh
      (cached/freeable empty, num_free_slots == cache_size) and every worker's physical
      encoder_cache count == 0 (step 9); also reports enable_prefix_caching as CONFIGURED at
      launch (no verified live introspection API for this exists -- see report's own caveat)
  9.  generate the SAME image again (step 10)
  10. report_vllm_encoder_cache_state(engine) one more time -- prove a cache entry exists AGAIN
      (recomputation occurred, not a stale hit silently returning wrong/missing state) (step 11)
  11. no weight perturbation anywhere in this script (step 12)

Persists a report JSON. Every step's evidence is in the report, not merely a pass/fail verdict.

Usage:
    python -m neural_thickets_repro.diagnostics.stage7b_encoder_cache_v011_diagnostic --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # same runtime fix as Gate 1/2 -- see eval_base_image_aware.py

from ..config import load_config
from ..env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from ..run_stage7b_anatomical_calibration import build_stage7b_engine_config
from ..vlm_adapter import (
    bootstrap_ray,
    build_image_aware_requests,
    ensure_full_encoder_cache_reset_exposed,
    report_vllm_encoder_cache_state,
    reset_vllm_encoder_cache_full,
    resolve_model_snapshot,
    verify_workers_can_import_external_root,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"


def _worker_entry_counts(state: Dict[str, Any]) -> list:
    return [w.get("encoder_cache_entry_count") for w in state["worker_side"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "stage7b_anatomical_calibration" / "encoder_cache_v011_diagnostic_report.json"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "Stage 7B encoder-cache v0.11.0 compatibility diagnostic",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision")
    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    engine_config = build_stage7b_engine_config()
    assert engine_config["enable_prefix_caching"] is False
    print(f"engine_config: {json.dumps(engine_config, indent=2)}")

    import ray
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore
    from data_handlers.gqa import GQAHandler  # type: ignore

    from ..run_global_visual_thicket_pilot import launch_stage6_engine

    handler = GQAHandler()
    task_datas = handler.load_data(str(EXTERNAL_ROOT / "data" / "gqa" / "train.parquet"), split="train", max_samples=None)
    task_datas = [d for d in task_datas if "image_path" in d][:1]
    if not task_datas:
        print("No example with an image was found -- cannot populate a real encoder-cache entry.", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    requests = build_image_aware_requests(task_datas, tokenizer)
    sampling_params = SamplingParams(temperature=0.0, seed=cfg.reproducibility.global_seed, max_tokens=cfg.evaluation.max_tokens)

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines, pgs = None, None
    report: Dict[str, Any] = {}
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        print("1. Exposing the full encoder-cache reset mechanism (must run before engine launch)...")
        ensure_full_encoder_cache_reset_exposed(EXTERNAL_ROOT)

        print("2. Launching the Stage-7B engine (enable_prefix_caching=False, no weight perturbation)...")
        engines, pgs = launch_stage6_engine(
            model_path, precision=engine_config["precision"], gpu_memory_utilization=engine_config["gpu_memory_utilization"],
            max_model_len=engine_config["max_model_len"], tensor_parallel_size=engine_config["tensor_parallel_size"],
            enable_prefix_caching=engine_config["enable_prefix_caching"],
        )
        engine = engines[0]

        print("3. Initial reset (also reports vllm_version + native-vs-compat path)...")
        initial_reset_report = reset_vllm_encoder_cache_full(engine)
        vllm_version = initial_reset_report["engine_side"]["vllm_version"]
        compat_path = initial_reset_report["engine_side"]["path"]
        print(f"   vllm_version={vllm_version}")
        print(f"   native reset_encoder_cache() present: {compat_path == 'native_reset_encoder_cache'}")
        print(f"   compatibility path selected: {compat_path}")

        print("4. Inspecting initial (post-reset) encoder-cache state...")
        state_initial = report_vllm_encoder_cache_state(engine)
        initial_cached = state_initial["engine_side"]["scheduler_state"]["cached_entry_count"]
        initial_worker_counts = _worker_entry_counts(state_initial)
        print(f"   scheduler cached_entry_count={initial_cached}, worker entry counts={initial_worker_counts}")

        print("5. Generating for one real image example under theta_0 (no perturbation)...")
        first_outputs = ray.get(engine.generate.remote(requests, sampling_params, use_tqdm=False))
        first_text = first_outputs[0].outputs[0].text
        print(f"   generated: {first_text!r}")

        print("6. Inspecting encoder-cache state after generation (expect a populated entry)...")
        state_populated = report_vllm_encoder_cache_state(engine)
        populated_cached = state_populated["engine_side"]["scheduler_state"]["cached_entry_count"]
        populated_worker_counts = _worker_entry_counts(state_populated)
        print(f"   scheduler cached_entry_count={populated_cached}, worker entry counts={populated_worker_counts}")
        step6_pass = populated_cached > 0 or any(c > 0 for c in populated_worker_counts)
        print(f"   {'PASS' if step6_pass else 'FAIL'}: encoder cache populated after generation")

        print("7. Executing the full, verified reset...")
        reset_report = reset_vllm_encoder_cache_full(engine)

        print("8. Inspecting encoder-cache state after the reset (expect fresh/empty)...")
        state_reset = report_vllm_encoder_cache_state(engine)
        reset_scheduler_state = state_reset["engine_side"]["scheduler_state"]
        reset_worker_counts = _worker_entry_counts(state_reset)
        step8_pass = (
            reset_scheduler_state["cached_entry_count"] == 0
            and reset_scheduler_state["freeable_entry_count"] == 0
            and reset_scheduler_state["num_free_slots"] == reset_scheduler_state["cache_size"]
            and all(c == 0 for c in reset_worker_counts)
        )
        print(f"   scheduler_state={reset_scheduler_state}, worker entry counts={reset_worker_counts}")
        print(f"   {'PASS' if step8_pass else 'FAIL'}: scheduler + every worker fresh/empty after reset")
        print(f"   enable_prefix_caching (as configured at launch): {engine_config['enable_prefix_caching']}")

        print("9. Regenerating the SAME image (proving recomputation, not a stale/missing hit)...")
        second_outputs = ray.get(engine.generate.remote(requests, sampling_params, use_tqdm=False))
        second_text = second_outputs[0].outputs[0].text
        print(f"   generated: {second_text!r}")

        print("10. Inspecting encoder-cache state after the second generation...")
        state_repopulated = report_vllm_encoder_cache_state(engine)
        repopulated_cached = state_repopulated["engine_side"]["scheduler_state"]["cached_entry_count"]
        repopulated_worker_counts = _worker_entry_counts(state_repopulated)
        step10_pass = repopulated_cached > 0 or any(c > 0 for c in repopulated_worker_counts)
        print(f"   scheduler cached_entry_count={repopulated_cached}, worker entry counts={repopulated_worker_counts}")
        print(f"   {'PASS' if step10_pass else 'FAIL'}: encoder cache repopulated after the second generation (recomputed, not a stale/missing hit)")

        overall_pass = step6_pass and step8_pass and step10_pass

        report = {
            "vllm_version": vllm_version,
            "native_reset_encoder_cache_present": compat_path == "native_reset_encoder_cache",
            "compatibility_path_selected": compat_path,
            "enable_prefix_caching_configured": engine_config["enable_prefix_caching"],
            "example_id": str(task_datas[0].get("question_id", "unknown")),
            "state_initial": state_initial,
            "state_after_first_generation": state_populated,
            "step6_encoder_cache_populated_after_generation": {"pass": step6_pass, "cached_entry_count": populated_cached, "worker_entry_counts": populated_worker_counts},
            "reset_report": reset_report,
            "state_after_reset": state_reset,
            "step8_scheduler_and_worker_fresh_after_reset": {"pass": step8_pass, "scheduler_state": reset_scheduler_state, "worker_entry_counts": reset_worker_counts},
            "state_after_second_generation": state_repopulated,
            "step10_encoder_cache_repopulated_after_second_generation": {"pass": step10_pass, "cached_entry_count": repopulated_cached, "worker_entry_counts": repopulated_worker_counts},
            "first_generation_text": first_text,
            "second_generation_text": second_text,
            "generations_match": first_text.strip() == second_text.strip(),
            "overall": "PASS" if overall_pass else "FAIL",
        }
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nOVERALL: {report.get('overall', 'FAIL (report incomplete)')}")
    if report.get("overall") != "PASS":
        print(
            "Do not proceed to another Stage-7B cache-safety smoke until this passes. A step-6 "
            "failure means vLLM did not cache anything for this request shape on this runtime -- "
            "re-examine the premise before assuming the reset itself is broken. A step-8 failure "
            "means the compatibility path's attribute names (see vlm_adapter.py's own docstring, "
            "divergence #9) are WRONG for this exact pinned build and must be corrected against "
            "the report's own before/after evidence. A step-10 failure after step-8 passing is "
            "the most concerning case: the cache is verifiably clear, but recomputation somehow "
            "did not happen -- inspect the raw generation texts in the report.",
            file=sys.stderr,
        )
    print(f"Wrote {out_path}")
    return 0 if report.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
