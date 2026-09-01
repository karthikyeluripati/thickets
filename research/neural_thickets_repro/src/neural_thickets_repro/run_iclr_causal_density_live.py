"""LIVE execution -- Phase 5 (base-control gate) + Phase 6 (decisive pilot) of the isolated 7B
causal-density pilot. GPU required. Binds the CPU-tested iclr_causal_density.evaluator/driver
injected callables to the real, already-established primitives this repository uses everywhere
else: scoped_perturbation.scoped_apply_perturbation, run_global_visual_thicket_pilot's
reset_to_base_weights_via_rpc/verify_exact_fixed_base_restoration_via_rpc/RayEngineLLMAdapter,
benchmarks.runner.run_benchmark, scopes.scope_requires_encoder_cache_reset,
vlm_adapter.reset_vllm_encoder_cache_full. Never a new perturbation/restoration/scoring
mechanism -- only the orchestration (dataset loading, subset/shuffle-manifest verification,
engine launch, base-control gate, candidate population loop) is new.

STRICTLY 7B-only. Never imports, references, or dispatches anything from the 32B S1/S2
modules -- see test_run_iclr_causal_density_live.py's own structural guard.

Usage (on the pod):
    python -m neural_thickets_repro.run_iclr_causal_density_live --phase base_control
    python -m neural_thickets_repro.run_iclr_causal_density_live --phase decisive_pilot
"""
from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

_FORBIDDEN_32B_72B_SUBSTRINGS = (
    "stage11_coarse_anatomical_atlas_32b", "stage11_32b_s2_live_evidence", "stage11_32b_s2_live_v3_solver_probe",
    "--scale 32B", "--scale 72B", "32B", "72B",
)

# Live fix (image-token-overflow bug, first exposed by a high-resolution TextVQA audit image
# producing a 16,215-token prompt against the frozen max_model_len=4096 budget): caps Qwen2.5-VL's
# own smart-resize vision-token count via the standard `max_pixels` processor kwarg. 1024*28*28
# pixels -> 1024 merged-patch-grid cells -> 256 vision tokens after the model's 2x2 spatial merge,
# leaving ample headroom under 4096 for prompt text + generation on every capability's questions.
# This does NOT change max_model_len, any scoring/perturbation logic, or any frozen design constant
# -- it only bounds how large an input image vLLM's processor is allowed to tokenize.
QWEN2_5_VL_MAX_PIXELS = 1024 * 28 * 28


def _ensure_no_32b_72b_in_argv(argv: Optional[Sequence[str]]) -> None:
    if not argv:
        return
    joined = " ".join(argv)
    for token in _FORBIDDEN_32B_72B_SUBSTRINGS:
        if token in joined:
            raise ValueError(f"run_iclr_causal_density_live.py refuses argv containing {token!r} -- this pilot is strictly 7B-only.")


def _collective_rpc_single_worker(engine: Any, method, args: tuple = (), *, label: str, ray_get=None) -> Any:
    if ray_get is None:
        import ray
        ray_get = ray.get
    results = ray_get(engine.collective_rpc.remote(method, args=args))
    if not isinstance(results, list):
        raise RuntimeError(f"collective_rpc({label!r}) returned {type(results).__name__}, expected a list.")
    if len(results) != 1:
        raise RuntimeError(f"collective_rpc({label!r}) returned {len(results)} per-worker results; this pilot is TP=1-only, expected 1.")
    return results[0]


def _real_apply_perturbation(engine: Any, seed: int, r: float, scope: str, *, ray_get=None) -> Dict[str, Any]:
    """Bound as evaluator.py's own `apply_perturbation` callable. Dispatches the SAME,
    unmodified scoped_apply_perturbation (scoped_perturbation.py) run_scoped_randopt.py
    already uses, in relative_l2 scale mode (the preregistered scale mode -- radii are
    relative-L2 fractions, never raw sigma values).
    """
    from .scoped_perturbation import scoped_apply_perturbation

    return _collective_rpc_single_worker(engine, scoped_apply_perturbation, args=(seed, r, scope, "relative_l2"), label="scoped_apply_perturbation", ray_get=ray_get)


def _real_reset_to_base_weights(engine: Any) -> None:
    from .run_global_visual_thicket_pilot import reset_to_base_weights_via_rpc

    reset_to_base_weights_via_rpc(engine)


def _real_verify_restoration(engine: Any) -> bool:
    from .run_global_visual_thicket_pilot import verify_exact_fixed_base_restoration_via_rpc

    result = verify_exact_fixed_base_restoration_via_rpc(engine)
    return bool(result.get("ok"))


def _real_reset_vllm_encoder_cache_full(engine: Any) -> None:
    from .vlm_adapter import reset_vllm_encoder_cache_full

    reset_vllm_encoder_cache_full(engine)


def _load_capability_pool_and_frozen_subsets(cap: str, adapter, cfg, persisted_subset_manifest, persisted_shuffle_manifests):
    """Reloads capability `cap`'s full pool, rebuilds the frozen selection/audit subsets and
    their frozen shuffled variants, and verifies EVERY rebuild matches the previously-persisted
    manifests exactly (subset_manifest.json / shuffle_manifest.json, both committed to this
    branch) -- proves this run is using the identical frozen data, never silently re-sampled.
    """
    from .iclr_causal_density.shuffle_manifest import build_frozen_shuffled_variant
    from .iclr_causal_density.subsets import build_selection_and_audit_subsets, ensure_subset_manifest_unchanged
    from .iclr_causal_density.design import SHUFFLE_SEED, SUBSET_SELECTION_SEED

    t0 = time.time()
    print(f"[{cap}] loading pool...", flush=True)
    pool = adapter.load_examples(cfg)
    print(f"[{cap}] pool n={len(pool)} ({time.time() - t0:.1f}s)", flush=True)

    selection, audit, current_manifest = build_selection_and_audit_subsets(pool, cap, seed=SUBSET_SELECTION_SEED)
    ensure_subset_manifest_unchanged(persisted_subset_manifest[cap], current_manifest)
    print(f"[{cap}] selection/audit subsets verified identical to the frozen manifest", flush=True)

    selection_shuffled, sel_shuf_manifest = build_frozen_shuffled_variant(selection, cap, "selection", adapter, seed=SHUFFLE_SEED)
    audit_shuffled, aud_shuf_manifest = build_frozen_shuffled_variant(audit, cap, "audit", adapter, seed=SHUFFLE_SEED)
    if sel_shuf_manifest != persisted_shuffle_manifests[f"{cap}:selection"]:
        raise RuntimeError(f"[{cap}] rebuilt selection shuffle manifest does not match the frozen, committed one.")
    if aud_shuf_manifest != persisted_shuffle_manifests[f"{cap}:audit"]:
        raise RuntimeError(f"[{cap}] rebuilt audit shuffle manifest does not match the frozen, committed one.")
    print(f"[{cap}] shuffle manifests verified identical to the frozen manifest ({time.time() - t0:.1f}s total)", flush=True)

    from .benchmarks.image_sanity import make_text_only_variant

    text_only_supported = adapter.supports_text_only_condition()
    return {
        "selection_correct": selection, "selection_shuffled": selection_shuffled,
        "selection_text_only": make_text_only_variant(selection) if text_only_supported else None,
        "audit_correct": audit, "audit_shuffled": audit_shuffled,
        "audit_text_only": make_text_only_variant(audit) if text_only_supported else None,
        "text_only_supported": text_only_supported,
    }


def _score_condition(run_benchmark, benchmark, examples, llm_adapter, tokenizer, sampling_params, generation_batch_size: int, *, allow_missing_image: bool = False) -> Dict[str, Any]:
    result = run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, allow_missing_image=allow_missing_image, max_requests_per_generate=generation_batch_size)
    return {
        "aggregate_score": result.aggregate_metrics["primary_metric"],
        "parser_failure_rate": result.aggregate_metrics.get("parser_failure_rate"),
        "per_example_scores": {r.example_id: r.score.score for r in result.per_example},
        "generation_hash": result.generation_hash(),
    }


def run_base_control_gate(engine, capability_frozen_data: Dict[str, Dict[str, Any]], adapters: Dict[str, Any], tokenizer, sampling_params, *, generation_batch_size: int, run_benchmark) -> Dict[str, Any]:
    """Phase 5: unperturbed model, all 5 capabilities, both subsets, all 3 (or 2, where text-
    only is unsupported) visual conditions. Records deterministic per-condition scores; the
    gate itself (meaningful visual advantage, complete schema-valid output) is evaluated by the
    caller from this report.
    """
    from .run_global_visual_thicket_pilot import RayEngineLLMAdapter

    llm_adapter = RayEngineLLMAdapter(engine)
    report: Dict[str, Any] = {}
    for cap, data in capability_frozen_data.items():
        adapter = adapters[cap]
        cap_report: Dict[str, Any] = {}
        for subset_role in ("selection", "audit"):
            for condition, key in (("correct_image", f"{subset_role}_correct"), ("shuffled_image", f"{subset_role}_shuffled"), ("text_only", f"{subset_role}_text_only")):
                examples = data[key]
                if examples is None:
                    cap_report[f"{subset_role}:{condition}"] = None
                    continue
                print(f"[base-control] {cap} {subset_role} {condition} (n={len(examples)})...", flush=True)
                cap_report[f"{subset_role}:{condition}"] = _score_condition(
                    run_benchmark, adapter, examples, llm_adapter, tokenizer, sampling_params, generation_batch_size,
                    allow_missing_image=(condition == "text_only"),
                )
        report[cap] = cap_report
    return report


def evaluate_base_control_gate(report: Dict[str, Any]) -> Dict[str, Any]:
    """Applies the preregistered base-control gate criteria (task spec Phase 5): deterministic
    evaluation (checked separately by a repeat pass, not here), valid correct-image
    performance, meaningful visual advantage over shuffled/text-only, complete schema-valid
    output for every (capability, subset, condition) cell that should exist.
    """
    failures = []
    for cap, cap_report in report.items():
        for subset_role in ("selection", "audit"):
            correct = cap_report.get(f"{subset_role}:correct_image")
            shuffled = cap_report.get(f"{subset_role}:shuffled_image")
            text_only = cap_report.get(f"{subset_role}:text_only")
            if correct is None or shuffled is None:
                failures.append(f"{cap}/{subset_role}: missing correct_image or shuffled_image result")
                continue
            if correct["aggregate_score"] <= shuffled["aggregate_score"]:
                failures.append(f"{cap}/{subset_role}: no visual advantage over shuffled_image (correct={correct['aggregate_score']}, shuffled={shuffled['aggregate_score']})")
            if text_only is not None and correct["aggregate_score"] <= text_only["aggregate_score"]:
                failures.append(f"{cap}/{subset_role}: no visual advantage over text_only (correct={correct['aggregate_score']}, text_only={text_only['aggregate_score']})")
    return {"pass": not failures, "failures": failures}


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_no_32b_72b_in_argv(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["base_control", "decisive_pilot"], required=True)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "iclr_causal_density"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--generation-batch-size", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true", default=True)
    args = parser.parse_args(argv)

    from .iclr_causal_density.design import CAPABILITIES, FROZEN_DESIGN
    from .iclr_causal_density.subsets import load_subset_manifest
    from .iclr_causal_density.shuffle_manifest import load_shuffle_manifest
    from .config import load_capability_benchmark_config
    from .run_capability_benchmark_gate import load_adapter
    from .scaling_common import resolve_immutable_model_revision

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    persisted_subset_manifest = load_subset_manifest(REPO_ROOT / "reports" / "iclr_causal_density" / "subset_manifest.json")
    persisted_shuffle_manifests = load_shuffle_manifest(REPO_ROOT / "reports" / "iclr_causal_density" / "shuffle_manifest.json")

    resolution = resolve_immutable_model_revision(FROZEN_DESIGN.model_name, "main")
    print(f"Resolved model revision: {resolution}", flush=True)
    (output_root / "model_revision_resolution.json").write_text(json.dumps(resolution, indent=2))

    capability_configs = {}
    adapters = {}
    for cap in CAPABILITIES:
        cfg = load_capability_benchmark_config(REPO_ROOT / "configs" / "benchmarks" / f"{cap}.yaml")
        capability_configs[cap] = cfg
        adapters[cap] = load_adapter(cfg.dataset.adapter)

    capability_frozen_data = {}
    for cap in CAPABILITIES:
        capability_frozen_data[cap] = _load_capability_pool_and_frozen_subsets(cap, adapters[cap], capability_configs[cap], persisted_subset_manifest, persisted_shuffle_manifests)

    print("All 5 capabilities' frozen selection/audit/shuffle subsets verified identical to the committed manifests.", flush=True)

    import ray
    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines  # type: ignore
    from .vlm_adapter import bootstrap_ray, ensure_full_encoder_cache_reset_exposed, resolve_model_snapshot, verify_workers_can_import_external_root
    from .run_global_visual_thicket_pilot import launch_stage6_engine, store_base_weights_via_rpc

    resolved_snapshot_path = resolve_model_snapshot(FROZEN_DESIGN.model_name, resolution["resolved_revision"])
    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(resolved_snapshot_path)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    engines, pgs = None, None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        # Live regression (decisive-pilot candidate #1, vision_encoder scope): MUST run before
        # launch_stage6_engine -- see vlm_adapter.py's own ensure_full_encoder_cache_reset_exposed
        # docstring. Without this, the Ray-wrapped RandOptNcclLLM actor never exposes
        # 'reset_encoder_cache_full', and reset_vllm_encoder_cache_full (needed for every
        # vision_encoder/full_vlm candidate, per scopes.scope_requires_encoder_cache_reset) hard-
        # fails rather than silently falling back to the worker-only reset already confirmed
        # insufficient on GPU. Unconditional, mirroring run_global_visual_thicket_pilot.py's own
        # Stage 6 main() exactly -- both base_control and decisive_pilot pay this one-time,
        # inexpensive monkey-patch cost regardless of phase.
        ensure_full_encoder_cache_reset_exposed(EXTERNAL_ROOT)

        engines, pgs = launch_stage6_engine(
            resolved_snapshot_path, precision="bfloat16", tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            mm_processor_kwargs={"max_pixels": QWEN2_5_VL_MAX_PIXELS},
        )
        engine = engines[0]
        store_base_weights_via_rpc(engine)
        print("Confirmed working CPU/GPU base snapshot (store_base_weights_via_rpc).", flush=True)

        # HARD FAIL BEFORE any candidate evaluation if the cache reset mechanism doesn't
        # actually WORK end-to-end against the LIVE engine (not merely that it was exposed
        # pre-launch) -- the same one-time precondition-verification discipline Stage 6's own
        # main() already established, reused here rather than reinvented.
        try:
            _real_reset_vllm_encoder_cache_full(engine)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"This pilot requires a working full multimodal-encoder-cache reset -- "
                f"verification failed against the live engine ({type(exc).__name__}: {exc}). "
                f"Refusing to start candidate evaluation without a proven-working cache-"
                f"invalidation path."
            ) from exc
        print("Confirmed working multimodal-encoder-cache reset.", flush=True)

        from .benchmarks.runner import run_benchmark

        if args.phase == "base_control":
            report = run_base_control_gate(engine, capability_frozen_data, adapters, tokenizer, sampling_params, generation_batch_size=args.generation_batch_size, run_benchmark=run_benchmark)
            (output_root / "base_control_report.json").write_text(json.dumps(report, indent=2, default=str))
            gate = evaluate_base_control_gate(report)
            (output_root / "base_control_gate.json").write_text(json.dumps(gate, indent=2))
            print(f"BASE CONTROL GATE: {'PASS' if gate['pass'] else 'FAIL'}", flush=True)
            for f in gate["failures"]:
                print(f"  FAIL: {f}", file=sys.stderr)
            return 0 if gate["pass"] else 1

        # decisive_pilot
        from .iclr_causal_density.candidates import build_candidate_population, validate_candidate_population, write_candidate_manifest
        from .iclr_causal_density.driver import run_candidate_population_rpc, summarize_population_run
        from .iclr_causal_density.evaluator import CapabilityAuditData, evaluate_one_candidate_all_capabilities
        from .run_global_visual_thicket_pilot import RayEngineLLMAdapter
        from .scopes import scope_requires_encoder_cache_reset

        candidates = build_candidate_population()
        validate_candidate_population(candidates)
        write_candidate_manifest(candidates, output_root / "candidate_manifest.json")
        print(f"Candidate population built: {len(candidates)} candidates.", flush=True)

        capability_audit_data = {
            cap: CapabilityAuditData(
                capability=cap, benchmark=adapters[cap], dataset_source=capability_configs[cap].dataset.source,
                correct_examples=data["audit_correct"], shuffled_examples=data["audit_shuffled"],
                text_only_examples=data["audit_text_only"],
            )
            for cap, data in capability_frozen_data.items()
        }

        # Live regression (decisive-pilot candidate #1): `engine` is the raw Ray actor handle --
        # only .generate.remote() is callable on it directly, never .generate(). run_benchmark
        # needs a synchronous-generate() adapter, exactly like run_base_control_gate already
        # builds above -- built once here, reused for every one of the 600 candidates.
        llm_adapter = RayEngineLLMAdapter(engine)

        def _bound_evaluate_one_candidate(candidate):
            return evaluate_one_candidate_all_capabilities(
                engine, candidate, capability_audit_data, tokenizer, sampling_params,
                run_benchmark=run_benchmark, apply_perturbation=_real_apply_perturbation,
                reset_to_base_weights=_real_reset_to_base_weights, scope_requires_encoder_cache_reset=scope_requires_encoder_cache_reset,
                reset_vllm_encoder_cache_full=_real_reset_vllm_encoder_cache_full, verify_restoration=_real_verify_restoration,
                scope_isolation_precondition_ok=True, decoding_config={"temperature": 0.0, "max_tokens": 256},
                source_commit=resolution.get("resolved_revision", "unknown"), run_id=f"iclr_causal_density_pilot_{int(time.time())}",
                model_name=FROZEN_DESIGN.model_name, model_revision=resolution["resolved_revision"], llm=llm_adapter,
            )

        def _progress(outcome):
            print(f"  [{outcome.status}] {outcome.candidate_id} (scope={outcome.scope}, radius={outcome.radius}, seed={outcome.seed}) n_rows={outcome.n_rows}" + (f" error={outcome.error}" if outcome.error else ""), flush=True)

        results_path = output_root / "results.jsonl"
        outcomes = run_candidate_population_rpc(
            candidates, results_path, evaluate_one_candidate=_bound_evaluate_one_candidate,
            expected_capabilities=CAPABILITIES, expected_conditions=("correct_image", "shuffled_image", "text_only"),
            fail_fast=args.fail_fast, progress_callback=_progress,
        )
        summary = summarize_population_run(candidates, outcomes, expected_rows_per_candidate=len(CAPABILITIES) * 3)
        (output_root / "run_summary.json").write_text(json.dumps(summary.to_dict(), indent=2))
        print(f"Run summary: {json.dumps(summary.to_dict(), indent=2)}", flush=True)
        return 0 if summary.run_complete else 1
    finally:
        if engines is not None:
            cleanup_engines(engines, pgs)
        elif ray_owned_by_us and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
