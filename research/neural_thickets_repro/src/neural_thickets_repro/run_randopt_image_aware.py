"""Gate 2: small-scale RandOpt reproduction, image-aware. PREPARES ONLY -- not executed by
us here (no GPU on this machine). GPU required to actually run.

Why this exists instead of reusing run_randopt.py: GATE1_DIAGNOSIS.md confirmed on GPU that
the released RandOpt code never constructs multi_modal_data, so images never reach the
model -- the exact bug that caused the Gate 1 hard fail (17.94% vs published 56.6%; fixed
for the baseline case by eval_base_image_aware.py, reaching 54.19% march-era scoring).
run_randopt.py subprocess-wraps the UNMODIFIED external randopt.py, which would hit the
identical image-blindness in its candidate-sampling and ensemble loops -- reusing it here
would reproduce a known-broken result, not a meaningful Gate 2 test. This script is the
image-aware equivalent for Gate 2, same relationship eval_base_image_aware.py has to
eval_base.py for Gate 1.

Reused UNMODIFIED from external/RandOpt (see REPRO_SPEC.md for citations -- external/RandOpt
itself is never edited):
  - core/engine.py: launch_engines / cleanup_engines (Ray + vLLM engine lifecycle), called
    WITH multimodal=True this time since we now actually attach multi_modal_data.
  - utils/worker_extn.py: WorkerExtension.perturb_self_weights / restore_self_weights,
    invoked via Ray collective_rpc -- the ACTUAL weight-perturbation mechanism. Not
    reimplemented: reimplementing risks silently diverging from the paper's method.
  - data_handlers/gqa.py: GQAHandler -- data loading, answer extraction/normalization,
    scoring (compute_reward, extract_answer_for_voting, is_voted_answer_correct).

Ours (Gate-0-validated pure Python, no vLLM/GPU dependency):
  - sample_candidates(): mirrors randopt.py:run_sampling's seed/sigma assignment scheme
    (np.random.default_rng, N unique seeds without replacement, sigma with replacement --
    described in REPRO_SPEC.md, not copied).
  - topk_voting.select_top_k / majority_vote (tests/test_topk_voting.py).
  - ledger.CandidateLedger for resumability (tests/test_ledger.py).
  - vlm_adapter.build_image_aware_requests -- the ONE deliberate deviation from the released
    code, identical to eval_base_image_aware.py's fix: prompts carry multi_modal_data.
    Nothing else about the algorithm changes.

Known simplification (documented, not hidden): num_engines is fixed at 1 for this
smoke-test scale (correctness over throughput -- randopt.py's own multi-engine batching
across run_sampling/run_ensemble_evaluation is a throughput optimization, not a correctness
requirement at N=20-ish scale). A full N=5000 run would want to match that batching; not
implemented here, out of scope until a smoke test is reviewed and scaling up is authorized.

Usage (smoke test, GPU required):
    python -m neural_thickets_repro.run_randopt_image_aware \
        --config configs/gqa_repro.yaml --sigma-candidate sigma_default --N 20 --K 5

--test-samples optionally caps the ensemble-evaluation test set (upstream default, and this
script's default, is None = the full 12,578 testdev examples per expert -- matching upstream
fidelity but expensive at K>1; pass e.g. --test-samples 200 for a genuinely fast mechanical
smoke test of the pipeline itself).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Same runtime-compatibility fix as eval_base_image_aware.py, forced before any
# torch/vllm/ray import anywhere in this process -- see that module for the full
# explanation (RuntimeError: Cannot re-initialize CUDA in forked subprocess). Not a
# reproduction-behavior change.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np

from .config import load_config
from .env_check import (
    GateBlockedError,
    assert_feasible,
    check_cuda,
    check_disk,
    check_gate_artifact,
    check_module,
)
from .ledger import CandidateLedger, CandidateRecord
from .topk_voting import majority_vote, select_top_k
from .vlm_adapter import (
    bootstrap_ray,
    build_image_aware_requests,
    resolve_model_snapshot,
    verify_workers_can_import_external_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

# The ACCEPTED Gate 1 result (image-aware full baseline), not the old blind results/base/.
GATE1_ARTIFACT = REPO_ROOT / "results" / "base_image_aware" / "metrics.json"


def _assert_spawn_configured() -> None:
    value = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if value != "spawn":
        raise RuntimeError(
            f"VLLM_WORKER_MULTIPROC_METHOD must be 'spawn' before vLLM initializes, got "
            f"{value!r}. Not a reproduction-behavior issue; a runtime multiprocessing/CUDA "
            f"launch misconfiguration."
        )


def sample_candidates(n: int, sigma_values: List[float], seed: int) -> List[Tuple[int, float]]:
    """Mirrors randopt.py:run_sampling's candidate assignment (described, not copied -- see
    REPRO_SPEC.md "Candidate (seed, sigma) assignment"): N unique seeds drawn without
    replacement, one sigma per candidate drawn WITH replacement from sigma_values, both from
    a single np.random.default_rng(seed) stream, in that order.
    """
    rng = np.random.default_rng(seed=seed)
    seeds = rng.choice(2**31, size=n, replace=False).tolist()
    sigmas = rng.choice(sigma_values, size=n).tolist()
    return list(zip((int(s) for s in seeds), (float(s) for s in sigmas)))


def run_sampling_phase(engine, handler, selection_requests, selection_datas, sampling_params,
                        candidates: List[Tuple[int, float]], ledger: CandidateLedger) -> Dict[Tuple[int, float], float]:
    """Perturb -> generate on the selection set (image-aware) -> score -> restore, once per
    candidate. Resumable: candidates already marked "done" in the ledger are skipped and
    their stored score reused, so an interrupted run doesn't repeat completed work.
    """
    import ray

    scores: Dict[Tuple[int, float], float] = {}
    for rec in ledger.load_all().values():
        if rec.status == "done" and rec.selection_score is not None:
            scores[(rec.seed, rec.sigma)] = rec.selection_score

    for candidate_id in ledger.iter_pending(range(len(candidates))):
        seed, sigma = candidates[candidate_id]
        start = time.time()

        ray.get(engine.collective_rpc.remote("perturb_self_weights", args=(seed, sigma, False)))
        outputs = ray.get(engine.generate.remote(selection_requests, sampling_params, use_tqdm=False))
        responses = [o.outputs[0].text for o in outputs]
        reward = float(np.mean([
            handler.compute_reward(resp, d["ground_truth"]) for resp, d in zip(responses, selection_datas)
        ])) if responses else 0.0
        ray.get(engine.collective_rpc.remote("restore_self_weights", args=(seed, sigma, False)))

        scores[(seed, sigma)] = reward
        ledger.append(CandidateRecord(
            candidate_id=candidate_id, seed=seed, sigma=sigma, selection_score=reward,
            rank=None, status="done", runtime_seconds=time.time() - start,
        ))
        print(f"  candidate {candidate_id + 1}/{len(candidates)}: seed={seed} sigma={sigma} score={reward:.4f}")

    return scores


def run_ensemble_phase(engine, handler, test_requests, test_datas, sampling_params,
                        top_k_candidates: List[Tuple[int, float]]) -> Dict:
    """Perturb -> generate on the test set (image-aware) -> extract answers -> restore, for
    each of the top-K candidates (ordered highest-selection-score first, matching upstream),
    then majority-vote per example via topk_voting.majority_vote.
    """
    import ray

    all_answers: List[List[str]] = []
    for rank, (seed, sigma) in enumerate(top_k_candidates):
        ray.get(engine.collective_rpc.remote("perturb_self_weights", args=(seed, sigma, False)))
        outputs = ray.get(engine.generate.remote(test_requests, sampling_params, use_tqdm=False))
        answers = [handler.extract_answer_for_voting(o.outputs[0].text) or "" for o in outputs]
        ray.get(engine.collective_rpc.remote("restore_self_weights", args=(seed, sigma, False)))
        all_answers.append(answers)
        print(f"  ensemble member {rank + 1}/{len(top_k_candidates)}: seed={seed} sigma={sigma} done")

    correct = 0
    predictions = []
    for idx, d in enumerate(test_datas):
        voted = majority_vote(all_answers, idx)
        is_correct = bool(voted) and handler.is_voted_answer_correct(voted, d["ground_truth"])
        correct += int(is_correct)
        predictions.append({
            "example_id": str(d["question_id"]),
            "reference_answer": d["ground_truth"]["answer"],
            "voted_answer": voted,
            "correct": is_correct,
            "expert_answers": [ans[idx] for ans in all_answers],
        })

    n = len(test_datas)
    return {"accuracy": correct / n if n else 0.0, "n": n, "correct": correct, "predictions": predictions}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--N", type=int, default=20, help="population size -- smoke-test default, NOT cfg.randopt.N (5000); pass explicitly for a larger run")
    parser.add_argument("--K", type=int, default=5, help="ensemble size -- smoke-test default, NOT cfg.randopt.K (50)")
    parser.add_argument(
        "--sigma-candidate", required=True,
        help="name of a config.randopt.sigma_candidates entry -- required, never defaulted",
    )
    parser.add_argument(
        "--test-samples", type=int, default=None,
        help="cap the ensemble-evaluation test set (default: None = all 12,578, matching upstream); use a small number for a fast mechanical smoke test",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.sigma_candidate not in cfg.randopt.sigma_candidates:
        print(
            f"--sigma-candidate must be one of {sorted(cfg.randopt.sigma_candidates)}, got "
            f"{args.sigma_candidate!r}. Sigma is a first-class unresolved reproduction "
            f"variable (see REPRO_SPEC.md) -- never silently defaulted.",
            file=sys.stderr,
        )
        return 1
    sigma_values = cfg.randopt.sigma_candidates[args.sigma_candidate]

    try:
        assert_feasible(
            "Gate 2 (image-aware RandOpt smoke test)",
            [
                check_cuda(), check_module("vllm"), check_module("ray"),
                check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb),
                check_gate_artifact(GATE1_ARTIFACT),
            ],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        print("Run `python -m neural_thickets_repro.validate_env` for the full picture.", file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision", "dataset.selection_split", "dataset.test_split")

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"randopt_image_aware_N{args.N}_K{args.K}_{args.sigma_candidate}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = CandidateLedger(out_dir / "candidates.jsonl")

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    _assert_spawn_configured()
    import ray  # used directly below (ray.is_initialized()) and via engine.*.remote calls
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines, launch_engines  # type: ignore
    from data_handlers.gqa import GQAHandler  # type: ignore

    handler = GQAHandler()
    selection_datas = handler.load_data(
        str(EXTERNAL_ROOT / "data" / "gqa" / "train.parquet"), split="train",
        max_samples=cfg.dataset.selection_set_size,
    )
    test_datas = handler.load_data(
        str(EXTERNAL_ROOT / "data" / "gqa" / "testdev.parquet"), split="test",
        max_samples=args.test_samples,
    )
    selection_datas = [d for d in selection_datas if "image_path" in d]
    test_datas = [d for d in test_datas if "image_path" in d]
    print(f"Selection set: {len(selection_datas)} examples | Test set: {len(test_datas)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    selection_requests = build_image_aware_requests(selection_datas, tokenizer)
    test_requests = build_image_aware_requests(test_datas, tokenizer)
    sampling_params = SamplingParams(temperature=0.0, seed=cfg.reproducibility.global_seed, max_tokens=cfg.evaluation.max_tokens)

    candidates = sample_candidates(args.N, sigma_values, cfg.reproducibility.global_seed)

    print(f"Sigma candidate: {args.sigma_candidate} = {sigma_values}  (UNRESOLVED assumption, see REPRO_SPEC.md)")

    # Mirrors upstream randopt.py:main()'s own Ray bootstrap -- launch_engines() assumes an
    # active Ray session and never starts one itself (see vlm_adapter.py divergence #5).
    # ray_owned_by_us tracks whether THIS call started Ray, so we only shut down a session
    # we actually own, never one we merely connected to.
    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines = None
    pgs = None
    try:
        # Catches "ModuleNotFoundError: No module named 'core'" on Ray workers in seconds
        # via a trivial remote call, before the slower/more opaque launch_engines()
        # actor-startup path (placement groups + vLLM engine init) would otherwise surface
        # it. Run unconditionally, regardless of ray_owned_by_us -- see vlm_adapter.py
        # divergence #6. Inside this try so a failure here still reaches the Ray-shutdown
        # finally below, same as a launch_engines() failure would.
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        # enable_prefix_caching intentionally NOT overridden: launch_engines' own default is
        # False, which is what makes it safe to repeatedly re-generate the same 200 selection-
        # set prompts across different perturbed weight states without stale KV-cache reuse --
        # verified against the pinned source, see GATE2_CACHE_SAFETY_REVIEW.md. Do not set this
        # to True without re-reading that review first.
        engines, pgs = launch_engines(1, model_path, precision=cfg.model.precision, tensor_parallel_size=1, multimodal=True)
        engine = engines[0]
        try:
            print(f"\n=== Sampling phase: N={args.N} candidates ===")
            scores = run_sampling_phase(engine, handler, selection_requests, selection_datas, sampling_params, candidates, ledger)

            top_k = select_top_k(scores, args.K)
            print(f"\n=== Selected top-{args.K} of {args.N} ===")
            for rank, (seed, sigma) in enumerate(top_k):
                print(f"  {rank + 1}. seed={seed} sigma={sigma} score={scores[(seed, sigma)]:.4f}")

            print(f"\n=== Ensemble evaluation: K={args.K} on {len(test_datas)} test examples ===")
            ensemble_results = run_ensemble_phase(engine, handler, test_requests, test_datas, sampling_params, top_k)
        finally:
            # cleanup_engines (unmodified upstream) already calls ray.shutdown()
            # unconditionally once engines exist -- do not shut down again below.
            cleanup_engines(engines, pgs)
    finally:
        # Only reached without cleanup_engines having run if launch_engines() itself threw
        # before returning engines/pgs -- in that case Ray was never shut down via the path
        # above. Shut it down ourselves, but only if we're the ones who started it.
        if engines is None and ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps({
        "N": args.N, "K": args.K, "sigma_candidate": args.sigma_candidate, "sigma_values": sigma_values,
        "top_k_candidates": [{"seed": s, "sigma": sg, "selection_score": scores[(s, sg)]} for s, sg in top_k],
        "ensemble_accuracy": ensemble_results["accuracy"],
        "ensemble_n": ensemble_results["n"],
        "ensemble_correct": ensemble_results["correct"],
    }, indent=2))
    predictions_path = out_dir / "ensemble_predictions.jsonl"
    with predictions_path.open("w") as f:
        for rec in ensemble_results["predictions"]:
            f.write(json.dumps(rec) + "\n")

    print("\n=== Gate 2 smoke test complete ===")
    print(f"Ensemble (K={args.K}) accuracy: {ensemble_results['accuracy']:.4f} ({ensemble_results['correct']}/{ensemble_results['n']})")
    print(f"Wrote {results_path}")
    print(f"Wrote {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
