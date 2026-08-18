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

Restoration mode (required, --restoration-mode, never defaulted): diagnostics/
gate2_restoration_ab.py's GPU A/B evidence found that a single perturb_self_weights ->
restore_self_weights cycle (seed=999999999, sigma=0.01) leaves visible residual drift
(max abs parameter drift 3.125e-02 after 10 repeated same-seed cycles, roughly stable rather
than growing -- raw-output restore only 1/5, extracted-answer restore 3/5), while
apply_perturbation -> reset_to_base_weights showed exactly zero drift and 5/5 exact restore
on both raw output and extracted answer. Both are real, reused-unmodified WorkerExtension
mechanisms (utils/worker_extn.py) -- which one is "correct" depends on what's being asked:
  released_compat: perturb_self_weights(seed, sigma, False) -> evaluate -> restore_self_weights
    (seed, sigma, False). Behaviorally identical to released RandOpt's own perturb/restore
    (each candidate's weights are a delta from whatever the CURRENT, possibly-still-drifted
    state is) -- use this for paper-code reproduction / Gate 2 mechanics fidelity.
  fixed_base: apply_perturbation(seed, sigma) -> evaluate -> reset_to_base_weights(). Every
    candidate is independently sampled from the exact stored pretrained base weights
    launch_engines() already keeps, never from a possibly-drifted current state -- use this
    for WACV-extension experiments, where independent-per-candidate sampling around theta is
    the actual scientific claim being tested, not released-code fidelity.
These are two different scientific interpretations of "perturb from theta," not a bugfix
replacing the other -- --restoration-mode is required with no default so a run's results are
never ambiguous about which one produced them. See _perturb_call/_restore_call below for the
exact per-mode call shapes, and REPRO_SPEC.md's "Gate 2 restoration semantics" row for the
full A/B evidence this decision is based on.

Usage (smoke test, GPU required):
    python -m neural_thickets_repro.run_randopt_image_aware \
        --config configs/gqa_repro.yaml --sigma-candidate sigma_default --N 20 --K 5 \
        --restoration-mode released_compat

--test-samples optionally caps the ensemble-evaluation test set (upstream default, and this
script's default, is None = the full 12,578 testdev examples per expert -- matching upstream
fidelity but expensive at K>1; pass e.g. --test-samples 200 for a genuinely fast mechanical
smoke test of the pipeline itself).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
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


def _git_commit() -> str:
    """Mirrors eval_base_image_aware.py's own helper (small enough to keep this script
    self-contained rather than importing a private helper across an unrelated Gate-1 module).
    """
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


def _assert_spawn_configured() -> None:
    value = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if value != "spawn":
        raise RuntimeError(
            f"VLLM_WORKER_MULTIPROC_METHOD must be 'spawn' before vLLM initializes, got "
            f"{value!r}. Not a reproduction-behavior issue; a runtime multiprocessing/CUDA "
            f"launch misconfiguration."
        )


# Confirmed against the pinned utils/worker_extn.py:WorkerExtension checkout (see module
# docstring's "Restoration mode" section and REPRO_SPEC.md). Neither method name here is
# reimplemented -- both are dispatched via Ray collective_rpc, unmodified upstream code.
RESTORATION_MODES: Tuple[str, ...] = ("released_compat", "fixed_base")


def _perturb_call(restoration_mode: str, seed: int, sigma: float) -> Tuple[str, tuple]:
    """(method_name, args) for WorkerExtension's perturb-side call, per restoration mode.
    released_compat: perturb_self_weights(seed, sigma, negate=False).
    fixed_base: apply_perturbation(seed, sigma) -- no negate/sync flag; this is the ONLY
    place restoration_mode affects candidate sampling/selection/voting -- it changes which
    WorkerExtension method is dispatched and with what args, nothing about which (seed,
    sigma) pairs are drawn or how scores/votes are computed from the resulting generations.
    """
    if restoration_mode == "released_compat":
        return "perturb_self_weights", (seed, sigma, False)
    if restoration_mode == "fixed_base":
        return "apply_perturbation", (seed, sigma)
    raise ValueError(f"Unknown restoration_mode {restoration_mode!r}, expected one of {RESTORATION_MODES}")


def _restore_call(restoration_mode: str, seed: int, sigma: float) -> Tuple[str, tuple]:
    """(method_name, args) for WorkerExtension's restore-side call, per restoration mode.
    released_compat: restore_self_weights(seed, sigma, negate=False) -- must reuse the SAME
    (seed, sigma) as the matching perturb call, or the result isn't the original weights.
    fixed_base: reset_to_base_weights() -- no arguments at all; (seed, sigma) are accepted
    here only so both restore functions share one call signature, then discarded, since
    reset_to_base_weights derives everything from the base weights launch_engines() already
    stores, not from the just-applied perturbation delta.
    """
    if restoration_mode == "released_compat":
        return "restore_self_weights", (seed, sigma, False)
    if restoration_mode == "fixed_base":
        return "reset_to_base_weights", ()
    raise ValueError(f"Unknown restoration_mode {restoration_mode!r}, expected one of {RESTORATION_MODES}")


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
                        candidates: List[Tuple[int, float]], ledger: CandidateLedger,
                        restoration_mode: str) -> Dict[Tuple[int, float], float]:
    """Perturb -> generate on the selection set (image-aware) -> score -> restore, once per
    candidate. Resumable: candidates already marked "done" in the ledger are skipped and
    their stored score reused, so an interrupted run doesn't repeat completed work.

    restoration_mode selects ONLY which WorkerExtension perturb/restore call is dispatched
    (see _perturb_call/_restore_call) -- candidate sampling, scoring, and everything else
    below is identical regardless of mode.
    """
    import ray

    scores: Dict[Tuple[int, float], float] = {}
    for rec in ledger.load_all().values():
        if rec.status == "done" and rec.selection_score is not None:
            scores[(rec.seed, rec.sigma)] = rec.selection_score

    for candidate_id in ledger.iter_pending(range(len(candidates))):
        seed, sigma = candidates[candidate_id]
        start = time.time()

        perturb_method, perturb_args = _perturb_call(restoration_mode, seed, sigma)
        restore_method, restore_args = _restore_call(restoration_mode, seed, sigma)

        ray.get(engine.collective_rpc.remote(perturb_method, args=perturb_args))
        outputs = ray.get(engine.generate.remote(selection_requests, sampling_params, use_tqdm=False))
        responses = [o.outputs[0].text for o in outputs]
        reward = float(np.mean([
            handler.compute_reward(resp, d["ground_truth"]) for resp, d in zip(responses, selection_datas)
        ])) if responses else 0.0
        ray.get(engine.collective_rpc.remote(restore_method, args=restore_args))

        scores[(seed, sigma)] = reward
        ledger.append(CandidateRecord(
            candidate_id=candidate_id, seed=seed, sigma=sigma, selection_score=reward,
            rank=None, status="done", runtime_seconds=time.time() - start,
            restoration_mode=restoration_mode,
        ))
        print(f"  candidate {candidate_id + 1}/{len(candidates)}: seed={seed} sigma={sigma} score={reward:.4f}")

    return scores


def run_ensemble_phase(engine, handler, test_requests, test_datas, sampling_params,
                        top_k_candidates: List[Tuple[int, float]], restoration_mode: str) -> Dict:
    """Perturb -> generate on the test set (image-aware) -> extract answers -> restore, for
    each of the top-K candidates (ordered highest-selection-score first, matching upstream),
    then majority-vote per example via topk_voting.majority_vote.

    restoration_mode selects ONLY which WorkerExtension perturb/restore call is dispatched
    (see _perturb_call/_restore_call) -- voting is identical regardless of mode.
    """
    import ray

    all_answers: List[List[str]] = []
    for rank, (seed, sigma) in enumerate(top_k_candidates):
        perturb_method, perturb_args = _perturb_call(restoration_mode, seed, sigma)
        restore_method, restore_args = _restore_call(restoration_mode, seed, sigma)

        ray.get(engine.collective_rpc.remote(perturb_method, args=perturb_args))
        outputs = ray.get(engine.generate.remote(test_requests, sampling_params, use_tqdm=False))
        answers = [handler.extract_answer_for_voting(o.outputs[0].text) or "" for o in outputs]
        ray.get(engine.collective_rpc.remote(restore_method, args=restore_args))
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
        "--restoration-mode", required=True, choices=RESTORATION_MODES,
        help=(
            "released_compat (perturb_self_weights/restore_self_weights, paper-code "
            "reproduction) or fixed_base (apply_perturbation/reset_to_base_weights, "
            "WACV-ready independent-per-candidate sampling) -- two different scientific "
            "interpretations of 'perturb from theta', not a bugfix for the other. Required, "
            "never defaulted -- see module docstring's 'Restoration mode' section."
        ),
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

    # restoration_mode is part of the default out_dir path (not just an internal field) so
    # that released_compat and fixed_base runs sharing the same N/K/sigma_candidate never
    # collide on the same candidates.jsonl -- their scores for the "same" (seed, sigma)
    # candidate are NOT interchangeable (different restoration mechanics), so a shared ledger
    # would silently mix modes together on resume.
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"randopt_image_aware_N{args.N}_K{args.K}_{args.sigma_candidate}_{args.restoration_mode}"
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
    print(f"Restoration mode: {args.restoration_mode}")

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
            print(f"\n=== Sampling phase: N={args.N} candidates ({args.restoration_mode}) ===")
            scores = run_sampling_phase(
                engine, handler, selection_requests, selection_datas, sampling_params,
                candidates, ledger, args.restoration_mode,
            )

            top_k = select_top_k(scores, args.K)
            print(f"\n=== Selected top-{args.K} of {args.N} ===")
            for rank, (seed, sigma) in enumerate(top_k):
                print(f"  {rank + 1}. seed={seed} sigma={sigma} score={scores[(seed, sigma)]:.4f}")

            print(f"\n=== Ensemble evaluation: K={args.K} on {len(test_datas)} test examples ({args.restoration_mode}) ===")
            ensemble_results = run_ensemble_phase(
                engine, handler, test_requests, test_datas, sampling_params, top_k, args.restoration_mode,
            )
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
        "restoration_mode": args.restoration_mode,
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

    run_metadata = {
        "restoration_mode": args.restoration_mode,
        "restoration_mechanism": {
            "released_compat": "perturb_self_weights(seed, sigma, False) / restore_self_weights(seed, sigma, False)",
            "fixed_base": "apply_perturbation(seed, sigma) / reset_to_base_weights()",
        }[args.restoration_mode],
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "model_snapshot_path": model_path,
        "N": args.N, "K": args.K,
        "sigma_candidate": args.sigma_candidate, "sigma_values": sigma_values,
        "selection_set_size": len(selection_datas),
        "test_set_size": len(test_datas),
        "global_seed": cfg.reproducibility.global_seed,
        "our_repo_git_commit": _git_commit(),
        "external_randopt_commit": "536df0a308f3990b6270c991fbb96bd0b779a58e",
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    print("\n=== Gate 2 smoke test complete ===")
    print(f"Restoration mode: {args.restoration_mode}")
    print(f"Ensemble (K={args.K}) accuracy: {ensemble_results['accuracy']:.4f} ({ensemble_results['correct']}/{ensemble_results['n']})")
    print(f"Wrote {results_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {run_metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
