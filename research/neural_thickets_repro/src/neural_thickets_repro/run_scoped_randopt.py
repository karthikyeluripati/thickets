"""Scoped RandOpt: component-localization RandOpt for the WACV extension (first milestone).
PREPARES ONLY -- not executed by us here (no GPU on this machine). GPU required to actually
run, and even then this milestone authorizes only diagnostics/scope_isolation_gpu_check.py,
never this script's own candidate search (see SCOPED_PERTURBATION_DESIGN.md).

Estimates, per component m, rho_{m,t} = P[S(theta + Delta_m) > S(theta)] by perturbing ONLY
a selected subset of model parameters (scopes.py) instead of the whole LM, with perturbation
magnitude made comparable across components of very different size via relative-L2 scaling
(scoped_perturbation.py), always restoring to the exact stored pretrained base.

run_randopt_image_aware.py (Gate 2, frozen) is not modified in any way here -- not even
additively. This script IMPORTS its sample_candidates() (read-only reuse, raw_sigma mode
only) and RESTORATION_MODES constant, exactly as it also reuses CandidateLedger/
topk_voting/vlm_adapter/GQAHandler/launch_engines/cleanup_engines -- all unchanged upstream
mechanisms, none of them reimplemented here.

Restoration mode: scoped scientific experiments require --restoration-mode fixed_base
(apply_perturbation-equivalent -> evaluate -> reset_to_base_weights, exact per-candidate
restore to the stored base -- see REPRO_SPEC.md "Gate 2 restoration semantics" for the A/B
evidence this is based on). released_compat is accepted at the CLI/argparse level (so the
rejection below is an explicit, readable message rather than a vanished flag) but this script
hard-fails immediately if anything else is passed -- released_compat may drift across
repeated perturbation cycles, which is exactly what scope-isolation science cannot tolerate.

Perturbation scope: neither apply_perturbation nor any other WorkerExtension method takes a
scope/component argument (confirmed against the pinned utils/worker_extn.py checkout --
SCOPED_PERTURBATION_DESIGN.md), so scoped perturbation is a local, package-side extension
(scoped_perturbation.scoped_apply_perturbation, dispatched via vLLM's own
collective_rpc(Callable, ...)) rather than a change to external/RandOpt, which is never
edited. Restoration after evaluation reuses the existing, unmodified, string-dispatched
reset_to_base_weights call -- exactly what fixed_base already uses today.

Candidate sampling depends on --perturbation-scale-mode (see candidate_sampling.py's module
docstring for the full reasoning):
  raw_sigma: candidates come from run_randopt_image_aware.py's own sample_candidates(N,
    sigma_values, global_seed) -- --sigma-candidate required, unchanged semantics, the
    (seed, sigma) pair used exactly as fixed_base already uses it today.
  relative_l2: candidates vary by SEED ONLY -- candidate_sampling.sample_candidate_seeds(N,
    global_seed) paired with the FIXED requested r from --relative-l2. No sigma_candidate
    value is ever sampled or used in this mode; --sigma-candidate is rejected, not silently
    ignored. The per-scope derived sigma (r * ||theta_m||_2 / sqrt(d_m)) is a run-level
    constant, computed once per candidate from that candidate's scope manifest (the manifest
    itself doesn't depend on candidate seed) and recorded on every candidate's ledger record.

Measurement (thicket_metrics.py): the primary scientific output is the THICKET, not merely
final accuracy -- expert density rho_{m,t}(r) = (1/N) sum_i 1[S(theta+Delta_i) > S(theta)]
and the candidate score distribution around it (mean/std/quantiles/deltas, a 95% Wilson CI
for expert density), computed against an EXPLICITLY evaluated base_score (see
compute_base_score() -- the exact unperturbed model on the SAME selection subset candidates
are scored against, restored to the exact stored base first, never inferred from a
historical/hard-coded baseline like the old Gate 1 result). Written per run to
thicket_metrics.json alongside the existing results.json/ensemble_predictions.jsonl
(Top-K/ensemble machinery, preserved for compatibility but not the headline result of this
experiment). See analysis/aggregate_coarse_thicket.py for building a cross-scope comparison
table from several completed runs' thicket_metrics.json files (hard-fails on any mismatched
task/model/dataset-subset/N/seed/radius/scoring/candidate-seed-sequence).

--relative-l2 (radius r) is an explicit experimental axis, not something this script tunes,
grids, or auto-selects -- exactly one r per invocation, chosen by the caller.

Usage (GPU required; NOT executed this milestone -- see
diagnostics/scope_isolation_gpu_check.py for the one authorized GPU validation):
    python -m neural_thickets_repro.run_scoped_randopt \
        --config configs/gqa_repro.yaml --N 20 --K 5 --restoration-mode fixed_base \
        --perturbation-scope vision_encoder \
        --perturbation-scale-mode relative_l2 --relative-l2 0.01
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

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # same runtime fix as every GPU script in this project

import numpy as np

from dataclasses import asdict

from .candidate_sampling import sample_candidate_seeds
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
from .run_randopt_image_aware import RESTORATION_MODES, sample_candidates
from .scoped_perturbation import NOISE_SEMANTICS, PERTURBATION_SCALE_MODES, scoped_apply_perturbation
from .scopes import PERTURBATION_SCOPES, scope_requires_encoder_cache_reset
from .thicket_metrics import aggregate_thicket_run
from .topk_voting import majority_vote, select_top_k
from .vlm_adapter import (
    bootstrap_ray,
    build_image_aware_requests,
    reset_vllm_encoder_cache,
    resolve_model_snapshot,
    verify_workers_can_import_external_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

GATE1_ARTIFACT = REPO_ROOT / "results" / "base_image_aware" / "metrics.json"


def _git_commit() -> str:
    """Duplicated, not imported, from run_randopt_image_aware.py -- consistent with this
    project's existing self-containment convention for small helpers across scripts.
    """
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
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


def _validate_collective_rpc_results(results, *, label: str):
    """Same TP=1 list-unwrap validation as diagnostics/gate2_restoration_ab.py -- duplicated
    rather than cross-imported from a diagnostics-only module, consistent with this project's
    convention. vLLM's collective_rpc returns a LIST of per-worker results even under TP=1;
    never index into it as a bare value.
    """
    if not isinstance(results, list):
        raise RuntimeError(
            f"collective_rpc({label!r}) returned {type(results).__name__}, expected vLLM's "
            f"own list-of-per-worker-results contract. Got: {results!r}"
        )
    if len(results) != 1:
        raise RuntimeError(
            f"collective_rpc({label!r}) returned {len(results)} per-worker results; this "
            f"script is TP=1-only (launch_engines(..., tensor_parallel_size=1, ...) below) "
            f"and expects exactly 1."
        )
    return results[0]


def _collective_rpc_single_worker(engine, method, args=(), *, label: str):
    import ray

    results = ray.get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def _candidate_key(seed: int, sigma_or_r: float) -> Tuple[int, float]:
    return (seed, sigma_or_r)


def compute_base_score(engine, handler, selection_requests, selection_datas, sampling_params) -> Tuple[float, List[str]]:
    """Explicitly evaluates the EXACT unperturbed base model on the SAME selection subset
    candidates are scored against, using the identical scoring path (handler.compute_reward)
    -- never inferred from a historical/hard-coded baseline (e.g. the Gate 1 result, which
    used a different subset/prompt path entirely). Restores to the exact stored base FIRST
    (reset_to_base_weights, the real unmodified upstream method) so this is never
    contaminated by whatever state the engine happened to be in beforehand -- must be called
    before any candidate sampling begins.

    Returns (base_score, base_responses) -- the raw text outputs are kept so callers can
    additionally compare a candidate's raw output against base directly (see
    run_sampling_phase), since a 200-example exact-match SCORE can legitimately tie even when
    the underlying raw outputs differ -- a weaker but more sensitive signal than delta_score
    for confirming a perturbation actually reached inference.
    """
    import ray

    _collective_rpc_single_worker(engine, "reset_to_base_weights", args=(), label="reset_to_base_weights")
    outputs = ray.get(engine.generate.remote(selection_requests, sampling_params, use_tqdm=False))
    responses = [o.outputs[0].text for o in outputs]
    base_score = float(np.mean([
        handler.compute_reward(resp, d["ground_truth"]) for resp, d in zip(responses, selection_datas)
    ])) if responses else 0.0
    return base_score, responses


def run_sampling_phase(
    engine, handler, selection_requests, selection_datas, sampling_params,
    candidates: List[Tuple[int, float]], ledger: CandidateLedger,
    scope: str, scale_mode: str, base_score: float, base_responses: List[str],
) -> Dict[Tuple[int, float], float]:
    """Perturb (scoped) -> generate on the selection set -> score -> restore (exact base),
    once per candidate. Resumable, same pattern as run_randopt_image_aware.py's own sampling
    phase. Candidate sampling/scoring/selection logic is otherwise identical regardless of
    scope/scale_mode -- only the perturb dispatch (scoped_apply_perturbation) differs.

    base_score (see compute_base_score) drives every candidate's delta_score/is_expert/is_tie
    -- the same explicit value for every candidate in this run, never re-derived per candidate.

    For scopes that can change vision-encoder/projector output
    (scopes.scope_requires_encoder_cache_reset), vlm_adapter.reset_vllm_encoder_cache is
    called after perturbing and before generating -- otherwise the SAME selection_requests
    images (built once, reused every candidate) could be served cached embeddings computed
    under a stale weight state. base_responses is used only for an informational raw-output-
    diff count printed alongside each candidate -- never used for scoring/selection.
    """
    import ray

    scores: Dict[Tuple[int, float], float] = {}
    for rec in ledger.load_all().values():
        if rec.status == "done" and rec.selection_score is not None:
            key_second = rec.sigma if scale_mode == "raw_sigma" else rec.requested_relative_l2
            scores[_candidate_key(rec.seed, key_second)] = rec.selection_score

    for candidate_id in ledger.iter_pending(range(len(candidates))):
        seed, sigma_or_r = candidates[candidate_id]
        start = time.time()

        perturb_result = _collective_rpc_single_worker(
            engine, scoped_apply_perturbation, args=(seed, sigma_or_r, scope, scale_mode),
            label="scoped_apply_perturbation",
        )
        encoder_cache_reset = False
        if scope_requires_encoder_cache_reset(scope):
            reset_vllm_encoder_cache(engine)
            encoder_cache_reset = True
        outputs = ray.get(engine.generate.remote(selection_requests, sampling_params, use_tqdm=False))
        responses = [o.outputs[0].text for o in outputs]
        reward = float(np.mean([
            handler.compute_reward(resp, d["ground_truth"]) for resp, d in zip(responses, selection_datas)
        ])) if responses else 0.0
        _collective_rpc_single_worker(engine, "reset_to_base_weights", args=(), label="reset_to_base_weights")

        delta_score = reward - base_score
        is_expert = delta_score > 0.0
        is_tie = delta_score == 0.0
        n_raw_changed = sum(1 for a, b in zip(base_responses, responses) if a.strip() != b.strip())

        scores[_candidate_key(seed, sigma_or_r)] = reward
        ledger.append(CandidateRecord(
            candidate_id=candidate_id, seed=seed, sigma=perturb_result["derived_sigma"],
            selection_score=reward, rank=None, status="done", runtime_seconds=time.time() - start,
            restoration_mode="fixed_base",
            perturbation_scope=scope, perturbation_scale_mode=scale_mode,
            requested_relative_l2=perturb_result["requested_relative_l2"],
            scope_param_count=perturb_result["scope_param_count"],
            scope_element_count=perturb_result["scope_total_element_count"],
            scope_base_l2_norm=perturb_result["scope_base_l2_norm"],
            actual_perturbation_l2=perturb_result["actual_perturbation_l2"],
            noise_semantics=perturb_result["noise_semantics"],
            base_score=base_score, delta_score=delta_score, is_expert=is_expert, is_tie=is_tie,
        ))
        label = "sigma" if scale_mode == "raw_sigma" else "r"
        status = "EXPERT" if is_expert else ("TIE" if is_tie else "regression")
        cache_note = f" encoder_cache_reset={encoder_cache_reset}" if scope_requires_encoder_cache_reset(scope) else ""
        print(
            f"  candidate {candidate_id + 1}/{len(candidates)}: seed={seed} {label}={sigma_or_r} "
            f"derived_sigma={perturb_result['derived_sigma']:.6g} score={reward:.4f} "
            f"raw_outputs_changed_vs_base={n_raw_changed}/{len(responses)}{cache_note} "
            f"delta={delta_score:+.4f} ({status})"
        )

    return scores


def run_ensemble_phase(
    engine, handler, test_requests, test_datas, sampling_params,
    top_k_candidates: List[Tuple[int, float]], scope: str, scale_mode: str,
) -> Dict:
    """Perturb (scoped) -> generate on the test set -> extract answers -> restore, for each
    of the top-K candidates, then majority-vote -- same pattern/reuse as
    run_randopt_image_aware.py's own ensemble phase.

    Same encoder-cache-reset requirement as run_sampling_phase, for the same reason: the
    test_requests images are also built once and reused across every top-K candidate here.
    """
    import ray

    all_answers: List[List[str]] = []
    for rank, (seed, sigma_or_r) in enumerate(top_k_candidates):
        _collective_rpc_single_worker(
            engine, scoped_apply_perturbation, args=(seed, sigma_or_r, scope, scale_mode),
            label="scoped_apply_perturbation",
        )
        encoder_cache_reset = False
        if scope_requires_encoder_cache_reset(scope):
            reset_vllm_encoder_cache(engine)
            encoder_cache_reset = True
        outputs = ray.get(engine.generate.remote(test_requests, sampling_params, use_tqdm=False))
        answers = [handler.extract_answer_for_voting(o.outputs[0].text) or "" for o in outputs]
        _collective_rpc_single_worker(engine, "reset_to_base_weights", args=(), label="reset_to_base_weights")
        all_answers.append(answers)
        cache_note = f" encoder_cache_reset={encoder_cache_reset}" if scope_requires_encoder_cache_reset(scope) else ""
        print(f"  ensemble member {rank + 1}/{len(top_k_candidates)}: seed={seed} done{cache_note}")

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
    parser.add_argument("--N", type=int, default=20, help="population size -- pass explicitly for a larger run")
    parser.add_argument("--K", type=int, default=5, help="ensemble size")
    parser.add_argument("--perturbation-scope", required=True, choices=PERTURBATION_SCOPES)
    parser.add_argument("--perturbation-scale-mode", required=True, choices=PERTURBATION_SCALE_MODES)
    parser.add_argument(
        "--sigma-candidate", default=None,
        help="required iff --perturbation-scale-mode raw_sigma; rejected for relative_l2 (no sigma_candidate draw exists in that mode)",
    )
    parser.add_argument(
        "--relative-l2", type=float, default=None,
        help="required iff --perturbation-scale-mode relative_l2 (the fixed r used for every candidate); rejected for raw_sigma",
    )
    parser.add_argument(
        "--restoration-mode", required=True, choices=RESTORATION_MODES,
        help="MUST be fixed_base for scoped WACV experiments -- released_compat is accepted here only so rejection is an explicit message, not a vanished CLI option; see module docstring.",
    )
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    if args.restoration_mode != "fixed_base":
        print(
            f"--restoration-mode must be 'fixed_base' for scoped WACV experiments (got "
            f"{args.restoration_mode!r}). released_compat may drift across repeated "
            f"perturbation cycles (REPRO_SPEC.md 'Gate 2 restoration semantics') -- exactly "
            f"what scope-isolation science cannot tolerate. Use run_randopt_image_aware.py "
            f"if you specifically want released_compat's paper-code-fidelity behavior.",
            file=sys.stderr,
        )
        return 1

    if args.perturbation_scale_mode == "raw_sigma":
        if args.sigma_candidate is None:
            print("--sigma-candidate is required when --perturbation-scale-mode raw_sigma.", file=sys.stderr)
            return 1
        if args.relative_l2 is not None:
            print("--relative-l2 is not accepted when --perturbation-scale-mode raw_sigma.", file=sys.stderr)
            return 1
    else:
        if args.relative_l2 is None:
            print("--relative-l2 is required when --perturbation-scale-mode relative_l2.", file=sys.stderr)
            return 1
        if args.sigma_candidate is not None:
            print(
                "--sigma-candidate is not accepted when --perturbation-scale-mode relative_l2 -- "
                "there is no sigma_candidate draw in this mode; candidates vary by seed only, "
                "paired with the one fixed --relative-l2 value.",
                file=sys.stderr,
            )
            return 1

    cfg = load_config(args.config)

    sigma_values = None
    if args.perturbation_scale_mode == "raw_sigma":
        if args.sigma_candidate not in cfg.randopt.sigma_candidates:
            print(
                f"--sigma-candidate must be one of {sorted(cfg.randopt.sigma_candidates)}, "
                f"got {args.sigma_candidate!r}.",
                file=sys.stderr,
            )
            return 1
        sigma_values = cfg.randopt.sigma_candidates[args.sigma_candidate]

    try:
        assert_feasible(
            "Scoped RandOpt (WACV component localization)",
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

    scale_id = f"raw_sigma_{args.sigma_candidate}" if args.perturbation_scale_mode == "raw_sigma" else f"relative_l2_r{args.relative_l2}"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"scoped_randopt_N{args.N}_K{args.K}_{args.perturbation_scope}_{scale_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = CandidateLedger(out_dir / "candidates.jsonl")

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    _assert_spawn_configured()
    import ray
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

    if args.perturbation_scale_mode == "raw_sigma":
        candidates = sample_candidates(args.N, sigma_values, cfg.reproducibility.global_seed)
        print(f"Sigma candidate: {args.sigma_candidate} = {sigma_values}  (UNRESOLVED assumption, see REPRO_SPEC.md)")
    else:
        seeds = sample_candidate_seeds(args.N, cfg.reproducibility.global_seed)
        candidates = [(seed, args.relative_l2) for seed in seeds]
        print(f"Relative-L2: r={args.relative_l2} (fixed across all {args.N} candidates -- candidates vary by seed only)")

    print(f"Perturbation scope: {args.perturbation_scope} | scale mode: {args.perturbation_scale_mode}")

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines = None
    pgs = None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        engines, pgs = launch_engines(1, model_path, precision=cfg.model.precision, tensor_parallel_size=1, multimodal=True)
        engine = engines[0]
        try:
            print(f"\n=== Base score: exact unperturbed model on the selection subset (n={len(selection_datas)}) ===")
            base_score, base_responses = compute_base_score(engine, handler, selection_requests, selection_datas, sampling_params)
            print(f"  base_score = {base_score:.4f}")

            print(f"\n=== Sampling phase: N={args.N} candidates (scope={args.perturbation_scope}) ===")
            if scope_requires_encoder_cache_reset(args.perturbation_scope):
                print(f"  scope {args.perturbation_scope!r} can affect vision-encoder output -- "
                      f"encoder cache will be reset after every perturbation, before generation")
            scores = run_sampling_phase(
                engine, handler, selection_requests, selection_datas, sampling_params,
                candidates, ledger, args.perturbation_scope, args.perturbation_scale_mode,
                base_score, base_responses,
            )

            top_k = select_top_k(scores, args.K)
            print(f"\n=== Selected top-{args.K} of {args.N} ===")
            for rank, (seed, sigma_or_r) in enumerate(top_k):
                print(f"  {rank + 1}. seed={seed} score={scores[(seed, sigma_or_r)]:.4f}")

            print(f"\n=== Ensemble evaluation: K={args.K} on {len(test_datas)} test examples ===")
            ensemble_results = run_ensemble_phase(
                engine, handler, test_requests, test_datas, sampling_params, top_k,
                args.perturbation_scope, args.perturbation_scale_mode,
            )
        finally:
            cleanup_engines(engines, pgs)
    finally:
        if engines is None and ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps({
        "restoration_mode": "fixed_base",
        "perturbation_scope": args.perturbation_scope,
        "perturbation_scale_mode": args.perturbation_scale_mode,
        "sigma_candidate": args.sigma_candidate, "sigma_values": sigma_values,
        "requested_relative_l2": args.relative_l2,
        "N": args.N, "K": args.K,
        "top_k_candidates": [{"seed": s, "sigma_or_r": sg, "selection_score": scores[(s, sg)]} for s, sg in top_k],
        "ensemble_accuracy": ensemble_results["accuracy"],
        "ensemble_n": ensemble_results["n"],
        "ensemble_correct": ensemble_results["correct"],
    }, indent=2))
    predictions_path = out_dir / "ensemble_predictions.jsonl"
    with predictions_path.open("w") as f:
        for rec in ensemble_results["predictions"]:
            f.write(json.dumps(rec) + "\n")

    run_metadata = {
        "restoration_mode": "fixed_base",
        "restoration_mechanism": "apply_perturbation-equivalent (scoped_apply_perturbation) / reset_to_base_weights()",
        "perturbation_scope": args.perturbation_scope,
        "perturbation_scale_mode": args.perturbation_scale_mode,
        "sigma_candidate": args.sigma_candidate, "sigma_values": sigma_values,
        "requested_relative_l2": args.relative_l2,
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "model_snapshot_path": model_path,
        "N": args.N, "K": args.K,
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

    # Primary scientific output of this run -- expert density / score distribution over the
    # candidate population, NOT Top-1/ensemble accuracy (results.json/predictions above are
    # preserved for compatibility with the existing Top-K machinery, not the headline result).
    done_records = sorted(
        (r for r in ledger.load_all().values() if r.status == "done"),
        key=lambda r: r.candidate_id,
    )
    thicket_stats = aggregate_thicket_run(base_score, [(r.seed, r.selection_score) for r in done_records])
    thicket_metrics_path = out_dir / "thicket_metrics.json"
    thicket_metrics_path.write_text(json.dumps({
        "task": "gqa",
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "scope": args.perturbation_scope,
        "N": args.N,
        "perturbation_scale_mode": args.perturbation_scale_mode,
        "sigma_candidate": args.sigma_candidate,
        "requested_relative_l2": args.relative_l2,
        "global_seed": cfg.reproducibility.global_seed,
        "restoration_mode": "fixed_base",
        "noise_semantics": NOISE_SEMANTICS,
        "base_score": base_score,
        "selection_set_size": len(selection_datas),
        "selection_example_ids": [str(d["question_id"]) for d in selection_datas],
        "dataset_revision": cfg.dataset.revision,
        "dataset_selection_split": cfg.dataset.selection_split,
        "scoring_protocol": "gqa_image_aware_v1",
        "candidate_seed_sequence": [r.seed for r in done_records],
        "scope_param_count": done_records[0].scope_param_count if done_records else None,
        "scope_element_count": done_records[0].scope_element_count if done_records else None,
        "scope_base_l2_norm": done_records[0].scope_base_l2_norm if done_records else None,
        "expert_count": thicket_stats.expert_count,
        "tie_count": thicket_stats.tie_count,
        "regression_count": thicket_stats.regression_count,
        "expert_density": thicket_stats.expert_density,
        "expert_density_ci_95": [thicket_stats.expert_density_ci_lower, thicket_stats.expert_density_ci_upper],
        "mean_score": thicket_stats.mean_score,
        "std_score": thicket_stats.std_score,
        "mean_delta": thicket_stats.mean_delta,
        "median_delta": thicket_stats.median_delta,
        "min_delta": thicket_stats.min_delta,
        "max_delta": thicket_stats.max_delta,
        "score_quantiles": {
            "25": thicket_stats.score_quantile_25,
            "50": thicket_stats.score_quantile_50,
            "75": thicket_stats.score_quantile_75,
        },
        "best_candidate_score": thicket_stats.best_candidate_score,
        "best_candidate_seed": thicket_stats.best_candidate_seed,
        "candidate_records": [asdict(r) for r in done_records],
    }, indent=2))

    print("\n=== Scoped RandOpt run complete ===")
    print(f"Scope: {args.perturbation_scope} | scale mode: {args.perturbation_scale_mode}")
    print(f"Base score: {base_score:.4f}")
    print(
        f"Expert density rho_m(r) = {thicket_stats.expert_density:.4f} "
        f"(95% CI [{thicket_stats.expert_density_ci_lower:.4f}, {thicket_stats.expert_density_ci_upper:.4f}]) "
        f"-- {thicket_stats.expert_count} experts / {thicket_stats.tie_count} ties / "
        f"{thicket_stats.regression_count} regressions out of {thicket_stats.n}"
    )
    print(f"Best candidate: seed={thicket_stats.best_candidate_seed} score={thicket_stats.best_candidate_score:.4f}")
    print(f"(Top-K ensemble accuracy, preserved for compatibility, not the primary result: {ensemble_results['accuracy']:.4f})")
    print(f"Wrote {results_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {run_metadata_path}")
    print(f"Wrote {thicket_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
