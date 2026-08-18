"""Gate 2 restoration-semantics A/B diagnostic -- GPU required. Narrowly scoped: this does
NOT redesign RandOpt and does NOT pick a restoration mode for Gate 2. It produces
side-by-side evidence on one fixed perturbation so that decision can be made deliberately.

Triggered by: gate2_gpu_preflight.py found that after a SINGLE perturb_self_weights ->
restore_self_weights cycle (seed=999999999, sigma=0.01, on 5 fixed examples), only 1/5
outputs exactly matched base -- not a cache-masking bug (prefix caching is off; see
GATE2_CACHE_SAFETY_REVIEW.md), and step 3 (output changes after perturbation) passed 5/5, so
perturbation itself is taking effect correctly. REPRO_SPEC.md's "Perturbation formula" row
already documents *why* this is plausible: restore_self_weights reseeds identically and
subtracts the same noise rather than restoring from a stored weight copy
(utils/worker_extn.py:perturb_self_weights/restore_self_weights, native dtype, no fp32
accumulation). bf16 add-then-subtract is not guaranteed bitwise-exact -- rounding can leave
a residual a few ULPs off theta whenever theta and the sigma*epsilon delta differ enough in
magnitude, which is plausible at sigma=0.01 against bf16 weights.

Two restoration paths, BOTH reusing external/RandOpt mechanisms unmodified (external/RandOpt
itself is never edited or reimplemented -- see REPRO_SPEC.md's reproduction-integrity
principle):

  Path A ("released_compat"): perturb_self_weights -> generate -> restore_self_weights ->
    generate. Exactly what randopt.py's own run_sampling/run_ensemble_evaluation do, and
    what run_randopt_image_aware.py currently calls.
  Path B ("fixed_base"): apply_perturbation -> generate -> reset_to_base_weights ->
    generate. utils/worker_extn.py:WorkerExtension also exposes these; per the pinned
    source, apply_perturbation derives the perturbed weights from the base weights
    launch_engines() already stores at engine-launch time, rather than from perturb/restore
    deltas applied to whatever the current (possibly already-drifted) weights are.

Both paths run on the SAME fixed test examples and (seed, sigma) as gate2_gpu_preflight.py
(imported from there, not re-declared, so "same fixed perturbation" holds by construction),
repeated --cycles times in a row, to see whether restoration error accumulates across
repeated cycles (expected if each cycle perturbs/restores relative to the current,
possibly-already-drifted state) or stays flat (expected if each cycle re-derives from an
unchanging stored base). Path B deliberately runs immediately after Path A's cycles without
any explicit re-clean in between -- if apply_perturbation/reset_to_base_weights truly
re-derive from launch_engines()'s stored base rather than current state, Path B's own first
cycle should already show no residue from Path A. That is itself part of the evidence this
diagnostic is designed to surface, not an oversight.

Signature verification: apply_perturbation/reset_to_base_weights were originally assumed to
accept the same (seed, sigma, <bool>) argument shape perturb_self_weights/restore_self_weights
do, because external/RandOpt isn't available in this development environment to check
directly (GPU-only pod). _verify_worker_extension_signatures() below replaces that assumption
with an explicit check, run on the driver directly against the pinned checkout (now available
on the pod) before launch_engines() spends any GPU time -- prints every signature and hard-
fails on a parameter-count mismatch. That check is necessary but not sufficient (it can't see
parameter names/order/defaults), so the try/except around Path B in main() remains the actual
safety net: if the real call convention differs in some way the count check can't catch, Ray's
collective_rpc raises a clear exception, caught and reported verbatim in the output report as
path_b_fixed_base.error, with Path A's results left intact -- rather than silently producing a
partial or misleading Path B result.

Every collective_rpc call in this module goes through _collective_rpc_single_worker(), which
explicitly validates and unwraps vLLM's list-of-per-worker-results return shape (collective_rpc
returns a LIST even with exactly one TP-rank worker) instead of indexing into it as if it were
a bare value -- this diagnostic is TP=1-only (see launch_engines() in main()) and fails clearly
rather than guessing if that shape assumption is ever violated.

Raw parameter drift (max/mean abs diff, relative L2, fraction of elements differing, and a
per-cycle check that visual.*-prefixed tensors are EXACTLY unchanged) is measured via a
diagnostic-only function dispatched through vLLM's own
collective_rpc(method: Union[str, Callable], ...) -- not a method added to WorkerExtension;
_diag_snapshot_base/_diag_diff_from_base below are ordinary functions in OUR OWN package,
shipped to the worker process by vLLM's existing RPC machinery (the same mechanism the
string-form perturb_self_weights/restore_self_weights calls already go through elsewhere in
this project), and they only ever READ worker_self.model_runner.model's parameters -- they
never call, wrap, or reimplement perturb_self_weights/restore_self_weights/
apply_perturbation/reset_to_base_weights. The actual math is diagnostics/perturb_restore_drift.py's
already-unit-tested measure_drift/check_vision_frozen (that module's own drift audit runs
our from-scratch perturb_cpu.py reimplementation directly on a loaded HF model, bypassing
vLLM/Ray entirely -- a different, complementary check; this script instead exercises the
REAL upstream WorkerExtension methods through the real vLLM/Ray engine, which is the only
way to test apply_perturbation/reset_to_base_weights at all, since perturb_cpu.py has no
equivalents for those). If this vLLM version does not support Callable dispatch, or
model_runner.model isn't reachable at that attribute path, drift measurement degrades to
"unavailable" in the report (drift_measurement_available: false) rather than crashing -- the
output/answer/correctness comparisons above it are computed independently and are unaffected
either way.

Usage:
    python -m neural_thickets_repro.diagnostics.gate2_restoration_ab --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # same runtime fix as Gate 1/2 -- see eval_base_image_aware.py

from ..config import load_config
from ..env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from ..vlm_adapter import (
    bootstrap_ray,
    build_image_aware_requests,
    resolve_model_snapshot,
    verify_workers_can_import_external_root,
)
from .gate2_gpu_preflight import N_PREFLIGHT_EXAMPLES, TEST_SEED, TEST_SIGMA
from .perturb_restore_drift import check_vision_frozen, measure_drift

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"


def _diag_snapshot_base(worker_self) -> str:
    """Dispatched via collective_rpc(_diag_snapshot_base, args=()) -- runs INSIDE the vLLM
    worker process (worker_self is that worker instance, vLLM's own convention for Callable
    RPC dispatch). Stores a live GPU-resident clone of the model's full state_dict as an
    attribute on the worker instance -- diagnostic-only in-memory state, not written to
    WorkerExtension's class definition, not persisted to disk. Kept on-GPU (not .cpu()) to
    keep repeated diffing fast; costs roughly one extra copy of the model's parameters in GPU
    memory for the diagnostic's duration (a few GB at 3B params in bf16 -- comfortably inside
    an L40S's 48GB alongside the already-running engine).
    """
    model = worker_self.model_runner.model
    worker_self._diag_base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return f"snapshotted {len(worker_self._diag_base_state)} tensors"


def _diag_diff_from_base(worker_self) -> Dict:
    """Dispatched via collective_rpc(_diag_diff_from_base, args=()) -- companion to
    _diag_snapshot_base, must run in the same worker process afterward (relies on the
    instance attribute it sets). Reuses perturb_restore_drift.py's already-unit-tested
    measure_drift/check_vision_frozen rather than reimplementing the same math a second time
    -- see that module's docstring for what each field means. Returns only small aggregate
    scalars (never per-parameter tensors) to keep the Ray object-store payload trivial.
    """
    if not hasattr(worker_self, "_diag_base_state"):
        raise RuntimeError("_diag_snapshot_base was never called on this worker before _diag_diff_from_base")
    model = worker_self.model_runner.model
    base_state = worker_self._diag_base_state
    drift = measure_drift(model, base_state)
    drift["vision_frozen"] = check_vision_frozen(model, base_state)
    return drift


def _validate_collective_rpc_results(results, *, label: str):
    """Pure validation/unwrap logic, no ray dependency -- split out from
    _collective_rpc_single_worker() below purely so it can be unit tested directly (with
    plain lists) without ray installed; see tests/test_gate2_restoration_ab.py.

    vLLM's collective_rpc returns a LIST of per-worker results (one per TP rank), not a bare
    value -- every call site in this module must go through here rather than treating
    ray.get(engine.collective_rpc.remote(...)) as the value directly (that was the bug: a
    single-worker [drift_dict] indexed as if it were drift_dict itself, raising
    "TypeError: list indices must be integers or slices, not str").

    This diagnostic's launch_engines(..., tensor_parallel_size=1, ...) call (see main())
    means exactly one worker should ever respond, so this validates that explicitly and
    unwraps it -- it does NOT blindly take [0] without checking length/type first.

    Deliberately NOT extended to aggregate across workers for TP>1: correctly combining
    per-shard drift stats (max_abs_drift needs max-across-shards, mean_abs_drift needs
    sum-of-sums divided by sum-of-counts, not an average of already-averaged per-shard
    ratios) is untested and unnecessary for how this diagnostic is actually invoked. If this
    is ever run with TP>1, this is the one place that needs extending -- it fails clearly
    here instead of silently reporting one shard's numbers as the whole model's.
    """
    if not isinstance(results, list):
        raise RuntimeError(
            f"collective_rpc({label!r}) returned {type(results).__name__}, expected vLLM's "
            f"own list-of-per-worker-results contract. Got: {results!r}"
        )
    if len(results) != 1:
        raise RuntimeError(
            f"collective_rpc({label!r}) returned {len(results)} per-worker results; this "
            f"diagnostic is TP=1-only (launch_engines(..., tensor_parallel_size=1, ...) in "
            f"main() below) and expects exactly 1. If you intended to run this under tensor "
            f"parallelism, _collective_rpc_single_worker needs extending to aggregate "
            f"per-shard drift correctly (see its docstring) -- it deliberately does not guess."
        )
    return results[0]


def _collective_rpc_single_worker(engine, method, args=(), *, label: str):
    import ray

    results = ray.get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def _verify_worker_extension_signatures(worker_extension_cls) -> None:
    """Runs on the DRIVER, directly against the pinned external/RandOpt checkout's
    utils.worker_extn.WorkerExtension class (importable now that main() has put
    EXTERNAL_ROOT on sys.path) -- introspecting the class object directly is equivalent to
    introspecting it on a worker, since it's the same unmodified class definition either way.

    Replaces this module's earlier documented ASSUMPTION that apply_perturbation/
    reset_to_base_weights accept the same (seed, sigma, bool) argument shape
    perturb_self_weights/restore_self_weights do -- that assumption was made because
    external/RandOpt isn't available in the development environment this diagnostic was
    written in. Now that the pod has the real checkout, this checks it explicitly, BEFORE
    launch_engines() spends any GPU time, and prints every signature so it can also be
    eyeballed directly in the run's console output.

    Only compares parameter COUNT, not names/defaults/order -- inspect.signature can't tell
    us the calling convention is semantically equivalent, only that it's shaped the same. A
    matching count is necessary but not sufficient; the try/except around Path B in main()
    remains the actual safety net against a real mismatch.
    """
    import inspect

    required = ("perturb_self_weights", "restore_self_weights", "apply_perturbation", "reset_to_base_weights")
    missing = [name for name in required if not hasattr(worker_extension_cls, name)]
    if missing:
        raise RuntimeError(
            f"utils.worker_extn.WorkerExtension is missing method(s) {missing} -- cannot run "
            f"this diagnostic's Path A/B as designed. Re-check the pinned source."
        )

    signatures = {name: inspect.signature(getattr(worker_extension_cls, name)) for name in required}
    for name, sig in signatures.items():
        print(f"  {name}{sig}")

    if len(signatures["apply_perturbation"].parameters) != len(signatures["perturb_self_weights"].parameters):
        raise RuntimeError(
            f"apply_perturbation{signatures['apply_perturbation']} has a different parameter "
            f"count than perturb_self_weights{signatures['perturb_self_weights']} -- "
            f"_run_path_cycles calls both with the same (seed, sigma, False) args= tuple. "
            f"Update that call site for apply_perturbation before trusting Path B's results."
        )
    if len(signatures["reset_to_base_weights"].parameters) != len(signatures["restore_self_weights"].parameters):
        raise RuntimeError(
            f"reset_to_base_weights{signatures['reset_to_base_weights']} has a different "
            f"parameter count than restore_self_weights{signatures['restore_self_weights']} -- "
            f"_run_path_cycles calls both with the same (seed, sigma, False) args= tuple. "
            f"Update that call site for reset_to_base_weights before trusting Path B's results."
        )


def _generate_and_score(engine, requests, task_datas, handler, sampling_params):
    import ray

    outputs = ray.get(engine.generate.remote(requests, sampling_params, use_tqdm=False))
    raw = [o.outputs[0].text for o in outputs]
    answers = [handler.extract_answer_for_voting(t) or "" for t in raw]
    correct = [bool(a) and handler.is_voted_answer_correct(a, d["ground_truth"]) for a, d in zip(answers, task_datas)]
    return raw, answers, correct


def _run_path_cycles(
    engine, label: str, perturb_method: str, restore_method: str,
    requests, task_datas, handler, sampling_params,
    base_raw: List[str], base_answers: List[str], base_correct: List[bool],
    n_cycles: int, drift_available: bool,
) -> List[Dict]:
    n_total = len(task_datas)
    cycles = []
    for cycle in range(1, n_cycles + 1):
        _collective_rpc_single_worker(engine, perturb_method, args=(TEST_SEED, TEST_SIGMA, False), label=perturb_method)
        _collective_rpc_single_worker(engine, restore_method, args=(TEST_SEED, TEST_SIGMA, False), label=restore_method)

        raw, answers, correct = _generate_and_score(engine, requests, task_datas, handler, sampling_params)

        n_raw_match = sum(1 for a, b in zip(base_raw, raw) if a.strip() == b.strip())
        n_answer_match = sum(1 for a, b in zip(base_answers, answers) if a == b)
        n_correctness_agree = sum(1 for a, b in zip(base_correct, correct) if a == b)

        drift = (
            _collective_rpc_single_worker(engine, _diag_diff_from_base, args=(), label="_diag_diff_from_base")
            if drift_available else None
        )

        cycles.append({
            "cycle": cycle,
            "n_raw_output_matches_base": n_raw_match,
            "n_extracted_answer_matches_base": n_answer_match,
            "n_correctness_agrees_with_base": n_correctness_agree,
            "n_total": n_total,
            "drift": drift,
        })
        drift_msg = f" max_abs_drift={drift['max_abs_drift']:.3e} vision_frozen={drift['vision_frozen']}" if drift else ""
        print(
            f"  [{label}] cycle {cycle}/{n_cycles}: raw_match={n_raw_match}/{n_total} "
            f"answer_match={n_answer_match}/{n_total} correctness_agree={n_correctness_agree}/{n_total}{drift_msg}"
        )
    return cycles


def _drift_summary(cycles: List[Dict]) -> Optional[Dict]:
    if not cycles or cycles[0]["drift"] is None:
        return None
    first, last = cycles[0]["drift"], cycles[-1]["drift"]
    return {
        "first_cycle": first,
        "last_cycle": last,
        "max_abs_drift_ratio_last_over_first": (
            (last["max_abs_drift"] / first["max_abs_drift"]) if first["max_abs_drift"] > 0 else None
        ),
        "fraction_elements_differing_ratio_last_over_first": (
            (last["fraction_elements_differing"] / first["fraction_elements_differing"])
            if first["fraction_elements_differing"] > 0 else None
        ),
    }


def _visual_frozen_every_cycle(cycles: List[Dict]) -> Optional[bool]:
    drift_cycles = [c["drift"] for c in cycles if c["drift"] is not None]
    if not drift_cycles:
        return None
    return all(d["vision_frozen"] for d in drift_cycles)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "gate2_diagnosis" / "restoration_ab_report.json"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "Gate 2 restoration A/B diagnostic",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision", "dataset.selection_split", "dataset.test_split")

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    import ray
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines, launch_engines  # type: ignore
    from data_handlers.gqa import GQAHandler  # type: ignore
    from utils.worker_extn import WorkerExtension  # type: ignore

    print("Verifying WorkerExtension method signatures against the pinned external/RandOpt checkout...")
    _verify_worker_extension_signatures(WorkerExtension)

    handler = GQAHandler()
    task_datas = handler.load_data(
        str(EXTERNAL_ROOT / "data" / "gqa" / "train.parquet"), split="train", max_samples=None,
    )
    task_datas = [d for d in task_datas if "image_path" in d][:N_PREFLIGHT_EXAMPLES]
    if len(task_datas) < N_PREFLIGHT_EXAMPLES:
        print(f"WARNING: only {len(task_datas)} examples with images available (wanted {N_PREFLIGHT_EXAMPLES})", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    requests = build_image_aware_requests(task_datas, tokenizer)
    sampling_params = SamplingParams(temperature=0.0, seed=cfg.reproducibility.global_seed, max_tokens=cfg.evaluation.max_tokens)

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines = None
    pgs = None
    path_a: List[Dict] = []
    path_b: List[Dict] = []
    path_b_error: Optional[str] = None
    base_raw: List[str] = []
    base_answers: List[str] = []
    base_correct: List[bool] = []
    drift_available = False
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        engines, pgs = launch_engines(1, model_path, precision=cfg.model.precision, tensor_parallel_size=1, multimodal=True)
        engine = engines[0]
        try:
            print("Snapshotting base parameters for drift measurement (diagnostic-only, in-worker)...")
            try:
                msg = _collective_rpc_single_worker(engine, _diag_snapshot_base, args=(), label="_diag_snapshot_base")
                print(f"  {msg}")
                drift_available = True
            except Exception as exc:  # noqa: BLE001 -- any failure here only downgrades drift reporting, never aborts the diagnostic
                print(
                    f"  WARNING: raw parameter drift measurement unavailable ({type(exc).__name__}: {exc}). "
                    f"Continuing with output/answer/correctness comparisons only -- those do not depend on this.",
                    file=sys.stderr,
                )

            print(f"Generating base outputs (n={len(task_datas)})...")
            base_raw, base_answers, base_correct = _generate_and_score(engine, requests, task_datas, handler, sampling_params)

            print(f"\n=== Path A (released_compat): perturb_self_weights / restore_self_weights, {args.cycles} cycles ===")
            path_a = _run_path_cycles(
                engine, "A", "perturb_self_weights", "restore_self_weights",
                requests, task_datas, handler, sampling_params,
                base_raw, base_answers, base_correct, args.cycles, drift_available,
            )

            print(f"\n=== Path B (fixed_base): apply_perturbation / reset_to_base_weights, {args.cycles} cycles ===")
            print("(runs immediately after Path A's cycles with no explicit re-clean -- see module docstring)")
            try:
                path_b = _run_path_cycles(
                    engine, "B", "apply_perturbation", "reset_to_base_weights",
                    requests, task_datas, handler, sampling_params,
                    base_raw, base_answers, base_correct, args.cycles, drift_available,
                )
            except Exception as exc:  # noqa: BLE001 -- reported in the output report, Path A results stay intact either way
                path_b_error = (
                    f"{type(exc).__name__}: {exc}. _verify_worker_extension_signatures() above already "
                    f"confirmed apply_perturbation/reset_to_base_weights have the same parameter COUNT as "
                    f"perturb_self_weights/restore_self_weights, so this is likely a mismatch in parameter "
                    f"names/order/defaults/semantics that count-checking can't catch -- inspect the printed "
                    f"signatures in this run's console output and adjust the args passed by _run_path_cycles "
                    f"before re-running Path B."
                )
                print(f"  Path B FAILED: {path_b_error}", file=sys.stderr)
        finally:
            cleanup_engines(engines, pgs)
    finally:
        if engines is None and ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    report = {
        "n_examples": len(task_datas),
        "test_seed": TEST_SEED,
        "test_sigma": TEST_SIGMA,
        "cycles_requested": args.cycles,
        "drift_measurement_available": drift_available,
        "path_a_released_compat": {
            "methods": {"perturb": "perturb_self_weights", "restore": "restore_self_weights"},
            "cycles": path_a,
            "drift_summary": _drift_summary(path_a),
            "visual_weights_frozen_every_cycle": _visual_frozen_every_cycle(path_a),
        },
        "path_b_fixed_base": {
            "methods": {"perturb": "apply_perturbation", "restore": "reset_to_base_weights"},
            "cycles": path_b,
            "drift_summary": _drift_summary(path_b),
            "visual_weights_frozen_every_cycle": _visual_frozen_every_cycle(path_b),
            "error": path_b_error,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nWrote {out_path}")
    print(
        "\nThis diagnostic does not choose a restoration mode -- review path_a_released_compat vs "
        "path_b_fixed_base (including the full per-cycle drift series in the report) before deciding "
        "between released_compat (paper-code-faithful, may drift across repeated perturbation cycles per "
        "this evidence) and fixed_base (every candidate independently perturbed from the exact stored "
        "pretrained base -- changes what the search explores, not merely a bugfix). Gate 2 N=20 remains "
        "blocked until this is reviewed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
