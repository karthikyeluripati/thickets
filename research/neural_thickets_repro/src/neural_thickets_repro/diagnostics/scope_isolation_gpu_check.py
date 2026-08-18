"""Scope-isolation GPU mechanical validation -- the ONE GPU execution authorized for the
scoped-RandOpt (WACV) milestone (see SCOPED_PERTURBATION_DESIGN.md). Does NOT run a
candidate search of any size -- no N=20/50/5000, no full GQA evaluation, no all-scope sweep.
Pure weight-level verification: no generation/tokenizer/GQA data needed at all, since Test
A/B only check which PARAMETERS changed, never model outputs.

Before either test, dispatches a Callable that reads worker_self.model_runner.model's real
named_parameters() and reports (printed AND written to the report): representative runtime
parameter names (proof of the actual namespace, not an assumption -- see
SCOPED_PERTURBATION_DESIGN.md's Phase 1 correction that the HF checkpoint's flat keys are NOT
necessarily what the loaded/wrapped runtime module exposes), the detected LM namespace
convention + layer count (scopes.discover_lm_layer_indices), and scopes.build_scope_manifest's
selected/element counts (+ any aliases) for EVERY ONE of the seven coarse scopes -- not just
the two under test -- so the full scope map is visible for review before any weight is
touched.

Test A (vision_encoder): snapshot base -> perturb ONLY vision_encoder
(scoped_perturbation.scoped_apply_perturbation, the local package-side extension -- never
touches external/RandOpt) -> verify >=1 selected vision-encoder param changed -> verify the
LITERAL COMPLEMENT of vision_encoder's selected storage across the entire runtime model
(not an enumerated "merger + LM" list, which can silently miss tensors belonging to neither
named group) is exactly unchanged -> reset_to_base_weights (real, unmodified upstream method,
string-dispatched exactly as fixed_base already uses it) -> verify entire model exactly
equals base.

Test B (lm_middle): same shape -- perturb only lm_middle, verify the literal complement of
lm_middle's selected storage (every other runtime tensor, whatever component it belongs to --
this correctly includes non-layer LM tensors like embeddings/final-norm that aren't part of
ANY lm_early/middle/late third, which an enumerated "vision + merger + lm_early + lm_late"
list would silently omit) is exactly unchanged, reset, verify exact full-model match.

Drift measurement reuses diagnostics/perturb_restore_drift.py's already-unit-tested
measure_drift (extended with an optional param_filter for exactly this in-scope/
out-of-scope split) rather than a third reimplementation of the same math.

Usage:
    python -m neural_thickets_repro.diagnostics.scope_isolation_gpu_check --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # same runtime fix as every GPU script in this project

from ..config import load_config
from ..env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from ..scopes import PERTURBATION_SCOPES, build_scope_manifest, discover_lm_layer_indices
from ..scoped_perturbation import scoped_apply_perturbation
from ..vlm_adapter import bootstrap_ray, resolve_model_snapshot, verify_workers_can_import_external_root
from .perturb_restore_drift import measure_drift

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

TEST_SEED = 999_999_999  # fixed, clearly not drawn from any real candidate's RNG stream -- same convention as gate2_gpu_preflight.py
TEST_SIGMA = 0.01  # deliberately large-ish so a real behavior change is easy to detect


def _try_named_parameters_no_duplicate(model):
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return None


def _diag_snapshot_base(worker_self) -> str:
    model = worker_self.model_runner.model
    worker_self._scope_diag_base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return f"snapshotted {len(worker_self._scope_diag_base_state)} tensors"


def _diag_report_all_scopes(worker_self) -> Dict:
    """Runs BEFORE either perturbation. Returns representative names, detected LM namespace
    convention/layer count, and a manifest summary for every one of the seven coarse scopes.
    """
    model = worker_self.model_runner.model
    named_parameters = list(model.named_parameters())
    all_names = [name for name, _ in named_parameters]
    alias_view = _try_named_parameters_no_duplicate(model)

    convention, layer_indices = discover_lm_layer_indices(all_names)

    scope_summaries = {}
    for scope in PERTURBATION_SCOPES:
        manifest = build_scope_manifest(scope, named_parameters, alias_named_parameters=alias_view)
        scope_summaries[scope] = {
            "selected_param_count": manifest.selected_param_count,
            "total_element_count": manifest.total_element_count,
            "base_l2_norm": manifest.base_l2_norm,
            "representative_names": manifest.representative_names,
            "aliases": manifest.aliases,
        }

    return {
        "n_total_parameters": len(named_parameters),
        "representative_param_names": all_names[:15],
        "detected_lm_namespace_convention": convention,
        "detected_lm_layer_count": len(layer_indices),
        "scope_summaries": scope_summaries,
    }


def _diag_scope_drift(worker_self, scope_name: str) -> Dict:
    """Runs AFTER a perturbation, before reset. Splits drift into in-scope (the perturbed
    scope's own selected params) vs out-of-scope (everything else) using measure_drift's
    param_filter -- reused math, not reimplemented.

    "Everything else" is deliberately the LITERAL complement of the scope's selected storage
    set across the entire (deduplicated) runtime model -- computed via full_vlm's own
    manifest, not an enumerated list of "the other named components" (e.g. "vision + merger
    + the other LM thirds"), which can silently miss tensors that don't belong to any of the
    named groups (e.g. non-layer LM tensors like embeddings/final-norm that aren't part of
    ANY lm_early/middle/late third). out_of_scope_param_count is reported explicitly so the
    report itself proves the full complement was checked, not just a boolean pass/fail.
    """
    if not hasattr(worker_self, "_scope_diag_base_state"):
        raise RuntimeError("_diag_snapshot_base was never called on this worker before _diag_scope_drift")
    model = worker_self.model_runner.model
    base_state = worker_self._scope_diag_base_state
    named_parameters = list(model.named_parameters())
    manifest = build_scope_manifest(scope_name, named_parameters)
    full_vlm_manifest = build_scope_manifest("full_vlm", named_parameters)
    selected_names = set(manifest.selected_param_names)

    in_scope = measure_drift(model, base_state, param_filter=lambda n: n in selected_names)
    out_of_scope = measure_drift(model, base_state, param_filter=lambda n: n not in selected_names)
    return {
        "in_scope": in_scope,
        "out_of_scope": out_of_scope,
        "scope_param_count": manifest.selected_param_count,
        "out_of_scope_param_count": full_vlm_manifest.selected_param_count - manifest.selected_param_count,
        "full_vlm_param_count": full_vlm_manifest.selected_param_count,
    }


def _diag_full_model_drift(worker_self) -> Dict:
    if not hasattr(worker_self, "_scope_diag_base_state"):
        raise RuntimeError("_diag_snapshot_base was never called on this worker before _diag_full_model_drift")
    model = worker_self.model_runner.model
    return measure_drift(model, worker_self._scope_diag_base_state)


def _validate_collective_rpc_results(results, *, label: str):
    """Same TP=1 list-unwrap validation as gate2_restoration_ab.py / run_scoped_randopt.py --
    duplicated, consistent with this project's convention. Never index [0] without checking.
    """
    if not isinstance(results, list):
        raise RuntimeError(
            f"collective_rpc({label!r}) returned {type(results).__name__}, expected vLLM's "
            f"own list-of-per-worker-results contract. Got: {results!r}"
        )
    if len(results) != 1:
        raise RuntimeError(
            f"collective_rpc({label!r}) returned {len(results)} per-worker results; this "
            f"diagnostic is TP=1-only and expects exactly 1."
        )
    return results[0]


def _rpc(engine, method, args=(), *, label: str):
    import ray

    results = ray.get(engine.collective_rpc.remote(method, args=args))
    return _validate_collective_rpc_results(results, label=label)


def _run_isolation_test(engine, test_label: str, scope: str) -> Dict:
    print(f"\n=== Test {test_label} ({scope}) ===")

    print(f"  Perturbing ONLY {scope} (seed={TEST_SEED}, sigma={TEST_SIGMA})...")
    _rpc(engine, scoped_apply_perturbation, args=(TEST_SEED, TEST_SIGMA, scope, "raw_sigma"), label="scoped_apply_perturbation")

    print("  Measuring in-scope vs out-of-scope drift...")
    drift = _rpc(engine, _diag_scope_drift, args=(scope,), label="_diag_scope_drift")

    in_scope_changed = drift["in_scope"]["max_abs_drift"] > 0.0
    out_of_scope_unchanged = drift["out_of_scope"]["max_abs_drift"] == 0.0
    print(
        f"    in-scope ({drift['scope_param_count']} tensors): max_abs_drift="
        f"{drift['in_scope']['max_abs_drift']:.3e} -- {'PASS (changed)' if in_scope_changed else 'FAIL (no change detected)'}"
    )
    print(
        f"    out-of-scope ({drift['out_of_scope_param_count']} tensors -- literal complement "
        f"of {scope}'s selected storage across all {drift['full_vlm_param_count']} runtime "
        f"tensors): max_abs_drift={drift['out_of_scope']['max_abs_drift']:.3e} -- "
        f"{'PASS (exactly unchanged)' if out_of_scope_unchanged else 'FAIL (leaked outside scope)'}"
    )

    print(f"  Resetting to base ({scope})...")
    _rpc(engine, "reset_to_base_weights", args=(), label="reset_to_base_weights")

    full_drift_after_reset = _rpc(engine, _diag_full_model_drift, args=(), label="_diag_full_model_drift")
    reset_exact = full_drift_after_reset["max_abs_drift"] == 0.0
    print(f"  Full-model drift after reset: max_abs_drift={full_drift_after_reset['max_abs_drift']:.3e} -- {'PASS (exact)' if reset_exact else 'FAIL (not exact)'}")

    test_pass = in_scope_changed and out_of_scope_unchanged and reset_exact
    return {
        "scope": scope,
        "in_scope_changed": in_scope_changed,
        "out_of_scope_unchanged": out_of_scope_unchanged,
        "reset_exact": reset_exact,
        "pass": test_pass,
        "in_scope_drift": drift["in_scope"],
        "out_of_scope_drift": drift["out_of_scope"],
        "full_model_drift_after_reset": full_drift_after_reset,
        "scope_param_count": drift["scope_param_count"],
        "out_of_scope_param_count": drift["out_of_scope_param_count"],
        "full_vlm_param_count": drift["full_vlm_param_count"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "scope_diagnosis" / "scope_isolation_report.json"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "Scope isolation GPU check",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision")

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    import ray

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines, launch_engines  # type: ignore

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines = None
    pgs = None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        engines, pgs = launch_engines(1, model_path, precision=cfg.model.precision, tensor_parallel_size=1, multimodal=True)
        engine = engines[0]
        try:
            print("Snapshotting base parameters (diagnostic-only, in-worker)...")
            snapshot_msg = _rpc(engine, _diag_snapshot_base, args=(), label="_diag_snapshot_base")
            print(f"  {snapshot_msg}")

            print("\nReporting all seven coarse scopes BEFORE any perturbation...")
            pre_report = _rpc(engine, _diag_report_all_scopes, args=(), label="_diag_report_all_scopes")
            print(f"  {pre_report['n_total_parameters']} total parameters")
            print(f"  detected LM namespace convention: {pre_report['detected_lm_namespace_convention']} ({pre_report['detected_lm_layer_count']} layers)")
            print("  representative parameter names:")
            for name in pre_report["representative_param_names"]:
                print(f"    {name}")
            for scope, summary in pre_report["scope_summaries"].items():
                alias_note = f" aliases={summary['aliases']}" if summary["aliases"] else ""
                print(
                    f"  scope={scope}: selected={summary['selected_param_count']} tensors, "
                    f"elements={summary['total_element_count']}, base_l2={summary['base_l2_norm']:.4f}{alias_note}"
                )

            test_a = _run_isolation_test(engine, "A", "vision_encoder")
            test_b = _run_isolation_test(engine, "B", "lm_middle")
        finally:
            cleanup_engines(engines, pgs)
    finally:
        if engines is None and ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    overall_pass = test_a["pass"] and test_b["pass"]
    report = {
        "pre_perturbation_report": pre_report,
        "test_seed": TEST_SEED,
        "test_sigma": TEST_SIGMA,
        "test_a_vision_encoder": test_a,
        "test_b_lm_middle": test_b,
        "overall": "PASS" if overall_pass else "FAIL",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nOVERALL: {report['overall']}")
    print(f"Test A (vision_encoder): {'PASS' if test_a['pass'] else 'FAIL'}")
    print(f"Test B (lm_middle): {'PASS' if test_b['pass'] else 'FAIL'}")
    if not overall_pass:
        print(
            "This is a mechanical scope-isolation check only -- it does not run a candidate "
            "search. Do not proceed to any N-candidate scoped run until this passes; inspect "
            "the report for which sub-check (in_scope_changed / out_of_scope_unchanged / "
            "reset_exact) failed.",
            file=sys.stderr,
        )
    print(f"Wrote {out_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
