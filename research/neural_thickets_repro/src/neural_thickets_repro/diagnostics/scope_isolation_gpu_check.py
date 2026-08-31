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

Tests C/D/E (vision_early / vision_middle / vision_late): identical shape, reusing the exact
same _run_isolation_test helper -- no new diagnostic framework -- for the three
fine-localization-inside-vision-encoder scopes added for the vision-encoder sub-scope
milestone (see SCOPED_PERTURBATION_DESIGN.md).

Tests F/G (vision_late_a / vision_late_b): same helper again, for the finer 5/5 split inside
vision_late added for the vision_late sub-scope milestone (see SCOPED_PERTURBATION_DESIGN.md
"vision_late sub-scopes" addendum).

Tests H/I (full_lm / full_vlm): same helper again -- the remaining two of the isolated
iclr_causal_density pilot's three preregistered scopes (vision_encoder is already Test A;
reports/iclr_causal_density/preregistration.md). This diagnostic's overall PASS is the
required precondition that pilot's own paired-condition evaluator checks before evaluating
any candidate in a given scope (see iclr_causal_density/evaluator.py's own docstring).

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
from ..run_global_visual_thicket_pilot import launch_stage6_engine, store_base_weights_via_rpc
from ..scopes import PERTURBATION_SCOPES, ScopeSelectionError, build_scope_manifest, discover_lm_layer_indices
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
    """ALIASES worker_self._base_weights (upstream's own store_base_weights() clone, already
    required to exist -- store_base_weights_via_rpc is called once in main() before this) as
    worker_self._scope_diag_base_state, rather than making a SEPARATE GPU-resident full clone
    of the model's state_dict. A third full-size (~15GB at 7B) copy of the weights -- on top
    of the live model's own weights and upstream's _base_weights clone -- genuinely does not
    fit a single 48GB L40S at 7B (confirmed live: torch.OutOfMemoryError during this exact
    clone). upstream's own store_base_weights() already clones every named_parameter()
    (never buffers) via p.data.clone() -- the identical values/tensors this diagnostic needs
    for drift comparison, so reusing that object is correct, not merely convenient. The one
    real difference from the previous state_dict()-based snapshot: buffers (e.g. any
    registered-but-non-trainable tensors) are no longer covered by the drift checks below --
    acceptable because neither scoped_apply_perturbation nor reset_to_base_weights ever
    touches buffers either (both iterate named_parameters()-derived name sets only), so a
    buffer's drift could never actually change via either mechanism this diagnostic exists to
    verify in the first place.
    """
    if not hasattr(worker_self, "_base_weights"):
        raise RuntimeError("_diag_snapshot_base: worker_self has no _base_weights -- store_base_weights_via_rpc must be called first.")
    worker_self._scope_diag_base_state = worker_self._base_weights
    return f"aliased {len(worker_self._scope_diag_base_state)} tensors from upstream's own store_base_weights() clone"


def _diag_report_all_scopes(worker_self) -> Dict:
    """Runs BEFORE either perturbation. Returns representative names, detected LM namespace
    convention/layer count, and a manifest summary for every one of the twelve coarse scopes
    this diagnostic knows about.

    ARCHITECTURE-DEPENDENT SCOPES (this repair pass, discovered live running against
    Qwen2.5-VL-7B-Instruct for the first time): lm_early/lm_middle/lm_late require the LM's
    layer count to be evenly divisible by 3 (partition_layers_into_thirds); vision_early/
    vision_middle/vision_late/vision_late_a/vision_late_b require EXACTLY 32 vision-encoder
    blocks (the 3B model's own count, hardcoded in scopes.py's own fixed 11/11/10 and 5/5
    partitions). Qwen2.5-VL-7B-Instruct has 28 LM layers -- NOT divisible by 3 --
    build_scope_manifest("lm_middle", ...) hard-raises ScopeSelectionError for this
    architecture, live-confirmed. This is a real, permanent architectural fact about 7B, not a
    transient failure -- catching it here per-scope (never silently skipping ALL scopes, never
    weakening vision_encoder/full_lm/full_vlm's own selection logic, which have no such
    divisibility/count dependency at all) lets the report -- and, more importantly, the
    isolation TESTS this pilot actually requires (vision_encoder/full_lm/full_vlm) -- proceed
    for this architecture, with the inapplicable scopes explicitly marked, never silently
    dropped from the JSON.
    """
    model = worker_self.model_runner.model
    named_parameters = list(model.named_parameters())
    all_names = [name for name, _ in named_parameters]
    alias_view = _try_named_parameters_no_duplicate(model)

    convention, layer_indices = discover_lm_layer_indices(all_names)

    scope_summaries = {}
    for scope in PERTURBATION_SCOPES:
        try:
            manifest = build_scope_manifest(scope, named_parameters, alias_named_parameters=alias_view)
        except ScopeSelectionError as exc:
            scope_summaries[scope] = {"applicable": False, "reason": str(exc)}
            continue
        scope_summaries[scope] = {
            "applicable": True,
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


def _run_isolation_test_or_skip(engine, test_label: str, scope: str) -> Dict:
    """ONLY for scopes NOT in this pilot's own required set (vision_encoder/full_lm/
    full_vlm -- Tests A/H/I, which always call _run_isolation_test directly, strictly, never
    through this wrapper): lm_early/lm_middle/lm_late/vision_early/vision_middle/vision_late/
    vision_late_a/vision_late_b (Tests B-G, the pre-existing WACV fine-localization scopes)
    depend on architecture-specific counts (LM layers divisible by 3; exactly 32 vision
    blocks) that do not hold for every model this diagnostic might ever be pointed at --
    confirmed live: Qwen2.5-VL-7B-Instruct has 28 LM layers, so build_scope_manifest("lm_
    middle", ...) hard-raises ScopeSelectionError. Catching that HERE (never around Tests
    A/H/I) lets this diagnostic's actually-required tests still run and their own pass/fail
    criteria stay completely unweakened; the skipped test is marked applicable=False,
    pass=None (never True, never silently counted as a pass) in the report.
    """
    try:
        result = _run_isolation_test(engine, test_label, scope)
        result["applicable"] = True
        return result
    except ScopeSelectionError as exc:
        print(f"\n=== Test {test_label} ({scope}) === SKIPPED -- not applicable to this model architecture: {exc}")
        return {"scope": scope, "applicable": False, "skip_reason": str(exc), "pass": None}


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
    from core.engine import cleanup_engines  # type: ignore

    ray_owned_by_us = bootstrap_ray(EXTERNAL_ROOT)

    engines = None
    pgs = None
    try:
        verify_workers_can_import_external_root(EXTERNAL_ROOT)

        # launch_stage6_engine, not external/RandOpt's own launch_engines: launch_engines()
        # accepts no max_model_len at all, so vLLM falls back to the model's full native
        # context (Qwen2.5-VL's is very large) -- at 7B this makes the KV-cache reservation
        # OOM regardless of gpu_memory_utilization (confirmed live on this pod: still OOMs
        # even at 0.60, the value that fixes every OTHER real-model script in this repo).
        # launch_stage6_engine is this repo's own, already-established, already-validated fix
        # for exactly this failure mode (see its own docstring's "Deliberately does NOT call
        # launch_engines()" section) -- STAGE6_MAX_MODEL_LEN (4096, irrelevant to correctness
        # here since this diagnostic never generates text at all -- pure weight-level check).
        # Mirrors launch_engines' own single-engine (TP=1) return shape ([engine], [pg]), so
        # cleanup_engines (still imported from external/RandOpt, unmodified) works unchanged.
        # UNLIKE launch_engines, this does NOT auto-store base weights on creation -- done
        # explicitly below, exactly once, before this diagnostic's own snapshot/perturb/reset
        # cycle (which relies on the upstream, string-dispatched "reset_to_base_weights" RPC
        # already having a base to reset to).
        #
        # gpu_memory_utilization=0.50 (NOT Stage6's own 0.60 default, and NOT this fix's own
        # first attempt at 0.40): confirmed live, both directions --
        #   - 0.60: OOM'd inside scoped_apply_perturbation's own (unmodified) delta.detach().
        #     float().pow(2).sum() on the largest scope's largest tensor once weights (15.6GB)
        #     + upstream's _base_weights clone (another ~15.6GB) + a 0.60-sized KV-cache
        #     reservation (7.67GB) had already consumed ~39GB of the 44.4GB usable card.
        #   - 0.40: vLLM itself refused to start ("No available memory for the cache blocks")
        #     -- weights (15.6GB) plus vLLM's own ~3.4GB fixed activation/workspace overhead
        #     already exceed a 0.40 budget (17.76GB) before any KV cache is even allocated.
        # This diagnostic NEVER calls engine.generate() (pure weight-level check -- see module
        # docstring), so its KV cache is pure overhead it should minimize, but vLLM still needs
        # a nonzero minimum -- 0.50 (22.2GB: weights + a real, if small, KV cache) is the
        # bisection between the two confirmed failure points, leaving ~22.2GB outside vLLM's
        # own reservation for this diagnostic's own base_weights clone + transient perturbation
        # work, without touching scoped_perturbation.py's own (unmodified) arithmetic at all.
        engines, pgs = launch_stage6_engine(model_path, precision=cfg.model.precision, tensor_parallel_size=1, gpu_memory_utilization=0.50)
        engine = engines[0]
        try:
            store_base_weights_via_rpc(engine)
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
                if not summary.get("applicable", True):
                    print(f"  scope={scope}: NOT APPLICABLE to this model architecture -- {summary['reason']}")
                    continue
                alias_note = f" aliases={summary['aliases']}" if summary["aliases"] else ""
                print(
                    f"  scope={scope}: selected={summary['selected_param_count']} tensors, "
                    f"elements={summary['total_element_count']}, base_l2={summary['base_l2_norm']:.4f}{alias_note}"
                )

            # Tests A/H/I are this pilot's OWN required scope-isolation precondition (vision_
            # encoder/full_lm/full_vlm -- reports/iclr_causal_density/preregistration.md) --
            # ALWAYS run strictly via _run_isolation_test (never the _or_skip wrapper): none of
            # the three depend on layer-count/block-count divisibility, so a failure here is a
            # REAL isolation bug, never architecture-inapplicability, and must never be
            # silently downgraded to a skip.
            test_a = _run_isolation_test(engine, "A", "vision_encoder")
            test_a["applicable"] = True
            # Tests B-G: the pre-existing WACV fine-localization scopes (SCOPED_PERTURBATION_
            # DESIGN.md) -- NOT part of this pilot's own required set, and (confirmed live)
            # structurally inapplicable to Qwen2.5-VL-7B-Instruct's 28-layer LM / vision-block
            # count -- run via _run_isolation_test_or_skip so an architecture mismatch here
            # never blocks Tests A/H/I. Reported for transparency, never gates overall_pass.
            test_b = _run_isolation_test_or_skip(engine, "B", "lm_middle")
            test_c = _run_isolation_test_or_skip(engine, "C", "vision_early")
            test_d = _run_isolation_test_or_skip(engine, "D", "vision_middle")
            test_e = _run_isolation_test_or_skip(engine, "E", "vision_late")
            test_f = _run_isolation_test_or_skip(engine, "F", "vision_late_a")
            test_g = _run_isolation_test_or_skip(engine, "G", "vision_late_b")
            # H/I: the iclr_causal_density pilot's own two remaining preregistered scopes
            # (vision_encoder above already covers its third) -- same strict, never-skipped
            # dispatch as Test A.
            test_h = _run_isolation_test(engine, "H", "full_lm")
            test_h["applicable"] = True
            test_i = _run_isolation_test(engine, "I", "full_vlm")
            test_i["applicable"] = True
        finally:
            cleanup_engines(engines, pgs)
    finally:
        if engines is None and ray_owned_by_us and ray.is_initialized():
            ray.shutdown()

    # overall_pass gates ONLY on this pilot's own required scopes (A/H/I) -- Tests B-G are
    # informational/legacy and, for an architecture where they are inapplicable, contribute
    # neither a PASS nor a silent FAIL to the gate this pilot's evaluator.py precondition
    # actually checks. A genuine isolation FAILURE (not merely "not applicable") on any
    # APPLICABLE test (required or legacy) still fails overall_pass -- "Do not weaken
    # tolerances or remove the failing scope" applies in full to every test that DID run.
    required_tests = {"A": test_a, "H": test_h, "I": test_i}
    legacy_tests = {"B": test_b, "C": test_c, "D": test_d, "E": test_e, "F": test_f, "G": test_g}
    required_pass = all(t["pass"] for t in required_tests.values())
    legacy_applicable_pass = all(t["pass"] for t in legacy_tests.values() if t.get("applicable"))
    overall_pass = required_pass and legacy_applicable_pass

    report = {
        "pre_perturbation_report": pre_report,
        "test_seed": TEST_SEED,
        "test_sigma": TEST_SIGMA,
        "test_a_vision_encoder": test_a,
        "test_b_lm_middle": test_b,
        "test_c_vision_early": test_c,
        "test_d_vision_middle": test_d,
        "test_e_vision_late": test_e,
        "test_f_vision_late_a": test_f,
        "test_g_vision_late_b": test_g,
        "test_h_full_lm": test_h,
        "test_i_full_vlm": test_i,
        "required_scopes_pass": required_pass,
        "required_scopes": sorted(required_tests),
        "overall": "PASS" if overall_pass else "FAIL",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    def _status(t: Dict) -> str:
        if not t.get("applicable", True):
            return "SKIPPED (not applicable to this architecture)"
        return "PASS" if t["pass"] else "FAIL"

    print(f"\nOVERALL: {report['overall']} (required_scopes_pass={required_pass})")
    print(f"Test A (vision_encoder) [REQUIRED]: {_status(test_a)}")
    print(f"Test B (lm_middle): {_status(test_b)}")
    print(f"Test C (vision_early): {_status(test_c)}")
    print(f"Test D (vision_middle): {_status(test_d)}")
    print(f"Test E (vision_late): {_status(test_e)}")
    print(f"Test F (vision_late_a): {_status(test_f)}")
    print(f"Test G (vision_late_b): {_status(test_g)}")
    print(f"Test H (full_lm) [REQUIRED]: {_status(test_h)}")
    print(f"Test I (full_vlm) [REQUIRED]: {_status(test_i)}")
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
