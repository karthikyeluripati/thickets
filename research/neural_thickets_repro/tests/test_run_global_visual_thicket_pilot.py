"""Tests for run_global_visual_thicket_pilot.py -- CPU-only. The real GPU/Ray/vLLM engine is
never launched for real; RPC dispatch is tested against a FAKE Ray-actor-shaped engine (see
_FakeRayEngine below) whose `.collective_rpc.remote(...)`/`.generate.remote(...)` return raw
values directly (paired with `ray_get=lambda x: x` injected into every call), so no real Ray
cluster is needed while still exercising the exact `.remote(...)`-then-`ray.get(...)` call
shape a real Ray actor handle presents.

This suite covers the FIXED-BASE RESTORATION repair pass: store_base_weights called exactly
once, reset_to_base_weights before AND after every perturbation, restore_self_weights NEVER
called by the Stage-6 lifecycle, exact (not tolerance-based) restoration verification,
checkpoint/resume durability, stale-output (run-signature) safety, and the paper-summary
incomplete-run refusal.
"""
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neural_thickets_repro.benchmarks.base import Example, ExampleScore, ParsedPrediction
from neural_thickets_repro.perturb_cpu import _generate_noise
from neural_thickets_repro.run_global_visual_thicket_pilot import (
    BASE_SNAPSHOT_MODE,
    CapabilityContext,
    CheckpointManifest,
    IncompatibleCheckpointError,
    IncompleteRunError,
    PERTURBATION_SEMANTICS,
    PILOT_CAPABILITIES,
    PilotConfigError,
    RESTORATION_MODE,
    RestorationFailedError,
    STAGE6_GPU_MEMORY_UTILIZATION,
    STAGE6_MAX_MODEL_LEN,
    UPSTREAM_SIGMA_GRID,
    WORKER_PERTURB_METHOD,
    WORKER_RESET_TO_BASE_METHOD,
    WORKER_RESTORE_METHOD,
    WORKER_STORE_BASE_METHOD,
    append_candidate_rows,
    build_d_map_context,
    build_delta_matrix,
    build_pilot_plan,
    build_run_manifest_summary,
    build_stage6_checkpoint_manifest,
    build_stage6_engine_config,
    build_stage6_perturbation_population,
    compute_diversity_summary,
    compute_figure2_summary,
    compute_mask_info_via_rpc,
    compute_run_signature,
    ensure_checkpoint_manifest,
    evaluate_one_perturbation_rpc,
    format_pilot_plan,
    load_completed_perturbation_rows,
    load_or_compute_baseline_scores,
    load_records,
    perturb_via_rpc,
    reset_to_base_weights_via_rpc,
    restore_via_rpc,
    run_pilot_rpc,
    store_base_weights_via_rpc,
    verify_exact_fixed_base_restoration_via_rpc,
    write_paper_summary,
)
from neural_thickets_repro.thicket.perturbation import PerturbationManifest, generate_perturbation_population


def _raw_config(**overrides):
    cfg = {
        "model": {"name": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "rev1", "family": "qwen2_5_vl", "scale": "3B"},
        "pilot": {
            "capabilities": list(PILOT_CAPABILITIES),
            "sigma_grid": list(UPSTREAM_SIGMA_GRID),
            "perturbations_per_sigma": 64,
            "examples_per_capability": 50,
            "base_seed": 42,
        },
        "outputs": {"root": "results/visual_thicket_global_3b_pilot"},
    }
    cfg.update(overrides)
    return cfg


def _identity_ray_get(x):
    return x


# =============================================================================================
# Fake Ray-actor-shaped engine -- fixed-base lifecycle (store_base_weights/reset_to_base_
# weights), using REAL small torch tensors so the exact-restoration Callable (which calls
# diagnostics.perturb_restore_drift.measure_drift -- real tensor arithmetic) works end to end.
# =============================================================================================


class _FakeRayEngine:
    """Duck-types a Ray actor handle wrapping a vLLM engine with worker_extension_cls=
    utils.worker_extn.WorkerExtension already attached. Deliberately has NO `llm_engine`/
    `model_executor`/`driver_worker` attribute at all.
    """

    def __init__(self, param_shapes, visual_prefixes=("visual.",)):
        self._values = {name: torch.zeros(n) for name, n in param_shapes.items()}
        self._base_weights = None
        self._visual_prefixes = visual_prefixes
        self.collective_rpc_calls = []
        self.generate_call_count = 0
        self.collective_rpc = SimpleNamespace(remote=self._collective_rpc)
        self.generate = SimpleNamespace(remote=self._generate)

    def _should_perturb(self, name):
        return not name.startswith(self._visual_prefixes)

    def _named_parameters(self):
        return list(self._values.items())

    def _perturb(self, seed, sigma, negate):
        sign = -1.0 if negate else 1.0
        for name, t in self._values.items():
            if self._should_perturb(name):
                noise = _generate_noise(t, seed)
                self._values[name] = t + sign * sigma * noise

    def _collective_rpc(self, method, args=()):
        label = method if isinstance(method, str) else getattr(method, "__name__", "callable")
        self.collective_rpc_calls.append((label, args))
        if method == WORKER_PERTURB_METHOD:
            seed, sigma, negate = args
            self._perturb(seed, sigma, negate)
            return [True]
        if method == WORKER_RESTORE_METHOD:
            seed, sigma, negate = args
            self._perturb(seed, sigma, not negate)
            return [True]
        if method == WORKER_STORE_BASE_METHOD:
            self._base_weights = {name: t.clone() for name, t in self._values.items()}
            return [True]
        if method == WORKER_RESET_TO_BASE_METHOD:
            if self._base_weights is None:
                raise RuntimeError("Must call store_base_weights first")
            self._values = {name: t.clone() for name, t in self._base_weights.items()}
            return [True]
        if callable(method):
            worker_kwargs = dict(
                model_runner=SimpleNamespace(model=SimpleNamespace(named_parameters=self._named_parameters)),
                _should_perturb=self._should_perturb,
            )
            if self._base_weights is not None:
                worker_kwargs["_base_weights"] = self._base_weights
            worker_self = SimpleNamespace(**worker_kwargs)
            return [method(worker_self, *args)]
        raise ValueError(f"unsupported collective_rpc method {method!r}")

    def _generate(self, requests, sampling_params, use_tqdm=True):
        self.generate_call_count += 1
        return [SimpleNamespace(outputs=[SimpleNamespace(text="ok")]) for _ in requests]


def _fake_engine():
    return _FakeRayEngine({"visual.blocks.0.weight": 2, "model.layers.0.weight": 3, "model.layers.1.weight": 2})


def _fake_engine_with_base_stored():
    engine = _fake_engine()
    store_base_weights_via_rpc(engine, ray_get=_identity_ray_get)
    return engine


# --- upstream sigma grid discovery/config -----------------------------------------------------


def test_upstream_sigma_grid_matches_confirmed_pinned_commit_default():
    assert UPSTREAM_SIGMA_GRID == (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01)


def test_perturbation_semantics_and_restoration_mode_are_frozen_and_distinct():
    assert PERTURBATION_SEMANTICS == "global_gaussian_upstream"
    assert RESTORATION_MODE == "fixed_base"
    assert PERTURBATION_SEMANTICS != RESTORATION_MODE


# --- PilotPlan arithmetic + dry-run formatting -------------------------------------------------


def test_build_pilot_plan_from_default_config():
    plan = build_pilot_plan(_raw_config())
    assert plan.capabilities == PILOT_CAPABILITIES
    assert plan.sigma_grid == UPSTREAM_SIGMA_GRID
    assert plan.total_unique_perturbations == 6 * 64
    assert plan.total_perturbation_capability_evaluations == 6 * 64 * 3
    assert plan.baseline_evaluations == 3 * 50
    assert plan.total_model_example_evaluations == 6 * 64 * 3 * 50 + 3 * 50


def test_build_pilot_plan_smoke_overrides_do_not_touch_config_dict():
    raw = _raw_config()
    plan = build_pilot_plan(raw, perturbations_per_sigma=2, subset_size=5)
    assert plan.perturbations_per_sigma == 2
    assert plan.examples_per_capability == 5
    assert raw["pilot"]["perturbations_per_sigma"] == 64
    assert raw["pilot"]["examples_per_capability"] == 50


def test_build_pilot_plan_rejects_wrong_capability_set():
    raw = _raw_config()
    raw["pilot"]["capabilities"] = ["visual_grounding", "counting", "spatial_reasoning"]
    with pytest.raises(PilotConfigError):
        build_pilot_plan(raw)


def test_build_pilot_plan_rejects_invented_sigma_grid():
    raw = _raw_config()
    raw["pilot"]["sigma_grid"] = [0.001, 0.5]
    with pytest.raises(PilotConfigError):
        build_pilot_plan(raw)


def test_build_pilot_plan_accepts_sigma_grid_in_any_order():
    raw = _raw_config()
    raw["pilot"]["sigma_grid"] = list(reversed(UPSTREAM_SIGMA_GRID))
    plan = build_pilot_plan(raw)
    assert set(plan.sigma_grid) == set(UPSTREAM_SIGMA_GRID)


def test_format_pilot_plan_prints_every_required_field_and_no_dollar_cost():
    plan = build_pilot_plan(_raw_config())
    text = format_pilot_plan(plan)
    for expected in (
        "model_name", "model_revision", "capabilities", "sigma_grid", "perturbations_per_sigma",
        "total_unique_perturbations", "examples_per_capability", "total_perturbation_x_capability_evaluations",
        "total_model_example_evaluations", "expected_model_loading_strategy", "output_dir",
        "run_signature", "perturbation_semantics", "restoration_mode",
    ):
        assert expected in text
    assert "$" not in text


def test_format_pilot_plan_documents_fixed_base_lifecycle_not_frontend_access():
    plan = build_pilot_plan(_raw_config())
    text = format_pilot_plan(plan)
    assert "collective_rpc" in text
    assert "reset_to_base_weights" in text
    assert "store_base_weights" in text
    assert "restore_self_weights is NEVER called" in text
    assert "llm_engine.model_executor" not in text.replace("Zero frontend llm_engine.model_executor access anywhere in this path.", "")


# --- run signature / stale-output safety (spec section 6) --------------------------------------


def test_compute_run_signature_is_full_when_values_match_paper_config():
    assert compute_run_signature(64, 50, 64, 50) == "full"


def test_compute_run_signature_is_smoke_when_either_value_differs():
    assert compute_run_signature(2, 50, 64, 50) == "smoke_p2_n50"
    assert compute_run_signature(64, 5, 64, 50) == "smoke_p64_n5"
    assert compute_run_signature(2, 5, 64, 50) == "smoke_p2_n5"


def test_build_pilot_plan_full_config_gets_full_output_dir_suffix():
    plan = build_pilot_plan(_raw_config())
    assert plan.run_signature == "full"
    assert plan.output_dir.name == "full"


def test_build_pilot_plan_override_gets_smoke_output_dir_suffix():
    plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=2, subset_size=5)
    assert plan.run_signature == "smoke_p2_n5"
    assert plan.output_dir.name == "smoke_p2_n5"


def test_full_and_smoke_plans_never_share_an_output_directory():
    full_plan = build_pilot_plan(_raw_config())
    smoke_plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=2, subset_size=5)
    assert full_plan.output_dir != smoke_plan.output_dir


def test_explicit_output_dir_still_gets_run_signature_appended():
    plan = build_pilot_plan(_raw_config(), output_dir="/tmp/custom_root", perturbations_per_sigma=2, subset_size=5)
    assert str(plan.output_dir).replace("\\", "/") == "/tmp/custom_root/smoke_p2_n5"


# --- D_map construction -------------------------------------------------------------------------


class _FixedPoolBenchmark:
    capability = "fake_capability"

    def __init__(self, examples):
        self._examples = examples

    def load_examples(self, cfg):
        return self._examples

    def subset_selection_rule(self):
        return "shuffled_prefix"


def _pool(n):
    return [Example(example_id=f"ex{i}", image=None, prompt_input={"q": i}, target=i) for i in range(n)]


def test_build_d_map_context_selects_requested_size_and_persists(tmp_path):
    benchmark = _FixedPoolBenchmark(_pool(200))
    ctx = build_d_map_context(benchmark, cfg=None, capability="visual_grounding", n=50, seed=1, subset_ids_dir=tmp_path)
    assert len(ctx.examples) == 50
    assert (tmp_path / "visual_grounding_d_map_50.json").exists()


def test_build_d_map_context_is_deterministic_across_calls(tmp_path):
    benchmark = _FixedPoolBenchmark(_pool(200))
    ctx1 = build_d_map_context(benchmark, cfg=None, capability="visual_grounding", n=50, seed=1, subset_ids_dir=tmp_path)
    ctx2 = build_d_map_context(benchmark, cfg=None, capability="visual_grounding", n=50, seed=1, subset_ids_dir=tmp_path)
    assert [e.example_id for e in ctx1.examples] == [e.example_id for e in ctx2.examples]
    assert ctx1.subset_hash == ctx2.subset_hash


# --- RPC dispatch: store_base_weights / reset_to_base_weights / perturb / mask / verify --------


def test_store_base_weights_via_rpc_dispatches_upstream_method():
    engine = _fake_engine()
    store_base_weights_via_rpc(engine, ray_get=_identity_ray_get)
    assert engine.collective_rpc_calls == [(WORKER_STORE_BASE_METHOD, ())]
    assert engine._base_weights is not None


def test_reset_to_base_weights_via_rpc_dispatches_upstream_method():
    engine = _fake_engine_with_base_stored()
    engine.collective_rpc_calls.clear()
    reset_to_base_weights_via_rpc(engine, ray_get=_identity_ray_get)
    assert engine.collective_rpc_calls == [(WORKER_RESET_TO_BASE_METHOD, ())]


def test_reset_to_base_weights_raises_if_store_base_weights_never_called():
    engine = _fake_engine()
    with pytest.raises(RuntimeError, match="store_base_weights"):
        reset_to_base_weights_via_rpc(engine, ray_get=_identity_ray_get)


def test_perturb_via_rpc_dispatches_upstream_method_name_and_args():
    engine = _fake_engine()
    perturb_via_rpc(engine, seed=1, sigma=0.01, ray_get=_identity_ray_get)
    assert engine.collective_rpc_calls == [(WORKER_PERTURB_METHOD, (1, 0.01, False))]


def test_restore_via_rpc_still_dispatches_the_historical_upstream_method_when_called_directly():
    """restore_self_weights remains a real, correctly-wired call (spec: "may remain in the
    repository for reproduction") -- it is simply never invoked by the Stage-6 lifecycle
    itself (see the separate never-called assertions below).
    """
    engine = _fake_engine()
    restore_via_rpc(engine, seed=1, sigma=0.01, ray_get=_identity_ray_get)
    assert engine.collective_rpc_calls == [(WORKER_RESTORE_METHOD, (1, 0.01, False))]


def test_perturb_then_reset_returns_to_stored_base_exactly():
    engine = _fake_engine_with_base_stored()
    base_snapshot = {k: v.clone() for k, v in engine._values.items()}
    perturb_via_rpc(engine, seed=7, sigma=0.05, ray_get=_identity_ray_get)
    assert not all(torch.equal(engine._values[k], base_snapshot[k]) for k in base_snapshot)  # actually perturbed
    reset_to_base_weights_via_rpc(engine, ray_get=_identity_ray_get)
    assert all(torch.equal(engine._values[k], base_snapshot[k]) for k in base_snapshot)  # exactly restored


def test_compute_mask_info_via_rpc_excludes_visual_and_computed_inside_worker():
    engine = _fake_engine()
    info = compute_mask_info_via_rpc(engine, ray_get=_identity_ray_get)
    assert info["param_count"] == 2  # model.layers.0/1, not visual.blocks.0
    assert "mask_hash" in info


def test_verify_exact_fixed_base_restoration_via_rpc_passes_when_reset_correctly():
    engine = _fake_engine_with_base_stored()
    perturb_via_rpc(engine, seed=3, sigma=0.02, ray_get=_identity_ray_get)
    reset_to_base_weights_via_rpc(engine, ray_get=_identity_ray_get)
    result = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=_identity_ray_get)
    assert result["ok"] is True
    assert result["max_abs_drift"] == 0.0


def test_verify_exact_fixed_base_restoration_via_rpc_fails_when_never_reset():
    engine = _fake_engine_with_base_stored()
    perturb_via_rpc(engine, seed=3, sigma=0.5, ray_get=_identity_ray_get)  # never reset
    result = verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=_identity_ray_get)
    assert result["ok"] is False
    assert result["max_abs_drift"] > 0.0


def test_never_accesses_llm_engine_or_model_executor_attributes():
    engine = _fake_engine_with_base_stored()
    assert not hasattr(engine, "llm_engine")
    assert not hasattr(engine, "model_executor")
    assert not hasattr(engine, "driver_worker")
    perturb_via_rpc(engine, seed=1, sigma=0.01, ray_get=_identity_ray_get)
    reset_to_base_weights_via_rpc(engine, ray_get=_identity_ray_get)
    compute_mask_info_via_rpc(engine, ray_get=_identity_ray_get)
    verify_exact_fixed_base_restoration_via_rpc(engine, ray_get=_identity_ray_get)


# --- evaluate_one_perturbation_rpc: fixed-base lifecycle ----------------------------------------


class _RecordingBenchmark:
    def __init__(self, name):
        self.capability = name

    def load_examples(self, cfg):
        raise NotImplementedError

    def subset_selection_rule(self):
        return "shuffled_prefix"

    def prepare_image(self, example):
        return "fake_image"

    def build_prompt(self, example):
        return []

    def parse_prediction(self, raw_generation, example):
        return ParsedPrediction(parsed=raw_generation, parse_ok=True)

    def score_example(self, parsed, example):
        return ExampleScore(score=1.0, correct=True)

    def aggregate_metrics(self, scores):
        return {"primary_metric": 0.5, "parser_failure_rate": 0.0}


def _fake_tokenizer():
    return SimpleNamespace(apply_chat_template=lambda messages, add_generation_prompt, tokenize: "TEXT")


def _fake_llm_adapter():
    def _generate(requests, sampling_params, use_tqdm=True):
        return [SimpleNamespace(outputs=[SimpleNamespace(text="ok")]) for _ in requests]

    return SimpleNamespace(generate=_generate)


def _build_contexts(capabilities, n_examples=3):
    from neural_thickets_repro.thicket.data_roles import partition_data_roles

    contexts = {}
    for capability in capabilities:
        examples = [Example(example_id=f"{capability}_{j}", image=None, prompt_input={}, target=0) for j in range(n_examples)]
        partition = partition_data_roles([e.example_id for e in examples], sizes={"map": n_examples}, seed=1)
        bench = _RecordingBenchmark(capability)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=bench, examples=examples, partition=partition, subset_hash=partition.manifest_hash, base_score=0.4)
    return contexts


def _manifest(seed=1, sigma=0.01):
    return PerturbationManifest(seed=seed, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=sigma)


def test_same_perturbation_id_appears_for_all_capabilities():
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = _manifest()

    records = evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)

    assert len(records) == 3
    assert {r.perturbation_id for r in records} == {manifest.perturbation_id}
    assert {r.capability for r in records} == set(PILOT_CAPABILITIES)


def test_evaluate_one_perturbation_rpc_calls_reset_before_and_after_perturb():
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = _manifest()
    engine.collective_rpc_calls.clear()

    evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)

    labels = [label for label, _ in engine.collective_rpc_calls]
    assert labels[0] == WORKER_RESET_TO_BASE_METHOD  # reset before perturb
    assert labels[1] == WORKER_PERTURB_METHOD
    assert labels[1:].count(WORKER_RESET_TO_BASE_METHOD) == 1
    assert labels[-2] == WORKER_RESET_TO_BASE_METHOD  # reset after evaluating, before verify
    assert labels[-1] == "verify_exact_fixed_base_restoration_rpc"


def test_evaluate_one_perturbation_rpc_receives_exactly_the_manifest_seed_and_sigma():
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = _manifest(seed=1480723517, sigma=0.01)
    engine.collective_rpc_calls.clear()

    evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)

    perturb_call = next(c for c in engine.collective_rpc_calls if c[0] == WORKER_PERTURB_METHOD)
    assert perturb_call[1] == (1480723517, 0.01, False)


def test_evaluate_one_perturbation_rpc_never_calls_restore_self_weights():
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = _manifest()
    engine.collective_rpc_calls.clear()

    evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)

    assert all(label != WORKER_RESTORE_METHOD for label, _ in engine.collective_rpc_calls)


def test_evaluate_one_perturbation_rpc_restores_exactly_after_evaluating():
    engine = _fake_engine_with_base_stored()
    base_snapshot = {k: v.clone() for k, v in engine._values.items()}
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = _manifest()

    evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)

    assert all(torch.equal(engine._values[k], base_snapshot[k]) for k in base_snapshot)


def test_evaluate_one_perturbation_rpc_aborts_on_exact_restoration_failure():
    """Forces a real restoration failure: an engine whose reset_to_base_weights is broken
    (never actually resets) must trip RestorationFailedError.
    """
    engine = _fake_engine_with_base_stored()

    def _broken_collective_rpc(method, args=()):
        if method == WORKER_RESET_TO_BASE_METHOD:
            engine.collective_rpc_calls.append((WORKER_RESET_TO_BASE_METHOD, args))
            return [True]  # silently does nothing -- current values stay perturbed
        return _FakeRayEngine._collective_rpc(engine, method, args)

    engine.collective_rpc = SimpleNamespace(remote=_broken_collective_rpc)
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = _manifest(sigma=0.5)

    with pytest.raises(RestorationFailedError):
        evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)


def test_no_perturbation_accumulation_across_candidates():
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    base_snapshot = {k: v.clone() for k, v in engine._values.items()}

    evaluate_one_perturbation_rpc(engine, _manifest(seed=1, sigma=0.02), contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)
    evaluate_one_perturbation_rpc(engine, _manifest(seed=2, sigma=0.02), contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)

    assert all(torch.equal(engine._values[k], base_snapshot[k]) for k in base_snapshot)


# --- build_stage6_perturbation_population: unchanged by the restoration-mode fix ---------------


def test_build_stage6_perturbation_population_every_seed_in_numpy_uint32_domain():
    from neural_thickets_repro.run_global_visual_thicket_pilot import NUMPY_SEED_DOMAIN

    plan = build_pilot_plan(_raw_config())
    population = build_stage6_perturbation_population(plan, base_seed=20260823, parameter_mask_hash="hash1")
    assert all(0 <= m.seed < NUMPY_SEED_DOMAIN for m in population)


def test_build_stage6_perturbation_population_full_grid_has_384_unique_seeds():
    plan = build_pilot_plan(_raw_config())  # paper config: 6 sigmas x 64 = 384
    population = build_stage6_perturbation_population(plan, base_seed=20260823, parameter_mask_hash="hash1")
    assert len(population) == 384
    assert len({m.seed for m in population}) == 384
    assert len({m.perturbation_id for m in population}) == 384


def test_build_stage6_perturbation_population_is_deterministic_and_independent_of_restoration_mode():
    """The population-generation function takes no restoration-mode parameter at all -- it
    cannot be affected by this repair pass's restoration-mode change. Calling it twice with
    identical inputs still reproduces the identical population (same seeds, same IDs).
    """
    import inspect

    sig = inspect.signature(build_stage6_perturbation_population)
    assert "restoration_mode" not in sig.parameters

    plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=5)
    pop_1 = build_stage6_perturbation_population(plan, base_seed=20260823, parameter_mask_hash="hash1")
    pop_2 = build_stage6_perturbation_population(plan, base_seed=20260823, parameter_mask_hash="hash1")
    assert [m.seed for m in pop_1] == [m.seed for m in pop_2]
    assert [m.perturbation_id for m in pop_1] == [m.perturbation_id for m in pop_2]


def test_known_failed_candidate_sigma_and_domain_membership_are_reproducible():
    """The real RunPod failure reported perturbation_id=5a417b7937eca5ad522e9c6b,
    seed=1480723517, sigma=0.01. The exact perturbation_id/seed also depend on the real
    model's own parameter_mask_hash (computed on the live GPU worker), which this CPU-only
    session cannot regenerate -- but the reported seed and sigma are independently checkable
    facts: 0.01 is a real member of the frozen sigma grid, and 1480723517 is a valid uint32
    worker seed, exactly the domain build_stage6_perturbation_population guarantees for every
    manifest post-fix.
    """
    from neural_thickets_repro.run_global_visual_thicket_pilot import NUMPY_SEED_DOMAIN

    reported_seed = 1480723517
    reported_sigma = 0.01
    assert reported_sigma in UPSTREAM_SIGMA_GRID
    assert 0 <= reported_seed < NUMPY_SEED_DOMAIN


# --- checkpoint manifest: identity + hard-fail on incompatibility -------------------------------


def _checkpoint(**overrides):
    kwargs = dict(
        experiment_id="visual_thicket_global_3b_pilot", run_signature="full", restoration_mode="fixed_base",
        perturbation_semantics="global_gaussian_upstream", model_revision="rev1",
        subset_hashes={"visual_grounding": "h1", "ocr_text_recognition_grounded": "h2", "spatial_reasoning": "h3"},
        subset_size=50, perturbations_per_sigma=64, expected_unique_perturbations=384, expected_result_rows=1152,
    )
    kwargs.update(overrides)
    return CheckpointManifest(**kwargs)


def test_checkpoint_manifest_round_trips_through_json():
    checkpoint = _checkpoint()
    restored = CheckpointManifest.from_dict(json.loads(json.dumps(checkpoint.to_dict())))
    assert restored == checkpoint


def test_ensure_checkpoint_manifest_creates_when_absent(tmp_path):
    path = tmp_path / "checkpoint_manifest.json"
    checkpoint = _checkpoint()
    result = ensure_checkpoint_manifest(path, checkpoint)
    assert result == checkpoint
    assert path.exists()


def test_ensure_checkpoint_manifest_passes_when_identical(tmp_path):
    path = tmp_path / "checkpoint_manifest.json"
    checkpoint = _checkpoint()
    ensure_checkpoint_manifest(path, checkpoint)
    result = ensure_checkpoint_manifest(path, checkpoint)  # resume attempt
    assert result == checkpoint


def test_ensure_checkpoint_manifest_hard_fails_on_restoration_mode_mismatch(tmp_path):
    """The exact real-world hazard this guards against: a prior run's checkpoint used the OLD
    subtractive restoration_mode -- resuming it into a fixed_base run must never be silent.
    """
    path = tmp_path / "checkpoint_manifest.json"
    old_checkpoint = _checkpoint(restoration_mode="released_compat_subtractive")
    ensure_checkpoint_manifest(path, old_checkpoint)

    new_checkpoint = _checkpoint(restoration_mode="fixed_base")
    with pytest.raises(IncompatibleCheckpointError):
        ensure_checkpoint_manifest(path, new_checkpoint)


def test_ensure_checkpoint_manifest_hard_fails_on_subset_hash_mismatch(tmp_path):
    path = tmp_path / "checkpoint_manifest.json"
    ensure_checkpoint_manifest(path, _checkpoint())
    with pytest.raises(IncompatibleCheckpointError):
        ensure_checkpoint_manifest(path, _checkpoint(subset_hashes={"visual_grounding": "DIFFERENT", "ocr_text_recognition_grounded": "h2", "spatial_reasoning": "h3"}))


def test_ensure_checkpoint_manifest_hard_fails_on_run_signature_mismatch(tmp_path):
    path = tmp_path / "checkpoint_manifest.json"
    ensure_checkpoint_manifest(path, _checkpoint(run_signature="full"))
    with pytest.raises(IncompatibleCheckpointError):
        ensure_checkpoint_manifest(path, _checkpoint(run_signature="smoke_p2_n5"))


# --- checkpoint/resume: persistence, skip-completed, rerun-incomplete --------------------------


def test_append_candidate_rows_and_load_records_round_trip(tmp_path):
    results_path = tmp_path / "results.jsonl"
    contexts = _build_contexts(PILOT_CAPABILITIES)
    engine = _fake_engine_with_base_stored()
    manifest = _manifest()
    records = evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)

    append_candidate_rows(results_path, records)

    loaded = load_records(results_path)
    assert len(loaded) == 3
    assert {r.perturbation_id for r in loaded} == {manifest.perturbation_id}


def test_load_completed_perturbation_rows_excludes_incomplete_groups(tmp_path):
    results_path = tmp_path / "results.jsonl"
    contexts = _build_contexts(PILOT_CAPABILITIES)
    engine = _fake_engine_with_base_stored()
    complete_manifest = _manifest(seed=1)
    incomplete_manifest = _manifest(seed=2)

    complete_records = evaluate_one_perturbation_rpc(engine, complete_manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)
    append_candidate_rows(results_path, complete_records)
    incomplete_records = evaluate_one_perturbation_rpc(engine, incomplete_manifest, contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)
    append_candidate_rows(results_path, incomplete_records[:2])  # simulate a crash mid-write -- only 2/3 rows

    completed = load_completed_perturbation_rows(results_path, PILOT_CAPABILITIES)
    assert set(completed) == {complete_manifest.perturbation_id}


def test_run_pilot_rpc_persists_a_candidate_only_after_restoration_passes(tmp_path):
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=1, subset_size=3, output_dir=str(tmp_path))

    records = run_pilot_rpc(plan, contexts, engine, _fake_tokenizer(), None, base_seed=42, parameter_mask_hash="hash1", ray_get=_identity_ray_get)

    results_path = plan.output_dir / "results.jsonl"
    assert results_path.exists()
    on_disk = load_records(results_path)
    assert len(on_disk) == len(records) == plan.total_perturbation_capability_evaluations


def test_run_pilot_rpc_resumes_and_skips_already_completed_candidates(tmp_path):
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=2, subset_size=3, output_dir=str(tmp_path))

    # First (interrupted) partial run: complete only.
    run_pilot_rpc(plan, contexts, engine, _fake_tokenizer(), None, base_seed=42, parameter_mask_hash="hash1", ray_get=_identity_ray_get)
    first_call_count = engine.generate_call_count

    # "Resume": same plan/engine identity -- must not re-evaluate already-persisted candidates.
    engine2 = _fake_engine_with_base_stored()
    engine2.generate_call_count = 0
    records = run_pilot_rpc(plan, contexts, engine2, _fake_tokenizer(), None, base_seed=42, parameter_mask_hash="hash1", ray_get=_identity_ray_get)

    assert engine2.generate_call_count == 0  # nothing re-run -- everything was already complete
    assert len(records) == plan.total_perturbation_capability_evaluations


def test_run_pilot_rpc_reruns_an_incomplete_candidate_after_interruption(tmp_path):
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=1, subset_size=3, output_dir=str(tmp_path))

    population = build_stage6_perturbation_population(plan, base_seed=42, parameter_mask_hash="hash1")
    # Simulate an interrupted candidate: only 2/3 capability rows durably persisted.
    incomplete_records = evaluate_one_perturbation_rpc(engine, population[0], contexts, _fake_tokenizer(), None, ray_get=_identity_ray_get)
    build_stage6_checkpoint_manifest(plan, contexts)  # sanity: constructible
    ensure_checkpoint_manifest(plan.output_dir / "checkpoint_manifest.json", build_stage6_checkpoint_manifest(plan, contexts))
    append_candidate_rows(plan.output_dir / "results.jsonl", incomplete_records[:2])

    records = run_pilot_rpc(plan, contexts, engine, _fake_tokenizer(), None, base_seed=42, parameter_mask_hash="hash1", ray_get=_identity_ray_get)

    by_pid = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, set()).add(r.capability)
    assert all(caps == set(PILOT_CAPABILITIES) for caps in by_pid.values())  # the incomplete one was fully rerun


def test_run_pilot_rpc_hard_fails_on_incompatible_existing_checkpoint(tmp_path):
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=1, subset_size=3, output_dir=str(tmp_path))

    incompatible = build_stage6_checkpoint_manifest(plan, contexts)
    object.__setattr__(incompatible, "restoration_mode", "released_compat_subtractive")  # simulate the OLD run's checkpoint
    (plan.output_dir).mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "checkpoint_manifest.json").write_text(json.dumps(incompatible.to_dict(), indent=2))

    with pytest.raises(IncompatibleCheckpointError):
        run_pilot_rpc(plan, contexts, engine, _fake_tokenizer(), None, base_seed=42, parameter_mask_hash="hash1", ray_get=_identity_ray_get)


def test_run_pilot_rpc_full_config_expects_1152_rows_and_384_unique_perturbations(tmp_path):
    plan = build_pilot_plan(_raw_config(), output_dir=str(tmp_path))
    assert plan.total_perturbation_capability_evaluations == 1152
    assert plan.total_unique_perturbations == 384


def test_run_pilot_rpc_uses_stage6_seed_domain_and_still_aligns_ids_across_capabilities(tmp_path):
    from neural_thickets_repro.run_global_visual_thicket_pilot import NUMPY_SEED_DOMAIN

    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    plan = build_pilot_plan(_raw_config(), perturbations_per_sigma=3, subset_size=2, output_dir=str(tmp_path))

    records = run_pilot_rpc(plan, contexts, engine, _fake_tokenizer(), None, base_seed=20260823, parameter_mask_hash="hash1", ray_get=_identity_ray_get)

    by_pid = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, set()).add(r.capability)
        assert 0 <= r.seed < NUMPY_SEED_DOMAIN
    assert all(caps == set(PILOT_CAPABILITIES) for caps in by_pid.values())
    assert len(by_pid) == plan.total_unique_perturbations


# --- baseline persistence (spec section 8) -------------------------------------------------------


def test_load_or_compute_baseline_scores_computes_and_persists_when_absent(tmp_path):
    engine = _fake_engine_with_base_stored()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    baseline_path = tmp_path / "baseline_scores.json"

    load_or_compute_baseline_scores(baseline_path, contexts, model_revision="rev1", run_signature="full", llm_adapter=_fake_llm_adapter(), tokenizer=_fake_tokenizer(), sampling_params=None)

    assert baseline_path.exists()
    for ctx in contexts.values():
        assert ctx.base_score == 0.5  # _RecordingBenchmark's fixed primary_metric


def test_load_or_compute_baseline_scores_reuses_when_compatible(tmp_path):
    contexts_1 = _build_contexts(PILOT_CAPABILITIES)
    baseline_path = tmp_path / "baseline_scores.json"
    load_or_compute_baseline_scores(baseline_path, contexts_1, model_revision="rev1", run_signature="full", llm_adapter=_fake_llm_adapter(), tokenizer=_fake_tokenizer(), sampling_params=None)

    contexts_2 = _build_contexts(PILOT_CAPABILITIES)
    for capability in contexts_2:
        contexts_2[capability].subset_hash = contexts_1[capability].subset_hash  # identical identity
    load_or_compute_baseline_scores(baseline_path, contexts_2, model_revision="rev1", run_signature="full", llm_adapter=_fake_llm_adapter(), tokenizer=_fake_tokenizer(), sampling_params=None)

    for capability in contexts_2:
        assert contexts_2[capability].base_score == contexts_1[capability].base_score


def test_load_or_compute_baseline_scores_hard_fails_on_subset_hash_mismatch(tmp_path):
    contexts_1 = _build_contexts(PILOT_CAPABILITIES)
    baseline_path = tmp_path / "baseline_scores.json"
    load_or_compute_baseline_scores(baseline_path, contexts_1, model_revision="rev1", run_signature="full", llm_adapter=_fake_llm_adapter(), tokenizer=_fake_tokenizer(), sampling_params=None)

    contexts_2 = _build_contexts(PILOT_CAPABILITIES)  # fresh contexts -> different subset_hash (different seeded partition object identity is fine, but let's force a mismatch explicitly)
    contexts_2["visual_grounding"].subset_hash = "definitely_different"
    with pytest.raises(IncompatibleCheckpointError):
        load_or_compute_baseline_scores(baseline_path, contexts_2, model_revision="rev1", run_signature="full", llm_adapter=_fake_llm_adapter(), tokenizer=_fake_tokenizer(), sampling_params=None)


def test_load_or_compute_baseline_scores_hard_fails_on_model_revision_mismatch(tmp_path):
    contexts_1 = _build_contexts(PILOT_CAPABILITIES)
    baseline_path = tmp_path / "baseline_scores.json"
    load_or_compute_baseline_scores(baseline_path, contexts_1, model_revision="rev1", run_signature="full", llm_adapter=_fake_llm_adapter(), tokenizer=_fake_tokenizer(), sampling_params=None)

    contexts_2 = _build_contexts(PILOT_CAPABILITIES)
    for capability in contexts_2:
        contexts_2[capability].subset_hash = contexts_1[capability].subset_hash
    with pytest.raises(IncompatibleCheckpointError):
        load_or_compute_baseline_scores(baseline_path, contexts_2, model_revision="rev2", run_signature="full", llm_adapter=_fake_llm_adapter(), tokenizer=_fake_tokenizer(), sampling_params=None)


# --- Figure-2 metrics + Spectral Discordance ----------------------------------------------------


def _fake_record(perturbation_id, capability, sigma, delta):
    from neural_thickets_repro.thicket.schema import ExperimentResultRecord

    return ExperimentResultRecord(
        experiment_id="e", perturbation_id=perturbation_id, model_family="x", model_scale="3B", model_revision="r",
        perturbation_mode="global_gaussian_upstream", anatomy_region=None, radius=None, sigma=sigma, seed=1,
        parameter_mask_hash="h", capability=capability, dataset_role="map", subset_hash="sh",
        base_score=0.5, perturbed_score=0.5 + delta, delta=delta, parser_failure_rate=0.0,
        per_example_result_path=None, per_example_result_hash="rh", runtime_metadata={},
    )


def test_compute_figure2_summary_groups_by_capability_and_sigma():
    records = [
        _fake_record("p0", "visual_grounding", 0.001, 0.1),
        _fake_record("p1", "visual_grounding", 0.001, -0.1),
        _fake_record("p0", "spatial_reasoning", 0.001, 0.2),
        _fake_record("p2", "visual_grounding", 0.01, 0.05),
    ]
    summary = compute_figure2_summary(records)
    assert set(summary["visual_grounding"]) == {"0.001", "0.01"}
    vg_001 = summary["visual_grounding"]["0.001"]
    assert vg_001["n"] == 2
    assert vg_001["mean"] == pytest.approx(0.0)
    assert vg_001["probability_of_improvement"] == pytest.approx(0.5)
    assert set(vg_001["solution_density"]) == {0.0, 0.02, 0.05}


def test_build_delta_matrix_aligns_perturbations_across_capabilities():
    records = [
        _fake_record("p0", "visual_grounding", 0.001, 0.1),
        _fake_record("p0", "spatial_reasoning", 0.001, -0.2),
        _fake_record("p1", "visual_grounding", 0.001, 0.3),
        _fake_record("p1", "spatial_reasoning", 0.001, 0.4),
    ]
    pids, caps, matrix = build_delta_matrix(records)
    assert pids == ("p0", "p1")
    assert caps == ("spatial_reasoning", "visual_grounding")
    assert matrix.shape == (2, 2)


def test_build_delta_matrix_raises_on_missing_cell():
    records = [_fake_record("p0", "visual_grounding", 0.001, 0.1)]
    records.append(_fake_record("p1", "visual_grounding", 0.001, 0.2))
    records.append(_fake_record("p1", "spatial_reasoning", 0.001, 0.2))
    with pytest.raises(ValueError):
        build_delta_matrix(records)


def test_compute_diversity_summary_reports_spectral_discordance():
    records = []
    rng = np.random.default_rng(0)
    for i in range(20):
        delta = float(rng.normal())
        records.append(_fake_record(f"p{i}", "visual_grounding", 0.001, delta))
        records.append(_fake_record(f"p{i}", "spatial_reasoning", 0.001, delta))
    summary = compute_diversity_summary(records)
    assert summary["spectral_discordance"] == pytest.approx(0.0, abs=1e-6)
    assert len(summary["perturbation_ids"]) == 20


# --- paper summary: refuses incomplete runs (spec section 6) ------------------------------------


def test_build_run_manifest_summary_complete_run():
    checkpoint = _checkpoint(expected_unique_perturbations=2, expected_result_rows=6)
    records = [_fake_record(f"p{i}", cap, 0.001, 0.1) for i in range(2) for cap in PILOT_CAPABILITIES]
    manifest = build_run_manifest_summary(checkpoint, records)
    assert manifest["run_complete"] is True
    assert manifest["actual_unique_perturbations"] == 2
    assert manifest["actual_result_rows"] == 6
    assert manifest["restoration_mode"] == "fixed_base"


def test_build_run_manifest_summary_incomplete_run():
    checkpoint = _checkpoint(expected_unique_perturbations=2, expected_result_rows=6)
    records = [_fake_record("p0", cap, 0.001, 0.1) for cap in PILOT_CAPABILITIES]  # only 1/2 perturbations
    manifest = build_run_manifest_summary(checkpoint, records)
    assert manifest["run_complete"] is False
    assert manifest["actual_unique_perturbations"] == 1
    assert manifest["actual_result_rows"] == 3


def test_write_paper_summary_refuses_incomplete_run(tmp_path):
    checkpoint = _checkpoint(expected_unique_perturbations=2, expected_result_rows=6)
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(checkpoint.to_dict(), indent=2))
    records = [_fake_record("p0", cap, 0.001, 0.1) for cap in PILOT_CAPABILITIES]
    with (tmp_path / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    with pytest.raises(IncompleteRunError):
        write_paper_summary(tmp_path)

    # run_manifest.json is still written even though the summary was refused.
    assert (tmp_path / "run_manifest.json").exists()
    assert not (tmp_path / "figure2_summary.json").exists()


def test_write_paper_summary_succeeds_for_a_complete_run(tmp_path):
    checkpoint = _checkpoint(subset_hashes={"visual_grounding": "h1", "ocr_text_recognition_grounded": "h2", "spatial_reasoning": "h3"}, expected_unique_perturbations=2, expected_result_rows=6)
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(checkpoint.to_dict(), indent=2))
    records = [_fake_record(f"p{i}", cap, 0.001, 0.1) for i in range(2) for cap in PILOT_CAPABILITIES]
    with (tmp_path / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    manifest = write_paper_summary(tmp_path)

    assert manifest["run_complete"] is True
    assert (tmp_path / "figure2_summary.json").exists()
    assert (tmp_path / "diversity_summary.json").exists()


def test_write_paper_summary_raises_if_never_started(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_paper_summary(tmp_path)


# --- Stage-6 engine config -------------------------------------------------------------------


def test_build_stage6_engine_config_matches_the_required_settings():
    config = build_stage6_engine_config()
    assert config["max_model_len"] == STAGE6_MAX_MODEL_LEN == 4096
    assert config["gpu_memory_utilization"] == STAGE6_GPU_MEMORY_UTILIZATION == 0.60
    assert config["restoration_mode"] == "fixed_base"
    assert config["perturbation_semantics"] == "global_gaussian_upstream"
    assert config["base_snapshot_mode"] == BASE_SNAPSHOT_MODE == "store_base_weights"


def test_format_runtime_compatibility_diagnostic_includes_every_required_field():
    from neural_thickets_repro.run_global_visual_thicket_pilot import format_runtime_compatibility_diagnostic

    text = format_runtime_compatibility_diagnostic(
        {"model_name": "Qwen/Qwen2.5-VL-3B-Instruct", "requested_revision": "rev1", "resolved_snapshot_path": "/fake/snap"},
        worker_extension_cls="utils.worker_extn.WorkerExtension", vllm_version="0.11.0", engine_mode="V1 (default)",
        engine_config=build_stage6_engine_config(),
    )
    for expected in (
        "vllm_version", "engine_mode", "worker_extension_cls", "model_name", "requested_revision",
        "resolved_snapshot_path", "max_model_len: 4096", "gpu_memory_utilization: 0.6", "restoration_mode: fixed_base",
    ):
        assert expected in text


def test_format_base_snapshot_confirmation_reports_stored_true():
    from neural_thickets_repro.run_global_visual_thicket_pilot import format_base_snapshot_confirmation

    text = format_base_snapshot_confirmation(0.60, "store_base_weights")
    assert "gpu_memory_utilization: 0.6" in text
    assert "base_snapshot_mode: store_base_weights" in text
    assert "base_snapshot_stored: True" in text


def test_get_vllm_version_never_raises_when_vllm_missing(monkeypatch):
    from neural_thickets_repro.run_global_visual_thicket_pilot import get_vllm_version

    monkeypatch.setitem(sys.modules, "vllm", None)  # simulates ImportError on `import vllm`
    assert isinstance(get_vllm_version(), str)


def test_detect_vllm_engine_mode_reports_v0_when_env_var_set(monkeypatch):
    from neural_thickets_repro.run_global_visual_thicket_pilot import detect_vllm_engine_mode

    monkeypatch.setenv("VLLM_USE_V1", "0")
    assert "V0" in detect_vllm_engine_mode()


def test_detect_vllm_engine_mode_reports_v1_by_default(monkeypatch):
    from neural_thickets_repro.run_global_visual_thicket_pilot import detect_vllm_engine_mode

    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    assert "V1" in detect_vllm_engine_mode()


# --- model-revision resolution ------------------------------------------------------------------


def test_resolve_and_report_model_snapshot_calls_vlm_adapter_resolver(monkeypatch):
    import neural_thickets_repro.vlm_adapter as vlm_adapter
    from neural_thickets_repro.run_global_visual_thicket_pilot import resolve_and_report_model_snapshot

    calls = []

    def _fake_resolve(model_name, revision):
        calls.append((model_name, revision))
        return "/fake/local/snapshot/path"

    monkeypatch.setattr(vlm_adapter, "resolve_model_snapshot", _fake_resolve)

    info = resolve_and_report_model_snapshot("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3")

    assert calls == [("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3")]
    assert info == {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "requested_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "resolved_snapshot_path": "/fake/local/snapshot/path",
    }


# --- launch_stage6_engine (still uses 4096 / no store_base_weights inside the launcher) --------


class _FakeStage6ActorHandle:
    def __init__(self, engine_kwargs):
        self.engine_kwargs = engine_kwargs
        self.collective_rpc_calls = []
        self.collective_rpc = SimpleNamespace(remote=self._collective_rpc_remote)

    def _collective_rpc_remote(self, method, args=()):
        self.collective_rpc_calls.append((method, args))
        return "fake_result"


def _install_fake_ray_and_core_engine(monkeypatch, remote_actor_calls):
    class _FakeRandOptNcclLLM:
        pass

    def _ray_remote(**decorator_kwargs):
        def _decorator(cls):
            class _FakeActorClass:
                @staticmethod
                def remote(**engine_kwargs):
                    remote_actor_calls.append(engine_kwargs)
                    return _FakeStage6ActorHandle(engine_kwargs)

            return _FakeActorClass

        return _decorator

    fake_ray_module = types.ModuleType("ray")
    fake_ray_module.remote = _ray_remote
    fake_ray_module.get = lambda x, timeout=None: x

    fake_pg_module = types.ModuleType("ray.util.placement_group")
    fake_pg_module.placement_group = lambda bundles, lifetime=None: SimpleNamespace(ready=lambda: "ready_marker")

    fake_strategies_module = types.ModuleType("ray.util.scheduling_strategies")
    fake_strategies_module.PlacementGroupSchedulingStrategy = lambda **kwargs: SimpleNamespace(**kwargs)

    fake_ray_util_module = types.ModuleType("ray.util")

    fake_core_engine_module = types.ModuleType("core.engine")
    fake_core_engine_module.RandOptNcclLLM = _FakeRandOptNcclLLM
    fake_core_module = types.ModuleType("core")
    fake_core_module.engine = fake_core_engine_module

    monkeypatch.setitem(sys.modules, "ray", fake_ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", fake_ray_util_module)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", fake_pg_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", fake_strategies_module)
    monkeypatch.setitem(sys.modules, "core", fake_core_module)
    monkeypatch.setitem(sys.modules, "core.engine", fake_core_engine_module)


def test_launch_stage6_engine_passes_max_model_len_4096(monkeypatch):
    from neural_thickets_repro.run_global_visual_thicket_pilot import launch_stage6_engine

    remote_actor_calls = []
    _install_fake_ray_and_core_engine(monkeypatch, remote_actor_calls)
    launch_stage6_engine("/fake/snapshot/path")
    assert remote_actor_calls[0]["max_model_len"] == 4096


def test_launch_stage6_engine_defaults_to_gpu_memory_utilization_060(monkeypatch):
    from neural_thickets_repro.run_global_visual_thicket_pilot import launch_stage6_engine

    remote_actor_calls = []
    _install_fake_ray_and_core_engine(monkeypatch, remote_actor_calls)
    launch_stage6_engine("/fake/snapshot/path")
    assert remote_actor_calls[0]["gpu_memory_utilization"] == 0.60


def test_launch_stage6_engine_never_calls_store_base_weights_itself(monkeypatch):
    from neural_thickets_repro.run_global_visual_thicket_pilot import launch_stage6_engine

    remote_actor_calls = []
    _install_fake_ray_and_core_engine(monkeypatch, remote_actor_calls)
    engines, pgs = launch_stage6_engine("/fake/snapshot/path")
    engine = engines[0]
    assert engine.collective_rpc_calls == []
    assert all(call[0] != WORKER_STORE_BASE_METHOD for call in engine.collective_rpc_calls)


def test_launch_stage6_engine_loads_exactly_one_engine(monkeypatch):
    from neural_thickets_repro.run_global_visual_thicket_pilot import launch_stage6_engine

    remote_actor_calls = []
    _install_fake_ray_and_core_engine(monkeypatch, remote_actor_calls)
    engines, pgs = launch_stage6_engine("/fake/snapshot/path")
    assert len(engines) == 1
    assert len(pgs) == 1
    assert len(remote_actor_calls) == 1


# --- dry-run CLI ---------------------------------------------------------------------------------


def test_main_dry_run_prints_plan_and_returns_zero(capsys):
    from neural_thickets_repro.run_global_visual_thicket_pilot import main

    rc = main(["--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "total_model_example_evaluations" in captured.out


def test_main_dry_run_with_smoke_overrides(capsys):
    from neural_thickets_repro.run_global_visual_thicket_pilot import main

    rc = main(["--dry-run", "--perturbations-per-sigma", "2", "--subset-size", "5"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "perturbations_per_sigma: 2" in captured.out
    assert "examples_per_capability (D_map): 5" in captured.out


# --- full main() GPU-path wiring: store_base_weights once, reset before/after, never restore ---


def test_main_real_run_calls_store_base_weights_exactly_once_and_resets_around_every_candidate(tmp_path, monkeypatch):
    import neural_thickets_repro.run_global_visual_thicket_pilot as pilot_mod
    import neural_thickets_repro.vlm_adapter as vlm_adapter

    fake_engine = _fake_engine()
    launch_calls = []

    def _fake_launch_stage6_engine(model_path, **kwargs):
        launch_calls.append((model_path, kwargs))
        return [fake_engine], ["fake_pg"]

    monkeypatch.setattr(pilot_mod, "launch_stage6_engine", _fake_launch_stage6_engine)

    fake_core_engine_module = types.ModuleType("core.engine")
    fake_core_engine_module.cleanup_engines = lambda engines, pgs: None
    fake_core_module = types.ModuleType("core")
    fake_core_module.engine = fake_core_engine_module
    monkeypatch.setitem(sys.modules, "core", fake_core_module)
    monkeypatch.setitem(sys.modules, "core.engine", fake_core_engine_module)

    fake_ray_module = types.ModuleType("ray")
    fake_ray_module.get = _identity_ray_get
    monkeypatch.setitem(sys.modules, "ray", fake_ray_module)

    fake_vllm_module = types.ModuleType("vllm")
    fake_vllm_module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module)

    fake_transformers_module = types.ModuleType("transformers")
    fake_transformers_module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda path: _fake_tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers_module)

    monkeypatch.setattr(vlm_adapter, "resolve_model_snapshot", lambda name, rev: "/fake/snapshot/path")
    monkeypatch.setattr(vlm_adapter, "bootstrap_ray", lambda root: True)
    monkeypatch.setattr(vlm_adapter, "verify_workers_can_import_external_root", lambda root: None)

    def _fake_build_d_map_context(benchmark, cfg, capability, n, seed, subset_ids_dir):
        from neural_thickets_repro.thicket.data_roles import partition_data_roles

        examples = [Example(example_id=f"{capability}_{j}", image=None, prompt_input={}, target=0) for j in range(3)]
        partition = partition_data_roles([e.example_id for e in examples], sizes={"map": 3}, seed=seed)
        return CapabilityContext(capability=capability, benchmark=_RecordingBenchmark(capability), examples=examples, partition=partition, subset_hash=partition.manifest_hash)

    monkeypatch.setattr(pilot_mod, "build_d_map_context", _fake_build_d_map_context)

    raw_config_path = tmp_path / "pilot_config.yaml"
    import yaml

    raw_config_path.write_text(yaml.safe_dump(_raw_config(outputs={"root": str(tmp_path / "out")})))

    rc = pilot_mod.main(["--config", str(raw_config_path), "--output-dir", str(tmp_path / "out"), "--perturbations-per-sigma", "1", "--subset-size", "3"])

    assert rc == 0
    assert len(launch_calls) == 1  # exactly one engine launch -- one model load
    assert launch_calls[0][1]["max_model_len"] == 4096
    assert launch_calls[0][1]["gpu_memory_utilization"] == 0.60

    store_base_calls = [c for c in fake_engine.collective_rpc_calls if c[0] == WORKER_STORE_BASE_METHOD]
    assert len(store_base_calls) == 1  # store_base_weights called EXACTLY ONCE

    restore_calls = [c for c in fake_engine.collective_rpc_calls if c[0] == WORKER_RESTORE_METHOD]
    assert restore_calls == []  # restore_self_weights NEVER called

    reset_calls = [c for c in fake_engine.collective_rpc_calls if c[0] == WORKER_RESET_TO_BASE_METHOD]
    perturb_calls = [c for c in fake_engine.collective_rpc_calls if c[0] == WORKER_PERTURB_METHOD]
    assert len(reset_calls) == 2 * len(perturb_calls)  # reset before AND after every perturbation

    verify_calls = [c for c in fake_engine.collective_rpc_calls if c[0] == "verify_exact_fixed_base_restoration_rpc"]
    assert len(verify_calls) == len(perturb_calls)  # restoration verification runs after every candidate

    # run_signature ("smoke_p1_n3", since perturbations_per_sigma/subset_size were overridden)
    # is always appended to --output-dir -- see stale-output-safety fix.
    run_output_dir = tmp_path / "out" / "smoke_p1_n3"
    assert (run_output_dir / "model_resolution.json").exists()
    assert (run_output_dir / "engine_config.json").exists()
    assert (run_output_dir / "checkpoint_manifest.json").exists()
    assert (run_output_dir / "baseline_scores.json").exists()
    assert (run_output_dir / "results.jsonl").exists()
    assert (run_output_dir / "run_manifest.json").exists()
    assert (run_output_dir / "figure2_summary.json").exists()
    assert (run_output_dir / "diversity_summary.json").exists()

    engine_config_on_disk = json.loads((run_output_dir / "engine_config.json").read_text())
    assert engine_config_on_disk["restoration_mode"] == "fixed_base"
    assert engine_config_on_disk["max_model_len"] == 4096
    assert engine_config_on_disk["gpu_memory_utilization"] == 0.60
