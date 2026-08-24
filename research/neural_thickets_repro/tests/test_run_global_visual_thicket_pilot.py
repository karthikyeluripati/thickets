"""Tests for run_global_visual_thicket_pilot.py -- CPU-only. The real GPU/Ray/vLLM engine is
never launched for real; RPC dispatch is tested against a FAKE Ray-actor-shaped engine (see
_FakeRayEngine below) whose `.collective_rpc.remote(...)`/`.generate.remote(...)` return raw
values directly (paired with `ray_get=lambda x: x` injected into every call), so no real Ray
cluster is needed while still exercising the exact `.remote(...)`-then-`ray.get(...)` call
shape a real Ray actor handle presents. This directly tests the Stage-6 repair-pass fix: the
runner must dispatch every weight-touching operation through collective_rpc, NEVER through a
frontend `llm_engine.model_executor` attribute path (which does not exist under vLLM V1).
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from neural_thickets_repro.benchmarks.base import Example, ExampleScore, ParsedPrediction
from neural_thickets_repro.run_global_visual_thicket_pilot import (
    CapabilityContext,
    PILOT_CAPABILITIES,
    PilotConfigError,
    RestorationFailedError,
    UPSTREAM_SIGMA_GRID,
    WORKER_PERTURB_METHOD,
    WORKER_RESTORE_METHOD,
    build_d_map_context,
    build_delta_matrix,
    build_pilot_plan,
    compute_diversity_summary,
    compute_figure2_summary,
    compute_mask_info_via_rpc,
    compute_restoration_fingerprint_via_rpc,
    evaluate_one_perturbation_rpc,
    format_pilot_plan,
    perturb_via_rpc,
    restore_via_rpc,
    run_pilot_rpc,
    verify_restoration_via_rpc,
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
# Fake Ray-actor-shaped engine
# =============================================================================================


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def float(self):
        return self

    def norm(self):
        return _FakeScalar(sum(v * v for v in self.values) ** 0.5)


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def _noise_for(name, seed, length):
    """Deterministic pseudo-noise, purely a function of (name, seed) -- not real Gaussian
    math (that's already covered by thicket.perturbation's own CPU tests); only needs to be
    reproducible and exactly invertible by an equal-and-opposite call for these lifecycle tests.
    """
    return [((hash((name, seed, i)) % 1000) / 500.0 - 1.0) for i in range(length)]


class _FakeRayEngine:
    """Duck-types a Ray actor handle wrapping a vLLM engine with
    worker_extension_cls=utils.worker_extn.WorkerExtension already attached: `.collective_rpc
    .remote(method_or_callable, args)` and `.generate.remote(requests, sampling_params,
    use_tqdm)` both return their result directly (no ObjectRef) -- callers must pass
    `ray_get=lambda x: x`. Deliberately has NO `llm_engine`/`model_executor`/`driver_worker`
    attribute at all, so any code path that tries the old broken frontend access fails loudly
    with AttributeError rather than silently working here.
    """

    def __init__(self, param_shapes, visual_prefixes=("visual.",)):
        self._values = {name: [0.0] * n for name, n in param_shapes.items()}
        self._visual_prefixes = visual_prefixes
        self.collective_rpc_calls = []
        self.generate_call_count = 0
        self.collective_rpc = SimpleNamespace(remote=self._collective_rpc)
        self.generate = SimpleNamespace(remote=self._generate)

    def _should_perturb(self, name):
        return not name.startswith(self._visual_prefixes)

    def _tensors(self):
        return [(name, _FakeTensor(vals)) for name, vals in self._values.items()]

    def _apply_noise(self, seed, sigma, negate):
        sign = -1.0 if negate else 1.0
        for name, vals in self._values.items():
            if self._should_perturb(name):
                noise = _noise_for(name, seed, len(vals))
                self._values[name] = [v + sign * sigma * n for v, n in zip(vals, noise)]

    def _collective_rpc(self, method, args=()):
        label = method if isinstance(method, str) else getattr(method, "__name__", "callable")
        self.collective_rpc_calls.append((label, args))
        if method == WORKER_PERTURB_METHOD:
            seed, sigma, negate = args
            self._apply_noise(seed, sigma, negate)
            return [True]
        if method == WORKER_RESTORE_METHOD:
            seed, sigma, negate = args
            self._apply_noise(seed, sigma, not negate)
            return [True]
        if callable(method):
            worker_self = SimpleNamespace(
                model_runner=SimpleNamespace(model=SimpleNamespace(named_parameters=self._tensors)),
                _should_perturb=self._should_perturb,
            )
            return [method(worker_self, *args)]
        raise ValueError(f"unsupported collective_rpc method {method!r}")

    def _generate(self, requests, sampling_params, use_tqdm=True):
        self.generate_call_count += 1
        return [SimpleNamespace(outputs=[SimpleNamespace(text="ok")]) for _ in requests]


def _fake_engine():
    return _FakeRayEngine({"visual.blocks.0.weight": 2, "model.layers.0.weight": 3, "model.layers.1.weight": 2})


# --- upstream sigma grid discovery/config -----------------------------------------------------


def test_upstream_sigma_grid_matches_confirmed_pinned_commit_default():
    assert UPSTREAM_SIGMA_GRID == (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01)


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
    ):
        assert expected in text
    assert "$" not in text


def test_format_pilot_plan_documents_collective_rpc_not_frontend_access():
    plan = build_pilot_plan(_raw_config())
    text = format_pilot_plan(plan)
    assert "collective_rpc" in text
    assert "model_executor" not in text or "Zero frontend" in text  # only appears in the explicit "we don't do this" sentence
    assert "llm_engine.model_executor" not in text.replace("Zero frontend llm_engine.model_executor access anywhere in this path.", "")


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


# --- RPC dispatch: perturb / restore / mask-info / restoration-verification --------------------


def test_perturb_via_rpc_dispatches_upstream_method_name_and_args():
    engine = _fake_engine()
    perturb_via_rpc(engine, seed=1, sigma=0.01, ray_get=_identity_ray_get)
    assert engine.collective_rpc_calls == [(WORKER_PERTURB_METHOD, (1, 0.01, False))]


def test_restore_via_rpc_dispatches_upstream_method_name_and_args():
    engine = _fake_engine()
    restore_via_rpc(engine, seed=1, sigma=0.01, ray_get=_identity_ray_get)
    assert engine.collective_rpc_calls == [(WORKER_RESTORE_METHOD, (1, 0.01, False))]


def test_perturb_then_restore_returns_to_base_fingerprint():
    engine = _fake_engine()
    base_fingerprint = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)
    perturb_via_rpc(engine, seed=7, sigma=0.05, ray_get=_identity_ray_get)
    restore_via_rpc(engine, seed=7, sigma=0.05, ray_get=_identity_ray_get)
    after_fingerprint = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)
    assert after_fingerprint == pytest.approx(base_fingerprint)


def test_compute_mask_info_via_rpc_excludes_visual_and_computed_inside_worker():
    engine = _fake_engine()
    info = compute_mask_info_via_rpc(engine, ray_get=_identity_ray_get)
    assert info["param_count"] == 2  # model.layers.0/1, not visual.blocks.0
    assert "mask_hash" in info


def test_verify_restoration_via_rpc_passes_when_untouched():
    engine = _fake_engine()
    base = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)
    result = verify_restoration_via_rpc(engine, base, atol=1e-9, rtol=0.0, ray_get=_identity_ray_get)
    assert result["ok"] is True


def test_verify_restoration_via_rpc_fails_on_injected_drift():
    engine = _fake_engine()
    base = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)
    perturb_via_rpc(engine, seed=3, sigma=0.5, ray_get=_identity_ray_get)  # never restored
    result = verify_restoration_via_rpc(engine, base, atol=1e-9, rtol=0.0, ray_get=_identity_ray_get)
    assert result["ok"] is False
    assert result["n_failing"] > 0


def test_never_accesses_llm_engine_or_model_executor_attributes():
    """The fake engine has neither attribute at all -- proves the RPC helpers never touch them."""
    engine = _fake_engine()
    assert not hasattr(engine, "llm_engine")
    assert not hasattr(engine, "model_executor")
    assert not hasattr(engine, "driver_worker")
    perturb_via_rpc(engine, seed=1, sigma=0.01, ray_get=_identity_ray_get)
    restore_via_rpc(engine, seed=1, sigma=0.01, ray_get=_identity_ray_get)
    compute_mask_info_via_rpc(engine, ray_get=_identity_ray_get)
    compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)
    # no AttributeError raised anywhere above -- nothing needed those attributes.


# --- evaluate_one_perturbation_rpc / run_pilot_rpc: full lifecycle -----------------------------


class _RecordingBenchmark:
    capability_field_name = "capability"

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


def _build_contexts(capabilities, n_examples=3):
    from neural_thickets_repro.thicket.data_roles import partition_data_roles

    contexts = {}
    for capability in capabilities:
        examples = [Example(example_id=f"{capability}_{j}", image=None, prompt_input={}, target=0) for j in range(n_examples)]
        partition = partition_data_roles([e.example_id for e in examples], sizes={"map": n_examples}, seed=1)
        bench = _RecordingBenchmark(capability)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=bench, examples=examples, partition=partition, subset_hash=partition.manifest_hash, base_score=0.4)
    return contexts


def test_same_perturbation_id_appears_for_all_capabilities():
    engine = _fake_engine()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.01)
    base_fingerprint = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)

    records = evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, base_fingerprint, restoration_atol=1e-6, restoration_rtol=0.0, ray_get=_identity_ray_get)

    assert len(records) == 3
    assert {r.perturbation_id for r in records} == {manifest.perturbation_id}
    assert {r.capability for r in records} == set(PILOT_CAPABILITIES)


def test_evaluate_one_perturbation_rpc_dispatches_perturb_then_restore_in_order():
    engine = _fake_engine()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.01)
    base_fingerprint = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)
    engine.collective_rpc_calls.clear()

    evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, base_fingerprint, restoration_atol=1e-6, restoration_rtol=0.0, ray_get=_identity_ray_get)

    labels = [label for label, _ in engine.collective_rpc_calls]
    assert labels[0] == WORKER_PERTURB_METHOD
    assert labels[-2] == WORKER_RESTORE_METHOD  # last call is verify_restoration_rpc
    assert labels[-1] == "verify_restoration_rpc"
    assert engine.generate_call_count == 3  # one generate per capability


def test_evaluate_one_perturbation_rpc_aborts_on_restoration_failure():
    engine = _fake_engine()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    manifest = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.01)
    # a base fingerprint that does NOT match the engine's actual (post-restore) state -- forces
    # verify_restoration_via_rpc to report failure, proving the abort path is real.
    bogus_base_fingerprint = {"model.layers.0.weight": 999.0, "model.layers.1.weight": 999.0}

    with pytest.raises(RestorationFailedError):
        evaluate_one_perturbation_rpc(engine, manifest, contexts, _fake_tokenizer(), None, bogus_base_fingerprint, restoration_atol=1e-9, restoration_rtol=0.0, ray_get=_identity_ray_get)


def test_no_perturbation_accumulation_across_candidates():
    engine = _fake_engine()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    base_fingerprint = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)

    manifest_a = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.02)
    manifest_b = PerturbationManifest(seed=2, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.02)

    evaluate_one_perturbation_rpc(engine, manifest_a, contexts, _fake_tokenizer(), None, base_fingerprint, restoration_atol=1e-6, restoration_rtol=0.0, ray_get=_identity_ray_get)
    evaluate_one_perturbation_rpc(engine, manifest_b, contexts, _fake_tokenizer(), None, base_fingerprint, restoration_atol=1e-6, restoration_rtol=0.0, ray_get=_identity_ray_get)

    final_fingerprint = compute_restoration_fingerprint_via_rpc(engine, ray_get=_identity_ray_get)
    assert final_fingerprint == pytest.approx(base_fingerprint)


def test_run_pilot_rpc_produces_full_population_and_shared_ids_across_capabilities():
    engine = _fake_engine()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    raw = _raw_config()
    plan = build_pilot_plan(raw, perturbations_per_sigma=2, subset_size=3)

    records = run_pilot_rpc(plan, contexts, engine, _fake_tokenizer(), None, base_seed=42, parameter_mask_hash="hash1", ray_get=_identity_ray_get)

    assert len(records) == plan.total_perturbation_capability_evaluations
    by_pid = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, set()).add(r.capability)
    assert all(caps == set(PILOT_CAPABILITIES) for caps in by_pid.values())
    assert len(by_pid) == plan.total_unique_perturbations


def test_run_pilot_rpc_reuses_the_same_engine_for_every_perturbation():
    """'One model load' at this level of the code: run_pilot_rpc/evaluate_one_perturbation_rpc
    never construct a new engine -- they only ever call methods on the ONE `engine` object
    passed in. Verified here by counting the (large) number of generate calls against a single
    unreplaced fake engine instance.
    """
    engine = _fake_engine()
    contexts = _build_contexts(PILOT_CAPABILITIES)
    raw = _raw_config()
    plan = build_pilot_plan(raw, perturbations_per_sigma=2, subset_size=3)

    run_pilot_rpc(plan, contexts, engine, _fake_tokenizer(), None, base_seed=42, parameter_mask_hash="hash1", ray_get=_identity_ray_get)

    assert engine.generate_call_count == plan.total_perturbation_capability_evaluations


# --- deterministic pilot population reproduction -----------------------------------------------


def test_pilot_population_generation_is_reproducible_per_sigma():
    pop_1 = generate_perturbation_population(mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.001)
    pop_2 = generate_perturbation_population(mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.001)
    assert [m.perturbation_id for m in pop_1] == [m.perturbation_id for m in pop_2]


def test_pilot_population_differs_across_sigmas():
    pop_a = generate_perturbation_population(mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.0001)
    pop_b = generate_perturbation_population(mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.01)
    assert {m.seed for m in pop_a}.isdisjoint({m.seed for m in pop_b})


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


# --- runtime compatibility diagnostic ------------------------------------------------------------


def test_format_runtime_compatibility_diagnostic_includes_every_required_field():
    from neural_thickets_repro.run_global_visual_thicket_pilot import format_runtime_compatibility_diagnostic

    text = format_runtime_compatibility_diagnostic(
        {"model_name": "Qwen/Qwen2.5-VL-3B-Instruct", "requested_revision": "rev1", "resolved_snapshot_path": "/fake/snap"},
        worker_extension_cls="utils.worker_extn.WorkerExtension", vllm_version="0.11.0", engine_mode="V1 (default)",
    )
    for expected in ("vllm_version", "engine_mode", "worker_extension_cls", "model_name", "requested_revision", "resolved_snapshot_path"):
        assert expected in text


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


# --- model-revision resolution (Task 4) ----------------------------------------------------------


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


# --- full main() GPU-path wiring: one model load, zero frontend access -------------------------


def test_main_real_run_launches_engine_exactly_once_and_never_touches_frontend_attrs(tmp_path, monkeypatch):
    import neural_thickets_repro.run_global_visual_thicket_pilot as pilot_mod
    import neural_thickets_repro.vlm_adapter as vlm_adapter

    fake_engine = _fake_engine()
    launch_calls = []

    def _fake_launch_engines(num_engines, model_name, precision, tensor_parallel_size, multimodal):
        launch_calls.append((num_engines, model_name, precision, tensor_parallel_size, multimodal))
        return [fake_engine], ["fake_pg"]

    fake_core_engine_module = types.ModuleType("core.engine")
    fake_core_engine_module.launch_engines = _fake_launch_engines
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
    assert launch_calls[0][1] == "/fake/snapshot/path"  # the RESOLVED snapshot path, not the bare model name
    assert (tmp_path / "out" / "model_resolution.json").exists()
    assert (tmp_path / "out" / "results.jsonl").exists()
    assert (tmp_path / "out" / "figure2_summary.json").exists()
    assert (tmp_path / "out" / "diversity_summary.json").exists()
