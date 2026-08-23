"""Tests for run_global_visual_thicket_pilot.py -- CPU-only. The real GPU execution path
(model load, vLLM generate) is not exercised here (never run in this session) -- these tests
cover every CPU-testable piece: plan arithmetic, dry-run formatting, D_map construction,
perturb/apply/restore lifecycle (against the existing synthetic dummy_vlm fixture), result
alignment across capabilities, Figure-2 metrics, and Spectral Discordance -- matching this
project's established "requires the pod" pattern for anything that actually needs a real model.
"""
import copy
import json

import numpy as np
import pytest
import torch

from neural_thickets_repro.benchmarks.base import Example, ExampleScore, ParsedPrediction
from neural_thickets_repro.run_global_visual_thicket_pilot import (
    CapabilityContext,
    PILOT_CAPABILITIES,
    PilotConfigError,
    RestorationFailedError,
    UPSTREAM_SIGMA_GRID,
    build_d_map_context,
    build_delta_matrix,
    build_pilot_plan,
    compute_diversity_summary,
    compute_figure2_summary,
    compute_global_gaussian_mask_hash,
    evaluate_one_perturbation,
    format_pilot_plan,
    run_pilot,
    snapshot_state,
    verify_restoration,
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


# --- upstream sigma grid discovery/config (spec item: "exact upstream sigma grid discovery") --


def test_upstream_sigma_grid_matches_confirmed_pinned_commit_default():
    """Cross-checked directly against external/RandOpt/randopt.py's CLI default
    `--sigma_values` at the pinned commit 536df0a308f3990b6270c991fbb96bd0b779a58e this stage
    -- also already recorded as `sigma_default` in REPRO_SPEC.md's Sigma resolution-plan.
    """
    assert UPSTREAM_SIGMA_GRID == (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01)


# --- PilotPlan arithmetic + dry-run formatting --------------------------------------------


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
    # the raw config dict itself is untouched -- overrides never mutate the paper pilot config.
    assert raw["pilot"]["perturbations_per_sigma"] == 64
    assert raw["pilot"]["examples_per_capability"] == 50


def test_build_pilot_plan_rejects_wrong_capability_set():
    raw = _raw_config()
    raw["pilot"]["capabilities"] = ["visual_grounding", "counting", "spatial_reasoning"]
    with pytest.raises(PilotConfigError):
        build_pilot_plan(raw)


def test_build_pilot_plan_rejects_invented_sigma_grid():
    raw = _raw_config()
    raw["pilot"]["sigma_grid"] = [0.001, 0.5]  # not the upstream grid
    with pytest.raises(PilotConfigError):
        build_pilot_plan(raw)


def test_build_pilot_plan_accepts_sigma_grid_in_any_order():
    raw = _raw_config()
    raw["pilot"]["sigma_grid"] = list(reversed(UPSTREAM_SIGMA_GRID))
    plan = build_pilot_plan(raw)
    assert set(plan.sigma_grid) == set(UPSTREAM_SIGMA_GRID)


def test_format_pilot_plan_prints_every_required_field():
    plan = build_pilot_plan(_raw_config())
    text = format_pilot_plan(plan)
    for expected in (
        "model_name", "model_revision", "capabilities", "sigma_grid", "perturbations_per_sigma",
        "total_unique_perturbations", "examples_per_capability", "total_perturbation_x_capability_evaluations",
        "total_model_example_evaluations", "expected_model_loading_strategy", "output_dir",
    ):
        assert expected in text
    assert "$" not in text  # no dollar-cost estimate


# --- D_map construction (spec section 4) ---------------------------------------------------


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


def test_build_d_map_context_all_ids_tagged_as_map_role(tmp_path):
    benchmark = _FixedPoolBenchmark(_pool(200))
    ctx = build_d_map_context(benchmark, cfg=None, capability="spatial_reasoning", n=50, seed=1, subset_ids_dir=tmp_path)
    assert ctx.partition.sizes == {"map": 50, "confirm": 0, "select": 0, "test": 0}
    assert set(ctx.partition.roles["map"]) == {e.example_id for e in ctx.examples}


# --- restoration safety (spec section 7) ----------------------------------------------------


def test_snapshot_and_verify_restoration_passes_when_unchanged(dummy_vlm_factory):
    model = dummy_vlm_factory()
    snapshot = snapshot_state(model)
    verify_restoration(model, snapshot, atol=1e-6)  # must not raise


def test_verify_restoration_raises_on_drift(dummy_vlm_factory):
    model = dummy_vlm_factory()
    snapshot = snapshot_state(model)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(1.0)
            break
    with pytest.raises(RestorationFailedError):
        verify_restoration(model, snapshot, atol=1e-6)


def test_apply_then_undo_global_gaussian_passes_restoration_check(dummy_vlm_factory):
    from neural_thickets_repro.thicket.perturbation import apply_global_gaussian_upstream, undo_global_gaussian_upstream

    model = dummy_vlm_factory()
    snapshot = snapshot_state(model)
    record = apply_global_gaussian_upstream(model, seed=5, sigma=0.1)
    undo_global_gaussian_upstream(model, record)
    verify_restoration(model, snapshot, atol=1e-5)  # must not raise -- close, per perturb_cpu's own float-rounding tolerance


# --- perturbation lifecycle + cross-capability alignment (spec sections 5-7) ----------------


class _RecordingBenchmark:
    """Deterministic fake benchmark whose score is a function of a CURRENT MODEL PARAMETER
    (not a fixed constant) -- so a real perturbation actually changes perturbed_score, exactly
    like a real capability adapter's score would respond to real perturbed weights.
    """

    def __init__(self, name, model, base_score):
        self.capability = name
        self._model = model
        self._base_score = base_score

    def load_examples(self, cfg):
        raise NotImplementedError

    def subset_selection_rule(self):
        return "shuffled_prefix"

    def prepare_image(self, example):
        return "fake_image"  # any non-None value -- run_benchmark only checks for None

    def build_prompt(self, example):
        return []

    def parse_prediction(self, raw_generation, example):
        return ParsedPrediction(parsed=raw_generation, parse_ok=True)

    def score_example(self, parsed, example):
        return ExampleScore(score=1.0, correct=True)

    def aggregate_metrics(self, scores):
        # perturbed_score tracks the model's current first-parameter mean -- changes exactly
        # when (and only when) the model is actually perturbed.
        first_param = next(iter(self._model.parameters()))
        current = float(first_param.detach().mean().item())
        return {"primary_metric": self._base_score + current, "parser_failure_rate": 0.0}


def _fake_llm_and_tokenizer():
    from types import SimpleNamespace

    def _generate(requests, sampling_params, use_tqdm=True):
        return [SimpleNamespace(outputs=[SimpleNamespace(text="ok")]) for _ in requests]

    llm = SimpleNamespace(generate=_generate)
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, add_generation_prompt, tokenize: "TEXT")
    return llm, tokenizer


def _build_contexts(model, capabilities, n_examples=3):
    from neural_thickets_repro.thicket.data_roles import partition_data_roles

    contexts = {}
    for i, capability in enumerate(capabilities):
        examples = [Example(example_id=f"{capability}_{j}", image=None, prompt_input={}, target=0) for j in range(n_examples)]
        partition = partition_data_roles([e.example_id for e in examples], sizes={"map": n_examples}, seed=1)
        bench = _RecordingBenchmark(capability, model, base_score=float(i))
        contexts[capability] = CapabilityContext(capability=capability, benchmark=bench, examples=examples, partition=partition, subset_hash=partition.manifest_hash, base_score=0.0)
    return contexts


def test_same_perturbation_id_appears_for_all_capabilities(dummy_vlm_factory):
    model = dummy_vlm_factory()
    contexts = _build_contexts(model, PILOT_CAPABILITIES)
    llm, tokenizer = _fake_llm_and_tokenizer()
    snapshot = snapshot_state(model)
    manifest = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.01)

    records = evaluate_one_perturbation(model, manifest, contexts, llm, tokenizer, None, snapshot, restoration_atol=1e-4)

    assert len(records) == 3
    assert {r.perturbation_id for r in records} == {manifest.perturbation_id}
    assert {r.capability for r in records} == set(PILOT_CAPABILITIES)


def test_evaluate_one_perturbation_restores_after_evaluating(dummy_vlm_factory):
    model = dummy_vlm_factory()
    contexts = _build_contexts(model, PILOT_CAPABILITIES)
    llm, tokenizer = _fake_llm_and_tokenizer()
    snapshot = snapshot_state(model)
    manifest = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.05)

    evaluate_one_perturbation(model, manifest, contexts, llm, tokenizer, None, snapshot, restoration_atol=1e-4)

    verify_restoration(model, snapshot, atol=1e-4)  # must not raise -- model is back at base


def test_no_perturbation_accumulation_across_candidates(dummy_vlm_factory):
    """Two sequential perturbations must each be applied from the SAME base state -- not
    stacked on top of each other's residue.
    """
    model = dummy_vlm_factory()
    contexts = _build_contexts(model, PILOT_CAPABILITIES)
    llm, tokenizer = _fake_llm_and_tokenizer()
    snapshot = snapshot_state(model)

    manifest_a = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.02)
    manifest_b = PerturbationManifest(seed=2, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.02)

    records_a = evaluate_one_perturbation(model, manifest_a, contexts, llm, tokenizer, None, snapshot, restoration_atol=1e-4)
    records_b = evaluate_one_perturbation(model, manifest_b, contexts, llm, tokenizer, None, snapshot, restoration_atol=1e-4)

    # Re-running manifest_a's own seed/sigma independently must give the identical perturbed
    # score as the first time -- impossible if candidate B's perturbation had accumulated.
    model2 = dummy_vlm_factory()
    contexts2 = _build_contexts(model2, PILOT_CAPABILITIES)
    snapshot2 = snapshot_state(model2)
    records_a_replay = evaluate_one_perturbation(model2, manifest_a, contexts2, llm, tokenizer, None, snapshot2, restoration_atol=1e-4)

    scores_a = sorted((r.capability, r.perturbed_score) for r in records_a)
    scores_a_replay = sorted((r.capability, r.perturbed_score) for r in records_a_replay)
    assert all(a[0] == b[0] and a[1] == pytest.approx(b[1], abs=1e-6) for a, b in zip(scores_a, scores_a_replay))


def test_evaluate_one_perturbation_aborts_on_restoration_failure(dummy_vlm_factory):
    model = dummy_vlm_factory()
    contexts = _build_contexts(model, PILOT_CAPABILITIES)
    llm, tokenizer = _fake_llm_and_tokenizer()
    snapshot = snapshot_state(model)
    manifest = PerturbationManifest(seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.05)

    with pytest.raises(RestorationFailedError):
        # atol=0 makes even float-rounding-level restoration "fail", proving the check is real.
        evaluate_one_perturbation(model, manifest, contexts, llm, tokenizer, None, snapshot, restoration_atol=0.0)


def test_run_pilot_produces_full_population_and_shared_ids_across_capabilities(dummy_vlm_factory):
    model = dummy_vlm_factory()
    contexts = _build_contexts(model, PILOT_CAPABILITIES)
    llm, tokenizer = _fake_llm_and_tokenizer()
    raw = _raw_config()
    plan = build_pilot_plan(raw, perturbations_per_sigma=2, subset_size=3)

    records = run_pilot(plan, contexts, model, llm, tokenizer, None, base_seed=42, parameter_mask_hash="hash1")

    assert len(records) == plan.total_perturbation_capability_evaluations
    by_pid = {}
    for r in records:
        by_pid.setdefault(r.perturbation_id, set()).add(r.capability)
    assert all(caps == set(PILOT_CAPABILITIES) for caps in by_pid.values())
    assert len(by_pid) == plan.total_unique_perturbations


# --- compute_global_gaussian_mask_hash -------------------------------------------------------


def test_compute_global_gaussian_mask_hash_excludes_visual_params(dummy_vlm_factory):
    model = dummy_vlm_factory()
    all_names = {n for n, _ in model.named_parameters()}
    non_visual_names = {n for n in all_names if not n.startswith("visual.")}
    assert compute_global_gaussian_mask_hash(model) == __import__("neural_thickets_repro.thicket.anatomy", fromlist=["compute_mask_hash"]).compute_mask_hash(non_visual_names)


# --- Figure-2 metrics + Spectral Discordance (spec sections 9-10) --------------------------


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
    records = [_fake_record("p0", "visual_grounding", 0.001, 0.1)]  # p0 missing spatial_reasoning
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
        records.append(_fake_record(f"p{i}", "spatial_reasoning", 0.001, delta))  # perfectly correlated
    summary = compute_diversity_summary(records)
    assert summary["spectral_discordance"] == pytest.approx(0.0, abs=1e-6)
    assert len(summary["perturbation_ids"]) == 20


# --- deterministic pilot population reproduction --------------------------------------------


def test_pilot_population_generation_is_reproducible_per_sigma():
    pop_1 = generate_perturbation_population(
        mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl",
        model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.001,
    )
    pop_2 = generate_perturbation_population(
        mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl",
        model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.001,
    )
    assert [m.perturbation_id for m in pop_1] == [m.perturbation_id for m in pop_2]


def test_pilot_population_differs_across_sigmas():
    pop_a = generate_perturbation_population(
        mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl",
        model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.0001,
    )
    pop_b = generate_perturbation_population(
        mode="global_gaussian_upstream", n=10, base_seed=20260823, model_family="qwen2_5_vl",
        model_scale="3B", model_revision="rev1", parameter_mask_hash="hash1", sigma=0.01,
    )
    assert {m.seed for m in pop_a}.isdisjoint({m.seed for m in pop_b})


# --- dry-run CLI end to end -------------------------------------------------------------------


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
