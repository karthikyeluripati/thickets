"""Tests for run_stage11_coarse_anatomical_atlas_32b.py -- the narrow, TP=4-aware Stage-11 S2
(coarse anatomy) 32B runner. CPU-only, no GPU/ray/vllm import -- matches this project's established
convention (fake collective_rpc_all_workers dispatch, marker-exception continuation proofs).
"""
from __future__ import annotations

import inspect
import json

import pytest

import neural_thickets_repro.run_stage11_coarse_anatomical_atlas_32b as module
import neural_thickets_repro.run_stage11_coarse_anatomical_atlas_7b as anatomy_7b
import neural_thickets_repro.thicket.distributed_perturbation as dp_module
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_D_MAP_N, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_RADII, STAGE8_REGIONS
from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import STAGE8_AUTHORITATIVE_SUBSET_HASHES
from neural_thickets_repro.run_stage11_coarse_anatomical_atlas_7b import Stage11DirectionAssignment
from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest


def _identity_ray_get(x):
    return x


# =================================================================================================
# Properties 1-3, 14: plan counts + region set
# =================================================================================================


def test_smoke_plan_counts_9_perturbations_54_rows_270_evaluations():
    plan = module.build_stage11_32b_smoke_plan(model_name="Qwen/Qwen2.5-VL-32B-Instruct", model_revision="a" * 40, output_root="/tmp/out")
    assert plan.total_unique_perturbations == 9
    assert plan.total_perturbation_capability_evaluations == 54
    assert plan.total_perturbed_model_example_evaluations == 270
    assert plan.is_smoke is True


def test_smoke_plan_d_map_n_is_5():
    plan = module.build_stage11_32b_smoke_plan(model_name="Qwen/Qwen2.5-VL-32B-Instruct", model_revision="a" * 40, output_root="/tmp/out")
    assert plan.d_map_n == 5
    assert plan.n_directions_per_cell == 1


def test_full_plan_counts_576_perturbations_3456_rows_172800_evaluations():
    plan = module.build_stage11_32b_plan(model_name="Qwen/Qwen2.5-VL-32B-Instruct", model_revision="a" * 40, output_root="/tmp/out")
    assert plan.total_unique_perturbations == 576
    assert plan.total_perturbation_capability_evaluations == 3456
    assert plan.total_perturbed_model_example_evaluations == 172800
    assert plan.is_smoke is False
    assert plan.d_map_n == 50
    assert plan.n_directions_per_cell == 64


def test_plan_has_exactly_3_anatomy_regions():
    plan = module.build_stage11_32b_plan(model_name="Qwen/Qwen2.5-VL-32B-Instruct", model_revision="a" * 40, output_root="/tmp/out")
    assert len(plan.regions) == 3
    assert set(plan.regions) == {"vision", "multimodal_connector_or_merger", "language"}
    assert plan.regions == STAGE8_REGIONS


def test_dry_run_prints_frozen_smoke_totals(capsys):
    rc = module.main(["--smoke", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total_unique_perturbations=9" in out
    assert "total_perturbation_x_capability_evaluations=54" in out
    assert "total_perturbed_model_example_evaluations=270" in out


def test_dry_run_prints_frozen_full_totals(capsys):
    rc = module.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total_unique_perturbations=576" in out
    assert "total_perturbation_x_capability_evaluations=3456" in out
    assert "total_perturbed_model_example_evaluations=172800" in out


# =================================================================================================
# Property 4/5: direction family reuse within a region across radii + independent namespaces
# across regions (reuses build_stage11_population/validate_stage11_direction_seed_reuse BY IMPORT
# -- unmodified, generic functions -- so these tests prove THIS module wires them correctly, not
# that the underlying logic is new).
# =================================================================================================


def test_direction_family_reused_across_radii_within_region_and_validates():
    plan = module.build_stage11_32b_plan(model_name="Qwen/Qwen2.5-VL-32B-Instruct", model_revision="a" * 40, output_root="/tmp/out", n_directions_per_cell=4)
    seed_bank = build_scaling_direction_seed_bank(module.STAGE11_32B_BASE_SEED, "32B", plan.regions, 4)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = module.build_stage11_population(plan, seed_bank, mask_hashes)
    module.validate_stage11_direction_seed_reuse(plan, population)  # must not raise

    for region in plan.regions:
        seeds_by_radius = {radius: {a.direction_seed for a in population[(region, radius)]} for radius in plan.radii}
        radii_iter = iter(seeds_by_radius.values())
        first = next(radii_iter)
        assert all(seeds == first for seeds in radii_iter)  # SAME seed set reused for every radius within a region


def test_direction_namespaces_independent_across_regions():
    seed_bank = build_scaling_direction_seed_bank(module.STAGE11_32B_BASE_SEED, "32B", STAGE8_REGIONS, 8)
    seeds_by_region = {r: set(seed_bank[r]) for r in STAGE8_REGIONS}
    regions = list(seeds_by_region)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            assert seeds_by_region[regions[i]].isdisjoint(seeds_by_region[regions[j]]), f"{regions[i]} and {regions[j]} share a direction seed"


def test_direction_seed_bank_is_scale_namespaced_distinct_from_7b_and_whole_model():
    """32B S2's own base seed must never coincide with 7B anatomy's STAGE11_BASE_SEED or with
    the scaling-common per-scale derivation whole_model already uses for other scales.
    """
    assert module.STAGE11_32B_BASE_SEED != 20260904  # 7B module's own literal STAGE11_BASE_SEED
    bank_32b = build_scaling_direction_seed_bank(module.STAGE11_32B_BASE_SEED, "32B", STAGE8_REGIONS, 4)
    bank_7b_style = build_scaling_direction_seed_bank(20260904, "7B", STAGE8_REGIONS, 4)
    assert bank_32b != bank_7b_style


# =================================================================================================
# Property 6/7/9/10: distributed region-only v3 evaluator wiring, TP4 rank consensus, restoration
# =================================================================================================

_FAKE_APPLY_RESULT_TEMPLATE = {
    "seed": 42, "requested_relative_l2": 0.05, "designed_relative_l2": 0.05, "realized_relative_l2": 0.05,
    "absolute_radius_error": 0.0, "relative_radius_error": 0.0, "realized_abs_error": 0.0,
    "radius_acceptance_mode": "strict", "quantization_limited": False,
    "accepted_scalar": 0.123, "solver_iterations": 2, "quantization_plateau": False,
    "radius_realization_method": "fixed_direction_bf16_quantization_aware_v3_distributed",
    "theta_l2_norm": 10.0, "raw_noise_l2_norm": 200.0, "realized_epsilon_l2_norm": 0.5, "region_param_count": 4,
}


def _build_distributed_assignment(*, region="vision", radius=0.05, seed=42, direction_index=0):
    manifest = PerturbationManifest(
        seed=seed, perturbation_mode="anatomical_relative_l2", model_family="qwen2_5_vl", model_scale="32B",
        model_revision="a" * 40, parameter_mask_hash="fakehash",
        anatomy_region=region, radius=radius, sigma=None,
    )
    return Stage11DirectionAssignment(manifest=manifest, region=region, direction_index=direction_index, direction_seed=seed)


def _make_fake_collective_rpc_all_workers(call_order, *, tensor_parallel_size=4, restoration_ok=True, solver_rank_disagreement=False):
    def _fake(engine, method, args=(), *, label, expected_world_size, ray_get=None):
        call_order.append(label)
        assert expected_world_size == tensor_parallel_size
        if label == "distributed_v3_solver_candidate":
            results = [dict(_FAKE_APPLY_RESULT_TEMPLATE) for _ in range(tensor_parallel_size)]
            if solver_rank_disagreement:
                results[1]["accepted_scalar"] = 999.0
            return results
        if label.startswith("reset_to_base_weights_cpu"):
            return [None] * tensor_parallel_size
        if label.startswith("verify_exact_fixed_base_restoration_cpu"):
            return [
                {"ok": restoration_ok, "max_abs_drift": 0.0 if restoration_ok else 0.05, "fraction_elements_differing": 0.0 if restoration_ok else 0.01}
                for _ in range(tensor_parallel_size)
            ]
        raise AssertionError(f"unexpected collective_rpc_all_workers label {label!r}")
    return _fake


class _FakeRunResult:
    def __init__(self, primary_metric=0.6):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}

    def generation_hash(self):
        return "fakehash"


def _fake_contexts(n=5):
    contexts = {}
    for capability in STAGE8_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=list(ids), partition=partition, subset_hash=partition.manifest_hash, base_score=0.5)
    return contexts


def _fake_run_benchmark_recording(call_order):
    def _run(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        call_order.append("evaluate_capability")
        return _FakeRunResult(primary_metric=0.6)
    return _run


def test_evaluate_one_stage11_32b_candidate_distributed_rpc_transactional_ordering(monkeypatch):
    call_order = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(call_order))
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda engine: call_order.append("reset_vllm_encoder_cache_full"))

    assignment = _build_distributed_assignment(region="vision")
    records = module.evaluate_one_stage11_32b_candidate_distributed_rpc(
        engine=object(), assignment=assignment, region_param_names=("vision.a.weight", "vision.b.weight"),
        capability_contexts=_fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark_recording(call_order), ray_get=_identity_ray_get, tensor_parallel_size=4,
    )
    assert len(records) == 6
    assert {r.capability for r in records} == set(STAGE8_CAPABILITIES)
    assert all(r.anatomy_region == "vision" for r in records)
    assert all(r.model_scale == "32B" for r in records)
    assert all(r.runtime_metadata["region"] == "vision" for r in records)
    assert all(r.runtime_metadata["tensor_parallel_size"] == 4 for r in records)
    assert all(r.runtime_metadata["distributed_rank_consensus_verified"] is True for r in records)
    assert all(r.experiment_id == "stage11_coarse_anatomical_atlas_32b" for r in records)

    expected_prefix = ["distributed_v3_solver_candidate", "reset_vllm_encoder_cache_full"] + ["evaluate_capability"] * 6
    assert call_order[: len(expected_prefix)] == expected_prefix
    remaining = call_order[len(expected_prefix):]
    assert remaining[0].startswith("reset_to_base_weights_cpu")
    assert remaining[1].startswith("verify_exact_fixed_base_restoration_cpu")
    assert remaining[2] == "reset_vllm_encoder_cache_full"


def test_evaluate_one_stage11_32b_candidate_hard_fails_on_rank_disagreement(monkeypatch):
    """Property 7 -- TP4 rank consensus: verify_solver_rank_consensus must actually run and hard
    -fail on a real rank disagreement, never silently accepted from rank 0 alone.
    """
    from neural_thickets_repro.thicket.distributed_v3_solver import SolverRankConsensusError

    call_order = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(call_order, solver_rank_disagreement=True))
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda engine: call_order.append("reset_vllm_encoder_cache_full"))

    assignment = _build_distributed_assignment(region="language")
    with pytest.raises(SolverRankConsensusError):
        module.evaluate_one_stage11_32b_candidate_distributed_rpc(
            engine=object(), assignment=assignment, region_param_names=("language.a.weight",),
            capability_contexts=_fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark_recording(call_order), ray_get=_identity_ray_get, tensor_parallel_size=4,
        )
    assert "evaluate_capability" not in call_order
    assert any(label.startswith("reset_to_base_weights_cpu") for label in call_order)  # restore-on-failure still attempted


def test_evaluate_one_stage11_32b_candidate_raises_on_restoration_failure(monkeypatch):
    """Property 10 -- exact restoration verification."""
    call_order = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(call_order, restoration_ok=False))
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda engine: call_order.append("reset_vllm_encoder_cache_full"))

    assignment = _build_distributed_assignment(region="multimodal_connector_or_merger")
    with pytest.raises(module.RestorationFailedError):
        module.evaluate_one_stage11_32b_candidate_distributed_rpc(
            engine=object(), assignment=assignment, region_param_names=("merger.a.weight",),
            capability_contexts=_fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark_recording(call_order), ray_get=_identity_ray_get, tensor_parallel_size=4,
        )


def test_evaluator_uses_the_real_distributed_v3_solver_not_a_shortcut():
    """Property 9 -- outside-region unchanged: the real distributed solver
    (scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed) is what actually
    enforces out-of-region drift == 0 (aggregate_distributed_out_of_region_drift, raising
    CorrectionOutOfRegionDriftError on any nonzero drift -- see thicket/distributed_v3_solver.py's
    own module docstring/tests). This proves THIS module's evaluator genuinely dispatches that
    real function (source-level), rather than a shortcut that skips the check.
    """
    source = inspect.getsource(module.evaluate_one_stage11_32b_candidate_distributed_rpc)
    assert "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed" in source
    assert "verify_solver_rank_consensus" in source
    assert "build_shard_specs_for_region" in source


# =================================================================================================
# Property 8: global anatomy completeness/disjointness -- proves main() actually dispatches the
# real, already rank-consensus-checked, generic distributed anatomy audit.
# =================================================================================================


def test_main_dispatches_the_real_distributed_anatomy_audit_with_rank_consensus():
    source = inspect.getsource(module.main)
    assert "report_global_anatomy_audit_rpc" in source
    assert "verify_anatomy_audit_rank_consensus" in source
    assert "ensure_scaling_anatomy_audit_passes" in source
    assert "collective_rpc_all_workers" in source


def test_module_sets_nccl_p2p_disable_before_any_ray_or_vllm_use():
    """Regression test for a real hang found live on a 4xL40S TP=4 pod (2026-09-04): this
    module's engine launch + first collective RPC hung indefinitely (1h10m elapsed, 100% GPU
    utilization on all 4 ranks, near-zero GPU memory resident, zero forward progress) inside
    NCCL collective init on this pod's NVLink-less/cross-NUMA topology -- the exact failure
    mode the two S2/S1 live-readiness probe scripts already work around via
    os.environ.setdefault("NCCL_P2P_DISABLE", "1"), set at their own module top level. This
    module was missing that fix. Verified here at module level (not just inside main()) since
    the fix must take effect before ANY ray/vllm import, which can happen from more than one
    code path."""
    assert module.os.environ.get("NCCL_P2P_DISABLE") == "1"


def test_main_exposes_encoder_cache_reset_before_launch_and_verifies_it_live():
    """Regression test for a real bug found live on a 4xL40S TP=4 pod (2026-09-04): the 32B
    S2 dispatcher launched the engine without first calling
    ensure_stage7b_encoder_cache_reset_mechanism_exposed(), so the Ray-wrapped engine actor
    never had reset_encoder_cache_full available -- the baseline repeatability preflight then
    hard-failed with 'the engine actor does not expose reset_encoder_cache_full', exactly the
    failure mode vlm_adapter.ensure_full_encoder_cache_reset_exposed's own docstring predicts
    for a script that launches before calling it. Fixed by mirroring stage8's own proven
    pre-launch/post-launch call pair exactly (both already tested end-to-end on real GPU
    hardware in Stage 7B/8/9) -- this test locks in that both calls are present and correctly
    ordered relative to launch_stage6_engine, without requiring GPU/vllm/ray to run."""
    source = inspect.getsource(module.main)
    assert "ensure_stage7b_encoder_cache_reset_mechanism_exposed" in source
    assert "ensure_encoder_cache_reset_available" in source

    pre_launch_call_idx = source.index("ensure_stage7b_encoder_cache_reset_mechanism_exposed(EXTERNAL_ROOT)")
    launch_idx = source.index("engines, pgs = launch_stage6_engine(")
    post_launch_verify_idx = source.index("ensure_encoder_cache_reset_available(engine)")
    store_base_weights_idx = source.index('collective_rpc_all_workers(engine, store_base_weights_cpu_rpc')

    assert pre_launch_call_idx < launch_idx, (
        "ensure_stage7b_encoder_cache_reset_mechanism_exposed() must run BEFORE launch_stage6_engine() "
        "-- Ray only picks up the monkey-patched method if it exists on the class before ray.remote() wraps it."
    )
    assert store_base_weights_idx < post_launch_verify_idx, (
        "ensure_encoder_cache_reset_available(engine) must run AFTER the engine is launched and the "
        "CPU base snapshot is stored -- it proves the reset works end-to-end against the LIVE engine."
    )


# =================================================================================================
# Property 11/12/13: checkpoint identity + resume semantics
# =================================================================================================


def _make_plan(tmp_path, **kwargs):
    return module.build_stage11_32b_smoke_plan(model_name="Qwen/Qwen2.5-VL-32B-Instruct", model_revision="a" * 40, output_root=str(tmp_path), **kwargs)


def _make_checkpoint_inputs(plan):
    seed_bank = build_scaling_direction_seed_bank(module.STAGE11_32B_BASE_SEED, "32B", plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    contexts = _fake_contexts(n=plan.d_map_n)
    audit = {"regions": {r: {"mask_hash": mask_hashes[r], "n_tensors": 1, "n_elements": 1} for r in plan.regions}}
    return seed_bank, mask_hashes, contexts, audit


def test_checkpoint_manifest_written_on_first_call_and_reused_on_second(tmp_path):
    plan = _make_plan(tmp_path)
    seed_bank, mask_hashes, contexts, audit = _make_checkpoint_inputs(plan)
    current = module.build_stage11_32b_checkpoint_manifest(plan, contexts, mask_hashes, seed_bank, audit)
    path = plan.output_dir / "checkpoint_manifest.json"
    first = module.ensure_stage11_32b_checkpoint_manifest(path, current)
    assert first == current
    assert path.exists()
    second = module.ensure_stage11_32b_checkpoint_manifest(path, current)  # identical current -> reuse, no raise
    assert second == current


def test_checkpoint_identity_mismatch_fails_closed(tmp_path):
    """Property 13: model revision / region hashes / radii / subset hashes / direction bank /
    scientific-contract mismatch must refuse to resume.
    """
    plan = _make_plan(tmp_path)
    seed_bank, mask_hashes, contexts, audit = _make_checkpoint_inputs(plan)
    current = module.build_stage11_32b_checkpoint_manifest(plan, contexts, mask_hashes, seed_bank, audit)
    path = plan.output_dir / "checkpoint_manifest.json"
    module.ensure_stage11_32b_checkpoint_manifest(path, current)

    different_plan = _make_plan(tmp_path)
    different_mask_hashes = {r: f"DIFFERENT_hash_{r}" for r in plan.regions}
    different = module.build_stage11_32b_checkpoint_manifest(different_plan, contexts, different_mask_hashes, seed_bank, audit)
    with pytest.raises(module.IncompatibleStage11_32BCheckpointError):
        module.ensure_stage11_32b_checkpoint_manifest(path, different)


def test_run_stage11_32b_rpc_skips_completed_perturbations(tmp_path):
    """Property 11: a perturbation_id with all 6 capability rows already on disk is skipped --
    the injected evaluator is never called for it.
    """
    plan = _make_plan(tmp_path)
    seed_bank, mask_hashes, contexts, audit = _make_checkpoint_inputs(plan)
    region_param_names_by_region = {r: (f"{r}.a.weight",) for r in plan.regions}

    population = module.build_stage11_population(plan, seed_bank, mask_hashes)
    # pre-populate results.jsonl with a COMPLETE set of 6 rows for the FIRST perturbation only
    from neural_thickets_repro.thicket.schema import ExperimentResultRecord

    first_cell = next(iter(population.values()))
    first_pid = first_cell[0].manifest.perturbation_id
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        ExperimentResultRecord(
            experiment_id="stage11_coarse_anatomical_atlas_32b", perturbation_id=first_pid,
            model_family="qwen2_5_vl", model_scale="32B", model_revision=plan.model_revision,
            perturbation_mode="anatomical_relative_l2", anatomy_region=first_cell[0].region,
            radius=first_cell[0].manifest.radius, sigma=None, seed=first_cell[0].manifest.seed,
            parameter_mask_hash=first_cell[0].manifest.parameter_mask_hash, capability=cap,
            dataset_role="map", subset_hash=contexts[cap].subset_hash, base_score=0.5, perturbed_score=0.55,
            delta=0.05, parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash="h", runtime_metadata={},
        )
        for cap in STAGE8_CAPABILITIES
    ]
    from neural_thickets_repro.run_global_visual_thicket_pilot import append_candidate_rows

    append_candidate_rows(plan.output_dir / "results.jsonl", records)

    calls = []

    def _fake_evaluate(engine, assignment, region_param_names, capability_contexts, tokenizer, sampling_params, *, run_benchmark, ray_get=None, generation_batch_size=None, rss_checkpoint=None):
        calls.append(assignment.manifest.perturbation_id)
        return [records[0]]  # arbitrary non-empty return

    newly_written = module.run_stage11_32b_rpc(
        plan, contexts, engine=object(), tokenizer=None, sampling_params=None, seed_bank=seed_bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes, anatomy_audit=audit,
        run_benchmark=None, evaluate_one_candidate=_fake_evaluate,
    )
    assert first_pid not in calls  # already-complete perturbation was skipped
    total_expected = plan.total_unique_perturbations
    assert len(calls) == total_expected - 1  # every OTHER perturbation was still attempted


def test_run_stage11_32b_rpc_recovers_incomplete_candidate_without_duplicating_complete_rows(tmp_path):
    """Property 12: a perturbation_id with only a PARTIAL set of capability rows on disk (crash
    mid-candidate) is treated as NOT complete -- it is re-run from scratch, and because
    append_candidate_rows is only ever called ONCE per successful candidate (after ALL
    capabilities + restore + verify succeed), no duplicate rows for an already-complete
    perturbation_id are ever created.
    """
    plan = _make_plan(tmp_path)
    seed_bank, mask_hashes, contexts, audit = _make_checkpoint_inputs(plan)
    region_param_names_by_region = {r: (f"{r}.a.weight",) for r in plan.regions}
    population = module.build_stage11_population(plan, seed_bank, mask_hashes)
    first_cell = next(iter(population.values()))
    first_pid = first_cell[0].manifest.perturbation_id

    from neural_thickets_repro.run_global_visual_thicket_pilot import append_candidate_rows
    from neural_thickets_repro.thicket.schema import ExperimentResultRecord

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    partial_record = ExperimentResultRecord(
        experiment_id="stage11_coarse_anatomical_atlas_32b", perturbation_id=first_pid,
        model_family="qwen2_5_vl", model_scale="32B", model_revision=plan.model_revision,
        perturbation_mode="anatomical_relative_l2", anatomy_region=first_cell[0].region,
        radius=first_cell[0].manifest.radius, sigma=None, seed=first_cell[0].manifest.seed,
        parameter_mask_hash=first_cell[0].manifest.parameter_mask_hash, capability=STAGE8_CAPABILITIES[0],
        dataset_role="map", subset_hash=contexts[STAGE8_CAPABILITIES[0]].subset_hash, base_score=0.5, perturbed_score=0.55,
        delta=0.05, parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash="h", runtime_metadata={},
    )
    append_candidate_rows(plan.output_dir / "results.jsonl", [partial_record])  # ONLY 1 of 6 capability rows -- simulates a mid-candidate crash

    calls = []

    def _fake_evaluate(engine, assignment, region_param_names, capability_contexts, tokenizer, sampling_params, *, run_benchmark, ray_get=None, generation_batch_size=None, rss_checkpoint=None):
        calls.append(assignment.manifest.perturbation_id)
        return [
            ExperimentResultRecord(
                experiment_id="stage11_coarse_anatomical_atlas_32b", perturbation_id=assignment.manifest.perturbation_id,
                model_family="qwen2_5_vl", model_scale="32B", model_revision=plan.model_revision,
                perturbation_mode="anatomical_relative_l2", anatomy_region=assignment.region,
                radius=assignment.manifest.radius, sigma=None, seed=assignment.manifest.seed,
                parameter_mask_hash=assignment.manifest.parameter_mask_hash, capability=cap,
                dataset_role="map", subset_hash=contexts[cap].subset_hash, base_score=0.5, perturbed_score=0.55,
                delta=0.05, parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash="h2", runtime_metadata={},
            )
            for cap in STAGE8_CAPABILITIES
        ]

    module.run_stage11_32b_rpc(
        plan, contexts, engine=object(), tokenizer=None, sampling_params=None, seed_bank=seed_bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes, anatomy_audit=audit,
        run_benchmark=None, evaluate_one_candidate=_fake_evaluate,
    )
    assert first_pid in calls  # the incomplete perturbation WAS re-attempted, never silently skipped

    from neural_thickets_repro.run_global_visual_thicket_pilot import load_records

    all_records = load_records(plan.output_dir / "results.jsonl")
    rows_for_first_pid = [r for r in all_records if r.perturbation_id == first_pid]
    # exactly 6 rows for first_pid on disk: the original partial 1 + the fresh complete 6 would be
    # 7 IF duplicated -- but a correctly-behaving loop only ever calls append_candidate_rows once
    # more (with a fresh, complete 6-row set) for this perturbation_id, so the file legitimately
    # contains 1 (stale partial) + 6 (fresh complete) = 7 raw lines, but NEVER a complete set
    # counted twice: exactly one COMPLETE (6-capability) group exists among them, and no
    # capability's row appears more than twice in total (once stale-partial, once fresh).
    fresh_rows = [r for r in rows_for_first_pid if r.per_example_result_hash == "h2"]
    assert len(fresh_rows) == 6
    assert {r.capability for r in fresh_rows} == set(STAGE8_CAPABILITIES)


# =================================================================================================
# Property 15: full N=50 authoritative subset-hash enforcement -- reuses run_subset_hash_check/
# ensure_subset_hashes_match_stage8 BY IMPORT (already tested generically in test_run_stage11_
# whole_model_scaling.py); this proves THIS module actually calls them on the full path.
# =================================================================================================


def test_full_run_path_calls_the_authoritative_subset_hash_gate():
    source = inspect.getsource(module.main)
    assert "run_subset_hash_check" in source
    assert "ensure_subset_gate_passes" in source
    assert "STAGE8_AUTHORITATIVE_SUBSET_HASHES" not in source  # never re-typed here -- reused by import inside the imported function itself


def test_authoritative_subset_hashes_reused_by_import_unchanged():
    assert module.STAGE8_AUTHORITATIVE_SUBSET_HASHES is STAGE8_AUTHORITATIVE_SUBSET_HASHES


# =================================================================================================
# Property 17: readiness-gated main() -- blocked without valid S2 evidence, authorized with it.
# =================================================================================================


class _ReachedSharedLifecycleMarker(Exception):
    pass


def _patch_resolve_model_snapshot_to_raise_marker(monkeypatch):
    import neural_thickets_repro.vlm_adapter as vlm_adapter

    def _boom(*a, **k):
        raise _ReachedSharedLifecycleMarker()

    monkeypatch.setattr(vlm_adapter, "resolve_model_snapshot", _boom)


_FAKE_REVISION = "7cfb30d71a1f4f49a57592323337a4a4727301da"


def _write_valid_base_evidence(evidence_dir, *, revision=_FAKE_REVISION):
    import neural_thickets_repro.stage11_32b_readiness as readiness

    evidence_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "resolved_revision": {"model_name": readiness.FROZEN_32B_MODEL_NAME, "resolved_revision": revision},
        "model_load": {"ok": True, "config": {"tensor_parallel_size": 4, "dtype": "bfloat16", "base_snapshot_mode": "cpu_base_weights", "gpu_memory_utilization": 0.60, "max_model_len": 4096, "enable_prefix_caching": False}},
        "gate_results": {g: readiness.GATE_PASS for g in readiness.GATE_IDS},
        "smoke_permitted": True,
    }
    (evidence_dir / "stage11_32b_live_readiness_report.json").write_text(json.dumps(base))


def _write_valid_s2_evidence(evidence_dir, *, revision=_FAKE_REVISION):
    import neural_thickets_repro.stage11_32b_s2_live_evidence as s2_evidence

    evidence_dir.mkdir(parents=True, exist_ok=True)
    region_pass = {
        "solver_error": None, "acceptance_mode": "strict",
        "rank_consensus": {"core_fields_ok": True, "full_bracket_trajectory": {"ok": True}},
        "restoration": {"ok": True}, "g4_g5_final": {"G4": "PASS", "G5": "PASS"},
    }
    artifact = {
        "resolved_revision": {"model_name": "Qwen/Qwen2.5-VL-32B-Instruct", "resolved_revision": revision},
        "tensor_parallel_size": 4, "regions": {r: dict(region_pass) for r in s2_evidence.S2_REGIONS}, "scientific_rows_written": 0,
    }
    (evidence_dir / "stage11_32b_s2_live_v3_solver_probe_report.json").write_text(json.dumps(artifact))


def test_32b_s2_blocked_without_any_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(module, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    rc = module.main([
        "--smoke", "--output-root", str(tmp_path / "out"),
        "--base-live-evidence-dir", str(tmp_path / "no_base"), "--s2-live-evidence-dir", str(tmp_path / "no_s2"),
    ])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_s2_readiness_gate_report.json")).read_text())
    assert report["live_evidence_found"] is False
    assert report["all_gates_pass"] is False


def test_32b_s2_blocked_when_s1_evidence_present_but_s2_probe_missing(tmp_path, monkeypatch):
    """The critical evidence-binding fix this milestone requires: S1's own (single-region,
    whole_model-scoped) evidence must NEVER, by itself, authorize S2.
    """
    monkeypatch.setattr(module, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(module, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    base_dir = tmp_path / "base"
    _write_valid_base_evidence(base_dir)
    rc = module.main([
        "--smoke", "--output-root", str(tmp_path / "out"),
        "--base-live-evidence-dir", str(base_dir), "--s2-live-evidence-dir", str(tmp_path / "no_s2"),
    ])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_s2_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False


def test_32b_s2_authorized_by_valid_multi_region_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(module, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    base_dir, s2_dir = tmp_path / "base", tmp_path / "s2"
    _write_valid_base_evidence(base_dir)
    _write_valid_s2_evidence(s2_dir)
    _patch_resolve_model_snapshot_to_raise_marker(monkeypatch)

    with pytest.raises(_ReachedSharedLifecycleMarker):
        module.main([
            "--smoke", "--output-root", str(tmp_path / "out"),
            "--base-live-evidence-dir", str(base_dir), "--s2-live-evidence-dir", str(s2_dir),
        ])
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_s2_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is True


def test_32b_s2_authorized_with_evidence_reaches_shared_lifecycle_for_full_too(tmp_path, monkeypatch):
    """Full (no --smoke) is gated identically to smoke -- never distinguished by this gate."""
    monkeypatch.setattr(module, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(module, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    base_dir, s2_dir = tmp_path / "base", tmp_path / "s2"
    _write_valid_base_evidence(base_dir)
    _write_valid_s2_evidence(s2_dir)
    _patch_resolve_model_snapshot_to_raise_marker(monkeypatch)

    with pytest.raises(_ReachedSharedLifecycleMarker):
        module.main([
            "--output-root", str(tmp_path / "out"),
            "--base-live-evidence-dir", str(base_dir), "--s2-live-evidence-dir", str(s2_dir),
        ])


def test_one_missing_region_in_s2_evidence_blocks_authorization(tmp_path, monkeypatch):
    """Property 16 (integration level): if the S2 probe artifact is missing even one region
    (e.g. an interrupted probe run), main() must stay blocked.
    """
    import neural_thickets_repro.stage11_32b_s2_live_evidence as s2_evidence

    monkeypatch.setattr(module, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(module, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": _FAKE_REVISION, "requested_revision": "main"})
    base_dir, s2_dir = tmp_path / "base", tmp_path / "s2"
    _write_valid_base_evidence(base_dir)
    s2_dir.mkdir(parents=True, exist_ok=True)
    region_pass = {
        "solver_error": None, "acceptance_mode": "strict",
        "rank_consensus": {"core_fields_ok": True, "full_bracket_trajectory": {"ok": True}},
        "restoration": {"ok": True}, "g4_g5_final": {"G4": "PASS", "G5": "PASS"},
    }
    incomplete_regions = {r: dict(region_pass) for r in s2_evidence.S2_REGIONS if r != "language"}
    artifact = {
        "resolved_revision": {"model_name": "Qwen/Qwen2.5-VL-32B-Instruct", "resolved_revision": _FAKE_REVISION},
        "tensor_parallel_size": 4, "regions": incomplete_regions, "scientific_rows_written": 0,
    }
    (s2_dir / "stage11_32b_s2_live_v3_solver_probe_report.json").write_text(json.dumps(artifact))

    rc = module.main(["--smoke", "--output-root", str(tmp_path / "out"), "--base-live-evidence-dir", str(base_dir), "--s2-live-evidence-dir", str(s2_dir)])
    assert rc == 0
    report = json.loads(next((tmp_path / "out").rglob("stage11_32b_s2_readiness_gate_report.json")).read_text())
    assert report["all_gates_pass"] is False


# =================================================================================================
# Property 18/20: 7B anatomy module + engine config are untouched by this module's existence.
# =================================================================================================


def test_7b_anatomy_module_still_tp1_only_unchanged():
    source = inspect.getsource(anatomy_7b._collective_rpc_single_worker) + inspect.getsource(anatomy_7b._validate_collective_rpc_results)
    assert "TP=1-only" in source


def test_new_module_never_imports_or_mutates_7b_module_state():
    assert not hasattr(anatomy_7b, "STAGE11_32B_REGIONS")
    assert not hasattr(anatomy_7b, "evaluate_one_stage11_32b_candidate_distributed_rpc")


def test_engine_config_never_requests_quantization_and_matches_frozen_contract():
    from neural_thickets_repro.stage11_32b_readiness import build_32b_engine_config

    cfg = build_32b_engine_config(tensor_parallel_size=4)
    assert cfg["tensor_parallel_size"] == 4
    assert cfg["max_model_len"] == 4096
    assert cfg["gpu_memory_utilization"] == pytest.approx(0.60)
    assert cfg["enforce_eager"] is True
    assert cfg["enable_prefix_caching"] is False
    assert cfg["precision"] == "bfloat16"
    assert cfg["base_snapshot_mode"] == "cpu_base_weights"
