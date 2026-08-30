"""Tests for run_stage11_whole_model_scaling.py -- CPU-only, same fake-engine philosophy as
test_run_stage11_coarse_anatomical_atlas_7b.py: the real GPU/Ray/vLLM engine is never touched;
RPC dispatch is tested against a fake, persistent-worker-shaped engine, using REAL small torch
tensors for the lifecycle/anatomy-audit tests that need genuine weight mutation/restoration.
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

import neural_thickets_repro.run_stage11_whole_model_scaling as module
from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_D_MAP_N, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_RADII
from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import STAGE8_AUTHORITATIVE_SUBSET_HASHES
from neural_thickets_repro.scaling_common import SCALING_MODEL_REGISTRY, WHOLE_MODEL_REGION_LABEL, WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest
from neural_thickets_repro.run_stage11_whole_model_scaling import (
    SUBSET_GATE_MODE_FULL,
    SUBSET_GATE_MODE_SMOKE,
    WholeModelCheckpointManifest,
    WholeModelDirectionAssignment,
    WholeModelDirectionSeedReuseViolationError,
    IncompatibleWholeModelCheckpointError,
    Stage11SmokeSubsetNondeterminismError,
    Stage11SubsetHashMismatchError,
    build_subset_gate_report,
    build_whole_model_checkpoint_manifest,
    build_whole_model_plan,
    build_whole_model_population,
    build_whole_model_smoke_plan,
    compute_whole_model_run_signature,
    ensure_smoke_subset_determinism,
    ensure_subset_gate_passes,
    ensure_subset_hashes_match_stage8,
    ensure_whole_model_checkpoint_manifest,
    evaluate_one_whole_model_candidate_rpc,
    run_smoke_subset_determinism_check,
    run_subset_hash_check,
    run_whole_model_rpc,
    validate_whole_model_direction_seed_reuse,
)


def _identity_ray_get(x):
    return x


@pytest.fixture(autouse=True)
def _fake_encoder_cache_reset(monkeypatch):
    def _fake_reset(engine):
        if hasattr(engine, "calls"):
            engine.calls.append("reset_vllm_encoder_cache_full")
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", _fake_reset)


# =================================================================================================
# 1. Plan counts -- full (192/1152/57600) and smoke (3/18/90)
# =================================================================================================


def test_whole_model_plan_full_counts():
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root="/tmp/whatever")
    assert plan.region_labels == (WHOLE_MODEL_REGION_LABEL,)
    assert plan.total_unique_perturbations == 1 * 3 * 64 == 192
    assert plan.total_perturbation_capability_evaluations == 192 * 6 == 1152
    assert plan.total_perturbed_model_example_evaluations == 1152 * 50 == 57600
    assert plan.run_signature == "stage11_7b_whole_model_v1"
    assert plan.is_smoke is False


def test_whole_model_plan_smoke_counts():
    spec = SCALING_MODEL_REGISTRY["3B"]
    plan = build_whole_model_smoke_plan(spec=spec, model_revision="a" * 40, output_root="/tmp/whatever")
    assert plan.total_unique_perturbations == 1 * 3 * 1 == 3
    assert plan.total_perturbation_capability_evaluations == 3 * 6 == 18
    assert plan.total_perturbed_model_example_evaluations == 18 * 5 == 90
    assert plan.is_smoke is True


def test_whole_model_smoke_plan_32b_counts_exactly_3_perturbations_18_rows():
    """The frozen smoke design (1 direction x 3 radii = 3 perturbations, x 6 capabilities = 18
    rows) is scale-generic -- 32B must produce the IDENTICAL counts 3B/7B already do, never a
    separately-invented 32B-specific smoke size.
    """
    spec = SCALING_MODEL_REGISTRY["32B"]
    plan = build_whole_model_smoke_plan(spec=spec, model_revision="a" * 40, output_root="/tmp/whatever")
    assert plan.total_unique_perturbations == 3
    assert plan.total_perturbation_capability_evaluations == 18
    assert plan.is_smoke is True
    assert plan.scale_label == "32B"


def test_whole_model_run_signature_varies_by_scale():
    sig_3b = compute_whole_model_run_signature("3B", STAGE8_RADII, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_D_MAP_N, 10)
    sig_7b = compute_whole_model_run_signature("7B", STAGE8_RADII, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_D_MAP_N, 10)
    assert sig_3b == "stage11_3b_whole_model_v1"
    assert sig_7b == "stage11_7b_whole_model_v1"
    assert sig_3b != sig_7b


def test_whole_model_run_signature_never_collides_with_the_anatomy_track_naming():
    sig = compute_whole_model_run_signature("7B", STAGE8_RADII, STAGE8_N_DIRECTIONS_PER_CELL, STAGE8_D_MAP_N, 10)
    assert sig != "stage11_coarse_anatomical_atlas_7b_v1"
    assert sig != "stage11_7b_anatomy_v1"


# =================================================================================================
# 2. Historical disqualification -- the old global pilot is never used as a whole_model anchor
# =================================================================================================


def test_historical_disqualification_note_names_the_actual_reason():
    assert "global_gaussian_upstream" in WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE
    assert "visual" in WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE.lower()


def test_module_never_loads_the_historical_global_pilot_results_as_a_baseline():
    """Source-scan: run_stage11_whole_model_scaling.py must not reference the historical 3B
    global pilot's OWN results directory/run-signature as a whole_model data source -- only its
    (unrelated, still-reused) engine-launch/RPC-transport helper functions.
    """
    source = inspect.getsource(module)
    assert "global_visual_thicket_pilot" not in source.lower().replace("run_global_visual_thicket_pilot", "")
    # the only mentions of "global" in this module are the reused infra import path itself and
    # the historical-disqualification note -- never a "reuse those results" code path.
    assert "stage6_visual_thicket_pilot_results" not in source
    assert "global_gaussian_upstream" not in source or "WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE" in dir(module)


# =================================================================================================
# 3. Population / direction-seed reuse
# =================================================================================================


def test_build_whole_model_population_reuses_each_seed_across_all_radii():
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root="/tmp/x", n_directions_per_cell=4)
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), 4)
    population = build_whole_model_population(plan, bank, mask_hash="deadbeef")
    validate_whole_model_direction_seed_reuse(plan, population)  # must not raise
    assert set(population.keys()) == set(plan.radii)
    for radius, assignments in population.items():
        assert len(assignments) == 4


def test_validate_whole_model_direction_seed_reuse_detects_duplicate_ids():
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root="/tmp/x", n_directions_per_cell=2)
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), 2)
    population = build_whole_model_population(plan, bank, mask_hash="deadbeef")
    # forcibly duplicate one radius's assignments onto another to break the invariant
    radii = list(population.keys())
    broken = dict(population)
    broken[radii[1]] = population[radii[0]]
    with pytest.raises(WholeModelDirectionSeedReuseViolationError):
        validate_whole_model_direction_seed_reuse(plan, broken)


# =================================================================================================
# 4. Subset-hash gate against Stage-8's authoritative 3B manifests
# =================================================================================================


def _fake_contexts(n=5, subset_hashes=None):
    contexts = {}
    for capability in STAGE8_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        subset_hash = subset_hashes.get(capability) if subset_hashes else partition.manifest_hash
        # examples length == n (not always []) so tests can check example counts (e.g. the
        # smoke subset-determinism gate, which asserts every capability has exactly N=5 examples).
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=list(ids), partition=partition, subset_hash=subset_hash, base_score=0.5)
    return contexts


def test_subset_hash_check_passes_with_authoritative_hashes():
    contexts = _fake_contexts(subset_hashes=STAGE8_AUTHORITATIVE_SUBSET_HASHES)
    report = run_subset_hash_check(contexts)
    assert report["all_match"] is True
    ensure_subset_hashes_match_stage8(report)  # must not raise


def test_subset_hash_check_fails_with_wrong_hashes():
    contexts = _fake_contexts()
    report = run_subset_hash_check(contexts)
    assert report["all_match"] is False
    with pytest.raises(Stage11SubsetHashMismatchError):
        ensure_subset_hashes_match_stage8(report)


# =================================================================================================
# 4b. MODE-AWARE subset gate -- reproduces and fixes the live smoke-vs-N=50-hash failure class
# =================================================================================================


def test_full_mode_subset_gate_still_hard_fails_on_a_single_mismatched_n50_hash():
    """Regression (task item 4): the FULL-mode gate must remain strict -- even ONE of the six
    N=50 live subset hashes differing from STAGE8_AUTHORITATIVE_SUBSET_HASHES is a hard stop.
    """
    contexts = _fake_contexts(subset_hashes=dict(STAGE8_AUTHORITATIVE_SUBSET_HASHES))
    # corrupt exactly one capability's live hash
    one_capability = next(iter(contexts))
    from dataclasses import replace as _dc_replace
    contexts[one_capability] = _dc_replace(contexts[one_capability], subset_hash="corrupted" * 4)

    full_report = run_subset_hash_check(contexts)
    assert full_report["all_match"] is False
    gate_report = build_subset_gate_report(is_smoke=False, d_map_n=STAGE8_D_MAP_N, full_subset_hash_report=full_report)
    assert gate_report["subset_gate_mode"] == SUBSET_GATE_MODE_FULL
    with pytest.raises(Stage11SubsetHashMismatchError):
        ensure_subset_gate_passes(gate_report)


def test_full_mode_subset_gate_passes_when_all_six_hashes_match():
    contexts = _fake_contexts(subset_hashes=dict(STAGE8_AUTHORITATIVE_SUBSET_HASHES))
    full_report = run_subset_hash_check(contexts)
    gate_report = build_subset_gate_report(is_smoke=False, d_map_n=STAGE8_D_MAP_N, full_subset_hash_report=full_report)
    ensure_subset_gate_passes(gate_report)  # must not raise


def test_smoke_mode_never_raises_subset_hash_mismatch_merely_because_n5_differs_from_n50():
    """Reproduces the EXACT reported failure class: an N=5 smoke subset hash will NEVER equal
    the N=50 authoritative Stage-8 hash (different sample sizes by construction) -- smoke mode
    must not raise Stage11SubsetHashMismatchError over this, and must instead pass its own
    deterministic-reconstruction gate.
    """
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE8_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    pass_b = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)  # independently "rebuilt", same deterministic hashes

    # Sanity: these N=5 hashes indeed differ from every N=50 authoritative hash (the root cause).
    for cap in STAGE8_CAPABILITIES:
        assert pass_a[cap].subset_hash != STAGE8_AUTHORITATIVE_SUBSET_HASHES[cap]

    determinism_report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    assert determinism_report["all_deterministic"] is True
    assert determinism_report["all_n_matches_expected"] is True
    gate_report = build_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=determinism_report)
    assert gate_report["subset_gate_mode"] == SUBSET_GATE_MODE_SMOKE
    ensure_subset_gate_passes(gate_report)  # must NOT raise Stage11SubsetHashMismatchError (or anything else)


def test_two_independent_smoke_manifest_builds_are_identical():
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE8_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    pass_b = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    for cap in STAGE8_CAPABILITIES:
        assert report[cap]["pass_a_subset_hash"] == report[cap]["pass_b_subset_hash"]
        assert report[cap]["matches"] is True


def test_changing_one_smoke_example_breaks_determinism_check():
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE8_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    changed_hashes = dict(n5_hash_by_cap)
    one_capability = STAGE8_CAPABILITIES[0]
    changed_hashes[one_capability] = "a_different_n5_hash_from_a_changed_example"
    pass_b = _fake_contexts(n=5, subset_hashes=changed_hashes)

    determinism_report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    assert determinism_report["all_deterministic"] is False
    assert determinism_report[one_capability]["matches"] is False
    gate_report = build_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=determinism_report)
    with pytest.raises(Stage11SmokeSubsetNondeterminismError):
        ensure_subset_gate_passes(gate_report)


def test_smoke_determinism_check_fails_if_a_capability_does_not_have_exactly_n5_examples():
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE8_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    pass_b = _fake_contexts(n=7, subset_hashes=n5_hash_by_cap)  # wrong N -- e.g. a config drift
    report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    assert report["all_n_matches_expected"] is False
    with pytest.raises(Stage11SmokeSubsetNondeterminismError):
        ensure_smoke_subset_determinism(report)


def test_subset_gate_report_requires_the_matching_sub_report_for_its_mode():
    with pytest.raises(ValueError):
        build_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=None)
    with pytest.raises(ValueError):
        build_subset_gate_report(is_smoke=False, d_map_n=50, full_subset_hash_report=None)


def test_ensure_subset_gate_passes_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        ensure_subset_gate_passes({"subset_gate_mode": "not_a_real_mode"})


def test_subset_gate_runs_before_engine_launch_and_before_any_candidate_row(runtime_wrapped_vlm_32vision_factory=None):
    """Task item 7: a failing subset gate must exit BEFORE any GPU engine is launched, before the
    live anatomy audit, and before any Stage-11 candidate row is evaluated/checkpointed -- proven
    structurally from main()'s own source ordering (the gate call sites, engine launch, and the
    candidate-lifecycle entry point all appear as distinct, findable statements in one function).
    """
    source = inspect.getsource(module.main)
    gate_pos = source.index("ensure_subset_gate_passes(subset_gate_report)")
    engine_launch_pos = source.index("launch_stage6_engine(")
    anatomy_audit_pos = source.index("report_scaling_anatomy_audit")
    run_rpc_pos = source.index("run_whole_model_rpc(")
    assert gate_pos < engine_launch_pos < anatomy_audit_pos < run_rpc_pos


def test_smoke_subset_gate_never_calls_full_mode_equality_check(monkeypatch):
    """Structural proof the smoke path cannot accidentally fall through to the N=50 comparison."""
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE8_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    pass_b = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    determinism_report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    gate_report = build_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=determinism_report)

    def _should_never_be_called(report):
        raise AssertionError("ensure_subset_hashes_match_stage8 must never be called for a smoke gate report")

    monkeypatch.setattr(module, "ensure_subset_hashes_match_stage8", _should_never_be_called)
    ensure_subset_gate_passes(gate_report)  # must not raise -- dispatches to the smoke path only


# =================================================================================================
# 5. Checkpoint identity
# =================================================================================================


def _region_mask_hash():
    return "deadbeef" * 8


def _fake_audit():
    return {"regions": {WHOLE_MODEL_REGION_LABEL: {"mask_hash": _region_mask_hash(), "n_tensors": 10, "n_elements": 1000}}}


def test_checkpoint_manifest_round_trips_through_dict():
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root="/tmp/x")
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), plan.n_directions_per_cell)
    checkpoint = build_whole_model_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hash(), bank, _fake_audit())
    restored = WholeModelCheckpointManifest.from_dict(checkpoint.to_dict())
    assert restored == checkpoint
    assert checkpoint.scale_label == "7B"
    assert checkpoint.track == "whole_model"


def test_ensure_whole_model_checkpoint_manifest_rejects_a_mismatched_resume(tmp_path):
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root=str(tmp_path))
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), plan.n_directions_per_cell)
    checkpoint_a = build_whole_model_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hash(), bank, _fake_audit())
    path = tmp_path / "checkpoint_manifest.json"
    ensure_whole_model_checkpoint_manifest(path, checkpoint_a)

    different_audit = {"regions": {WHOLE_MODEL_REGION_LABEL: {"mask_hash": "different" * 8, "n_tensors": 10, "n_elements": 1000}}}
    checkpoint_b = build_whole_model_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hash(), bank, different_audit)
    with pytest.raises(IncompatibleWholeModelCheckpointError):
        ensure_whole_model_checkpoint_manifest(path, checkpoint_b)


# =================================================================================================
# 6. Candidate lifecycle -- bounded memory, restores exactly, Ray-unrecoverable failure path
# =================================================================================================


class _FakeWholeModelEngine:
    def __init__(self, model):
        self._model = model
        self._base_weights = None
        self.calls = []
        self._worker_self = SimpleNamespace(
            model_runner=SimpleNamespace(model=model), reset_to_base_weights=self._reset_to_base_weights, _should_perturb=should_perturb,
        )
        self.collective_rpc = SimpleNamespace(remote=self._collective_rpc)

    def store_base_weights(self):
        self._base_weights = {name: p.detach().clone() for name, p in self._model.named_parameters()}
        self._worker_self._base_weights = self._base_weights

    def _reset_to_base_weights(self):
        if self._base_weights is None:
            raise RuntimeError("store_base_weights not called")
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                p.copy_(self._base_weights[name])

    def _collective_rpc(self, method, args=()):
        label = method if isinstance(method, str) else getattr(method, "__name__", "callable")
        self.calls.append(label)
        if method == "reset_to_base_weights":
            self._reset_to_base_weights()
            return [True]
        if callable(method):
            return [method(self._worker_self, *args)]
        raise ValueError(f"unsupported method {method!r}")


class _FakeWholeModelRunResult:
    def __init__(self, primary_metric=0.6):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}

    def generation_hash(self):
        return "fakehash"


def _fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
    return _FakeWholeModelRunResult(primary_metric=0.6)


def _build_assignment_and_engine(model, *, radius=0.05, seed=42, direction_index=0):
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names = atlas.region("full_model").param_names
    manifest = PerturbationManifest(
        seed=seed, perturbation_mode="anatomical_relative_l2", model_family="qwen2_5_vl", model_scale="7B",
        model_revision="a" * 40, parameter_mask_hash=atlas.region("full_model").mask_hash,
        anatomy_region=WHOLE_MODEL_REGION_LABEL, radius=radius, sigma=None,
    )
    assignment = WholeModelDirectionAssignment(manifest=manifest, direction_index=direction_index, direction_seed=seed)
    engine = _FakeWholeModelEngine(model)
    engine.store_base_weights()
    return assignment, engine, region_param_names


def test_evaluate_one_whole_model_candidate_rpc_produces_a_record_per_capability(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    records = evaluate_one_whole_model_candidate_rpc(
        engine, assignment, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert len(records) == 6
    assert {r.capability for r in records} == set(STAGE8_CAPABILITIES)
    assert all(r.anatomy_region == WHOLE_MODEL_REGION_LABEL for r in records)
    assert all(r.model_scale == "7B" for r in records)


def test_evaluate_one_whole_model_candidate_rpc_restores_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    evaluate_one_whole_model_candidate_rpc(
        engine, assignment, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), before[name])


def test_evaluate_one_whole_model_candidate_rpc_rejects_wrong_perturbation_mode(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    bad_manifest = PerturbationManifest(
        seed=1, perturbation_mode="global_gaussian_upstream", model_family="qwen2_5_vl", model_scale="7B",
        model_revision="a" * 40, parameter_mask_hash="h", anatomy_region=None, radius=None, sigma=0.001,
    )
    assignment = WholeModelDirectionAssignment(manifest=bad_manifest, direction_index=0, direction_seed=1)
    engine = _FakeWholeModelEngine(model)
    engine.store_base_weights()
    with pytest.raises(ValueError):
        evaluate_one_whole_model_candidate_rpc(
            engine, assignment, atlas.region("full_model").param_names, _fake_contexts(), tokenizer=None,
            sampling_params=None, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )


class _FakeUnrecoverableRayError(RuntimeError):
    pass


def test_evaluate_one_whole_model_candidate_rpc_skips_restoration_rpc_on_ray_unrecoverable_error(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    monkeypatch.setattr(module, "_is_ray_unrecoverable_error", lambda exc: isinstance(exc, _FakeUnrecoverableRayError))

    def _failing_run_benchmark(*args, **kwargs):
        raise _FakeUnrecoverableRayError("actor died")

    reset_calls_before = engine.calls.count("reset_to_base_weights")
    with pytest.raises(_FakeUnrecoverableRayError):
        evaluate_one_whole_model_candidate_rpc(
            engine, assignment, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_failing_run_benchmark, ray_get=_identity_ray_get,
        )
    assert engine.calls.count("reset_to_base_weights") == reset_calls_before


# =================================================================================================
# 32B ONLY: TP=4-aware candidate evaluator -- reuses collective_rpc_all_workers as its dispatch
# mechanism (never engine.collective_rpc directly), so these tests intercept it and record call
# order rather than needing a real TP-sharded model.
# =================================================================================================

import neural_thickets_repro.thicket.distributed_perturbation as dp_module
from neural_thickets_repro.run_stage11_whole_model_scaling import (
    evaluate_one_whole_model_candidate_distributed_rpc,
    reset_to_base_weights_distributed_rpc,
    restore_and_verify_distributed_rpc,
)

_FAKE_APPLY_RESULT_TEMPLATE = {
    # realized == requested EXACTLY -- keeps this fake comfortably within REALIZED_RADIUS_TOLERANCE
    # regardless of its exact value; this test is about dispatch/ordering, not radius-precision
    # (that's the real solver's own, separately-tested job).
    "region": "whole_model", "seed": 42, "direction_seed": 42, "requested_relative_l2": 0.05,
    "designed_relative_l2": 0.05, "realized_relative_l2": 0.05, "absolute_radius_error": 0.0,
    "relative_radius_error": 0.0, "realized_abs_error": 0.0,
    "radius_acceptance_mode": "strict", "quantization_limited": False,
    "accepted_scalar": 0.123, "final_scale": 0.123, "solver_iterations": 2, "correction_iterations": 2,
    "quantization_plateau": False, "radius_realization_method": "fixed_direction_bf16_quantization_aware_v3_distributed",
    "theta_l2_norm": 10.0, "raw_noise_l2_norm": 200.0, "realized_epsilon_l2_norm": 0.5, "region_param_count": 4,
}


def _build_distributed_assignment(*, radius=0.05, seed=42, direction_index=0):
    manifest = PerturbationManifest(
        seed=seed, perturbation_mode="anatomical_relative_l2", model_family="qwen2_5_vl", model_scale="32B",
        model_revision="a" * 40, parameter_mask_hash="fakehash",
        anatomy_region=WHOLE_MODEL_REGION_LABEL, radius=radius, sigma=None,
    )
    return WholeModelDirectionAssignment(manifest=manifest, direction_index=direction_index, direction_seed=seed)


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


def _fake_run_benchmark_recording(call_order):
    def _run(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        call_order.append("evaluate_capability")
        return _FakeWholeModelRunResult(primary_metric=0.6)
    return _run


def test_evaluate_one_whole_model_candidate_distributed_rpc_transactional_ordering(monkeypatch):
    """Exact same transactional ordering as the TP=1 evaluator (perturb -> reset cache -> evaluate
    6 capabilities -> restore -> verify -> reset cache), reused end to end -- only the dispatch
    mechanism differs.
    """
    call_order = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(call_order))
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda engine: call_order.append("reset_vllm_encoder_cache_full"))

    assignment = _build_distributed_assignment()
    records = evaluate_one_whole_model_candidate_distributed_rpc(
        engine=object(), assignment=assignment, region_param_names=("a.weight", "b.weight"),
        capability_contexts=_fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark_recording(call_order), ray_get=_identity_ray_get, tensor_parallel_size=4,
    )
    assert len(records) == 6
    assert {r.capability for r in records} == set(STAGE8_CAPABILITIES)
    assert all(r.model_scale == "32B" for r in records)
    assert all(r.runtime_metadata["tensor_parallel_size"] == 4 for r in records)
    assert all(r.runtime_metadata["distributed_rank_consensus_verified"] is True for r in records)

    expected_prefix = ["distributed_v3_solver_candidate", "reset_vllm_encoder_cache_full"] + ["evaluate_capability"] * 6
    assert call_order[: len(expected_prefix)] == expected_prefix
    remaining = call_order[len(expected_prefix):]
    assert remaining[0].startswith("reset_to_base_weights_cpu")
    assert remaining[1].startswith("verify_exact_fixed_base_restoration_cpu")
    assert remaining[2] == "reset_vllm_encoder_cache_full"


def test_evaluate_one_whole_model_candidate_distributed_rpc_hard_fails_on_rank_disagreement(monkeypatch):
    """verify_solver_rank_consensus (the existing, unmodified consensus check) must actually be
    invoked and must hard-fail on a real rank disagreement -- never silently accepted from rank 0
    alone. Restore is still attempted (the except-block's own responsibility) before re-raising.
    """
    from neural_thickets_repro.thicket.distributed_v3_solver import SolverRankConsensusError

    call_order = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(call_order, solver_rank_disagreement=True))
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda engine: call_order.append("reset_vllm_encoder_cache_full"))

    assignment = _build_distributed_assignment()
    with pytest.raises(SolverRankConsensusError):
        evaluate_one_whole_model_candidate_distributed_rpc(
            engine=object(), assignment=assignment, region_param_names=("a.weight",),
            capability_contexts=_fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark_recording(call_order), ray_get=_identity_ray_get, tensor_parallel_size=4,
        )
    # restore was attempted (proving the except-block's restore-on-failure path ran) even though
    # the candidate never got past the consensus check to evaluate any capability
    assert "evaluate_capability" not in call_order
    assert any(label.startswith("reset_to_base_weights_cpu") for label in call_order)


def test_evaluate_one_whole_model_candidate_distributed_rpc_raises_on_restoration_failure(monkeypatch):
    call_order = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(call_order, restoration_ok=False))
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda engine: call_order.append("reset_vllm_encoder_cache_full"))

    assignment = _build_distributed_assignment()
    with pytest.raises(module.RestorationFailedError):
        evaluate_one_whole_model_candidate_distributed_rpc(
            engine=object(), assignment=assignment, region_param_names=("a.weight",),
            capability_contexts=_fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark_recording(call_order), ray_get=_identity_ray_get, tensor_parallel_size=4,
        )


def test_reset_to_base_weights_distributed_rpc_dispatches_to_every_rank(monkeypatch):
    calls = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(calls))
    reset_to_base_weights_distributed_rpc(object(), tensor_parallel_size=4, ray_get=_identity_ray_get)
    assert calls == ["reset_to_base_weights_cpu_baseline_preflight"]


def test_restore_and_verify_distributed_rpc_aggregates_across_ranks(monkeypatch):
    calls = []
    monkeypatch.setattr(dp_module, "collective_rpc_all_workers", _make_fake_collective_rpc_all_workers(calls))
    result = restore_and_verify_distributed_rpc(object(), tensor_parallel_size=4, ray_get=_identity_ray_get)
    assert result["ok"] is True
    assert result["global_max_abs_drift"] == 0.0
    assert calls == ["reset_to_base_weights_cpu_candidate", "verify_exact_fixed_base_restoration_cpu_candidate"]


def test_run_whole_model_rpc_uses_injected_evaluate_one_candidate(tmp_path):
    """run_whole_model_rpc's own outer loop (population/checkpoint/telemetry/results.jsonl) is
    REUSED unchanged for 32B -- only the per-candidate evaluator differs, via injection.
    """
    spec = SCALING_MODEL_REGISTRY["32B"]
    plan = build_whole_model_smoke_plan(spec=spec, model_revision="a" * 40, output_root=str(tmp_path))
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "32B", (WHOLE_MODEL_REGION_LABEL,), plan.n_directions_per_cell)
    audit = {"regions": {WHOLE_MODEL_REGION_LABEL: {"mask_hash": "fakehash", "n_tensors": 1, "n_elements": 1}}}

    calls = []

    def _fake_evaluate_one_candidate(engine, assignment, region_param_names, capability_contexts, tokenizer, sampling_params, *, run_benchmark, ray_get=None, generation_batch_size=None, rss_checkpoint=None):
        calls.append(assignment.manifest.perturbation_id)
        return []

    n_rows = run_whole_model_rpc(
        plan, _fake_contexts(), object(), tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names=("a.weight",), mask_hash="fakehash", anatomy_audit=audit,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get, evaluate_one_candidate=_fake_evaluate_one_candidate,
    )
    assert len(calls) == plan.total_unique_perturbations == 3
    assert n_rows == 0


def test_run_whole_model_rpc_default_evaluator_is_the_legacy_tp1_one(runtime_wrapped_vlm_32vision_factory, tmp_path):
    """Omitting evaluate_one_candidate entirely (the 3B/7B call shape, unchanged) must still use
    evaluate_one_whole_model_candidate_rpc -- never silently fall back to the distributed one.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_smoke_plan(spec=spec, model_revision="a" * 40, output_root=str(tmp_path))
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), plan.n_directions_per_cell)
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    mask_hash = atlas.region("full_model").mask_hash
    region_param_names = atlas.region("full_model").param_names
    audit = {"regions": {WHOLE_MODEL_REGION_LABEL: {"mask_hash": mask_hash, "n_tensors": len(region_param_names), "n_elements": 1}}}

    engine = _FakeWholeModelEngine(model)
    engine.store_base_weights()

    n_rows = run_whole_model_rpc(
        plan, _fake_contexts(), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names=region_param_names, mask_hash=mask_hash, anatomy_audit=audit,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert n_rows == 3 * 6  # 3 perturbations x 6 capabilities, evaluated via the real legacy single-worker RPC path
    assert "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3" in engine.calls


def test_run_whole_model_rpc_writes_zero_rows_when_the_first_candidate_ooms(tmp_path, runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """Reproduces the reported failure class structurally: an OOM (simulated here as a plain
    RuntimeError, since a real torch.cuda.OutOfMemoryError needs a GPU) during the FIRST
    candidate's scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 call must leave
    ZERO completed perturbations and ZERO scientific result rows -- no capability was ever
    evaluated for that candidate, and no later candidate runs either (the exception aborts the
    whole run_whole_model_rpc loop, not just one iteration).
    """
    model = runtime_wrapped_vlm_32vision_factory()
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root=str(tmp_path), n_directions_per_cell=1)
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), 1)
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    mask_hash = atlas.region("full_model").mask_hash
    region_param_names = atlas.region("full_model").param_names
    audit = {"regions": {WHOLE_MODEL_REGION_LABEL: {"mask_hash": mask_hash, "n_tensors": len(region_param_names), "n_elements": 1}}}

    engine = _FakeWholeModelEngine(model)
    engine.store_base_weights()

    def _simulated_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.03 GiB (simulated, no real GPU involved).")

    monkeypatch.setattr(module, "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3", _simulated_oom)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_whole_model_rpc(
            plan, _fake_contexts(), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
            region_param_names=region_param_names, mask_hash=mask_hash, anatomy_audit=audit,
            run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )

    results_path = plan.output_dir / "results.jsonl"
    telemetry_path = plan.output_dir / "candidate_memory_telemetry.jsonl"
    assert (not results_path.exists()) or results_path.read_text().strip() == ""
    assert (not telemetry_path.exists()) or telemetry_path.read_text().strip() == ""
    # checkpoint_manifest.json IS written (once, before the candidate loop) -- that's run
    # IDENTITY/provenance, never a scientific candidate row.
    assert (plan.output_dir / "checkpoint_manifest.json").exists()


def test_run_whole_model_rpc_writes_zero_rows_when_restoration_verification_ooms(tmp_path, runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """Reproduces the SECOND reported failure class: an OOM during POST-CANDIDATE exact-
    restoration verification (verify_exact_fixed_base_restoration_via_rpc, AFTER all 6
    capabilities were already evaluated) must ALSO leave ZERO completed perturbations and ZERO
    scientific result rows -- append_candidate_rows is only ever reached after
    evaluate_one_whole_model_candidate_rpc RETURNS, and this exception happens before that
    return, regardless of how much real evaluation work already happened for this candidate.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root=str(tmp_path), n_directions_per_cell=1)
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), 1)
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    mask_hash = atlas.region("full_model").mask_hash
    region_param_names = atlas.region("full_model").param_names
    audit = {"regions": {WHOLE_MODEL_REGION_LABEL: {"mask_hash": mask_hash, "n_tensors": len(region_param_names), "n_elements": 1}}}

    engine = _FakeWholeModelEngine(model)
    engine.store_base_weights()

    def _simulated_verify_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 4.06 GiB (simulated, no real GPU involved).")

    monkeypatch.setattr(module, "verify_exact_fixed_base_restoration_via_rpc", _simulated_verify_oom)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_whole_model_rpc(
            plan, _fake_contexts(), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
            region_param_names=region_param_names, mask_hash=mask_hash, anatomy_audit=audit,
            run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )

    results_path = plan.output_dir / "results.jsonl"
    telemetry_path = plan.output_dir / "candidate_memory_telemetry.jsonl"
    assert (not results_path.exists()) or results_path.read_text().strip() == ""
    assert (not telemetry_path.exists()) or telemetry_path.read_text().strip() == ""
    assert (plan.output_dir / "checkpoint_manifest.json").exists()


def test_run_whole_model_rpc_persists_rows_and_resumes(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    spec = SCALING_MODEL_REGISTRY["7B"]
    plan = build_whole_model_plan(spec=spec, model_revision="a" * 40, output_root=str(tmp_path), n_directions_per_cell=1)
    from neural_thickets_repro.scaling_common import build_scaling_direction_seed_bank

    bank = build_scaling_direction_seed_bank(1, "7B", (WHOLE_MODEL_REGION_LABEL,), 1)
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    mask_hash = atlas.region("full_model").mask_hash
    region_param_names = atlas.region("full_model").param_names
    audit = {"regions": {WHOLE_MODEL_REGION_LABEL: {"mask_hash": mask_hash, "n_tensors": len(region_param_names), "n_elements": 1}}}

    engine = _FakeWholeModelEngine(model)
    engine.store_base_weights()

    n_written = run_whole_model_rpc(
        plan, _fake_contexts(), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names=region_param_names, mask_hash=mask_hash, anatomy_audit=audit,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert n_written == plan.total_perturbation_capability_evaluations  # 3 radii x 1 direction x 6 capabilities = 18

    # resume: nothing new to do
    n_written_again = run_whole_model_rpc(
        plan, _fake_contexts(), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names=region_param_names, mask_hash=mask_hash, anatomy_audit=audit,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert n_written_again == 0
