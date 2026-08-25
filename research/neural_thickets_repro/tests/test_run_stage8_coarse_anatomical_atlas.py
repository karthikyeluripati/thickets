"""Tests for run_stage8_coarse_anatomical_atlas.py -- CPU-only. The real GPU/Ray/vLLM engine is
never launched; RPC dispatch is tested against a fake, persistent-worker-shaped engine (same
philosophy as tests/test_run_stage7b_anatomical_calibration.py's _FakeCalibrationEngine), using
REAL small torch tensors for the lifecycle tests that need genuine weight mutation/restoration.
"""
from types import SimpleNamespace

import pytest
import torch

import neural_thickets_repro.run_stage8_coarse_anatomical_atlas as module
from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.run_stage7b_anatomical_calibration import (
    ENABLE_PREFIX_CACHING,
    FULL_CALIBRATION_RADII,
    FULL_CALIBRATION_REGIONS,
    MULTIMODAL_CACHE_POLICY,
    PERTURBATION_MODE,
    RADIUS_REALIZATION_METHOD,
)
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import (
    STAGE8_CAPABILITIES,
    STAGE8_D_MAP_N,
    STAGE8_N_DIRECTIONS_PER_CELL,
    STAGE8_RADII,
    STAGE8_REGIONS,
    STAGE8_SMOKE_D_MAP_N,
    STAGE8_SMOKE_N_DIRECTIONS,
    BaselineNondeterminismError,
    DirectionSeedReuseViolationError,
    IncompatibleStage8CheckpointError,
    Stage8CheckpointManifest,
    build_stage8_checkpoint_manifest,
    build_stage8_direction_seed_bank,
    build_stage8_plan,
    build_stage8_population,
    build_stage8_smoke_plan,
    compute_direction_seed_bank_hash,
    compute_stage8_run_signature,
    ensure_baseline_repeatability,
    ensure_stage8_checkpoint_manifest,
    evaluate_one_stage8_candidate_rpc,
    run_baseline_repeatability_preflight_rpc,
    run_stage8_rpc,
    validate_stage8_direction_seed_reuse,
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
# 1. Frozen full config
# =================================================================================================


def test_stage8_uses_exactly_six_frozen_capabilities():
    assert STAGE8_CAPABILITIES == (
        "visual_grounding", "counting", "spatial_reasoning",
        "ocr_text_recognition_grounded", "relational_reasoning", "fine_grained_recognition",
    )
    assert len(STAGE8_CAPABILITIES) == 6


def test_stage8_regions_are_the_three_l1_anatomy_regions_byte_identical_to_stage7a():
    assert STAGE8_REGIONS == FULL_CALIBRATION_REGIONS == ("vision", "multimodal_connector_or_merger", "language")


def test_stage8_radii_are_exactly_three_and_match_the_frozen_stage7b_common_radii():
    assert STAGE8_RADII == (FULL_CALIBRATION_RADII[0], FULL_CALIBRATION_RADII[1], FULL_CALIBRATION_RADII[3])
    assert len(STAGE8_RADII) == 3


def test_stage8_never_includes_the_calibration_scale_destructive_radii():
    destructive = {FULL_CALIBRATION_RADII[4], FULL_CALIBRATION_RADII[5]}
    assert destructive == {0.1784941427189971, 0.3569882854379942}
    assert not (set(STAGE8_RADII) & destructive)


def test_stage8_n_directions_per_cell_is_64():
    assert STAGE8_N_DIRECTIONS_PER_CELL == 64


def test_stage8_d_map_n_is_50():
    assert STAGE8_D_MAP_N == 50


def test_default_stage8_plan_is_the_frozen_full_identity():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    assert plan.regions == STAGE8_REGIONS
    assert plan.radii == STAGE8_RADII
    assert plan.capabilities == STAGE8_CAPABILITIES
    assert plan.n_directions_per_cell == 64
    assert plan.d_map_n == 50
    assert plan.is_smoke is False
    assert plan.run_signature == "stage8_coarse_anatomical_atlas_3b_v1"


def test_full_stage8_plan_totals_576_perturbations_3456_rows_172800_evaluations():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    assert plan.total_unique_perturbations == 3 * 3 * 64 == 576
    assert plan.total_perturbation_capability_evaluations == 576 * 6 == 3456
    assert plan.total_perturbed_model_example_evaluations == 3456 * 50 == 172_800


def test_build_stage8_plan_rejects_destructive_radii():
    with pytest.raises(ValueError):
        build_stage8_plan(model_name="m", model_revision="rev1", output_root="out", radii=(0.1784941427189971,))


def test_build_stage8_plan_rejects_empty_regions():
    with pytest.raises(ValueError):
        build_stage8_plan(model_name="m", model_revision="rev1", output_root="out", regions=())


def test_build_stage8_plan_rejects_unrecognized_d_map_size():
    with pytest.raises(module.DatasetRoleViolationError):
        build_stage8_plan(model_name="m", model_revision="rev1", output_root="out", d_map_n=17)


def test_perturbation_mode_is_anatomical_relative_l2():
    assert PERTURBATION_MODE == "anatomical_relative_l2"


def test_radius_realization_method_reused_by_identity_from_stage7b():
    assert module.RADIUS_REALIZATION_METHOD == RADIUS_REALIZATION_METHOD == "fixed_direction_bf16_quantization_aware_v3"


def test_cache_policy_and_prefix_caching_reused_by_identity_from_stage7b():
    assert module.MULTIMODAL_CACHE_POLICY == MULTIMODAL_CACHE_POLICY == "full_encoder_reset_vllm011_verified_v2"
    assert module.ENABLE_PREFIX_CACHING == ENABLE_PREFIX_CACHING is False


def test_stage8_plan_enable_prefix_caching_is_false():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    assert plan.enable_prefix_caching is False


# =================================================================================================
# 2. Smoke config
# =================================================================================================


def test_smoke_config_is_1_direction_5_examples():
    assert STAGE8_SMOKE_N_DIRECTIONS == 1
    assert STAGE8_SMOKE_D_MAP_N == 5


def test_smoke_plan_totals_9_perturbations_54_rows_270_evaluations():
    plan = build_stage8_smoke_plan(model_name="m", model_revision="rev1", output_root="out")
    assert plan.is_smoke is True
    assert plan.regions == STAGE8_REGIONS  # all 3 regions, same as full
    assert plan.radii == STAGE8_RADII  # all 3 radii, same as full
    assert plan.capabilities == STAGE8_CAPABILITIES  # all 6 capabilities, same as full
    assert plan.total_unique_perturbations == 3 * 3 * 1 == 9
    assert plan.total_perturbation_capability_evaluations == 9 * 6 == 54
    assert plan.total_perturbed_model_example_evaluations == 54 * 5 == 270


def test_smoke_and_full_plans_never_share_an_output_directory():
    full_plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    smoke_plan = build_stage8_smoke_plan(model_name="m", model_revision="rev1", output_root="out")
    assert full_plan.output_dir != smoke_plan.output_dir
    assert full_plan.run_signature != smoke_plan.run_signature


def test_compute_stage8_run_signature_never_returns_the_full_literal_for_any_deviation():
    sig = compute_stage8_run_signature(STAGE8_REGIONS, STAGE8_RADII, 1, 5)
    assert sig != "stage8_coarse_anatomical_atlas_3b_v1"
    assert sig.startswith("stage8_smoke_")


# =================================================================================================
# 3. Direction-family seed bank + population
# =================================================================================================


def test_direction_seed_bank_has_64_seeds_per_region_for_the_full_config():
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, STAGE8_REGIONS, 64)
    assert set(bank.keys()) == set(STAGE8_REGIONS)
    for region, seeds in bank.items():
        assert len(seeds) == 64
        assert len(set(seeds)) == 64  # no internal duplicate within one region's own bank


def test_direction_seed_bank_is_deterministic():
    bank1 = build_stage8_direction_seed_bank(42, STAGE8_REGIONS, 8)
    bank2 = build_stage8_direction_seed_bank(42, STAGE8_REGIONS, 8)
    assert bank1 == bank2


def test_direction_seed_bank_is_not_a_function_of_radius():
    """The bank-building call itself never even takes a radius argument -- this test pins that
    the SAME seed for (region, i) is what gets reused verbatim across every radius in
    build_stage8_population (see the next test), which is the actual scientific requirement.
    """
    bank = build_stage8_direction_seed_bank(42, ("vision",), 4)
    assert len(bank["vision"]) == 4


def test_same_direction_seed_reused_across_all_radii_within_a_region():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out", n_directions_per_cell=4)
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage8_population(plan, bank, mask_hashes)

    for region in plan.regions:
        seeds_per_radius = [
            [a.direction_seed for a in population[(region, radius)]] for radius in plan.radii
        ]
        assert all(s == seeds_per_radius[0] for s in seeds_per_radius), f"region {region} seed sequence differs across radii"


def test_population_has_576_unique_perturbation_ids_for_the_full_config():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage8_population(plan, bank, mask_hashes)
    all_ids = [a.manifest.perturbation_id for cell in population.values() for a in cell]
    assert len(all_ids) == 576
    assert len(set(all_ids)) == 576


def test_population_is_deterministic():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out", n_directions_per_cell=4)
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    pop1 = build_stage8_population(plan, bank, mask_hashes)
    pop2 = build_stage8_population(plan, bank, mask_hashes)
    ids1 = {(k, tuple(a.manifest.perturbation_id for a in v)) for k, v in pop1.items()}
    ids2 = {(k, tuple(a.manifest.perturbation_id for a in v)) for k, v in pop2.items()}
    assert ids1 == ids2


def test_population_requires_seed_bank_for_every_region():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    with pytest.raises(ValueError):
        build_stage8_population(plan, {"vision": tuple(range(64))}, mask_hashes)


def test_validate_stage8_direction_seed_reuse_passes_for_a_correct_population():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out", n_directions_per_cell=4)
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage8_population(plan, bank, mask_hashes)
    validate_stage8_direction_seed_reuse(plan, population)  # must not raise


def test_validate_stage8_direction_seed_reuse_detects_wrong_repeat_count():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out", n_directions_per_cell=4)
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = dict(build_stage8_population(plan, bank, mask_hashes))
    # Drop one radius cell for "vision" entirely -- its seeds now only repeat twice, not 3x.
    del population[("vision", plan.radii[-1])]
    with pytest.raises(DirectionSeedReuseViolationError):
        validate_stage8_direction_seed_reuse(plan, population)


# =================================================================================================
# 4. Checkpoint identity
# =================================================================================================


def _fake_contexts(n=5):
    contexts = {}
    for capability in STAGE8_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=[], partition=partition, subset_hash=partition.manifest_hash, base_score=0.5)
    return contexts


def _region_mask_hashes(plan):
    return {r: f"hash_{r}" for r in plan.regions}


def test_checkpoint_manifest_includes_direction_seed_bank_hash():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    checkpoint = build_stage8_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank)
    assert checkpoint.direction_seed_bank_hash == compute_direction_seed_bank_hash(bank)
    assert checkpoint.expected_unique_perturbations == 576
    assert checkpoint.expected_result_rows == 3456


def test_checkpoint_manifest_round_trips_through_json():
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root="out")
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    checkpoint = build_stage8_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank)
    restored = Stage8CheckpointManifest.from_dict(checkpoint.to_dict())
    assert restored == checkpoint


def test_ensure_stage8_checkpoint_manifest_creates_when_absent(tmp_path):
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root=str(tmp_path))
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    checkpoint = build_stage8_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank)
    path = tmp_path / "checkpoint_manifest.json"
    result = ensure_stage8_checkpoint_manifest(path, checkpoint)
    assert path.exists()
    assert result == checkpoint


def test_ensure_stage8_checkpoint_manifest_hard_fails_on_seed_bank_mismatch(tmp_path):
    plan = build_stage8_plan(model_name="m", model_revision="rev1", output_root=str(tmp_path))
    bank_a = build_stage8_direction_seed_bank(1, plan.regions, plan.n_directions_per_cell)
    bank_b = build_stage8_direction_seed_bank(2, plan.regions, plan.n_directions_per_cell)
    checkpoint_a = build_stage8_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank_a)
    checkpoint_b = build_stage8_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank_b)
    path = tmp_path / "checkpoint_manifest.json"
    ensure_stage8_checkpoint_manifest(path, checkpoint_a)
    with pytest.raises(IncompatibleStage8CheckpointError):
        ensure_stage8_checkpoint_manifest(path, checkpoint_b)


def test_ensure_stage8_checkpoint_manifest_hard_fails_on_model_revision_mismatch(tmp_path):
    plan_a = build_stage8_plan(model_name="m", model_revision="rev1", output_root=str(tmp_path))
    plan_b = build_stage8_plan(model_name="m", model_revision="rev2", output_root=str(tmp_path))
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan_a.regions, plan_a.n_directions_per_cell)
    checkpoint_a = build_stage8_checkpoint_manifest(plan_a, _fake_contexts(), _region_mask_hashes(plan_a), bank)
    checkpoint_b = build_stage8_checkpoint_manifest(plan_b, _fake_contexts(), _region_mask_hashes(plan_b), bank)
    path = tmp_path / "checkpoint_manifest.json"
    ensure_stage8_checkpoint_manifest(path, checkpoint_a)
    with pytest.raises(IncompatibleStage8CheckpointError):
        ensure_stage8_checkpoint_manifest(path, checkpoint_b)


# =================================================================================================
# 5. Baseline repeatability preflight
# =================================================================================================


class _FakeBaselineEngine:
    def __init__(self):
        self.calls = []


class _FakeRunResultDeterministic:
    def __init__(self, primary_metric=0.5, gen="samehash", parsed="sameparsed"):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}
        self._gen = gen
        self._parsed = parsed

    def generation_hash(self):
        return self._gen

    def parsed_prediction_hash(self):
        return self._parsed


def _fake_reset_to_base_weights_via_rpc(engine, *, ray_get=None):
    engine.calls.append("reset_to_base_weights_via_rpc")


def test_baseline_preflight_passes_when_both_passes_agree(monkeypatch):
    monkeypatch.setattr(module, "reset_to_base_weights_via_rpc", _fake_reset_to_base_weights_via_rpc)
    engine = _FakeBaselineEngine()
    contexts = _fake_contexts(n=2)

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params):
        return _FakeRunResultDeterministic()

    report = run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert set(report.keys()) == set(STAGE8_CAPABILITIES)
    for cap, r in report.items():
        assert r["deterministic"] is True
    ensure_baseline_repeatability(report)  # must not raise


def test_baseline_preflight_evaluates_twice_per_capability_with_reset_between(monkeypatch):
    monkeypatch.setattr(module, "reset_to_base_weights_via_rpc", _fake_reset_to_base_weights_via_rpc)
    engine = _FakeBaselineEngine()
    contexts = _fake_contexts(n=1)
    call_count = {"n": 0}

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params):
        call_count["n"] += 1
        return _FakeRunResultDeterministic()

    run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert call_count["n"] == 2 * len(STAGE8_CAPABILITIES)
    assert engine.calls.count("reset_to_base_weights_via_rpc") == 2 * len(STAGE8_CAPABILITIES)
    assert engine.calls.count("reset_vllm_encoder_cache_full") == 2 * len(STAGE8_CAPABILITIES)


def test_baseline_preflight_hard_fails_when_one_capability_disagrees(monkeypatch):
    monkeypatch.setattr(module, "reset_to_base_weights_via_rpc", _fake_reset_to_base_weights_via_rpc)
    engine = _FakeBaselineEngine()
    contexts = _fake_contexts(n=1)
    call_count = {"n": 0}

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params):
        call_count["n"] += 1
        # Make fine_grained_recognition's SECOND pass disagree -- named in the failure message,
        # matching the spec's "especially important for fine_grained_recognition" note.
        if benchmark == contexts["fine_grained_recognition"].benchmark and call_count["n"] % 2 == 0:
            return _FakeRunResultDeterministic(primary_metric=0.9, gen="differenthash")
        return _FakeRunResultDeterministic()

    report = run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert report["fine_grained_recognition"]["deterministic"] is False
    with pytest.raises(BaselineNondeterminismError, match="fine_grained_recognition"):
        ensure_baseline_repeatability(report)


def test_ensure_baseline_repeatability_never_silently_averages():
    """A hand-built report with mismatched scores for one capability must raise, never be
    silently passed through by any averaging/tolerance logic -- there is none.
    """
    report = {
        "visual_grounding": {"deterministic": True, "score_match": True, "generation_hash_match": True, "parsed_prediction_hash_match": True},
        "counting": {"deterministic": False, "score_match": False, "generation_hash_match": True, "parsed_prediction_hash_match": True},
    }
    with pytest.raises(BaselineNondeterminismError):
        ensure_baseline_repeatability(report)


# =================================================================================================
# 6. Real-tensor candidate lifecycle (mirrors test_run_stage7b_anatomical_calibration.py)
# =================================================================================================


class _FakeStage8Engine:
    def __init__(self, model):
        self._model = model
        self._base_weights = None
        self.calls = []
        self._worker_self = SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            reset_to_base_weights=self._reset_to_base_weights,
            _should_perturb=should_perturb,
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


class _FakeStage8RunResult:
    def __init__(self, primary_metric=0.6):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}

    def generation_hash(self):
        return "fakehash"


def _fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params):
    return _FakeStage8RunResult(primary_metric=0.6)


def _build_assignment_and_engine(model, *, region="language", radius=0.05, seed=42, direction_index=0):
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names = atlas.region(region).param_names
    from neural_thickets_repro.thicket.perturbation import PerturbationManifest

    manifest = PerturbationManifest(
        seed=seed, perturbation_mode=PERTURBATION_MODE, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash=atlas.region(region).mask_hash,
        anatomy_region=region, radius=radius, sigma=None,
    )
    assignment = module.Stage8DirectionAssignment(manifest=manifest, region=region, direction_index=direction_index, direction_seed=seed)
    engine = _FakeStage8Engine(model)
    engine.store_base_weights()
    return assignment, engine, region_param_names


def test_evaluate_one_stage8_candidate_rpc_produces_a_record_per_capability(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()

    records = evaluate_one_stage8_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert len(records) == 6
    assert {r.capability for r in records} == set(STAGE8_CAPABILITIES)
    for r in records:
        assert r.experiment_id == module.EXPERIMENT_ID
        assert r.anatomy_region == "language"
        assert r.perturbed_score == pytest.approx(0.6)
        assert r.runtime_metadata["direction_family_id"] == "language:0"
        assert r.runtime_metadata["direction_seed"] == 42
        assert r.runtime_metadata["direction_index"] == 0
        assert r.runtime_metadata["region"] == "language"
        assert r.runtime_metadata["cache_reset_after_restoration"] is True


def test_evaluate_one_stage8_candidate_rpc_restores_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    base_snapshot = {k: v.clone() for k, v in engine._base_weights.items()}
    contexts = _fake_contexts()

    evaluate_one_stage8_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_snapshot[name])


def test_evaluate_one_stage8_candidate_rpc_resets_cache_twice(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()
    engine.calls.clear()

    evaluate_one_stage8_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert engine.calls[-3:] == ["reset_to_base_weights", "verify_exact_fixed_base_restoration_rpc", "reset_vllm_encoder_cache_full"]
    assert engine.calls.count("reset_vllm_encoder_cache_full") == 2


def test_evaluate_one_stage8_candidate_rpc_never_touches_outside_region(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model, region="vision")
    contexts = _fake_contexts()
    vision_names = set(region_param_names)
    outside_before = {n: p.detach().clone() for n, p in model.named_parameters() if n not in vision_names}

    evaluate_one_stage8_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    for name, p in model.named_parameters():
        if name not in vision_names:
            assert torch.equal(p.detach(), outside_before[name])


def test_evaluate_one_stage8_candidate_rpc_rejects_wrong_perturbation_mode(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    from neural_thickets_repro.thicket.perturbation import PerturbationManifest

    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    bad_manifest = PerturbationManifest(
        seed=1, perturbation_mode="global_gaussian_upstream", model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="h", anatomy_region=None, radius=None, sigma=0.001,
    )
    assignment = module.Stage8DirectionAssignment(manifest=bad_manifest, region="language", direction_index=0, direction_seed=1)
    engine = _FakeStage8Engine(model)
    engine.store_base_weights()

    with pytest.raises(ValueError):
        evaluate_one_stage8_candidate_rpc(
            engine, assignment, atlas.region("language").param_names, _fake_contexts(), tokenizer=None,
            sampling_params=None, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )


# =================================================================================================
# 7. run_stage8_rpc: checkpoint-only-after-full-success
# =================================================================================================


def test_run_stage8_rpc_persists_rows_only_after_full_success(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names_by_region = {"language": atlas.region("language").param_names}
    mask_hashes = {"language": atlas.region("language").mask_hash}

    plan = module.build_stage8_plan(
        model_name="qwen2_5_vl_3b", model_revision="rev1", output_root=str(tmp_path),
        regions=("language",), radii=(0.05,), n_directions_per_cell=2, d_map_n=5,
    )
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage8Engine(model)
    engine.store_base_weights()

    records = run_stage8_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert len(records) == 2 * 6  # 2 directions x 6 capabilities
    results_path = plan.output_dir / "results.jsonl"
    assert results_path.exists()
    lines = results_path.read_text().strip().split("\n")
    assert len(lines) == 12
    checkpoint_path = plan.output_dir / "checkpoint_manifest.json"
    assert checkpoint_path.exists()


def test_run_stage8_rpc_resumes_skipping_completed_candidates(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names_by_region = {"language": atlas.region("language").param_names}
    mask_hashes = {"language": atlas.region("language").mask_hash}

    plan = module.build_stage8_plan(
        model_name="qwen2_5_vl_3b", model_revision="rev1", output_root=str(tmp_path),
        regions=("language",), radii=(0.05,), n_directions_per_cell=2, d_map_n=5,
    )
    bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage8Engine(model)
    engine.store_base_weights()

    run_stage8_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    call_count_before = len(engine.calls)

    # Second call against the SAME output_dir must find everything already complete and do no new work.
    records_again = run_stage8_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert len(records_again) == 12
    results_path = plan.output_dir / "results.jsonl"
    assert len(results_path.read_text().strip().split("\n")) == 12  # never duplicated


# =================================================================================================
# 8. No best-radius / no capability-optimization selection logic
# =================================================================================================


def test_no_best_radius_or_capability_selection_logic_exists():
    import inspect

    source = inspect.getsource(module)
    for forbidden in ("best_radius", "select_best", "optimal_radius", "top_capability"):
        assert forbidden not in source
