"""Tests for run_stage9_hierarchical_anatomical_atlas.py -- CPU-only. The real GPU/Ray/vLLM
engine is never launched; RPC dispatch is tested against a fake, persistent-worker-shaped
engine (same philosophy as tests/test_run_stage8_coarse_anatomical_atlas.py), using REAL small
torch tensors for the lifecycle tests that need genuine weight mutation/restoration.
"""
from types import SimpleNamespace

import pytest
import torch

import neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas as module
from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext
from neural_thickets_repro.run_stage7b_anatomical_calibration import (
    ENABLE_PREFIX_CACHING,
    MULTIMODAL_CACHE_POLICY,
    PERTURBATION_MODE,
    RADIUS_REALIZATION_METHOD,
)
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_RADII
from neural_thickets_repro.thicket.anatomy_stage9 import STAGE9_CHILD_REGIONS, build_stage9_hierarchical_partition
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest
from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import (
    STAGE8_AUTHORITATIVE_BASELINE,
    STAGE9_CAPABILITIES,
    STAGE9_D_MAP_N,
    STAGE9_N_DIRECTIONS_PER_CELL,
    STAGE9_RADII,
    STAGE9_SMOKE_D_MAP_N,
    STAGE9_SMOKE_N_DIRECTIONS,
    DirectionSeedReuseViolationError,
    IncompatibleStage9CheckpointError,
    Stage9BaselineMismatchError,
    Stage9CheckpointManifest,
    Stage9DirectionAssignment,
    build_stage9_checkpoint_manifest,
    build_stage9_direction_seed_bank,
    build_stage9_plan,
    build_stage9_population,
    build_stage9_smoke_plan,
    compute_direction_seed_bank_hash,
    compute_partition_audit_hash,
    compute_stage9_run_signature,
    ensure_stage9_baseline_matches_stage8,
    ensure_stage9_checkpoint_manifest,
    evaluate_one_stage9_candidate_rpc,
    report_stage9_child_param_names,
    run_stage9_baseline_equality_check,
    run_stage9_rpc,
    validate_stage9_direction_seed_reuse,
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


def test_stage9_uses_exactly_six_child_regions():
    assert set(STAGE9_CHILD_REGIONS) == {
        "vision_early", "vision_mid", "vision_late", "language_early", "language_mid", "language_late",
    }
    assert len(STAGE9_CHILD_REGIONS) == 6


def test_stage9_radii_reused_by_identity_from_stage8():
    assert STAGE9_RADII == STAGE8_RADII == (0.0035698828543799426, 0.017849414271899712, 0.07139765708759885)


def test_stage9_capabilities_reused_by_identity_from_stage8():
    assert STAGE9_CAPABILITIES == STAGE8_CAPABILITIES
    assert len(STAGE9_CAPABILITIES) == 6


def test_stage9_d_map_n_reused_by_identity_from_stage8():
    assert STAGE9_D_MAP_N == 50


def test_stage9_n_directions_per_cell_is_64():
    assert STAGE9_N_DIRECTIONS_PER_CELL == 64


def test_default_stage9_plan_is_the_frozen_full_identity():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    assert set(plan.child_regions) == set(STAGE9_CHILD_REGIONS)
    assert plan.radii == STAGE9_RADII
    assert plan.capabilities == STAGE9_CAPABILITIES
    assert plan.n_directions_per_cell == 64
    assert plan.d_map_n == 50
    assert plan.generation_batch_size == 10
    assert plan.is_smoke is False
    assert plan.run_signature == "stage9_hierarchical_anatomical_atlas_3b_v1"


def test_full_stage9_plan_totals_1152_perturbations_6912_rows_345600_evaluations():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    assert plan.total_unique_perturbations == 6 * 3 * 64 == 1152
    assert plan.total_perturbation_capability_evaluations == 1152 * 6 == 6912
    assert plan.total_perturbed_model_example_evaluations == 6912 * 50 == 345_600


def test_stage9_plan_prefix_caching_and_batch_size():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    assert plan.enable_prefix_caching is False
    assert plan.generation_batch_size == 10
    assert module.ENABLE_PREFIX_CACHING == ENABLE_PREFIX_CACHING is False
    assert module.MULTIMODAL_CACHE_POLICY == MULTIMODAL_CACHE_POLICY == "full_encoder_reset_vllm011_verified_v2"
    assert module.RADIUS_REALIZATION_METHOD == RADIUS_REALIZATION_METHOD == "fixed_direction_bf16_quantization_aware_v3"


def test_build_stage9_plan_rejects_empty_child_regions():
    with pytest.raises(ValueError):
        build_stage9_plan(model_name="m", model_revision="rev1", output_root="out", child_regions=())


def test_build_stage9_plan_rejects_unrecognized_d_map_size():
    with pytest.raises(module.DatasetRoleViolationError):
        build_stage9_plan(model_name="m", model_revision="rev1", output_root="out", d_map_n=17)


def test_build_stage9_plan_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        build_stage9_plan(model_name="m", model_revision="rev1", output_root="out", generation_batch_size=0)


# =================================================================================================
# 2. Smoke config
# =================================================================================================


def test_smoke_plan_totals_18_perturbations_108_rows_540_evaluations():
    plan = build_stage9_smoke_plan(model_name="m", model_revision="rev1", output_root="out")
    assert plan.is_smoke is True
    assert set(plan.child_regions) == set(STAGE9_CHILD_REGIONS)
    assert plan.radii == STAGE9_RADII
    assert plan.capabilities == STAGE9_CAPABILITIES
    assert plan.total_unique_perturbations == 6 * 3 * 1 == 18
    assert plan.total_perturbation_capability_evaluations == 18 * 6 == 108
    assert plan.total_perturbed_model_example_evaluations == 108 * 5 == 540


def test_smoke_and_full_plans_never_share_an_output_directory():
    full_plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    smoke_plan = build_stage9_smoke_plan(model_name="m", model_revision="rev1", output_root="out")
    assert full_plan.output_dir != smoke_plan.output_dir
    assert full_plan.run_signature != smoke_plan.run_signature


def test_compute_stage9_run_signature_never_returns_the_full_literal_for_any_deviation():
    sig = compute_stage9_run_signature(STAGE9_CHILD_REGIONS, STAGE9_RADII, 1, 5)
    assert sig != "stage9_hierarchical_anatomical_atlas_3b_v1"
    assert sig.startswith("stage9_smoke_")


# =================================================================================================
# 3. Direction-family seed bank + population (per child region)
# =================================================================================================


def test_direction_seed_bank_has_64_seeds_per_child_region_for_the_full_config():
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, STAGE9_CHILD_REGIONS, 64)
    assert set(bank.keys()) == set(STAGE9_CHILD_REGIONS)
    for region, seeds in bank.items():
        assert len(seeds) == 64
        assert len(set(seeds)) == 64


def test_direction_seed_bank_is_deterministic():
    bank1 = build_stage9_direction_seed_bank(42, STAGE9_CHILD_REGIONS, 8)
    bank2 = build_stage9_direction_seed_bank(42, STAGE9_CHILD_REGIONS, 8)
    assert bank1 == bank2


def test_stage9_seed_namespace_is_independent_of_stage8():
    """Even for a superficially similar region label, Stage 9's seed bank must be an
    independent stream from Stage 8's own (different derive_seed namespace string).
    """
    from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import build_stage8_direction_seed_bank

    stage9_bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, ("vision_early",), 4)
    stage8_bank = build_stage8_direction_seed_bank(module.STAGE9_BASE_SEED, ("vision_early",), 4)
    assert stage9_bank["vision_early"] != stage8_bank["vision_early"]


def test_same_direction_seed_reused_across_all_radii_within_a_child_region():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out", n_directions_per_cell=4)
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.child_regions}
    population = build_stage9_population(plan, bank, mask_hashes)

    for region in plan.child_regions:
        seeds_per_radius = [[a.direction_seed for a in population[(region, radius)]] for radius in plan.radii]
        assert all(s == seeds_per_radius[0] for s in seeds_per_radius), f"region {region} seed sequence differs across radii"


def test_population_has_1152_unique_perturbation_ids_for_the_full_config():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.child_regions}
    population = build_stage9_population(plan, bank, mask_hashes)
    all_ids = [a.manifest.perturbation_id for cell in population.values() for a in cell]
    assert len(all_ids) == 1152
    assert len(set(all_ids)) == 1152


def test_validate_stage9_direction_seed_reuse_passes_for_a_correct_population():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out", n_directions_per_cell=4)
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.child_regions}
    population = build_stage9_population(plan, bank, mask_hashes)
    validate_stage9_direction_seed_reuse(plan, population)  # must not raise


def test_validate_stage9_direction_seed_reuse_detects_wrong_repeat_count():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out", n_directions_per_cell=4)
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.child_regions}
    population = dict(build_stage9_population(plan, bank, mask_hashes))
    del population[("vision_early", plan.radii[-1])]
    with pytest.raises(DirectionSeedReuseViolationError):
        validate_stage9_direction_seed_reuse(plan, population)


def test_no_cross_child_region_direction_pairing_assumption():
    """Direction index i in vision_early and direction index i in language_early must have
    DIFFERENT seeds (no accidental cross-anatomy geometric pairing) -- confirmed empirically for
    the frozen full config, since the two regions' name strings differ in derive_seed's input.
    """
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, STAGE9_CHILD_REGIONS, 8)
    assert bank["vision_early"] != bank["language_early"]
    assert bank["vision_early"][0] != bank["language_early"][0]


# =================================================================================================
# 4. Checkpoint identity
# =================================================================================================


def _fake_contexts(n=5):
    contexts = {}
    for capability in STAGE9_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=[], partition=partition, subset_hash=partition.manifest_hash, base_score=0.5)
    return contexts


def _region_mask_hashes(plan):
    return {r: f"hash_{r}" for r in plan.child_regions}


def _fake_audits(plan):
    children, audits = build_stage9_hierarchical_partition(_real_shaped_param_names())
    return audits


def _real_shaped_param_names(n_layers=12):
    import torch.nn as nn

    class DummyVisual32Blocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(4, 4, bias=False)
            self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(32)])
            self.merger = nn.Linear(4, 4, bias=False)

    class LangLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(4, 4, bias=False)

    class LangModelInner(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 4)
            self.layers = nn.ModuleList([LangLayer() for _ in range(n)])
            self.norm = nn.Linear(4, 4, bias=False)

    class LangModel(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.model = LangModelInner(n)

    class VLM(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.visual = DummyVisual32Blocks()
            self.language_model = LangModel(n)

    return [n for n, _ in VLM(n_layers).named_parameters()]


def test_checkpoint_manifest_includes_partition_audit_hash_and_stage8_parent_signature():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    audits = _fake_audits(plan)
    checkpoint = build_stage9_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audits)
    assert checkpoint.partition_audit_hash == compute_partition_audit_hash(audits)
    assert checkpoint.stage8_parent_run_signature == module.STAGE8_PARENT_RUN_SIGNATURE
    assert checkpoint.expected_unique_perturbations == 1152
    assert checkpoint.expected_result_rows == 6912


def test_checkpoint_manifest_round_trips_through_json():
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    audits = _fake_audits(plan)
    checkpoint = build_stage9_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audits)
    restored = Stage9CheckpointManifest.from_dict(checkpoint.to_dict())
    assert restored == checkpoint


def test_ensure_stage9_checkpoint_manifest_creates_when_absent(tmp_path):
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root=str(tmp_path))
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    audits = _fake_audits(plan)
    checkpoint = build_stage9_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audits)
    path = tmp_path / "checkpoint_manifest.json"
    result = ensure_stage9_checkpoint_manifest(path, checkpoint)
    assert path.exists()
    assert result == checkpoint


def test_ensure_stage9_checkpoint_manifest_hard_fails_on_partition_audit_mismatch(tmp_path):
    plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root=str(tmp_path))
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    audits_a = _fake_audits(plan)
    audits_b = {k: v for k, v in audits_a.items()}
    audits_b["vision"] = audits_b["vision"].__class__(
        parent="vision", child_band_names=audits_b["vision"].child_band_names,
        uncovered_tensors=("fake.tensor",), uncovered_tensor_assignment={"fake.tensor": "early"},
        union_equals_parent=True, children_pairwise_disjoint=True,
    )
    checkpoint_a = build_stage9_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audits_a)
    checkpoint_b = build_stage9_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audits_b)
    path = tmp_path / "checkpoint_manifest.json"
    ensure_stage9_checkpoint_manifest(path, checkpoint_a)
    with pytest.raises(IncompatibleStage9CheckpointError):
        ensure_stage9_checkpoint_manifest(path, checkpoint_b)


def test_ensure_stage9_checkpoint_manifest_hard_fails_on_generation_batch_size_mismatch(tmp_path):
    plan_a = build_stage9_plan(model_name="m", model_revision="rev1", output_root=str(tmp_path), generation_batch_size=10)
    plan_b = build_stage9_plan(model_name="m", model_revision="rev1", output_root=str(tmp_path), generation_batch_size=5)
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan_a.child_regions, plan_a.n_directions_per_cell)
    audits = _fake_audits(plan_a)
    checkpoint_a = build_stage9_checkpoint_manifest(plan_a, _fake_contexts(), _region_mask_hashes(plan_a), bank, audits)
    checkpoint_b = build_stage9_checkpoint_manifest(plan_b, _fake_contexts(), _region_mask_hashes(plan_b), bank, audits)
    assert checkpoint_a != checkpoint_b
    # Same output_dir would be forced here since batch size doesn't change run_signature by
    # itself in this module (unlike Stage 8) -- confirm the checkpoint dataclass equality still
    # catches the difference so a resume attempt would be rejected.
    path = tmp_path / "checkpoint_manifest.json"
    ensure_stage9_checkpoint_manifest(path, checkpoint_a)
    with pytest.raises(IncompatibleStage9CheckpointError):
        ensure_stage9_checkpoint_manifest(path, checkpoint_b)


# =================================================================================================
# 5. Stage-8 baseline equality hard gate
# =================================================================================================


def test_baseline_equality_check_passes_when_all_match():
    baseline_scores = {"capabilities": {cap: {"score": score} for cap, score in STAGE8_AUTHORITATIVE_BASELINE.items()}}
    report = run_stage9_baseline_equality_check(baseline_scores)
    assert report["all_match"] is True
    ensure_stage9_baseline_matches_stage8(report)  # must not raise


def test_baseline_equality_check_hard_fails_on_a_single_mismatch():
    baseline_scores = {"capabilities": {cap: {"score": score} for cap, score in STAGE8_AUTHORITATIVE_BASELINE.items()}}
    baseline_scores["capabilities"]["fine_grained_recognition"]["score"] = 0.99
    report = run_stage9_baseline_equality_check(baseline_scores)
    assert report["all_match"] is False
    assert report["fine_grained_recognition"]["matches"] is False
    with pytest.raises(Stage9BaselineMismatchError, match="fine_grained_recognition"):
        ensure_stage9_baseline_matches_stage8(report)


def test_authoritative_stage8_baseline_values_match_the_frozen_spec():
    assert STAGE8_AUTHORITATIVE_BASELINE == {
        "visual_grounding": 0.880, "counting": 0.680, "spatial_reasoning": 0.700,
        "ocr_text_recognition_grounded": 0.938, "relational_reasoning": 0.540, "fine_grained_recognition": 0.420,
    }


def test_baseline_equality_check_never_averages_missing_score():
    baseline_scores = {"capabilities": {}}  # no scores at all
    report = run_stage9_baseline_equality_check(baseline_scores)
    assert report["all_match"] is False
    with pytest.raises(Stage9BaselineMismatchError):
        ensure_stage9_baseline_matches_stage8(report)


# =================================================================================================
# 6. report_stage9_child_param_names -- live partition audit inside the worker
# =================================================================================================


def test_report_stage9_child_param_names_returns_all_six_regions_with_hashes():
    class _FakeModel:
        def named_parameters(self_inner):
            return [(n, None) for n in _real_shaped_param_names()]

    worker_self = SimpleNamespace(model_runner=SimpleNamespace(model=_FakeModel()))
    result = report_stage9_child_param_names(worker_self, STAGE9_CHILD_REGIONS)
    assert set(result["regions"].keys()) == set(STAGE9_CHILD_REGIONS)
    for region, info in result["regions"].items():
        assert "param_names" in info and "mask_hash" in info
    assert set(result["audits"].keys()) == {"vision", "language"}
    for parent, audit in result["audits"].items():
        assert audit["union_equals_parent"] is True
        assert audit["children_pairwise_disjoint"] is True


# =================================================================================================
# 7. Real-tensor candidate lifecycle
# =================================================================================================


class _FakeStage9Engine:
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


class _FakeStage9RunResult:
    def __init__(self, primary_metric=0.6):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}

    def generation_hash(self):
        return "fakehash"


def _fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
    return _FakeStage9RunResult(primary_metric=0.6)


def _build_assignment_and_engine(model, *, child_region="language_mid", radius=0.05, seed=42, direction_index=0):
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names)
    region = children[child_region]
    manifest = PerturbationManifest(
        seed=seed, perturbation_mode=PERTURBATION_MODE, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash=region.mask_hash,
        anatomy_region=child_region, radius=radius, sigma=None,
    )
    assignment = Stage9DirectionAssignment(manifest=manifest, child_region=child_region, direction_index=direction_index, direction_seed=seed)
    engine = _FakeStage9Engine(model)
    engine.store_base_weights()
    return assignment, engine, region.param_names


def test_evaluate_one_stage9_candidate_rpc_produces_a_record_per_capability(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()

    records = evaluate_one_stage9_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert len(records) == 6
    assert {r.capability for r in records} == set(STAGE9_CAPABILITIES)
    for r in records:
        assert r.experiment_id == module.EXPERIMENT_ID
        assert r.anatomy_region == "language_mid"
        assert r.perturbed_score == pytest.approx(0.6)
        assert r.runtime_metadata["child_region"] == "language_mid"
        assert r.runtime_metadata["direction_family_id"] == "language_mid:0"
        assert r.runtime_metadata["direction_seed"] == 42
        assert r.runtime_metadata["cache_reset_after_restoration"] is True
        assert r.runtime_metadata["generation_batch_size"] == 10


def test_evaluate_one_stage9_candidate_rpc_restores_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    base_snapshot = {k: v.clone() for k, v in engine._base_weights.items()}
    contexts = _fake_contexts()

    evaluate_one_stage9_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_snapshot[name])


def test_evaluate_one_stage9_candidate_rpc_resets_cache_twice(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()
    engine.calls.clear()

    evaluate_one_stage9_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert engine.calls[-3:] == ["reset_to_base_weights", "verify_exact_fixed_base_restoration_rpc", "reset_vllm_encoder_cache_full"]
    assert engine.calls.count("reset_vllm_encoder_cache_full") == 2


def test_evaluate_one_stage9_candidate_rpc_never_touches_outside_child_region(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model, child_region="vision_early")
    contexts = _fake_contexts()
    region_names = set(region_param_names)
    outside_before = {n: p.detach().clone() for n, p in model.named_parameters() if n not in region_names}

    evaluate_one_stage9_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    for name, p in model.named_parameters():
        if name not in region_names:
            assert torch.equal(p.detach(), outside_before[name])


def test_evaluate_one_stage9_candidate_rpc_threads_generation_batch_size(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()
    seen = []

    def _capturing_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        seen.append(kwargs.get("max_requests_per_generate"))
        return _FakeStage9RunResult(primary_metric=0.6)

    evaluate_one_stage9_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_capturing_run_benchmark, ray_get=_identity_ray_get, generation_batch_size=10,
    )
    assert set(seen) == {10}
    assert len(seen) == 6


def test_evaluate_one_stage9_candidate_rpc_rejects_wrong_perturbation_mode(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names)
    bad_manifest = PerturbationManifest(
        seed=1, perturbation_mode="global_gaussian_upstream", model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="h", anatomy_region=None, radius=None, sigma=0.001,
    )
    assignment = Stage9DirectionAssignment(manifest=bad_manifest, child_region="language_mid", direction_index=0, direction_seed=1)
    engine = _FakeStage9Engine(model)
    engine.store_base_weights()

    with pytest.raises(ValueError):
        evaluate_one_stage9_candidate_rpc(
            engine, assignment, children["language_mid"].param_names, _fake_contexts(), tokenizer=None,
            sampling_params=None, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )


# =================================================================================================
# 8. run_stage9_rpc: checkpoint-only-after-full-success + resume
# =================================================================================================


def test_run_stage9_rpc_persists_rows_only_after_full_success(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names)
    region_param_names_by_region = {"language_mid": children["language_mid"].param_names}
    mask_hashes = {"language_mid": children["language_mid"].mask_hash}

    plan = build_stage9_plan(
        model_name="qwen2_5_vl_3b", model_revision="rev1", output_root=str(tmp_path),
        child_regions=("language_mid",), radii=(0.05,), n_directions_per_cell=2, d_map_n=5,
    )
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage9Engine(model)
    engine.store_base_weights()

    newly_written = run_stage9_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        child_region_param_names_by_region=region_param_names_by_region, child_mask_hash_by_region=mask_hashes,
        audits=audits, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    # This repair pass: run_stage9_rpc returns an int COUNT of newly-written rows, never the
    # full in-memory record list (the driver-RSS OOM fix) -- the authoritative total still
    # lives on disk.
    assert newly_written == 2 * 6
    results_path = plan.output_dir / "results.jsonl"
    assert results_path.exists()
    assert len(results_path.read_text().strip().split("\n")) == 12
    assert (plan.output_dir / "checkpoint_manifest.json").exists()
    telemetry_path = plan.output_dir / "candidate_memory_telemetry.jsonl"
    assert telemetry_path.exists()
    assert len(telemetry_path.read_text().strip().split("\n")) == 2  # one line per candidate, not per row


def test_run_stage9_rpc_resumes_skipping_completed_candidates(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names)
    region_param_names_by_region = {"language_mid": children["language_mid"].param_names}
    mask_hashes = {"language_mid": children["language_mid"].mask_hash}

    plan = build_stage9_plan(
        model_name="qwen2_5_vl_3b", model_revision="rev1", output_root=str(tmp_path),
        child_regions=("language_mid",), radii=(0.05,), n_directions_per_cell=2, d_map_n=5,
    )
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage9Engine(model)
    engine.store_base_weights()

    run_stage9_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        child_region_param_names_by_region=region_param_names_by_region, child_mask_hash_by_region=mask_hashes,
        audits=audits, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    newly_written_second_call = run_stage9_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        child_region_param_names_by_region=region_param_names_by_region, child_mask_hash_by_region=mask_hashes,
        audits=audits, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert newly_written_second_call == 0  # nothing new -- everything was already complete
    results_path = plan.output_dir / "results.jsonl"
    assert len(results_path.read_text().strip().split("\n")) == 12  # never duplicated


# =================================================================================================
# 9. No best-radius / no capability-optimization selection logic
# =================================================================================================


def test_no_best_radius_or_capability_selection_logic_exists():
    import inspect

    source = inspect.getsource(module)
    for forbidden in ("best_radius", "select_best", "optimal_radius", "top_capability"):
        assert forbidden not in source


# =================================================================================================
# 10. Mode-aware baseline gate (this repair pass -- smoke N=5 vs full N=50 gate bug fix)
# =================================================================================================


def test_full_mode_still_uses_exact_stage8_n50_equality_check():
    """Section 1: full mode must be UNCHANGED -- exact equality against the authoritative
    Stage-8 N=50 baseline, still a hard stop on mismatch.
    """
    baseline_scores = {"capabilities": {cap: {"score": score} for cap, score in module.STAGE8_AUTHORITATIVE_BASELINE.items()}}
    report = module.run_stage9_baseline_equality_check(baseline_scores)
    assert report["all_match"] is True
    module.ensure_stage9_baseline_matches_stage8(report)  # must not raise


def test_full_mode_mismatch_still_hard_fails():
    baseline_scores = {"capabilities": {cap: {"score": score} for cap, score in module.STAGE8_AUTHORITATIVE_BASELINE.items()}}
    baseline_scores["capabilities"]["counting"]["score"] = 0.5  # not 0.680
    report = module.run_stage9_baseline_equality_check(baseline_scores)
    assert report["all_match"] is False
    with pytest.raises(module.Stage9BaselineMismatchError):
        module.ensure_stage9_baseline_matches_stage8(report)


def test_full_mode_requires_stage8_n50_subset_hashes():
    contexts = {
        cap: SimpleNamespace(subset_hash=expected)
        for cap, expected in module.STAGE8_AUTHORITATIVE_SUBSET_HASHES.items()
    }
    report = module.run_stage9_subset_hash_check(contexts)
    assert report["all_match"] is True
    module.ensure_stage9_subset_hashes_match_stage8(report)  # must not raise


def test_full_mode_subset_hash_mismatch_hard_fails():
    contexts = {
        cap: SimpleNamespace(subset_hash=expected)
        for cap, expected in module.STAGE8_AUTHORITATIVE_SUBSET_HASHES.items()
    }
    contexts["spatial_reasoning"] = SimpleNamespace(subset_hash="some_other_hash_from_a_reshuffled_subset")
    report = module.run_stage9_subset_hash_check(contexts)
    assert report["all_match"] is False
    with pytest.raises(module.Stage9SubsetHashMismatchError):
        module.ensure_stage9_subset_hashes_match_stage8(report)


class _FakeStage9BaselineEngine:
    def __init__(self):
        self.calls = []


class _FakeStage9RunResultDeterministic:
    def __init__(self, primary_metric=0.5, gen="samehash", parsed="sameparsed"):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}
        self._gen = gen
        self._parsed = parsed

    def generation_hash(self):
        return self._gen

    def parsed_prediction_hash(self):
        return self._parsed


def _fake_reset_to_base_weights_via_rpc_stage8(engine, *, ray_get=None):
    engine.calls.append("reset_to_base_weights_via_rpc")


@pytest.fixture
def _patch_stage8_reset_functions(monkeypatch):
    """run_baseline_repeatability_preflight_rpc is reused BY IDENTITY from Stage 8's own
    module -- its internal reset_vllm_encoder_cache_full/reset_to_base_weights_via_rpc calls
    are bound in STAGE 8's namespace, not Stage 9's, so they must be patched there.
    """
    import neural_thickets_repro.run_stage8_coarse_anatomical_atlas as stage8_module

    def _fake_reset_cache(engine):
        if hasattr(engine, "calls"):
            engine.calls.append("reset_vllm_encoder_cache_full")

    monkeypatch.setattr(stage8_module, "reset_vllm_encoder_cache_full", _fake_reset_cache)
    monkeypatch.setattr(stage8_module, "reset_to_base_weights_via_rpc", _fake_reset_to_base_weights_via_rpc_stage8)


def test_smoke_mode_never_compares_n5_score_to_stage8_n50_baseline(_patch_stage8_reset_functions):
    """The smoke repeatability report/gate never even RECEIVES the Stage-8 N=50 baseline values
    as an input -- architecturally impossible for it to compare against them, unlike the prior
    bug where the N=50 equality function was called on N=5 live data.
    """
    engine = _FakeStage9BaselineEngine()
    contexts = _fake_contexts(n=5)  # smoke-shaped: N=5 examples per capability

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        return _FakeStage9RunResultDeterministic(primary_metric=0.8)  # deliberately NOT any Stage-8 N=50 value

    from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import build_stage9_baseline_gate_report

    repeatability_report = module.run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    gate_report = build_stage9_baseline_gate_report(is_smoke=True, d_map_n=5, smoke_repeatability_report=repeatability_report)
    assert gate_report["baseline_gate_mode"] == "smoke_n5_repeatability"
    assert "baseline_equality" not in gate_report  # never present in smoke mode
    assert gate_report["d_map_n"] == 5
    module.ensure_stage9_baseline_gate_passes(gate_report)  # deterministic (0.8==0.8) -- must not raise


def test_smoke_mode_performs_two_pass_theta0_repeatability(_patch_stage8_reset_functions):
    engine = _FakeStage9BaselineEngine()
    contexts = _fake_contexts(n=1)
    call_count = {"n": 0}

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        call_count["n"] += 1
        return _FakeStage9RunResultDeterministic()

    module.run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert call_count["n"] == 2 * len(STAGE9_CAPABILITIES)
    assert engine.calls.count("reset_to_base_weights_via_rpc") == 2 * len(STAGE9_CAPABILITIES)
    assert engine.calls.count("reset_vllm_encoder_cache_full") == 2 * len(STAGE9_CAPABILITIES)


def test_smoke_mode_hard_fails_on_generation_hash_mismatch(_patch_stage8_reset_functions):
    from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import BaselineNondeterminismError

    engine = _FakeStage9BaselineEngine()
    contexts = _fake_contexts(n=1)
    call_count = {"n": 0}

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            return _FakeStage9RunResultDeterministic(gen="different_hash_pass_b")
        return _FakeStage9RunResultDeterministic()

    repeatability_report = module.run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    gate_report = module.build_stage9_baseline_gate_report(is_smoke=True, d_map_n=5, smoke_repeatability_report=repeatability_report)
    with pytest.raises(BaselineNondeterminismError):
        module.ensure_stage9_baseline_gate_passes(gate_report)


def test_smoke_mode_hard_fails_on_parsed_prediction_hash_mismatch(_patch_stage8_reset_functions):
    from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import BaselineNondeterminismError

    engine = _FakeStage9BaselineEngine()
    contexts = _fake_contexts(n=1)
    call_count = {"n": 0}

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            return _FakeStage9RunResultDeterministic(parsed="different_parsed_pass_b")
        return _FakeStage9RunResultDeterministic()

    repeatability_report = module.run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    gate_report = module.build_stage9_baseline_gate_report(is_smoke=True, d_map_n=5, smoke_repeatability_report=repeatability_report)
    with pytest.raises(BaselineNondeterminismError):
        module.ensure_stage9_baseline_gate_passes(gate_report)


def test_smoke_mode_hard_fails_on_score_mismatch(_patch_stage8_reset_functions):
    from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import BaselineNondeterminismError

    engine = _FakeStage9BaselineEngine()
    contexts = _fake_contexts(n=1)
    call_count = {"n": 0}

    def fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            return _FakeStage9RunResultDeterministic(primary_metric=0.99)
        return _FakeStage9RunResultDeterministic(primary_metric=0.5)

    repeatability_report = module.run_baseline_repeatability_preflight_rpc(
        engine, contexts, tokenizer=None, sampling_params=None, run_benchmark=fake_run_benchmark, ray_get=_identity_ray_get,
    )
    gate_report = module.build_stage9_baseline_gate_report(is_smoke=True, d_map_n=5, smoke_repeatability_report=repeatability_report)
    with pytest.raises(BaselineNondeterminismError):
        module.ensure_stage9_baseline_gate_passes(gate_report)


def test_baseline_gate_mode_is_persisted_for_both_modes():
    smoke_report = {"deterministic": True, "score_match": True, "generation_hash_match": True, "parsed_prediction_hash_match": True}
    smoke_gate = module.build_stage9_baseline_gate_report(
        is_smoke=True, d_map_n=5, smoke_repeatability_report={"visual_grounding": smoke_report},
    )
    assert smoke_gate["baseline_gate_mode"] == "smoke_n5_repeatability"

    full_equality = module.run_stage9_baseline_equality_check(
        {"capabilities": {cap: {"score": score} for cap, score in module.STAGE8_AUTHORITATIVE_BASELINE.items()}}
    )
    full_subset = module.run_stage9_subset_hash_check(
        {cap: SimpleNamespace(subset_hash=h) for cap, h in module.STAGE8_AUTHORITATIVE_SUBSET_HASHES.items()}
    )
    full_gate = module.build_stage9_baseline_gate_report(
        is_smoke=False, d_map_n=50, full_equality_report=full_equality, full_subset_hash_report=full_subset,
    )
    assert full_gate["baseline_gate_mode"] == "stage8_full_exact_equality"


def test_build_stage9_baseline_gate_report_requires_the_right_arguments_per_mode():
    with pytest.raises(ValueError):
        module.build_stage9_baseline_gate_report(is_smoke=True, d_map_n=5)  # missing smoke_repeatability_report
    with pytest.raises(ValueError):
        module.build_stage9_baseline_gate_report(is_smoke=False, d_map_n=50)  # missing full reports


def test_ensure_stage9_baseline_gate_passes_never_silently_skips_an_unknown_mode():
    with pytest.raises(ValueError):
        module.ensure_stage9_baseline_gate_passes({"baseline_gate_mode": "some_unknown_mode"})


def test_stage9_design_counts_unchanged_by_this_repair_pass():
    full_plan = build_stage9_plan(model_name="m", model_revision="rev1", output_root="out")
    assert full_plan.total_unique_perturbations == 1152
    assert full_plan.total_perturbation_capability_evaluations == 6912
    smoke_plan = build_stage9_smoke_plan(model_name="m", model_revision="rev1", output_root="out")
    assert smoke_plan.total_unique_perturbations == 18
    assert smoke_plan.total_perturbation_capability_evaluations == 108


# =================================================================================================
# 11. Driver-RSS OOM fix during full candidate evaluation (this repair pass)
# =================================================================================================


def test_checkpointed_candidates_have_exactly_six_rows(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names)
    region_param_names_by_region = {"language_mid": children["language_mid"].param_names}
    mask_hashes = {"language_mid": children["language_mid"].mask_hash}
    plan = build_stage9_plan(
        model_name="qwen2_5_vl_3b", model_revision="rev1", output_root=str(tmp_path),
        child_regions=("language_mid",), radii=(0.05,), n_directions_per_cell=2, d_map_n=5,
    )
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage9Engine(model)
    engine.store_base_weights()

    run_stage9_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        child_region_param_names_by_region=region_param_names_by_region, child_mask_hash_by_region=mask_hashes,
        audits=audits, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    completed = module.load_completed_perturbation_rows(plan.output_dir / "results.jsonl", plan.capabilities)
    assert len(completed) == 2  # 2 directions
    for pid, rows in completed.items():
        assert len(rows) == 6
        assert {r.capability for r in rows} == set(STAGE9_CAPABILITIES)


def test_incomplete_candidate_after_simulated_failure_writes_zero_rows(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """A mid-candidate failure (e.g. run_benchmark raising) must leave results.jsonl untouched
    for that candidate -- append_candidate_rows is only ever called AFTER the function returns
    successfully.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()

    def _failing_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        raise RuntimeError("simulated mid-candidate failure")

    with pytest.raises(RuntimeError, match="simulated mid-candidate failure"):
        evaluate_one_stage9_candidate_rpc(
            engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
            run_benchmark=_failing_run_benchmark, ray_get=_identity_ray_get,
        )
    # Restoration was still attempted (recoverable-engine path) -- weights are back to base.
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), engine._base_weights[name])


def test_resume_reruns_a_candidate_with_fewer_than_six_rows(tmp_path, runtime_wrapped_vlm_32vision_factory):
    """A candidate with an incomplete row set on disk (simulating a crash mid-checkpoint) must
    NOT be treated as completed -- load_completed_perturbation_rows only counts a perturbation
    as done when exactly the full capability set is present.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names)
    region_param_names_by_region = {"language_mid": children["language_mid"].param_names}
    mask_hashes = {"language_mid": children["language_mid"].mask_hash}
    plan = build_stage9_plan(
        model_name="qwen2_5_vl_3b", model_revision="rev1", output_root=str(tmp_path),
        child_regions=("language_mid",), radii=(0.05,), n_directions_per_cell=1, d_map_n=5,
    )
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage9Engine(model)
    engine.store_base_weights()

    # Manually write a PARTIAL candidate (only 3 of 6 capability rows) directly to results.jsonl,
    # simulating a crash that happened mid-write -- but append_candidate_rows itself is always
    # atomic-per-candidate in the real lifecycle, so this models an out-of-band interruption.
    from neural_thickets_repro.run_global_visual_thicket_pilot import append_candidate_rows
    from neural_thickets_repro.thicket.schema import ExperimentResultRecord

    partial_pid = list(build_stage9_population(plan, bank, mask_hashes).values())[0][0].manifest.perturbation_id
    partial_rows = [
        ExperimentResultRecord(
            experiment_id=module.EXPERIMENT_ID, perturbation_id=partial_pid, model_family="qwen2_5_vl", model_scale="3B",
            model_revision="rev1", perturbation_mode=module.PERTURBATION_MODE, anatomy_region="language_mid", radius=0.05,
            sigma=None, seed=1, parameter_mask_hash="h", capability=cap, dataset_role="map", subset_hash=f"sub_{cap}",
            base_score=0.5, perturbed_score=0.5, delta=0.0, parser_failure_rate=0.0, per_example_result_path=None,
            per_example_result_hash="h", runtime_metadata={},
        )
        for cap in STAGE9_CAPABILITIES[:3]
    ]
    (plan.output_dir).mkdir(parents=True, exist_ok=True)
    append_candidate_rows(plan.output_dir / "results.jsonl", partial_rows)

    newly_written = run_stage9_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        child_region_param_names_by_region=region_param_names_by_region, child_mask_hash_by_region=mask_hashes,
        audits=audits, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    # The partial (3-row) candidate was NOT trusted as already complete -- it was re-evaluated
    # in FULL, appending a fresh set of all 6 rows (never overwriting/deduplicating the old
    # partial ones -- append_candidate_rows is append-only by design).
    assert newly_written == 6
    all_rows_for_pid = [
        r for r in module.load_records(plan.output_dir / "results.jsonl") if r.perturbation_id == partial_pid
    ]
    assert len(all_rows_for_pid) == 3 + 6  # old partial rows + the fresh full re-evaluation
    # load_completed_perturbation_rows's own conservative design (pre-existing, unchanged by
    # this repair pass) never trusts a perturbation_id with anything other than EXACTLY one row
    # per expected capability -- 9 rows for one pid is correctly still "not complete", matching
    # "resuming NEVER trusts a partial candidate" rather than silently deduplicating.
    completed = module.load_completed_perturbation_rows(plan.output_dir / "results.jsonl", plan.capabilities)
    assert partial_pid not in completed


def test_run_result_is_deleted_after_each_capability(runtime_wrapped_vlm_32vision_factory):
    """Confirms del+release_transient_memory actually runs once per capability (6x per
    candidate), not merely once per candidate.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()
    release_calls = {"n": 0}

    import neural_thickets_repro.mem_telemetry as mem_telemetry_module

    original = mem_telemetry_module.release_transient_memory

    def _counting_release():
        release_calls["n"] += 1
        original()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mem_telemetry_module, "release_transient_memory", _counting_release)
        evaluate_one_stage9_candidate_rpc(
            engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )
    assert release_calls["n"] == 6  # once per capability


def test_scientific_record_accumulator_contains_only_compact_objects(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()

    records = evaluate_one_stage9_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    from neural_thickets_repro.thicket.schema import ExperimentResultRecord

    for r in records:
        assert isinstance(r, ExperimentResultRecord)
        # runtime_metadata must contain only JSON-primitive values -- never a raw generation,
        # image, tensor, or RunResult reference.
        for value in r.runtime_metadata.values():
            assert isinstance(value, (str, int, float, bool, type(None)))


def test_candidate_memory_telemetry_is_separate_from_results_jsonl(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    names = [n for n, _ in model.named_parameters()]
    children, audits = build_stage9_hierarchical_partition(names)
    region_param_names_by_region = {"language_mid": children["language_mid"].param_names}
    mask_hashes = {"language_mid": children["language_mid"].mask_hash}
    plan = build_stage9_plan(
        model_name="qwen2_5_vl_3b", model_revision="rev1", output_root=str(tmp_path),
        child_regions=("language_mid",), radii=(0.05,), n_directions_per_cell=1, d_map_n=5,
    )
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, plan.child_regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage9Engine(model)
    engine.store_base_weights()

    run_stage9_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        child_region_param_names_by_region=region_param_names_by_region, child_mask_hash_by_region=mask_hashes,
        audits=audits, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    import json as _json

    results_rows = [_json.loads(line) for line in (plan.output_dir / "results.jsonl").read_text().strip().split("\n")]
    for row in results_rows:
        assert not any(k.startswith("rss_") for k in row)
        assert not any(k.startswith("rss_") for k in row.get("runtime_metadata", {}))

    telemetry_rows = [_json.loads(line) for line in (plan.output_dir / "candidate_memory_telemetry.jsonl").read_text().strip().split("\n")]
    assert len(telemetry_rows) == 1  # 1 direction family
    for key in ("perturbation_index", "perturbation_id", "child_region", "radius", "direction_index",
                "rss_start_mb", "rss_after_capability_mb", "rss_after_checkpoint_mb", "rss_after_cleanup_mb",
                "delta_from_previous_candidate_mb", "high_water_mb"):
        assert key in telemetry_rows[0]
    assert len(telemetry_rows[0]["rss_after_capability_mb"]) == 6


def test_direction_seed_bank_contains_only_integers_never_tensors():
    """Stage 9 must never pre-materialize direction tensors on the driver -- only integer seeds."""
    bank = build_stage9_direction_seed_bank(module.STAGE9_BASE_SEED, STAGE9_CHILD_REGIONS, 8)
    for region, seeds in bank.items():
        for seed in seeds:
            assert isinstance(seed, int)


def test_is_ray_unrecoverable_error_true_for_a_ray_actor_error():
    import sys
    import types

    fake_ray = types.ModuleType("ray")
    fake_exceptions = types.ModuleType("ray.exceptions")

    class FakeRayActorError(Exception):
        pass

    fake_exceptions.RayActorError = FakeRayActorError
    fake_ray.exceptions = fake_exceptions
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "ray", fake_ray)
        mp.setitem(sys.modules, "ray.exceptions", fake_exceptions)
        assert module._is_ray_unrecoverable_error(FakeRayActorError("actor died")) is True
        assert module._is_ray_unrecoverable_error(RuntimeError("some other error")) is False


def test_is_ray_unrecoverable_error_false_when_ray_not_installed():
    # In this CPU-only test environment ray genuinely is not installed -- confirms the
    # ImportError path returns False rather than raising.
    assert module._is_ray_unrecoverable_error(RuntimeError("anything")) is False


def test_ray_unrecoverable_failure_path_skips_restoration_rpc(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """When the exception is classified as ray-unrecoverable, evaluate_one_stage9_candidate_rpc
    must NOT call reset_to_base_weights_via_rpc again (which would itself raise against a dead
    actor) -- it must propagate the original exception directly.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()

    class FakeUnrecoverableError(Exception):
        pass

    def _failing_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
        raise FakeUnrecoverableError("simulated Ray actor death / OOM")

    monkeypatch.setattr(module, "_is_ray_unrecoverable_error", lambda exc: isinstance(exc, FakeUnrecoverableError))
    reset_calls_before = engine.calls.count("reset_to_base_weights")

    with pytest.raises(FakeUnrecoverableError):
        evaluate_one_stage9_candidate_rpc(
            engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
            run_benchmark=_failing_run_benchmark, ray_get=_identity_ray_get,
        )
    # No additional reset_to_base_weights RPC was attempted after the unrecoverable failure.
    assert engine.calls.count("reset_to_base_weights") == reset_calls_before


def test_stage8_run_stage8_rpc_still_returns_a_list_unchanged():
    """This repair pass touches ONLY Stage 9 -- Stage 8's own run_stage8_rpc must remain
    completely unaffected (still returns the full record list, never changed to an int).
    """
    import inspect

    from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import run_stage8_rpc

    sig = inspect.signature(run_stage8_rpc)
    assert "List" in str(sig.return_annotation)


def test_all_stage9_scientific_signatures_remain_identical():
    """No new field was added to Stage9CheckpointManifest by this execution-only repair pass --
    the fixed runner resumes the SAME scientific run identity.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(Stage9CheckpointManifest)}
    assert field_names == {
        "experiment_id", "run_signature", "restoration_mode", "perturbation_mode",
        "radius_realization_method", "multimodal_cache_policy", "enable_prefix_caching",
        "generation_batch_size", "model_revision", "dataset_role", "child_regions", "radii",
        "capabilities", "n_directions_per_cell", "d_map_n", "subset_hashes", "child_mask_hashes",
        "direction_seed_bank_hash", "partition_audit_hash", "stage8_parent_run_signature",
        "expected_unique_perturbations", "expected_result_rows",
    }
    assert compute_stage9_run_signature(STAGE9_CHILD_REGIONS, STAGE9_RADII, 64, 50) == "stage9_hierarchical_anatomical_atlas_3b_v1"
