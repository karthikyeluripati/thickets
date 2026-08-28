"""Tests for run_stage11_coarse_anatomical_atlas_7b.py -- CPU-only. The real GPU/Ray/vLLM engine
and the real Qwen2.5-VL-7B-Instruct checkpoint are never touched; RPC dispatch is tested against
a fake, persistent-worker-shaped engine (same philosophy as test_run_stage8_coarse_anatomical_
atlas.py / test_run_stage9_hierarchical_anatomical_atlas.py), using REAL small torch tensors for
the lifecycle/anatomy-audit tests that need genuine weight mutation/restoration/norms.
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

import neural_thickets_repro.run_stage11_coarse_anatomical_atlas_7b as module
from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext
from neural_thickets_repro.run_stage7b_anatomical_calibration import (
    ENABLE_PREFIX_CACHING,
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
    build_stage8_direction_seed_bank,
)
from neural_thickets_repro.run_stage9_hierarchical_anatomical_atlas import STAGE8_AUTHORITATIVE_SUBSET_HASHES
from neural_thickets_repro.run_stage11_whole_model_scaling import (
    SUBSET_GATE_MODE_FULL,
    SUBSET_GATE_MODE_SMOKE,
    Stage11SmokeSubsetNondeterminismError,
    ensure_smoke_subset_determinism,
    run_smoke_subset_determinism_check,
)
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest
from neural_thickets_repro.run_stage11_coarse_anatomical_atlas_7b import (
    STAGE11_CAPABILITIES,
    STAGE11_D_MAP_N,
    STAGE11_N_DIRECTIONS_PER_CELL,
    STAGE11_RADII,
    STAGE11_REGIONS,
    STAGE11_SMOKE_D_MAP_N,
    STAGE11_SMOKE_N_DIRECTIONS,
    DirectionSeedReuseViolationError,
    IncompatibleStage11CheckpointError,
    ModelRevisionResolutionError,
    Stage11CheckpointManifest,
    Stage11DirectionAssignment,
    Stage11SubsetHashMismatchError,
    build_stage11_checkpoint_manifest,
    build_stage11_direction_seed_bank,
    build_stage11_plan,
    build_stage11_population,
    build_stage11_smoke_plan,
    build_stage11_subset_gate_report,
    compute_anatomy_audit_hash,
    compute_direction_seed_bank_hash,
    compute_stage11_run_signature,
    ensure_stage11_anatomy_audit_passes,
    ensure_stage11_checkpoint_manifest,
    ensure_stage11_subset_gate_passes,
    ensure_stage11_subset_hashes_match_stage8,
    evaluate_one_stage11_candidate_rpc,
    report_stage11_anatomy_audit,
    resolve_immutable_model_revision,
    run_stage11_rpc,
    run_stage11_subset_hash_check,
    validate_stage11_direction_seed_reuse,
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
# 1. Frozen full config -- byte-identical reuse from Stage 8
# =================================================================================================


def test_stage11_regions_radii_capabilities_reused_by_identity_from_stage8():
    assert STAGE11_REGIONS == STAGE8_REGIONS == ("vision", "multimodal_connector_or_merger", "language")
    assert STAGE11_RADII == STAGE8_RADII == (0.0035698828543799426, 0.017849414271899712, 0.07139765708759885)
    assert STAGE11_CAPABILITIES == STAGE8_CAPABILITIES
    assert STAGE11_N_DIRECTIONS_PER_CELL == STAGE8_N_DIRECTIONS_PER_CELL == 64
    assert STAGE11_D_MAP_N == STAGE8_D_MAP_N == 50


def test_default_stage11_plan_is_the_frozen_full_identity():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    assert plan.regions == STAGE11_REGIONS
    assert plan.radii == STAGE11_RADII
    assert plan.capabilities == STAGE11_CAPABILITIES
    assert plan.n_directions_per_cell == 64
    assert plan.d_map_n == 50
    assert plan.generation_batch_size == 10
    assert plan.is_smoke is False
    assert plan.run_signature == "stage11_coarse_anatomical_atlas_7b_v1"
    assert plan.model_scale == "7B"
    assert plan.model_family == "qwen2_5_vl"


def test_full_stage11_plan_matches_stage8_candidate_budget_exactly():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    assert plan.total_unique_perturbations == 3 * 3 * 64 == 576
    assert plan.total_perturbation_capability_evaluations == 576 * 6 == 3456
    assert plan.total_perturbed_model_example_evaluations == 3456 * 50 == 172_800


def test_stage11_plan_prefix_caching_and_batch_size():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    assert plan.enable_prefix_caching is False
    assert plan.generation_batch_size == 10
    assert module.ENABLE_PREFIX_CACHING == ENABLE_PREFIX_CACHING is False
    assert module.MULTIMODAL_CACHE_POLICY == MULTIMODAL_CACHE_POLICY == "full_encoder_reset_vllm011_verified_v2"
    assert module.RADIUS_REALIZATION_METHOD == RADIUS_REALIZATION_METHOD == "fixed_direction_bf16_quantization_aware_v3"


def test_build_stage11_plan_rejects_empty_regions():
    with pytest.raises(ValueError):
        build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out", regions=())


def test_build_stage11_plan_rejects_unrecognized_d_map_size():
    with pytest.raises(module.DatasetRoleViolationError):
        build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out", d_map_n=17)


def test_build_stage11_plan_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out", generation_batch_size=0)


# =================================================================================================
# 2. Smoke config -- 9 perturbations / 54 rows / 270 evaluations
# =================================================================================================


def test_smoke_plan_totals_9_perturbations_54_rows_270_evaluations():
    plan = build_stage11_smoke_plan(model_name="m", model_revision="a" * 40, output_root="out")
    assert plan.is_smoke is True
    assert plan.regions == STAGE11_REGIONS
    assert plan.radii == STAGE11_RADII
    assert plan.total_unique_perturbations == 3 * 3 * 1 == 9
    assert plan.total_perturbation_capability_evaluations == 9 * 6 == 54
    assert plan.total_perturbed_model_example_evaluations == 54 * 5 == 270


def test_smoke_and_full_plans_never_share_an_output_directory():
    full_plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    smoke_plan = build_stage11_smoke_plan(model_name="m", model_revision="a" * 40, output_root="out")
    assert full_plan.output_dir != smoke_plan.output_dir
    assert full_plan.run_signature != smoke_plan.run_signature


def test_compute_stage11_run_signature_never_returns_the_full_literal_for_any_deviation():
    sig = compute_stage11_run_signature(STAGE11_REGIONS, STAGE11_RADII, 1, 5, 10)
    assert sig != "stage11_coarse_anatomical_atlas_7b_v1"
    assert sig.startswith("stage11_smoke_")


def test_stage11_full_run_signature_never_collides_with_stage8s():
    assert module._FULL_RUN_SIGNATURE != "stage8_coarse_anatomical_atlas_3b_v2_batched10"
    assert "7b" in module._FULL_RUN_SIGNATURE
    assert "3b" not in module._FULL_RUN_SIGNATURE


# =================================================================================================
# 3. Model revision resolution -- never invents a hash
# =================================================================================================


def test_resolve_immutable_model_revision_passes_through_an_already_pinned_sha():
    sha = "a" * 40
    result = resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", sha)
    assert result["resolved_revision"] == sha
    assert result["resolution_method"] == "already_pinned"


def test_resolve_immutable_model_revision_resolves_a_mutable_ref_via_hf_api(monkeypatch):
    class _FakeModelInfo:
        sha = "b" * 40

    class _FakeHfApi:
        def model_info(self, repo_id, revision):
            assert repo_id == "Qwen/Qwen2.5-VL-7B-Instruct"
            assert revision == "main"
            return _FakeModelInfo()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    result = resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")
    assert result["resolved_revision"] == "b" * 40
    assert result["resolution_method"] == "resolved_via_hf_api"


def test_resolve_immutable_model_revision_hard_fails_when_hub_call_raises(monkeypatch):
    class _FailingHfApi:
        def model_info(self, repo_id, revision):
            raise RuntimeError("network unavailable")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _FailingHfApi)

    with pytest.raises(ModelRevisionResolutionError):
        resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")


def test_resolve_immutable_model_revision_hard_fails_on_a_malformed_sha(monkeypatch):
    class _FakeModelInfo:
        sha = "not-a-real-sha"

    class _FakeHfApi:
        def model_info(self, repo_id, revision):
            return _FakeModelInfo()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    with pytest.raises(ModelRevisionResolutionError):
        resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")


def test_resolve_immutable_model_revision_never_invents_a_hash_when_api_returns_none(monkeypatch):
    class _FakeModelInfo:
        sha = None

    class _FakeHfApi:
        def model_info(self, repo_id, revision):
            return _FakeModelInfo()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    with pytest.raises(ModelRevisionResolutionError):
        resolve_immutable_model_revision("Qwen/Qwen2.5-VL-7B-Instruct", "main")


# =================================================================================================
# 4. Hard 7B anatomy audit -- tensor counts, element counts, norms, percentages, hashes
# =================================================================================================


def test_anatomy_audit_reports_all_three_l1_regions(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    audit = report_stage11_anatomy_audit(worker, STAGE11_REGIONS)
    assert set(audit["regions"].keys()) == set(STAGE11_REGIONS)
    for region, info in audit["regions"].items():
        assert info["n_tensors"] > 0
        assert info["n_elements"] > 0
        assert info["l2_norm"] > 0
        assert 0.0 < info["percentage_of_total_elements"] <= 100.0
        assert isinstance(info["mask_hash"], str) and len(info["mask_hash"]) == 64  # sha256 hex


def test_anatomy_audit_union_equals_full_model_and_pairwise_disjoint(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    audit = report_stage11_anatomy_audit(worker, STAGE11_REGIONS)
    assert audit["union_equals_full_model"] is True
    assert audit["pairwise_disjoint"] is True
    assert audit["uncovered_by_full_model"] == []
    ensure_stage11_anatomy_audit_passes(audit, STAGE11_REGIONS)  # must not raise


def test_anatomy_audit_percentages_sum_to_100_across_the_three_l1_regions(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    audit = report_stage11_anatomy_audit(worker, STAGE11_REGIONS)
    total_pct = sum(info["percentage_of_total_elements"] for info in audit["regions"].values())
    assert total_pct == pytest.approx(100.0, abs=1e-6)


def test_anatomy_audit_element_counts_sum_to_total_model_elements(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    audit = report_stage11_anatomy_audit(worker, STAGE11_REGIONS)
    total_from_regions = sum(info["n_elements"] for info in audit["regions"].values())
    assert total_from_regions == audit["total_model_elements"]


def test_ensure_stage11_anatomy_audit_passes_rejects_a_missing_region():
    fake_audit = {"regions": {"vision": {"n_tensors": 1}}, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": []}
    with pytest.raises(RuntimeError):
        ensure_stage11_anatomy_audit_passes(fake_audit, STAGE11_REGIONS)


def test_ensure_stage11_anatomy_audit_passes_rejects_an_empty_region():
    fake_audit = {
        "regions": {r: {"n_tensors": 0 if r == "vision" else 1} for r in STAGE11_REGIONS},
        "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": [],
    }
    with pytest.raises(RuntimeError):
        ensure_stage11_anatomy_audit_passes(fake_audit, STAGE11_REGIONS)


def test_ensure_stage11_anatomy_audit_passes_rejects_uncovered_parameters():
    fake_audit = {
        "regions": {r: {"n_tensors": 1} for r in STAGE11_REGIONS},
        "union_equals_full_model": False, "pairwise_disjoint": True, "uncovered_by_full_model": ["some.param"],
    }
    with pytest.raises(RuntimeError):
        ensure_stage11_anatomy_audit_passes(fake_audit, STAGE11_REGIONS)


def test_compute_anatomy_audit_hash_is_deterministic(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    audit1 = report_stage11_anatomy_audit(worker, STAGE11_REGIONS)
    audit2 = report_stage11_anatomy_audit(worker, STAGE11_REGIONS)
    assert compute_anatomy_audit_hash(audit1) == compute_anatomy_audit_hash(audit2)


# =================================================================================================
# 5. Independent 7B direction-seed namespace
# =================================================================================================


def test_direction_seed_bank_has_64_seeds_per_region_for_the_full_config():
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, STAGE11_REGIONS, 64)
    assert set(bank.keys()) == set(STAGE11_REGIONS)
    for region, seeds in bank.items():
        assert len(seeds) == 64
        assert len(set(seeds)) == 64


def test_direction_seed_bank_is_deterministic():
    bank1 = build_stage11_direction_seed_bank(42, STAGE11_REGIONS, 8)
    bank2 = build_stage11_direction_seed_bank(42, STAGE11_REGIONS, 8)
    assert bank1 == bank2


def test_stage11_seed_namespace_is_independent_of_stage8_even_with_the_same_base_seed_and_region():
    """Section 5's explicit requirement: 3B seed-i and 7B seed-i are NOT geometrically paired.
    Even reusing STAGE8's own base seed and a shared region label, Stage 11's namespace string
    ("stage11_direction_family") differs from Stage 8's ("stage8_direction_family"), so the
    derived seed streams are provably independent.
    """
    stage11_bank = build_stage11_direction_seed_bank(module.STAGE8_BASE_SEED, ("vision",), 8)
    stage8_bank = build_stage8_direction_seed_bank(module.STAGE8_BASE_SEED, ("vision",), 8)
    assert stage11_bank["vision"] != stage8_bank["vision"]
    assert stage11_bank["vision"][0] != stage8_bank["vision"][0]


def test_same_direction_seed_reused_across_all_radii_within_a_region():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out", n_directions_per_cell=4)
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage11_population(plan, bank, mask_hashes)

    for region in plan.regions:
        seeds_per_radius = [[a.direction_seed for a in population[(region, radius)]] for radius in plan.radii]
        assert all(s == seeds_per_radius[0] for s in seeds_per_radius)


def test_population_has_576_unique_perturbation_ids_for_the_full_config():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage11_population(plan, bank, mask_hashes)
    all_ids = [a.manifest.perturbation_id for cell in population.values() for a in cell]
    assert len(all_ids) == 576
    assert len(set(all_ids)) == 576


def test_validate_stage11_direction_seed_reuse_passes_for_a_correct_population():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out", n_directions_per_cell=4)
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage11_population(plan, bank, mask_hashes)
    validate_stage11_direction_seed_reuse(plan, population)  # must not raise


def test_validate_stage11_direction_seed_reuse_detects_missing_cell():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out", n_directions_per_cell=4)
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = dict(build_stage11_population(plan, bank, mask_hashes))
    del population[("vision", plan.radii[-1])]
    with pytest.raises(DirectionSeedReuseViolationError):
        validate_stage11_direction_seed_reuse(plan, population)


# =================================================================================================
# 6. Same-example D_map subset-hash gate against Stage-8's authoritative 3B manifests
# =================================================================================================


def _fake_contexts(n=5, subset_hashes=None):
    contexts = {}
    for capability in STAGE11_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        subset_hash = subset_hashes.get(capability) if subset_hashes else partition.manifest_hash
        # examples length == n (not always []) so tests can check example counts (e.g. the
        # smoke subset-determinism gate, which asserts every capability has exactly N=5 examples).
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=list(ids), partition=partition, subset_hash=subset_hash, base_score=0.5)
    return contexts


def test_subset_hash_check_passes_when_contexts_use_the_authoritative_stage8_hashes():
    contexts = _fake_contexts(subset_hashes=STAGE8_AUTHORITATIVE_SUBSET_HASHES)
    report = run_stage11_subset_hash_check(contexts)
    assert report["all_match"] is True
    ensure_stage11_subset_hashes_match_stage8(report)  # must not raise


def test_subset_hash_check_fails_when_a_capability_uses_a_different_subset():
    contexts = _fake_contexts()  # random per-test partition hashes -- never equal the authoritative ones
    report = run_stage11_subset_hash_check(contexts)
    assert report["all_match"] is False
    with pytest.raises(Stage11SubsetHashMismatchError):
        ensure_stage11_subset_hashes_match_stage8(report)


# =================================================================================================
# 6b. MODE-AWARE subset gate -- reproduces and fixes the live 7B-anatomy-smoke N=5-vs-N=50 failure
# (the same bug class already fixed for run_stage11_whole_model_scaling.py; the smoke-determinism
# machinery below is reused BY IMPORT from that module, never reimplemented).
# =================================================================================================


def test_full_mode_subset_gate_still_hard_fails_on_a_single_mismatched_n50_hash():
    """(6D/6E) FULL mode must remain strict -- even ONE of the six N=50 live subset hashes
    differing from STAGE8_AUTHORITATIVE_SUBSET_HASHES is a hard stop.
    """
    contexts = _fake_contexts(subset_hashes=dict(STAGE8_AUTHORITATIVE_SUBSET_HASHES))
    from dataclasses import replace as _dc_replace
    one_capability = next(iter(contexts))
    contexts[one_capability] = _dc_replace(contexts[one_capability], subset_hash="corrupted" * 4)

    full_report = run_stage11_subset_hash_check(contexts)
    assert full_report["all_match"] is False
    gate_report = build_stage11_subset_gate_report(is_smoke=False, d_map_n=STAGE11_D_MAP_N, full_subset_hash_report=full_report)
    assert gate_report["subset_gate_mode"] == SUBSET_GATE_MODE_FULL
    with pytest.raises(Stage11SubsetHashMismatchError):
        ensure_stage11_subset_gate_passes(gate_report)


def test_full_mode_subset_gate_passes_when_all_six_hashes_match():
    contexts = _fake_contexts(subset_hashes=dict(STAGE8_AUTHORITATIVE_SUBSET_HASHES))
    full_report = run_stage11_subset_hash_check(contexts)
    gate_report = build_stage11_subset_gate_report(is_smoke=False, d_map_n=STAGE11_D_MAP_N, full_subset_hash_report=full_report)
    ensure_stage11_subset_gate_passes(gate_report)  # must not raise


def test_smoke_mode_never_raises_subset_hash_mismatch_merely_because_n5_differs_from_n50():
    """(6A) Reproduces the EXACT reported failure class: an N=5 smoke subset hash will NEVER equal
    the N=50 authoritative Stage-8 hash (different sample sizes by construction) -- smoke mode
    must not raise Stage11SubsetHashMismatchError over this, and must instead pass its own
    deterministic-reconstruction gate.
    """
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE11_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    pass_b = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)  # independently "rebuilt", same deterministic hashes

    for cap in STAGE11_CAPABILITIES:
        assert pass_a[cap].subset_hash != STAGE8_AUTHORITATIVE_SUBSET_HASHES[cap]

    determinism_report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    assert determinism_report["all_deterministic"] is True
    assert determinism_report["all_n_matches_expected"] is True
    gate_report = build_stage11_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=determinism_report)
    assert gate_report["subset_gate_mode"] == SUBSET_GATE_MODE_SMOKE
    ensure_stage11_subset_gate_passes(gate_report)  # must NOT raise Stage11SubsetHashMismatchError (or anything else)


def test_two_independent_smoke_manifest_builds_are_identical():
    """(6B) Two independently built N=5 smoke manifests must match exactly."""
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE11_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    pass_b = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    for cap in STAGE11_CAPABILITIES:
        assert report[cap]["pass_a_subset_hash"] == report[cap]["pass_b_subset_hash"]
        assert report[cap]["matches"] is True


def test_changing_one_smoke_example_breaks_determinism_check():
    """(6C) Changing one N=5 example fails smoke determinism."""
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE11_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    changed_hashes = dict(n5_hash_by_cap)
    one_capability = STAGE11_CAPABILITIES[0]
    changed_hashes[one_capability] = "a_different_n5_hash_from_a_changed_example"
    pass_b = _fake_contexts(n=5, subset_hashes=changed_hashes)

    determinism_report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    assert determinism_report["all_deterministic"] is False
    assert determinism_report[one_capability]["matches"] is False
    gate_report = build_stage11_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=determinism_report)
    with pytest.raises(Stage11SmokeSubsetNondeterminismError):
        ensure_stage11_subset_gate_passes(gate_report)


def test_subset_gate_report_requires_the_matching_sub_report_for_its_mode():
    with pytest.raises(ValueError):
        build_stage11_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=None)
    with pytest.raises(ValueError):
        build_stage11_subset_gate_report(is_smoke=False, d_map_n=50, full_subset_hash_report=None)


def test_ensure_stage11_subset_gate_passes_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        ensure_stage11_subset_gate_passes({"subset_gate_mode": "not_a_real_mode"})


def test_subset_gate_runs_before_engine_launch_and_before_any_candidate_row():
    """(6G) A failing subset gate must exit BEFORE any GPU engine is launched, before the live
    anatomy audit, and before any Stage-11 candidate row is evaluated/checkpointed -- proven
    structurally from main()'s own source ordering.
    """
    source = inspect.getsource(module.main)
    gate_pos = source.index("ensure_stage11_subset_gate_passes(subset_gate_report)")
    engine_launch_pos = source.index("launch_stage6_engine(")
    anatomy_audit_pos = source.index("report_stage11_anatomy_audit")
    run_rpc_pos = source.index("run_stage11_rpc(")
    assert gate_pos < engine_launch_pos < anatomy_audit_pos < run_rpc_pos


def test_smoke_subset_gate_never_calls_full_mode_equality_check(monkeypatch):
    """(6F) Structural proof the smoke path cannot accidentally fall through to the N=50
    comparison.
    """
    n5_hash_by_cap = {cap: f"n5_hash_{cap}" for cap in STAGE11_CAPABILITIES}
    pass_a = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    pass_b = _fake_contexts(n=5, subset_hashes=n5_hash_by_cap)
    determinism_report = run_smoke_subset_determinism_check(pass_a, pass_b, d_map_n=5)
    gate_report = build_stage11_subset_gate_report(is_smoke=True, d_map_n=5, smoke_determinism_report=determinism_report)

    def _should_never_be_called(report):
        raise AssertionError("ensure_stage11_subset_hashes_match_stage8 must never be called for a smoke gate report")

    monkeypatch.setattr(module, "ensure_stage11_subset_hashes_match_stage8", _should_never_be_called)
    ensure_stage11_subset_gate_passes(gate_report)  # must not raise -- dispatches to the smoke path only


def test_smoke_and_full_gate_modes_share_the_imported_whole_model_smoke_machinery():
    """(2) Proof the 7B-anatomy runner reuses run_stage11_whole_model_scaling's smoke-determinism
    helper BY IDENTITY rather than a duplicated reimplementation -- the two tracks cannot drift
    into different smoke policies again.
    """
    import neural_thickets_repro.run_stage11_whole_model_scaling as whole_model_module
    assert module.run_smoke_subset_determinism_check is whole_model_module.run_smoke_subset_determinism_check
    assert module.ensure_smoke_subset_determinism is whole_model_module.ensure_smoke_subset_determinism


def test_stage11_full_and_smoke_design_totals_unchanged_by_this_repair_pass():
    """(6H/6I) This repair pass touches gate SELECTION only -- the frozen 576/3456/N50 full design
    and 9/54/N5 smoke design must be byte-identical to before.
    """
    full_plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    assert full_plan.total_unique_perturbations == 576
    assert full_plan.total_perturbation_capability_evaluations == 3456
    assert full_plan.d_map_n == 50

    smoke_plan = build_stage11_smoke_plan(model_name="m", model_revision="a" * 40, output_root="out")
    assert smoke_plan.total_unique_perturbations == 9
    assert smoke_plan.total_perturbation_capability_evaluations == 54
    assert smoke_plan.d_map_n == 5


# =================================================================================================
# 7. Checkpoint identity -- includes anatomy_audit_hash + stage8_parent_run_signature
# =================================================================================================


def _region_mask_hashes(plan):
    return {r: f"hash_{r}" for r in plan.regions}


def _fake_anatomy_audit():
    return {
        "regions": {r: {"mask_hash": f"hash_{r}", "n_tensors": 3, "n_elements": 100} for r in STAGE11_REGIONS},
        "total_model_elements": 300, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": [],
    }


def test_checkpoint_manifest_includes_anatomy_audit_hash_and_stage8_parent_signature():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    audit = _fake_anatomy_audit()
    checkpoint = build_stage11_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audit)
    assert checkpoint.anatomy_audit_hash == compute_anatomy_audit_hash(audit)
    assert checkpoint.stage8_parent_run_signature == module.STAGE8_PARENT_RUN_SIGNATURE == "stage8_coarse_anatomical_atlas_3b_v2_batched10"
    assert checkpoint.expected_unique_perturbations == 576
    assert checkpoint.expected_result_rows == 3456


def test_checkpoint_manifest_round_trips_through_json():
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root="out")
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    audit = _fake_anatomy_audit()
    checkpoint = build_stage11_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audit)
    restored = Stage11CheckpointManifest.from_dict(checkpoint.to_dict())
    assert restored == checkpoint


def test_ensure_stage11_checkpoint_manifest_creates_when_absent(tmp_path):
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root=str(tmp_path))
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    audit = _fake_anatomy_audit()
    checkpoint = build_stage11_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audit)
    path = tmp_path / "checkpoint_manifest.json"
    result = ensure_stage11_checkpoint_manifest(path, checkpoint)
    assert path.exists()
    assert result == checkpoint


def test_ensure_stage11_checkpoint_manifest_hard_fails_on_anatomy_mismatch(tmp_path):
    plan = build_stage11_plan(model_name="m", model_revision="a" * 40, output_root=str(tmp_path))
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    audit_a = _fake_anatomy_audit()
    checkpoint_a = build_stage11_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audit_a)
    path = tmp_path / "checkpoint_manifest.json"
    ensure_stage11_checkpoint_manifest(path, checkpoint_a)

    audit_b = _fake_anatomy_audit()
    audit_b["regions"]["vision"]["n_elements"] = 999999  # a differently-derived 7B anatomy
    checkpoint_b = build_stage11_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan), bank, audit_b)
    with pytest.raises(IncompatibleStage11CheckpointError):
        ensure_stage11_checkpoint_manifest(path, checkpoint_b)


# =================================================================================================
# 8. Candidate lifecycle -- bounded memory, v3 + bracket-expansion acceptance, Ray-unrecoverable
#    failure path, no all-record accumulator
# =================================================================================================


class _FakeStage11Engine:
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


class _FakeStage11RunResult:
    def __init__(self, primary_metric=0.6):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}

    def generation_hash(self):
        return "fakehash"


def _fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params, **kwargs):
    return _FakeStage11RunResult(primary_metric=0.6)


def _build_assignment_and_engine(model, *, region="language", radius=0.05, seed=42, direction_index=0):
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names = atlas.region(region).param_names
    manifest = PerturbationManifest(
        seed=seed, perturbation_mode=PERTURBATION_MODE, model_family="qwen2_5_vl", model_scale="7B",
        model_revision="a" * 40, parameter_mask_hash=atlas.region(region).mask_hash,
        anatomy_region=region, radius=radius, sigma=None,
    )
    assignment = Stage11DirectionAssignment(manifest=manifest, region=region, direction_index=direction_index, direction_seed=seed)
    engine = _FakeStage11Engine(model)
    engine.store_base_weights()
    return assignment, engine, region_param_names


def test_evaluate_one_stage11_candidate_rpc_produces_a_record_per_capability(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    contexts = _fake_contexts()

    records = evaluate_one_stage11_candidate_rpc(
        engine, assignment, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert len(records) == 6
    assert {r.capability for r in records} == set(STAGE11_CAPABILITIES)
    assert all(r.model_scale == "7B" for r in records)
    assert all(r.runtime_metadata["stage8_parent_run_signature"] == "stage8_coarse_anatomical_atlas_3b_v2_batched10" for r in records)


def test_evaluate_one_stage11_candidate_rpc_restores_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    evaluate_one_stage11_candidate_rpc(
        engine, assignment, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), before[name])


def test_evaluate_one_stage11_candidate_rpc_releases_run_result_after_each_capability(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    calls = {"n": 0}

    def _counting_release():
        calls["n"] += 1

    monkeypatch.setattr("neural_thickets_repro.mem_telemetry.release_transient_memory", _counting_release)

    evaluate_one_stage11_candidate_rpc(
        engine, assignment, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert calls["n"] == 6


def test_evaluate_one_stage11_candidate_rpc_rejects_wrong_perturbation_mode(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    bad_manifest = PerturbationManifest(
        seed=1, perturbation_mode="global_gaussian_upstream", model_family="qwen2_5_vl", model_scale="7B",
        model_revision="a" * 40, parameter_mask_hash="h", anatomy_region=None, radius=None, sigma=0.001,
    )
    assignment = Stage11DirectionAssignment(manifest=bad_manifest, region="language", direction_index=0, direction_seed=1)
    engine = _FakeStage11Engine(model)
    engine.store_base_weights()

    with pytest.raises(ValueError):
        evaluate_one_stage11_candidate_rpc(
            engine, assignment, atlas.region("language").param_names, _fake_contexts(), tokenizer=None,
            sampling_params=None, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )


class _FakeUnrecoverableRayError(RuntimeError):
    pass


def test_evaluate_one_stage11_candidate_rpc_skips_restoration_rpc_on_ray_unrecoverable_error(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)

    monkeypatch.setattr(module, "_is_ray_unrecoverable_error", lambda exc: isinstance(exc, _FakeUnrecoverableRayError))

    def _failing_run_benchmark(*args, **kwargs):
        raise _FakeUnrecoverableRayError("actor died")

    reset_calls_before = engine.calls.count("reset_to_base_weights")
    with pytest.raises(_FakeUnrecoverableRayError):
        evaluate_one_stage11_candidate_rpc(
            engine, assignment, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_failing_run_benchmark, ray_get=_identity_ray_get,
        )
    assert engine.calls.count("reset_to_base_weights") == reset_calls_before  # no extra RPC against a dead engine


def test_evaluate_one_stage11_candidate_rpc_still_restores_on_a_recoverable_failure(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    model = runtime_wrapped_vlm_32vision_factory()
    assignment, engine, region_param_names = _build_assignment_and_engine(model)
    monkeypatch.setattr(module, "_is_ray_unrecoverable_error", lambda exc: False)

    def _failing_run_benchmark(*args, **kwargs):
        raise ValueError("some ordinary failure")

    with pytest.raises(ValueError):
        evaluate_one_stage11_candidate_rpc(
            engine, assignment, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_failing_run_benchmark, ray_get=_identity_ray_get,
        )
    assert "reset_to_base_weights" in engine.calls  # still attempted for a recoverable failure


# =================================================================================================
# 9. run_stage11_rpc: bounded-memory accumulator-free loop, checkpoint-only-after-success, resume
# =================================================================================================


def test_run_stage11_rpc_persists_rows_only_after_full_success_and_writes_telemetry(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names_by_region = {"language": atlas.region("language").param_names}
    mask_hashes = {"language": atlas.region("language").mask_hash}

    plan = build_stage11_plan(
        model_name="qwen2_5_vl_7b", model_revision="a" * 40, output_root=str(tmp_path),
        regions=("language",), radii=(0.05,), n_directions_per_cell=2, d_map_n=5,
    )
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    contexts = _fake_contexts(n=5)
    engine = _FakeStage11Engine(model)
    engine.store_base_weights()
    audit = {"regions": {"language": {"mask_hash": mask_hashes["language"], "n_tensors": 1, "n_elements": 1}}, "total_model_elements": 1, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": []}

    newly_written = run_stage11_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        anatomy_audit=audit, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert newly_written == 2 * 6
    results_path = plan.output_dir / "results.jsonl"
    assert results_path.exists()
    assert len(results_path.read_text().strip().split("\n")) == 12
    assert (plan.output_dir / "checkpoint_manifest.json").exists()
    telemetry_path = plan.output_dir / "candidate_memory_telemetry.jsonl"
    assert telemetry_path.exists()
    assert len(telemetry_path.read_text().strip().split("\n")) == 2  # one line per candidate, not per row


def test_run_stage11_rpc_returns_an_int_not_a_record_list(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names_by_region = {"language": atlas.region("language").param_names}
    mask_hashes = {"language": atlas.region("language").mask_hash}
    plan = build_stage11_plan(
        model_name="qwen2_5_vl_7b", model_revision="a" * 40, output_root=str(tmp_path),
        regions=("language",), radii=(0.05,), n_directions_per_cell=1, d_map_n=5,
    )
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    engine = _FakeStage11Engine(model)
    engine.store_base_weights()
    audit = {"regions": {"language": {"mask_hash": mask_hashes["language"], "n_tensors": 1, "n_elements": 1}}, "total_model_elements": 1, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": []}

    result = run_stage11_rpc(
        plan, _fake_contexts(n=5), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        anatomy_audit=audit, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert isinstance(result, int)


def test_run_stage11_rpc_resumes_skipping_completed_candidates(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names_by_region = {"language": atlas.region("language").param_names}
    mask_hashes = {"language": atlas.region("language").mask_hash}
    plan = build_stage11_plan(
        model_name="qwen2_5_vl_7b", model_revision="a" * 40, output_root=str(tmp_path),
        regions=("language",), radii=(0.05,), n_directions_per_cell=2, d_map_n=5,
    )
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    engine = _FakeStage11Engine(model)
    engine.store_base_weights()
    audit = {"regions": {"language": {"mask_hash": mask_hashes["language"], "n_tensors": 1, "n_elements": 1}}, "total_model_elements": 1, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": []}

    first = run_stage11_rpc(
        plan, _fake_contexts(n=5), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        anatomy_audit=audit, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert first == 12

    second = run_stage11_rpc(
        plan, _fake_contexts(n=5), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        anatomy_audit=audit, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )
    assert second == 0  # nothing new -- both candidates already completed
    results_path = plan.output_dir / "results.jsonl"
    assert len(results_path.read_text().strip().split("\n")) == 12  # unchanged, never duplicated


def test_run_stage11_rpc_hard_fails_on_incompatible_checkpoint(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names_by_region = {"language": atlas.region("language").param_names}
    mask_hashes = {"language": atlas.region("language").mask_hash}
    plan = build_stage11_plan(
        model_name="qwen2_5_vl_7b", model_revision="a" * 40, output_root=str(tmp_path),
        regions=("language",), radii=(0.05,), n_directions_per_cell=1, d_map_n=5,
    )
    bank = build_stage11_direction_seed_bank(module.STAGE11_BASE_SEED, plan.regions, plan.n_directions_per_cell)
    engine = _FakeStage11Engine(model)
    engine.store_base_weights()
    audit = {"regions": {"language": {"mask_hash": mask_hashes["language"], "n_tensors": 1, "n_elements": 1}}, "total_model_elements": 1, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": []}
    run_stage11_rpc(
        plan, _fake_contexts(n=5), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
        anatomy_audit=audit, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    different_audit = {"regions": {"language": {"mask_hash": "DIFFERENT_HASH", "n_tensors": 1, "n_elements": 1}}, "total_model_elements": 1, "union_equals_full_model": True, "pairwise_disjoint": True, "uncovered_by_full_model": []}
    with pytest.raises(IncompatibleStage11CheckpointError):
        run_stage11_rpc(
            plan, _fake_contexts(n=5), engine, tokenizer=None, sampling_params=None, seed_bank=bank,
            region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hashes,
            anatomy_audit=different_audit, run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )
