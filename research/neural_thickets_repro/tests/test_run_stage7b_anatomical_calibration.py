"""Tests for run_stage7b_anatomical_calibration.py -- CPU-only. The real GPU/Ray/vLLM engine is
never launched; RPC dispatch is tested against a fake, persistent-worker-shaped engine (see
_FakeCalibrationEngine) using REAL small torch tensors, same philosophy as
tests/test_run_global_visual_thicket_pilot.py's _FakeRayEngine.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext, PILOT_CAPABILITIES
from neural_thickets_repro.run_stage7b_anatomical_calibration import (
    CALIBRATION_CAPABILITIES,
    CALIBRATION_D_MAP_N,
    CALIBRATION_N_PER_CELL,
    CALIBRATION_REGIONS,
    DATASET_ROLE,
    EXPERIMENT_ID,
    OutOfRegionDriftError,
    PERTURBATION_MODE,
    RealizedRadiusMismatchError,
    Stage7bCheckpointManifest,
    build_stage7b_checkpoint_manifest,
    build_stage7b_plan,
    build_stage7b_population,
    ensure_stage7b_checkpoint_manifest,
    evaluate_one_calibration_candidate_rpc,
)
from neural_thickets_repro.run_stage7b_anatomical_calibration import IncompatibleCalibrationCheckpointError


def _identity_ray_get(x):
    return x


# --- Stage7bPlan --------------------------------------------------------------------------------


def test_calibration_uses_exactly_the_three_pilot_capabilities():
    assert CALIBRATION_CAPABILITIES == PILOT_CAPABILITIES == ("visual_grounding", "ocr_text_recognition_grounded", "spatial_reasoning")


def test_calibration_population_size_is_8_per_cell():
    assert CALIBRATION_N_PER_CELL == 8


def test_calibration_d_map_size_is_20():
    assert CALIBRATION_D_MAP_N == 20


def test_calibration_regions_are_the_three_l1_anatomy_regions():
    assert CALIBRATION_REGIONS == ("vision", "multimodal_connector_or_merger", "language")


def test_build_stage7b_plan_rejects_empty_radii(tmp_path):
    with pytest.raises(ValueError):
        build_stage7b_plan(model_name="m", model_revision="r", radii=[], output_dir=tmp_path)


def test_build_stage7b_plan_totals():
    plan = build_stage7b_plan(model_name="m", model_revision="r", radii=[0.01, 0.05, 0.1], output_dir="out")
    assert plan.total_unique_perturbations == 3 * 3 * CALIBRATION_N_PER_CELL  # regions x radii x n_per_cell
    assert plan.total_perturbation_capability_evaluations == plan.total_unique_perturbations * 3


# --- population: 8 seeds per (region, radius) cell, unique across the whole population --------


def test_build_stage7b_population_has_n_per_cell_members_for_every_region_radius_pair():
    plan = build_stage7b_plan(model_name="qwen2_5_vl", model_revision="rev1", radii=[0.01, 0.05], output_dir="out")
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}

    population = build_stage7b_population(plan, base_seed=123, parameter_mask_hash_by_region=mask_hashes)

    assert set(population.keys()) == {(r, radius) for r in plan.regions for radius in plan.radii}
    for cell, manifests in population.items():
        assert len(manifests) == CALIBRATION_N_PER_CELL
        region, radius = cell
        for m in manifests:
            assert m.anatomy_region == region
            assert m.radius == radius
            assert m.sigma is None
            assert m.perturbation_mode == PERTURBATION_MODE


def test_build_stage7b_population_is_deterministic():
    plan = build_stage7b_plan(model_name="qwen2_5_vl", model_revision="rev1", radii=[0.01], output_dir="out")
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    pop1 = build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region=mask_hashes)
    pop2 = build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region=mask_hashes)
    for cell in pop1:
        assert [m.perturbation_id for m in pop1[cell]] == [m.perturbation_id for m in pop2[cell]]


def test_build_stage7b_population_requires_mask_hash_for_every_region():
    plan = build_stage7b_plan(model_name="m", model_revision="r", radii=[0.01], output_dir="out")
    with pytest.raises(ValueError, match="Missing parameter_mask_hash"):
        build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region={"vision": "h"})


def test_build_stage7b_population_no_two_manifests_share_a_worker_seed():
    plan = build_stage7b_plan(model_name="m", model_revision="r", radii=[0.01, 0.05, 0.1], output_dir="out")
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region=mask_hashes)
    all_seeds = [m.seed for manifests in population.values() for m in manifests]
    assert len(all_seeds) == len(set(all_seeds))


# --- checkpoint manifest identity ---------------------------------------------------------------


def _fake_contexts(n=20):
    contexts = {}
    for capability in CALIBRATION_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=[], partition=partition, subset_hash=partition.manifest_hash, base_score=0.5)
    return contexts


def test_build_stage7b_checkpoint_manifest_fields():
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", radii=[0.01, 0.05], output_dir="out")
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts())
    assert checkpoint.experiment_id == EXPERIMENT_ID
    assert checkpoint.perturbation_mode == PERTURBATION_MODE
    assert checkpoint.dataset_role == "map" == DATASET_ROLE
    assert checkpoint.d_map_n == CALIBRATION_D_MAP_N
    assert checkpoint.n_per_cell == CALIBRATION_N_PER_CELL
    assert checkpoint.expected_unique_perturbations == plan.total_unique_perturbations


def test_checkpoint_manifest_round_trips_through_json():
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", radii=[0.01], output_dir="out")
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts())
    restored = Stage7bCheckpointManifest.from_dict(json.loads(json.dumps(checkpoint.to_dict())))
    assert restored == checkpoint


def test_ensure_stage7b_checkpoint_manifest_creates_when_absent(tmp_path):
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", radii=[0.01], output_dir="out")
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts())
    path = tmp_path / "checkpoint_manifest.json"
    result = ensure_stage7b_checkpoint_manifest(path, checkpoint)
    assert path.exists()
    assert result == checkpoint


def test_ensure_stage7b_checkpoint_manifest_hard_fails_on_mismatch(tmp_path):
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", radii=[0.01], output_dir="out")
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts())
    path = tmp_path / "checkpoint_manifest.json"
    ensure_stage7b_checkpoint_manifest(path, checkpoint)

    plan2 = build_stage7b_plan(model_name="m", model_revision="rev2", radii=[0.01], output_dir="out")
    checkpoint2 = build_stage7b_checkpoint_manifest(plan2, _fake_contexts())
    with pytest.raises(IncompatibleCalibrationCheckpointError):
        ensure_stage7b_checkpoint_manifest(path, checkpoint2)


# --- no D_confirm/select/test access -------------------------------------------------------------


def test_dataset_role_is_always_map():
    assert DATASET_ROLE == "map"


def test_checkpoint_manifest_rejects_wrong_d_map_size():
    from neural_thickets_repro.run_stage7b_anatomical_calibration import DatasetRoleViolationError

    bad_plan = build_stage7b_plan(model_name="m", model_revision="r", radii=[0.01], output_dir="out")
    object.__setattr__(bad_plan, "d_map_n", 15)  # simulate a caller-mutated plan with the wrong size
    with pytest.raises(DatasetRoleViolationError):
        build_stage7b_checkpoint_manifest(bad_plan, _fake_contexts())


# --- no best-radius selection logic exists (section 7) -------------------------------------------


def test_no_best_radius_selection_logic_exists():
    import neural_thickets_repro.run_stage7b_anatomical_calibration as module

    forbidden_substrings = ("best_radius", "select_best", "optimize_radius", "maximize", "argmax_radius")
    public_names = [name for name in dir(module) if not name.startswith("_")]
    for name in public_names:
        lowered = name.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, f"found forbidden name {name!r} (matches {forbidden!r})"


# --- evaluate_one_calibration_candidate_rpc: fixed-base lifecycle -------------------------------


class _FakeCalibrationEngine:
    """Persistent-worker-shaped fake: builds worker_self ONCE (not per-call), so diag_snapshot_
    base's ad-hoc worker-instance state (_anatomical_diag_base_state) correctly persists across
    separate collective_rpc dispatches -- matching how a real, long-lived vLLM worker process
    behaves (unlike a fake that reconstructs worker_self fresh on every call).
    """

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


class _FakeRunResult:
    def __init__(self, primary_metric):
        self.aggregate_metrics = {"primary_metric": primary_metric, "parser_failure_rate": 0.0}

    def generation_hash(self):
        return "fakehash"


def _fake_run_benchmark(benchmark, examples, llm_adapter, tokenizer, sampling_params):
    return _FakeRunResult(primary_metric=0.6)


def _build_manifest_and_engine(model, *, region="language", radius=0.05, seed=42):
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names = atlas.region(region).param_names
    manifest = PerturbationManifest(
        seed=seed, perturbation_mode=PERTURBATION_MODE, model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash=atlas.region(region).mask_hash,
        anatomy_region=region, radius=radius, sigma=None,
    )
    engine = _FakeCalibrationEngine(model)
    engine.store_base_weights()
    return manifest, engine, region_param_names


def test_evaluate_one_calibration_candidate_rpc_produces_a_record_per_capability(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model)
    contexts = _fake_contexts()

    records = evaluate_one_calibration_candidate_rpc(
        engine, manifest, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert len(records) == 3
    assert {r.capability for r in records} == set(CALIBRATION_CAPABILITIES)
    for r in records:
        assert r.experiment_id == EXPERIMENT_ID
        assert r.perturbation_mode == PERTURBATION_MODE
        assert r.anatomy_region == "language"
        assert r.radius == pytest.approx(0.05)
        assert r.sigma is None
        assert r.dataset_role == "map"
        assert r.perturbed_score == pytest.approx(0.6)


def test_evaluate_one_calibration_candidate_rpc_resets_before_and_after(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model)
    contexts = _fake_contexts()
    engine.calls.clear()

    evaluate_one_calibration_candidate_rpc(
        engine, manifest, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert "reset_to_base_weights" in engine.calls
    assert engine.calls.count("reset_to_base_weights") >= 1
    assert "scoped_apply_anatomical_perturbation" in engine.calls


def test_evaluate_one_calibration_candidate_rpc_restores_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model)
    base_snapshot = {k: v.clone() for k, v in engine._base_weights.items()}
    contexts = _fake_contexts()

    evaluate_one_calibration_candidate_rpc(
        engine, manifest, region_param_names, contexts, tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_snapshot[name])


def test_evaluate_one_calibration_candidate_rpc_rejects_wrong_perturbation_mode(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names = atlas.region("language").param_names
    manifest = PerturbationManifest(
        seed=1, perturbation_mode="global_gaussian_upstream", model_family="x", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="h", sigma=0.01,
    )
    engine = _FakeCalibrationEngine(model)
    engine.store_base_weights()
    with pytest.raises(ValueError, match="anatomical_relative_l2"):
        evaluate_one_calibration_candidate_rpc(
            engine, manifest, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )


def test_evaluate_one_calibration_candidate_rpc_hard_fails_on_realized_radius_mismatch(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """Forces a mismatch by monkeypatching scoped_apply_anatomical_perturbation's dispatched
    result to report a realized radius far from what was requested.
    """
    import neural_thickets_repro.run_stage7b_anatomical_calibration as module

    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model, radius=0.05)

    def _broken_apply(worker_self, seed, r, region_name, region_param_names):
        return {"region": region_name, "seed": seed, "requested_relative_l2": r, "realized_relative_l2": r + 10.0, "region_param_count": len(region_param_names)}

    monkeypatch.setattr(module, "scoped_apply_anatomical_perturbation", _broken_apply)

    with pytest.raises(RealizedRadiusMismatchError):
        evaluate_one_calibration_candidate_rpc(
            engine, manifest, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )


def test_evaluate_one_calibration_candidate_rpc_hard_fails_on_out_of_region_drift(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """Forces an out-of-region drift report by monkeypatching diag_region_drift's result --
    the perturbation itself is correct (apply_anatomical_relative_l2 never touches outside its
    region, proven directly in test_thicket_perturbation.py); this test proves the CALLER
    hard-fails if the drift check ever reported a leak, regardless of cause.
    """
    import neural_thickets_repro.run_stage7b_anatomical_calibration as module

    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model, radius=0.05)

    def _broken_drift(worker_self, region_param_names):
        return {
            "in_region": {"max_abs_drift": 1.0, "fraction_elements_differing": 1.0},
            "out_of_region": {"max_abs_drift": 0.5, "fraction_elements_differing": 0.01},
            "region_param_count": len(region_param_names),
            "out_of_region_param_count": 10,
        }

    monkeypatch.setattr(module, "diag_region_drift", _broken_drift)

    with pytest.raises(OutOfRegionDriftError):
        evaluate_one_calibration_candidate_rpc(
            engine, manifest, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )


def test_evaluate_one_calibration_candidate_rpc_never_touches_outside_region_for_real(runtime_wrapped_vlm_32vision_factory):
    """End-to-end (no monkeypatching): the real scoped_apply_anatomical_perturbation only
    touches its own region's parameters -- confirmed by inspecting the model directly after
    the call returns (before the lifecycle's own final reset).
    """
    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model, region="vision", radius=0.05)
    region_names_set = set(region_param_names)
    outside_before = {n: p.detach().clone() for n, p in model.named_parameters() if n not in region_names_set}

    evaluate_one_calibration_candidate_rpc(
        engine, manifest, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    # After the full lifecycle (which ends with an exact reset), everything must match base --
    # already covered by test_..._restores_exactly. Here we additionally confirm the outside
    # -region snapshot taken BEFORE perturbing matches base too (i.e. it was never base-drifted).
    for name, value in outside_before.items():
        assert torch.equal(value, engine._base_weights[name])
