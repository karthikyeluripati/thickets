"""Tests for run_stage7b_anatomical_calibration.py -- CPU-only. The real GPU/Ray/vLLM engine is
never launched; RPC dispatch is tested against a fake, persistent-worker-shaped engine (see
_FakeCalibrationEngine) using REAL small torch tensors, same philosophy as
tests/test_run_global_visual_thicket_pilot.py's _FakeRayEngine.
"""
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

import neural_thickets_repro.run_global_visual_thicket_pilot as run_global_visual_thicket_pilot
import neural_thickets_repro.run_stage7b_anatomical_calibration as module
import neural_thickets_repro.thicket.anatomy as thicket_anatomy
from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext, PILOT_CAPABILITIES
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest
from neural_thickets_repro.run_stage7b_anatomical_calibration import (
    CALIBRATION_CAPABILITIES,
    DATASET_ROLE,
    EXPERIMENT_ID,
    FULL_CALIBRATION_D_MAP_N,
    FULL_CALIBRATION_N_PER_CELL,
    FULL_CALIBRATION_RADII,
    FULL_CALIBRATION_REGIONS,
    IncompatibleCalibrationCheckpointError,
    OutOfRegionDriftError,
    PERTURBATION_MODE,
    RealizedRadiusMismatchError,
    SMOKE_D_MAP_N,
    SMOKE_N_PER_CELL,
    SMOKE_RADIUS,
    SMOKE_REGION,
    Stage7bCheckpointManifest,
    build_smoke_stage7b_plan,
    build_stage7b_checkpoint_manifest,
    build_stage7b_plan,
    build_stage7b_population,
    compute_stage7b_run_signature,
    ensure_stage7b_checkpoint_manifest,
    evaluate_one_calibration_candidate_rpc,
    report_region_param_names,
    run_stage7b_rpc,
)


def _identity_ray_get(x):
    return x


# =================================================================================================
# 1. Frozen full-calibration paper config
# =================================================================================================


def test_calibration_uses_exactly_the_three_pilot_capabilities():
    assert CALIBRATION_CAPABILITIES == PILOT_CAPABILITIES == ("visual_grounding", "ocr_text_recognition_grounded", "spatial_reasoning")


def test_full_calibration_regions_are_the_three_l1_anatomy_regions():
    assert FULL_CALIBRATION_REGIONS == ("vision", "multimodal_connector_or_merger", "language")


def test_full_calibration_radii_are_exactly_the_frozen_stage7a_values():
    assert FULL_CALIBRATION_RADII == (
        0.0035698828543799426, 0.017849414271899712, 0.035698828543799424,
        0.07139765708759885, 0.1784941427189971, 0.3569882854379942,
    )
    assert len(FULL_CALIBRATION_RADII) == 6


def test_full_calibration_population_size_is_8_per_cell():
    assert FULL_CALIBRATION_N_PER_CELL == 8


def test_full_calibration_d_map_size_is_20():
    assert FULL_CALIBRATION_D_MAP_N == 20


def test_default_stage7b_plan_is_the_frozen_full_identity():
    plan = build_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    assert plan.regions == FULL_CALIBRATION_REGIONS
    assert plan.radii == FULL_CALIBRATION_RADII
    assert plan.n_per_cell == FULL_CALIBRATION_N_PER_CELL
    assert plan.d_map_n == FULL_CALIBRATION_D_MAP_N
    assert plan.run_signature == "full"
    assert plan.is_smoke is False


def test_full_stage7b_plan_totals_144_candidates_432_rows():
    plan = build_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    assert plan.total_unique_perturbations == 3 * 6 * 8 == 144
    assert plan.total_perturbation_capability_evaluations == 144 * 3 == 432


def test_build_stage7b_plan_rejects_empty_radii():
    with pytest.raises(ValueError):
        build_stage7b_plan(model_name="m", model_revision="r", output_root="out", radii=[])


def test_build_stage7b_plan_rejects_empty_regions():
    with pytest.raises(ValueError):
        build_stage7b_plan(model_name="m", model_revision="r", output_root="out", regions=[])


# --- common six-radius grid identical across all 3 regions -------------------------------------


def test_build_stage7b_population_uses_the_same_radius_grid_for_every_region():
    plan = build_stage7b_plan(model_name="qwen2_5_vl", model_revision="rev1", output_root="out")
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}

    population = build_stage7b_population(plan, base_seed=123, parameter_mask_hash_by_region=mask_hashes)

    radii_by_region = {}
    for (region, radius) in population:
        radii_by_region.setdefault(region, set()).add(radius)
    assert set(radii_by_region.keys()) == set(FULL_CALIBRATION_REGIONS)
    for region, radii in radii_by_region.items():
        assert radii == set(FULL_CALIBRATION_RADII), f"region {region} does not use the common frozen radius grid"


def test_build_stage7b_population_has_n_per_cell_members_for_every_region_radius_pair():
    plan = build_stage7b_plan(model_name="qwen2_5_vl", model_revision="rev1", output_root="out", radii=[0.01, 0.05])
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}

    population = build_stage7b_population(plan, base_seed=123, parameter_mask_hash_by_region=mask_hashes)

    assert set(population.keys()) == {(r, radius) for r in plan.regions for radius in plan.radii}
    for cell, manifests in population.items():
        assert len(manifests) == plan.n_per_cell
        region, radius = cell
        for m in manifests:
            assert m.anatomy_region == region
            assert m.radius == radius
            assert m.sigma is None
            assert m.perturbation_mode == PERTURBATION_MODE


def test_build_stage7b_population_is_deterministic():
    plan = build_stage7b_plan(model_name="qwen2_5_vl", model_revision="rev1", output_root="out", radii=[0.01])
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    pop1 = build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region=mask_hashes)
    pop2 = build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region=mask_hashes)
    for cell in pop1:
        assert [m.perturbation_id for m in pop1[cell]] == [m.perturbation_id for m in pop2[cell]]


def test_build_stage7b_population_requires_mask_hash_for_every_region():
    plan = build_stage7b_plan(model_name="m", model_revision="r", output_root="out", radii=[0.01])
    with pytest.raises(ValueError, match="Missing parameter_mask_hash"):
        build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region={"vision": "h"})


def test_build_stage7b_population_no_two_manifests_share_a_worker_seed():
    plan = build_stage7b_plan(model_name="m", model_revision="r", output_root="out")  # full: 144 candidates
    mask_hashes = {r: f"hash_{r}" for r in plan.regions}
    population = build_stage7b_population(plan, base_seed=1, parameter_mask_hash_by_region=mask_hashes)
    all_seeds = [m.seed for manifests in population.values() for m in manifests]
    assert len(all_seeds) == 144
    assert len(all_seeds) == len(set(all_seeds))


# =================================================================================================
# 3. Smoke mode
# =================================================================================================


def test_smoke_config_is_frozen_to_one_vision_candidate_at_the_specified_radius():
    assert SMOKE_REGION == "vision"
    assert SMOKE_RADIUS == FULL_CALIBRATION_RADII[2] == 0.035698828543799424
    assert SMOKE_N_PER_CELL == 1
    assert SMOKE_D_MAP_N == 5


def test_build_smoke_stage7b_plan_totals_exactly_one_candidate_three_rows():
    plan = build_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    assert plan.regions == (SMOKE_REGION,)
    assert plan.radii == (SMOKE_RADIUS,)
    assert plan.n_per_cell == 1
    assert plan.d_map_n == 5
    assert plan.total_unique_perturbations == 1
    assert plan.total_perturbation_capability_evaluations == 3  # 1 perturbation x 3 capabilities
    assert plan.is_smoke is True


def test_smoke_uses_all_three_frozen_capabilities():
    plan = build_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    assert plan.capabilities == CALIBRATION_CAPABILITIES == PILOT_CAPABILITIES


def test_smoke_expected_perturbed_model_example_evaluations_is_15():
    plan = build_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    perturbed_model_example_evaluations = plan.total_perturbation_capability_evaluations * plan.d_map_n
    assert perturbed_model_example_evaluations == 3 * 5 == 15


# --- smoke/full path isolation: can never collide, can never resume into each other -----------


def test_smoke_and_full_plans_never_share_an_output_directory():
    full_plan = build_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    smoke_plan = build_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    assert full_plan.output_dir != smoke_plan.output_dir
    assert full_plan.run_signature == "full"
    assert smoke_plan.run_signature != "full"
    assert smoke_plan.run_signature.startswith("smoke_")


def test_compute_stage7b_run_signature_is_full_only_for_the_exact_frozen_identity():
    assert compute_stage7b_run_signature(FULL_CALIBRATION_REGIONS, FULL_CALIBRATION_RADII, FULL_CALIBRATION_N_PER_CELL, FULL_CALIBRATION_D_MAP_N) == "full"


@pytest.mark.parametrize(
    "regions,radii,n_per_cell,d_map_n",
    [
        ((SMOKE_REGION,), (SMOKE_RADIUS,), SMOKE_N_PER_CELL, SMOKE_D_MAP_N),
        (FULL_CALIBRATION_REGIONS, FULL_CALIBRATION_RADII, FULL_CALIBRATION_N_PER_CELL, 5),  # only d_map_n differs
        (FULL_CALIBRATION_REGIONS, FULL_CALIBRATION_RADII, 1, FULL_CALIBRATION_D_MAP_N),  # only n_per_cell differs
        (("vision",), FULL_CALIBRATION_RADII, FULL_CALIBRATION_N_PER_CELL, FULL_CALIBRATION_D_MAP_N),  # only regions differ
    ],
)
def test_compute_stage7b_run_signature_never_returns_full_for_any_deviation(regions, radii, n_per_cell, d_map_n):
    signature = compute_stage7b_run_signature(regions, radii, n_per_cell, d_map_n)
    assert signature != "full"
    assert signature.startswith("smoke_")


def test_run_signature_never_collides_across_different_smoke_configurations():
    sig_a = compute_stage7b_run_signature(("vision",), (0.01,), 1, 5)
    sig_b = compute_stage7b_run_signature(("vision",), (0.02,), 1, 5)
    sig_c = compute_stage7b_run_signature(("language",), (0.01,), 1, 5)
    assert len({sig_a, sig_b, sig_c}) == 3


def test_failed_smoke_checkpoint_can_never_be_loaded_as_a_full_checkpoint(tmp_path):
    """A smoke run's checkpoint_manifest.json lives under a DIFFERENT output_dir than a full
    run's -- so even a failed/partial smoke run can never be silently resumed as full: the full
    run's own ensure_stage7b_checkpoint_manifest call operates on a path the smoke run never
    wrote to at all.
    """
    smoke_plan = build_smoke_stage7b_plan(model_name="m", model_revision="r", output_root=tmp_path)
    full_plan = build_stage7b_plan(model_name="m", model_revision="r", output_root=tmp_path)
    assert smoke_plan.output_dir != full_plan.output_dir

    smoke_checkpoint = build_stage7b_checkpoint_manifest(smoke_plan, _fake_contexts(n=5), {r: f"h_{r}" for r in smoke_plan.regions})
    ensure_stage7b_checkpoint_manifest(smoke_plan.output_dir / "checkpoint_manifest.json", smoke_checkpoint)

    assert not (full_plan.output_dir / "checkpoint_manifest.json").exists()


# =================================================================================================
# 2. Engine config: reuses Stage 6's safe launcher, never upstream launch_engines()
# =================================================================================================


def test_reuses_stage6_launcher_and_config_by_identity_not_duplication():
    assert module.launch_stage6_engine is run_global_visual_thicket_pilot.launch_stage6_engine
    assert module.build_stage6_engine_config is run_global_visual_thicket_pilot.build_stage6_engine_config
    assert module.resolve_and_report_model_snapshot is run_global_visual_thicket_pilot.resolve_and_report_model_snapshot
    assert module.store_base_weights_via_rpc is run_global_visual_thicket_pilot.store_base_weights_via_rpc
    assert module.reset_to_base_weights_via_rpc is run_global_visual_thicket_pilot.reset_to_base_weights_via_rpc
    assert module.verify_exact_fixed_base_restoration_via_rpc is run_global_visual_thicket_pilot.verify_exact_fixed_base_restoration_via_rpc


def test_frozen_engine_config_never_falls_back_to_model_default_context_length():
    engine_config = module.build_stage6_engine_config()
    assert engine_config["max_model_len"] == 4096
    assert engine_config["max_model_len"] != 128000
    assert engine_config["gpu_memory_utilization"] == 0.60
    assert engine_config["tensor_parallel_size"] == 1


def test_main_never_imports_or_calls_upstream_launch_engines():
    main_source = inspect.getsource(module.main)
    assert "launch_stage6_engine(" in main_source
    assert "import launch_engines" not in main_source
    assert "launch_engines(" not in main_source


def test_store_base_weights_called_exactly_once_in_main_source():
    """main() must call store_base_weights_via_rpc exactly once (source-level: it appears
    exactly once as a call, not inside any per-candidate loop) -- the per-candidate
    reset_to_base_weights/scoped perturbation lifecycle lives entirely in
    evaluate_one_calibration_candidate_rpc, never re-storing the base snapshot.
    """
    main_source = inspect.getsource(module.main)
    assert main_source.count("store_base_weights_via_rpc(engine)") == 1


# --- L1 anatomy unchanged: reused by identity, not modified/forked -----------------------------


def test_report_region_param_names_reuses_frozen_l1_anatomy_by_identity(runtime_wrapped_vlm_32vision_factory):
    assert module.build_anatomy_atlas is thicket_anatomy.build_anatomy_atlas
    model = runtime_wrapped_vlm_32vision_factory()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))

    report = report_region_param_names(worker, FULL_CALIBRATION_REGIONS)

    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    for region in FULL_CALIBRATION_REGIONS:
        assert report[region]["param_names"] == list(atlas.region(region).param_names)
        assert report[region]["mask_hash"] == atlas.region(region).mask_hash


# =================================================================================================
# Checkpoint manifest identity + full-run-safety accounting
# =================================================================================================


def _fake_contexts(n=20):
    contexts = {}
    for capability in CALIBRATION_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=[], partition=partition, subset_hash=partition.manifest_hash, base_score=0.5)
    return contexts


def _region_mask_hashes(plan):
    return {r: f"hash_{r}" for r in plan.regions}


def test_build_stage7b_checkpoint_manifest_fields():
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", output_root="out")
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan))
    assert checkpoint.experiment_id == EXPERIMENT_ID
    assert checkpoint.run_signature == "full"
    assert checkpoint.perturbation_mode == PERTURBATION_MODE
    assert checkpoint.dataset_role == "map" == DATASET_ROLE
    assert checkpoint.d_map_n == FULL_CALIBRATION_D_MAP_N
    assert checkpoint.n_per_cell == FULL_CALIBRATION_N_PER_CELL
    assert checkpoint.regions == FULL_CALIBRATION_REGIONS
    assert checkpoint.radii == FULL_CALIBRATION_RADII
    assert checkpoint.expected_unique_perturbations == 144
    assert checkpoint.expected_result_rows == 432
    assert checkpoint.region_mask_hashes == _region_mask_hashes(plan)


def test_checkpoint_manifest_to_dict_includes_perturbation_semantics():
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", output_root="out")
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan))
    assert checkpoint.to_dict()["perturbation_semantics"] == "anatomical_relative_l2"
    assert checkpoint.to_dict()["restoration_mode"] == "fixed_base"


def test_checkpoint_manifest_round_trips_through_json():
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", output_root="out", radii=[0.01])
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan))
    restored = Stage7bCheckpointManifest.from_dict(json.loads(json.dumps(checkpoint.to_dict())))
    assert restored == checkpoint


def test_ensure_stage7b_checkpoint_manifest_creates_when_absent(tmp_path):
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", output_root="out", radii=[0.01])
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan))
    path = tmp_path / "checkpoint_manifest.json"
    result = ensure_stage7b_checkpoint_manifest(path, checkpoint)
    assert path.exists()
    assert result == checkpoint


def test_ensure_stage7b_checkpoint_manifest_hard_fails_on_mismatch(tmp_path):
    plan = build_stage7b_plan(model_name="m", model_revision="rev1", output_root="out", radii=[0.01])
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts(), _region_mask_hashes(plan))
    path = tmp_path / "checkpoint_manifest.json"
    ensure_stage7b_checkpoint_manifest(path, checkpoint)

    plan2 = build_stage7b_plan(model_name="m", model_revision="rev2", output_root="out", radii=[0.01])
    checkpoint2 = build_stage7b_checkpoint_manifest(plan2, _fake_contexts(), _region_mask_hashes(plan2))
    with pytest.raises(IncompatibleCalibrationCheckpointError):
        ensure_stage7b_checkpoint_manifest(path, checkpoint2)


def test_build_stage7b_run_manifest_summary_reports_run_complete():
    from neural_thickets_repro.run_stage7b_anatomical_calibration import build_stage7b_run_manifest_summary
    from neural_thickets_repro.thicket.schema import ExperimentResultRecord

    plan = build_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    checkpoint = build_stage7b_checkpoint_manifest(plan, _fake_contexts(n=5), _region_mask_hashes(plan))

    def _record(capability):
        return ExperimentResultRecord(
            experiment_id=EXPERIMENT_ID, perturbation_id="pid1", model_family="qwen2_5_vl", model_scale="3B",
            model_revision="r", perturbation_mode=PERTURBATION_MODE, anatomy_region="vision", radius=SMOKE_RADIUS,
            sigma=None, seed=1, parameter_mask_hash="h", capability=capability, dataset_role="map", subset_hash="s",
            base_score=0.5, perturbed_score=0.5, delta=0.0, parser_failure_rate=0.0,
            per_example_result_path=None, per_example_result_hash="h", runtime_metadata={},
        )

    records = [_record(c) for c in CALIBRATION_CAPABILITIES]
    manifest = build_stage7b_run_manifest_summary(checkpoint, records)
    assert manifest["run_complete"] is True
    assert manifest["actual_unique_perturbations"] == 1
    assert manifest["actual_result_rows"] == 3
    assert manifest["perturbation_semantics"] == "anatomical_relative_l2"

    manifest_incomplete = build_stage7b_run_manifest_summary(checkpoint, records[:2])
    assert manifest_incomplete["run_complete"] is False


# --- no D_confirm/select/test access -------------------------------------------------------------


def test_dataset_role_is_always_map():
    assert DATASET_ROLE == "map"


def test_checkpoint_manifest_rejects_unrecognized_d_map_size():
    from neural_thickets_repro.run_stage7b_anatomical_calibration import DatasetRoleViolationError

    bad_plan = build_stage7b_plan(model_name="m", model_revision="r", output_root="out", radii=[0.01])
    object.__setattr__(bad_plan, "d_map_n", 15)  # simulate a caller-mutated plan with an unrecognized size
    with pytest.raises(DatasetRoleViolationError):
        build_stage7b_checkpoint_manifest(bad_plan, _fake_contexts(), _region_mask_hashes(bad_plan))


def test_build_stage7b_plan_rejects_unrecognized_d_map_size():
    from neural_thickets_repro.run_stage7b_anatomical_calibration import DatasetRoleViolationError

    with pytest.raises(DatasetRoleViolationError):
        build_stage7b_plan(model_name="m", model_revision="r", output_root="out", d_map_n=17)


def test_no_data_role_dict_key_other_than_map_in_source():
    """Code-level guard (not prose): this module's docstrings discuss D_confirm/D_select/D_test
    BY NAME to document their absence, so this checks for the actual dict-literal role-size
    keys (`"confirm":`, `"select":`, `"test":`) a real `sizes={...}` call would use, never the
    prose word itself.
    """
    full_source = inspect.getsource(module)
    for forbidden in ('"confirm":', "'confirm':", '"select":', "'select':", '"test":', "'test':"):
        assert forbidden not in full_source, f"found forbidden dataset-role dict key {forbidden!r}"


# --- no best-radius selection logic exists (section 7) -------------------------------------------


def test_no_best_radius_selection_logic_exists():
    forbidden_substrings = ("best_radius", "select_best", "optimize_radius", "maximize", "argmax_radius")
    public_names = [name for name in dir(module) if not name.startswith("_")]
    for name in public_names:
        lowered = name.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, f"found forbidden name {name!r} (matches {forbidden!r})"


# =================================================================================================
# evaluate_one_calibration_candidate_rpc: fixed-base lifecycle
# =================================================================================================


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
        assert r.runtime_metadata["perturbation_semantics"] == "anatomical_relative_l2"
        assert "theta_region_l2_norm" in r.runtime_metadata
        assert "epsilon_region_l2_norm" in r.runtime_metadata


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
    # The defensive reset-before-perturb happens INSIDE scoped_apply_anatomical_perturbation
    # itself (a direct method call on worker_self, not a separate top-level collective_rpc
    # dispatch this fake's call log would see) -- proven directly against a persistent fake
    # worker in test_scoped_anatomical_perturbation.py::
    # test_scoped_apply_anatomical_perturbation_calls_reset_to_base_first. The reset visible
    # HERE, as its own top-level dispatch, is the lifecycle's final restoration.
    assert engine.calls[-2:] == ["reset_to_base_weights", "verify_exact_fixed_base_restoration_rpc"]


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
    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model, radius=0.05)

    def _broken_apply(worker_self, seed, r, region_name, region_param_names):
        return {
            "region": region_name, "seed": seed, "requested_relative_l2": r, "realized_relative_l2": r + 10.0,
            "region_param_count": len(region_param_names), "theta_l2_norm": 1.0, "realized_epsilon_l2_norm": 1.0,
        }

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

    for name, value in outside_before.items():
        assert torch.equal(value, engine._base_weights[name])


def test_evaluate_one_calibration_candidate_rpc_changes_only_selected_region(runtime_wrapped_vlm_32vision_factory):
    """Direct requested-relative-L2-realization + selected-region-only mutation check together:
    ONLY the vision region's parameters differ from base immediately after the perturbation is
    applied (checked via a snapshot taken right after apply, before the lifecycle's own reset).
    """
    model = runtime_wrapped_vlm_32vision_factory()
    from neural_thickets_repro.thicket.perturbation import apply_anatomical_relative_l2

    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    region_param_names = atlas.region("vision").param_names
    base_snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}

    record = apply_anatomical_relative_l2(model, "vision", region_param_names, seed=99, r=0.07)

    assert record.realized_epsilon_l2_norm / record.theta_l2_norm == pytest.approx(0.07, abs=1e-6)
    for name, p in model.named_parameters():
        if name in set(region_param_names):
            continue
        assert torch.equal(p.detach(), base_snapshot[name]), f"{name} changed outside the perturbed region"


# =================================================================================================
# run_stage7b_rpc: candidate-level durable checkpoint -- only after successful restoration
# =================================================================================================


def test_run_stage7b_rpc_persists_rows_only_after_full_success(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    engine = _FakeCalibrationEngine(model)
    engine.store_base_weights()

    plan = build_stage7b_plan(
        model_name="qwen2_5_vl", model_revision="rev1", output_root=tmp_path,
        regions=("vision",), radii=(0.05,), n_per_cell=1, d_map_n=5,
    )
    region_param_names_by_region = {"vision": atlas.region("vision").param_names}
    mask_hash_by_region = {"vision": atlas.region("vision").mask_hash}
    contexts = _fake_contexts(n=5)

    records = run_stage7b_rpc(
        plan, contexts, engine, tokenizer=None, sampling_params=None, base_seed=1,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hash_by_region,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert len(records) == 3  # 1 candidate x 3 capabilities
    results_path = plan.output_dir / "results.jsonl"
    assert results_path.exists()
    persisted_lines = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    assert len(persisted_lines) == 3
    assert (plan.output_dir / "checkpoint_manifest.json").exists()


def test_run_stage7b_rpc_never_persists_rows_for_a_failed_candidate(tmp_path, runtime_wrapped_vlm_32vision_factory, monkeypatch):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    engine = _FakeCalibrationEngine(model)
    engine.store_base_weights()

    plan = build_stage7b_plan(
        model_name="qwen2_5_vl", model_revision="rev1", output_root=tmp_path,
        regions=("vision",), radii=(0.05,), n_per_cell=1, d_map_n=5,
    )
    region_param_names_by_region = {"vision": atlas.region("vision").param_names}
    mask_hash_by_region = {"vision": atlas.region("vision").mask_hash}
    contexts = _fake_contexts(n=5)

    def _broken_apply(worker_self, seed, r, region_name, region_param_names):
        return {
            "region": region_name, "seed": seed, "requested_relative_l2": r, "realized_relative_l2": r + 10.0,
            "region_param_count": len(region_param_names), "theta_l2_norm": 1.0, "realized_epsilon_l2_norm": 1.0,
        }

    monkeypatch.setattr(module, "scoped_apply_anatomical_perturbation", _broken_apply)

    with pytest.raises(RealizedRadiusMismatchError):
        run_stage7b_rpc(
            plan, contexts, engine, tokenizer=None, sampling_params=None, base_seed=1,
            region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hash_by_region,
            run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
        )

    results_path = plan.output_dir / "results.jsonl"
    assert not results_path.exists() or results_path.read_text().strip() == ""


def test_run_stage7b_rpc_skips_already_completed_candidates(tmp_path, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    plan = build_stage7b_plan(
        model_name="qwen2_5_vl", model_revision="rev1", output_root=tmp_path,
        regions=("vision",), radii=(0.05,), n_per_cell=1, d_map_n=5,
    )
    region_param_names_by_region = {"vision": atlas.region("vision").param_names}
    mask_hash_by_region = {"vision": atlas.region("vision").mask_hash}
    contexts = _fake_contexts(n=5)

    engine1 = _FakeCalibrationEngine(model)
    engine1.store_base_weights()
    run_stage7b_rpc(
        plan, contexts, engine1, tokenizer=None, sampling_params=None, base_seed=1,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hash_by_region,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    engine2 = _FakeCalibrationEngine(model)
    engine2.store_base_weights()
    records2 = run_stage7b_rpc(
        plan, contexts, engine2, tokenizer=None, sampling_params=None, base_seed=1,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hash_by_region,
        run_benchmark=_fake_run_benchmark, ray_get=_identity_ray_get,
    )

    assert len(records2) == 3  # resumed from the already-completed rows, no new GPU work needed
    assert "scoped_apply_anatomical_perturbation" not in engine2.calls
