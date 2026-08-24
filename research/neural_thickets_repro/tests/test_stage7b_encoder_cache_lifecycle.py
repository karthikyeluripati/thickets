"""Tests for the Stage 7B multimodal-encoder-cache-reset lifecycle fix (this repair pass).

Root cause this fixes (confirmed at commit 0307f99, see analysis/
stage7b_anatomical_calibration_analysis.py's compute_data_integrity_report and
run_stage7b_anatomical_calibration.py's MULTIMODAL_CACHE_POLICY docstring): vLLM caches its
multimodal-encoder OUTPUT for a given image input; Stage 7B perturbs vision/connector weights
but never invalidated that cache, so vision/connector-perturbed generation silently reused the
BASE model's cached image embeddings -- every vision/connector row in the analyzed run had
delta EXACTLY 0.0 and a collapsed per_example_result_hash across a 100x radius span.

Two layers of coverage here:
  1. A minimal, standalone fixture (_FakeVLLMEncoderCache) that reproduces the ACTUAL failure
     mechanism directly -- cache a representation under one weight state, show it's returned
     stale after a weight change UNLESS reset() is called, matching vLLM's own real
     mm_processor_cache/encoder-cache-manager behavior (see vlm_adapter.py's own docstring for
     the confirmed real-vLLM-source citations).
  2. Integration-level tests against the REAL evaluate_one_calibration_candidate_rpc /
     run_stage7b_rpc (via the same CPU-only fake-engine philosophy as
     tests/test_run_stage7b_anatomical_calibration.py), proving the corrected lifecycle actually
     calls reset_vllm_encoder_cache_full at the right two points for every region type, that a
     failure hard-fails, and that no row is ever checkpointed without both resets having
     succeeded.

The real GPU/Ray/vLLM engine is never launched; ray is not installed in this CPU test
environment, so every call to vlm_adapter.reset_vllm_encoder_cache_full (which does `import ray`
internally) is monkeypatched at the run_stage7b_anatomical_calibration module-attribute level.
"""
from types import SimpleNamespace

import pytest
import torch

import neural_thickets_repro.run_stage7b_anatomical_calibration as module
from neural_thickets_repro.perturb_cpu import should_perturb
from neural_thickets_repro.run_global_visual_thicket_pilot import CapabilityContext, PILOT_CAPABILITIES
from neural_thickets_repro.run_stage7b_anatomical_calibration import (
    EncoderCacheResetUnavailableError,
    PERTURBATION_MODE,
    build_cache_safety_smoke_stage7b_plan,
    build_stage7b_plan,
    ensure_encoder_cache_reset_available,
    ensure_stage7b_encoder_cache_reset_mechanism_exposed,
    evaluate_one_calibration_candidate_rpc,
    run_stage7b_rpc,
)
from neural_thickets_repro.thicket.anatomy import build_anatomy_atlas
from neural_thickets_repro.thicket.data_roles import partition_data_roles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest


def _identity_ray_get(x):
    return x


# =================================================================================================
# 1. Standalone reproduction of the ACTUAL failure mechanism (item 5)
# =================================================================================================


class _FakeVLLMEncoderCache:
    """Minimal stand-in for vLLM's real multimodal-encoder-output cache: keyed by image id,
    returns whatever representation was computed the FIRST time that image was seen, regardless
    of whether the weights used to compute it are still current -- exactly like the real
    scheduler-side encoder_cache_manager / worker-side cached tensor vlm_adapter.py's own
    docstring cites (vllm.v1.core.sched.scheduler.Scheduler.reset_encoder_cache /
    vllm.v1.worker.gpu_worker.Worker.reset_encoder_cache). `reset()` mirrors
    reset_vllm_encoder_cache_full's real effect: clears the cache so the next encode()
    recomputes.
    """

    def __init__(self):
        self._cache = {}

    def encode(self, image_id: str, current_weight_version: str) -> str:
        if image_id not in self._cache:
            self._cache[image_id] = current_weight_version
        return self._cache[image_id]

    def reset(self) -> None:
        self._cache.clear()


def test_stale_cache_reproduces_the_real_bug_without_a_reset():
    """1. base vision encoder result for image X is cached. 2. vision-region weights change.
    3. without reset, the stale cached representation is returned -- reproducing EXACTLY the
    real symptom (generation output invariant to a real, confirmed weight change).
    """
    cache = _FakeVLLMEncoderCache()
    base_result = cache.encode("image_X", current_weight_version="base_weights")
    assert base_result == "base_weights"

    # vision-region weights change (a real, substantial perturbation was applied)
    stale_result = cache.encode("image_X", current_weight_version="vision_perturbed_weights")

    assert stale_result == "base_weights"  # STALE -- the bug, reproduced directly
    assert stale_result != "vision_perturbed_weights"


def test_full_reset_makes_the_encoder_path_recompute_under_changed_weights():
    """4. with full reset, the encoder path recomputes under the changed weights."""
    cache = _FakeVLLMEncoderCache()
    cache.encode("image_X", current_weight_version="base_weights")

    cache.reset()  # the fix
    fresh_result = cache.encode("image_X", current_weight_version="vision_perturbed_weights")

    assert fresh_result == "vision_perturbed_weights"


# =================================================================================================
# 2. ensure_stage7b_encoder_cache_reset_mechanism_exposed / ensure_encoder_cache_reset_available
# =================================================================================================


def test_ensure_stage7b_encoder_cache_reset_mechanism_exposed_succeeds_when_underlying_call_succeeds(monkeypatch):
    monkeypatch.setattr(module, "ensure_full_encoder_cache_reset_exposed", lambda external_root: None)
    ensure_stage7b_encoder_cache_reset_mechanism_exposed("some/external/root")  # must not raise


def test_ensure_stage7b_encoder_cache_reset_mechanism_exposed_hard_fails_when_unavailable(monkeypatch):
    def _broken(external_root):
        raise ImportError("core.engine not importable")
    monkeypatch.setattr(module, "ensure_full_encoder_cache_reset_exposed", _broken)
    with pytest.raises(EncoderCacheResetUnavailableError, match="BEFORE the engine is launched"):
        ensure_stage7b_encoder_cache_reset_mechanism_exposed("some/external/root")


def test_ensure_encoder_cache_reset_available_succeeds_when_underlying_call_succeeds(monkeypatch):
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda engine: None)
    ensure_encoder_cache_reset_available(engine=object())  # must not raise


def test_ensure_encoder_cache_reset_available_hard_fails_when_unavailable(monkeypatch):
    def _broken(engine):
        raise RuntimeError("the engine actor does not expose 'reset_encoder_cache_full'")
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", _broken)
    with pytest.raises(EncoderCacheResetUnavailableError, match="proven-working cache-invalidation path"):
        ensure_encoder_cache_reset_available(engine=object())


# =================================================================================================
# 3. Integration: the REAL Stage-7B lifecycle calls the reset path at the right two points,
#    for EVERY region type, and hard-fails / never checkpoints on cache-reset failure.
# =================================================================================================


class _FakeCalibrationEngine:
    """Same persistent-worker-shaped fake as tests/test_run_stage7b_anatomical_calibration.py's
    own _FakeCalibrationEngine -- duplicated locally (this project's established convention of
    self-contained test files) rather than cross-imported.
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


def _tracking_run_benchmark(engine):
    """A fake run_benchmark that ALSO appends to the engine's call log, so the exact
    interleaving of cache resets vs. capability evaluation can be asserted, not just their
    total counts.
    """
    def _run(benchmark, examples, llm_adapter, tokenizer, sampling_params):
        engine.calls.append("run_benchmark")
        return _FakeRunResult(primary_metric=0.6)
    return _run


def _fake_contexts(n=5):
    contexts = {}
    for capability in PILOT_CAPABILITIES:
        ids = [f"{capability}_{i}" for i in range(n)]
        partition = partition_data_roles(ids, sizes={"map": n}, seed=1)
        contexts[capability] = CapabilityContext(capability=capability, benchmark=None, examples=[], partition=partition, subset_hash=partition.manifest_hash, base_score=0.5)
    return contexts


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


@pytest.mark.parametrize("region", ["vision", "multimodal_connector_or_merger", "language"])
def test_cache_reset_occurs_before_evaluation_and_after_restoration_for_every_region_type(runtime_wrapped_vlm_32vision_factory, monkeypatch, region):
    """The cache reset must fire for ALL region types, including language -- the immediately
    preceding candidate may have perturbed vision/connector and left candidate-specific
    embeddings cached, so this must never depend on which region the CURRENT candidate touches.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model, region=region, radius=0.05)
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda e: e.calls.append("reset_vllm_encoder_cache_full"))

    records = evaluate_one_calibration_candidate_rpc(
        engine, manifest, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
        run_benchmark=_tracking_run_benchmark(engine), ray_get=_identity_ray_get,
    )

    assert len(records) == 3
    reset_indices = [i for i, c in enumerate(engine.calls) if c == "reset_vllm_encoder_cache_full"]
    apply_index = engine.calls.index("scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3")
    first_benchmark_index = engine.calls.index("run_benchmark")
    restoration_index = engine.calls.index("verify_exact_fixed_base_restoration_rpc")

    assert len(reset_indices) == 2, f"expected exactly 2 cache resets for region={region!r}, got call log {engine.calls}"
    # Reset #1: strictly after the accepted perturbation, strictly before ANY capability evaluation.
    assert apply_index < reset_indices[0] < first_benchmark_index
    # Reset #2: strictly after the verified restoration.
    assert reset_indices[1] > restoration_index

    for r in records:
        assert r.runtime_metadata["multimodal_cache_policy"] == module.MULTIMODAL_CACHE_POLICY
        assert r.runtime_metadata["cache_reset_before_evaluation"] is True
        assert r.runtime_metadata["cache_reset_after_restoration"] is True


def test_cache_reset_failure_before_evaluation_hard_fails_and_persists_nothing(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    model = runtime_wrapped_vlm_32vision_factory()
    manifest, engine, region_param_names = _build_manifest_and_engine(model, region="vision", radius=0.05)

    def _broken_reset(e):
        raise RuntimeError("simulated cache-reset failure")
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", _broken_reset)

    with pytest.raises(RuntimeError, match="simulated cache-reset failure"):
        evaluate_one_calibration_candidate_rpc(
            engine, manifest, region_param_names, _fake_contexts(), tokenizer=None, sampling_params=None,
            run_benchmark=_tracking_run_benchmark(engine), ray_get=_identity_ray_get,
        )
    # No capability was ever evaluated -- the cache reset gate blocked it.
    assert "run_benchmark" not in engine.calls


def test_cache_reset_failure_after_restoration_hard_fails_and_the_caller_persists_nothing(tmp_path, runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """If the POST-restoration reset fails, evaluate_one_calibration_candidate_rpc must never
    return records -- so run_stage7b_rpc (which only checkpoints after a successful return)
    persists nothing for this candidate, even though evaluation itself succeeded.
    """
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

    call_count = {"n": 0}

    def _fails_only_the_second_time(e):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated post-restoration cache-reset failure")
        e.calls.append("reset_vllm_encoder_cache_full")

    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", _fails_only_the_second_time)

    with pytest.raises(RuntimeError, match="simulated post-restoration cache-reset failure"):
        run_stage7b_rpc(
            plan, _fake_contexts(), engine, tokenizer=None, sampling_params=None, base_seed=1,
            region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hash_by_region,
            run_benchmark=_tracking_run_benchmark(engine), ray_get=_identity_ray_get,
        )

    results_path = plan.output_dir / "results.jsonl"
    assert not results_path.exists() or results_path.read_text().strip() == ""


def test_cache_reset_brackets_every_candidate_across_a_multi_candidate_run_regardless_of_region_order(tmp_path, runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """Two sequential candidates of DIFFERENT region types (vision then language) -- the reset
    must fire around EACH candidate independently (4 total resets for 2 candidates), never
    relying on candidate ordering to "carry over" a single reset.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    atlas = build_anatomy_atlas([n for n, _ in model.named_parameters()])
    engine = _FakeCalibrationEngine(model)
    engine.store_base_weights()
    monkeypatch.setattr(module, "reset_vllm_encoder_cache_full", lambda e: e.calls.append("reset_vllm_encoder_cache_full"))

    plan = build_stage7b_plan(
        model_name="qwen2_5_vl", model_revision="rev1", output_root=tmp_path,
        regions=("vision", "language"), radii=(0.05,), n_per_cell=1, d_map_n=5,
    )
    region_param_names_by_region = {r: atlas.region(r).param_names for r in plan.regions}
    mask_hash_by_region = {r: atlas.region(r).mask_hash for r in plan.regions}

    engine.calls.clear()
    records = run_stage7b_rpc(
        plan, _fake_contexts(), engine, tokenizer=None, sampling_params=None, base_seed=1,
        region_param_names_by_region=region_param_names_by_region, parameter_mask_hash_by_region=mask_hash_by_region,
        run_benchmark=_tracking_run_benchmark(engine), ray_get=_identity_ray_get,
    )

    assert len(records) == 2 * 3  # 2 candidates x 3 capabilities
    assert engine.calls.count("reset_vllm_encoder_cache_full") == 4  # 2 per candidate x 2 candidates


# =================================================================================================
# 4. Cache policy is part of run/checkpoint identity -- old no-cache-reset checkpoint can't resume
# =================================================================================================


def test_old_no_cache_reset_checkpoint_cannot_resume_into_the_corrected_plan(tmp_path):
    """Simulates the REAL scenario: the old run analyzed at commit 0307f99 was persisted under
    run_signature "full_fixed_direction_bf16_quantization_aware_v3" (no cache-policy suffix at
    all). The corrected plan's run_signature/output_dir must be structurally disjoint, so its
    ensure_stage7b_checkpoint_manifest call never even sees, let alone resumes, that old
    checkpoint.
    """
    old_output_dir = tmp_path / "full_fixed_direction_bf16_quantization_aware_v3"
    old_output_dir.mkdir(parents=True)
    (old_output_dir / "results.jsonl").write_text('{"fake": "old no-cache-reset provenance row"}\n')

    corrected_plan = build_stage7b_plan(model_name="m", model_revision="rev1", output_root=tmp_path)

    assert corrected_plan.output_dir != old_output_dir
    assert corrected_plan.run_signature == "full_fixed_direction_bf16_quantization_aware_v3_cache_reset_v1"
    assert not (corrected_plan.output_dir / "results.jsonl").exists()
    assert (old_output_dir / "results.jsonl").exists()  # old provenance untouched


# =================================================================================================
# 5. Cache-safety smoke config/counts (item 4)
# =================================================================================================


def test_cache_safety_smoke_plan_has_the_exact_specified_shape():
    plan = build_cache_safety_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    assert set(plan.regions) == {"vision", "multimodal_connector_or_merger", "language"}
    assert len(plan.radii) == 1
    assert plan.n_per_cell == 1
    assert plan.d_map_n == 5
    assert plan.total_unique_perturbations == 3
    assert plan.total_perturbation_capability_evaluations == 9
    perturbed_model_example_evaluations = plan.total_perturbation_capability_evaluations * plan.d_map_n
    assert perturbed_model_example_evaluations == 45
    assert plan.is_smoke is True


def test_cache_safety_smoke_output_is_disjoint_from_full_and_execution_smoke():
    cache_smoke = build_cache_safety_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    from neural_thickets_repro.run_stage7b_anatomical_calibration import build_smoke_stage7b_plan
    full_plan = build_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    execution_smoke = build_smoke_stage7b_plan(model_name="m", model_revision="r", output_root="out")
    assert cache_smoke.output_dir not in (full_plan.output_dir, execution_smoke.output_dir)
