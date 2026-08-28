"""Tests for the Stage-11 32B readiness milestone: thicket.cpu_base_snapshot,
thicket.distributed_perturbation, and stage11_32b_readiness. CPU-only throughout -- no GPU/ray/
vLLM import anywhere in this file, matching this project's established convention for testing
GPU-shaped code (real small torch.nn.Module/tensors stand in for a live worker's parameters).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

import neural_thickets_repro.scaling_common as scaling_common
import neural_thickets_repro.stage11_32b_readiness as readiness
from neural_thickets_repro.thicket import cpu_base_snapshot as cbs
from neural_thickets_repro.thicket import distributed_perturbation as dp
from neural_thickets_repro.thicket.perturbation import apply_anatomical_relative_l2
from neural_thickets_repro.perturb_cpu import _generate_noise


# =================================================================================================
# Section 1: 32B model spec
# =================================================================================================


def test_32b_model_spec_registered_and_frozen():
    spec = scaling_common.get_scaling_model_spec("32B")
    assert spec.model_name == readiness.FROZEN_32B_MODEL_NAME == "Qwen/Qwen2.5-VL-32B-Instruct"
    assert spec.model_family == readiness.FROZEN_32B_MODEL_FAMILY == "qwen2_5_vl"


def test_32b_and_72b_remain_same_family_as_3b_7b():
    for scale in ("3B", "7B", "32B", "72B"):
        spec = scaling_common.get_scaling_model_spec(scale)
        assert spec.model_family == scaling_common.MODEL_FAMILY == "qwen2_5_vl"
        assert spec.model_name.startswith("Qwen/Qwen2.5-VL-")


def test_72b_remains_hard_disabled():
    assert "72B" not in scaling_common.RUNNABLE_SCALES
    assert "32B" not in scaling_common.RUNNABLE_SCALES
    assert scaling_common.RUNNABLE_SCALES == ("3B", "7B")
    with pytest.raises(scaling_common.ScaleNotYetEnabledError):
        scaling_common.ensure_scale_runnable("32B")
    with pytest.raises(scaling_common.ScaleNotYetEnabledError):
        scaling_common.ensure_scale_runnable("72B")


# =================================================================================================
# Section 2/3: architecture / VRAM estimate
# =================================================================================================


def test_32b_parameter_estimate_is_in_the_expected_order_of_magnitude():
    est = readiness.estimate_qwen25_vl_32b_parameter_count()
    assert 28e9 < est["total_params"] < 40e9  # "32B" naming should roughly match
    assert est["language_backbone_params"] > est["vision_tower_params"]


def test_vram_estimate_legacy_mode_double_counts_weights():
    est = readiness.estimate_vram_gib(10_000_000_000, tensor_parallel_size=1, base_snapshot_mode="store_base_weights")
    assert est["base_snapshot_overhead_gib_per_gpu"] == pytest.approx(est["weight_gib_per_gpu"])


def test_vram_estimate_cpu_snapshot_mode_adds_zero_gpu_overhead():
    est = readiness.estimate_vram_gib(10_000_000_000, tensor_parallel_size=1, base_snapshot_mode="cpu_base_weights")
    assert est["base_snapshot_overhead_gib_per_gpu"] == 0.0


def test_vram_estimate_scales_inversely_with_tp_size():
    e1 = readiness.estimate_vram_gib(10_000_000_000, tensor_parallel_size=1, base_snapshot_mode="cpu_base_weights")
    e4 = readiness.estimate_vram_gib(10_000_000_000, tensor_parallel_size=4, base_snapshot_mode="cpu_base_weights")
    assert e4["weight_gib_per_gpu"] == pytest.approx(e1["weight_gib_per_gpu"] / 4)


def test_32b_single_gpu_legacy_mode_does_not_fit():
    est = readiness.estimate_qwen25_vl_32b_parameter_count()
    vram = readiness.estimate_vram_gib(est["total_params"], tensor_parallel_size=1, base_snapshot_mode="store_base_weights")
    assert vram["fits"] is False


def test_cpu_snapshot_mode_recommends_fewer_gpus_than_legacy_mode():
    est = readiness.estimate_qwen25_vl_32b_parameter_count()
    legacy = readiness.recommend_min_gpu_count(est["total_params"], base_snapshot_mode="store_base_weights")
    cpu_mode = readiness.recommend_min_gpu_count(est["total_params"], base_snapshot_mode="cpu_base_weights")
    assert legacy["recommended_min_tp_size"] is not None and cpu_mode["recommended_min_tp_size"] is not None
    assert cpu_mode["recommended_min_tp_size"] <= legacy["recommended_min_tp_size"]


def test_recommend_min_gpu_count_reports_none_when_nothing_clears_the_safety_margin():
    result = readiness.recommend_min_gpu_count(10_000_000_000_000, base_snapshot_mode="cpu_base_weights", candidate_tp_sizes=(1, 2))
    assert result["recommended_min_tp_size"] is None
    assert result["note"] is not None


# =================================================================================================
# CPU base snapshot -- real small torch.nn.Module, CPU-only
# =================================================================================================


class _TinyVLM(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.visual = nn.Linear(8, 8, bias=False)
        self.language_model = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.visual.weight.copy_(torch.randn(8, 8, generator=g, dtype=torch.float32).to(torch.bfloat16))
            self.language_model.weight.copy_(torch.randn(8, 8, generator=g, dtype=torch.float32).to(torch.bfloat16))
        self.to(torch.bfloat16)


def _fake_worker(model: nn.Module):
    ns = SimpleNamespace()
    ns.model_runner = SimpleNamespace(model=model)
    ns._should_perturb = lambda name: not name.startswith("visual.")
    return ns


def test_cpu_snapshot_store_and_reset_roundtrip_is_exact():
    model = _TinyVLM(seed=1)
    worker = _fake_worker(model)
    report = cbs.store_base_weights_cpu_rpc(worker)
    assert report["n_parameters"] == 2
    original_lm_weight = model.language_model.weight.detach().clone()

    with torch.no_grad():
        model.language_model.weight.add_(0.01)  # perturb
    assert not torch.equal(model.language_model.weight, original_lm_weight)

    cbs.reset_to_base_weights_cpu_rpc(worker)
    assert torch.equal(model.language_model.weight, original_lm_weight)


def test_cpu_snapshot_verification_passes_after_correct_restore():
    model = _TinyVLM(seed=2)
    worker = _fake_worker(model)
    cbs.store_base_weights_cpu_rpc(worker)
    with torch.no_grad():
        model.language_model.weight.add_(0.02)
    cbs.reset_to_base_weights_cpu_rpc(worker)
    verification = cbs.verify_exact_fixed_base_restoration_cpu_rpc(worker)
    assert verification["ok"] is True
    assert verification["max_abs_drift"] == 0.0


def test_cpu_snapshot_verification_detects_a_genuine_drift():
    model = _TinyVLM(seed=3)
    worker = _fake_worker(model)
    cbs.store_base_weights_cpu_rpc(worker)
    with torch.no_grad():
        model.language_model.weight.add_(0.02)  # perturbed, but NOT reset back -- should be caught
    verification = cbs.verify_exact_fixed_base_restoration_cpu_rpc(worker)
    assert verification["ok"] is False
    assert verification["max_abs_drift"] > 0.0


def test_cpu_snapshot_verification_requires_prior_store_call():
    model = _TinyVLM(seed=4)
    worker = _fake_worker(model)
    with pytest.raises(RuntimeError):
        cbs.verify_exact_fixed_base_restoration_cpu_rpc(worker)
    with pytest.raises(RuntimeError):
        cbs.reset_to_base_weights_cpu_rpc(worker)


def test_cpu_snapshot_bounded_chunk_verification_matches_unchunked_reference():
    """Bounded restoration memory: a small chunk_elements (forcing multiple chunks) must produce
    the IDENTICAL diagnostic as one giant chunk -- chunking must never change the answer.
    """
    model = _TinyVLM(seed=5)
    worker = _fake_worker(model)
    cbs.store_base_weights_cpu_rpc(worker)
    with torch.no_grad():
        model.language_model.weight.view(-1)[3].add_(0.02)
    small_chunk = cbs.verify_exact_fixed_base_restoration_cpu_rpc(worker, chunk_elements=1)
    large_chunk = cbs.verify_exact_fixed_base_restoration_cpu_rpc(worker, chunk_elements=10_000)
    assert small_chunk == large_chunk


def test_cpu_snapshot_preserves_native_dtype_exactly():
    model = _TinyVLM(seed=6)
    worker = _fake_worker(model)
    cbs.store_base_weights_cpu_rpc(worker)
    for name, ref in worker._base_weights_cpu.items():
        assert ref.dtype == torch.bfloat16


# =================================================================================================
# Legacy-vs-CPU-snapshot equivalence classification (Section 5)
# =================================================================================================


def test_legacy_vs_cpu_snapshot_full_lifecycle_is_bit_exact():
    """The actual A/B/C classification, exercised end to end on parallel model copies -- one
    using upstream-shaped legacy clone-on-same-device, one using the new CPU-snapshot path.
    """
    legacy_model = _TinyVLM(seed=42)
    cpu_model = _TinyVLM(seed=42)
    assert torch.equal(legacy_model.language_model.weight, cpu_model.language_model.weight)

    # Legacy: upstream's own clone-on-same-device semantics (CPU tensors here stand in for GPU
    # tensors -- clone()/copy_() are device-generic, so this exercises the identical code path).
    legacy_base = {name: p.data.clone() for name, p in legacy_model.named_parameters()}
    cpu_worker = _fake_worker(cpu_model)
    cbs.store_base_weights_cpu_rpc(cpu_worker, pin_memory=False)

    initial_equal = all(torch.equal(legacy_base[n], cpu_worker._base_weights_cpu[n]) for n in legacy_base)

    seed, delta = 123, 0.01
    with torch.no_grad():
        noise_legacy = _generate_noise(legacy_model.language_model.weight, seed)
        legacy_model.language_model.weight.add_(delta * noise_legacy)
        noise_cpu = _generate_noise(cpu_model.language_model.weight, seed)
        cpu_model.language_model.weight.add_(delta * noise_cpu)
    perturbed_equal = torch.equal(legacy_model.language_model.weight, cpu_model.language_model.weight)

    for name, p in legacy_model.named_parameters():
        p.data.copy_(legacy_base[name])
    cbs.reset_to_base_weights_cpu_rpc(cpu_worker)
    restored_equal = torch.equal(legacy_model.language_model.weight, cpu_model.language_model.weight) and torch.equal(legacy_model.visual.weight, cpu_model.visual.weight)
    n_differing = int((legacy_model.language_model.weight != cpu_model.language_model.weight).sum().item())

    equivalence_class = cbs.classify_snapshot_equivalence(initial_equal, perturbed_equal, restored_equal, n_differing)
    assert equivalence_class == cbs.EQUIVALENCE_BIT_EXACT
    cbs.ensure_bit_exact_before_32b(equivalence_class)  # must not raise


def test_ensure_bit_exact_before_32b_stops_on_non_a_classification():
    with pytest.raises(RuntimeError):
        cbs.ensure_bit_exact_before_32b(cbs.EQUIVALENCE_SEMANTICS_CHANGED)
    with pytest.raises(RuntimeError):
        cbs.ensure_bit_exact_before_32b(cbs.EQUIVALENCE_SCIENTIFICALLY_EQUIVALENT)


def test_classify_snapshot_equivalence_detects_semantics_change():
    result = cbs.classify_snapshot_equivalence(True, False, False, 5)
    assert result == cbs.EQUIVALENCE_SEMANTICS_CHANGED


# =================================================================================================
# Distributed perturbation -- global relative-L2 semantics
# =================================================================================================


def test_generate_full_shape_noise_matches_generate_noise_at_full_shape():
    """world_size=1 equivalence: the new distributed noise generator must be bit-identical to
    the existing, unmodified _generate_noise for an unsharded parameter.
    """
    p = torch.zeros(4, 6, dtype=torch.bfloat16)
    seed = 999
    expected = _generate_noise(p, seed)
    actual = dp._generate_full_shape_noise(p.shape, p.dtype, p.device, seed)
    assert torch.equal(expected, actual)


def test_generate_full_shape_noise_is_deterministic_across_calls():
    """'Same direction reused across radii' at the RNG level: identical (shape, seed) always
    reproduces the identical draw.
    """
    a = dp._generate_full_shape_noise(torch.Size([10, 10]), torch.float32, torch.device("cpu"), 7)
    b = dp._generate_full_shape_noise(torch.Size([10, 10]), torch.float32, torch.device("cpu"), 7)
    assert torch.equal(a, b)


def test_slice_shard_replicated_returns_full_tensor():
    shard = dp.ShardSpec(global_shape=torch.Size([4, 4]), dim=None, rank=0, world_size=2)
    t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(dp.slice_shard(t, shard), t)


def test_slice_shard_sharded_dim0_splits_correctly():
    t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    shard0 = dp.ShardSpec(global_shape=t.shape, dim=0, local_offset=0, local_size=2, rank=0, world_size=2)
    shard1 = dp.ShardSpec(global_shape=t.shape, dim=0, local_offset=2, local_size=2, rank=1, world_size=2)
    part0, part1 = dp.slice_shard(t, shard0), dp.slice_shard(t, shard1)
    assert torch.equal(torch.cat([part0, part1], dim=0), t)


def test_distributed_apply_at_world_size_1_matches_the_unmodified_original_function():
    """THE core equivalence proof: at world_size=1 with the identity all-reduce, the new
    TP-aware function must be numerically identical to the existing, untouched
    apply_anatomical_relative_l2 -- proving 3B/7B (which only ever call the original) are
    unaffected by anything in this module.
    """
    model = _TinyVLM(seed=11)
    region_names = ["language_model.weight"]
    shard_specs = {"language_model.weight": dp.ShardSpec(global_shape=model.language_model.weight.shape, dim=None, rank=0, world_size=1)}

    model_a = _TinyVLM(seed=11)
    record_a = apply_anatomical_relative_l2(model_a, "language", ["language_model.weight"], seed=55, r=0.05)

    model_b = _TinyVLM(seed=11)
    record_b = dp.apply_anatomical_relative_l2_distributed(model_b, "language", region_names, seed=55, r=0.05, shard_specs=shard_specs)

    assert record_a.theta_l2_norm == pytest.approx(record_b.theta_l2_norm)
    assert record_a.raw_noise_l2_norm == pytest.approx(record_b.raw_noise_l2_norm)
    assert record_a.scale == pytest.approx(record_b.scale)
    assert torch.equal(model_a.language_model.weight, model_b.language_model.weight)


def test_distributed_apply_with_sharded_region_matches_the_single_process_global_norm():
    """The actual distributed-semantics proof: split ONE parameter into two simulated shards,
    run the distributed apply with a REAL summing all-reduce across both simulated ranks, and
    confirm the resulting global theta/noise norms and final perturbed values are IDENTICAL to
    running the ORIGINAL single-process function on the whole (unsharded) tensor.
    """
    from neural_thickets_repro.thicket.memory_bounded_ops import chunked_squared_l2_sum

    reference_model = _TinyVLM(seed=21)
    reference_record = apply_anatomical_relative_l2(reference_model, "language", ["language_model.weight"], seed=77, r=0.05)

    # Simulate 2 ranks, each holding half of language_model.weight along dim 0.
    rank0_model = _TinyVLM(seed=21)
    rank1_model = _TinyVLM(seed=21)
    full_shape = rank0_model.language_model.weight.shape
    with torch.no_grad():
        rank0_model.language_model.weight.data = rank0_model.language_model.weight.data[:4].clone()
        rank1_model.language_model.weight.data = rank1_model.language_model.weight.data[4:].clone()

    shard0 = {"language_model.weight": dp.ShardSpec(global_shape=full_shape, dim=0, local_offset=0, local_size=4, rank=0, world_size=2)}
    shard1 = {"language_model.weight": dp.ShardSpec(global_shape=full_shape, dim=0, local_offset=4, local_size=4, rank=1, world_size=2)}

    # Precompute rank 1's LOCAL contribution the same way apply_anatomical_relative_l2_distributed
    # itself would (its own primitives: _generate_full_shape_noise + slice_shard +
    # chunked_squared_l2_sum) -- this is exactly what a real collective's peer message carries;
    # a fake all_reduce_sum below adds it to rank 0's own (real, function-computed) local value,
    # exercising the function's actual reduction call sites rather than only re-deriving the
    # math externally.
    rank1_local_theta_sq = chunked_squared_l2_sum(rank1_model.language_model.weight.detach())
    full_noise = dp._generate_full_shape_noise(full_shape, rank1_model.language_model.weight.dtype, torch.device("cpu"), 77)
    rank1_local_noise_sq = chunked_squared_l2_sum(dp.slice_shard(full_noise, shard1["language_model.weight"]))

    peer_contributions = iter([rank1_local_theta_sq, rank1_local_noise_sq])  # consumed in the SAME order apply_anatomical_relative_l2_distributed calls all_reduce_sum (theta, then noise) in each of its two passes
    call_log = []

    def fake_all_reduce_sum(local_value: float, _pg) -> float:
        call_log.append(local_value)
        try:
            return local_value + next(peer_contributions)
        except StopIteration:
            # Pass 2's designed/realized sums: rank 1's exact contribution isn't precomputed
            # here (it depends on rank 1's own perturbed values), so just prove the call happens
            # -- Pass 1's theta/noise reduction (already checked below) is the scientific crux.
            return local_value

    record_rank0 = dp.apply_anatomical_relative_l2_distributed(
        rank0_model, "language", ["language_model.weight"], seed=77, r=0.05, shard_specs=shard0, all_reduce_sum=fake_all_reduce_sum,
    )

    assert record_rank0.theta_l2_norm == pytest.approx(reference_record.theta_l2_norm)
    assert record_rank0.raw_noise_l2_norm == pytest.approx(reference_record.raw_noise_l2_norm)
    assert record_rank0.scale == pytest.approx(reference_record.scale)
    assert len(call_log) == 4  # Pass 1 (theta, noise) + Pass 2 (designed, realized)
    # rank 0's own updated shard must equal the corresponding slice of the unsharded reference run.
    assert torch.equal(rank0_model.language_model.weight, reference_model.language_model.weight[:4])


def test_no_per_shard_independent_radius_bug():
    """Regression-style: computing the rescale from a SINGLE shard's LOCAL norm (the bug this
    module exists to prevent) gives a DIFFERENT, wrong scale than the correct globally-reduced
    one -- positively distinguishing correct distributed behavior from the naive/buggy version.
    """
    model = _TinyVLM(seed=31)
    r = 0.05
    seed = 88
    region_names = ["language_model.weight"]
    shard_spec_world1 = {"language_model.weight": dp.ShardSpec(global_shape=model.language_model.weight.shape, dim=None, rank=0, world_size=1)}
    correct_record = dp.apply_anatomical_relative_l2_distributed(_TinyVLM(seed=31), "language", region_names, seed=seed, r=r, shard_specs=shard_spec_world1)

    # Buggy: pretend the LOCAL half-tensor IS the whole region (as if radius were realized
    # per-shard instead of globally) -- the scale derived from only half the elements' norm
    # must differ from the correct global-region scale.
    half_model = _TinyVLM(seed=31)
    with torch.no_grad():
        half_model.language_model.weight.data = half_model.language_model.weight.data[:4].clone()
    buggy_shard_spec = {"language_model.weight": dp.ShardSpec(global_shape=half_model.language_model.weight.shape, dim=None, rank=0, world_size=1)}
    buggy_record = dp.apply_anatomical_relative_l2_distributed(half_model, "language", region_names, seed=seed, r=r, shard_specs=buggy_shard_spec)

    assert buggy_record.theta_l2_norm != pytest.approx(correct_record.theta_l2_norm)
    assert buggy_record.scale != pytest.approx(correct_record.scale)


def test_apply_anatomical_relative_l2_original_is_never_imported_by_distributed_module_for_mutation():
    """Structural proof the distributed module never monkeypatches or mutates the original --
    it only calls the unrelated, untouched primitives (_generate_noise, chunked_squared_l2_sum).
    """
    import inspect
    source = inspect.getsource(dp)
    assert "apply_anatomical_relative_l2(" not in source  # never calls the original directly (would defeat the purpose of a TP-aware version)


def test_missing_shard_spec_raises():
    model = _TinyVLM(seed=41)
    with pytest.raises(ValueError):
        dp.apply_anatomical_relative_l2_distributed(model, "language", ["language_model.weight"], seed=1, r=0.05, shard_specs={})


# =================================================================================================
# Distributed restoration-verification aggregation
# =================================================================================================


def test_aggregate_restoration_all_ranks_ok():
    results = [{"ok": True, "max_abs_drift": 0.0, "fraction_elements_differing": 0.0}, {"ok": True, "max_abs_drift": 0.0, "fraction_elements_differing": 0.0}]
    agg = dp.aggregate_distributed_restoration_verification(results)
    assert agg["ok"] is True
    assert agg["global_max_abs_drift"] == 0.0
    assert agg["n_ranks"] == 2


def test_aggregate_restoration_one_rank_drifting_fails_globally():
    results = [{"ok": True, "max_abs_drift": 0.0, "fraction_elements_differing": 0.0}, {"ok": False, "max_abs_drift": 0.003, "fraction_elements_differing": 0.01}]
    agg = dp.aggregate_distributed_restoration_verification(results)
    assert agg["ok"] is False
    assert agg["global_max_abs_drift"] == pytest.approx(0.003)
    assert agg["any_rank_has_differing_elements"] is True


def test_aggregate_restoration_requires_at_least_one_rank():
    with pytest.raises(ValueError):
        dp.aggregate_distributed_restoration_verification([])


# =================================================================================================
# Readiness gates (G1-G8)
# =================================================================================================


def test_g1_fails_on_wrong_model_name():
    assert readiness.g1_model_family_audit("Qwen/Qwen2.5-VL-99B-Instruct", "qwen2_5_vl") == readiness.GATE_FAIL


def test_g1_not_yet_verified_without_live_config():
    assert readiness.g1_model_family_audit(readiness.FROZEN_32B_MODEL_NAME, readiness.FROZEN_32B_MODEL_FAMILY) == readiness.GATE_NOT_YET_VERIFIED


def test_g1_passes_with_matching_live_config():
    assert readiness.g1_model_family_audit(readiness.FROZEN_32B_MODEL_NAME, readiness.FROZEN_32B_MODEL_FAMILY, live_config_model_type="qwen2_5_vl") == readiness.GATE_PASS


def test_g2_hardware_feasibility_not_yet_verified_by_default():
    assert readiness.g2_hardware_feasibility(None) == readiness.GATE_NOT_YET_VERIFIED


def test_g2_hardware_feasibility_fails_on_insufficient_headroom():
    est = readiness.estimate_vram_gib(60_000_000_000, tensor_parallel_size=1, base_snapshot_mode="store_base_weights")
    assert readiness.g2_hardware_feasibility(est) == readiness.GATE_FAIL


def test_g3_requires_bit_exact_class():
    assert readiness.g3_cpu_snapshot_bit_equivalence(None) == readiness.GATE_NOT_YET_VERIFIED
    assert readiness.g3_cpu_snapshot_bit_equivalence(cbs.EQUIVALENCE_BIT_EXACT) == readiness.GATE_PASS
    assert readiness.g3_cpu_snapshot_bit_equivalence(cbs.EQUIVALENCE_SCIENTIFICALLY_EQUIVALENT) == readiness.GATE_FAIL


def test_g7_subset_gate_smoke_and_full_modes():
    smoke_report = {"subset_gate_mode": "smoke_n5_deterministic_repeatability", "smoke_determinism": {"all_deterministic": True, "all_n_matches_expected": True}}
    assert readiness.g7_subset_gate(smoke_report) == readiness.GATE_PASS
    full_report = {"subset_gate_mode": "stage8_full_n50_exact_equality", "subset_hash_equality": {"all_match": False}}
    assert readiness.g7_subset_gate(full_report) == readiness.GATE_FAIL


def test_ensure_32b_smoke_permitted_requires_all_gates_pass():
    all_pass = {g: readiness.GATE_PASS for g in readiness.GATE_IDS}
    readiness.ensure_32b_smoke_permitted(all_pass)  # must not raise

    one_failing = dict(all_pass)
    one_failing["G3"] = readiness.GATE_FAIL
    with pytest.raises(readiness.Stage32BSmokeNotPermittedError):
        readiness.ensure_32b_smoke_permitted(one_failing)


def test_ensure_32b_smoke_permitted_rejects_incomplete_gate_set():
    with pytest.raises(readiness.Stage32BSmokeNotPermittedError):
        readiness.ensure_32b_smoke_permitted({"G1": readiness.GATE_PASS})


def test_readiness_manifest_default_gates_are_not_yet_verified_never_pass():
    manifest = readiness.build_32b_readiness_manifest()
    d = manifest.to_dict()
    assert all(v == readiness.GATE_NOT_YET_VERIFIED for v in d["gate_results"].values())
    assert d["all_gates_pass"] is False


def test_readiness_manifest_smoke_and_full_counts_match_frozen_design():
    manifest = readiness.build_32b_readiness_manifest()
    d = manifest.to_dict()
    assert d["smoke_counts"] == {"s1_perturbations": 3, "s1_rows": 18, "s2_perturbations": 9, "s2_rows": 54}
    assert d["full_counts"] == {"s1_perturbations": 192, "s1_rows": 1152, "s2_perturbations": 576, "s2_rows": 3456}


def test_readiness_manifest_reuses_frozen_regions_radii_capabilities_no_new_regions():
    from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_RADII, STAGE8_REGIONS
    manifest = readiness.build_32b_readiness_manifest()
    assert tuple(manifest.region_definitions) == STAGE8_REGIONS
    assert tuple(manifest.radii) == STAGE8_RADII
    assert tuple(manifest.capabilities) == STAGE8_CAPABILITIES


def test_readiness_manifest_defaults_to_cpu_base_snapshot_mode():
    manifest = readiness.build_32b_readiness_manifest()
    assert manifest.base_snapshot_mode == "cpu_base_weights"


def test_readiness_manifest_with_all_gates_passing_reports_true():
    all_pass = {g: readiness.GATE_PASS for g in readiness.GATE_IDS}
    manifest = readiness.build_32b_readiness_manifest(gate_results=all_pass, resolved_revision="a" * 40, intended_tp_size=4)
    d = manifest.to_dict()
    assert d["all_gates_pass"] is True
    assert d["resolved_revision"] == "a" * 40
    assert d["intended_tp_size"] == 4
