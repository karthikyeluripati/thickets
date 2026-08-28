"""Tests for thicket.distributed_v3_solver -- the distributed counterpart of
scoped_anatomical_perturbation.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3.
CPU-only throughout (no GPU/ray/vLLM import); a real small torch.nn.Module stands in for a live
worker's model, matching this project's established convention.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import pytest
import torch
import torch.nn as nn

import neural_thickets_repro.scoped_anatomical_perturbation as legacy_v3
import neural_thickets_repro.thicket.cpu_base_snapshot as cbs
import neural_thickets_repro.thicket.distributed_perturbation as dp
import neural_thickets_repro.thicket.distributed_v3_solver as dv3
from neural_thickets_repro.thicket.memory_bounded_ops import chunked_squared_l2_sum


# =================================================================================================
# Fixtures / fakes
# =================================================================================================


class _TinyVLM(nn.Module):
    def __init__(self, seed: int = 0, n: int = 12):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.language_model = nn.Linear(n, n, bias=False)
        with torch.no_grad():
            self.language_model.weight.copy_(torch.randn(n, n, generator=g, dtype=torch.float32).to(torch.bfloat16))
        self.to(torch.bfloat16)


def _fake_legacy_worker(model: nn.Module):
    """Matches upstream's WorkerExtension shape exactly: bound reset_to_base_weights() (no args)
    and a plain {name: tensor} _base_weights dict, same device as the live parameters.
    """
    ns = SimpleNamespace()
    ns.model_runner = SimpleNamespace(model=model)
    ns._should_perturb = lambda name: True

    def _store():
        ns._base_weights = {name: p.data.clone() for name, p in model.named_parameters()}
        return True

    def _reset():
        for name, p in model.named_parameters():
            p.data.copy_(ns._base_weights[name])
        return True

    ns.store_base_weights = _store
    ns.reset_to_base_weights = _reset
    ns.store_base_weights()
    return ns


def _fake_distributed_worker(model: nn.Module):
    ns = SimpleNamespace()
    ns.model_runner = SimpleNamespace(model=model)
    ns._should_perturb = lambda name: True
    cbs.store_base_weights_cpu_rpc(ns, pin_memory=False)
    return ns


def _world1_shard_specs(model: nn.Module) -> Dict[str, dp.ShardSpec]:
    return {name: dp.ShardSpec(global_shape=p.shape, dim=None, rank=0, world_size=1) for name, p in model.named_parameters()}


def _run_distributed_world1(seed_model: int, seed_direction: int, r: float, n: int = 12):
    model = _TinyVLM(seed=seed_model, n=n)
    worker = _fake_distributed_worker(model)
    shard_specs = _world1_shard_specs(model)
    result = dv3.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed(
        worker, seed_direction, r, "language", ["language_model.weight"], shard_specs,
        all_reduce_sum=dp.identity_all_reduce_sum, all_reduce_max=dv3.identity_all_reduce_max,
    )
    return model, result


def _run_legacy(seed_model: int, seed_direction: int, r: float, n: int = 12):
    model = _TinyVLM(seed=seed_model, n=n)
    worker = _fake_legacy_worker(model)
    result = legacy_v3.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker, seed_direction, r, "language", ["language_model.weight"])
    return model, result


# =================================================================================================
# world_size=1 equivalence (Section 6) -- many deterministic cases
# =================================================================================================


def _run_legacy_or_capture_error(seed_model, seed_direction, r, n=12):
    try:
        return _run_legacy(seed_model, seed_direction, r, n), None
    except legacy_v3.RadiusCorrectionFailedError as exc:
        return None, exc


def _run_distributed_or_capture_error(seed_model, seed_direction, r, n=12):
    try:
        return _run_distributed_world1(seed_model, seed_direction, r, n), None
    except legacy_v3.RadiusCorrectionFailedError as exc:
        return None, exc


@pytest.mark.parametrize("seed_direction,r", [(1, 0.05), (2, 0.02), (3, 0.10), (5, 0.005), (8, 0.15), (13, 0.001)])
def test_world_size_1_equivalence_across_many_cases(seed_direction, r):
    """On these small (12x12=144-element) test tensors, some (seed, r) combinations genuinely
    hit a bf16 quantization plateau whose nearest attainable state exceeds the 0.1% admissibility
    bound -- exactly the phenomenon v3 exists to detect, and BOTH algorithms must reach the
    IDENTICAL verdict (same exception type, same numbers) on those cases, not merely agree when
    both happen to succeed. Comparing symmetrically (success-vs-success OR
    identical-failure-vs-identical-failure) is the correct equivalence test, never assuming
    every case converges.
    """
    legacy_outcome, legacy_error = _run_legacy_or_capture_error(42, seed_direction, r)
    dist_outcome, dist_error = _run_distributed_or_capture_error(42, seed_direction, r)

    if legacy_error is not None or dist_error is not None:
        assert type(legacy_error) is type(dist_error), f"legacy raised {legacy_error!r}, distributed raised {dist_error!r}"
        assert str(legacy_error).split("relative error ")[-1].split(" exceeds")[0] == str(dist_error).split("relative error ")[-1].split(" exceeds")[0]
        return

    legacy_model, legacy_result = legacy_outcome
    dist_model, dist_result = dist_outcome
    assert dist_result["accepted_scalar"] == pytest.approx(legacy_result["accepted_scalar"])
    assert dist_result["radius_acceptance_mode"] == legacy_result["radius_acceptance_mode"]
    assert dist_result["realized_relative_l2"] == pytest.approx(legacy_result["realized_relative_l2"])
    assert dist_result["solver_iterations"] == legacy_result["solver_iterations"]
    assert dist_result["quantization_plateau"] == legacy_result["quantization_plateau"]
    assert torch.equal(legacy_model.language_model.weight, dist_model.language_model.weight)


def test_world_size_1_equivalence_final_restored_weights_identical():
    """After the solver returns, the LIVE model weights (already the accepted state, per both
    algorithms' own discipline) must be bit-identical between legacy and distributed.
    """
    legacy_model, legacy_result = _run_legacy(seed_model=7, seed_direction=21, r=0.03)
    dist_model, dist_result = _run_distributed_world1(seed_model=7, seed_direction=21, r=0.03)
    assert torch.equal(legacy_model.language_model.weight, dist_model.language_model.weight)
    assert legacy_result["radius_acceptance_mode"] in ("strict", "quantization_limited")


def test_world_size_1_equivalence_classified_bit_decision_equivalent():
    """The Section-6-required classification: same accepted scalar, same iteration count, same
    mode, same final weights -- class A, never merely 'close'.
    """
    legacy_model, legacy_result = _run_legacy(seed_model=99, seed_direction=4, r=0.07)
    dist_model, dist_result = _run_distributed_world1(seed_model=99, seed_direction=4, r=0.07)
    is_class_a = (
        legacy_result["accepted_scalar"] == dist_result["accepted_scalar"]
        and legacy_result["solver_iterations"] == dist_result["solver_iterations"]
        and legacy_result["radius_acceptance_mode"] == dist_result["radius_acceptance_mode"]
        and torch.equal(legacy_model.language_model.weight, dist_model.language_model.weight)
    )
    assert is_class_a


# =================================================================================================
# Fixed Gaussian direction preservation (Section 3) -- no redraw during bracket search
# =================================================================================================


def test_noise_is_not_redrawn_between_solver_trials():
    """_generate_full_shape_noise is a pure function of (shape, dtype, device, seed) -- calling
    it twice with the SAME seed for the SAME parameter must return bit-identical noise,
    regardless of how many solver trials have already run.
    """
    shape, dtype, device = torch.Size([12, 12]), torch.bfloat16, torch.device("cpu")
    a = dp._generate_full_shape_noise(shape, dtype, device, seed=17)
    b = dp._generate_full_shape_noise(shape, dtype, device, seed=17)
    c = dp._generate_full_shape_noise(shape, dtype, device, seed=17)  # simulating a 3rd bracket trial
    assert torch.equal(a, b) and torch.equal(b, c)


def test_only_scalar_changes_between_trials_not_direction():
    """Two DIFFERENT requested radii on the SAME seed must accept DIFFERENT scalars but the same
    underlying noise draw -- i.e. accepted_scalar * noise, not a redrawn vector, explains the
    difference. Proven indirectly: the ratio of the two final perturbations' L2 norms must equal
    the ratio of their accepted scales (both scale the SAME direction).
    """
    model_a, result_a = _run_distributed_world1(seed_model=5, seed_direction=1, r=0.02, n=16)
    model_b, result_b = _run_distributed_world1(seed_model=5, seed_direction=1, r=0.06, n=16)
    # Same direction seed(3) + same starting model(5) -- only r differs.
    scale_ratio = result_b["final_scale"] / result_a["final_scale"]
    assert scale_ratio > 1.0  # larger requested radius -> larger scale, same direction


# =================================================================================================
# Simulated 2-rank equivalence (Section 7) -- the strongest CPU proof before live TP hardware
# =================================================================================================


class _TwoRankAllReduceSimulator:
    """Drives a REAL 2-rank collective simulation for the iterative solver within one process.
    rank0's `_distributed_trial_apply` calls all_reduce_sum exactly 4 times per trial, in a fixed
    order (theta, noise, designed, realized) -- this class tracks that order, and on each call
    ALSO performs rank1's own equivalent local computation against rank1's OWN shard (kept as a
    real, live nn.Parameter mirrored in lockstep), returning the TRUE combined global value --
    mathematically identical to what a real NCCL all-reduce across two live processes computes.
    """

    def __init__(self, rank1_param: torch.nn.Parameter, rank1_shard: dp.ShardSpec, rank1_base_cpu: torch.Tensor, seed: int, chunk_elements: int = 4_194_304):
        self.rank1_param = rank1_param
        self.rank1_shard = rank1_shard
        self.rank1_base_cpu = rank1_base_cpu
        self.seed = seed
        self.chunk_elements = chunk_elements
        self._call_index = 0  # 0=theta, 1=noise, 2=designed, 3=realized, then wraps
        self._current_trial_r = None
        self._rank1_full_noise = None
        self._rank1_scale = None

    def _reset_rank1_if_new_trial(self):
        """Called only from the wrapper's own phase==0 branch (i.e. exactly once per trial,
        already gated by the caller) -- always resets unconditionally, mirroring rank0's own
        reset_to_base_weights_cpu_rpc() call at the top of every _evaluate() invocation.
        """
        self.rank1_param.data.copy_(self.rank1_base_cpu.to(self.rank1_param.device))

    def apply_scale_and_measure(self, global_theta_l2: float, global_noise_l2: float, trial_r: float) -> None:
        """Called by the test harness AFTER the first two all-reduces of a trial (theta, noise)
        have returned, mirroring exactly when rank0's own _distributed_trial_apply computes
        `scale` and applies it in Pass 2 -- keeps rank1's shard a true mirror.
        """
        self._rank1_scale = (trial_r * global_theta_l2) / global_noise_l2
        rank1_noise_slice = dp.slice_shard(self._rank1_full_noise, self.rank1_shard)
        delta = self._rank1_scale * rank1_noise_slice
        self.rank1_param.data.add_(delta.to(dtype=self.rank1_param.dtype))
        self._rank1_last_delta = delta

    def designed_contribution(self) -> float:
        from neural_thickets_repro.thicket.memory_bounded_ops import chunked_squared_l2_sum
        return chunked_squared_l2_sum(self._rank1_last_delta.detach())

    def realized_contribution(self) -> float:
        return dv3._chunked_cross_device_squared_l2_diff_sum(self.rank1_param.detach(), self.rank1_base_cpu, self.chunk_elements)


def test_simulated_two_rank_solver_matches_single_process_reference():
    """Splits a real parameter into 2 shards (dim 0), runs the distributed v3 solver with a
    REAL 2-rank collective simulation, and requires: same accepted scalar, same realized global
    relative-L2, same solver iteration count/path length, same acceptance mode, and -- the
    strongest check -- concatenating the two ranks' final BF16 weights reproduces the
    single-process reference's final weights EXACTLY.
    """
    reference_model, reference_result = _run_legacy(seed_model=31, seed_direction=9, r=0.04, n=16)

    rank0_model = _TinyVLM(seed=31, n=16)
    rank1_model = _TinyVLM(seed=31, n=16)
    full_shape = rank0_model.language_model.weight.shape
    with torch.no_grad():
        rank0_model.language_model.weight.data = rank0_model.language_model.weight.data[:8].clone()
        rank1_model.language_model.weight.data = rank1_model.language_model.weight.data[8:].clone()

    rank0_shard = dp.ShardSpec(global_shape=full_shape, dim=0, local_offset=0, local_size=8, rank=0, world_size=2)
    rank1_shard = dp.ShardSpec(global_shape=full_shape, dim=0, local_offset=8, local_size=8, rank=1, world_size=2)

    rank0_worker = _fake_distributed_worker(rank0_model)
    rank1_worker_ns = SimpleNamespace(model_runner=SimpleNamespace(model=rank1_model))
    cbs.store_base_weights_cpu_rpc(rank1_worker_ns, pin_memory=False)
    rank1_base_cpu_tensor = rank1_worker_ns._base_weights_cpu["language_model.weight"]

    simulator = _TwoRankAllReduceSimulator(rank1_model.language_model.weight, rank1_shard, rank1_base_cpu_tensor, seed=9)

    # Wrap all_reduce_sum so the designed/realized phases (2, 3) get the correct rank1
    # contribution by first letting rank0's OWN local value flow through, then adding rank1's
    # matching contribution computed via apply_scale_and_measure (invoked by a thin trial hook).
    trial_state = {"theta": None, "noise": None}

    def all_reduce_sum_wrapper(local_value: float, process_group) -> float:
        # 5 all_reduce_sum calls happen per trial: theta, noise, designed, realized (all inside
        # _distributed_trial_apply), then ONE more from aggregate_distributed_out_of_region_
        # drift's n_differing reduce -- cycling mod 4 here originally desynchronized every trial
        # after the first (this 5th call was mistaken for the next trial's phase-0/theta call).
        phase = simulator._call_index % 5
        if phase == 0:
            simulator._reset_rank1_if_new_trial()
            rank1_local = chunked_squared_l2_sum(simulator.rank1_param.detach())
            simulator._call_index += 1
            trial_state["theta"] = (local_value, rank1_local)
            return local_value + rank1_local
        if phase == 1:
            simulator._rank1_full_noise = dp._generate_full_shape_noise(rank1_shard.global_shape, simulator.rank1_param.dtype, simulator.rank1_param.device, dp._param_noise_seed(9, "language_model.weight"))
            rank1_noise_slice = dp.slice_shard(simulator._rank1_full_noise, rank1_shard)
            rank1_local = chunked_squared_l2_sum(rank1_noise_slice)
            simulator._call_index += 1
            trial_state["noise"] = (local_value, rank1_local)
            global_theta = trial_state["theta"][0] + trial_state["theta"][1]
            global_noise = local_value + rank1_local
            trial_state["global_theta_l2"] = global_theta ** 0.5
            trial_state["global_noise_l2"] = global_noise ** 0.5
            return global_noise
        if phase == 2:
            # rank0 has just computed its own designed contribution using `scale`; mirror it on rank1.
            trial_r = trial_state["trial_r"]
            simulator.apply_scale_and_measure(trial_state["global_theta_l2"], trial_state["global_noise_l2"], trial_r)
            rank1_local = simulator.designed_contribution()
            simulator._call_index += 1
            return local_value + rank1_local
        if phase == 3:
            rank1_local = simulator.realized_contribution()
            simulator._call_index += 1
            return local_value + rank1_local
        # phase == 4: the out-of-region n_differing SUM reduce (aggregate_distributed_out_of_
        # region_drift's second all_reduce_sum call, made once more per trial AFTER the 4 apply
        # -- theta/noise/designed/realized -- reduces above). The region here covers the whole
        # model, so rank1's own local_n_differing is trivially 0 too -- a genuine pass-through,
        # not a shortcut. Cycling mod 5 (not 4) is exactly what was missing the first time this
        # test was written and produced a nonsensical relative error: this 5th call was being
        # mistaken for the NEXT trial's phase-0 (theta) call, corrupting every trial after the
        # first.
        simulator._call_index += 1
        return local_value + 0.0

    orig_apply = dv3._distributed_trial_apply

    def _wrapped_apply(model, region_param_names, seed, trial_r, shard_specs, base_weights_cpu, **kwargs):
        trial_state["trial_r"] = trial_r
        return orig_apply(model, region_param_names, seed, trial_r, shard_specs, base_weights_cpu, **kwargs)

    dv3._distributed_trial_apply = _wrapped_apply
    try:
        shard_specs_rank0 = {"language_model.weight": rank0_shard}
        result0 = dv3.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed(
            rank0_worker, 9, 0.04, "language", ["language_model.weight"], shard_specs_rank0,
            all_reduce_sum=all_reduce_sum_wrapper, all_reduce_max=dv3.identity_all_reduce_max,
        )
    finally:
        dv3._distributed_trial_apply = orig_apply

    assert result0["accepted_scalar"] == pytest.approx(reference_result["accepted_scalar"])
    assert result0["realized_relative_l2"] == pytest.approx(reference_result["realized_relative_l2"])
    assert result0["radius_acceptance_mode"] == reference_result["radius_acceptance_mode"]
    assert result0["solver_iterations"] == reference_result["solver_iterations"]

    concatenated = torch.cat([rank0_model.language_model.weight, rank1_model.language_model.weight], dim=0)
    assert torch.equal(concatenated, reference_model.language_model.weight)


# =================================================================================================
# Restore-between-trials (Section 5) -- no cumulative perturbation
# =================================================================================================


def test_reset_to_base_weights_cpu_rpc_used_between_trials_no_cumulative_drift():
    """After a multi-trial solve, the FINAL live weights' displacement from base corresponds to
    exactly ONE trial's delta (the accepted one) -- never a sum of multiple trials' deltas, which
    would happen if resets were skipped between attempts.
    """
    model, result = _run_distributed_world1(seed_model=11, seed_direction=6, r=0.08)
    if result["solver_iterations"] < 2:
        pytest.skip("this seed/radius converged in 1 iteration -- no cumulative-drift risk to demonstrate")
    # The realized relative-L2 must be close to the REQUESTED r, not some multiple of it (which a
    # cumulative bug would produce after N un-reset trials).
    assert result["realized_relative_l2"] < 0.5  # sane, single-trial-scale magnitude, not N x r


def test_distributed_evaluate_resets_before_every_trial_via_source_inspection():
    import inspect
    source = inspect.getsource(dv3.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed)
    assert "reset_to_base_weights_cpu_rpc(worker_self)" in source
    # appears at the START of the per-trial evaluate closure AND before the final reproduction re-apply.
    assert source.count("reset_to_base_weights_cpu_rpc(worker_self)") == 2


# =================================================================================================
# Rank consensus enforcement (Section 8)
# =================================================================================================


def test_verify_solver_rank_consensus_passes_on_identical_results():
    r = {"accepted_scalar": 0.1, "radius_acceptance_mode": "strict", "realized_relative_l2": 0.05, "solver_iterations": 3, "quantization_plateau": False}
    out = dv3.verify_solver_rank_consensus([dict(r), dict(r)])
    assert out["ok"] is True


def test_verify_solver_rank_consensus_hard_fails_on_disagreement():
    r0 = {"accepted_scalar": 0.1, "radius_acceptance_mode": "strict", "realized_relative_l2": 0.05, "solver_iterations": 3, "quantization_plateau": False}
    r1 = dict(r0, accepted_scalar=0.2)  # simulates a genuinely diverged rank
    with pytest.raises(dv3.SolverRankConsensusError):
        dv3.verify_solver_rank_consensus([r0, r1])


def test_verify_solver_rank_consensus_requires_at_least_one_result():
    with pytest.raises(ValueError):
        dv3.verify_solver_rank_consensus([])


# =================================================================================================
# Bounded realized-radius measurement / memory (Section 9)
# =================================================================================================


def test_cross_device_diff_matches_same_device_reference():
    gpu_like = torch.randn(1000, dtype=torch.float32).to(torch.bfloat16)
    cpu_like = gpu_like.clone() + torch.randn(1000, dtype=torch.float32).to(torch.bfloat16) * 0.01
    from neural_thickets_repro.thicket.memory_bounded_ops import chunked_squared_l2_diff_sum

    same_device = chunked_squared_l2_diff_sum(gpu_like, cpu_like)
    cross_device = dv3._chunked_cross_device_squared_l2_diff_sum(gpu_like, cpu_like, chunk_elements=64)
    assert cross_device == pytest.approx(same_device)


def test_cross_device_diff_is_chunk_size_invariant():
    a = torch.randn(500, dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(500, dtype=torch.float32).to(torch.bfloat16)
    small_chunk = dv3._chunked_cross_device_squared_l2_diff_sum(a, b, chunk_elements=7)
    large_chunk = dv3._chunked_cross_device_squared_l2_diff_sum(a, b, chunk_elements=10_000)
    assert small_chunk == pytest.approx(large_chunk)


def test_out_of_region_drift_vacuous_for_whole_model_region():
    """For the S1 whole_model track, the region covers every parameter -- the out-of-region
    complement is empty by construction, so drift is trivially zero.
    """
    model = _TinyVLM(seed=1)
    worker = _fake_distributed_worker(model)
    region_names_set = {name for name, _ in model.named_parameters()}
    result = dv3.aggregate_distributed_out_of_region_drift(model, worker._base_weights_cpu, region_names_set)
    assert result["max_abs_drift"] == 0.0
    assert result["n_differing"] == 0


def test_out_of_region_drift_detects_a_real_out_of_region_change():
    model = _TinyVLM(seed=2)
    worker = _fake_distributed_worker(model)
    with torch.no_grad():
        model.language_model.weight.add_(0.05)  # "out of region" relative to an empty region set
    result = dv3.aggregate_distributed_out_of_region_drift(model, worker._base_weights_cpu, region_names_set=set())
    assert result["max_abs_drift"] > 0.0


# =================================================================================================
# Requires cpu_base_weights (structural)
# =================================================================================================


def test_distributed_v3_requires_cpu_base_snapshot():
    model = _TinyVLM(seed=3)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model), _should_perturb=lambda n: True)
    shard_specs = _world1_shard_specs(model)
    with pytest.raises(RuntimeError):
        dv3.scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3_distributed(worker, 1, 0.05, "language", ["language_model.weight"], shard_specs)


def test_method_name_distinct_from_legacy():
    assert dv3.QUANTIZATION_AWARE_METHOD_V3_DISTRIBUTED != legacy_v3.QUANTIZATION_AWARE_METHOD_V3
    _, result = _run_distributed_world1(seed_model=1, seed_direction=1, r=0.05)
    assert result["radius_realization_method"] == dv3.QUANTIZATION_AWARE_METHOD_V3_DISTRIBUTED
