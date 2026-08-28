"""Tests for the Stage-11 32B whole-model smoke END-TO-END WIRING milestone: dispatcher
integration, run_stage11_whole_model_scaling.py's 32B branch, vLLM shard-mapping, and the live
G3/G4/G5 gate-check functions. CPU-only, no GPU/ray/vLLM import -- matches this project's
established convention (see test_run_stage11_whole_model_scaling.py's own fake-engine tests).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Dict

import pytest
import torch
import torch.nn as nn

import neural_thickets_repro.run_stage11_visual_thicket_scaling as dispatcher
import neural_thickets_repro.run_stage11_whole_model_scaling as whole_model
import neural_thickets_repro.scaling_common as scaling_common
import neural_thickets_repro.stage11_32b_readiness as readiness
from neural_thickets_repro.thicket import cpu_base_snapshot as cbs
from neural_thickets_repro.thicket import distributed_perturbation as dp
from neural_thickets_repro.thicket import vllm_shard_mapping as vsm


# =================================================================================================
# Top-level 32B dispatch (dispatcher)
# =================================================================================================


def test_dispatcher_32b_whole_model_smoke_dry_run_succeeds():
    rc = dispatcher.main(["--scale", "32B", "--track", "whole_model", "--smoke", "--dry-run"])
    assert rc == 0


def test_dispatcher_32b_anatomy_is_blocked():
    rc = dispatcher.main(["--scale", "32B", "--track", "anatomy", "--smoke"])
    assert rc == 1


def test_dispatcher_32b_without_smoke_is_blocked():
    rc = dispatcher.main(["--scale", "32B", "--track", "whole_model"])
    assert rc == 1


def test_dispatcher_72b_still_hard_rejected():
    rc = dispatcher.main(["--scale", "72B", "--track", "whole_model"])
    assert rc == 1
    rc_anatomy = dispatcher.main(["--scale", "72B", "--track", "anatomy"])
    assert rc_anatomy == 1


def test_dispatcher_3b_7b_dry_run_unaffected(capsys):
    rc3 = dispatcher.main(["--scale", "3B", "--track", "whole_model", "--smoke", "--dry-run"])
    out3 = capsys.readouterr().out
    rc7 = dispatcher.main(["--scale", "7B", "--track", "whole_model", "--smoke", "--dry-run"])
    out7 = capsys.readouterr().out
    assert rc3 == 0 and rc7 == 0
    assert "scale=3B" in out3 and "scale=7B" in out7


# =================================================================================================
# run_stage11_whole_model_scaling.py -- 32B branch
# =================================================================================================


def test_32b_scale_is_now_a_valid_cli_choice():
    rc = whole_model.main(["--scale", "32B", "--smoke", "--dry-run"])
    assert rc == 0


def test_32b_smoke_dry_run_matches_frozen_design_totals(capsys):
    rc = whole_model.main(["--scale", "32B", "--smoke", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total_unique_perturbations=3" in out
    assert "total_perturbation_x_capability_evaluations=18" in out


def test_32b_full_run_without_smoke_refused():
    rc = whole_model.main(["--scale", "32B"])
    assert rc == 1


def test_32b_smoke_blocked_by_v3_solver_gap_writes_gate_report(tmp_path, monkeypatch):
    """Full (non-dry-run) 32B path with the environment/Hub gates faked open, to reach the new
    readiness pre-flight branch -- must write a real gate report and return 0 (a clean,
    honest 'blocked' exit), never crash, never proceed toward an engine launch.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": "a" * 40, "requested_revision": "main"})
    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path)])
    assert rc == 0
    report_files = list(tmp_path.rglob("stage11_32b_readiness_gate_report.json"))
    assert len(report_files) == 1
    import json
    report = json.loads(report_files[0].read_text())
    assert report["gate_results"]["G4"] == readiness.GATE_FAIL
    assert report["all_gates_pass"] is False


def test_32b_never_reaches_legacy_engine_config(tmp_path, monkeypatch):
    """Structural proof the 32B branch returns before build_stage7b_engine_config (the 3B/7B-only,
    TP=1-hardcoded, legacy-base-snapshot-hardcoded config builder) is ever called.
    """
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: {"resolved_revision": "a" * 40, "requested_revision": "main"})

    def _should_never_be_called():
        raise AssertionError("build_stage7b_engine_config must never be called on the 32B branch")

    monkeypatch.setattr(whole_model, "build_stage7b_engine_config", _should_never_be_called)
    rc = whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path)])
    assert rc == 0  # would have raised AssertionError above if the legacy path were reached


def test_3b_7b_source_unchanged_by_32b_branch_presence():
    """The 32B branch is an early return; confirms via source inspection that the existing
    3B/7B code (build_stage7b_engine_config, store_base_weights_via_rpc, run_whole_model_rpc)
    is still present, unparameterized by any 32B-specific conditional.
    """
    source = inspect.getsource(whole_model.main)
    assert 'if plan.scale_label == "32B":' in source
    # the existing 3B/7B call chain still appears exactly once, after the 32B branch's own `return`
    assert source.count("build_stage7b_engine_config()") == 1


# =================================================================================================
# 32B engine configuration (Section 3)
# =================================================================================================


def test_32b_engine_config_matches_task_spec():
    cfg = readiness.build_32b_engine_config()
    assert cfg["tensor_parallel_size"] == 4
    assert cfg["gpu_memory_utilization"] == pytest.approx(0.60)
    assert cfg["max_model_len"] == 4096
    assert cfg["enforce_eager"] is True
    assert cfg["enable_prefix_caching"] is False
    assert cfg["precision"] == "bfloat16"
    assert cfg["base_snapshot_mode"] == "cpu_base_weights"


def test_32b_engine_config_rejects_invalid_tp_size():
    with pytest.raises(ValueError):
        readiness.build_32b_engine_config(tensor_parallel_size=0)


def test_32b_engine_config_never_requests_quantization():
    cfg = readiness.build_32b_engine_config()
    assert "quant" not in str(cfg).lower()
    assert cfg["precision"] == "bfloat16"


# =================================================================================================
# Immutable revision persistence (reuses the existing, unmodified generic mechanism)
# =================================================================================================


def test_32b_revision_resolution_is_persisted_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(whole_model, "assert_feasible", lambda *a, **k: None)
    fake_resolution = {"resolved_revision": "b" * 40, "requested_revision": "main"}
    monkeypatch.setattr(whole_model, "resolve_immutable_model_revision", lambda *a, **k: fake_resolution)
    whole_model.main(["--scale", "32B", "--smoke", "--output-root", str(tmp_path)])
    revision_files = list(tmp_path.rglob("model_revision_resolution.json"))
    assert len(revision_files) == 1
    import json
    assert json.loads(revision_files[0].read_text()) == fake_resolution


def test_resumed_checkpoint_manifest_rejects_a_different_revision():
    """Existing, unmodified mechanism: a checkpoint persisted under one model_revision must hard
    -fail if a later invocation tries to resume under a DIFFERENT one -- this is what makes
    'once resolved, all subsequent executions must use that exact SHA' true, generically, for
    every scale including 32B, without any new code.
    """
    from neural_thickets_repro.run_stage11_whole_model_scaling import (
        IncompatibleWholeModelCheckpointError, WholeModelCheckpointManifest, ensure_whole_model_checkpoint_manifest,
    )
    import tempfile
    from pathlib import Path

    def _manifest(rev):
        return WholeModelCheckpointManifest(
            experiment_id="x", run_signature="y", scale_label="32B", track="whole_model", restoration_mode="fixed_base",
            perturbation_mode="anatomical_relative_l2", radius_realization_method="fixed_direction_bf16_quantization_aware_v3",
            multimodal_cache_policy="p", enable_prefix_caching=False, generation_batch_size=10, model_revision=rev,
            dataset_role="map", radii=(0.1,), capabilities=("a",), n_directions_per_cell=1, d_map_n=5,
            subset_hashes={"a": "h"}, whole_model_mask_hash="m", direction_seed_bank_hash="s", anatomy_audit_hash="aud",
            expected_unique_perturbations=1, expected_result_rows=1,
        )

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "checkpoint_manifest.json"
        ensure_whole_model_checkpoint_manifest(path, _manifest("a" * 40))
        with pytest.raises(IncompatibleWholeModelCheckpointError):
            ensure_whole_model_checkpoint_manifest(path, _manifest("c" * 40))


# =================================================================================================
# Real vLLM TP shard-metadata mapping (Section 5 -- "the main engineering task")
# =================================================================================================


def test_shard_spec_from_attributes_replicated_when_no_dim_and_tp_size_1():
    spec = vsm.build_shard_spec_from_attributes(torch.Size([8, 8]), output_dim=None, input_dim=None, tp_size=1, tp_rank=0)
    assert spec.is_replicated
    assert spec.world_size == 1


def test_shard_spec_from_attributes_recognized_replicated_under_tp_gt_1():
    """A norm weight (no output_dim/input_dim) living inside a tp_size=4 layer -- the documented
    'no dim attribute = replicated' convention, not an ambiguity.
    """
    spec = vsm.build_shard_spec_from_attributes(torch.Size([5120]), output_dim=None, input_dim=None, tp_size=4, tp_rank=2)
    assert spec.is_replicated
    assert spec.world_size == 4
    assert spec.rank == 2


def test_shard_spec_from_attributes_column_sharded():
    """output_dim=0, local shape already the shard -- global_shape/local_offset recovered per
    vLLM's own documented start_idx = tp_rank * shard_size convention.
    """
    spec = vsm.build_shard_spec_from_attributes(torch.Size([1280, 5120]), output_dim=0, input_dim=None, tp_size=4, tp_rank=2)
    assert spec.dim == 0
    assert spec.local_size == 1280
    assert spec.global_shape == torch.Size([5120, 5120])
    assert spec.local_offset == 2560  # tp_rank(2) * shard_size(1280)


def test_shard_spec_from_attributes_row_sharded():
    spec = vsm.build_shard_spec_from_attributes(torch.Size([5120, 1280]), output_dim=None, input_dim=1, tp_size=4, tp_rank=1)
    assert spec.dim == 1
    assert spec.local_offset == 1280  # tp_rank(1) * shard_size(1280)


def test_shard_spec_hard_fails_on_both_dims_set():
    with pytest.raises(vsm.AmbiguousShardMappingError):
        vsm.build_shard_spec_from_attributes(torch.Size([8, 8]), output_dim=0, input_dim=1, tp_size=4, tp_rank=0)


def test_shard_spec_hard_fails_on_out_of_range_dim():
    with pytest.raises(vsm.AmbiguousShardMappingError):
        vsm.build_shard_spec_from_attributes(torch.Size([8, 8]), output_dim=5, input_dim=None, tp_size=4, tp_rank=0)


class _FakeTPLinear(nn.Module):
    """Mimics vLLM's documented attribute convention: tp_size/tp_rank on the owning module,
    output_dim set directly on the weight Parameter via a plain attribute assignment (the same
    mechanism set_weight_attrs uses under the hood).
    """
    def __init__(self, local_out: int, in_features: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(local_out, in_features))
        self.weight.output_dim = 0
        self.tp_size = tp_size
        self.tp_rank = tp_rank


class _FakeReplicatedNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # no output_dim/input_dim, no tp_size -- plain replicated norm


class _FakeTPModel(nn.Module):
    def __init__(self, tp_size: int, tp_rank: int):
        super().__init__()
        self.proj = _FakeTPLinear(local_out=32 // tp_size, in_features=16, tp_size=tp_size, tp_rank=tp_rank)
        self.norm = _FakeReplicatedNorm(16)


def test_build_shard_specs_for_region_end_to_end_fake_vllm_model():
    model = _FakeTPModel(tp_size=4, tp_rank=2)
    specs = vsm.build_shard_specs_for_region(model, ["proj.weight", "norm.weight"])
    assert specs["proj.weight"].dim == 0
    assert specs["proj.weight"].global_shape == torch.Size([32, 16])
    assert specs["proj.weight"].local_offset == 16  # tp_rank(2) * shard_size(8)
    assert specs["norm.weight"].is_replicated


def test_build_shard_specs_for_region_missing_parameter_hard_fails():
    model = _FakeTPModel(tp_size=4, tp_rank=0)
    with pytest.raises(vsm.AmbiguousShardMappingError):
        vsm.build_shard_specs_for_region(model, ["does.not.exist"])


def test_ensure_uniform_tp_size_detects_mismatch():
    specs = {"a": dp.ShardSpec(global_shape=torch.Size([8]), dim=0, local_offset=0, local_size=2, rank=0, world_size=2)}
    with pytest.raises(vsm.AmbiguousShardMappingError):
        vsm.ensure_uniform_tp_size(specs, expected_tp_size=4)
    vsm.ensure_uniform_tp_size(specs, expected_tp_size=2)  # must not raise


# =================================================================================================
# Live G3/G4/G5 gate-check functions (worker-side, testable via fakes)
# =================================================================================================


class _TinyVLM(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.language_model = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.language_model.weight.copy_(torch.randn(8, 8, generator=g, dtype=torch.float32).to(torch.bfloat16))
        self.to(torch.bfloat16)


def _fake_worker(model):
    ns = SimpleNamespace()
    ns.model_runner = SimpleNamespace(model=model)
    ns._should_perturb = lambda name: True
    return ns


def test_g3_live_check_reports_bit_exact_on_a_real_module():
    model = _TinyVLM(seed=1)
    worker = _fake_worker(model)
    facts = cbs.g3_live_cpu_cuda_equivalence_check_rpc(worker, probe_param_name="language_model.weight", seed=42, delta=0.01)
    equivalence_class = cbs.classify_snapshot_equivalence(**facts)
    assert equivalence_class == cbs.EQUIVALENCE_BIT_EXACT


def test_g3_live_check_requires_existing_parameter():
    model = _TinyVLM(seed=2)
    worker = _fake_worker(model)
    with pytest.raises(RuntimeError):
        cbs.g3_live_cpu_cuda_equivalence_check_rpc(worker, probe_param_name="does.not.exist", seed=1)


def test_g4_g5_live_check_world_size_1_within_tolerance():
    model = _TinyVLM(seed=3)
    worker = _fake_worker(model)
    worker.tensor_parallel_size = 1
    worker.rank = 0
    # language_model.weight has neither output_dim/input_dim and worker has no vLLM-shaped
    # module wrapping -- resolves to replicated/world_size=1 via the fallback path.
    result = dp.g4_g5_live_relative_l2_check_rpc(worker, ["language_model.weight"], seed=5, r=0.05)
    # The ONE-SHOT apply (no bf16-bracketed correction -- that's the v3 solver's job, not yet
    # distributed-aware, see stage11_32b_readiness.V3_SOLVER_DISTRIBUTED_EXTENSION_NOTE) only
    # needs to be CLOSE here -- this test proves the global-norm-reduction plumbing is wired
    # correctly end to end, not bf16 radius exactness (a different, already-solved problem).
    assert result["realized_relative_l2"] == pytest.approx(0.05, abs=5e-3)


def test_classify_g4_g5_live_check_requires_synchronized_global_values():
    consistent = [{"theta_l2_norm": 1.0, "raw_noise_l2_norm": 2.0, "scale": 0.5, "realized_relative_l2": 0.05, "requested_r": 0.05}] * 2
    assert dp.classify_g4_g5_live_check(consistent) is True

    inconsistent = [
        {"theta_l2_norm": 1.0, "raw_noise_l2_norm": 2.0, "scale": 0.5, "realized_relative_l2": 0.05, "requested_r": 0.05},
        {"theta_l2_norm": 1.5, "raw_noise_l2_norm": 2.0, "scale": 0.5, "realized_relative_l2": 0.05, "requested_r": 0.05},  # a rank that never got the all-reduced value -- the exact "no per-rank normalization" bug
    ]
    assert dp.classify_g4_g5_live_check(inconsistent) is False


def test_classify_g4_g5_live_check_empty_list_fails_closed():
    assert dp.classify_g4_g5_live_check([]) is False


# =================================================================================================
# 32B anatomy completeness gate (design-level -- reused frozen regions, no execution)
# =================================================================================================


def test_32b_readiness_manifest_region_completeness_reuses_frozen_atlas():
    """No NEW regions were invented for 32B -- the manifest reuses the exact frozen
    vision/connector/language partition, the same one 3B/7B's own live anatomy audits already
    proved complete/disjoint.
    """
    manifest = readiness.build_32b_readiness_manifest()
    assert set(manifest.region_definitions) == {"vision", "multimodal_connector_or_merger", "language"}


# =================================================================================================
# Candidate transactional persistence -- proof the existing mechanism is untouched
# =================================================================================================


def test_evaluate_one_whole_model_candidate_rpc_source_has_no_32b_conditional():
    """The real per-candidate transactional lifecycle (perturb -> evaluate 6 capabilities ->
    restore -> verify -> only then return rows) is completely unparameterized by scale -- no
    32B-specific branch was added to it (the 32B path diverges much earlier, in main(), before
    this function is ever reached) -- this positively confirms 3B/7B's transactional guarantees
    were not touched.
    """
    source = inspect.getsource(whole_model.evaluate_one_whole_model_candidate_rpc)
    assert "32B" not in source
    assert "cpu_base_weights" not in source
    assert "tensor_parallel" not in source


# =================================================================================================
# No 3B/7B behavior regression -- direct equivalence checks
# =================================================================================================


def test_3b_and_7b_still_use_ensure_scale_runnable_unchanged():
    with pytest.raises(scaling_common.ScaleNotYetEnabledError):
        scaling_common.ensure_scale_runnable("72B")
    scaling_common.ensure_scale_runnable("3B")  # must not raise
    scaling_common.ensure_scale_runnable("7B")  # must not raise


def test_3b_7b_dry_run_plans_never_take_the_32b_branch(capsys):
    rc = whole_model.main(["--scale", "7B", "--smoke", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "32B readiness gate report" not in out
    assert "STOP AND REPORT" not in out
