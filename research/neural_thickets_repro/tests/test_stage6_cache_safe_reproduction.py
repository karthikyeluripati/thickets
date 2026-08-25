"""Tests for the cache-safe Stage-6 reproduction (this repair pass -- prefix-KV-cache audit).

Root cause under investigation: historical Stage 6's completed full run launched with
enable_prefix_caching left at vLLM's own default (confirmed True on the pinned build via
Stage 7B's own live log, same shared launch_stage6_engine() code path); its candidate loop
reuses the SAME capability_contexts (and therefore the same prompts/images) across every
perturbation candidate while language weights change between them; reset_prefix_cache() is
never called anywhere in this project's actual code. This makes stage6_cache_safety_status
"cache_suspect" (never proven safe, never proven invalid) -- these tests cover the NEW,
additive, cache-safe reproduction infrastructure (never overwriting or resuming the historical
run), not a claim about whether the historical result is actually wrong.
"""
from __future__ import annotations

import pytest

from neural_thickets_repro.run_global_visual_thicket_pilot import (
    DEFAULT_PERTURBATIONS_PER_SIGMA,
    DEFAULT_SUBSET_SIZE,
    PILOT_CAPABILITIES,
    STAGE6_CACHE_SAFE_ENABLE_PREFIX_CACHING,
    STAGE6_CACHE_SAFE_MULTIMODAL_CACHE_POLICY,
    STAGE6_CACHE_SAFE_RUN_LABEL,
    STAGE6_CACHE_SAFE_SMOKE_EXAMPLES_PER_CAPABILITY,
    STAGE6_CACHE_SAFE_SMOKE_PERTURBATIONS_PER_SIGMA,
    STAGE6_CACHE_SAFE_SMOKE_SIGMA_GRID,
    UPSTREAM_SIGMA_GRID,
    CheckpointManifest,
    PilotConfigError,
    PilotPlan,
    build_cache_safe_pilot_plan,
    build_cache_safe_smoke_pilot_plan,
    build_pilot_plan,
    build_stage6_cache_safe_engine_config,
    build_stage6_checkpoint_manifest,
    build_stage6_engine_config,
    compute_cache_safe_run_signature,
    compute_run_signature,
)


def _raw_config(**overrides):
    cfg = {
        "model": {"name": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "rev1", "family": "qwen2_5_vl", "scale": "3B"},
        "pilot": {
            "capabilities": list(PILOT_CAPABILITIES),
            "sigma_grid": list(UPSTREAM_SIGMA_GRID),
            "perturbations_per_sigma": DEFAULT_PERTURBATIONS_PER_SIGMA,
            "examples_per_capability": DEFAULT_SUBSET_SIZE,
            "base_seed": 42,
        },
        "outputs": {"root": "results/visual_thicket_global_3b_pilot"},
    }
    cfg.update(overrides)
    return cfg


# =================================================================================================
# Historical Stage 6 configuration remains untouched
# =================================================================================================


def test_historical_compute_run_signature_unchanged():
    assert compute_run_signature(64, 50, 64, 50) == "full"
    assert compute_run_signature(2, 5, 64, 50) == "smoke_p2_n5"


def test_historical_build_pilot_plan_unaffected_by_the_new_cache_fields():
    plan = build_pilot_plan(_raw_config())
    assert plan.run_signature == "full"
    assert plan.multimodal_cache_policy is None
    assert plan.enable_prefix_caching is None


def test_historical_engine_config_has_no_prefix_caching_key():
    config = build_stage6_engine_config()
    assert "enable_prefix_caching" not in config


def test_historical_checkpoint_manifest_round_trips_with_none_cache_fields():
    plan = build_pilot_plan(_raw_config())
    checkpoint = build_stage6_checkpoint_manifest(plan, {})
    assert checkpoint.multimodal_cache_policy is None
    assert checkpoint.enable_prefix_caching is None
    restored = CheckpointManifest.from_dict(checkpoint.to_dict())
    assert restored == checkpoint


def test_legacy_checkpoint_dict_missing_cache_fields_entirely_still_parses():
    """The REAL historical checkpoint_manifest.json predates these fields entirely -- from_dict
    must not KeyError on a dict that simply doesn't have them.
    """
    legacy_dict = {
        "experiment_id": "visual_thicket_global_3b_pilot", "run_signature": "full", "restoration_mode": "fixed_base",
        "perturbation_semantics": "global_gaussian_upstream", "model_revision": "rev1",
        "subset_hashes": {}, "subset_size": 50, "perturbations_per_sigma": 64,
        "expected_unique_perturbations": 384, "expected_result_rows": 1152,
    }
    checkpoint = CheckpointManifest.from_dict(legacy_dict)
    assert checkpoint.multimodal_cache_policy is None
    assert checkpoint.enable_prefix_caching is None


# =================================================================================================
# Corrected Stage-6 execution policy: prefix caching disabled
# =================================================================================================


def test_cache_safe_engine_config_disables_prefix_caching():
    config = build_stage6_cache_safe_engine_config()
    assert config["enable_prefix_caching"] is False


def test_cache_safe_engine_config_matches_historical_except_prefix_caching():
    historical = build_stage6_engine_config()
    cache_safe = build_stage6_cache_safe_engine_config()
    assert "enable_prefix_caching" not in historical
    assert cache_safe["enable_prefix_caching"] is False
    for key in historical:
        assert cache_safe[key] == historical[key]


def test_cache_safe_engine_config_reuses_the_verified_v011_multimodal_cache_policy():
    assert STAGE6_CACHE_SAFE_MULTIMODAL_CACHE_POLICY == "full_encoder_reset_vllm011_verified_v2"
    assert STAGE6_CACHE_SAFE_ENABLE_PREFIX_CACHING is False


# =================================================================================================
# Corrected Stage-6 run identity is isolated -- no old checkpoint may resume into it
# =================================================================================================


def test_cache_safe_run_signature_is_the_suggested_label_for_the_full_frozen_config():
    sig = compute_cache_safe_run_signature(64, 50, UPSTREAM_SIGMA_GRID)
    assert sig == STAGE6_CACHE_SAFE_RUN_LABEL == "stage6_global_gaussian_upstream_cache_safe_v2"


def test_cache_safe_run_signature_disjoint_from_historical_full_and_smoke():
    cache_safe_sig = compute_cache_safe_run_signature(64, 50, UPSTREAM_SIGMA_GRID)
    historical_full_sig = compute_run_signature(64, 50, 64, 50)
    historical_smoke_sig = compute_run_signature(2, 5, 64, 50)
    assert cache_safe_sig not in (historical_full_sig, historical_smoke_sig)


def test_cache_safe_pilot_plan_output_dir_disjoint_from_historical(tmp_path):
    raw_config = _raw_config(outputs={"root": str(tmp_path)})
    historical_plan = build_pilot_plan(raw_config)
    cache_safe_plan = build_cache_safe_pilot_plan(raw_config)
    assert cache_safe_plan.output_dir != historical_plan.output_dir
    assert cache_safe_plan.multimodal_cache_policy == STAGE6_CACHE_SAFE_MULTIMODAL_CACHE_POLICY
    assert cache_safe_plan.enable_prefix_caching is False


def test_old_historical_checkpoint_cannot_resume_into_the_cache_safe_plan(tmp_path):
    raw_config = _raw_config(outputs={"root": str(tmp_path)})
    historical_plan = build_pilot_plan(raw_config)
    historical_plan.output_dir.mkdir(parents=True)
    (historical_plan.output_dir / "results.jsonl").write_text('{"fake": "historical cache_suspect provenance row"}\n')

    cache_safe_plan = build_cache_safe_pilot_plan(raw_config)

    assert cache_safe_plan.output_dir != historical_plan.output_dir
    assert not (cache_safe_plan.output_dir / "results.jsonl").exists()
    assert (historical_plan.output_dir / "results.jsonl").exists()  # historical provenance untouched


# =================================================================================================
# Frozen scientific config preserved EXACTLY for the full cache-safe reproduction
# =================================================================================================


def test_cache_safe_full_plan_preserves_the_frozen_sigma_grid(tmp_path):
    plan = build_cache_safe_pilot_plan(_raw_config(outputs={"root": str(tmp_path)}))
    assert set(plan.sigma_grid) == set(UPSTREAM_SIGMA_GRID)
    assert len(plan.sigma_grid) == 6


def test_cache_safe_full_plan_preserves_64_candidates_per_sigma_and_3_capabilities(tmp_path):
    plan = build_cache_safe_pilot_plan(_raw_config(outputs={"root": str(tmp_path)}))
    assert plan.perturbations_per_sigma == 64
    assert set(plan.capabilities) == set(PILOT_CAPABILITIES)
    assert len(plan.capabilities) == 3


def test_cache_safe_full_plan_preserves_d_map_n_50(tmp_path):
    plan = build_cache_safe_pilot_plan(_raw_config(outputs={"root": str(tmp_path)}))
    assert plan.examples_per_capability == 50


def test_cache_safe_full_plan_counts_remain_384_perturbations_1152_rows(tmp_path):
    plan = build_cache_safe_pilot_plan(_raw_config(outputs={"root": str(tmp_path)}))
    assert plan.total_unique_perturbations == 6 * 64 == 384
    assert plan.total_perturbation_capability_evaluations == 384 * 3 == 1152


def test_cache_safe_pilot_plan_rejects_a_wrong_capability_set_same_as_historical(tmp_path):
    """build_cache_safe_pilot_plan reuses build_pilot_plan's OWN validation unchanged -- a
    config with an invented capability must still be rejected identically.
    """
    raw_config = _raw_config(outputs={"root": str(tmp_path)})
    raw_config["pilot"]["capabilities"] = ["visual_grounding", "ocr_text_recognition_grounded", "made_up_capability"]
    with pytest.raises(PilotConfigError):
        build_cache_safe_pilot_plan(raw_config)


# =================================================================================================
# Cache-safety smoke: exact counts
# =================================================================================================


def test_cache_safe_smoke_config():
    assert STAGE6_CACHE_SAFE_SMOKE_SIGMA_GRID == (0.0005, 0.001)
    assert STAGE6_CACHE_SAFE_SMOKE_PERTURBATIONS_PER_SIGMA == 2
    assert STAGE6_CACHE_SAFE_SMOKE_EXAMPLES_PER_CAPABILITY == 5


def test_cache_safe_smoke_plan_counts_are_4_perturbations_12_rows(tmp_path):
    plan = build_cache_safe_smoke_pilot_plan(_raw_config(outputs={"root": str(tmp_path)}))
    assert plan.total_unique_perturbations == 2 * 2 == 4
    assert plan.total_perturbation_capability_evaluations == 4 * 3 == 12
    perturbed_model_example_evaluations = plan.total_perturbation_capability_evaluations * plan.examples_per_capability
    assert perturbed_model_example_evaluations == 12 * 5 == 60


def test_cache_safe_smoke_plan_uses_all_3_frozen_capabilities(tmp_path):
    plan = build_cache_safe_smoke_pilot_plan(_raw_config(outputs={"root": str(tmp_path)}))
    assert set(plan.capabilities) == set(PILOT_CAPABILITIES)


def test_cache_safe_smoke_plan_has_cache_safe_identity(tmp_path):
    plan = build_cache_safe_smoke_pilot_plan(_raw_config(outputs={"root": str(tmp_path)}))
    assert plan.multimodal_cache_policy == STAGE6_CACHE_SAFE_MULTIMODAL_CACHE_POLICY
    assert plan.enable_prefix_caching is False
    assert plan.run_signature.startswith(STAGE6_CACHE_SAFE_RUN_LABEL)
    assert plan.run_signature != STAGE6_CACHE_SAFE_RUN_LABEL  # smoke must be its own, disjoint signature


def test_cache_safe_smoke_plan_output_disjoint_from_full_cache_safe_and_historical(tmp_path):
    raw_config = _raw_config(outputs={"root": str(tmp_path)})
    historical = build_pilot_plan(raw_config)
    cache_safe_full = build_cache_safe_pilot_plan(raw_config)
    cache_safe_smoke = build_cache_safe_smoke_pilot_plan(raw_config)
    output_dirs = {historical.output_dir, cache_safe_full.output_dir, cache_safe_smoke.output_dir}
    assert len(output_dirs) == 3  # all three structurally distinct
