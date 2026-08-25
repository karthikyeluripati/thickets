"""Tests for analysis/stage6_cache_safe_reproduction_analysis.py (this repair pass): the audit
comparing the historical Stage-6 full run against the cache-safe reproduction. Builds small
synthetic ExperimentResultRecord grids (never touches real results/ data) so every statistic
can be hand-verified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import stage6_cache_safe_reproduction_analysis as sca  # noqa: E402

from neural_thickets_repro.run_global_visual_thicket_pilot import (  # noqa: E402
    DEFAULT_PERTURBATIONS_PER_SIGMA,
    DEFAULT_SUBSET_SIZE,
    PILOT_CAPABILITIES,
    UPSTREAM_SIGMA_GRID,
    CheckpointManifest,
)
from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402

SMALL_SIGMAS = (0.0001, 0.0005, 0.001, 0.002)
BASE_SCORES = {"visual_grounding": 0.8, "ocr_text_recognition_grounded": 0.9, "spatial_reasoning": 0.7}


def _rec(
    *, capability: str, sigma: float, idx: int, delta: float, seed: Optional[int] = None,
    mask_hash: Optional[str] = None, model_revision: str = "rev1", per_example_hash: Optional[str] = None,
) -> ExperimentResultRecord:
    pid = f"p{sigma}_{idx}"
    base = BASE_SCORES[capability]
    return ExperimentResultRecord(
        experiment_id="exp", perturbation_id=pid, model_family="qwen2_5_vl", model_scale="3B",
        model_revision=model_revision, perturbation_mode="global_gaussian_upstream", anatomy_region=None,
        radius=None, sigma=sigma, seed=seed if seed is not None else idx, parameter_mask_hash=mask_hash or f"mask_{pid}",
        capability=capability, dataset_role="map", subset_hash=f"sub_{capability}", base_score=base,
        perturbed_score=round(base + delta, 10), delta=delta, parser_failure_rate=0.0,
        per_example_result_path=None, per_example_result_hash=per_example_hash,
        runtime_metadata={"restoration_mode": "fixed_base", "perturbation_semantics": "global_gaussian_upstream"},
    )


def _build_grid(
    delta_fn, *, sigmas: Tuple[float, ...] = SMALL_SIGMAS, n: int = 6, model_revision: str = "rev1",
) -> List[ExperimentResultRecord]:
    records = []
    for sigma in sigmas:
        for idx in range(n):
            for cap in PILOT_CAPABILITIES:
                records.append(_rec(
                    capability=cap, sigma=sigma, idx=idx, delta=delta_fn(sigma, cap, idx), model_revision=model_revision,
                ))
    return records


def _checkpoint(**overrides) -> CheckpointManifest:
    cfg = dict(
        experiment_id="visual_thicket_global_3b_pilot", run_signature="synthetic", restoration_mode="fixed_base",
        perturbation_semantics="global_gaussian_upstream", model_revision="rev1",
        subset_hashes={cap: f"sub_{cap}" for cap in PILOT_CAPABILITIES}, subset_size=DEFAULT_SUBSET_SIZE,
        perturbations_per_sigma=DEFAULT_PERTURBATIONS_PER_SIGMA,
        expected_unique_perturbations=DEFAULT_PERTURBATIONS_PER_SIGMA * len(UPSTREAM_SIGMA_GRID),
        expected_result_rows=DEFAULT_PERTURBATIONS_PER_SIGMA * len(UPSTREAM_SIGMA_GRID) * len(PILOT_CAPABILITIES),
    )
    cfg.update(overrides)
    return CheckpointManifest(**cfg)


def _useful_delta(sigma, cap, idx):
    """spatial_reasoning: useful at every SMALL_SIGMAS cell; others: transition (never useful)."""
    if cap == "spatial_reasoning":
        return [0.05, 0.05, 0.03, 0.03, -0.01, -0.01][idx % 6]
    return [-0.01, -0.01, 0.0, 0.0, -0.02, -0.02][idx % 6]


# =================================================================================================
# Part 1: validate_alignment
# =================================================================================================


def test_validate_alignment_all_checks_pass_on_matching_synthetic_full_scale_grid():
    hist = _build_grid(_useful_delta, sigmas=UPSTREAM_SIGMA_GRID, n=DEFAULT_PERTURBATIONS_PER_SIGMA)
    safe = _build_grid(_useful_delta, sigmas=UPSTREAM_SIGMA_GRID, n=DEFAULT_PERTURBATIONS_PER_SIGMA)
    v = sca.validate_alignment(hist, _checkpoint(), safe, _checkpoint(run_signature="cache_safe"))
    assert v["all_hard_verification_checks_pass"] is True
    assert v["candidate_alignment_exact"] is True
    assert v["n_seed_mismatches"] == 0
    assert v["n_parameter_mask_hash_mismatches"] == 0


def test_validate_alignment_detects_model_revision_mismatch():
    hist = _build_grid(_useful_delta, model_revision="rev1")
    safe = _build_grid(_useful_delta, model_revision="rev2")
    v = sca.validate_alignment(hist, _checkpoint(model_revision="rev1"), safe, _checkpoint(model_revision="rev2"))
    assert v["same_model_revision"] is False
    assert v["all_hard_verification_checks_pass"] is False


def test_validate_alignment_detects_seed_mismatch_for_common_candidate():
    hist = _build_grid(_useful_delta)
    safe = _build_grid(_useful_delta)
    # Mutate exactly one candidate's seed on the safe side.
    safe[0] = _rec(capability=safe[0].capability, sigma=safe[0].sigma, idx=0, delta=safe[0].delta, seed=999999)
    v = sca.validate_alignment(hist, _checkpoint(), safe, _checkpoint())
    assert v["same_perturbation_seeds_for_common_candidates"] is False
    assert v["n_seed_mismatches"] == 1
    assert v["candidate_alignment_exact"] is False


def test_validate_alignment_flags_a_non_frozen_smaller_grid_as_not_matching():
    """A deliberately smaller synthetic grid (n=6, not the frozen 64) must fail the frozen-
    design checks while still correctly reporting per-candidate alignment for the candidates
    that DO exist in common -- these are two independent things.
    """
    hist = _build_grid(_useful_delta, n=6)
    safe = _build_grid(_useful_delta, n=6)
    v = sca.validate_alignment(hist, _checkpoint(), safe, _checkpoint())
    assert v["64_perturbations_per_sigma"] is True  # checkpoint field itself still says 64
    assert v["384_unique_perturbations"] is False  # but actual records only have 4*6=24 unique ids
    assert v["1152_rows"] is False
    assert v["all_hard_verification_checks_pass"] is False
    assert v["candidate_alignment_exact"] is True  # per-candidate alignment is still exact


# =================================================================================================
# Part 2: compute_clean_stage6_summary
# =================================================================================================


def test_compute_clean_stage6_summary_matches_manual_computation():
    safe = _build_grid(_useful_delta, n=6)
    checkpoint = _checkpoint()
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    summary = sca.compute_clean_stage6_summary(safe, checkpoint, baseline)

    row = summary["capability_sigma_table"]["spatial_reasoning"]["0.0001"]
    deltas = [0.05, 0.05, 0.03, 0.03, -0.01, -0.01]
    assert row["n"] == 6
    assert row["mean_delta"] == pytest.approx(sum(deltas) / 6)
    assert row["regime"] == "useful"
    assert row["base_score"] == BASE_SCORES["spatial_reasoning"]
    # density_ge_0.0 is a NEW metric this pass adds beyond compute_radius_table's own set.
    assert row["density_ge_0.0"] == pytest.approx(4 / 6)


def test_compute_clean_stage6_summary_records_the_cache_safe_execution_identity():
    safe = _build_grid(_useful_delta, n=6)
    checkpoint = _checkpoint(multimodal_cache_policy="full_encoder_reset_vllm011_verified_v2", enable_prefix_caching=False)
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    summary = sca.compute_clean_stage6_summary(safe, checkpoint, baseline)
    assert summary["multimodal_cache_policy"] == "full_encoder_reset_vllm011_verified_v2"
    assert summary["enable_prefix_caching"] is False


# =================================================================================================
# Part 4: old vs cache-safe comparison
# =================================================================================================


def test_comparison_reports_perfect_agreement_when_runs_are_identical():
    hist = _build_grid(_useful_delta, n=6)
    safe = _build_grid(_useful_delta, n=6)
    v = sca.validate_alignment(hist, _checkpoint(), safe, _checkpoint())
    cmp = sca.compute_old_vs_cache_safe_comparison(hist, safe, v)
    assert cmp["candidate_for_candidate_comparison_performed"] is True
    cell = cmp["per_capability_sigma"]["spatial_reasoning|0.0001"]
    assert cell["pearson_delta"] == pytest.approx(1.0)
    assert cell["spearman_delta"] == pytest.approx(1.0)
    assert cell["mae_delta"] == pytest.approx(0.0)
    assert cell["fraction_exact_candidate_score_match"] == pytest.approx(1.0)
    assert cell["fraction_improvement_sign_changed"] == pytest.approx(0.0)
    assert cell["top10_jaccard"] == pytest.approx(1.0)
    assert cmp["overall"]["exact_score_agreement_fraction"] == pytest.approx(1.0)
    assert cmp["overall"]["changed_row_fraction"] == pytest.approx(0.0)


def test_comparison_detects_a_uniformly_shifted_run():
    hist = _build_grid(_useful_delta, n=6)
    safe = _build_grid(lambda s, c, i: _useful_delta(s, c, i) - 0.1, n=6)
    v = sca.validate_alignment(hist, _checkpoint(), safe, _checkpoint())
    cmp = sca.compute_old_vs_cache_safe_comparison(hist, safe, v)
    cell = cmp["per_capability_sigma"]["spatial_reasoning|0.0001"]
    assert cell["mean_signed_delta_shift_clean_minus_historical"] == pytest.approx(-0.1)
    assert cell["mae_delta"] == pytest.approx(0.1)
    assert cell["fraction_exact_candidate_score_match"] == pytest.approx(0.0)
    # historical deltas [.05,.05,.03,.03,-.01,-.01] shifted by -0.1 => [-.05,-.05,-.07,-.07,-.11,-.11]
    # every previously-positive delta (idx 0-3) flips sign; previously-negative ones (idx 4,5) stay negative.
    assert cell["fraction_improvement_sign_changed"] == pytest.approx(4 / 6)


def test_comparison_short_circuits_when_candidate_alignment_is_not_exact():
    hist = _build_grid(_useful_delta, n=6)
    safe = _build_grid(_useful_delta, n=6)
    safe[0] = _rec(capability=safe[0].capability, sigma=safe[0].sigma, idx=0, delta=safe[0].delta, seed=999999)
    v = sca.validate_alignment(hist, _checkpoint(), safe, _checkpoint())
    cmp = sca.compute_old_vs_cache_safe_comparison(hist, safe, v)
    assert cmp["candidate_for_candidate_comparison_performed"] is False
    assert "per_capability_sigma" not in cmp


def test_comparison_per_example_result_hash_agreement_fraction():
    hist = _build_grid(_useful_delta, n=6)
    safe = _build_grid(_useful_delta, n=6)
    for i, r in enumerate(hist):
        hist[i] = _rec(capability=r.capability, sigma=r.sigma, idx=int(r.perturbation_id.split("_")[-1]), delta=r.delta, per_example_hash="H")
    for i, r in enumerate(safe):
        # half get a matching hash, half a different one
        h = "H" if i % 2 == 0 else "DIFFERENT"
        safe[i] = _rec(capability=r.capability, sigma=r.sigma, idx=int(r.perturbation_id.split("_")[-1]), delta=r.delta, per_example_hash=h)
    v = sca.validate_alignment(hist, _checkpoint(), safe, _checkpoint())
    cmp = sca.compute_old_vs_cache_safe_comparison(hist, safe, v)
    assert cmp["overall"]["per_example_result_hash_available_for_n_rows"] == len(hist)
    assert cmp["overall"]["per_example_result_hash_exact_match_fraction"] == pytest.approx(0.5)


# =================================================================================================
# Part 3: spatial thicket survival
# =================================================================================================


def test_spatial_thicket_survival_true_when_useful_in_both_at_every_small_sigma():
    hist = _build_grid(_useful_delta, n=6)
    safe = _build_grid(_useful_delta, n=6)
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    clean_summary = sca.compute_clean_stage6_summary(safe, _checkpoint(), baseline)
    survival = sca.compute_spatial_thicket_survival(clean_summary, hist)
    assert survival["spatial_thicket_reproduces"] == "true"
    assert survival["n_sigmas_useful_cache_safe"] == 4


def test_spatial_thicket_survival_false_when_clean_never_useful():
    hist = _build_grid(_useful_delta, n=6)
    never_useful_delta = lambda s, c, i: -0.001  # near_base everywhere
    safe = _build_grid(never_useful_delta, n=6)
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    clean_summary = sca.compute_clean_stage6_summary(safe, _checkpoint(), baseline)
    survival = sca.compute_spatial_thicket_survival(clean_summary, hist)
    assert survival["spatial_thicket_reproduces"] == "false"
    assert survival["n_sigmas_useful_cache_safe"] == 0


def test_spatial_thicket_survival_partially_when_only_some_sigmas_remain_useful():
    hist = _build_grid(_useful_delta, n=6)

    def partial_delta(sigma, cap, idx):
        if cap == "spatial_reasoning" and sigma in (0.0005, 0.001):
            return _useful_delta(sigma, cap, idx)
        return _useful_delta(sigma, "visual_grounding", idx)  # non-useful pattern

    safe = _build_grid(partial_delta, n=6)
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    clean_summary = sca.compute_clean_stage6_summary(safe, _checkpoint(), baseline)
    survival = sca.compute_spatial_thicket_survival(clean_summary, hist)
    assert survival["spatial_thicket_reproduces"] == "partially"
    assert survival["n_sigmas_useful_cache_safe"] == 2


# =================================================================================================
# Part 6: specialization reproduction
# =================================================================================================


def test_specialization_reproduction_flags_nonpositive_transfer_for_a_clean_specialist_pattern():
    def specialist_delta(sigma, cap, idx):
        if cap == "spatial_reasoning":
            return [0.05, 0.05, 0.03, 0.03, -0.01, -0.01][idx % 6]
        return [-0.01, -0.02, -0.01, -0.02, -0.01, -0.02][idx % 6]  # always non-positive

    hist = _build_grid(specialist_delta, n=6)
    safe = _build_grid(specialist_delta, n=6)
    spec = sca.compute_specialization_reproduction(hist, safe)
    assert spec["spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_historical"] is True
    assert spec["spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_cache_safe"] is True
    assert spec["improving_all_3_capabilities_total_across_small_sigmas_historical"] == 0
    assert spec["improving_all_3_capabilities_total_across_small_sigmas_cache_safe"] == 0


# =================================================================================================
# Part 7: cache-impact classification
# =================================================================================================


def _dummy_comparison(exact_agreement, sign_flip, top10_jaccard):
    return {
        "overall": {"exact_score_agreement_fraction": exact_agreement},
        "per_capability_sigma": {
            "spatial_reasoning|0.0005": {"fraction_improvement_sign_changed": sign_flip, "top10_jaccard": top10_jaccard},
        },
    }


def test_classify_cache_impact_robust_when_high_agreement_and_conclusion_preserved():
    comparison = _dummy_comparison(0.99, 0.01, 0.95)
    survival = {"spatial_thicket_reproduces": "true"}
    specialization = {
        "spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_historical": True,
        "spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_cache_safe": True,
    }
    result = sca.classify_cache_impact(comparison, survival, specialization)
    assert result["classification"] == "A_ROBUST"


def test_classify_cache_impact_contaminated_when_low_agreement_but_conclusion_preserved():
    comparison = _dummy_comparison(0.43, 0.30, 0.30)
    survival = {"spatial_thicket_reproduces": "partially"}
    specialization = {
        "spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_historical": True,
        "spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_cache_safe": True,
    }
    result = sca.classify_cache_impact(comparison, survival, specialization)
    assert result["classification"] == "B_QUALITATIVELY_ROBUST_BUT_NUMERICALLY_CONTAMINATED"


def test_classify_cache_impact_invalidated_when_spatial_thicket_does_not_reproduce():
    comparison = _dummy_comparison(0.99, 0.01, 0.95)
    survival = {"spatial_thicket_reproduces": "false"}
    specialization = {
        "spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_historical": True,
        "spatial_positive_transfer_to_others_nonpositive_at_every_tested_sigma_cache_safe": True,
    }
    result = sca.classify_cache_impact(comparison, survival, specialization)
    assert result["classification"] == "C_INVALIDATED"


# =================================================================================================
# Part 9: Stage-8 radius recommendation
# =================================================================================================


def test_stage8_radius_recommendation_schema_and_common_radius_discipline():
    survival = {"spatial_thicket_reproduces": "partially"}
    cache_impact = {"classification": "B_QUALITATIVELY_ROBUST_BUT_NUMERICALLY_CONTAMINATED"}
    rec = sca.compute_stage8_radius_recommendation(survival, cache_impact)
    assert set(rec.keys()) == {
        "selected_common_radii", "selection_basis", "excluded_radii", "common_across_regions",
        "no_region_specific_optimization", "no_capability_specific_optimization",
        "stage6_spatial_thicket_reproduces", "stage6_cache_impact_classification",
        "proceed_to_stage8", "blocking_issue",
    }
    assert len(rec["selected_common_radii"]) in (2, 3)
    assert rec["common_across_regions"] is True
    assert rec["no_region_specific_optimization"] is True
    assert rec["no_capability_specific_optimization"] is True
    assert rec["proceed_to_stage8"] is True
    assert rec["blocking_issue"] is None


def test_stage8_radius_recommendation_excludes_the_known_destructive_radii():
    survival = {"spatial_thicket_reproduces": "partially"}
    cache_impact = {"classification": "B_QUALITATIVELY_ROBUST_BUT_NUMERICALLY_CONTAMINATED"}
    rec = sca.compute_stage8_radius_recommendation(survival, cache_impact)
    destructive = {0.1784941427189971, 0.3569882854379942}
    assert set(rec["selected_common_radii"]).isdisjoint(destructive)
    assert {float(k) for k in rec["excluded_radii"]} == destructive


def test_stage8_radius_recommendation_blocks_proceeding_when_invalidated():
    survival = {"spatial_thicket_reproduces": "false"}
    cache_impact = {"classification": "C_INVALIDATED"}
    rec = sca.compute_stage8_radius_recommendation(survival, cache_impact)
    assert rec["proceed_to_stage8"] is False
    assert rec["blocking_issue"] is not None


# =================================================================================================
# Part 8: Stage6 <-> Stage7B bridge -- no sigma/radius numerical equivalence
# =================================================================================================


def test_clean_stage6_summary_subset_size_is_50_never_conflated_with_stage7b_n20():
    """Stage 6 uses D_map N=50 (DEFAULT_SUBSET_SIZE); Stage 7B uses N=20
    (FULL_CALIBRATION_D_MAP_N). compute_clean_stage6_summary must report whatever subset_size
    the CHECKPOINT actually carries, never a hardcoded 20.
    """
    safe = _build_grid(_useful_delta, n=6)
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    checkpoint = _checkpoint(subset_size=DEFAULT_SUBSET_SIZE)
    summary = sca.compute_clean_stage6_summary(safe, checkpoint, baseline)
    assert summary["subset_size"] == 50 == DEFAULT_SUBSET_SIZE
    assert summary["subset_size"] != 20


def test_bridge_never_claims_sigma_radius_numerical_equivalence(tmp_path):
    safe = _build_grid(_useful_delta, n=6)
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    clean_summary = sca.compute_clean_stage6_summary(safe, _checkpoint(), baseline)
    bridge = sca.compute_stage6_stage7b_bridge(clean_summary, tmp_path)  # no stage7b dir present
    assert "NOT" in bridge["note"] and "equated" in bridge["note"]
    assert "no sigma" in bridge["qualitative_connection"].lower() or "numerical equivalence" in bridge["qualitative_connection"].lower()


# =================================================================================================
# Determinism
# =================================================================================================


def test_full_pipeline_is_deterministic_on_repeat_computation():
    hist = _build_grid(_useful_delta, n=6)
    safe = _build_grid(_useful_delta, n=6)
    baseline = {"capabilities": {cap: {"score": BASE_SCORES[cap]} for cap in PILOT_CAPABILITIES}}
    checkpoint = _checkpoint()

    def run_once():
        v = sca.validate_alignment(hist, checkpoint, safe, checkpoint)
        clean_summary = sca.compute_clean_stage6_summary(safe, checkpoint, baseline)
        cmp = sca.compute_old_vs_cache_safe_comparison(hist, safe, v)
        survival = sca.compute_spatial_thicket_survival(clean_summary, hist)
        spec = sca.compute_specialization_reproduction(hist, safe)
        impact = sca.classify_cache_impact(cmp, survival, spec)
        return sca._sanitize({"clean": clean_summary, "cmp": cmp, "survival": survival, "spec": spec, "impact": impact})

    import json as _json
    first = _json.dumps(run_once(), sort_keys=True)
    second = _json.dumps(run_once(), sort_keys=True)
    assert first == second
