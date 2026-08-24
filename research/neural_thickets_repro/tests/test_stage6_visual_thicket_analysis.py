"""Tests for analysis/stage6_visual_thicket_analysis.py -- CPU-only, synthetic
ExperimentResultRecord data (never the real results.jsonl, which lives outside the repo and
is not committed). `analysis/` is not on pytest's own pythonpath (pyproject.toml only adds
`src`), so this test file adds it manually, mirroring the script's own sys.path handling.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import stage6_visual_thicket_analysis as stage6  # noqa: E402

from neural_thickets_repro.thicket.schema import ExperimentResultRecord  # noqa: E402


def _record(perturbation_id, capability, sigma, delta, seed=1, base_score=0.5):
    return ExperimentResultRecord(
        experiment_id="visual_thicket_global_3b_pilot", perturbation_id=perturbation_id, model_family="qwen2_5_vl",
        model_scale="3B", model_revision="rev1", perturbation_mode="global_gaussian_upstream", anatomy_region=None,
        radius=None, sigma=sigma, seed=seed, parameter_mask_hash="hash1", capability=capability, dataset_role="map",
        subset_hash=f"sh_{capability}", base_score=base_score, perturbed_score=base_score + delta, delta=delta,
        parser_failure_rate=0.0, per_example_result_path=None, per_example_result_hash="rh", runtime_metadata={},
    )


CAPS = ("cap_a", "cap_b", "cap_c")


def _population(sigma, deltas_by_cap, seed_start=1, pid_prefix=None):
    """deltas_by_cap: {capability: [delta, delta, ...]} -- same length per capability, one
    perturbation per index, shared perturbation_id across capabilities at that index.
    `pid_prefix` (defaults to a sigma-derived tag) keeps perturbation_ids unique when several
    sigma populations are concatenated -- real Stage-6 IDs are unique across the WHOLE run,
    never just within one sigma bucket.
    """
    prefix = pid_prefix if pid_prefix is not None else f"s{sigma}_"
    n = len(next(iter(deltas_by_cap.values())))
    records = []
    for i in range(n):
        pid = f"{prefix}p{i}"
        for cap, deltas in deltas_by_cap.items():
            records.append(_record(pid, cap, sigma, deltas[i], seed=seed_start + i))
    return records


# --- _sanitize -----------------------------------------------------------------------------


def test_sanitize_replaces_nan_and_inf_with_none():
    obj = {"a": float("nan"), "b": [1.0, float("inf"), -float("inf")], "c": {"d": float("nan")}}
    result = stage6._sanitize(obj)
    assert result == {"a": None, "b": [1.0, None, None], "c": {"d": None}}


def test_sanitize_preserves_ordinary_values():
    obj = {"a": 1.5, "b": [1, 2, "x"], "c": None}
    assert stage6._sanitize(obj) == obj


# --- compute_sign_agreement_matrix / compute_improving_count_histogram ---------------------


def test_compute_sign_agreement_matrix():
    matrix = np.array([[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
    agreement = stage6.compute_sign_agreement_matrix(matrix)
    assert np.allclose(np.diag(agreement), 1.0)
    # cols 0,1 agree in sign for rows 0,1,2 (1,1)(-1,-1)(1,-1)->disagree; row3 (0,0)->agree(equal signs)
    # signs col0: [1,-1,1,0], col1: [1,-1,-1,0] -> equal at idx0,1,3 -> 3/4
    assert agreement[0, 1] == pytest.approx(3 / 4)


def test_compute_improving_count_histogram():
    matrix = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, -1.0]])
    hist = stage6.compute_improving_count_histogram(matrix)
    assert hist == {"0": 1, "1": 1, "2": 1, "3": 1}


# --- compute_threshold_transfer -------------------------------------------------------------


def test_compute_threshold_transfer_positive_source():
    matrix = np.array([[1.0, 2.0], [-1.0, 3.0], [0.5, -0.5]])
    transfer, counts = stage6.compute_threshold_transfer(matrix, threshold=0.0, strict=True)
    # source=0: rows where col0>0 -> rows 0,2 -> mean col0=(1+0.5)/2=0.75, mean col1=(2-0.5)/2=0.75
    assert counts[0] == 2
    assert transfer[0][0] == pytest.approx(0.75)
    assert transfer[0][1] == pytest.approx(0.75)


def test_compute_threshold_transfer_empty_selection_is_none():
    matrix = np.array([[-1.0, 2.0], [-2.0, 3.0]])
    transfer, counts = stage6.compute_threshold_transfer(matrix, threshold=0.0, strict=True)
    assert counts[0] == 0
    assert transfer[0] == [None, None]


def test_compute_threshold_transfer_strong_source_ge():
    matrix = np.array([[0.02, 1.0], [0.019, 2.0], [0.03, 3.0]])
    transfer, counts = stage6.compute_threshold_transfer(matrix, threshold=0.02, strict=False)
    assert counts[0] == 2  # rows 0 and 2 (>=0.02)
    assert transfer[0][1] == pytest.approx((1.0 + 3.0) / 2)


# --- classify_regime --------------------------------------------------------------------------


def test_classify_regime_destructive():
    assert stage6.classify_regime(mean_delta=-0.5, p_gt0=0.0, p_lt0=1.0, density_at_02=0.0) == "destructive"


def test_classify_regime_near_base():
    assert stage6.classify_regime(mean_delta=0.0, p_gt0=0.02, p_lt0=0.02, density_at_02=0.0) == "near_base"


def test_classify_regime_useful():
    assert stage6.classify_regime(mean_delta=0.02, p_gt0=0.6, p_lt0=0.2, density_at_02=0.5) == "useful"


def test_classify_regime_transition_otherwise():
    assert stage6.classify_regime(mean_delta=-0.01, p_gt0=0.2, p_lt0=0.3, density_at_02=0.1) == "transition"


# --- compute_delta_numeric_audit: the OCR diagnosis logic ------------------------------------


def test_delta_numeric_audit_flags_near_exact_multiples_as_floating_point_noise():
    records = [_record(f"p{i}", "binary_cap", 0.001, d) for i, d in enumerate([0.02000000000000018, -0.02, 0.0])]
    audit = stage6.compute_delta_numeric_audit(records)
    cell = audit["binary_cap"]["0.001"]
    assert cell["max_abs_distance_to_nearest_0.02_multiple"] < 1e-9


def test_delta_numeric_audit_flags_genuine_fine_grained_deltas():
    records = [_record(f"p{i}", "soft_cap", 0.001, d) for i, d in enumerate([0.004, 0.006, -0.01])]
    audit = stage6.compute_delta_numeric_audit(records)
    cell = audit["soft_cap"]["0.001"]
    assert cell["max_abs_distance_to_nearest_0.02_multiple"] > 0.005  # nowhere near a 0.02 multiple
    assert cell["min_positive_delta"] == pytest.approx(0.004)
    assert cell["max_positive_delta"] == pytest.approx(0.006)
    assert cell["n_delta_gt_0"] == 2
    assert cell["n_delta_ge_0.02"] == 0


def test_delta_numeric_audit_handles_no_positive_deltas():
    records = [_record(f"p{i}", "cap", 0.01, d) for i, d in enumerate([-0.1, -0.2, 0.0])]
    audit = stage6.compute_delta_numeric_audit(records)
    cell = audit["cap"]["0.01"]
    assert cell["min_positive_delta"] is None
    assert cell["max_positive_delta"] is None
    assert cell["max_abs_distance_to_nearest_0.02_multiple"] is None


# --- compute_baseline_headroom ---------------------------------------------------------------


def test_compute_baseline_headroom():
    baseline = {"capabilities": {"cap_a": {"score": 0.8, "subset_hash": "h"}, "cap_b": {"score": 0.5, "subset_hash": "h2"}}}
    headroom = stage6.compute_baseline_headroom(baseline)
    assert headroom["cap_a"]["headroom_1_minus_baseline"] == pytest.approx(0.2)
    assert headroom["cap_b"]["headroom_1_minus_baseline"] == pytest.approx(0.5)


# --- compute_radius_table: CIs + regime -------------------------------------------------------


def test_compute_radius_table_includes_wilson_and_bootstrap_cis():
    deltas = [0.02] * 40 + [-0.02] * 10 + [0.0] * 14
    records = [_record(f"p{i}", "cap", 0.001, d) for i, d in enumerate(deltas)]
    table = stage6.compute_radius_table(records)
    cell = table["cap"]["0.001"]
    assert "mean_delta_95ci_bootstrap" in cell
    assert "p_delta_gt_0_95ci_wilson" in cell
    assert "density_ge_0.02_95ci_wilson" in cell
    lower, upper = cell["mean_delta_95ci_bootstrap"]
    assert lower <= cell["mean_delta"] <= upper
    wilson_lower, wilson_upper = cell["p_delta_gt_0_95ci_wilson"]
    assert 0.0 <= wilson_lower <= cell["p_delta_gt_0"] <= wilson_upper <= 1.0
    assert cell["regime"] == "useful"


def test_compute_radius_table_is_deterministic():
    deltas = [0.02, -0.01, 0.0, 0.05, -0.03] * 10
    records = [_record(f"p{i}", "cap", 0.002, d) for i, d in enumerate(deltas)]
    table_1 = stage6.compute_radius_table(records)
    table_2 = stage6.compute_radius_table(records)
    assert table_1["cap"]["0.002"]["mean_delta_95ci_bootstrap"] == table_2["cap"]["0.002"]["mean_delta_95ci_bootstrap"]


# --- compute_diversity_by_sigma / compute_expert_overlap / compute_directional_transfer -------


def test_compute_diversity_by_sigma_groups_correctly_and_reports_required_fields():
    records = _population(0.001, {"cap_a": [0.1, -0.1, 0.2, 0.0], "cap_b": [0.1, -0.1, -0.2, 0.0], "cap_c": [-0.1, 0.1, 0.2, 0.0]})
    by_sigma = stage6.group_by_sigma(records)
    diversity = stage6.compute_diversity_by_sigma(by_sigma)
    entry = diversity["0.001"]
    assert entry["n_perturbations"] == 4
    assert set(entry) >= {"task_rank_correlation_matrix", "spectral_discordance", "expert_overlap_jaccard", "sign_agreement_matrix", "improving_count_histogram"}
    assert set(entry["expert_overlap_jaccard"]) == {"q_0.1", "q_0.2"}


def test_compute_diversity_by_sigma_handles_a_constant_column_without_raising():
    """A capability with an exactly-constant delta (e.g. a total-collapse floor, the real
    sigma=0.01 visual_grounding case in the pilot data) must not crash the computation.
    Because thicket.diversity's correlation/discordance operate on RANKS (percentile_rank_
    matrix), a constant column still gets a well-defined (if arbitrarily tie-broken by stable
    sort) rank sequence -- empirically this produces a finite number, not NaN, confirmed both
    here and against the real pilot data (sigma=0.01 -> spectral_discordance=0.4027...).
    """
    records = _population(0.01, {"cap_a": [-0.8, -0.8, -0.8, -0.8], "cap_b": [0.1, -0.1, 0.2, -0.2], "cap_c": [0.0, 0.1, -0.1, 0.05]})
    by_sigma = stage6.group_by_sigma(records)
    diversity = stage6.compute_diversity_by_sigma(by_sigma)  # must not raise
    sd = diversity["0.01"]["spectral_discordance"]
    assert sd is not None
    assert np.isfinite(sd)


def test_compute_expert_overlap_persists_actual_perturbation_ids():
    records = _population(0.001, {"cap_a": [0.1, 0.2, -0.1, 0.05, 0.3], "cap_b": [0.2, 0.1, -0.2, 0.0, 0.4]})
    by_sigma = stage6.group_by_sigma(records)
    overlap = stage6.compute_expert_overlap(by_sigma)
    entry = overlap["0.001"]
    assert "top_5" in entry and "top_10" in entry and "top_20pct" in entry
    top5_ids = entry["top_5"]["top_perturbation_ids"]["cap_a"]
    assert all("p" in pid for pid in top5_ids)
    assert "jaccard" in entry["top_5"]


def test_compute_directional_transfer_only_covers_configured_sigmas():
    records = _population(0.0005, {"cap_a": [0.1, -0.1], "cap_b": [0.1, 0.1]}) + _population(0.01, {"cap_a": [-0.5, -0.5], "cap_b": [-0.5, -0.5]})
    by_sigma = stage6.group_by_sigma(records)
    transfer = stage6.compute_directional_transfer(by_sigma)
    assert "0.0005" in transfer
    assert "0.01" not in transfer  # not one of DIRECTIONAL_TRANSFER_SIGMAS


# --- end-to-end main() ------------------------------------------------------------------------


def _write_synthetic_run(tmp_path, sigmas=(0.001, 0.01), n_per_sigma=4, capabilities=CAPS):
    records = []
    rng = np.random.default_rng(0)
    for sigma in sigmas:
        deltas_by_cap = {cap: rng.normal(scale=0.05, size=n_per_sigma).tolist() for cap in capabilities}
        records.extend(_population(sigma, deltas_by_cap))

    results_path = tmp_path / "results.jsonl"
    with results_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    checkpoint = {
        "experiment_id": "visual_thicket_global_3b_pilot", "run_signature": "full", "restoration_mode": "fixed_base",
        "perturbation_semantics": "global_gaussian_upstream", "model_revision": "rev1",
        "subset_hashes": {cap: f"sh_{cap}" for cap in capabilities}, "subset_size": 50,
        "perturbations_per_sigma": n_per_sigma, "expected_unique_perturbations": len(sigmas) * n_per_sigma,
        "expected_result_rows": len(sigmas) * n_per_sigma * len(capabilities),
    }
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(checkpoint, indent=2))

    baseline = {"model_revision": "rev1", "run_signature": "full", "capabilities": {cap: {"score": 0.7, "subset_hash": f"sh_{cap}"} for cap in capabilities}}
    (tmp_path / "baseline_scores.json").write_text(json.dumps(baseline, indent=2))
    return records


def test_main_writes_all_six_outputs(tmp_path):
    _write_synthetic_run(tmp_path)
    rc = stage6.main(["--results-dir", str(tmp_path)])
    assert rc == 0
    analysis_dir = tmp_path / "analysis"
    for name in ("radius_table.json", "diversity_by_sigma.json", "directional_transfer.json", "expert_overlap.json", "delta_numeric_audit.json", "stage6_analysis.md"):
        assert (analysis_dir / name).exists()
        if name.endswith(".json"):
            json.loads((analysis_dir / name).read_text())  # must be strictly valid JSON (no NaN)


def test_main_refuses_row_count_mismatch(tmp_path):
    _write_synthetic_run(tmp_path)
    checkpoint_path = tmp_path / "checkpoint_manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["expected_result_rows"] += 1
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="rows"):
        stage6.main(["--results-dir", str(tmp_path)])


def test_main_refuses_unique_perturbation_count_mismatch(tmp_path):
    _write_synthetic_run(tmp_path)
    checkpoint_path = tmp_path / "checkpoint_manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["expected_unique_perturbations"] += 1
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="unique perturbations"):
        stage6.main(["--results-dir", str(tmp_path)])


def test_main_report_never_alters_results_jsonl(tmp_path):
    _write_synthetic_run(tmp_path)
    before = (tmp_path / "results.jsonl").read_text()
    stage6.main(["--results-dir", str(tmp_path)])
    after = (tmp_path / "results.jsonl").read_text()
    assert before == after
