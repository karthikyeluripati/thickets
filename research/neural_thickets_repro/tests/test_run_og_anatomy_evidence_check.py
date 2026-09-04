"""Regression tests for run_og_anatomy_evidence_check.py.

Pure CPU, no GPU/vllm/ray import at any point -- this module and its tests only read
already-committed CSV/Markdown artifacts under results/.
"""

import json

import numpy as np
import pytest

from neural_thickets_repro import run_og_anatomy_evidence_check as m


# ------------------------------- 32B/72B guard ------------------------------------------------

def test_ensure_no_32b_72b_rejects_32b_token():
    with pytest.raises(ValueError, match="forbidden scale token"):
        m._ensure_no_32b_72b_in_argv(["--data-dir", "/some/32b_results"])


def test_ensure_no_32b_72b_rejects_72b_token_case_insensitive():
    with pytest.raises(ValueError, match="forbidden scale token"):
        m._ensure_no_32b_72b_in_argv(["--config", "STAGE_72B.yaml"])


def test_ensure_no_32b_72b_allows_ordinary_argv():
    m._ensure_no_32b_72b_in_argv(["--output-dir", "reports/og_anatomy_evidence_check"])  # no raise


# ------------------------------- _top_region_by_density ---------------------------------------

def _cell(capability, region, radius, density):
    return m.CellStat(
        capability=capability, region=region, radius=radius, n=64, mean_delta=0.0,
        std_delta=0.01, density_ge_0_02=density, density_ge_0_0=1.0, density_ge_0_05=0.0,
        positive_thicket_mass=0.0,
    )


def test_top_region_by_density_picks_highest():
    cells = [_cell("cap", "vision", 0.1, 0.3), _cell("cap", "language", 0.1, 0.5), _cell("cap", "multimodal_connector_or_merger", 0.1, 0.1)]
    top, second, densities = m._top_region_by_density(cells, "cap", 0.1)
    assert top == "language"
    assert second == "vision"
    assert densities == {"vision": 0.3, "language": 0.5, "multimodal_connector_or_merger": 0.1}


def test_top_region_by_density_tie_returns_none():
    cells = [_cell("cap", "vision", 0.1, 0.0), _cell("cap", "language", 0.1, 0.0), _cell("cap", "multimodal_connector_or_merger", 0.1, 0.0)]
    top, second, _ = m._top_region_by_density(cells, "cap", 0.1)
    assert top is None
    assert second is None


def test_top_region_by_density_missing_capability_returns_none():
    cells = [_cell("other_cap", "vision", 0.1, 0.5)]
    top, second, densities = m._top_region_by_density(cells, "cap", 0.1)
    assert top is None
    assert densities == {}


# ------------------------------- CellStat.approx_95ci_mean -------------------------------------

def test_approx_95ci_mean_widens_with_std_narrows_with_n():
    tight = m.CellStat("c", "r", 0.1, 64, 0.02, 0.01, 0.5, 1.0, 0.0, 0.0)
    lo, hi = tight.approx_95ci_mean()
    assert lo < 0.02 < hi
    wide = m.CellStat("c", "r", 0.1, 8, 0.02, 0.05, 0.5, 1.0, 0.0, 0.0)
    lo2, hi2 = wide.approx_95ci_mean()
    assert (hi2 - lo2) > (hi - lo)


def test_approx_95ci_mean_n_le_1_returns_degenerate_point():
    single = m.CellStat("c", "r", 0.1, 1, 0.02, 0.01, 0.5, 1.0, 0.0, 0.0)
    assert single.approx_95ci_mean() == (0.02, 0.02)


# ------------------------------- transfer / guided-search / cross-scale -----------------------

def test_transfer_results_against_real_local_raw_data_or_gracefully_not_measurable():
    """This repo's raw stage8 results.jsonl is gitignored -- present when locally
    restored (as it is in this developer's worktree), absent on a bare fresh clone.
    Either way the block must return a well-formed, honestly-labeled result."""
    result = m.compute_transfer_results()
    assert result["block"] == "STRUCTURED_TRANSFER"
    assert result["verdict"] in ("PASS", "FAIL", "NOT_MEASURABLE")
    if result["verdict"] == "NOT_MEASURABLE":
        assert "raw per-candidate" in result["reason"] or "not found" in result["reason"]
    else:
        assert "full_matrix_summary" in result


def test_transfer_results_not_measurable_when_raw_file_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "STAGE8_RAW_RESULTS_PATH", tmp_path / "does_not_exist.jsonl")
    result = m.compute_transfer_results()
    assert result["verdict"] == "NOT_MEASURABLE"
    assert "gitignored" in result["reason"] or "not found" in result["reason"]


def test_guided_search_results_against_real_local_raw_data_or_gracefully_not_measurable():
    result = m.compute_guided_search_results()
    assert result["block"] == "GUIDED_SEARCH_VALUE"
    assert result["verdict"] in ("PASS", "FAIL", "NOT_MEASURABLE")


def test_cross_scale_results_is_not_measurable_and_names_missing_cells():
    result = m.compute_cross_scale_results()
    assert result["block"] == "CROSS_SCALE_CONSISTENCY"
    assert result["verdict"] == "NOT_MEASURABLE"
    missing = result["exact_missing_S1_cells_required"]
    assert set(missing["capabilities"]) == set(m.CAPABILITIES)
    assert missing["total_missing_candidate_evaluations"] == len(m.CAPABILITIES) * len(m.REGIONS) * len(m.RADII_3B) * 64


# ------------------------------- compute_decision logic ---------------------------------------

def test_decision_stop_or_reframe_when_anatomy_fails():
    decision = m.compute_decision(
        {"verdict": "FAIL"}, {"verdict": "NOT_MEASURABLE"}, {"verdict": "NOT_MEASURABLE"}, {"verdict": "NOT_MEASURABLE"},
    )
    assert decision["decision"] == "STOP_OR_REFRAME"


def test_decision_go_strong_when_everything_passes():
    decision = m.compute_decision(
        {"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "PASS"},
    )
    assert decision["decision"] == "GO_STRONG"


def test_decision_go_anatomy_only_when_anatomy_passes_but_transfer_fails():
    decision = m.compute_decision(
        {"verdict": "PASS"}, {"verdict": "FAIL"}, {"verdict": "PASS"}, {"verdict": "PASS"},
    )
    assert decision["decision"] == "GO_ANATOMY_ONLY"


def test_decision_never_claims_go_strong_when_anatomy_is_not_pass():
    decision = m.compute_decision(
        {"verdict": "NOT_MEASURABLE"}, {"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "PASS"},
    )
    assert decision["decision"] != "GO_STRONG"


# ------------------------------- real-data integration (deterministic reproduction) -----------

def test_compute_anatomy_results_against_real_committed_data_is_deterministic():
    """Running the real committed analysis against the real committed artifacts twice must
    produce byte-identical output -- this is the reproducibility property the final report
    claims, verified here rather than merely asserted."""
    first = m.compute_anatomy_results()
    second = m.compute_anatomy_results()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_compute_anatomy_results_covers_all_six_capabilities():
    result = m.compute_anatomy_results()
    capabilities_seen = {row["capability"] for row in result["per_capability"]}
    assert capabilities_seen == set(m.CAPABILITIES)


def test_compute_anatomy_results_never_reports_pass_without_meeting_its_own_thresholds():
    result = m.compute_anatomy_results()
    if result["verdict"] == "PASS":
        assert result["criterion_2_at_least_3_capabilities_reproducible_3B_pattern"]["met"]
        assert result["criterion_3_survives_held_out_7B_evaluation"]["met"]
    else:
        assert result["verdict"] == "FAIL"


# ------------------------------- raw-data-driven logic (synthetic, unit-level) ----------------

def _write_synthetic_stage8_raw(path):
    """6 capabilities x 3 regions x 1 radius x 20 directions, small but structurally valid
    (every perturbation_id has exactly 6 rows, one per capability) synthetic dataset --
    exercises load_stage8_raw_rows / build_matrix / guided-search logic without depending
    on the real (gitignored, locally-restored-only) data file."""
    import random
    rng = random.Random(0)
    radius = m.RADII_3B[0]
    lines = []
    for region in m.REGIONS:
        for direction_index in range(20):
            pid = f"{region}-{direction_index}"
            seed = rng.randint(0, 2**31)
            for cap in m.CAPABILITIES:
                delta = rng.uniform(-0.03, 0.03)
                lines.append(json.dumps({
                    "perturbation_id": pid, "anatomy_region": region, "radius": radius,
                    "seed": seed, "capability": cap, "delta": delta,
                    "runtime_metadata": {"direction_index": direction_index},
                }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_stage8_raw_rows_reads_synthetic_file(tmp_path, monkeypatch):
    p = tmp_path / "results.jsonl"
    _write_synthetic_stage8_raw(p)
    monkeypatch.setattr(m, "STAGE8_RAW_RESULTS_PATH", p)
    rows = m.load_stage8_raw_rows()
    assert len(rows) == 3 * 20 * 6
    assert all(isinstance(r, m.RawCandidateRow) for r in rows)


def test_transfer_results_runs_on_synthetic_data_and_is_well_formed(tmp_path, monkeypatch):
    p = tmp_path / "results.jsonl"
    _write_synthetic_stage8_raw(p)
    monkeypatch.setattr(m, "STAGE8_RAW_RESULTS_PATH", p)
    result = m.compute_transfer_results()
    assert result["verdict"] in ("PASS", "FAIL")
    assert "full_matrix_summary" in result
    pairs_seen = {(e["source_capability"], e["target_capability"]) for e in result["full_matrix_summary"]}
    assert all(s != t for s, t in pairs_seen)  # never a capability transferring to itself


def test_guided_search_results_runs_on_synthetic_data_and_is_well_formed(tmp_path, monkeypatch):
    p = tmp_path / "results.jsonl"
    _write_synthetic_stage8_raw(p)
    monkeypatch.setattr(m, "STAGE8_RAW_RESULTS_PATH", p)
    result = m.compute_guided_search_results()
    # synthetic dataset only has 1 radius and 20 (not 64) directions per region/cap ->
    # the >=3-radii-and-64-direction gate should make this legitimately NOT_MEASURABLE-shaped
    # (empty per-radius lists), never a crash and never a fabricated PASS/FAIL from too little data.
    assert result["block"] == "GUIDED_SEARCH_VALUE"
    for cap_entry in result["per_capability"]:
        assert cap_entry["per_radius"] == []  # 20 != 64 directions -> correctly skipped, not faked


def test_bootstrap_ci_mean_is_deterministic_given_seed():
    deltas = np.array([0.01, 0.02, -0.01, 0.03, 0.0, 0.015, -0.02, 0.025])
    lo1, hi1 = m._bootstrap_ci_mean(deltas, seed=42)
    lo2, hi2 = m._bootstrap_ci_mean(deltas, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= np.mean(deltas) <= hi1


def test_main_writes_all_five_json_outputs_and_returns_zero(tmp_path):
    out_dir = tmp_path / "og_anatomy_out"
    rc = m.main(["--output-dir", str(out_dir)])
    assert rc == 0
    for name in (
        "anatomy_results.json", "transfer_results.json", "guided_search_results.json",
        "cross_scale_results.json", "paper_viability_decision.json",
    ):
        assert (out_dir / name).exists(), name
        json.loads((out_dir / name).read_text(encoding="utf-8"))  # must be valid JSON


def test_main_rejects_32b_argv(tmp_path):
    with pytest.raises(ValueError):
        m.main(["--output-dir", str(tmp_path / "32b_out")])
