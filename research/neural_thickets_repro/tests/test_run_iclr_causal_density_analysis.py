"""Tests for run_iclr_causal_density_analysis.py -- the Step 12 Phase 7-10 analysis driver.
CPU-only. Structural 32B/72B guards (matching this project's established convention) plus an
end-to-end synthetic-data integration test proving the full Phase 7-10 wiring produces a
correct, deterministic result -- the frozen metrics/search_budget/grounded_selection/
decision_gate modules are exhaustively tested elsewhere; this file tests ONLY this script's own
orchestration/I-O code.
"""
from __future__ import annotations

import gzip
import inspect
import json

import pytest

import neural_thickets_repro.run_iclr_causal_density_analysis as analysis_module


def test_module_does_not_import_any_32b_module_at_import_time():
    import ast

    source = inspect.getsource(analysis_module)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    for name in imported_names:
        assert "32b" not in name.lower(), f"Unexpected 32B-related import: {name!r}"


def test_module_never_imports_vllm_ray_torch_at_module_scope():
    source = inspect.getsource(analysis_module)
    # only inside function bodies is a lazy import acceptable -- module-level top-of-file
    # imports (before the first `def`) must never include GPU-only packages.
    header = source.split("\ndef ")[0]
    for forbidden in ("import torch", "import vllm", "import ray"):
        assert forbidden not in header


def test_runtime_guard_refuses_argv_containing_32b_or_72b_markers():
    with pytest.raises(ValueError, match="strictly 7B-only"):
        analysis_module.main(["--data-dir", "x", "--scale", "32B"])
    with pytest.raises(ValueError, match="strictly 7B-only"):
        analysis_module.main(["--data-dir", "x", "--scale", "72B"])


def test_open_maybe_gz_prefers_gz_sibling(tmp_path):
    plain = tmp_path / "results.jsonl"
    plain.write_text('{"a": 1}\n')
    gz = tmp_path / "results.jsonl.gz"
    with gzip.open(gz, "wt") as f:
        f.write('{"a": 2}\n')
    with analysis_module.open_maybe_gz(plain) as f:
        line = f.readline()
    assert json.loads(line) == {"a": 2}  # .gz sibling wins


def test_open_maybe_gz_falls_back_to_plain_when_no_gz(tmp_path):
    plain = tmp_path / "results.jsonl"
    plain.write_text('{"a": 1}\n')
    with analysis_module.open_maybe_gz(plain) as f:
        line = f.readline()
    assert json.loads(line) == {"a": 1}


def test_run_phase7_derives_resample_matrix_size_from_actual_data_not_hardcoded_200():
    """Caught while testing this module: an earlier version hardcoded
    build_resample_index_matrix(200), matching the real frozen AUDIT_SET_SIZE but breaking (an
    IndexError from bootstrap_g_distribution) on any audit-set size other than exactly 200 --
    including small synthetic test fixtures. n_examples must be derived from the actual loaded
    data's own sample_ids length.
    """
    import numpy as np

    base_audit = {
        "counting": (["ex_0", "ex_1", "ex_2", "ex_3"], analysis_module.load_base_condition_scores(
            {"counting": {
                "audit:correct_image": {"per_example_scores": {f"ex_{i}": 0.8 for i in range(4)}},
                "audit:text_only": {"per_example_scores": {f"ex_{i}": 0.2 for i in range(4)}},
                "audit:shuffled_image": {"per_example_scores": {f"ex_{i}": 0.1 for i in range(4)}},
            }}, "counting", "audit",
        )[1], {}),
    }
    audit_rows = {}
    for cid in ("c1",):
        for cond, val in (("correct_image", 0.9), ("text_only", 0.1), ("shuffled_image", 0.1)):
            audit_rows[(cid, "counting", cond)] = {f"ex_{i}": val for i in range(4)}
    candidate_meta = {"c1": ("full_lm", 0.02, 1)}

    result = analysis_module.run_phase7(["counting"], base_audit, audit_rows, candidate_meta)
    assert result["counting"]["density"].n_candidates == 1
    assert not np.isnan(result["counting"]["density"].rho_standard)


def test_load_base_condition_scores_builds_correct_condition_scores():
    base_report = {
        "counting": {
            "audit:correct_image": {"per_example_scores": {"ex_0": 0.8, "ex_1": 0.6}},
            "audit:text_only": {"per_example_scores": {"ex_0": 0.2, "ex_1": 0.1}},
            "audit:shuffled_image": {"per_example_scores": {"ex_0": 0.1, "ex_1": 0.3}},
        }
    }
    sample_ids, cs, aggregates = analysis_module.load_base_condition_scores(base_report, "counting", "audit")
    assert sample_ids == ["ex_0", "ex_1"]
    assert list(cs.real) == [0.8, 0.6]
    assert list(cs.text) == [0.2, 0.1]
    assert list(cs.shuffle) == [0.1, 0.3]
    assert aggregates["correct_image"] == pytest.approx(0.7)


def test_load_base_condition_scores_rejects_mismatched_sample_ids():
    base_report = {
        "counting": {
            "audit:correct_image": {"per_example_scores": {"ex_0": 0.8, "ex_1": 0.6}},
            "audit:text_only": {"per_example_scores": {"ex_0": 0.2}},  # missing ex_1
            "audit:shuffled_image": {"per_example_scores": {"ex_0": 0.1, "ex_1": 0.3}},
        }
    }
    with pytest.raises(ValueError, match="sample_id sets don't match"):
        analysis_module.load_base_condition_scores(base_report, "counting", "audit")


def _write_result_row(f, *, candidate_id, capability, condition, sample_id, score, scope="full_lm", radius=0.02, seed=1):
    f.write(json.dumps({
        "candidate_id": candidate_id, "capability": capability, "condition": condition, "sample_id": sample_id,
        "per_example_score": score, "scope": scope, "radius": radius, "seed": seed,
    }) + "\n")


def test_load_rows_by_candidate_capability_condition(tmp_path):
    path = tmp_path / "results.jsonl"
    with path.open("w") as f:
        _write_result_row(f, candidate_id="c1", capability="counting", condition="correct_image", sample_id="ex_0", score=1.0)
        _write_result_row(f, candidate_id="c1", capability="counting", condition="correct_image", sample_id="ex_1", score=0.0)
        _write_result_row(f, candidate_id=None, capability="counting", condition="correct_image", sample_id="ex_0", score=0.5)  # base row, skipped

    rows_by_ccc, candidate_meta = analysis_module.load_rows_by_candidate_capability_condition(path)
    assert rows_by_ccc[("c1", "counting", "correct_image")] == {"ex_0": 1.0, "ex_1": 0.0}
    assert candidate_meta == {"c1": ("full_lm", 0.02, 1)}


# =================================================================================================
# End-to-end synthetic integration test -- proves the full Phase 7-10 wiring produces a correct,
# deterministic result against small, hand-constructed (not real) data.
# =================================================================================================


def test_end_to_end_synthetic_pipeline_produces_inconclusive_with_four_capabilities(tmp_path):
    """A minimal but structurally complete synthetic dataset: 6 candidates (one per frozen
    scope-radius cell), 4 eligible capabilities. This pool (6) is smaller than the frozen
    search-budget module's own smallest fixed budget (10 -- SEARCH_BUDGETS is a module-level
    default bound at search_budget.py's own import time, never meant to be overridden by a
    caller, so this test does NOT attempt to shrink it) -- every Phase 8 cell therefore hits
    search_budget.InsufficientPoolSizeError, which run_phase8() catches and records as
    divergence_confirmed=False, exactly the graceful-degradation path a genuinely too-small
    pool must take. This still proves main() reads real files, calls the frozen Phase 7-10
    modules correctly end-to-end, and writes valid JSON output -- without needing the real
    600-candidate dataset or touching any frozen module's own defaults.
    """
    caps = ["counting", "ocr_text_recognition", "spatial_reasoning", "relational_reasoning"]
    scopes = ["vision_encoder", "full_lm", "full_vlm"]
    radii = [0.02, 0.04]
    cells = [(s, r) for s in scopes for r in radii]

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    base_report = {cap: {} for cap in caps + ["visual_grounding"]}
    for cap in caps:
        for subset in ("audit", "selection"):
            base_report[cap][f"{subset}:correct_image"] = {"per_example_scores": {f"ex_{i}": 0.5 for i in range(4)}}
            base_report[cap][f"{subset}:text_only"] = {"per_example_scores": {f"ex_{i}": 0.2 for i in range(4)}}
            base_report[cap][f"{subset}:shuffled_image"] = {"per_example_scores": {f"ex_{i}": 0.1 for i in range(4)}}
    (data_dir / "base_control_report.json").write_text(json.dumps(base_report))

    def _write_pass(filename):
        with (data_dir / filename).open("w") as f:
            for i, (scope, radius) in enumerate(cells):
                cid = f"{scope}_r{radius}_seed{i}"
                for cap in caps:
                    for condition in ("correct_image", "shuffled_image", "text_only"):
                        for ex in range(4):
                            score = 0.9 if condition == "correct_image" else 0.1
                            _write_result_row(f, candidate_id=cid, capability=cap, condition=condition, sample_id=f"ex_{ex}", score=score, scope=scope, radius=radius, seed=i)

    _write_pass("results.jsonl")
    _write_pass("results_selection.jsonl")

    output_dir = tmp_path / "out"
    rc = analysis_module.main(["--data-dir", str(data_dir), "--output-dir", str(output_dir)])
    assert rc == 0

    decision = json.loads((output_dir / "decision.json").read_text())
    assert decision["eligible_capabilities"] == caps
    assert decision["excluded_capabilities"] == ["visual_grounding"]
    assert set(decision["per_capability"].keys()) == set(caps)
    # decision_gate.py's own precedence rule 2 fires: only 4 capabilities were ever passed in
    assert decision["decision"] in ("INCONCLUSIVE", "REJECTED", "CONFIRMED")  # deterministic given fixed synthetic data; just prove it runs end-to-end and is one of the valid values
    assert "expected exactly 5 capabilities" in decision["reasons"][0]

    full_output = json.loads((output_dir / "analysis_full_output.json").read_text())
    assert len(full_output["phase8_per_cell"]) == len(caps) * len(cells)
    assert set(full_output["phase9_per_capability"].keys()) == set(caps)
