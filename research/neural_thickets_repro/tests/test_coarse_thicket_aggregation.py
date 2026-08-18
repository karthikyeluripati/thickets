"""Tests for coarse_thicket_aggregation.py -- pure Python, no GPU/ray/vllm needed."""
import json

import pytest

from neural_thickets_repro.coarse_thicket_aggregation import (
    AggregationMismatchError,
    assert_comparable,
    build_coarse_map_rows,
    format_table,
    load_thicket_metrics,
)


def _run(scope, **overrides):
    base = {
        "task": "gqa",
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scope": scope,
        "N": 20,
        "perturbation_scale_mode": "relative_l2",
        "requested_relative_l2": 0.01,
        "global_seed": 42,
        "restoration_mode": "fixed_base",
        "noise_semantics": "upstream_per_tensor_reseed",
        "base_score": 0.52,
        "selection_set_size": 200,
        "selection_example_ids": [f"q{i}" for i in range(200)],
        "dataset_revision": "rev1",
        "dataset_selection_split": "train",
        "scoring_protocol": "gqa_image_aware_v1",
        "candidate_seed_sequence": [100, 200, 300],
        "expert_count": 8,
        "tie_count": 1,
        "regression_count": 11,
        "expert_density": 0.4,
        "expert_density_ci_95": [0.22, 0.61],
        "mean_score": 0.5,
        "std_score": 0.05,
        "mean_delta": 0.01,
        "median_delta": 0.005,
        "min_delta": -0.1,
        "max_delta": 0.15,
        "score_quantiles": {"25": 0.45, "50": 0.5, "75": 0.55},
        "best_candidate_score": 0.67,
        "best_candidate_seed": 300,
    }
    base.update(overrides)
    return base


# --- load_thicket_metrics ---


def test_load_thicket_metrics_from_file(tmp_path):
    path = tmp_path / "thicket_metrics.json"
    path.write_text(json.dumps(_run("full_lm")))
    loaded = load_thicket_metrics(path)
    assert loaded["scope"] == "full_lm"


def test_load_thicket_metrics_from_run_directory(tmp_path):
    run_dir = tmp_path / "some_run"
    run_dir.mkdir()
    (run_dir / "thicket_metrics.json").write_text(json.dumps(_run("vision_encoder")))
    loaded = load_thicket_metrics(run_dir)
    assert loaded["scope"] == "vision_encoder"


def test_load_thicket_metrics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_thicket_metrics(tmp_path / "does_not_exist")


# --- assert_comparable ---


def test_assert_comparable_passes_for_consistent_runs():
    runs = [_run("full_lm"), _run("vision_encoder"), _run("lm_middle")]
    assert_comparable(runs)  # should not raise


def test_assert_comparable_single_run_never_raises():
    assert_comparable([_run("full_lm")])
    assert_comparable([])


def test_aggregation_refuses_mismatched_radii():
    runs = [_run("full_lm", requested_relative_l2=0.01), _run("vision_encoder", requested_relative_l2=0.02)]
    with pytest.raises(AggregationMismatchError, match="requested_relative_l2"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_candidate_seeds():
    runs = [
        _run("full_lm", candidate_seed_sequence=[100, 200, 300]),
        _run("vision_encoder", candidate_seed_sequence=[100, 200, 999]),
    ]
    with pytest.raises(AggregationMismatchError, match="candidate_seed_sequence"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_candidate_seed_order():
    """Same SET of seeds in a different order must still be refused -- 'same sequence' means
    same order too, not just same membership.
    """
    runs = [
        _run("full_lm", candidate_seed_sequence=[100, 200, 300]),
        _run("vision_encoder", candidate_seed_sequence=[300, 200, 100]),
    ]
    with pytest.raises(AggregationMismatchError, match="candidate_seed_sequence"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_dataset_revision():
    runs = [_run("full_lm", dataset_revision="rev1"), _run("vision_encoder", dataset_revision="rev2")]
    with pytest.raises(AggregationMismatchError, match="dataset_revision"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_selection_example_ids():
    runs = [
        _run("full_lm", selection_example_ids=["q0", "q1"]),
        _run("vision_encoder", selection_example_ids=["q0", "q2"]),
    ]
    with pytest.raises(AggregationMismatchError, match="selection_example_ids"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_scoring_protocol():
    runs = [_run("full_lm", scoring_protocol="gqa_image_aware_v1"), _run("vision_encoder", scoring_protocol="gqa_image_aware_v2")]
    with pytest.raises(AggregationMismatchError, match="scoring_protocol"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_task():
    runs = [_run("full_lm", task="gqa"), _run("vision_encoder", task="some_other_task")]
    with pytest.raises(AggregationMismatchError, match="task"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_model_revision():
    runs = [_run("full_lm", model_revision="rev_a"), _run("vision_encoder", model_revision="rev_b")]
    with pytest.raises(AggregationMismatchError, match="model_revision"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_n():
    runs = [_run("full_lm", N=20), _run("vision_encoder", N=50)]
    with pytest.raises(AggregationMismatchError, match="N"):
        assert_comparable(runs)


def test_aggregation_refuses_mismatched_global_seed():
    runs = [_run("full_lm", global_seed=42), _run("vision_encoder", global_seed=7)]
    with pytest.raises(AggregationMismatchError, match="global_seed"):
        assert_comparable(runs)


def test_aggregation_allows_differing_scope_only():
    """scope is the ONE field that's supposed to vary -- must not be flagged as a mismatch."""
    runs = [_run("full_lm"), _run("vision_merger"), _run("lm_early")]
    assert_comparable(runs)  # should not raise despite three different scope values


# --- build_coarse_map_rows / format_table ---


def test_build_coarse_map_rows_computes_best_delta():
    runs = [_run("full_lm", base_score=0.5, best_candidate_score=0.8)]
    rows = build_coarse_map_rows(runs)
    assert rows[0]["best_delta"] == pytest.approx(0.3)


def test_build_coarse_map_rows_preserves_core_fields():
    runs = [_run("vision_encoder", requested_relative_l2=0.03, N=20)]
    rows = build_coarse_map_rows(runs)
    row = rows[0]
    assert row["scope"] == "vision_encoder"
    assert row["r"] == 0.03
    assert row["N"] == 20
    assert row["base"] == 0.52
    assert row["density"] == 0.4
    assert row["ci_lower"] == 0.22
    assert row["ci_upper"] == 0.61


def test_format_table_contains_all_scopes_and_header():
    runs = [_run("full_lm"), _run("vision_encoder"), _run("lm_middle")]
    rows = build_coarse_map_rows(runs)
    table = format_table(rows)
    assert "scope" in table and "density" in table and "CI" in table
    assert "full_lm" in table
    assert "vision_encoder" in table
    assert "lm_middle" in table


def test_format_table_orders_scopes_canonically():
    # Deliberately built out of canonical order -- table must still render full_lm before
    # vision_encoder before lm_middle, matching PERTURBATION_SCOPES order.
    runs = [_run("lm_middle"), _run("full_lm"), _run("vision_encoder")]
    rows = build_coarse_map_rows(runs)
    table = format_table(rows)
    assert table.index("full_lm") < table.index("vision_encoder") < table.index("lm_middle")


def test_format_table_empty_rows_does_not_crash():
    table = format_table([])
    assert "scope" in table  # header still renders
