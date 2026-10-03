import numpy as np
import pytest

from thicket_runtime.partial_evaluation import (MatrixOracle, bounds, correlation, evaluate, joint_success,
                                               order_scores, run_policy, tie_keys, vote_answers)


def test_subset_cannot_see_late_winner_and_cost_is_observed_cells_only():
    r = np.array([[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=float)
    ids = ["late", "early"]
    oracle = MatrixOracle(r, np.arange(1, 13).reshape(2, 6))
    result = run_policy(oracle.observe, r.shape, ids, list(range(6)), {"family": "subset", "prompts": 2}, k=1)
    assert result["ranking"][0] == 1
    assert oracle.cost()["evaluations"] == 4
    assert oracle.cost()["generated_tokens"] == 18
    assert evaluate(result, oracle, r, ids, (1,))["best_regret"] == pytest.approx(1/3)
    with pytest.raises(ValueError, match="previously"):
        oracle.observe([0], [0])


def test_halving_never_resurrects_a_late_winner_using_full_scores():
    r = np.array([[0]*4 + [1]*28] + [[1]*4 + [0]*28]*7)
    o = MatrixOracle(r)
    result = run_policy(o.observe, r.shape, [str(i) for i in range(8)], list(range(32)),
                        {"family": "halving", "initial": 4}, k=1)
    assert 0 in result["eliminated"] and result["counts"][0] == 4
    assert result["ranking"][0] != 0
    assert len(result["ranking"]) == len(set(result["ranking"])) == 8
    assert o.cost()["evaluations"] == 80


def test_full_budget_recovers_exact_tie_broken_reference():
    r = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    ids = ["x", "y", "z"]
    o = MatrixOracle(r)
    result = run_policy(o.observe, r.shape, ids, [2, 0, 1], {"family": "subset", "prompts": 3}, k=1)
    metrics = evaluate(result, o, r, ids, (1, 2))
    assert result["ranking"] == order_scores(r.mean(axis=1), tie_keys(ids))
    assert metrics["best_regret"] == 0 and metrics["topk"]["2"]["recall"] == 1
    assert metrics["topk"]["1"]["boundary_tie_count"] == 2
    assert metrics["pair_fraction"] == 1


def test_bounds_full_set_exact_and_zero_success_not_false_certainty():
    lo, hi = bounds([0., .5, 1.], 8, 200, "wilson", 1.96, 100, 7)
    assert hi[0] > 0 and lo[-1] < 1
    lo, hi = bounds([0., .5, 1.], 200, 200, "hoeffding", .05, 100, 7)
    assert np.array_equal(lo, hi) and np.array_equal(lo, [0., .5, 1.])


def test_conservative_race_preserves_clear_top_and_never_double_counts():
    r = np.array([[1]*200]*10 + [[0]*200]*30)
    o = MatrixOracle(r)
    result = run_policy(o.observe, r.shape, [str(i) for i in range(40)], list(range(200)),
                        {"family": "hoeffding", "parameter": .05})
    assert set(result["ranking"][:10]) == set(range(10))
    assert all(result["counts"][i] == 200 for i in range(10))
    assert o.cost()["evaluations"] < 40*200


def test_gate_is_joint_not_averages_and_votes_follow_upstream_ties():
    g = {"maximum_pair_fraction": .4, "minimum_top10_recall": .9, "maximum_best_regret": .01}
    m = {"pair_fraction": .4, "topk": {"10": {"recall": .9}}, "best_regret": .010000000000000009}
    assert joint_success(m, g)
    assert not joint_success(dict(m, pair_fraction=.41), g)
    assert vote_answers([["1", ""], ["2", "3"]], [0, 1], 2) == ["1", "3"]
    assert vote_answers([["1"], ["2"]], [1, 0], 2) == ["2"]


def test_rank_correlations_handle_ties_without_external_stats_dependency():
    assert correlation([1, 1, 2], [1, 2, 3]) == pytest.approx(3**.5/2)
    assert correlation([1, 1, 2], [1, 2, 3], "kendall") == pytest.approx(2/6**.5)
    assert correlation([1, 2, 3], [3, 2, 1], "kendall") == -1
    assert correlation([1, 1], [2, 3]) is None
