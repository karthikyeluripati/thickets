import itertools

import pytest

from thicket_runtime.agreement import (compare_tokens, distribution, feasibility_gate,
                                       rank_correlation, simulate_prefix)
from thicket_runtime.shared_speculative import make_pairs


def test_prefix_stops_at_first_divergence_even_if_suffix_rejoins():
    b, c = [1, 2, 3, 4, 5, 6], [9, 2, 3, 4, 5, 6]
    metrics = compare_tokens(b, c)
    assert metrics["common_prefix_tokens"] == 0
    assert metrics["first_divergence_position"] == 0
    assert metrics["immediate_divergence"]
    s = simulate_prefix(b, c, 4)
    assert s["ideal_candidate_output_rounds"] == 6
    assert s["accepted_draft_tokens"] == 0
    assert s["wasted_positions_after_rejection"] == 3


def test_rejection_emits_corrected_token_and_never_uses_bonus():
    b, c = list(range(12)), [0, 1, 2, 3, 4, 99, 6, 7, 8, 9, 10, 11]
    s = simulate_prefix(b, c, 4)
    assert s["accepted_draft_tokens"] == 5
    assert s["verification_calls"] == 2
    assert s["fallback_token_rounds"] == 6
    assert s["ideal_candidate_output_rounds"] == 8
    assert s["wasted_positions_after_rejection"] == 2
    assert s["full_blocks_attempted"] == 2
    assert s["full_blocks_accepted"] == 1
    assert simulate_prefix(b, b, 4)["ideal_candidate_output_rounds"] == 3


def test_short_and_empty_traces_have_explicit_denominators():
    assert compare_tokens([1], [1])["first_k_match"]["2"] is False
    assert compare_tokens([1], [1])["first_k_available"]["2"] is False
    assert compare_tokens([], [])["common_prefix_fraction"] is None
    assert compare_tokens([1], [1, 2])["divergence_kind"] == "length_mismatch"
    assert simulate_prefix([1], [1, 2, 3], 4)["ideal_candidate_output_rounds"] == 3
    assert simulate_prefix([1, 2, 3], [1], 4)["ideal_candidate_output_rounds"] == 1
    assert simulate_prefix([], [], 4)["round_reduction"] is None
    with pytest.raises(ValueError):
        simulate_prefix([1], [1], 0)


def test_exhaustive_small_trace_round_bound_and_accounting():
    traces = [list(t) for n in range(5) for t in itertools.product((0, 1), repeat=n)]
    for b, c, k in itertools.product(traces, traces, (1, 2, 4)):
        s = simulate_prefix(b, c, k)
        p = compare_tokens(b, c)["common_prefix_tokens"]
        assert s["accepted_draft_tokens"] == p
        assert s["all_accepted_oracle_rounds"] <= s["ideal_candidate_output_rounds"] <= len(c)
        assert s["accepted_draft_tokens"] + int(s["rejected"]) + s["fallback_token_rounds"] == len(c)
        assert 0 <= s["wasted_positions_after_rejection"] < k
        if k == 1:
            assert s["rounds_saved"] == 0


def test_cap_equality_is_observed_not_complete_natural_trajectory():
    base = [{"token_ids": [1, 2], "prompt_token_ids": [3], "finish_reason": "length", "stop_reason": None, "reward": 0}]
    pairs = make_pairs(base, base, [{"id": "a"}])
    assert pairs[0]["complete_observed_token_equality"]
    assert not pairs[0]["uncensored_complete_equality"]
    assert pairs[0]["base_capped"]
    with pytest.raises(ValueError, match="tokenization"):
        make_pairs(base, [dict(base[0], prompt_token_ids=[4])], [{"id": "a"}])


def test_tied_reward_ranks_and_distributions():
    assert distribution([0, 2, 4, 6])["median"] == 3
    assert distribution([]) == {"n": 0}
    assert rank_correlation([1, 1, 2], [4, 4, 9]) == pytest.approx(1)
    assert rank_correlation([1, 1], [2, 3]) is None


def test_gate_requires_same_scale_and_block_on_both_workloads_and_controls():
    protocol = {"sigmas": [.001], "workloads": [{"name": "a"}, {"name": "b"}],
                "verifier_gate": {"block_sizes": [4], "minimum_candidates_per_sigma": 32,
                    "minimum_median_prefix": 4, "minimum_round_reduction": .2,
                    "minimum_initial_block_acceptance": .5}}
    good = {"candidates": 100, "common_prefix_tokens": {"median": 8},
            "simulation": {"4": {"token_weighted_round_reduction": .3}},
            "first_k": {"4": {"fraction_all_pairs": .6}}}
    summary = {"a": {"0.001": good}, "b": {"0.001": good}}
    assert feasibility_gate(summary, protocol, controls_passed=True, complete=True)["pass"]
    assert not feasibility_gate(summary, protocol, controls_passed=False, complete=True)["pass"]
    assert feasibility_gate(summary, protocol, controls_passed=True, complete=False)["status"] == "inconclusive"
    summary["b"] = {}
    assert feasibility_gate(summary, protocol, controls_passed=True, complete=True)["status"] == "inconclusive"
    summary["b"] = {"0.001": dict(good, common_prefix_tokens={"median": 1})}
    assert feasibility_gate(summary, protocol, controls_passed=True, complete=True)["status"] == "negative"
