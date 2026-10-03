from collections import Counter

import numpy as np
import pytest

from thicket_runtime.committee_selection import (VoteTable, frozen_gate, greedy_committee,
                                                matched_random, voting_details)


def reward(text, gold):
    return float(text == '#### ' + gold)


def naive(answers, members, gold):
    result = []
    for j, expected in enumerate(gold):
        votes = Counter(answers[i][j] for i in members if answers[i][j])
        winner = votes.most_common(1)[0][0] if votes else ''
        result.append(bool(winner and winner == expected))
    return sum(result)


def test_incremental_votes_match_counter_on_empty_answers_and_changing_ties():
    rng = np.random.default_rng(37)
    answers = rng.choice(['', '1', '2', '3'], size=(18, 29)).tolist()
    gold = rng.choice(['1', '2', '3'], size=29).tolist()
    table = VoteTable(answers, gold, reward)
    chosen = []
    for i in [7, 3, 10, 15, 5, 17, 0]:
        scores, _ = table.append_trials()
        for trial in set(range(18)) - set(chosen):
            assert int(scores[trial]) == naive(answers, chosen + [trial], gold)
        chosen.append(i)
        assert table.append(i) == naive(answers, chosen, gold)
    with pytest.raises(ValueError, match='duplicate'):
        table.append(7)


def test_greedy_really_optimizes_each_append_and_never_reads_test_columns():
    rng = np.random.default_rng(19)
    answers = rng.choice(['', '1', '2'], size=(12, 23)).tolist()
    gold = ['1'] * 23
    individual = [sum(a == '1' for a in row) for row in answers]
    ids = [str(i) for i in range(12)]
    committee, history = greedy_committee(answers, gold, individual, ids, reward, 8, 0)
    assert individual[committee[0]] == max(individual)
    for k in range(1, 8):
        previous = committee[:k]
        optimum = max(naive(answers, previous + [i], gold) for i in set(range(12)) - set(previous))
        assert history[k]['selection_correct'] == optimum
    assert sum(v['correct'] for v in voting_details(answers, committee, gold, reward)) == history[-1]['selection_correct']


def test_random_controls_preserve_exact_quality_multiset_for_every_prefix():
    scores = [10, 10, 9, 9, 9, 7, 7, 7]
    original = [0, 2, 5, 1, 3, 6]
    actual = matched_random(original, scores, list(map(str, range(8))), 101)
    assert len(set(actual)) == len(actual)
    assert [scores[i] for i in actual] == [scores[i] for i in original]
    assert actual == matched_random(original, scores, list(map(str, range(8))), 101)


def test_gate_requires_same_comparison_on_both_populations_and_exact_thresholds():
    config = {'efficiency_complementary_k': [5, 10], 'efficiency_standard_k': [20, 50],
              'maximum_accuracy_loss': .01, 'same_k': [3, 5, 10, 20], 'minimum_accuracy_gain': .02}
    def population():
        return {**{f'standard-k{k}': .5 for k in [3, 5, 10, 20, 50]},
                **{f'greedy-0-k{k}': .47 for k in [3, 5, 10, 20]}}
    pops = {'validation_a': population(), 'validation_b': population()}
    pops['validation_a']['greedy-0-k5'] = .49
    pops['validation_b']['greedy-0-k10'] = .49
    assert not frozen_gate(pops, config)['pass']
    pops['validation_b']['greedy-0-k5'] = .49
    assert frozen_gate(pops, config)['pass']
    pops = {'validation_a': population(), 'validation_b': population()}
    for row in pops.values():
        row['greedy-0-k3'] = .52
    assert frozen_gate(pops, config)['pass']
