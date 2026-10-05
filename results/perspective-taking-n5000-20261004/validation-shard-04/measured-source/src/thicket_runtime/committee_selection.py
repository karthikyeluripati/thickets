"""Greedy majority-vote selection using selection columns only; no search change."""
from collections import Counter
import hashlib

import numpy as np


def committee_ties(ids, seed):
    return [hashlib.sha256(f'committee-tie-v1:{seed}:{cid}'.encode()).hexdigest() for cid in ids]


class VoteTable:
    """Vectorized append trials exactly matching stable Counter vote ties."""
    def __init__(self, answers, gold, reward):
        self.answers = np.asarray(answers, dtype=object)
        c, p = self.answers.shape
        if len(gold) != p or not c or not p:
            raise ValueError('invalid committee data')
        self.codes = np.zeros((c, p), dtype=int)
        self.labels = []
        self.correct = np.zeros((p, c + 1), dtype=bool)
        for j in range(p):
            labels = [''] + sorted({str(a) for a in self.answers[:, j] if a})
            mapping = {a: i for i, a in enumerate(labels)}
            self.codes[:, j] = [mapping[a] for a in self.answers[:, j]]
            self.labels.append(labels)
            for i, answer in enumerate(labels):
                self.correct[j, i] = bool(answer and reward('#### ' + answer, str(gold[j])) > 0)
        self.q = np.arange(p)
        self.counts = np.zeros((p, c + 1), dtype=int)
        self.first = np.full((p, c + 1), c + 1, dtype=int)
        self.winners = np.zeros(p, dtype=int)
        self.members = []

    def append_trials(self):
        a = self.codes
        old_count = self.counts[self.q, a]
        first = self.first[self.q, a]
        winning_count = self.counts[self.q, self.winners]
        winning_first = self.first[self.q, self.winners]
        improve = (a != 0) & ((old_count + 1 > winning_count) |
                              ((old_count + 1 == winning_count) & (first < winning_first)))
        proposed = np.where(improve, a, self.winners)
        return self.correct[self.q, proposed].sum(axis=1), proposed

    def append(self, candidate):
        if candidate in self.members:
            raise ValueError('duplicate committee member')
        scores, proposed = self.append_trials()
        a = self.codes[candidate]
        valid = a != 0
        new = valid & (self.counts[self.q, a] == 0)
        self.first[self.q[new], a[new]] = len(self.members)
        self.counts[self.q[valid], a[valid]] += 1
        self.winners = proposed[candidate].copy()
        self.members.append(candidate)
        return int(scores[candidate])


def greedy_committee(answers, gold, individual_correct, candidate_ids, reward, size, tie_seed):
    table = VoteTable(answers, gold, reward)
    keys = committee_ties(candidate_ids, tie_seed)
    available = set(range(len(candidate_ids)))
    if not 0 < size <= len(available):
        raise ValueError('invalid committee size')
    history = []
    for step in range(size):
        scores, _ = table.append_trials()
        if step == 0:
            chosen = min(available, key=lambda i: (-individual_correct[i], keys[i]))
            tied = [i for i in available if individual_correct[i] == individual_correct[chosen]]
        else:
            chosen = min(available, key=lambda i: (-scores[i], -individual_correct[i], keys[i]))
            tied = [i for i in available if scores[i] == scores[chosen] and individual_correct[i] == individual_correct[chosen]]
        correct = table.append(chosen)
        history.append({'k': step + 1, 'added_index': chosen, 'candidate_id': candidate_ids[chosen],
                        'selection_correct': correct, 'tie_count_after_quality': len(tied),
                        'marginal_correct': correct - (history[-1]['selection_correct'] if history else 0)})
        available.remove(chosen)
    return table.members, history


def matched_random(sequence, individual_correct, candidate_ids, seed):
    rng = np.random.default_rng(seed)
    pools = {}
    for score in sorted(set(individual_correct)):
        candidates = sorted((i for i, s in enumerate(individual_correct) if s == score), key=lambda i: candidate_ids[i])
        pools[score] = rng.permutation(candidates).tolist()
    result = [pools[individual_correct[i]].pop() for i in sequence]
    if len(set(result)) != len(result):
        raise AssertionError('matched random control repeats an expert')
    return result


def voting_details(answers, committee, gold, reward):
    answers = np.asarray(answers, dtype=object)
    result = []
    for j, expected in enumerate(gold):
        counter = Counter(answers[i, j] for i in committee if answers[i, j])
        ranked = counter.most_common()
        winner, count = ranked[0] if ranked else ('', 0)
        second = ranked[1][1] if len(ranked) > 1 else 0
        valid = sum(counter.values())
        result.append({'answer': winner, 'correct': bool(winner and reward('#### ' + winner, str(expected)) > 0),
                       'winning_votes': count, 'runner_up_votes': second, 'valid_votes': valid,
                       'margin': count - second, 'margin_fraction': (count - second) / valid if valid else 0.,
                       'winning_tie_count': sum(n == count for a, n in ranked),
                       'counts': [[str(a), int(n)] for a, n in ranked]})
    return result


def frozen_gate(populations, config):
    """Identical predeclared comparisons must pass both held-out populations."""
    efficiency, accuracy = [], []
    for small in config['efficiency_complementary_k']:
        for large in config['efficiency_standard_k']:
            deltas = {p: populations[p][f'greedy-0-k{small}'] - populations[p][f'standard-k{large}']
                      for p in ('validation_a', 'validation_b')}
            efficiency.append({'complementary_k': small, 'standard_k': large, 'deltas': deltas,
                               'pass': all(d >= -config['maximum_accuracy_loss'] - 1e-12 for d in deltas.values())})
    for k in config['same_k']:
        deltas = {p: populations[p][f'greedy-0-k{k}'] - populations[p][f'standard-k{k}']
                  for p in ('validation_a', 'validation_b')}
        accuracy.append({'k': k, 'deltas': deltas,
                         'pass': all(d >= config['minimum_accuracy_gain'] - 1e-12 for d in deltas.values())})
    passed = any(r['pass'] for r in efficiency + accuracy)
    return {'pass': passed, 'decision': 'GO_stop_after_report' if passed else 'NO-GO_close_direction',
            'efficiency': efficiency, 'accuracy': accuracy}
