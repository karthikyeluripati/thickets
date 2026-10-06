"""Tests for the selection-vs-shared-response analysis."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import selection_vs_shared as s  # noqa: E402


def test_accounting_identity():
    rng = np.random.default_rng(0)
    for _ in range(500):
        gR, gT, mR, mT = rng.normal(0, 5, 4)
        r = s.accounting(gR, gT, mR, mT)
        assert r['gap'] == pytest.approx(r['group_mean_change'] + r['winner_advantage_change'])


def test_original_tie_rule():
    ids = ['a', 'b', 'c']
    sel = {'a': 50, 'b': 50, 'c': 49}
    search = {'a': 77, 'b': 78, 'c': 90}
    assert s.select_winner(ids, sel, search) == 'b'  # selection tie -> higher SEARCH count
    search = {'a': 77, 'b': 77, 'c': 90}
    assert s.select_winner(ids, sel, search) == min(['a', 'b'], key=s.rank_hash)  # then frozen hash


def test_stratified_halves_disjoint_balanced():
    codes = np.repeat(np.arange(12), [36, 30, 27, 26, 17, 17, 16, 15, 5, 4, 4, 3])
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = s.stratified_halves(codes, rng)
        assert len(a) == len(b) == 100 and not set(a) & set(b) and len(set(a) | set(b)) == 200
        for k in range(12):
            assert abs((codes[a] == k).sum() - (codes[b] == k).sum()) <= 1


def test_describe_and_position():
    g = [-1, 0, 2, 8, 8]
    d = s.describe(g)
    assert d['above_base'] == 3 and d['max'] == 8
    p = s.position(8, g)
    assert p['rank_best_1'] == 1 and p['tied_with'] == 1 and p['share_strictly_below'] == pytest.approx(0.6)


def test_position_non_member():
    p = s.position(8.0, [7.5, 1.0, 8.0], member=False)
    assert p['rank_best_1'] == 1 and p['tied_with'] == 1 and not p['winner_is_member']
