"""Exact-identity tests for the margin/additivity follow-up."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import margin_additivity as m  # noqa: E402


def test_margin_identity_random():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        LB, LC = rng.normal(-2, 2, 4), rng.normal(-2, 2, 4)
        y = int(rng.integers(4))
        r = m.margin_terms(LB, LC, y)
        assert r['m_C'] == pytest.approx(r['m_B'] + r['t'] - r['s'], abs=1e-12)
        assert r['s'] >= 0


def test_comparator_tie_lowest_letter_and_rank():
    r = m.margin_terms(np.array([-1., -1., -0.5, -3.]), np.zeros(4), 2)
    assert r['comparator'] == 'A' and r['gold_rank'] == 1 and r['m_B'] == pytest.approx(0.5)
    r = m.margin_terms(np.array([-1., -1., -2., -3.]), np.zeros(4), 1)
    assert r['gold_tied'] and r['m_B'] == 0 and r['gold_rank'] == 1


def test_gain_identity():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = 200; bc = rng.random(n) < .4; cc = np.where(bc, rng.random(n) < .9, rng.random(n) < .2)
        b, r, d = bc.mean(), cc[~bc].mean(), (~cc[bc]).mean()
        assert m.gain_identity(b, r, d) == pytest.approx(100 * (cc.mean() - bc.mean()))


def test_decomposition_identity():
    rng = np.random.default_rng(2)
    for _ in range(200):
        wT, wR = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6)); eT, eR = rng.normal(0, .1, 6), rng.normal(0, .1, 6)
        c, r = m.decompose(wT, wR, eT, eR)
        assert c + r == pytest.approx(100 * ((wT * eT).sum() - (wR * eR).sum()))


def test_bins_and_center():
    assert [m.margin_bin(x) for x in (0, -0.49, 0.5, 1.99, 2, -4, 9)] == ['[0,0.5)', '[0,0.5)', '[0.5,1)', '[1,2)', '[2,4)', '[4,inf)', '[4,inf)']
    z = m.center(np.array([[-1., -2., -3., -4.]]))
    assert z.sum() == pytest.approx(0) and z.argmax() == 0


def test_additive_prediction_parameter_free():
    zB = np.zeros((2, 4)); zI = [np.full((2, 4), 1.0), np.full((2, 4), 2.0)]
    assert np.allclose(m.additive_prediction(zB, zI), 3.0)
