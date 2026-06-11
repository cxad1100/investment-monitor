"""Unit tests for the pairs-trading universe and engine math."""

import numpy as np
import pandas as pd
import pytest

from tools.pairs_universe import UNIVERSE, candidate_pairs
from tools.pairs_engine import engle_granger
from tools.pairs_engine import half_life


def test_universe_entries_complete():
    for tk, meta in UNIVERSE.items():
        assert set(meta) == {"sector", "currency", "slippage_bps"}, tk
        assert meta["currency"] in ("USD", "EUR")
        assert meta["slippage_bps"] in (5, 10, 15)


def test_candidate_pairs_same_sector_and_currency():
    pairs = candidate_pairs()
    assert ("BAC", "JPM") in pairs            # same sector, same currency
    for a, b in pairs:
        assert UNIVERSE[a]["sector"] == UNIVERSE[b]["sector"]
        assert UNIVERSE[a]["currency"] == UNIVERSE[b]["currency"]
    # never cross-sector or cross-currency
    assert ("NVDA", "SIE.DE") not in pairs
    assert ("JPM", "DBK.DE") not in pairs


def test_candidate_pairs_count_reasonable():
    n = len(candidate_pairs())
    assert 30 <= n <= 80                      # ~52 with the curated universe


def make_cointegrated(n=750, seed=42, beta=0.8, alpha=0.5, phi=0.7, noise=0.02):
    """Price pair sharing a random-walk driver:
    log x = random walk; log y = alpha + beta*log x + AR(1) noise."""
    rng = np.random.default_rng(seed)
    lx = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = phi * eps[i - 1] + rng.normal(0, noise)
    ly = alpha + beta * lx + eps
    idx = pd.bdate_range("2022-01-03", periods=n)
    return (pd.Series(np.exp(ly), idx, name="YYY"),
            pd.Series(np.exp(lx), idx, name="XXX"))


def make_independent(n=750, seed=7):
    """Two unrelated random walks — correlated by luck at most, never cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    a = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)), idx, name="AAA")
    b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)), idx, name="BBB")
    return a, b


def test_detects_cointegrated_pair():
    y, x = make_cointegrated()
    r = engle_granger(y, x)
    assert r["pvalue"] < 0.05
    assert r["beta"] > 0
    assert {r["y"], r["x"]} == {"YYY", "XXX"}
    assert set(r) >= {"y", "x", "alpha", "beta", "pvalue", "spread"}


def test_rejects_independent_walks():
    a, b = make_independent()
    r = engle_granger(a, b)
    assert r["pvalue"] >= 0.05


def test_spread_is_ols_residual():
    y, x = make_cointegrated()
    r = engle_granger(y, x)
    # residual of OLS with constant has (near-)zero mean by construction
    assert abs(r["spread"].mean()) < 1e-10
    assert len(r["spread"]) == len(y)


def test_half_life_known_ar1():
    # AR(1) with phi=0.9: Δs = (phi-1)·s + ε, so HL ≈ ln2/0.1 ≈ 6.9 days
    rng = np.random.default_rng(3)
    s = np.zeros(5000)
    for i in range(1, 5000):
        s[i] = 0.9 * s[i - 1] + rng.normal(0, 1)
    hl = half_life(pd.Series(s, pd.bdate_range("2010-01-04", periods=5000)))
    assert 5.5 < hl < 8.5


def test_half_life_random_walk_is_huge():
    rng = np.random.default_rng(4)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 3000)),
                   pd.bdate_range("2012-01-02", periods=3000))
    assert half_life(rw) > 60
