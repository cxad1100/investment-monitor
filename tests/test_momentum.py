import numpy as np
import pandas as pd
import pytest

from tools.momentum import rebalance_dates, momentum_scores


def test_rebalance_dates_monthly_count():
    idx = pd.bdate_range("2022-01-03", periods=400)   # ~19 months
    dates = rebalance_dates(idx, "M")
    # one date per calendar month spanned, each a date present in the index
    months = len(set((d.year, d.month) for d in idx))
    assert len(dates) == months
    assert all(d in idx for d in dates)
    # each is the LAST index date in its month
    for d in dates:
        same_month = [x for x in idx if (x.year, x.month) == (d.year, d.month)]
        assert d == same_month[-1]


def _rw(seed, n=400, drift=0.0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


def test_momentum_scores_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    px = pd.DataFrame({"A": _rw(0), "B": _rw(1)}, index=idx)
    asof = idx[300]
    full = momentum_scores(px, asof)                 # later data present
    truncated = momentum_scores(px.loc[:asof], asof)  # later data removed
    pd.testing.assert_series_equal(full.sort_index(), truncated.sort_index())


def test_momentum_scores_is_12_1_return():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"UP": np.linspace(100.0, 200.0, 300),
                       "FLAT": np.full(300, 100.0)}, index=idx)
    s = momentum_scores(px, idx[-1], lookback=252, skip=21)
    assert s["UP"] > s["FLAT"]                       # trending beats flat
    assert abs(s["FLAT"]) < 1e-9                      # flat ~ zero momentum


def test_momentum_scores_insufficient_history_empty():
    idx = pd.bdate_range("2020-01-01", periods=100)
    px = pd.DataFrame({"A": _rw(2, 100)}, index=idx)
    assert momentum_scores(px, idx[-1], lookback=252, skip=21).empty
