import numpy as np
import pandas as pd
import pytest

from tools.momentum import rebalance_dates, momentum_scores, eligible, select_topk


def test_eligible_drops_illiquid_and_short_history():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({
        "LIQ":   np.linspace(100, 110, 300),         # ok
        "WIDE":  np.linspace(100, 110, 300),         # too-wide spread
        "SHORT": [np.nan] * 250 + list(np.linspace(100, 110, 50)),  # short history
    }, index=idx)
    slip = {"LIQ": 10, "WIDE": 80, "SHORT": 10}
    elig = eligible(px, idx[-1], slip, liq_max=30, min_obs=273)
    assert elig == {"LIQ"}


def test_select_topk_ranks_and_respects_eligibility():
    scores = pd.Series({"A": 0.5, "B": 0.9, "C": 0.1, "D": 0.7})
    picks = select_topk(scores, {"A", "B", "C", "D"}, k=2)
    assert picks == ["B", "D"]                       # highest two, ordered
    picks2 = select_topk(scores, {"A", "C"}, k=5)    # eligibility limits pool
    assert picks2 == ["A", "C"]                       # k larger than pool is fine


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


from tools.momentum import run_momentum


def _multi(seed=1, n=400, ncols=20):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = {f"T{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
            for i in range(ncols)}
    return pd.DataFrame(cols, idx)


def test_run_momentum_cost_monotonic_same_schedule():
    px = _multi()
    slip = {t: 10 for t in px.columns}
    r = run_momentum(px, slip, k=5, cost_mults=(0.0, 1.0, 2.0))
    n0 = r["runs"][0.0]["stats"]["net_return"]
    n1 = r["runs"][1.0]["stats"]["net_return"]
    n2 = r["runs"][2.0]["stats"]["net_return"]
    assert n0 >= n1 >= n2                              # more cost -> lower net
    assert len(r["holdings_log"]) >= 2                 # schedule is cost-independent
    assert all(len(h["picks"]) <= 5 for h in r["holdings_log"])


def test_run_momentum_equity_starts_at_capital():
    px = _multi()
    slip = {t: 10 for t in px.columns}
    r = run_momentum(px, slip, k=5, capital=10_000.0, cost_mults=(1.0,))
    eq = r["runs"][1.0]["equity"]
    assert abs(eq.iloc[0] - 10_000.0) < 1e-6
    assert "stats" in r["runs"][1.0] and "sharpe" in r["runs"][1.0]["stats"]
    assert eq.index[0] == r["holdings_log"][0]["date"]   # anchored at first rebalance


def test_run_momentum_no_history_returns_empty_schedule():
    idx = pd.bdate_range("2020-01-01", periods=100)
    px = pd.DataFrame({"A": np.linspace(100, 110, 100)}, index=idx)
    r = run_momentum(px, {"A": 10}, k=5, cost_mults=(1.0,))
    assert r["holdings_log"] == []
