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


from tools.momentum import benchmark_curves, equal_weight_curve


def test_benchmark_curve_buy_hold_normalized():
    idx = pd.bdate_range("2021-01-01", periods=100)
    bench = pd.DataFrame({"MSCI World": np.linspace(100, 120, 100)}, index=idx)
    window = idx[10:90]
    curves = benchmark_curves(bench, window, capital=10_000.0)
    c = curves["MSCI World"]
    assert abs(c.iloc[0] - 10_000.0) < 1e-6           # starts at capital
    assert c.iloc[-1] > c.iloc[0]                      # rising benchmark rises
    assert list(c.index) == list(window)


def test_equal_weight_curve_starts_at_capital():
    idx = pd.bdate_range("2021-01-01", periods=120)
    px = pd.DataFrame({"A": np.linspace(100, 130, 120),
                       "B": np.linspace(100, 110, 120)}, index=idx)
    window = idx[5:115]
    c = equal_weight_curve(px, ["A", "B"], window, capital=10_000.0)
    assert abs(c.iloc[0] - 10_000.0) < 1e-6
    assert c.iloc[-1] > c.iloc[0]


def test_equal_weight_curve_empty_tickers_returns_empty():
    idx = pd.bdate_range("2021-01-01", periods=10)
    px = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=idx)
    assert equal_weight_curve(px, [], idx, capital=10_000.0).empty


import re

import build_momentum_report as bmr


def _fake_gather():
    idx = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(3)
    px = pd.DataFrame({f"T{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 400)))
                       for i in range(20)}, index=idx)
    slip = {t: 10 for t in px.columns}
    from tools.momentum import run_momentum
    res = run_momentum(px, slip, k=5, cost_mults=(0.0, 1.0, 2.0), capital=10_000.0)
    return dict(res=res, prices=px, benchmarks=pd.DataFrame(index=idx), capital=10_000.0,
                meta={t: dict(name=t, local_id="000", country="X", sector="Y") for t in px.columns})


def test_public_report_has_no_euro_amounts():
    d = _fake_gather()
    html = bmr.build(d, public=True)
    euros = re.findall(r"€[0-9][0-9.,]*", html)
    assert all(e == "€1" for e in euros), euros


def test_private_report_builds_nonempty():
    d = _fake_gather()
    html = bmr.build(d, public=False)
    assert "<html" in html.lower() and "momentum" in html.lower()
    assert "Sharpe" in html
