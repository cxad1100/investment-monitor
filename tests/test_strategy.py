import re

import numpy as np
import pandas as pd

import build_strategy_report as bs
from tools.momentum import run_momentum
from tools.momentum_grid import _stats_slice


def _fake_d():
    idx = pd.bdate_range("2018-01-01", periods=500)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500)))
                       for i in range(15)}, index=idx)
    slip = {t: 10 for t in px.columns}
    res = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,))
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te, ve = pd.Timestamp("2019-06-30"), pd.Timestamp("2019-09-30")
    return dict(prices=px, res=res, benchmarks=pd.DataFrame(index=idx), capital=10_000.0,
                meta={t: dict(name=t, local_id="000", country="X", sector="Y") for t in px.columns},
                strategy=bs.STRATEGY,
                train=_stats_slice(eq, tr, eq.index[0], te, 10_000.0),
                val=_stats_slice(eq, tr, te + pd.Timedelta(days=1), ve, 10_000.0),
                test=_stats_slice(eq, tr, ve + pd.Timedelta(days=1), eq.index[-1], 10_000.0),
                n_dead=42, n_countries=1, n_live=10,
                significance=dict(
                    mc=dict(null_sharpe=np.array([0.1, 0.2, 0.3, 0.25]), strat_sharpe=0.6,
                            null_sharpe_median=0.22, p_sharpe=0.04, p_total=0.05, n_trials=1000),
                    dsr=dict(dsr=0.78, n_trials=32, T=20, sr_benchmark_annual=1.1,
                             sharpe_annual=1.4),
                    ci=dict(conf=95, sharpe=1.2, sharpe_lo=0.4, sharpe_hi=1.9,
                            cagr=0.3, cagr_lo=0.1, cagr_hi=0.5),
                    ppy=4.0),
                quant=dict(
                    perf=dict(sharpe=0.84, sortino=1.13, calmar=0.61, omega=1.17, ann_return=0.27,
                              ann_vol=0.32, max_dd=-0.44, dd_days=404, skew=-0.39, kurtosis=2.89,
                              var95=-0.03, cvar95=-0.05, worst_day=-0.13, best_day=0.09),
                    bench=dict(beta=0.82, alpha_ann=0.15, corr=0.43, tracking_error=0.29,
                               info_ratio=0.42, up_capture=0.85, down_capture=0.67),
                    trades=dict(n_trades=495, trades_per_year=60.0, hit_rate=0.53,
                                profit_factor=1.81, avg_win=528.0, avg_loss=-335.0, payoff=1.58),
                    roll=dict(roll_sharpe_min=-2.05, roll_sharpe_med=0.88, roll_sharpe_pos_frac=0.86),
                    grade=dict(score=62.2, letter="C", survivorship_corrected=False,
                               flags=["Survivorship NOT corrected.", "Regime.", "Multiple testing.",
                                      "Known decaying anomaly."]),
                    isin_overlap=0.02))


def test_strategy_page_builds():
    html = bs.build(_fake_d(), public=False)
    assert "<html" in html.lower() and "strategy" in html.lower()
    assert "Validation" in html and bs.STRATEGY.code in html


def test_strategy_public_no_euro_amounts():
    html = bs.build(_fake_d(), public=True)
    euros = re.findall(r"€[0-9][0-9.,]*", html)
    assert all(e == "€1" for e in euros), euros
