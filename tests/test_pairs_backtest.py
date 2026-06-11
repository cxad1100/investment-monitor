"""Backtester tests: exact P&L arithmetic and walk-forward orchestration."""

import numpy as np
import pandas as pd

from tools.pairs_backtest import backtest_stats, run_backtest, simulate_pair
from tests.test_pairs_engine import make_cointegrated, make_independent


def _frame():
    idx = pd.bdate_range("2024-01-01", periods=5)
    py = pd.Series([100.0, 100.0, 100.0, 110.0, 110.0], idx)
    px = pd.Series([100.0] * 5, idx)
    sig = pd.Series([0, 1, 1, 0, 0], idx, dtype=float)
    return py, px, sig


def test_simulate_pair_exact_pnl():
    """Hand-computed: beta=1, capital 2000 → N_y = N_x = 1000.
    signal=[0,1,1,0,0] → held=[0,0,1,1,0] (t+1 execution).
    Gross: day3 = 1000*10% - 1000*0 = 100; day4 held but flat prices = 0.
    Costs per turn = (1€ + 10bps*1000) * 2 legs = 4€; turns on day2 and day4 → 8€.
    Net = 92."""
    py, px, sig = _frame()
    res = simulate_pair(py, px, sig, beta=1.0, pair_capital=2000.0,
                        slip_y_bps=10, slip_x_bps=10, fee_eur=1.0, cost_mult=1.0)
    assert abs(res["gross"].sum() - 100.0) < 1e-9
    assert abs(res["costs"].sum() - 8.0) < 1e-9
    assert abs(res["pnl"].sum() - 92.0) < 1e-9
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["entry"] == py.index[2] and t["exit"] == py.index[4]
    assert t["side"] == 1 and t["days"] == 2
    assert abs(t["net"] - 92.0) < 1e-9


def test_simulate_pair_zero_cost_mult():
    py, px, sig = _frame()
    res = simulate_pair(py, px, sig, beta=1.0, pair_capital=2000.0,
                        slip_y_bps=10, slip_x_bps=10, fee_eur=1.0, cost_mult=0.0)
    assert abs(res["pnl"].sum() - 100.0) < 1e-9


def test_simulate_pair_force_close_at_window_end():
    py, px, _ = _frame()
    sig = pd.Series([0, 1, 1, 1, 1], py.index, dtype=float)   # never exits by itself
    res = simulate_pair(py, px, sig, beta=1.0, pair_capital=2000.0,
                        slip_y_bps=10, slip_x_bps=10)
    assert len(res["trades"]) == 1                            # closed by force
    assert res["trades"][0]["exit"] == py.index[-1]


def _synthetic_universe(n=400):
    y, x = make_cointegrated(n=n, seed=42)
    a, b = make_independent(n=n, seed=7)
    prices = pd.concat([y, x, a, b], axis=1)
    candidates = [("YYY", "XXX"), ("AAA", "BBB")]
    slippage = {c: 5 for c in prices.columns}
    return prices, candidates, slippage


def test_run_backtest_structure_and_cost_monotonicity():
    prices, cands, slip = _synthetic_universe()
    bt = run_backtest(prices, cands, slip, capital=10_000.0,
                      formation_days=120, trading_days=40)
    assert set(bt["runs"]) == {0.0, 1.0, 2.0}
    base = bt["runs"][1.0]
    eq = base["equity"]
    assert eq.index.equals(prices.index)
    assert eq.iloc[0] == 10_000.0
    assert len(base["trades"]) > 0                      # seeded — see plan header
    for t in base["trades"]:
        assert {"pair", "capital", "entry", "exit", "net", "days"} <= set(t)
    st = base["stats"]
    assert {"net_return", "sharpe", "max_drawdown", "n_trades",
            "win_rate", "avg_days", "total_costs"} <= set(st)
    # identical trades, only costs differ → frictionless ≥ realistic ≥ pessimistic
    n0 = bt["runs"][0.0]["stats"]["net_return"]
    n2 = bt["runs"][2.0]["stats"]["net_return"]
    assert n0 >= bt["runs"][1.0]["stats"]["net_return"] >= n2
    assert bt["runs"][0.0]["stats"]["total_costs"] == 0.0
    assert len(bt["windows"]) > 0
    for w in bt["windows"]:
        assert {"formation_end", "trade_start", "n_tested", "n_selected"} <= set(w)


def test_run_backtest_no_lookahead():
    """Truncating future data must not change past equity. Both runs roll
    windows from index 0, so all windows fully inside the truncation are
    identical."""
    prices, cands, slip = _synthetic_universe(n=400)
    full = run_backtest(prices, cands, slip, capital=10_000.0,
                        formation_days=120, trading_days=40,
                        cost_mults=(1.0,))["runs"][1.0]["equity"]
    trunc = run_backtest(prices.iloc[:250], cands, slip, capital=10_000.0,
                         formation_days=120, trading_days=40,
                         cost_mults=(1.0,))["runs"][1.0]["equity"]
    # windows inside first 250 rows: trading days 120-160, 160-200, 200-240
    assert np.allclose(full.iloc[:240].values, trunc.iloc[:240].values)
