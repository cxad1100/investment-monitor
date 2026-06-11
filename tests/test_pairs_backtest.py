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
