"""Vectorized pairs backtester with Trade Republic-style costs.

Execution model: a signal computed on the close of day t is executed at
the close of day t+1; P&L accrues from t+1 onward. Costs per traded leg:
fixed fee (€1) + slippage as a half-spread in bps of leg notional.
Shorting is simulated — Trade Republic offers no shorting.
"""

import numpy as np
import pandas as pd

from tools.pairs_engine import (
    generate_signals,
    pair_zscore,
    select_pairs,
    walkforward_windows,
)


def simulate_pair(py: pd.Series, px: pd.Series, signal: pd.Series, beta: float,
                  pair_capital: float, slip_y_bps: float, slip_x_bps: float,
                  fee_eur: float = 1.0, cost_mult: float = 1.0,
                  z: pd.Series | None = None) -> dict:
    """Daily net P&L of one pair over one trading window, plus a trade ledger.

    Long spread (+1) = long Y, short X, beta-weighted notionals:
    N_y = pair_capital/(1+beta), N_x = beta*N_y.
    """
    n_y = pair_capital / (1.0 + beta)
    n_x = beta * n_y
    held = signal.shift(1).fillna(0.0)          # t+1 execution
    held.iloc[-1] = 0.0                         # force-close at window end
    r_y = py.pct_change().fillna(0.0)
    r_x = px.pct_change().fillna(0.0)
    gross = held.shift(1).fillna(0.0) * (n_y * r_y - n_x * r_x)
    turns = held.diff().abs().fillna(0.0)
    per_turn = (fee_eur + slip_y_bps / 1e4 * n_y) + (fee_eur + slip_x_bps / 1e4 * n_x)
    costs = turns * per_turn * cost_mult
    pnl = gross - costs

    trades, open_t, open_i = [], None, None
    hv = held.to_numpy()
    for i, d in enumerate(held.index):
        if hv[i] != 0 and (i == 0 or hv[i - 1] == 0):
            open_t = dict(entry=d, side=int(hv[i]), gross=0.0, costs=0.0,
                          z_entry=None if z is None else float(z.iloc[i - 1]))
            open_i = i
        if open_t is not None:
            open_t["gross"] += float(gross.iloc[i])
            open_t["costs"] += float(costs.iloc[i])
        if open_t is not None and hv[i] == 0 and i > 0 and hv[i - 1] != 0:
            open_t.update(exit=d, days=i - open_i,
                          net=open_t["gross"] - open_t["costs"])
            trades.append(open_t)
            open_t = None
    return {"pnl": pnl, "gross": gross, "costs": costs, "trades": trades}


def backtest_stats(equity, trades, capital):
    raise NotImplementedError


def run_backtest(*args, **kwargs):
    raise NotImplementedError
