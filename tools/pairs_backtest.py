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


def backtest_stats(equity: pd.Series, trades: list[dict], capital: float) -> dict:
    rets = equity.pct_change().dropna()
    sd = float(rets.std())
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min())
    wins = sum(1 for t in trades if t["net"] > 0)
    return dict(
        net_return=float(equity.iloc[-1] / capital - 1.0),
        sharpe=sharpe,
        max_drawdown=dd,
        n_trades=len(trades),
        win_rate=wins / len(trades) if trades else None,
        avg_days=float(np.mean([t["days"] for t in trades])) if trades else None,
        total_costs=float(sum(t["costs"] for t in trades)),
    )


def run_backtest(prices: pd.DataFrame, candidates: list[tuple[str, str]],
                 slippage_bps: dict, capital: float = 10_000.0,
                 formation_days: int = 252, trading_days: int = 63,
                 p_max: float = 0.05, top_n: int = 10, entry: float = 2.0,
                 stop: float = 3.5, fee_eur: float = 1.0,
                 cost_mults: tuple = (0.0, 1.0, 2.0)) -> dict:
    """Walk-forward backtest. Pair selection and signals are computed once
    (they don't depend on costs); each cost multiple re-prices the same
    trades — that's the cost-sensitivity table.
    """
    windows = walkforward_windows(prices.index, formation_days, trading_days)
    legs, window_log = [], []
    for form_idx, trade_idx in windows:
        sel = select_pairs(prices.loc[form_idx], candidates,
                           p_max=p_max, top_n=top_n)
        pairs = sel["selected"]
        window_log.append(dict(formation_end=form_idx[-1], trade_start=trade_idx[0],
                               n_tested=sel["n_tested"], n_selected=len(pairs)))
        if not pairs:
            continue
        slice_cap = capital / len(pairs)
        for pr in pairs:
            py = prices.loc[trade_idx, pr["y"]]
            px = prices.loc[trade_idx, pr["x"]]
            if py.isna().any() or px.isna().any():
                continue
            z = pair_zscore(py, px, pr)
            sig = generate_signals(z, entry=entry, stop=stop)
            legs.append((trade_idx, pr, py, px, sig, z, slice_cap))

    runs = {}
    for m in cost_mults:
        daily = pd.Series(0.0, index=prices.index)
        trades = []
        for trade_idx, pr, py, px, sig, z, cap in legs:
            res = simulate_pair(py, px, sig, pr["beta"], cap,
                                slippage_bps[pr["y"]], slippage_bps[pr["x"]],
                                fee_eur=fee_eur, cost_mult=m, z=z)
            daily.loc[trade_idx] = daily.loc[trade_idx] + res["pnl"]
            for t in res["trades"]:
                trades.append({**t, "pair": f'{pr["y"]}/{pr["x"]}', "capital": cap})
        equity = capital + daily.cumsum()
        runs[m] = dict(equity=equity, trades=trades,
                       stats=backtest_stats(equity, trades, capital))
    return {"runs": runs, "windows": window_log,
            "start": str(windows[0][1][0].date()) if windows else None}
