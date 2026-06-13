"""Long-only cross-sectional momentum engine (pure functions, no I/O).

Strategy: at each monthly rebalance, rank every eligible name by 12-1 momentum
(trailing ~12 months skipping the most recent ~1 month), hold an equal-weight
global top-k until the next rebalance. Walk-forward, no look-ahead: ranks use
only data with index <= the rebalance date; returns accrue strictly after.

Holdings are computed once and are independent of trading costs; each cost
multiple re-prices the identical schedule (the cost-sensitivity table).
"""

import numpy as np
import pandas as pd

from tools.pairs_backtest import backtest_stats


def rebalance_dates(index, freq: str = "M") -> list[pd.Timestamp]:
    """Last trading day present in the index for each period (default month)."""
    idx = pd.DatetimeIndex(index)
    # Map deprecated "M" to "ME" for pandas compatibility
    actual_freq = "ME" if freq == "M" else freq
    last = pd.Series(idx, index=idx).resample(actual_freq).last().dropna()
    return list(last)


def momentum_scores(prices: pd.DataFrame, asof, lookback: int = 252,
                    skip: int = 21) -> pd.Series:
    """12-1 momentum per ticker: price(asof-skip) / price(asof-lookback) - 1.

    Uses only rows with index <= asof (no look-ahead). Returns an empty Series
    when there is not yet `lookback`+1 rows of history. inf/NaN dropped.
    """
    hist = prices.loc[:asof]
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    recent = hist.iloc[-(skip + 1)]              # ~skip days before asof
    base = hist.iloc[-(lookback + 1)]            # ~lookback days before asof
    scores = recent / base - 1.0
    return scores.replace([np.inf, -np.inf], np.nan).dropna()


def eligible(prices: pd.DataFrame, asof, slippage_bps: dict,
             liq_max: int = 30, min_obs: int = 273) -> set[str]:
    """Tradeable names at `asof`: tight half-spread, enough history, valid price."""
    hist = prices.loc[:asof]
    out: set[str] = set()
    for t in prices.columns:
        if slippage_bps.get(t, 10**9) > liq_max:
            continue
        col = hist[t].dropna()
        if len(col) >= min_obs and float(col.iloc[-1]) > 0:
            out.add(t)
    return out


def select_topk(scores: pd.Series, eligible_set: set[str], k: int) -> list[str]:
    """Top-k tickers by score, restricted to the eligible set, highest first."""
    s = scores[[t for t in scores.index if t in eligible_set]]
    return list(s.sort_values(ascending=False).head(k).index)


def run_momentum(prices: pd.DataFrame, slippage_bps: dict, *, k: int = 15,
                 lookback: int = 252, skip: int = 21, capital: float = 10_000.0,
                 cost_mults: tuple = (0.0, 1.0, 2.0), freq: str = "M",
                 liq_max: int = 30, fee_eur: float = 1.0) -> dict:
    """Walk-forward momentum backtest.

    Returns {"runs": {mult: {equity, trades, stats}}, "holdings_log": [...],
             "start": iso}. The schedule (holdings_log) is cost-independent;
    each cost multiple compounds the same equal-weight daily returns minus a
    rebalance cost drag.
    """
    dates = [d for d in rebalance_dates(prices.index, freq)
             if len(prices.loc[:d]) >= lookback + 1]
    holdings_log = []
    for i in range(len(dates) - 1):
        d = dates[i]
        scores = momentum_scores(prices, d, lookback, skip)
        elig = eligible(prices, d, slippage_bps, liq_max, lookback + skip)
        picks = select_topk(scores, elig, k)
        holdings_log.append(dict(date=d, next=dates[i + 1], picks=picks,
                                 scores={t: float(scores[t]) for t in picks}))

    runs = {}
    for mult in cost_mults:
        equity_val = capital
        eq_points = [(holdings_log[0]["date"], capital)] if holdings_log \
            else [(prices.index[0], capital)]
        trades = []
        prev: set[str] = set()
        for h in holdings_log:
            d, nxt, picks = h["date"], h["next"], h["picks"]
            if not picks:
                if prev:                                # liquidate to cash: charge exit
                    equity_val -= sum(fee_eur + slippage_bps[t] / 1e4 * (equity_val / len(prev))
                                      for t in prev) * mult
                prev = set()
                continue
            w = equity_val / len(picks)                 # equal notional per name
            traded = (set(picks) ^ prev)                # enters + exits
            cost = sum(fee_eur + slippage_bps[t] / 1e4 * w
                       for t in traded) * mult
            equity_val -= cost
            seg = prices.loc[d:nxt, picks]
            rets = seg.pct_change().iloc[1:].fillna(0.0)
            port_ret = rets.mean(axis=1)                # equal-weight daily return
            for day, r in port_ret.items():
                equity_val *= (1.0 + r)
                eq_points.append((day, equity_val))
            for t in picks:
                name_ret = float(seg[t].iloc[-1] / seg[t].iloc[0] - 1.0)
                c = (fee_eur + slippage_bps[t] / 1e4 * w) * mult \
                    if t in traded else 0.0
                trades.append(dict(pair=t, entry=d, exit=nxt, days=len(seg) - 1,
                                   gross=w * name_ret, costs=c,
                                   net=w * name_ret - c, capital=w))
            prev = set(picks)
        equity = pd.Series(dict(eq_points)).sort_index()
        runs[mult] = dict(equity=equity, trades=trades,
                       stats=backtest_stats(equity, trades, capital))
    return {"runs": runs, "holdings_log": holdings_log,
            "start": str(dates[0].date()) if dates else None}


def benchmark_curves(bench_prices: pd.DataFrame, window, capital: float) -> dict:
    """Buy-hold equity per benchmark over `window`, each normalized to `capital`
    at the window start. Missing benchmarks (all-NaN) are skipped."""
    out = {}
    for name in bench_prices.columns:
        s = bench_prices[name].reindex(window).ffill().dropna()
        if s.empty or float(s.iloc[0]) <= 0:
            continue
        out[name] = capital * s / float(s.iloc[0])
    return out


def equal_weight_curve(prices: pd.DataFrame, tickers: list[str], window,
                       capital: float) -> pd.Series:
    """Equity of an equal-weight buy-hold basket of `tickers` over `window`,
    normalized to `capital` at the start — the 'hold everything equally'
    honesty benchmark."""
    sub = prices.reindex(window)[tickers].ffill().dropna(how="all")
    norm = sub / sub.iloc[0]                          # each name -> 1.0 at start
    port = norm.mean(axis=1)                          # equal weight
    return capital * port
