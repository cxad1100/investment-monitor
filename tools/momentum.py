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
