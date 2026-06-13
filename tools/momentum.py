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
