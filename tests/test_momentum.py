import numpy as np
import pandas as pd
import pytest

from tools.momentum import rebalance_dates


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
