import numpy as np
import pandas as pd

from tools.dead_stocks import parse_index_changes, classify_dead, keep_real


def test_parse_index_changes_keeps_removals_only():
    rows = [
        {"date": "2020-08-24", "action": "removed", "ticker": "WDI.DE", "name": "Wirecard"},
        {"date": "2021-03-22", "action": "added",   "ticker": "ZAL.DE", "name": "Zalando"},
        {"date": "2022-01-24", "action": "deleted", "ticker": "XYZ.DE", "name": "Xyz"},
    ]
    out = parse_index_changes(rows)
    assert [r["ticker"] for r in out] == ["WDI.DE", "XYZ.DE"]           # removals only
    assert out[0]["removal_date"] == pd.Timestamp("2020-08-24")


def test_classify_dead_when_prices_stop():
    idx = pd.bdate_range("2019-01-01", periods=400)
    s = pd.Series(list(np.linspace(50, 2, 200)) + [np.nan] * 200, index=idx)  # stops mid
    today = idx[-1]
    dl = classify_dead(s, removal_date=idx[195], today=today, gap_days=20)
    assert dl == idx[199]                                  # last real bar = delisting


def test_classify_dead_rejects_still_trading():
    idx = pd.bdate_range("2019-01-01", periods=400)
    s = pd.Series(np.linspace(50, 80, 400), index=idx)     # never stops
    assert classify_dead(s, removal_date=idx[100], today=idx[-1]) is None


def test_keep_real_filters_penny_and_wide_spread():
    cands = [
        {"ticker": "WDI.DE", "last_price": 1.50, "spread_pct": 0.30},     # ok
        {"ticker": "PENNY",  "last_price": 0.40, "spread_pct": 0.30},     # < €1 -> drop
        {"ticker": "WIDE",   "last_price": 5.00, "spread_pct": 2.50},     # > 1.5% -> drop
        {"ticker": "NOPX",   "last_price": None, "spread_pct": 0.30},     # missing -> drop
    ]
    kept = {c["ticker"] for c in keep_real(cands, min_price=1.0, max_spread_pct=1.5)}
    assert kept == {"WDI.DE"}
