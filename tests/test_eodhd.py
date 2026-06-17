import json

import pandas as pd

from tools.eodhd import fetch_eod, delisted_candidates


def test_fetch_eod_parses_adjusted_close():
    payload = json.dumps([
        {"date": "2019-01-02", "close": 100.0, "adjusted_close": 98.0, "volume": 1000},
        {"date": "2019-01-03", "close": 102.0, "adjusted_close": 99.9, "volume": 1100},
    ])
    s = fetch_eod("WDI.XETRA", key="x", get_fn=lambda url: payload)
    assert list(s.index) == [pd.Timestamp("2019-01-02"), pd.Timestamp("2019-01-03")]
    assert s.iloc[0] == 98.0 and s.iloc[-1] == 99.9


def test_fetch_eod_none_on_free_tier_warning():
    warn = json.dumps([{"warning": "Data is limited by one year as you have free subscription"}])
    assert fetch_eod("WDI.XETRA", key="x", get_fn=lambda url: warn) is None


def test_fetch_eod_none_on_empty():
    assert fetch_eod("X.Y", key="x", get_fn=lambda url: "[]") is None


def test_fetch_eod_none_on_persistent_error():
    def boom(url):
        raise OSError("connection reset")
    assert fetch_eod("X.Y", key="x", get_fn=boom, retries=0) is None   # blip → skip, don't abort


def test_delisted_candidates_keeps_domestic_common_stock_only():
    rows = [
        {"Code": "WDI", "Name": "Wirecard AG", "Isin": "DE0007472060",
         "Exchange": "XETRA", "Currency": "EUR", "Type": "Common Stock"},   # DE death — keep
        {"Code": "0IIA", "Name": "GPF Physical Gold ETC", "Isin": "XS2265368097",
         "Exchange": "XETRA", "Currency": "EUR", "Type": "ETF"},            # ETC — drop
        {"Code": "02M", "Name": "The Mosaic Company", "Isin": "US61945C1036",
         "Exchange": "XETRA", "Currency": "EUR", "Type": "Common Stock"},   # foreign withdrawal — drop
    ]
    cands = delisted_candidates(rows)
    assert len(cands) == 1
    c = cands[0]
    assert c["ticker"] == "WDI.XETRA" and c["name"] == "Wirecard AG"
    assert "removal_date" in c and "spread_pct" in c
