import numpy as np
import pandas as pd

from tools.universe_assemble import assemble_meta, assemble_prices, delisting_map
from tools.universe_pit import PITUniverse


def test_assemble_meta_unions_and_marks_delisting():
    survivors = pd.DataFrame([
        {"ticker": "BMW.DE", "name": "BMW", "sector": "Vehicles", "country": "DE",
         "currency": "EUR", "slippage_bps": 10, "local_id": "519000"},
    ])
    misses = [{"ticker": "DDD", "name": "3 D SYS", "sector": "Internet & Software",
               "country": "USA", "currency": "EUR", "slippage_bps": 12, "local_id": "888346"}]
    dead = pd.DataFrame([{"ticker": "WDI.DE", "name": "Wirecard",
                          "sector": "Internet & Software", "delisting_date": "2020-08-21",
                          "slippage_bps": 15}])
    meta = assemble_meta(survivors, misses, dead)
    assert set(meta["ticker"]) == {"BMW.DE", "DDD", "WDI.DE"}
    row = meta.set_index("ticker")
    assert pd.isna(row.loc["BMW.DE", "delisting_date"])           # survivor
    assert pd.notna(row.loc["WDI.DE", "delisting_date"])          # dead
    assert row.loc["DDD", "sector"] == "Internet & Software"


def test_delisting_map_only_dead():
    meta = pd.DataFrame([
        {"ticker": "BMW.DE", "delisting_date": pd.NaT},
        {"ticker": "WDI.DE", "delisting_date": pd.Timestamp("2020-08-21")},
    ])
    dm = delisting_map(meta)
    assert dm == {"WDI.DE": pd.Timestamp("2020-08-21")}


def test_assemble_prices_merges_and_pit_sees_death():
    idx = pd.bdate_range("2019-01-01", periods=300)
    live = pd.DataFrame({"BMW.DE": np.linspace(60, 90, 300)}, index=idx)
    dead = pd.DataFrame({"WDI.DE": list(np.linspace(100, 2, 150))}, index=idx[:150])
    prices = assemble_prices([live, dead])
    assert "WDI.DE" in prices.columns and "BMW.DE" in prices.columns
    assert prices["WDI.DE"].dropna().index[-1] == idx[149]        # not ffilled past death
    pit = PITUniverse(prices, delisting={"WDI.DE": idx[149]})
    assert "WDI.DE" in pit.tradeable(idx[100], min_history_days=50)
    assert "WDI.DE" not in pit.tradeable(idx[200], min_history_days=50)
