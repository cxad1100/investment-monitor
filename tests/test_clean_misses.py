import pandas as pd

from tools.clean_misses import clean_name, resolve_misses


def test_clean_name_strips_broker_junk():
    assert clean_name("3 D SYS CORP.  DL-,001") == "3 D SYS CORP"
    assert clean_name("ACTIA GROUP SA INH.EO-,75") == "ACTIA GROUP SA"
    assert clean_name("ABB N") == "ABB"
    assert clean_name("A.P. MOELLER - MAERSK") == "A.P. MOELLER - MAERSK"   # untouched


def test_resolve_misses_uses_injected_resolver_and_slippage():
    rows = [
        {"Local_ID": "888346", "Name": "3 D SYS CORP.  DL-,001", "Country": "USA",
         "Sector": "Internet & Software", "Bid": 2.588, "Ask": 2.614},
        {"Local_ID": "000000", "Name": "UNRESOLVABLE NV", "Country": "—",
         "Sector": "Unknown", "Bid": 1.0, "Ask": 1.0},
    ]
    fake = {"3 D SYS CORP": "DDD", "UNRESOLVABLE NV": None}
    resolved = resolve_misses(rows, resolve_fn=lambda wkn, name, country: fake.get(name))
    assert len(resolved) == 1                              # only the one that resolved
    r = resolved[0]
    assert r["ticker"] == "DDD" and r["sector"] == "Internet & Software"
    assert 2 <= r["slippage_bps"] <= 50                    # from Bid/Ask half-spread
