"""Re-resolve the price-disparity-flagged tickers to the German listing whose
price actually matches the broker mid.

Yahoo search returns several German listings per company; _best_quote picks by
exchange rank, which sometimes lands a thin '0'-suffix line with stale/wrong data
(Broadcom 1YD0.F EUR8.80 instead of 1YD.F EUR337). Score and volume don't
distinguish them — but price does: the right listing's last close ~= the broker
mid. For each flagged WKN, gather the German EQUITY candidates, price each, and
keep the one with ratio closest to 1 (within [0.5, 2]). Updates cache + meta.

Skips the ~1000x cluster (ratio > 500) — those are correct mappings where the
broker scrape mis-scales the price, not the ticker.
"""

import json
import time

import numpy as np
import pandas as pd
import yfinance as yf

from tools.build_universe import ROOT, _parse_eur
from tools.wkn_resolve import _EXCH_RANK, _name_variants, _names_match, _yahoo_search

flags = pd.read_csv(ROOT / "data" / "universe_price_flags.csv")
fix = flags[(flags["flag"] == "DISPARITY") & (flags["ratio"] < 500)]   # broker mid is the reliable side
uni = pd.read_csv(ROOT / "data" / "universe.csv")
uni["mid"] = (uni["Bid"].map(_parse_eur) + uni["Ask"].map(_parse_eur)) / 2.0
midmap = dict(zip(uni["Local_ID"].astype(str), uni["mid"]))
namemap = dict(zip(uni["Local_ID"].astype(str), uni["Name"]))
meta = pd.read_csv(ROOT / "data" / "universe_meta.csv", dtype={"local_id": str})
tk2wkn = dict(zip(meta["ticker"], meta["local_id"]))


def _price(sym):
    try:
        s = yf.Ticker(sym).history(period="5d")["Close"].dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:
        return None


fixes, unresolved = {}, []
for _, row in fix.iterrows():
    tk = row["ticker"]
    wkn = str(tk2wkn.get(tk, ""))
    bm = midmap.get(wkn)
    nm = namemap.get(wkn)
    if not wkn or not bm or bm <= 0 or not isinstance(nm, str):
        continue
    cands = {}
    for v in _name_variants(nm):
        for q in (_yahoo_search(v) or []):
            if (q.get("quoteType") == "EQUITY" and q.get("exchange") in _EXCH_RANK
                    and _names_match(nm, q)):
                cands.setdefault(q["symbol"], q)
        time.sleep(0.3)
    best, best_d = None, 9e9
    for sym in cands:
        p = _price(sym)
        time.sleep(0.2)
        if not p:
            continue
        r = p / bm
        if 0.5 <= r <= 2 and abs(np.log(r)) < best_d:
            best_d, best = abs(np.log(r)), sym
    if best and best != tk:
        fixes[wkn] = best
    elif not best:
        unresolved.append((tk, nm[:30]))
    print(f"  {nm[:26]:26s} {tk:9s} (br {bm:8.2f}) -> {best or 'NO MATCH'}")

print(f"\nfixes: {len(fixes)}   no-match (left as-is): {len(unresolved)}")
cache = json.loads((ROOT / "data" / "wkn_ticker_map.json").read_text())
for wkn, sym in fixes.items():
    cache[wkn] = sym
(ROOT / "data" / "wkn_ticker_map.json").write_text(json.dumps(cache))
meta["ticker"] = meta["local_id"].astype(str).map(lambda w: fixes.get(w)).fillna(meta["ticker"])
meta = meta.drop_duplicates("ticker")
meta.to_csv(ROOT / "data" / "universe_meta.csv", index=False)
print(f"updated meta: {len(meta)} rows")
