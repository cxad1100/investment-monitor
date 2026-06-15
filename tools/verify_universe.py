"""Trust-but-verify the resolved universe before it ships.

Three checks against data/universe_meta.csv (resolved tickers) + data/universe.csv
(broker Bid/Ask):
  1. Ticker collision — many WKNs collapsing to one ticker = a fuzzy black hole.
  2. Price parity — Yahoo's last close vs the broker mid; a big ratio means the
     name fuzzy-matched the wrong company (broker EUR150 vs Yahoo EUR1.50).
  3. Zombie — ticker returns no price at all (delisted / dead). NOTE: under Yahoo
     rate-limiting a live ticker can also return nothing, so NO_PRICE is reported
     but NOT treated as proof of death.

Writes data/universe_price_flags.csv. Run when nothing else is hitting Yahoo.
"""

import time

import numpy as np
import pandas as pd
import yfinance as yf

from tools.build_universe import _parse_eur, ROOT

meta = pd.read_csv(ROOT / "data" / "universe_meta.csv", dtype={"local_id": str})
uni = pd.read_csv(ROOT / "data" / "universe.csv")
uni["mid"] = (uni["Bid"].map(_parse_eur) + uni["Ask"].map(_parse_eur)) / 2.0
mid = dict(zip(uni["Local_ID"].astype(str), uni["mid"]))

# 1. collision check
coll = meta["ticker"].value_counts()
print("=== ticker collisions (>3 WKNs -> one ticker = suspicious) ===")
print(coll[coll > 3].to_dict() or "none", " | max collision:", int(coll.max()))

# 2/3. price parity + zombie — chunked download (rate-limit friendly)
tickers = sorted(meta["ticker"].unique())
last: dict[str, float] = {}
for i in range(0, len(tickers), 200):
    ch = tickers[i : i + 200]
    try:
        raw = yf.download(ch, period="5d", auto_adjust=True, progress=False, threads=True)
        cl = raw["Close"] if "Close" in raw else raw
        if hasattr(cl, "columns"):
            for t in cl.columns:
                s = cl[t].dropna()
                if len(s):
                    last[t] = float(s.iloc[-1])
        else:
            s = cl.dropna()
            if len(s):
                last[ch[0]] = float(s.iloc[-1])
    except Exception as e:
        print("  chunk error", i, e)
    print(f"  priced {min(i + 200, len(tickers))}/{len(tickers)}  ({len(last)} have a price)")
    time.sleep(1)

rows = []
for r in meta.itertuples(index=False):
    bm = mid.get(str(r.local_id))
    yc = last.get(r.ticker)
    if yc is None:
        rows.append(dict(ticker=r.ticker, name=r.name, broker=bm, yahoo=None,
                         ratio=None, flag="NO_PRICE"))
    elif bm and bm > 0:
        ratio = yc / bm
        if ratio > 2 or ratio < 0.5:
            rows.append(dict(ticker=r.ticker, name=r.name, broker=round(bm, 2),
                             yahoo=round(yc, 2), ratio=round(ratio, 3), flag="DISPARITY"))

f = pd.DataFrame(rows)
f.to_csv(ROOT / "data" / "universe_price_flags.csv", index=False)
disp = f[f["flag"] == "DISPARITY"] if len(f) else f
noprice = f[f["flag"] == "NO_PRICE"] if len(f) else f
print(f"\n=== priced {len(last)}/{len(tickers)} | NO_PRICE {len(noprice)} (rate-limit ambiguous) "
      f"| PRICE_DISPARITY {len(disp)} (likely wrong mappings) ===")
if len(disp):
    worst = disp.reindex(disp["ratio"].sub(1).abs().sort_values(ascending=False).index)
    print(worst.head(30).to_string(index=False))
print("\nwrote data/universe_price_flags.csv")
