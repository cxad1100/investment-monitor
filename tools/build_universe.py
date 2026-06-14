"""Build the tradeable, sector-tagged pairs universe from the scraped broker list.

Input: `data/universe.csv` — every stock on the broker (Lang & Schwarz /
Trade Republic): Local_ID (WKN), Name, Country, Sector, live Bid/Ask, Perf_1Y.
This file is the complete registry; the tradeable universe is every asset that
resolves to a German EUR listing with price history (sector-optional).

Pipeline (offline, slow, network-bound — run once, cached):
  1. Parse Bid/Ask → half-spread → slippage_bps; drop untradeably-wide spreads
     and sub-€1 penny listings. Sector-optional (N/A sector kept as "Unknown").
  2. Resolve WKN → yfinance ticker (tools.wkn_resolve, cached to
     data/wkn_ticker_map.json): German via deterministic ISIN → Yahoo search,
     foreign by name. Skip unresolved.
  3. Verify each ticker returns recent history (chunked yf.download); drop dead.

Output: `data/universe_meta.csv`
  (ticker, local_id, name, country, sector, currency, slippage_bps)
which `tools/pairs_universe.py` loads as UNIVERSE. No global top-N cap — the pairs
engine bounds compute by pairing only within sector / country+sector groups.

Run:  .venv/bin/python -m tools.build_universe            # all tagged names
      .venv/bin/python -m tools.build_universe --limit 200  # debug subset
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from tools.wkn_resolve import resolve_ticker, spread_to_slippage, _load_cache, _save_cache

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "universe.csv"
OUT = ROOT / "data" / "universe_meta.csv"

MAX_SPREAD_PCT = 1.5       # drop names wider than this full bid/ask spread (untradeable)
MIN_PRICE = 1.0            # drop sub-€1 penny listings (price/momentum = tick noise)
CHUNK = 50                 # tickers per history-verify batch
CHUNK_PAUSE = 1.0
MAX_RETRY = 4
BACKOFF = 20


def _parse_eur(s) -> float:
    """Parse a broker price like '40.4000 €' (or German '1.234,56 €') to float."""
    t = str(s).replace("€", "").replace("\xa0", "").strip()
    if "," in t and "." in t:          # German thousands+decimal: 1.234,56
        t = t.replace(".", "").replace(",", ".")
    elif "," in t:                     # comma decimal: 24,30
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _has_history(tickers: list[str]) -> set[str]:
    """Tickers that return any recent daily data (drops dead/delisted lines)."""
    live: set[str] = set()
    n = len(tickers)
    for i in range(0, n, CHUNK):
        chunk = tickers[i : i + CHUNK]
        for attempt in range(MAX_RETRY):
            try:
                df = yf.download(chunk, period="1mo", auto_adjust=True,
                                 progress=False, threads=True)
                break
            except Exception as e:
                if ("too many" in str(e).lower() or "rate" in str(e).lower()) \
                        and attempt < MAX_RETRY - 1:
                    time.sleep(BACKOFF * (2 ** attempt))
                    continue
                df = None
                break
        if df is not None and not df.empty:
            close = df["Close"] if "Close" in df else df
            if isinstance(close, pd.Series):
                if close.notna().any():
                    live.add(chunk[0])
            else:
                for t in close.columns:
                    if close[t].notna().any():
                        live.add(t)
        print(f"  history {min(i+CHUNK, n)}/{n}  ({len(live)} live)")
        time.sleep(CHUNK_PAUSE)
    return live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="cap tagged rows processed (debug; 0 = all)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the history-verify stage (faster, keeps dead lines)")
    args = ap.parse_args()

    df = pd.read_csv(SRC).copy()
    # Sector-optional: keep every liquid, priced asset (momentum + future strats
    # don't need a sector; pairs still pairs only the sector-tagged ones).
    df["Country"] = df["Country"].fillna("—").replace({"N/A": "—"})
    df["Sector"] = df["Sector"].fillna("Unknown").replace({"N/A": "Unknown"})
    df["bid"] = df["Bid"].map(_parse_eur)
    df["ask"] = df["Ask"].map(_parse_eur)
    mid = (df["bid"] + df["ask"]) / 2.0
    df["spread_pct"] = (df["ask"] - df["bid"]) / mid * 100.0
    df = df[(mid >= MIN_PRICE) & (df["spread_pct"] >= 0) & (df["spread_pct"] <= MAX_SPREAD_PCT)]
    df = df.sort_values("spread_pct")            # tightest (most liquid) first
    if args.limit:
        df = df.head(args.limit)
    print(f"liquid, priced rows (all sectors): {len(df)}")

    print("resolving WKN → ticker (cached)…")
    cache = _load_cache()
    rows = []
    for j, r in enumerate(df.itertuples(index=False), 1):
        sym = resolve_ticker(r.Local_ID, r.Name, r.Country, cache=cache)
        if sym:
            rows.append(dict(ticker=sym, local_id=r.Local_ID, name=r.Name,
                             country=r.Country, sector=r.Sector, currency="EUR",
                             slippage_bps=spread_to_slippage(r.bid, r.ask)))
        if j % 50 == 0:
            _save_cache(cache)
            print(f"  resolved {j}/{len(df)}  ({len(rows)} hits)")
    _save_cache(cache)
    meta = pd.DataFrame(rows).drop_duplicates("ticker")
    print(f"  {len(meta)} unique tickers resolved")

    if not args.no_verify and len(meta):
        print("verifying history…")
        live = _has_history(list(meta["ticker"]))
        meta = meta[meta["ticker"].isin(live)]
        print(f"  {len(meta)} with live history")

    OUT.parent.mkdir(exist_ok=True)
    meta.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(meta)} rows)")
    if len(meta):
        print(meta.groupby(["country", "sector"]).size()
              .sort_values(ascending=False).head(12))


if __name__ == "__main__":
    main()
