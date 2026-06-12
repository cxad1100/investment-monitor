"""Build a tradeable, sector-tagged pairs universe from the raw LS/Frankfurt list.

The raw list (`data/universe_raw.csv`, ~15k names from Trade Republic / LS Exchange,
all `.F` Frankfurt-listed) is far too large to cointegration-test (C(15k,2) ≈ 119M
pairs) and most `.F` lines are illiquid secondary listings with ~0 volume. This
script narrows it to the top-N most liquid names and tags each with a real sector,
so the pairs engine can bucket candidates by sector+currency.

Two-stage, cheap-first (avoids 15k slow `.info` calls):
  1. Batch `yf.download` price+volume in chunks → median daily € turnover → rank.
  2. `.info` ONLY for the top-N survivors → sector / currency / display name.

Output: `data/universe_meta.csv`  (ticker,name,type,sector,currency,turnover_eur,slippage_bps)
which `tools/pairs_universe.py` loads as UNIVERSE.

Run (offline, slow ~10-15 min, network-bound):
    .venv/bin/python -m tools.build_universe --top 300
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Yahoo spews 404/delisted/timeout noise for the many dead .F lines — silence it.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "universe_raw.csv"
OUT = ROOT / "data" / "universe_meta.csv"

LIQUIDITY_PERIOD = "3mo"   # window for ranking turnover
CHUNK = 50                 # tickers per yf.download batch (small = rate-limit-safe)
CHUNK_PAUSE = 1.0          # seconds between chunks (throttle)
MAX_RETRY = 4              # per-chunk retries on rate-limit
BACKOFF = 20               # base seconds for exponential backoff


def _slippage_bps(turnover_eur: float) -> int:
    """Assumed half-spread per leg, tiered by daily € turnover."""
    if turnover_eur >= 50e6:
        return 5
    if turnover_eur >= 5e6:
        return 10
    if turnover_eur >= 1e6:
        return 15
    return 25


def _download_chunk(chunk: list[str]) -> pd.DataFrame | None:
    """yf.download with retry+exponential backoff on Yahoo rate limiting."""
    for attempt in range(MAX_RETRY):
        try:
            df = yf.download(chunk, period=LIQUIDITY_PERIOD, auto_adjust=True,
                             progress=False, threads=True)
            if df is not None and not df.empty:
                return df
            return None
        except Exception as e:
            msg = str(e).lower()
            if "too many" in msg or "rate" in msg:
                wait = BACKOFF * (2 ** attempt)
                print(f"    rate-limited; backing off {wait}s "
                      f"(attempt {attempt+1}/{MAX_RETRY})")
                time.sleep(wait)
                continue
            print(f"    chunk download failed ({e}); skipping")
            return None
    print("    gave up on chunk after retries")
    return None


def rank_by_liquidity(tickers: list[str]) -> pd.Series:
    """Median daily € turnover (Close × Volume) per ticker, descending.

    Dead/illiquid `.F` lines come back NaN or 0 and sink to the bottom.
    Small chunks + throttle + backoff keep Yahoo from rate-limiting the scan.
    """
    turnover: dict[str, float] = {}
    n = len(tickers)
    for i in range(0, n, CHUNK):
        chunk = tickers[i : i + CHUNK]
        df = _download_chunk(chunk)
        if df is not None:
            # single-ticker chunks come back without the outer column level
            close = df["Close"] if "Close" in df else df
            vol = df["Volume"] if "Volume" in df else None
            if vol is not None:
                if isinstance(close, pd.Series):    # 1 surviving ticker
                    close, vol = close.to_frame(chunk[0]), vol.to_frame(chunk[0])
                med = (close * vol).median()        # € turnover per ticker
                for t, v in med.items():
                    turnover[t] = float(v) if pd.notna(v) else 0.0
        live = sum(1 for v in turnover.values() if v > 0)
        print(f"  liquidity {min(i+CHUNK, n)}/{n}  ({live} live so far)")
        time.sleep(CHUNK_PAUSE)
    s = pd.Series(turnover, name="turnover_eur")
    return s[s > 0].sort_values(ascending=False)


def _map_sector(info: dict, kind: str) -> str:
    """yfinance sector for stocks; category for ETFs (so ETFs bucket too)."""
    if kind == "etf":
        return (info.get("category") or "ETF — Uncategorized").strip()
    return (info.get("sector") or "Unknown").strip()


def enrich(top: pd.Series, names: dict[str, str], kinds: dict[str, str]
           ) -> pd.DataFrame:
    """Fetch sector/currency for the survivors only (one `.info` each)."""
    rows = []
    n = len(top)
    for j, (t, turn) in enumerate(top.items(), 1):
        sector, currency = "Unknown", "EUR"
        info = {}
        for attempt in range(MAX_RETRY):
            try:
                info = yf.Ticker(t).info
                kind = kinds.get(t, "stock")
                sector = _map_sector(info, kind)
                currency = (info.get("currency") or "EUR").upper()
                break
            except Exception as e:
                if ("too many" in str(e).lower() or "rate" in str(e).lower()) \
                        and attempt < MAX_RETRY - 1:
                    time.sleep(BACKOFF * (2 ** attempt))
                    continue
                print(f"  info {j}/{n} {t}: failed ({e})")
                break
        disp = names.get(t) or info.get("longName") or info.get("shortName") or t
        rows.append(dict(ticker=t, name=disp, type=kinds.get(t, "stock"),
                         sector=sector, currency=currency,
                         turnover_eur=round(float(turn)),
                         slippage_bps=_slippage_bps(float(turn))))
        if j % 25 == 0:
            print(f"  enriched {j}/{n}")
        time.sleep(0.3)  # be gentle on Yahoo
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300,
                    help="keep the N most liquid names (default 300)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap raw tickers scanned (debug; 0 = all)")
    args = ap.parse_args()

    raw = pd.read_csv(RAW)
    if args.limit:
        raw = raw.head(args.limit)
    names = dict(zip(raw["yf_ticker"], raw["name"]))
    kinds = dict(zip(raw["yf_ticker"], raw["type"]))
    tickers = list(raw["yf_ticker"].dropna().unique())
    print(f"raw universe: {len(tickers)} tickers")

    print("stage 1 — ranking by liquidity…")
    ranked = rank_by_liquidity(tickers)
    print(f"  {len(ranked)} tickers with positive turnover")
    top = ranked.head(args.top)
    print(f"  keeping top {len(top)}")

    print("stage 2 — fetching sector/currency for survivors…")
    meta = enrich(top, names, kinds)
    meta = meta.sort_values("turnover_eur", ascending=False)
    OUT.parent.mkdir(exist_ok=True)
    meta.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(meta)} rows)")
    print(meta.groupby("sector").size().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
