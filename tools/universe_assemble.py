"""Assemble the survivorship-corrected momentum dataset: survivors ∪ resolved
misses ∪ dead EUR names → one 9y price frame (dead columns truncated at death,
NOT forward-filled past it) + one meta with a `delisting_date` column (NaT for
survivors). This is what the momentum engine + PITUniverse consume.

The CLI `main()` (wiring the live survivor fetch + FMP dead-price ingest) is added
once the dead-price source lands; the pure assembly below is unit-tested now.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_PRICES = ROOT / "data" / "momentum_prices.csv"
OUT_META = ROOT / "data" / "momentum_meta.csv"

_META_COLS = ["ticker", "name", "sector", "country", "currency",
              "slippage_bps", "local_id", "delisting_date"]


def assemble_meta(survivors: pd.DataFrame, misses: list[dict],
                  dead: pd.DataFrame) -> pd.DataFrame:
    """Union the three sources into one meta with a delisting_date column."""
    surv = survivors.copy()
    surv["delisting_date"] = pd.NaT
    miss = pd.DataFrame(misses)
    if not miss.empty:
        miss["delisting_date"] = pd.NaT
    dead = dead.copy()
    if not dead.empty:
        dead["delisting_date"] = pd.to_datetime(dead["delisting_date"])
    frames = []
    for df in (surv, miss, dead):
        if df.empty:
            continue
        for c in _META_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[_META_COLS])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def assemble_prices(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join price frames on the date index. Each frame's columns keep their
    own NaN tails (dead names already truncated at death by the dead-stock step),
    so no cross-name forward-fill resurrects a delisted line."""
    out = pd.concat(frames, axis=1).sort_index()
    return out.loc[:, ~out.columns.duplicated()]


def delisting_map(meta: pd.DataFrame) -> dict:
    """meta → {ticker: Timestamp} for the dead names only (PITUniverse input)."""
    d = meta.dropna(subset=["delisting_date"])
    return {r.ticker: pd.Timestamp(r.delisting_date) for r in d.itertuples(index=False)}
