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
        dead["delisting_date"] = pd.to_datetime(dead["delisting_date"], format="mixed")
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


def fetch_survivors(tickers: list[str], years: int = 9, chunk: int = 200) -> pd.DataFrame:
    """Robust chunked yfinance pull of adjusted closes. Keeps only the columns
    yfinance actually returns (many .F lines won't resolve) — never KeyErrors on a
    missing ticker, unlike pairs_universe.fetch_prices."""
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), chunk):
        raw = yf.download(tickers[i:i + chunk], period=f"{years}y",
                          auto_adjust=True, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) and "Close" \
            in raw.columns.get_level_values(0) else raw
        frames.append(close)
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out.dropna(how="all").ffill()


def main(refresh: bool = False, include_misses: bool = False):
    from tools.pairs_universe import _load_universe
    from tools.dead_stocks import load_dead_prices, OUT_META as DEAD_META

    surv_meta = pd.DataFrame([{"ticker": t, **m} for t, m in _load_universe().items()])
    misses = []
    if include_misses:
        from tools.clean_misses import load_misses, resolve_misses
        misses = resolve_misses(load_misses())
        print(f"resolved misses: {len(misses)}")

    dead_prices, _ = load_dead_prices()
    dead_meta = pd.read_csv(DEAD_META) if DEAD_META.exists() else pd.DataFrame(columns=["ticker"])

    meta = assemble_meta(surv_meta, misses, dead_meta)
    alive = meta[meta["delisting_date"].isna()]["ticker"].tolist()
    print(f"fetching {len(alive)} survivor histories (9y, yfinance, chunked)…")
    surv_prices = fetch_survivors(alive)
    prices = assemble_prices([surv_prices, dead_prices])

    prices.to_csv(OUT_PRICES)
    meta.to_csv(OUT_META, index=False)
    print(f"assembled {meta.shape[0]} names ({dead_meta.shape[0]} dead, "
          f"{surv_prices.shape[1]} live priced) · prices {prices.shape} "
          f"{prices.index[0].date()}→{prices.index[-1].date()}")


if __name__ == "__main__":
    import sys
    main(refresh="--refresh" in sys.argv, include_misses="--misses" in sys.argv)
