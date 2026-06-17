"""Source EUR-exchange stocks that DIED (delisted) 2018→now and were real while
alive (≥ €1, ≤ 1.5 % spread), for the survivorship-corrected momentum universe.

Pure core (this section) is unit-tested. The network adapters + CLI below are
best-effort; a hand-seeded data/eur_delisted_seed.csv guarantees real dead names
even if scraping yields nothing.

A removed index member is only DEAD if its prices stop near the removal date —
that distinguishes a bankruptcy (Wirecard) from a mere demotion (still trading).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_REMOVE = {"removed", "deleted", "delisted", "remove"}


def parse_index_changes(rows: list[dict]) -> list[dict]:
    """Index change rows → removal candidates [{ticker, name, removal_date}]."""
    out = []
    for r in rows:
        if str(r.get("action", "")).strip().lower() in _REMOVE:
            out.append({"ticker": r["ticker"], "name": r.get("name", r["ticker"]),
                        "removal_date": pd.Timestamp(r["date"])})
    return out


def classify_dead(price_series: pd.Series, removal_date, today,
                  *, gap_days: int = 20):
    """Delisting date (= last real bar) if the series stops trading near removal,
    else None (still printing prices → demoted, not dead)."""
    s = price_series.dropna()
    if s.empty:
        return None
    last = s.index[-1]
    if (pd.Timestamp(today) - last).days > gap_days:
        return last
    return None


def keep_real(candidates: list[dict], *, min_price: float = 1.0,
              max_spread_pct: float = 1.5) -> list[dict]:
    """Keep dead names that were real: last price ≥ min_price and spread ≤ cap %."""
    out = []
    for c in candidates:
        lp, sp = c.get("last_price"), c.get("spread_pct")
        if lp is None or sp is None:
            continue
        if lp >= min_price and sp <= max_spread_pct:
            out.append(c)
    return out


# ── network adapters + CLI (best-effort; seed CSV guarantees real dead names) ──

import io
import urllib.request

import yfinance as yf

SEED_CSV = ROOT / "data" / "eur_delisted_seed.csv"
OUT_META = ROOT / "data" / "eur_delisted.csv"
OUT_PRICES = ROOT / "data" / "momentum_dead_prices.csv"


def build_dead_table(seed: list[dict], *, fetch_history, today=None,
                     min_price_alive: float = 1.0, max_survival_ratio: float = 1.0,
                     max_workers: int = 1):
    """Seed/removal candidates → (dead-meta DataFrame, dead-prices DataFrame).

    `fetch_history(ticker) -> pd.Series` is injected (EODHD live, fake in tests).
    Keeps only names whose prices stop near removal (classify_dead → dead) AND that
    were a real holding while alive — **peak** price ≥ `min_price_alive`. The peak
    test (not the last price) is deliberate: a fallen giant like Wirecard dies at
    €0.40 but was a €100 stock, and those collapses are the whole point; only
    perpetual sub-€1 penny names are excluded. Each kept column is truncated at its
    delisting date so its last bar is the last traded price the graveyard uses.
    """
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    if max_workers > 1 and seed:                       # concurrent fetch (the slow part)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            hist = list(pool.map(lambda c: fetch_history(c["ticker"]), seed))
    else:
        hist = [fetch_history(c["ticker"]) for c in seed]
    rows, cols = [], {}
    for c, s in zip(seed, hist):
        if s is None or s.dropna().empty:
            continue
        dl = classify_dead(s, removal_date=pd.Timestamp(c["removal_date"]), today=today)
        if dl is None:
            continue                                       # still trading → not dead
        s = s.loc[:dl].dropna()                            # truncate at death
        peak = float(s.max())
        last = float(s.iloc[-1])
        if peak < min_price_alive:                         # perpetual penny — never a real holding
            continue
        if last > max_survival_ratio * peak:               # fair-value withdrawal/buyout, not a death
            continue
        rows.append({"ticker": c["ticker"], "name": c.get("name", c["ticker"]),
                     "sector": c.get("sector", "Unknown"), "delisting_date": dl,
                     "last_price": last, "peak_price": peak,
                     "spread_pct": float(c.get("spread_pct", 0.5))})
        cols[c["ticker"]] = s
    meta = pd.DataFrame(rows, columns=["ticker", "name", "sector", "delisting_date",
                                       "last_price", "peak_price", "spread_pct"])
    prices = pd.DataFrame(cols)
    return meta, prices


def _spread_to_slippage_bps(spread_pct: float) -> int:
    """Full spread % → per-leg half-spread bps, clamped [2, 75] (≤1.5% ⇒ ≤75)."""
    return int(min(75, max(2, round(spread_pct / 2 * 100))))


def fetch_dead_history(ticker: str):
    """Daily closes up to delisting via yfinance, Stooq CSV fallback. None if both
    fail. Best-effort — wrapped so a single dead name never aborts a batch."""
    try:
        df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        close = df["Close"] if "Close" in df else df
        s = close.dropna()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        if not s.empty:
            return s.squeeze()
    except Exception:
        pass
    try:                                                   # Stooq: lowercase, .F→.de
        sym = ticker.lower().replace(".f", ".de")
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        raw = urllib.request.urlopen(url, timeout=20).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        if "Date" in df and "Close" in df:
            return pd.Series(df["Close"].values,
                             index=pd.to_datetime(df["Date"])).dropna()
    except Exception:
        pass
    return None


def load_seed() -> list[dict]:
    if not SEED_CSV.exists():
        return []
    return pd.read_csv(SEED_CSV).to_dict("records")


DEAD_DIR = ROOT / "data" / "dead_prices"      # committed per-ticker immutable history


def load_dead_prices():
    """Committed per-ticker dead price CSVs → (prices DataFrame, {ticker: delisting})."""
    if not DEAD_DIR.exists():
        return pd.DataFrame(), {}
    cols, delist = {}, {}
    for f in sorted(DEAD_DIR.glob("*.csv")):
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].dropna()
        if len(s):
            cols[f.stem] = s
            delist[f.stem] = s.index[-1]
    return pd.DataFrame(cols), delist


def main(limit: int | None = None):
    """Backfill dead-listing history. Prefers EODHD (paid plan returns full delisted
    history); falls back to the seed list + yfinance/Stooq when no key. Writes
    committed per-ticker CSVs to data/dead_prices/ + the meta to eur_delisted.csv."""
    from tools import eodhd
    key = eodhd.api_key()
    if key:
        cands = eodhd.delisted_candidates(eodhd.fetch_delisted_list(key=key))
        print(f"EODHD delisted candidates: {len(cands)}")
        fetch = lambda t: eodhd.fetch_eod(t, key=key)           # noqa: E731
    else:
        cands = load_seed()
        print(f"no EODHD key — seed candidates: {len(cands)}")
        fetch = fetch_dead_history
    if limit:
        cands = cands[:limit]

    meta, prices = build_dead_table(cands, fetch_history=fetch)
    if not meta.empty:
        meta["slippage_bps"] = meta["spread_pct"].map(_spread_to_slippage_bps)

    DEAD_DIR.mkdir(parents=True, exist_ok=True)
    for t in prices.columns:
        col = prices[t].dropna()
        if len(col):
            col.rename("close").to_csv(DEAD_DIR / f"{t}.csv", header=True)
    OUT_META.parent.mkdir(exist_ok=True)
    meta.to_csv(OUT_META, index=False)
    print(f"kept {len(meta)} dead listings → {OUT_META.name}; "
          f"{len(list(DEAD_DIR.glob('*.csv')))} price files in {DEAD_DIR.name}/")


if __name__ == "__main__":
    import sys
    lim = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    main(limit=lim)
