"""EODHD client for the survivorship backfill.

The free tier gives the delisted master LIST but caps EOD history to 1 year (a
`[{"warning": …}]` response); the paid 'EOD Historical Data' plan (~$20/mo)
returns full delisted history ending at the delisting date. Network is injected
(`get_fn`) for tests — no live calls in the suite.
"""
import json
import pathlib
import time
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
_ENV = ROOT / ".env"
EXCHANGES = ("XETRA", "F")          # XETRA + Frankfurt — the TR-tradeable German listings


def api_key() -> str | None:
    """Read EODHD_API_KEY from .env (gitignored). None if absent."""
    if not _ENV.exists():
        return None
    for line in _ENV.read_text().splitlines():
        if line.strip().startswith("EODHD_API_KEY"):
            return line.split("=", 1)[1].strip()
    return None


def _http_get(url: str) -> str:
    return urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")


def fetch_eod(symbol: str, *, key: str | None = None, start: str = "2018-01-01",
              get_fn=_http_get, retries: int = 2, retry_delay: float = 1.0):
    """Daily adjusted closes for a (possibly delisted) EODHD symbol → Series.

    None if the free-tier 1-year-cap warning is returned (not upgraded), there is no
    data, or the request keeps failing. Transient network errors (a connection reset
    mid-batch) are retried with backoff so a single blip never aborts a 4k-name run.
    """
    key = key or api_key()
    url = (f"https://eodhd.com/api/eod/{symbol}"
           f"?api_token={key}&fmt=json&from={start}")
    data = None
    for attempt in range(retries + 1):
        try:
            data = json.loads(get_fn(url))
            break
        except Exception:
            if attempt >= retries:
                return None
            time.sleep(retry_delay * (attempt + 1))
    if not isinstance(data, list) or not data or "warning" in data[0]:
        return None
    df = pd.DataFrame(data)
    s = pd.Series(df["adjusted_close"].values,
                  index=pd.to_datetime(df["date"])).dropna()
    return s if len(s) else None


def delisted_candidates(rows: list[dict], *, default_spread_pct: float = 0.5,
                        isin_prefix: str | None = "DE") -> list[dict]:
    """EODHD delisted-list rows → true-death candidates.

    Restricted to domestic issuers (ISIN prefix, default "DE" — a German company
    off its home XETRA/Frankfurt listing is genuinely dead). Foreign cross-listings
    that merely *left* Frankfurt at fair value (Mosaic, Chubu) and mislabeled ETCs
    are excluded — they are not survivorship holes in the German universe. The
    in-window (2018→) + ≥€1 filtering happens after the EOD fetch (classify_dead /
    keep_real); spread is unknown for a dead listing, so a conservative default is
    used and the price floor drops penny noise.
    """
    out, seen = [], set()
    for r in rows:
        if r.get("Type") != "Common Stock":
            continue
        isin = str(r.get("Isin", ""))
        if isin_prefix and not isin.startswith(isin_prefix):
            continue
        key = isin or f"{r.get('Code')}.{r.get('Exchange')}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"ticker": f"{r['Code']}.{r['Exchange']}",
                    "name": str(r.get("Name", r["Code"])), "isin": isin,
                    "sector": "Unknown", "spread_pct": default_spread_pct,
                    "removal_date": "2018-01-01"})
    return out


def fetch_delisted_list(exchanges=EXCHANGES, *, key: str | None = None, get_fn=_http_get):
    """Live delisted master list across `exchanges` (works on the FREE tier)."""
    key = key or api_key()
    rows = []
    for exch in exchanges:
        url = (f"https://eodhd.com/api/exchange-symbol-list/{exch}"
               f"?api_token={key}&delisted=1&fmt=json")
        rows += json.loads(get_fn(url))
    return rows
