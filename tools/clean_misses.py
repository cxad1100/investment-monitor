"""Resolve the broker 'misses' (data/universe_misses.csv — 579 tradeable names the
auto-resolver couldn't map to a yfinance ticker) by scrubbing the German broker
name junk and retrying resolution. Expands the survivor universe; best-effort.
"""
import re
from pathlib import Path

import pandas as pd

from tools.wkn_resolve import resolve_ticker, spread_to_slippage

ROOT = Path(__file__).resolve().parent.parent
MISSES_CSV = ROOT / "data" / "universe_misses.csv"

# A trailing Lang & Schwarz junk token: a currency-denomination tail (DL-,001 =
# USD 0.001 par, EO-,75 = EUR 0.75 par), a holder/registry word (INH./NAM./O.N.,
# possibly concatenated with a denomination as in "INH.EO-,75"), or a lone share-
# class letter ("N", "SV"). Real company-type tails ("CORP.", "SA", "AG") and real
# words ("MAERSK") are NOT junk and are kept.
_JUNK_TOKEN = re.compile(
    r"""^(?:
        (?:DL|EO|SF|YC|NK|CHF|DKK|SEK|HK|JY|FF|LX|RU|ZY)[-.,].*   # currency par tail
      | (?:INH|NAM|VINK|REG|RC|SV|ON)\.?.*                         # holder/registry word
      | O\.N\.?                                                    # O.N.
      | [A-Z]                                                      # single class letter
    )$""",
    re.X,
)


def clean_name(name: str) -> str:
    """Strip a run of trailing broker par/class junk tokens; keep the real name."""
    s = re.sub(r"\s+", " ", str(name)).strip()
    toks = s.split(" ")
    while len(toks) > 1 and _JUNK_TOKEN.match(toks[-1]):
        toks.pop()
    return " ".join(toks).strip().rstrip(".").strip()


def resolve_misses(rows: list[dict], *, resolve_fn=resolve_ticker) -> list[dict]:
    """Each miss → resolved meta row, dropping unresolved. `resolve_fn(wkn, name,
    country) -> ticker|None` is injected (real one hits the network/cache)."""
    out = []
    for r in rows:
        cleaned = clean_name(r["Name"])
        tk = resolve_fn(str(r.get("Local_ID", "")), cleaned, str(r.get("Country", "")))
        if not tk:
            continue
        out.append({
            "ticker": tk, "name": cleaned,
            "sector": str(r.get("Sector", "Unknown")) or "Unknown",
            "country": str(r.get("Country", "—")), "currency": "EUR",
            "slippage_bps": spread_to_slippage(r.get("Bid"), r.get("Ask")),
            "local_id": str(r.get("Local_ID", "")),
        })
    return out


def load_misses() -> list[dict]:
    if not MISSES_CSV.exists():
        return []
    return pd.read_csv(MISSES_CSV).to_dict("records")
