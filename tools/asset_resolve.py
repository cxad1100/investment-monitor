"""Resolve an asset named in a video to a yfinance ticker.

Priority: explicit alias (indices / crypto / commodities / mega-caps the channel
talks about) -> reversed known names from COMPANY_NAMES + the broker UNIVERSE ->
the model's ticker_guess -> None (unresolved; the page flags it, the corpus still
keeps the thesis). Returns are percentage moves, so a USD listing is fine for the
math even though the rest of the monitor is EUR.
"""
from __future__ import annotations

import functools
import re

# symbol -> spoken synonyms (normalised on load)
_ALIASES = {
    "^GSPC": ["s&p 500", "s&p500", "sp500", "spx", "s and p 500", "the s&p", "s&p"],
    "^NDX": ["nasdaq 100", "nasdaq100", "ndx"],
    "^IXIC": ["nasdaq", "nasdaq composite"],
    "^DJI": ["dow", "dow jones", "djia"],
    "^RUT": ["russell 2000", "russell2000", "rut"],
    "^GDAXI": ["dax", "german dax"],
    "^VIX": ["vix", "volatility index"],
    "BTC-USD": ["bitcoin", "btc"],
    "ETH-USD": ["ethereum", "eth", "ether"],
    "GC=F": ["gold"],
    "SI=F": ["silver"],
    "CL=F": ["oil", "crude", "wti", "crude oil"],
    "^TNX": ["us 10 year", "10 year treasury", "10-year treasury", "us10y",
             "ten year yield", "10y", "10 year yield"],
    "EURUSD=X": ["eur/usd", "eurusd", "euro dollar", "eur usd"],
    "DX-Y.NYB": ["dxy", "dollar index", "us dollar index"],
    "NVDA": ["nvidia"],
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "TSLA": ["tesla"],
    "AMZN": ["amazon"],
    "META": ["meta", "facebook"],
    "GOOGL": ["google", "alphabet"],
    "NFLX": ["netflix"],
}


def _norm(s: str) -> str:
    """Lowercase, keep alphanumerics only ('S&P 500' -> 'sp500')."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


_LOOKUP = {_norm(syn): sym for sym, syns in _ALIASES.items() for syn in syns}
_LOOKUP.update({_norm(sym): sym for sym in _ALIASES})  # symbol spoken verbatim


@functools.lru_cache(maxsize=1)
def _known_names() -> dict:
    """Normalised display-name -> ticker from COMPANY_NAMES and the UNIVERSE.

    Lazy + guarded: the universe CSV / yfinance imports must not break resolution
    (or import of this module) if absent.
    """
    out: dict = {}
    try:
        from tools.portfolio_tools import COMPANY_NAMES
        for ticker, name in COMPANY_NAMES.items():
            out.setdefault(_norm(name), ticker)
    except Exception:
        pass
    try:
        from tools.pairs_universe import UNIVERSE
        for ticker, meta in UNIVERSE.items():
            name = meta.get("name")
            if name:
                out.setdefault(_norm(name), ticker)
    except Exception:
        pass
    return out


def resolve_asset(name: str, ticker_guess: str | None = None) -> str | None:
    """Best yfinance ticker for an asset, or None if unresolved."""
    key = _norm(name)
    if key and key in _LOOKUP:
        return _LOOKUP[key]
    if key:
        known = _known_names().get(key)
        if known:
            return known
    if ticker_guess:
        g = ticker_guess.strip().upper()
        if g and g.lower() != "null":
            return g
    return None
