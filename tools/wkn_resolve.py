"""Resolve a broker WKN (Local_ID) to a yfinance ticker, with an on-disk cache.

The scraped broker list (`universe.csv`) identifies each stock by
its German WKN, not a yfinance ticker — but cointegration needs price history.

- German securities: WKN → ISIN is deterministic (DE000 + WKN + ISIN check digit),
  and Yahoo's search endpoint resolves the ISIN to its XETRA `.DE` primary listing.
- Foreign securities (the WKN doesn't encode a DE ISIN): resolve by name through
  the same endpoint, best-effort.

`resolve_ticker` is network I/O; results are memoised to a JSON cache so the
offline universe build hits Yahoo once per name. The pure helpers
(`isin_from_wkn`, `spread_to_slippage`) are unit-tested.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "wkn_ticker_map.json"

_SEARCH = "https://query2.finance.yahoo.com/v1/finance/search?q="
_UA = {"User-Agent": "Mozilla/5.0"}
# Prefer the German listings (EUR-denominated, what the broker trades), then any.
_EXCH_RANK = {"GER": 0, "FRA": 1, "STU": 2, "MUN": 3, "DUS": 4, "HAM": 5, "BER": 6}


def _isin_check_digit(body: str) -> int:
    """ISIN check digit over the 11-char body (e.g. 'DE000' + 6-char WKN).
    Letters expand A=10..Z=35, then a Luhn pass over the digit string."""
    s = "".join(str(ord(c) - 55) if c.isalpha() else c for c in body)
    digits = [int(d) for d in s][::-1]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def isin_from_wkn(wkn: str) -> str:
    """Deterministic ISIN for a German-domiciled security from its 6-char WKN."""
    body = "DE000" + wkn.strip().upper()      # 11 chars
    return body + str(_isin_check_digit(body))


def spread_to_slippage(bid: float, ask: float, lo: int = 2, hi: int = 50) -> int:
    """Half bid/ask spread in bps (per-leg slippage), clamped to [lo, hi]."""
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return hi
    mid = (bid + ask) / 2.0
    if mid <= 0 or ask < bid:
        return hi
    half_bps = (ask - bid) / mid / 2.0 * 1e4
    return int(min(hi, max(lo, round(half_bps))))


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))


def _yahoo_search(query: str, timeout: float = 15.0, retries: int = 3) -> list[dict] | None:
    """Quotes for a query, or None if every attempt hard-failed (timeout/429).
    A definitive empty result is []; None is a *transient* failure the caller
    must NOT cache as a known-miss, or one rate-limit poisons the WKN forever."""
    url = _SEARCH + urllib.parse.quote(query)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r).get("quotes", [])
        except Exception:
            time.sleep(1.5 * (attempt + 1))           # back off (rate limit/timeout)
    return None


# Generic corporate/denomination words — ignored when matching a broker name to a
# Yahoo result, so "ORACLE CORP. DL-,01" matches "Oracle Corp." on the token ORACLE.
_STOP = {"INC", "CORP", "CO", "LTD", "PLC", "THE", "GROUP", "GRP", "HLDG", "HLDGS",
         "HOLDING", "HOLDINGS", "ORD", "REG", "SHS", "CLASS", "COM", "NEW", "AKT"}


def _name_tokens(s: str) -> set[str]:
    """Alpha tokens (>=3 chars) of a name, minus generic corporate words — used to
    confirm a Yahoo result actually corresponds to the queried name."""
    return {t for t in re.findall(r"[A-Z]{3,}", (s or "").upper()) if t not in _STOP}


def _clean_name(name: str) -> str:
    """Drop the ADR/denomination tail of a broker name ('…ADR/5', 'DL-,01',
    'EO-,01', 'O.N.', 'INH', …) for a Yahoo name search; dots are kept here."""
    s = re.sub(r"\bSP\.?\s*ADR\b|\bADR\s*/?\s*\d*\b", " ", (name or "").upper())  # ADR / ADR/5 / SP.ADR
    s = re.split(r"\s{2,}|\bDL[\s\-]|\bEO[\s\-]|\sO\.?N\.?\b|\bINH\b|\bVINK\b|\bVZO\b|\bNAM\b", s)[0]
    return re.sub(r"\s+", " ", s).strip(" ,-") or (name or "")


def _name_variants(name: str) -> list[str]:
    """Search forms to try in order: cleaned name as-is (keeps brand dots like
    'AMAZON.COM'), then with dots de-glued ('SEMICON.MANU.' -> 'SEMICON MANU').
    Some names match only one form (Amazon needs the dot, TSMC needs it gone)."""
    base = _clean_name(name)
    nodots = re.sub(r"\s+", " ", base.replace(".", " ")).strip(" ,-")
    return [base] if nodots == base else [base, nodots]


def _best_quote(quotes: list[dict] | None) -> dict | None:
    # German EUR exchanges only: the broker trades the German listing, and an
    # EUR-only universe keeps pairs FX-safe (no foreign USD/GBP/CHF listings).
    eq = [q for q in (quotes or [])
          if q.get("symbol") and q.get("quoteType") == "EQUITY"
          and q.get("exchange") in _EXCH_RANK]
    if not eq:
        return None
    eq.sort(key=lambda q: _EXCH_RANK[q["exchange"]])
    return eq[0]


def _names_match(query: str, quote: dict) -> bool:
    """True if the broker name and the quote's name share a real token — rejects
    Yahoo fuzzy non-matches (the DFK0.F='01 Quantum' attractor for unmatched names)."""
    cand = f"{quote.get('shortname', '')} {quote.get('longname', '')}"
    return bool(_name_tokens(query) & _name_tokens(cand))


def resolve_ticker(wkn: str, name: str, country: str, *,
                   cache: dict | None = None, throttle: float = 0.3) -> str | None:
    """WKN → yfinance ticker (the German EUR listing). German-domiciled issuers
    use the deterministic DE-ISIN (DE000+WKN is a *real* ISIN only for them);
    foreign / N-A names resolve by name search — for those the computed DE-ISIN
    is fake and Yahoo fuzzy-matches it to an unrelated stock (the DFK0.F bug).
    Either way the result is accepted only if its name actually matches the broker
    name (`_names_match`) and it is a German EUR listing (`_best_quote`) — this
    rejects Yahoo's fuzzy non-matches (the DFK0.F='01 Quantum' attractor) and keeps
    the universe FX-safe. Memoised by WKN; returns None if unresolved."""
    cache = _load_cache() if cache is None else cache
    if wkn in cache:                                   # "" cached = known-miss
        return cache[wkn] or None

    failed = False
    quote = None
    if str(country).strip().lower() in ("germany", "deutschland", "de"):
        q = _yahoo_search(isin_from_wkn(wkn))          # real DE-ISIN only for German issuers
        failed = q is None
        quote = _best_quote(q)
    if quote is None or not _names_match(name, quote):  # ISIN missed/garbage → by name
        quote = None
        for nm in _name_variants(name):                 # dotted form, then de-glued
            time.sleep(throttle)
            q = _yahoo_search(nm)
            failed = failed or q is None
            cand = _best_quote(q)
            if cand and _names_match(name, cand):
                quote = cand
                break
    sym = quote["symbol"] if (quote and _names_match(name, quote)) else None
    time.sleep(throttle)
    if sym is None and failed:
        return None                                    # transient — leave uncached, retry next run
    cache[wkn] = sym or ""                             # definitive: a hit, or a true no-match
    return sym
