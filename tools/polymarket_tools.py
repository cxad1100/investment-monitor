"""Polymarket Gamma API — prediction market signals for investment analysis.

Four independent fetchers, each targeting a distinct signal category:
  - fetch_geo_conflict_markets()   geopolitical / military risk
  - fetch_macro_markets()          Fed policy, recession, inflation
  - fetch_company_markets()        stock-specific (market cap, IPOs, AI)
  - fetch_all_investment_markets() combined deduplicated set
"""

import re
import requests

POLYMARKET_BASE = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

# ── Noise exclusion ──────────────────────────────────────────────────────────
EXCLUDE_PHRASES = [
    "gta vi", "world cup", "nba", "nfl", "nhl", "stanley cup", "super bowl",
    "oscar", "grammy", "champions league", "europa league", "premier league",
    "formula 1", "f1 ", "wimbledon", "us open", "tour de france",
    "miss universe", "miss world", "eurovision", "american idol",
    "fdv above", "fdv below",          # crypto token launches — not investable
    "metamask", "hyperbeat",           # specific low-relevance crypto projects
    "nomination",                      # 2028 political nominations — too far out
    "2028 us presidential",
]

# ── Keyword sets ─────────────────────────────────────────────────────────────

# Geopolitical: word-boundary matched to avoid false positives (war in Warnock etc.)
GEO_WORDS = ["iran", "russia", "ukraine", "taiwan", "china", "nato", "israel",
             "gaza", "opec", "north korea", "putin", "zelenskyy"]
GEO_PHRASES = [
    "invade", "invasion", "military clash", "ceasefire", "peace deal",
    "sanctions", "nuclear deal", "nuke", "regime fall", "blockade", "coup",
    "strait of hormuz", "oil embargo",
]

# Macro: Fed meetings (specific), annual path, recession, inflation
FOMC_MEETING_PHRASES = [
    "fed decrease interest rate", "fed increase interest rate",
    "no change in fed interest rate", "fomc meeting",
    "fed cut rates after", "fed raise rates after",
    "basis points after the",
]
MACRO_ANNUAL_PHRASES = [
    "fed rate cuts happen in 2026", "fed rate cuts happen in 2027",
    "rate cut happen in 2026", "rate hike in 2026",
    "recession by end of", "us recession",
    "cpi above", "cpi below", "inflation above", "inflation below",
    "gdp growth", "unemployment above", "unemployment below",
    "tariff", "trade war",
]

# Company / stock-specific
COMPANY_PHRASES = [
    # Market cap / stock-specific
    "largest company in the world by market cap",
    "ipo before", "ipo by", "ipo day",
    "market cap be greater than", "market cap between",
    "spacex", "stripe ipo", "databricks ipo", "kraken ipo", "openai ipo",
    "best ai model", "top ai model",
    "will tesla", "will nvidia", "will apple", "will microsoft",
    "will amazon", "will google", "will meta",
    # Bitcoin / crypto price levels
    "bitcoin hit", "bitcoin reach", "bitcoin dip",
    "btc hit", "btc reach", "btc dip",
    "bitcoin $", "bitcoin at $",
]


def parse_probability(market: dict) -> float:
    """Extract Yes probability from outcomePrices[0] or lastTradePrice."""
    prices = market.get("outcomePrices", [])
    candidates = ([prices[0]] if prices else []) + [market.get("lastTradePrice")]
    for src in candidates:
        if src is None:
            continue
        try:
            p = float(src)
            if 0.0 < p < 1.0:
                return round(p, 4)
        except (ValueError, TypeError):
            pass
    return 0.5


def _fetch_all_active(max_markets: int = 3000) -> list[dict]:
    """Paginate Gamma API — returns up to max_markets active markets."""
    markets: list[dict] = []
    offset = 0
    while len(markets) < max_markets:
        try:
            r = requests.get(
                f"{POLYMARKET_BASE}/markets",
                params={"active": "true", "limit": "100", "offset": str(offset)},
                headers=HEADERS, timeout=30,
            )
            r.raise_for_status()
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            markets.extend(batch)
            if len(batch) < 100:
                break
            offset += 100
        except Exception:
            break
    return markets


def _is_noise(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in EXCLUDE_PHRASES)


def _to_md(m: dict, category: str = "") -> dict:
    return {
        "id": m.get("id", ""),
        "question": m.get("question", ""),
        "probability": parse_probability(m),
        "volume": float(m.get("volume", 0) or 0),
        "end_date": m.get("endDate", ""),
        "category": category,
        "source": "polymarket",
    }


def _word_match(question: str, words: list[str]) -> bool:
    q = question.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", q) for w in words)


def _phrase_match(question: str, phrases: list[str]) -> bool:
    q = question.lower()
    return any(p in q for p in phrases)


# ── Public fetchers ──────────────────────────────────────────────────────────

def fetch_geo_conflict_markets(min_volume: float = 200_000.0) -> list[dict]:
    """Military / geopolitical conflict markets with investment implications."""
    raw = _fetch_all_active()
    out = []
    for m in raw:
        q = m.get("question", "")
        if _is_noise(q):
            continue
        if _word_match(q, GEO_WORDS) or _phrase_match(q, GEO_PHRASES):
            md = _to_md(m, "geo")
            if md["volume"] >= min_volume:
                out.append(md)
    return sorted(out, key=lambda x: x["volume"], reverse=True)[:40]


def fetch_macro_markets(min_volume: float = 200_000.0) -> list[dict]:
    """Fed policy (meeting-specific + annual path), recession, inflation markets."""
    raw = _fetch_all_active()
    out = []
    seen = set()
    for m in raw:
        q = m.get("question", "")
        if _is_noise(q) or m.get("id") in seen:
            continue
        if _phrase_match(q, FOMC_MEETING_PHRASES + MACRO_ANNUAL_PHRASES):
            md = _to_md(m, "macro")
            if md["volume"] >= min_volume:
                out.append(md)
                seen.add(m.get("id"))
    return sorted(out, key=lambda x: x["volume"], reverse=True)[:50]


def fetch_company_markets(min_volume: float = 100_000.0) -> list[dict]:
    """Stock-specific: market cap dominance, IPO events, AI leadership."""
    raw = _fetch_all_active()
    out = []
    for m in raw:
        q = m.get("question", "")
        if _is_noise(q):
            continue
        if _phrase_match(q, COMPANY_PHRASES):
            md = _to_md(m, "company")
            if md["volume"] >= min_volume:
                out.append(md)
    return sorted(out, key=lambda x: x["volume"], reverse=True)[:30]


def fetch_all_investment_markets() -> dict:
    """Single paginate pass → all three categories. Avoids triple API hammering."""
    raw = _fetch_all_active(max_markets=3000)
    geo, macro, company = [], [], []
    seen: set[str] = set()

    for m in raw:
        mid = m.get("id", "")
        if mid in seen:
            continue
        q = m.get("question", "")
        if _is_noise(q):
            continue
        vol = float(m.get("volume", 0) or 0)
        md = _to_md(m)

        is_geo = (_word_match(q, GEO_WORDS) or _phrase_match(q, GEO_PHRASES)) and vol >= 200_000
        is_macro = _phrase_match(q, FOMC_MEETING_PHRASES + MACRO_ANNUAL_PHRASES) and vol >= 200_000
        is_company = _phrase_match(q, COMPANY_PHRASES) and vol >= 100_000

        if is_geo:
            md["category"] = "geo"
            geo.append(md)
            seen.add(mid)
        elif is_macro:
            md["category"] = "macro"
            macro.append(md)
            seen.add(mid)
        elif is_company:
            md["category"] = "company"
            company.append(md)
            seen.add(mid)

    return {
        "geo": sorted(geo, key=lambda x: x["volume"], reverse=True)[:40],
        "macro": sorted(macro, key=lambda x: x["volume"], reverse=True)[:50],
        "company": sorted(company, key=lambda x: x["volume"], reverse=True)[:30],
    }


# ── Legacy aliases (used by collect_all.py) ─────────────────────────────────
def fetch_geopolitical_markets(min_volume: float = 200_000.0) -> list[dict]:
    return fetch_geo_conflict_markets(min_volume)


def fetch_earnings_markets(min_volume: float = 1000.0) -> list[dict]:
    return fetch_company_markets(min_volume)
