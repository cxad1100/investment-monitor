"""Polymarket Gamma API — extract prediction market probabilities."""

import re
import requests

POLYMARKET_BASE = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

# Whole-word matches (regex \b boundaries applied) — avoids "war" in "Warnock"
GEO_KEYWORDS_WORD = ["war", "iran", "russia", "ukraine", "taiwan", "china", "nato",
                      "israel", "gaza", "opec"]

# Phrase/substring matches (no word boundary needed — specific enough)
GEO_KEYWORDS_PHRASE = [
    "invade", "invasion", "military", "attack", "ceasefire", "peace deal",
    "sanctions", "nuclear", "nuke", "regime", "blockade", "coup",
    "north korea", "oil price",
]

MACRO_KEYWORDS = [
    "fed rate", "rate cut", "rate hike", "interest rate", "federal reserve",
    "recession", "inflation", "cpi", "gdp", "tariff", "trade war",
    "yield", "treasury", "powell", "ecb", "ipo", "merger",
    "bitcoin hit", "bitcoin reach", "bitcoin dip", "btc hit",
]

EARNINGS_KEYWORDS = ["earnings", "revenue", "beat", "miss", "guidance",
                      "quarterly", "profit", "eps", "results"]

# Exclude pure entertainment / sports noise even if keywords match
EXCLUDE_PHRASES = ["gta vi", "world cup", "nba", "nfl", "nhl", "stanley cup",
                    "super bowl", "oscar", "grammy", "nobel peace prize"]


def parse_probability(market: dict) -> float:
    """Extract Yes probability. outcomePrices[0] = Yes price (0-1 scale)."""
    prices = market.get("outcomePrices", [])
    if prices:
        try:
            p = float(prices[0])
            if 0.0 < p < 1.0:
                return round(p, 4)
        except (ValueError, TypeError):
            pass
    # Fallback to lastTradePrice
    ltp = market.get("lastTradePrice")
    if ltp is not None:
        try:
            p = float(ltp)
            if 0.0 < p < 1.0:
                return round(p, 4)
        except (ValueError, TypeError):
            pass
    return 0.5


def _fetch_all_active_markets(max_markets: int = 2000) -> list[dict]:
    """Paginate Gamma API to collect all active markets."""
    all_markets = []
    offset = 0
    while len(all_markets) < max_markets:
        try:
            resp = requests.get(
                f"{POLYMARKET_BASE}/markets",
                params={"active": "true", "limit": "100", "offset": str(offset)},
                headers=HEADERS, timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                break
            offset += 100
        except Exception:
            break
    return all_markets


def _is_excluded(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in EXCLUDE_PHRASES)


def _to_market_dict(m: dict) -> dict:
    return {
        "id": m.get("id", ""),
        "question": m.get("question", ""),
        "probability": parse_probability(m),
        "volume": float(m.get("volume", 0) or 0),
        "end_date": m.get("endDate", ""),
        "source": "polymarket",
    }


def _matches_geo_macro(question: str) -> bool:
    q = question.lower()
    if any(re.search(r'\b' + k + r'\b', q) for k in GEO_KEYWORDS_WORD):
        return True
    if any(k in q for k in GEO_KEYWORDS_PHRASE + MACRO_KEYWORDS):
        return True
    return False


def fetch_geopolitical_markets(min_volume: float = 500_000.0) -> list[dict]:
    """Fetch active geopolitical + macro prediction markets with investment relevance."""
    all_markets = _fetch_all_active_markets()
    result = []
    for m in all_markets:
        q = m.get("question", "")
        if _is_excluded(q.lower()) or not _matches_geo_macro(q):
            continue
        md = _to_market_dict(m)
        if md["volume"] >= min_volume:
            result.append(md)
    return sorted(result, key=lambda x: x["volume"], reverse=True)[:30]


def fetch_earnings_markets(min_volume: float = 1000.0) -> list[dict]:
    """Fetch active earnings beat/miss prediction markets."""
    all_markets = _fetch_all_active_markets()
    result = []
    for m in all_markets:
        q = m.get("question", "").lower()
        if _is_excluded(q) or not any(k in q for k in EARNINGS_KEYWORDS):
            continue
        md = _to_market_dict(m)
        if md["volume"] >= min_volume:
            result.append(md)
    return sorted(result, key=lambda x: x["volume"], reverse=True)
