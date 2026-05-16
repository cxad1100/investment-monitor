"""Polymarket Gamma API — extract prediction market probabilities."""

import requests

POLYMARKET_BASE = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

GEO_KEYWORDS = ["war", "attack", "invasion", "ceasefire", "peace", "sanctions",
                 "nuclear", "conflict", "military", "troops", "strike", "embargo",
                 "election", "coup", "treaty", "referendum", "crisis"]

EARNINGS_KEYWORDS = ["earnings", "revenue", "beat", "miss", "guidance",
                      "quarterly", "profit", "eps", "results"]


def parse_probability(market: dict) -> float:
    """Extract Yes probability from outcomePrices array."""
    prices = market.get("outcomePrices", [])
    if prices:
        try:
            return round(float(prices[0]), 4)
        except (ValueError, TypeError):
            pass
    return 0.5


def filter_by_volume(markets: list[dict], min_volume: float = 1000.0) -> list[dict]:
    """Keep only markets with sufficient trading volume."""
    result = []
    for m in markets:
        vol = float(m.get("volume", 0) or 0)
        if vol >= min_volume:
            result.append(m)
    return result


def _fetch_markets(params: dict) -> list[dict]:
    try:
        resp = requests.get(f"{POLYMARKET_BASE}/markets", params=params,
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else []
    except Exception:
        return []


def fetch_geopolitical_markets(min_volume: float = 2000.0) -> list[dict]:
    """Fetch active geopolitical prediction markets."""
    markets = _fetch_markets({"active": "true", "limit": "200"})
    geo = []
    for m in markets:
        q = m.get("question", "").lower()
        if any(k in q for k in GEO_KEYWORDS):
            geo.append({
                "id": m.get("id", ""),
                "question": m.get("question", ""),
                "probability": parse_probability(m),
                "volume": float(m.get("volume", 0) or 0),
                "end_date": m.get("endDate", ""),
                "source": "polymarket",
            })
    return filter_by_volume(geo, min_volume)


def fetch_earnings_markets(min_volume: float = 1000.0) -> list[dict]:
    """Fetch active earnings beat/miss prediction markets."""
    markets = _fetch_markets({"active": "true", "limit": "200"})
    earnings = []
    for m in markets:
        q = m.get("question", "").lower()
        if any(k in q for k in EARNINGS_KEYWORDS):
            earnings.append({
                "id": m.get("id", ""),
                "question": m.get("question", ""),
                "probability": parse_probability(m),
                "volume": float(m.get("volume", 0) or 0),
                "end_date": m.get("endDate", ""),
                "source": "polymarket",
            })
    return filter_by_volume(earnings, min_volume)
