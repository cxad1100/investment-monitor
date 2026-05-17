"""
WSB Reddit trend analyser.
Converts raw WSB signal into a structured current market trend definition:
which sectors retail traders are piling into, what momentum looks like,
and whether squeeze conditions are present.
"""

from collections import defaultdict

# Map WSB sector hype labels to standard sector names
SECTOR_MAP = {
    "Semiconductors":   "Information Technology",
    "AI":               "Information Technology",
    "Robotics":         "Industrials",
    "Photonics":        "Information Technology",
    "Defense":          "Industrials",
    "Energy":           "Energy",
    "Biotech":          "Healthcare",
    "Crypto":           "Financials",
}

# Score thresholds for narrative labels
HYPE_HIGH   = 0.5
HYPE_MEDIUM = 0.15


def _sector_label(score: float) -> str:
    if score >= HYPE_HIGH:
        return "strong momentum"
    if score >= HYPE_MEDIUM:
        return "building interest"
    return "light activity"


def compute_retail_trend(wsb_signal: dict, universe_map: dict, short_interest: dict) -> dict:
    """
    Synthesise WSB into a current trend object.

    Returns:
      sector_momentum: list of {sector, wsb_label, score, std_sector} sorted by score
      trending_stocks:  list of {ticker, company, mentions, sector, squeeze_flag}
      squeeze_plays:    tickers with both WSB attention AND high short interest
      trend_direction:  'risk_on' | 'defensive' | 'mixed'
      narrative:        plain-English summary of current retail trend
    """
    sector_hype    = wsb_signal.get("sector_hype", [])
    trending_ticks = wsb_signal.get("trending_tickers", [])
    squeeze_cands  = wsb_signal.get("squeeze_candidates", [])

    # ── Sector momentum ───────────────────────────────────────────────────────
    sector_momentum = []
    for item in sector_hype:
        wsb_label = item.get("sector", item.get("wsb_label", "Unknown"))
        score     = float(item.get("score", 0))
        std       = SECTOR_MAP.get(wsb_label, "Other")
        sector_momentum.append({
            "wsb_label":  wsb_label,
            "std_sector": std,
            "score":      round(score, 3),
            "momentum":   _sector_label(score),
        })
    sector_momentum.sort(key=lambda x: -x["score"])

    # ── Trending stocks with sector context ──────────────────────────────────
    trending_stocks = []
    for item in trending_ticks:
        ticker = item["ticker"]
        meta   = universe_map.get(ticker, {})
        si     = short_interest.get(ticker, {})
        short_float = si.get("short_float_pct", 0) or 0

        trending_stocks.append({
            "ticker":       ticker,
            "company":      meta.get("name", ticker),
            "sector":       meta.get("sector", "Unknown"),
            "mentions":     item["mentions_7d"],
            "squeeze_flag": item.get("squeeze_flag", False),
            "short_float":  round(short_float, 1),
        })

    # ── Squeeze plays: WSB interest + elevated short float ───────────────────
    squeeze_plays = [
        s for s in trending_stocks
        if s["short_float"] > 10 or s["squeeze_flag"]
    ]

    # ── Trend direction ───────────────────────────────────────────────────────
    risk_on_sectors   = {"Semiconductors", "AI", "Robotics", "Crypto", "Energy"}
    defensive_sectors = {"Biotech", "Healthcare", "Consumer Staples"}

    top_wsb_labels = {item.get("sector", item.get("wsb_label", "")) for item in sector_hype[:3]}
    risk_on_count   = len(top_wsb_labels & risk_on_sectors)
    defensive_count = len(top_wsb_labels & defensive_sectors)

    if risk_on_count > defensive_count:
        trend_direction = "risk_on"
    elif defensive_count > risk_on_count:
        trend_direction = "defensive"
    else:
        trend_direction = "mixed"

    # ── Narrative ─────────────────────────────────────────────────────────────
    top = sector_momentum[:3]
    if top:
        primary   = top[0]["wsb_label"]
        secondary = [t["wsb_label"] for t in top[1:]]
        parts = [f"Retail focus concentrated in {primary}"]
        if secondary:
            parts.append(f"with secondary interest in {' and '.join(secondary)}")

        top_tickers = [t["ticker"] for t in trending_stocks[:5]]
        if top_tickers:
            parts.append(f"Top tickers: {', '.join(top_tickers)}")

        if squeeze_plays:
            sq = ', '.join(s["ticker"] for s in squeeze_plays[:3])
            parts.append(f"Squeeze watch: {sq}")

        narrative = ". ".join(parts) + "."
    else:
        narrative = "Insufficient WSB activity this week for trend definition."

    return {
        "sector_momentum":  sector_momentum,
        "trending_stocks":  trending_stocks,
        "squeeze_plays":    squeeze_plays,
        "trend_direction":  trend_direction,
        "narrative":        narrative,
        "total_posts":      wsb_signal.get("total_posts_analyzed", 0),
    }
