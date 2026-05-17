"""
Cross-source theme detector.

Identifies current market themes by aggregating signal strength from:
  - News headlines (frequency + recency)
  - WSB Reddit (sector hype scores + ticker mentions)
  - Polymarket (trading volume as proxy for conviction)
  - Macro data (FRED trends, futures, regime)

Produces ranked themes with composite scores and sector/ticker implications.
Used in fast_scorer to enrich momentum signals beyond pure geopolitics.
"""

from __future__ import annotations
from collections import defaultdict

# ── Theme definitions ────────────────────────────────────────────────────────
# Each theme: keywords for matching each source, affected sectors + tickers.
THEMES: list[dict] = [
    {
        "id":           "ai_semiconductors",
        "label":        "AI & Semiconductors",
        "news_kw":      ["artificial intelligence", "ai model", "semiconductor", "chip", "gpu",
                         "nvidia", "amd", "tsmc", "asml", "deepseek", "chatgpt", "llm",
                         "generative ai", "machine learning", "inference", "h100", "blackwell"],
        "wsb_sectors":  ["Semiconductors", "AI", "Photonics", "Robotics"],
        "poly_kw":      ["ai model", "nvidia", "best ai", "top ai"],
        "sectors":      ["Information Technology"],
        "key_tickers":  ["NVDA", "AMD", "AMAT", "LRCX", "KLAC", "MU", "AVGO", "QCOM", "TSM"],
    },
    {
        "id":           "energy_geopolitics",
        "label":        "Energy & Geopolitics",
        "news_kw":      ["oil", "crude", "brent", "wti", "opec", "lng", "natural gas",
                         "iran", "hormuz", "energy", "petroleum", "barrel", "shale"],
        "wsb_sectors":  ["Energy"],
        "poly_kw":      ["iran", "oil", "opec", "invade iran", "iranian"],
        "sectors":      ["Energy"],
        "key_tickers":  ["XOM", "CVX", "COP", "APA", "EOG", "SLB", "HAL", "VG", "LNG"],
    },
    {
        "id":           "defense_rearmament",
        "label":        "Defense & Rearmament",
        "news_kw":      ["defense", "military", "nato", "weapon", "missile", "pentagon",
                         "defense budget", "lockheed", "raytheon", "northrop", "rearmament",
                         "ukraine aid", "arms", "f-35", "drone"],
        "wsb_sectors":  ["Defense"],
        "poly_kw":      ["nato", "russia invade", "ukraine", "military"],
        "sectors":      ["Industrials"],
        "key_tickers":  ["LMT", "RTX", "NOC", "GD", "LHX", "RKLB", "BA"],
    },
    {
        "id":           "fed_rates",
        "label":        "Fed Policy & Rates",
        "news_kw":      ["federal reserve", "fed", "rate cut", "rate hike", "fomc", "powell",
                         "interest rate", "monetary policy", "inflation", "cpi", "basis point"],
        "wsb_sectors":  [],
        "poly_kw":      ["fed rate", "rate cut", "no change in fed", "fomc", "rate hike in 2026",
                         "basis points after"],
        "sectors":      ["Financials", "Real Estate", "Utilities"],
        "key_tickers":  ["JPM", "BAC", "GS", "MS", "WFC", "SPG", "AMT"],
    },
    {
        "id":           "tariff_trade",
        "label":        "Tariffs & Trade",
        "news_kw":      ["tariff", "trade war", "trade deal", "import duty", "customs duty",
                         "trade policy", "trade deficit", "protectionism", "section 301",
                         "china trade", "trade negotiation", "exemption"],
        "wsb_sectors":  [],
        "poly_kw":      ["tariff", "trade war", "trade deal"],
        "sectors":      ["Industrials", "Consumer Discretionary", "Materials"],
        "key_tickers":  ["CAT", "DE", "MMM", "GE", "HON", "FCX", "NUE"],
    },
    {
        "id":           "crypto_fintech",
        "label":        "Crypto & Fintech",
        "news_kw":      ["bitcoin", "btc", "crypto", "cryptocurrency", "ethereum", "coinbase",
                         "defi", "blockchain", "digital asset", "stablecoin", "web3"],
        "wsb_sectors":  ["Crypto"],
        "poly_kw":      ["bitcoin", "btc", "crypto"],
        "sectors":      ["Financials", "Information Technology"],
        "key_tickers":  ["COIN", "MSTR", "PYPL", "SQ", "MARA", "RIOT"],
    },
    {
        "id":           "macro_recovery",
        "label":        "Macro Recovery / Risk-On",
        "news_kw":      ["economic recovery", "gdp growth", "consumer spending", "retail sales",
                         "jobs report", "payroll", "unemployment falls", "soft landing",
                         "bull market", "risk appetite", "rally", "market high"],
        "wsb_sectors":  [],
        "poly_kw":      ["recession", "gdp"],  # low recession prob = recovery signal
        "sectors":      ["Consumer Discretionary", "Financials", "Industrials"],
        "key_tickers":  ["AMZN", "HD", "MCD", "NKE", "LOW", "TGT"],
    },
    {
        "id":           "biotech_healthcare",
        "label":        "Biotech & Healthcare",
        "news_kw":      ["biotech", "pharmaceutical", "fda approval", "clinical trial",
                         "drug approval", "merger biotech", "gene therapy", "cancer treatment",
                         "obesity drug", "glp-1", "ozempic", "wegovy"],
        "wsb_sectors":  ["Biotech"],
        "poly_kw":      [],
        "sectors":      ["Healthcare"],
        "key_tickers":  ["LLY", "NVO", "MRNA", "ABBV", "BMY", "GILD", "AMGN", "ILMN"],
    },
    {
        "id":           "ipo_pipeline",
        "label":        "IPO Pipeline",
        "news_kw":      ["ipo", "initial public offering", "going public", "direct listing",
                         "stock market debut", "spacex ipo", "stripe", "databricks", "klarna",
                         "anthropic ipo", "pre-ipo"],
        "wsb_sectors":  [],
        "poly_kw":      ["ipo before", "ipo by", "spacex", "databricks", "kraken ipo"],
        "sectors":      ["Information Technology", "Financials"],
        "key_tickers":  [],
    },
    {
        "id":           "geopolitical_peace",
        "label":        "Peace Deals & De-escalation",
        "news_kw":      ["ceasefire", "peace deal", "peace talks", "diplomatic", "negotiations",
                         "treaty", "iran deal", "nuclear agreement", "ukraine peace",
                         "de-escalation", "sanctions lifted", "tariff removal"],
        "wsb_sectors":  [],
        "poly_kw":      ["ceasefire", "peace deal", "nuclear deal", "ukraine signs"],
        "sectors":      ["Energy", "Consumer Discretionary", "Industrials"],
        "key_tickers":  [],
    },
]

_THEME_BY_ID = {t["id"]: t for t in THEMES}


# ── Scoring helpers ──────────────────────────────────────────────────────────

def _news_score(headlines: list[dict], kws: list[str]) -> float:
    """Fraction of headlines mentioning any keyword × 100."""
    if not headlines:
        return 0.0
    hits = sum(1 for h in headlines if any(k in (h.get("title", "") + h.get("summary", "")).lower() for k in kws))
    return round(hits / len(headlines) * 100, 1)


def _wsb_score(sector_hype: list[dict], wsb_sectors: list[str]) -> float:
    """Max hype score across matching WSB sectors × 100."""
    if not wsb_sectors:
        return 0.0
    matches = [
        float(item.get("score", 0))
        for item in sector_hype
        if item.get("sector", item.get("wsb_label", "")) in wsb_sectors
    ]
    return round(max(matches) * 100, 1) if matches else 0.0


def _poly_score(all_markets: list[dict], kws: list[str], is_bearish_theme: bool = False) -> float:
    """Volume-weighted Polymarket interest score (0-100)."""
    if not all_markets or not kws:
        return 0.0
    total_vol = sum(float(m.get("volume", 0) or 0) for m in all_markets) or 1
    theme_vol = sum(
        float(m.get("volume", 0) or 0)
        for m in all_markets
        if any(k in m.get("question", "").lower() for k in kws)
    )
    return round(theme_vol / total_vol * 100, 1)


def _macro_score(macro: dict, sectors: list[str]) -> float:
    """Bonus if sectors are in current regime tailwinds."""
    tailwinds = set(macro.get("sector_tailwinds", []))
    headwinds = set(macro.get("sector_headwinds", []))
    sector_set = set(sectors)
    if sector_set & tailwinds:
        return 75.0
    if sector_set & headwinds:
        return 25.0
    return 50.0


# ── Main function ────────────────────────────────────────────────────────────

def score_themes(signals: dict) -> list[dict]:
    """
    Score all themes across news + WSB + Polymarket + macro.

    Returns themes sorted by composite score descending.
    Each entry: id, label, composite, news_score, wsb_score, poly_score,
    macro_score, sectors, key_tickers, top_news, top_poly_markets
    """
    headlines   = signals.get("news", {}).get("headlines", [])
    wsb         = signals.get("wsb", {})
    sector_hype = wsb.get("sector_hype", [])
    macro       = signals.get("macro", {})
    all_markets = (
        signals.get("polymarket_geo", [])
        + signals.get("polymarket_macro", [])
        + signals.get("polymarket_company", [])
    )
    trending_tickers = {t["ticker"]: t for t in wsb.get("trending_tickers", [])}

    results = []
    for theme in THEMES:
        ns = _news_score(headlines, theme["news_kw"])
        ws = _wsb_score(sector_hype, theme["wsb_sectors"])
        ps = _poly_score(all_markets, theme["poly_kw"])
        ms = _macro_score(macro, theme["sectors"])

        # Composite: news 35% + wsb 25% + poly 25% + macro 15%
        composite = round(0.35 * ns + 0.25 * ws + 0.25 * ps + 0.15 * (ms / 100 * 100), 1)

        # Top supporting headlines
        top_news = [
            h["title"]
            for h in headlines
            if any(k in (h.get("title", "") + h.get("summary", "")).lower() for k in theme["news_kw"])
        ][:5]

        # Top Polymarket markets for this theme
        top_poly = sorted(
            [m for m in all_markets if any(k in m.get("question", "").lower() for k in theme["poly_kw"])],
            key=lambda m: float(m.get("volume", 0) or 0), reverse=True
        )[:3]

        # WSB tickers matching this theme
        wsb_matches = [
            t for t in theme.get("key_tickers", [])
            if t in trending_tickers
        ]

        results.append({
            "id":           theme["id"],
            "label":        theme["label"],
            "composite":    composite,
            "news_score":   ns,
            "wsb_score":    ws,
            "poly_score":   ps,
            "macro_score":  ms,
            "sectors":      theme["sectors"],
            "key_tickers":  theme.get("key_tickers", []),
            "wsb_tickers":  wsb_matches,
            "top_news":     top_news,
            "top_poly":     [{"question": m["question"], "probability": m["probability"],
                              "volume": m.get("volume", 0)} for m in top_poly],
        })

    return sorted(results, key=lambda x: -x["composite"])


def get_sector_theme_score(sector: str, themes: list[dict]) -> float:
    """Return best composite theme score for a sector (0-100). Used in fast_scorer."""
    best = 0.0
    for t in themes:
        if sector in t["sectors"]:
            best = max(best, t["composite"])
    return best


def get_ticker_theme_score(ticker: str, themes: list[dict]) -> float:
    """Return best composite theme score where ticker is a key name."""
    best = 0.0
    for t in themes:
        if ticker in t.get("key_tickers", []):
            best = max(best, t["composite"])
    return best
