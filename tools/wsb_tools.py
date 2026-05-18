"""Reddit retail sentiment — r/wallstreetbets + r/investing + r/stocks + r/options."""

import re
from collections import defaultdict
import requests

# Subreddits to scrape
SUBREDDIT_URLS = [
    # WSB — high-velocity, speculative
    "https://www.reddit.com/r/wallstreetbets/hot.json?limit=100",
    "https://www.reddit.com/r/wallstreetbets/top.json?t=week&limit=100",
    # Broader retail investing
    "https://www.reddit.com/r/stocks/hot.json?limit=100",
    "https://www.reddit.com/r/stocks/top.json?t=week&limit=100",
    "https://www.reddit.com/r/investing/hot.json?limit=100",
    "https://www.reddit.com/r/investing/top.json?t=week&limit=100",
    "https://www.reddit.com/r/options/hot.json?limit=100",
    "https://www.reddit.com/r/StockMarket/hot.json?limit=100",
    "https://www.reddit.com/r/SecurityAnalysis/hot.json?limit=50",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

SQUEEZE_KEYWORDS = [
    "squeeze", "short interest", "gamma squeeze", "short float",
    "days to cover", "si%", "ftd", "fail to deliver", "short seller",
    "heavily shorted", "max pain",
]

BULLISH_KEYWORDS = [
    "buy", "bull", "long", "calls", "moon", "undervalued", "breakout",
    "bullish", "upside", "accumulate", "strong buy", "beat earnings",
    "raised guidance", "record revenue", "buyback", "dividend",
]

BEARISH_KEYWORDS = [
    "sell", "bear", "short", "puts", "crash", "overvalued", "breakdown",
    "bearish", "downside", "miss", "missed earnings", "cut guidance",
    "layoffs", "fraud", "lawsuit", "downgrade",
]

SECTOR_KEYWORDS = {
    "Semiconductors":   ["semiconductor", "chip", "foundry", "wafer", "tsmc", "amd", "nvidia", "intel", "qualcomm", "asml", "arm"],
    "AI":               ["artificial intelligence", "llm", "machine learning", "deep learning", "neural network", "ai stocks", "openai", "anthropic", "copilot"],
    "Robotics":         ["robot", "robotics", "automation", "autonomous", "humanoid", "boston dynamics"],
    "Defense":          ["defense", "military", "nato", "pentagon", "lockheed", "raytheon", "missile", "wartime", "geopolit"],
    "Biotech":          ["biotech", "fda", "clinical trial", "drug approval", "biopharma", "oncology", "phase 3", "phase 2"],
    "Energy":           ["oil", "energy stocks", "crude", "lng", "solar", "renewables", "drilling", "opec", "natural gas"],
    "Financials":       ["bank", "interest rate", "fed", "federal reserve", "credit", "fintech", "insurance", "rate cut", "rate hike"],
    "Crypto":           ["bitcoin", "btc", "ethereum", "defi", "crypto", "blockchain", "coinbase", "solana", "altcoin"],
    "Consumer":         ["retail", "consumer spending", "e-commerce", "amazon", "shopify", "consumer confidence"],
    "Healthcare":       ["healthcare", "hospital", "pharma", "drug", "vaccine", "medical device", "insurance", "medicare"],
}

EXCLUDE_TOKENS = {
    "A", "I", "AM", "PM", "US", "UK", "EU", "DD", "OP", "YOY", "IMO",
    "EOD", "CEO", "CFO", "ETF", "EPS", "FYI", "SEC", "IPO", "ATH", "ATL",
    "TA", "TD", "IV", "OI", "WSB", "SPY", "SPX", "VIX", "GDP", "CPI",
    "NOT", "THE", "AND", "OR", "BUT", "FOR", "ARE", "WAS", "HAS", "HAD",
    "BE", "TO", "IN", "IS", "IT", "OF", "ON", "AT", "BY", "DO", "IF",
    "NO", "SO", "UP", "AN", "AS", "MY", "HE", "WE", "GO", "ME", "ALL",
    "ITS", "CAN", "NOW", "GET", "GOT", "USE", "NEW", "ONE", "TWO", "THREE",
    "THEY", "THEIR", "THAT", "THIS", "WITH", "FROM", "HAVE", "BEEN",
    "WHEN", "WHAT", "WILL", "MORE", "VERY", "JUST", "ALSO", "ONLY",
    "OUT", "TOO", "HOW", "WHY", "WHO", "ANY", "IRS", "PNL", "ROI",
    "AH", "PM", "RSI", "MACD", "PE", "PB", "EV", "DCF", "FCF",
    "YTD", "QOQ", "MOM", "CAGR", "APY", "APR", "NAV", "AUM",
}


def extract_tickers_from_text(text: str) -> list[str]:
    """Extract $TICKER and likely standalone uppercase tickers."""
    dollar = re.findall(r'\$([A-Z]{1,5})\b', text)
    # Also match 1-5 uppercase letters surrounded by spaces/punctuation (no $)
    bare = re.findall(r'(?<![A-Z\$])([A-Z]{2,5})(?![A-Z])', text)
    combined = dollar + [t for t in bare if t not in EXCLUDE_TOKENS]
    return list(dict.fromkeys(combined))  # deduplicate, preserve order


def _sentiment_score(text: str) -> float:
    """Return sentiment in [-1, 1]: positive = bullish, negative = bearish."""
    lower = text.lower()
    bull = sum(1 for k in BULLISH_KEYWORDS if k in lower)
    bear = sum(1 for k in BEARISH_KEYWORDS if k in lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


def fetch_wsb_posts() -> list[dict]:
    """Fetch posts from multiple retail investing subreddits."""
    seen_ids: set[str] = set()
    posts = []
    for url in SUBREDDIT_URLS:
        subreddit = url.split("/r/")[1].split("/")[0]
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                p = child.get("data", {})
                post_id = p.get("id", "")
                if post_id in seen_ids:
                    continue
                seen_ids.add(post_id)
                posts.append({
                    "id":           post_id,
                    "title":        p.get("title", ""),
                    "score":        int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "text":         p.get("selftext", ""),
                    "subreddit":    subreddit,
                })
        except Exception:
            pass
    return posts


def analyze_wsb_signals(posts: list[dict]) -> dict:
    """
    Aggregate ticker mentions, sentiment, squeeze flags, and sector hype.

    Mention weight = 1 + log10(max(score, 1)) so high-upvote posts count more.
    """
    import math

    weighted_mentions: dict[str, float] = defaultdict(float)
    sentiment_accum:   dict[str, list[float]] = defaultdict(list)
    squeeze_posts:     dict[str, list[str]] = defaultdict(list)

    for post in posts:
        full_text = post["title"] + " " + post.get("text", "")
        weight = 1.0 + math.log10(max(post.get("score", 1), 1))
        tickers = extract_tickers_from_text(full_text)
        sent = _sentiment_score(full_text)

        for t in tickers:
            weighted_mentions[t] += weight
            sentiment_accum[t].append(sent)

        text_lower = full_text.lower()
        if any(k in text_lower for k in SQUEEZE_KEYWORDS):
            for t in tickers:
                squeeze_posts[t].append(post["title"])

    # Build per-ticker dict including mention count + sentiment
    ticker_mentions: dict[str, dict] = {}
    for ticker, w_count in sorted(weighted_mentions.items(), key=lambda x: -x[1])[:50]:
        sents = sentiment_accum[ticker]
        ticker_mentions[ticker] = {
            "mentions_7d":   round(w_count, 1),
            "squeeze_flag":  ticker in squeeze_posts,
            "sentiment":     round(sum(sents) / len(sents), 3) if sents else 0.0,
        }

    # Sector hype from raw keyword counts
    all_text_lower = " ".join(
        p["title"].lower() + " " + p.get("text", "").lower()
        for p in posts
    )
    sector_scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        count = sum(all_text_lower.count(k) for k in keywords)
        score = min(1.0, count / 40.0)
        if score > 0.05:
            sector_scores[sector] = round(score, 3)

    trending = [
        {"ticker": t, **v}
        for t, v in list(ticker_mentions.items())[:30]
    ]

    return {
        "trending_tickers": trending,
        "ticker_mentions":  ticker_mentions,
        "sector_hype": [
            {"sector": s, "score": sc}
            for s, sc in sorted(sector_scores.items(), key=lambda x: -x[1])
        ],
        "squeeze_candidates": list(squeeze_posts.keys())[:10],
        "total_posts_analyzed": len(posts),
        "subreddits": list({p["subreddit"] for p in posts}),
    }
