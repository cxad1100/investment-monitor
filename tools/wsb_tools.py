"""Reddit retail sentiment — multi-subreddit with comment depth and DD extraction."""

import re
import math
import time
from collections import defaultdict
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

# ── Subreddits ───────────────────────────────────────────────────────────────
SUBREDDIT_URLS = [
    "https://www.reddit.com/r/wallstreetbets/hot.json?limit=100",
    "https://www.reddit.com/r/wallstreetbets/top.json?t=week&limit=100",
    "https://www.reddit.com/r/wallstreetbets/new.json?limit=50",
    "https://www.reddit.com/r/stocks/hot.json?limit=100",
    "https://www.reddit.com/r/stocks/top.json?t=week&limit=100",
    "https://www.reddit.com/r/investing/hot.json?limit=100",
    "https://www.reddit.com/r/investing/top.json?t=week&limit=100",
    "https://www.reddit.com/r/options/hot.json?limit=100",
    "https://www.reddit.com/r/StockMarket/hot.json?limit=100",
    "https://www.reddit.com/r/SecurityAnalysis/hot.json?limit=50",
    "https://www.reddit.com/r/ValueInvesting/hot.json?limit=50",
]

# ── Signal keywords ──────────────────────────────────────────────────────────
SQUEEZE_KEYWORDS = [
    "squeeze", "short interest", "gamma squeeze", "short float",
    "days to cover", "si%", "ftd", "fail to deliver", "heavily shorted",
    "max pain", "short ladder", "naked short",
]

BULLISH_KEYWORDS = [
    "buy", "bull", "long", "calls", "moon", "undervalued", "breakout",
    "bullish", "upside", "accumulate", "strong buy", "beat earnings",
    "raised guidance", "record revenue", "buyback", "dividend increase",
    "all-time high", "ath", "catalyst", "beat", "blowout quarter",
]

BEARISH_KEYWORDS = [
    "sell", "bear", "short", "puts", "crash", "overvalued", "breakdown",
    "bearish", "downside", "miss", "missed earnings", "cut guidance",
    "layoffs", "fraud", "lawsuit", "downgrade", "debt", "default",
    "bankruptcy", "sec investigation", "restatement", "warning",
]

OPTIONS_KEYWORDS = [
    "call", "put", "strike", "expiry", "expiration", "iv crush", "iv spike",
    "open interest", "options chain", "leaps", "0dte", "weeklies",
    "delta", "theta", "gamma", "vega", "implied volatility",
]

DD_FLAIRS = {"dd", "due diligence", "analysis", "research", "technical analysis", "fundamentals"}

SECTOR_KEYWORDS = {
    "Semiconductors":   ["semiconductor", "chip", "foundry", "wafer", "tsmc", "amd", "nvidia", "intel", "qualcomm", "asml", "arm", "smc", "micron"],
    "AI":               ["artificial intelligence", "llm", "machine learning", "deep learning", "ai stocks", "openai", "anthropic", "copilot", "gemini", "grok"],
    "Robotics":         ["robot", "robotics", "automation", "autonomous", "humanoid", "boston dynamics", "figure ai"],
    "Defense":          ["defense", "military", "nato", "pentagon", "lockheed", "raytheon", "missile", "wartime", "geopolit", "weapons"],
    "Biotech":          ["biotech", "fda", "clinical trial", "drug approval", "biopharma", "oncology", "phase 3", "phase 2", "catalyst", "readout"],
    "Energy":           ["oil", "energy stocks", "crude", "lng", "solar", "renewables", "drilling", "opec", "natural gas", "uranium", "nuclear"],
    "Financials":       ["bank", "interest rate", "fed", "federal reserve", "credit", "fintech", "insurance", "rate cut", "rate hike", "yield curve"],
    "Crypto":           ["bitcoin", "btc", "ethereum", "defi", "crypto", "blockchain", "coinbase", "solana", "altcoin", "web3"],
    "Consumer":         ["retail", "consumer spending", "e-commerce", "amazon", "shopify", "consumer confidence", "holiday sales"],
    "Healthcare":       ["healthcare", "hospital", "pharma", "drug", "vaccine", "medical device", "insurance", "medicare", "medicaid"],
    "Real Estate":      ["reit", "real estate", "mortgage", "housing", "commercial real estate", "multifamily", "office vacancy"],
    "Macro":            ["fed", "inflation", "cpi", "recession", "gdp", "unemployment", "tariff", "trade war", "dollar", "treasury"],
}

EXCLUDE_TOKENS = {
    "A", "I", "AM", "PM", "US", "UK", "EU", "DD", "OP", "YOY", "IMO",
    "EOD", "CEO", "CFO", "ETF", "EPS", "FYI", "SEC", "IPO", "ATH", "ATL",
    "TA", "TD", "IV", "OI", "WSB", "SPY", "SPX", "VIX", "GDP", "CPI",
    "NOT", "THE", "AND", "OR", "BUT", "FOR", "ARE", "WAS", "HAS", "HAD",
    "BE", "TO", "IN", "IS", "IT", "OF", "ON", "AT", "BY", "DO", "IF",
    "NO", "SO", "UP", "AN", "AS", "MY", "HE", "WE", "GO", "ME", "ALL",
    "ITS", "CAN", "NOW", "GET", "GOT", "USE", "NEW", "ONE", "TWO",
    "THEY", "THEIR", "THAT", "THIS", "WITH", "FROM", "HAVE", "BEEN",
    "WHEN", "WHAT", "WILL", "MORE", "VERY", "JUST", "ALSO", "ONLY",
    "OUT", "TOO", "HOW", "WHY", "WHO", "ANY", "IRS", "PNL", "ROI",
    "AH", "RSI", "MACD", "PE", "PB", "EV", "DCF", "FCF",
    "YTD", "QOQ", "MOM", "CAGR", "APY", "APR", "NAV", "AUM",
    "OH", "OK", "LOL", "OMG", "TBH", "NGL", "FWIW", "AFAIK",
    "DCA", "YOLO", "FOMO", "HODL", "RIP", "TLDR", "TL",
}


def extract_tickers(text: str) -> list[str]:
    dollar = re.findall(r'\$([A-Z]{1,5})\b', text)
    bare = re.findall(r'(?<![A-Z\$])([A-Z]{2,5})(?![A-Z])', text)
    combined = dollar + [t for t in bare if t not in EXCLUDE_TOKENS and len(t) >= 2]
    return list(dict.fromkeys(combined))


def _sentiment(text: str) -> float:
    lower = text.lower()
    bull = sum(1 for k in BULLISH_KEYWORDS if k in lower)
    bear = sum(1 for k in BEARISH_KEYWORDS if k in lower)
    total = bull + bear
    return round((bull - bear) / total, 3) if total else 0.0


def _options_flag(text: str) -> bool:
    lower = text.lower()
    return sum(1 for k in OPTIONS_KEYWORDS if k in lower) >= 2


def _is_dd(post: dict) -> bool:
    flair = (post.get("flair") or "").lower()
    title_lower = post.get("title", "").lower()
    return any(d in flair or d in title_lower for d in DD_FLAIRS)


def _fetch_comments(post_id: str, subreddit: str, max_comments: int = 30) -> list[str]:
    """Fetch top-level comment texts from a single post."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=50&sort=top"
        r = requests.get(url, headers=HEADERS, timeout=20)
        data = r.json()
        comments = []
        if len(data) < 2:
            return []
        for child in data[1].get("data", {}).get("children", [])[:max_comments]:
            body = child.get("data", {}).get("body", "")
            score = child.get("data", {}).get("score", 0)
            if body and score > 5:  # only upvoted comments
                comments.append(body)
        return comments
    except Exception:
        return []


def fetch_wsb_posts(fetch_comment_depth: bool = True) -> list[dict]:
    """
    Fetch posts from multiple retail subreddits.
    For high-value posts (score > 500), also fetch top comments for richer signal.
    """
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
                if not post_id or post_id in seen_ids:
                    continue
                seen_ids.add(post_id)
                posts.append({
                    "id":           post_id,
                    "title":        p.get("title", ""),
                    "score":        int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "text":         p.get("selftext", ""),
                    "flair":        p.get("link_flair_text", ""),
                    "awards":       int(p.get("total_awards_received", 0)),
                    "subreddit":    subreddit,
                    "comments":     [],
                })
        except Exception:
            pass

    if not fetch_comment_depth:
        return posts

    # Fetch comments for high-signal posts: score > 500 OR DD flair OR many awards
    priority_posts = [
        p for p in posts
        if p["score"] > 500 or p["awards"] > 0 or _is_dd(p)
    ][:25]  # cap at 25 to avoid rate limits

    for p in priority_posts:
        comments = _fetch_comments(p["id"], p["subreddit"])
        p["comments"] = comments
        time.sleep(0.5)  # polite rate limiting

    return posts


def analyze_wsb_signals(posts: list[dict]) -> dict:
    """
    Aggregate weighted ticker signals with sentiment, options flow, squeeze,
    and DD extraction. Mention weight = log10(upvotes) so viral posts dominate.
    """
    weighted_mentions:  dict[str, float]       = defaultdict(float)
    sentiment_accum:    dict[str, list[float]] = defaultdict(list)
    options_tickers:    set[str]               = set()
    squeeze_posts:      dict[str, list[str]]   = defaultdict(list)
    dd_posts:           list[dict]             = []

    for post in posts:
        full_text = post["title"] + " " + post.get("text", "")
        comment_text = " ".join(post.get("comments", []))
        combined = full_text + " " + comment_text

        # Weight: upvotes (log) + awards boost + comment depth boost
        weight = (
            1.0 + math.log10(max(post.get("score", 1), 1))
            + post.get("awards", 0) * 0.5
            + math.log10(max(post.get("num_comments", 1), 1)) * 0.3
        )

        tickers = extract_tickers(combined)
        sent = _sentiment(combined)

        for t in tickers:
            weighted_mentions[t] += weight
            sentiment_accum[t].append(sent)

        lower = combined.lower()

        if any(k in lower for k in SQUEEZE_KEYWORDS):
            for t in tickers:
                squeeze_posts[t].append(post["title"])

        if _options_flag(combined):
            options_tickers.update(tickers)

        if _is_dd(post) and post.get("score", 0) > 100:
            dd_posts.append({
                "title":     post["title"],
                "score":     post["score"],
                "tickers":   tickers[:5],
                "sentiment": round(sent, 3),
                "subreddit": post.get("subreddit", ""),
                "summary":   (post.get("text", "") or "")[:300].strip(),
            })

    # Build per-ticker dict
    ticker_mentions: dict[str, dict] = {}
    for ticker, w in sorted(weighted_mentions.items(), key=lambda x: -x[1])[:60]:
        sents = sentiment_accum[ticker]
        ticker_mentions[ticker] = {
            "mentions_7d":    round(w, 1),
            "squeeze_flag":   ticker in squeeze_posts,
            "options_active": ticker in options_tickers,
            "sentiment":      round(sum(sents) / len(sents), 3) if sents else 0.0,
        }

    # Sector hype across all text
    all_text_lower = " ".join(
        p["title"].lower() + " " + p.get("text", "").lower() + " " + " ".join(p.get("comments", []))
        for p in posts
    )
    sector_scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        count = sum(all_text_lower.count(k) for k in keywords)
        score = min(1.0, count / 40.0)
        if score > 0.05:
            sector_scores[sector] = round(score, 3)

    trending = [{"ticker": t, **v} for t, v in list(ticker_mentions.items())[:30]]

    return {
        "trending_tickers":   trending,
        "ticker_mentions":    ticker_mentions,
        "dd_posts":           sorted(dd_posts, key=lambda x: -x["score"])[:10],
        "sector_hype": [
            {"sector": s, "score": sc}
            for s, sc in sorted(sector_scores.items(), key=lambda x: -x[1])
        ],
        "squeeze_candidates":  list(squeeze_posts.keys())[:10],
        "options_flow_tickers": list(options_tickers)[:20],
        "total_posts_analyzed": len(posts),
        "subreddits": list({p["subreddit"] for p in posts}),
    }
