"""Reddit r/wallstreetbets public JSON — ticker mentions, squeeze detection, sector hype."""

import re
from collections import Counter
import requests

WSB_URLS = [
    "https://www.reddit.com/r/wallstreetbets/hot.json?limit=100",
    "https://www.reddit.com/r/wallstreetbets/top.json?t=week&limit=100",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

SQUEEZE_KEYWORDS = ["squeeze", "short interest", "gamma squeeze", "short float",
                     "days to cover", "si%", "ftd", "fail to deliver"]

SECTOR_KEYWORDS = {
    "Semiconductors": ["semiconductor", "chip", "foundry", "wafer", "tsmc", "amd", "nvidia", "intel", "qualcomm", "asml"],
    "Photonics": ["photonics", "laser", "lidar", "optical", "fiber optic", "photonic"],
    "Robotics": ["robot", "robotics", "automation", "autonomous", "humanoid"],
    "AI": ["artificial intelligence", "llm", "machine learning", "deep learning", "neural network", "ai stocks"],
    "Defense": ["defense", "military", "nato", "pentagon", "lockheed", "raytheon", "missile", "wartime"],
    "Biotech": ["biotech", "fda", "clinical trial", "drug approval", "biopharma", "oncology"],
    "Energy": ["oil", "energy stocks", "crude", "lng", "solar", "renewables", "drilling"],
    "Crypto": ["bitcoin", "btc", "ethereum", "defi", "crypto", "blockchain", "coinbase"],
}

EXCLUDE_TOKENS = {
    "A", "I", "AM", "PM", "US", "UK", "EU", "DD", "OP", "YOY", "IMO",
    "EOD", "CEO", "ETF", "EPS", "FYI", "SEC", "IPO", "ATH", "ATL",
    "TA", "TD", "IV", "OI", "WSB", "SPY", "SPX", "VIX", "GDP", "CPI",
    "NOT", "THE", "AND", "OR", "BUT", "FOR", "ARE", "WAS", "HAS", "HAD",
    "BE", "TO", "IN", "IS", "IT", "OF", "ON", "AT", "BY", "DO", "IF",
    "NO", "SO", "UP", "AN", "AS", "MY", "HE", "WE", "GO", "ME",
}


def extract_tickers_from_text(text: str) -> list[str]:
    """Extract $TICKER patterns from text, filtering common false positives."""
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    return [t for t in dollar_tickers if t not in EXCLUDE_TOKENS]


def fetch_wsb_posts() -> list[dict]:
    """Fetch recent posts from r/wallstreetbets via public JSON endpoint."""
    posts = []
    for url in WSB_URLS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                p = child.get("data", {})
                posts.append({
                    "title": p.get("title", ""),
                    "score": int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "text": p.get("selftext", ""),
                })
        except Exception:
            pass
    return posts


def analyze_wsb_signals(posts: list[dict]) -> dict:
    """Extract trending tickers, squeeze candidates, and sector hype from posts."""
    all_tickers = []
    squeeze_posts: dict[str, list[str]] = {}

    for post in posts:
        full_text = post["title"] + " " + post.get("text", "")
        tickers = extract_tickers_from_text(full_text)
        all_tickers.extend(tickers)
        text_lower = full_text.lower()
        if any(k in text_lower for k in SQUEEZE_KEYWORDS):
            for t in tickers:
                if t not in squeeze_posts:
                    squeeze_posts[t] = []
                squeeze_posts[t].append(post["title"])

    ticker_counts = Counter(all_tickers)
    squeeze_candidates = list(squeeze_posts.keys())

    all_text_lower = " ".join(p["title"].lower() + " " + p.get("text", "").lower() for p in posts)
    sector_scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        count = sum(all_text_lower.count(k) for k in keywords)
        score = min(1.0, count / 30.0)
        if score > 0.05:
            sector_scores[sector] = round(score, 3)

    trending = [
        {
            "ticker": ticker,
            "mentions_7d": count,
            "squeeze_flag": ticker in squeeze_candidates,
        }
        for ticker, count in ticker_counts.most_common(30)
    ]

    return {
        "trending_tickers": trending,
        "sector_hype": [
            {"sector": s, "score": score}
            for s, score in sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        ],
        "squeeze_candidates": squeeze_candidates[:10],
        "total_posts_analyzed": len(posts),
    }
