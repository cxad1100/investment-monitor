"""
Cross-reference: news headlines <-> Polymarket markets.

For each Polymarket market, find the most relevant recent headlines.
For each headline, find which markets it relates to.
This surfaces which events have current news support vs are priced in silence.
"""

import re
from collections import defaultdict

# Topic keyword maps: each topic has a list of terms to match in headlines/questions
TOPIC_KEYWORDS = {
    "iran":        ["iran", "iranian", "tehran", "hormuz", "iaea", "nuclear deal", "us-iran"],
    "russia":      ["russia", "russian", "kremlin", "putin", "moscow"],
    "ukraine":     ["ukraine", "ukrainian", "kyiv", "zelensky", "donbas"],
    "china":       ["china", "chinese", "beijing", "prc", "xi jinping", "ccp"],
    "taiwan":      ["taiwan", "taiwanese", "tsmc", "strait", "taipei"],
    "fed":         ["federal reserve", "fed", "fomc", "powell", "rate cut", "rate hike",
                    "interest rate", "basis point", "monetary policy"],
    "inflation":   ["inflation", "cpi", "pce", "price index", "consumer price"],
    "recession":   ["recession", "gdp", "contraction", "economic slowdown", "downturn"],
    "tariff":      ["tariff", "trade war", "trade deal", "import duty", "customs"],
    "bitcoin":     ["bitcoin", "btc", "crypto", "cryptocurrency", "coinbase", "blockchain"],
    "nvidia":      ["nvidia", "nvda", "gpu", "ai chip", "h100", "blackwell"],
    "apple":       ["apple", "aapl", "iphone", "tim cook", "app store"],
    "tesla":       ["tesla", "tsla", "elon musk", "ev", "electric vehicle"],
    "microsoft":   ["microsoft", "msft", "azure", "openai", "copilot"],
    "amazon":      ["amazon", "amzn", "aws", "bezos"],
    "google":      ["google", "alphabet", "googl", "goog", "deepmind", "gemini", "youtube"],
    "meta":        ["meta", "facebook", "instagram", "zuckerberg", "llama"],
    "ai":          ["artificial intelligence", "ai model", "llm", "chatgpt", "deepseek",
                    "machine learning", "generative ai"],
    "oil":         ["oil", "crude", "brent", "wti", "opec", "petroleum", "barrel"],
    "gold":        ["gold", "bullion", "precious metal", "safe haven"],
    "spacex":      ["spacex", "elon musk", "starship", "starlink", "rocket"],
    "ipo":         ["ipo", "initial public offering", "going public", "stock market debut"],
    "nato":        ["nato", "alliance", "defense pact", "collective defense"],
    "semiconductor": ["semiconductor", "chip", "tsmc", "foundry", "wafer", "advanced node"],
}


def _text_topics(text: str) -> set[str]:
    """Return which topics are mentioned in a text string."""
    t = text.lower()
    found = set()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(k in t for k in kws):
            found.add(topic)
    return found


def match_news_to_markets(headlines: list[dict], all_markets: list[dict]) -> dict[str, list[dict]]:
    """
    For each market (by id), find headlines that share topic keywords.
    Returns {market_id: [headline_dicts]}.
    """
    # Pre-compute topics for each market question
    market_topics = {}
    for m in all_markets:
        market_topics[m["id"]] = _text_topics(m["question"])

    # Pre-compute topics for each headline
    headline_topics = []
    for h in headlines:
        topics = _text_topics(h.get("title", "") + " " + h.get("summary", ""))
        headline_topics.append((h, topics))

    result = defaultdict(list)
    for m in all_markets:
        m_topics = market_topics[m["id"]]
        if not m_topics:
            continue
        for h, h_topics in headline_topics:
            shared = m_topics & h_topics
            if shared:
                result[m["id"]].append({
                    "title": h.get("title", ""),
                    "source": h.get("source", ""),
                    "published": h.get("published", "")[:16],
                    "shared_topics": sorted(shared),
                    "match_score": len(shared),
                })
        # Sort by match strength, take top 5
        result[m["id"]].sort(key=lambda x: -x["match_score"])
        result[m["id"]] = result[m["id"]][:5]

    return dict(result)


def find_markets_for_headline(headline: dict, all_markets: list[dict]) -> list[dict]:
    """Given a headline, find which markets it's most relevant to."""
    h_topics = _text_topics(headline.get("title", "") + " " + headline.get("summary", ""))
    if not h_topics:
        return []

    matches = []
    for m in all_markets:
        m_topics = _text_topics(m["question"])
        shared = h_topics & m_topics
        if shared:
            matches.append({
                **m,
                "shared_topics": sorted(shared),
                "match_score": len(shared),
            })

    return sorted(matches, key=lambda x: -x["match_score"])[:3]


def build_news_market_summary(headlines: list[dict], all_markets: list[dict]) -> dict:
    """
    Build a cross-reference summary:
    - Which markets have news support (headlines found)
    - Which markets are priced in silence (no news)
    - Top news-driven topics across all markets
    """
    news_to_market = match_news_to_markets(headlines, all_markets)

    markets_with_news = [m for m in all_markets if news_to_market.get(m["id"])]
    markets_silent   = [m for m in all_markets if not news_to_market.get(m["id"])]

    # Topic frequency across all matched headlines
    topic_freq: dict[str, int] = defaultdict(int)
    for news_list in news_to_market.values():
        for item in news_list:
            for t in item["shared_topics"]:
                topic_freq[t] += 1
    top_topics = sorted(topic_freq.items(), key=lambda x: -x[1])[:8]

    return {
        "market_news_map": news_to_market,
        "markets_with_news": [m["question"][:60] for m in markets_with_news[:10]],
        "markets_silent": [m["question"][:60] for m in markets_silent[:5]],
        "top_topics": top_topics,
        "coverage_pct": round(len(markets_with_news) / len(all_markets) * 100) if all_markets else 0,
    }
