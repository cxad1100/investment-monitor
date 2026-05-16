"""RSS news feeds — sector and company headline scraping."""

import feedparser

RSS_FEEDS = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_tech": "https://feeds.reuters.com/reuters/technologyNews",
}


def fetch_news_headlines(max_per_feed: int = 20) -> dict:
    """Fetch recent headlines from RSS feeds."""
    all_headlines = []
    for feed_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                all_headlines.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:200],
                    "published": str(entry.get("published", "")),
                    "source": feed_name,
                })
        except Exception:
            pass
    return {
        "headlines": all_headlines,
        "total": len(all_headlines),
    }
