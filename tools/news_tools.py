"""RSS news feeds — finance and sector headline scraping."""

import feedparser

RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/rss/topstories",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "financial_times": "https://www.ft.com/rss/home",
    "seeking_alpha": "https://seekingalpha.com/feed.xml",
}


def fetch_news_headlines(max_per_feed: int = 20) -> dict:
    """Fetch recent headlines from finance RSS feeds."""
    all_headlines = []
    for feed_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                all_headlines.append({
                    "title": title,
                    "summary": entry.get("summary", "")[:200].strip(),
                    "published": str(entry.get("published", "")),
                    "source": feed_name,
                })
        except Exception:
            pass
    return {
        "headlines": all_headlines,
        "total": len(all_headlines),
    }
