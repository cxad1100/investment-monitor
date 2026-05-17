"""RSS news feeds — finance and sector headline scraping."""

import feedparser

RSS_FEEDS = {
    "yahoo_finance":     "https://finance.yahoo.com/rss/topstories",
    "marketwatch":       "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "financial_times":   "https://www.ft.com/rss/home",
    "seeking_alpha":     "https://seekingalpha.com/feed.xml",
    "cnbc_top":          "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "cnbc_finance":      "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "cnbc_earnings":     "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    "reuters_markets":   "https://feeds.reuters.com/reuters/businessNews",
    "wsj_markets":       "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "motley_fool":       "https://www.fool.com/feeds/index.aspx",
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
