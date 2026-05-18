"""RSS news feeds — finance, macro, sector, and regulatory headlines."""

import feedparser

RSS_FEEDS = {
    # Major finance / markets
    "yahoo_finance":        "https://finance.yahoo.com/rss/topstories",
    "marketwatch":          "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "cnbc_markets":         "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "cnbc_earnings":        "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    "reuters_business":     "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets":      "https://feeds.reuters.com/reuters/USmarket",
    "wsj_markets":          "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "financial_times":      "https://www.ft.com/rss/home",
    "barrons":              "https://www.barrons.com/xml/rss/3_7510.xml",
    "thestreet":            "https://www.thestreet.com/rss/",
    "benzinga":             "https://www.benzinga.com/feed",
    "motley_fool":          "https://www.fool.com/feeds/index.aspx",
    "seeking_alpha":        "https://seekingalpha.com/feed.xml",
    "investopedia":         "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline",
    # Macro / economy
    "ap_business":          "https://rsshub.app/apnews/topics/business",
    "guardian_business":    "https://www.theguardian.com/business/rss",
    "economist":            "https://www.economist.com/finance-and-economics/rss.xml",
    "project_syndicate":    "https://www.project-syndicate.org/rss",
    # Regulatory / filings
    "sec_edgar_8k":         "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom&startdt=2020-01-01&forms=8-K&_source=filing&hits.hits._source=period_of_report,entity_name,file_date,form_type&hits.hits.total=true",
}

# Separate SEC EDGAR 8-K via the proper search API
SEC_8K_URL = "https://efts.sec.gov/LATEST/search-index?forms=8-K&dateRange=custom&startdt={today}&hits.hits.total=true"


def _fetch_sec_8k(max_items: int = 20) -> list[dict]:
    """Fetch recent 8-K filings (material events) from SEC EDGAR."""
    import requests
    from datetime import date, timedelta
    try:
        start = (date.today() - timedelta(days=3)).isoformat()
        url = (
            "https://efts.sec.gov/LATEST/search-index?forms=8-K"
            f"&dateRange=custom&startdt={start}"
            "&hits.hits._source=period_of_report,entity_name,file_date,form_type"
        )
        resp = requests.get(url, headers={"User-Agent": "investment-research contact@example.com"}, timeout=15)
        hits = resp.json().get("hits", {}).get("hits", [])
        results = []
        for h in hits[:max_items]:
            src = h.get("_source", {})
            results.append({
                "title":     f"8-K: {src.get('entity_name', '?')}",
                "summary":   f"Filed {src.get('file_date', '?')}",
                "published": src.get("file_date", ""),
                "source":    "sec_8k",
            })
        return results
    except Exception:
        return []


def fetch_news_headlines(max_per_feed: int = 20) -> dict:
    """Fetch recent headlines from finance RSS feeds + SEC 8-K filings."""
    all_headlines = []

    for feed_name, url in RSS_FEEDS.items():
        if feed_name == "sec_edgar_8k":
            continue  # handled separately
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                all_headlines.append({
                    "title":     title,
                    "summary":   entry.get("summary", "")[:300].strip(),
                    "published": str(entry.get("published", "")),
                    "source":    feed_name,
                })
        except Exception:
            pass

    all_headlines.extend(_fetch_sec_8k(max_items=15))

    return {
        "headlines": all_headlines,
        "total":     len(all_headlines),
    }
