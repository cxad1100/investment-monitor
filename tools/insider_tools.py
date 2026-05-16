"""SEC EDGAR Form 4 — net insider buying/selling per company."""

import requests
from datetime import datetime, timedelta

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
HEADERS = {"User-Agent": "investment-research-bot admin@example.com"}


def fetch_insider_transactions(tickers: list[str], days_back: int = 90) -> dict[str, dict]:
    """Fetch Form 4 filings for each ticker from SEC EDGAR."""
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    results = {}

    for ticker in tickers:
        try:
            params = {
                "q": f'"{ticker}"',
                "dateRange": "custom",
                "startdt": start,
                "forms": "4",
                "_source": "filing",
            }
            resp = requests.get(EDGAR_SEARCH, params=params, headers=HEADERS, timeout=20)
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            buy_count = sum(1 for h in hits if "purchase" in str(h).lower())
            sell_count = sum(1 for h in hits if "sale" in str(h).lower())
            total = len(hits)

            if total == 0:
                signal = "neutral"
                net_buy_pct = 0.0
            elif buy_count > sell_count * 2:
                signal = "strong_buy"
                net_buy_pct = 0.8
            elif buy_count > sell_count:
                signal = "buy"
                net_buy_pct = 0.4
            elif sell_count > buy_count * 2:
                signal = "strong_sell"
                net_buy_pct = -0.8
            elif sell_count > buy_count:
                signal = "sell"
                net_buy_pct = -0.4
            else:
                signal = "neutral"
                net_buy_pct = 0.0

            results[ticker] = {
                "buy_filings": buy_count,
                "sell_filings": sell_count,
                "total_filings": total,
                "net_buy_pct_mktcap": net_buy_pct,
                "signal": signal,
                "days_back": days_back,
            }
        except Exception as e:
            results[ticker] = {"net_buy_pct_mktcap": 0.0, "signal": "neutral", "error": str(e)}

    return results
