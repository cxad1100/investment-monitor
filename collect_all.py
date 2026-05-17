"""
Master data collection script.
Runs all 13 collectors in sequence and produces data/signals.json.
No Claude API required. Runtime: ~20-30 minutes for full universe.

Usage:
  python collect_all.py
  python collect_all.py --fast  (skips per-ticker collectors for quick test)
"""

import argparse
import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

from config import FRED_API_KEY
from tools.universe_manager import refresh_universe, load_universe
from tools.fred_tools import fetch_fred_series, classify_regime
from tools.futures_tools import fetch_futures_signal
from tools.polymarket_tools import fetch_all_investment_markets
from tools.gdelt_tools import fetch_regional_conflict_indices
from tools.news_tools import fetch_news_headlines
from tools.yfinance_tools import fetch_price_history, fetch_fundamentals
from tools.insider_tools import fetch_insider_transactions
from tools.options_tools import fetch_options_signal
from tools.short_interest_tools import fetch_short_interest
from tools.wsb_tools import fetch_wsb_posts, analyze_wsb_signals
from tools.btc_tools import fetch_btc_signal
from tools.sentiment_tools import fetch_market_sentiment
from tools.news_polymarket_bridge import build_news_market_summary
from tools.wsb_trend import compute_retail_trend
from tools.macro_extended_tools import (
    fetch_bond_yields, fetch_currencies,
    fetch_sector_etf_performance, fetch_extended_commodities,
)
from fast_scorer import score_all_assets


def build_universe_map(universe: list[dict]) -> dict:
    return {row["yf_ticker"]: row for row in universe}


def build_earnings_catalog(fundamentals: dict, poly_earnings: list[dict]) -> dict:
    """Merge yfinance earnings data with Polymarket earnings markets."""
    catalog = {}
    for ticker, fund in fundamentals.items():
        catalog[ticker] = {
            "beat_probability": 0.5,
            "next_earnings_date": fund.get("next_earnings_date"),
            "consensus_eps": fund.get("eps"),
        }
    for market in poly_earnings:
        q = market["question"].upper()
        for ticker in catalog:
            if ticker in q or ticker.replace(".DE", "").replace(".PA", "") in q:
                catalog[ticker]["beat_probability"] = market["probability"]
                catalog[ticker]["polymarket_question"] = market["question"]
    return catalog


def compute_price_stats(price_data: dict) -> dict:
    returns = [v.get("return_1y", 0) for v in price_data.values() if "error" not in v]
    return {
        "avg_return_1y": round(float(np.mean(returns)), 2) if returns else 0.0,
        "median_return_1y": round(float(np.median(returns)), 2) if returns else 0.0,
    }


def collect(fast: bool = False) -> dict:
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        print("ERROR: FRED_API_KEY not set")
        sys.exit(1)

    Path("data").mkdir(exist_ok=True)
    print("\n" + "="*55)
    print("  SIGNAL COLLECTION")
    print("="*55)

    print("\n[1/9] Refreshing universe...")
    refresh_universe(validate=not fast)
    universe = load_universe()
    tickers = [row["yf_ticker"] for row in universe]
    print(f"      {len(tickers)} assets loaded")

    print("[2/9] FRED macro indicators...")
    fred_data = fetch_fred_series()
    macro_regime = classify_regime(fred_data)

    print("[3/9] Commodity futures...")
    futures_signal = fetch_futures_signal()
    macro_signal = {
        "regime": macro_regime.get("regime", "growth"),
        "risk_level": macro_regime.get("risk_level", "medium"),
        "sector_tailwinds": list(set(futures_signal.get("sector_tailwinds", []))),
        "sector_headwinds": list(set(futures_signal.get("sector_headwinds", []))),
        "futures_summary": futures_signal.get("summary", ""),
        "fred_indicators": fred_data,
    }

    print("[4/9] Polymarket prediction markets...")
    poly_data = fetch_all_investment_markets()
    poly_geo = poly_data["geo"]
    poly_macro = poly_data["macro"]
    poly_company = poly_data["company"]
    print(f"      {len(poly_geo)} geo · {len(poly_macro)} macro · {len(poly_company)} company markets")

    print("[5/9] GDELT regional conflict indices...")
    gdelt_data = fetch_regional_conflict_indices()

    print("[6/9] News RSS headlines...")
    news_data = fetch_news_headlines()

    print("[7/9] Price history + fundamentals...")
    sample_tickers = tickers[:50] if fast else tickers
    price_data = fetch_price_history(sample_tickers, period="1y")
    fundamentals = fetch_fundamentals(sample_tickers)
    price_stats = compute_price_stats(price_data)
    earnings_catalog = build_earnings_catalog(fundamentals, poly_earnings)

    print("[8/9] Insider flow, options, short interest...")
    rated_tickers = list({t for t in sample_tickers if t in price_data and t in fundamentals})[:200]
    insider_data = fetch_insider_transactions(rated_tickers)
    options_data = fetch_options_signal(rated_tickers)
    short_data = fetch_short_interest(rated_tickers)

    print("[9/9] WSB Reddit + BTC signal...")
    wsb_posts = fetch_wsb_posts()
    wsb_signal = analyze_wsb_signals(wsb_posts)
    btc_signal = fetch_btc_signal()

    print("[10/13] Market sentiment (VIX, Fear/Greed, credit spreads)...")
    sentiment = fetch_market_sentiment()

    print("[11/13] Bond yields + yield curve...")
    bond_yields = fetch_bond_yields()
    print(f"      {bond_yields.get('summary', 'N/A')}")

    print("[12/13] Currencies + sector ETF performance...")
    currencies = fetch_currencies()
    sector_etfs = fetch_sector_etf_performance()
    leaders = sector_etfs.get("leaders_1m", [])
    laggards = sector_etfs.get("laggards_1m", [])
    print(f"      ETF leaders 1M: {', '.join(leaders)} | laggards: {', '.join(laggards)}")

    print("[13/13] Extended commodities (silver, wheat, corn, soybeans)...")
    commodities_ext = fetch_extended_commodities()
    print(f"      {len(commodities_ext)} commodity signals collected")

    universe_map = build_universe_map(universe)

    signals = {
        "collected_at": datetime.now().isoformat(),
        "universe_size": len(tickers),
        "rated_ticker_count": len(rated_tickers),
        "macro": macro_signal,
        "polymarket_geo": poly_geo,
        "polymarket_macro": poly_macro,
        "polymarket_company": poly_company,
        "gdelt": gdelt_data,
        "news": news_data,
        "price_data": price_data,
        "price_stats": price_stats,
        "fundamentals": fundamentals,
        "earnings": earnings_catalog,
        "insider": insider_data,
        "options": options_data,
        "short_interest": short_data,
        "wsb": wsb_signal,
        "btc": btc_signal,
        "sentiment": sentiment,
        "bond_yields": bond_yields,
        "currencies": currencies,
        "sector_etfs": sector_etfs,
        "commodities_ext": commodities_ext,
        "news_market_bridge": news_market_summary,
        "retail_trend": retail_trend,
        "universe_map": {k: {"sector": v.get("sector", ""), "region": v.get("region", ""),
                             "type": v.get("type", "stock"), "name": v.get("name", "")}
                         for k, v in universe_map.items()},
        "events": [],
    }

    vix_val = sentiment.get("vix", {}).get("vix", "?")
    fg_score = sentiment.get("fear_greed", {}).get("score", "?")
    fg_rating = sentiment.get("fear_greed", {}).get("rating", "?")
    print(f"      VIX {vix_val} · Fear/Greed {fg_score} ({fg_rating}) · credit spreads {sentiment.get('credit_spreads', {}).get('spread_regime', '?')}")

    # Cross-reference: news → markets + WSB trend
    all_poly_markets = poly_geo + poly_macro + poly_company
    headlines = news_data.get("headlines", [])
    news_market_summary = build_news_market_summary(headlines, all_poly_markets)
    print(f"      News-market bridge: {news_market_summary['coverage_pct']}% of markets have news support")
    print(f"      Top news topics: {[t for t, _ in news_market_summary['top_topics'][:5]]}")

    retail_trend = compute_retail_trend(wsb_signal, universe_map, short_data)
    print(f"      Retail trend ({retail_trend['trend_direction']}): {retail_trend['narrative'][:80]}")

    print("\n[fast-scorer] Computing composite scores...")
    fast_scores = score_all_assets(signals)
    signals["fast_scores"] = fast_scores
    high_priority = [t for t, s in fast_scores.items() if s["score"] >= 70]
    print(f"[fast-scorer] {len(fast_scores)} scored, {len(high_priority)} flagged for deep rating")

    out_path = "data/signals.json"
    with open(out_path, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    print(f"\n[collect_all] signals.json saved")
    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all signals for weekly rating")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 50 tickers, skip validation")
    args = parser.parse_args()
    collect(fast=args.fast)
