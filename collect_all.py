"""
Master data collection script.
Runs all collectors in sequence and produces data/signals.json.
No Claude API required. Runtime: ~20-30 minutes for full universe.

Usage:
  python collect_all.py
  python collect_all.py --fast  (50 tickers, skip validation, ~5 min)
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
from tools.theme_detector import score_themes
from tools.macro_extended_tools import (
    fetch_bond_yields, fetch_currencies,
    fetch_sector_etf_performance, fetch_extended_commodities,
)
from fast_scorer import score_all_assets


def build_universe_map(universe: list[dict]) -> dict:
    return {row["yf_ticker"]: row for row in universe}


def build_earnings_catalog(fundamentals: dict, poly_markets: list[dict]) -> dict:
    """Merge yfinance earnings data with Polymarket earnings markets."""
    catalog = {}
    for ticker, fund in fundamentals.items():
        catalog[ticker] = {
            "beat_probability": 0.5,
            "next_earnings_date": fund.get("next_earnings_date"),
            "consensus_eps": fund.get("eps"),
        }
    for market in poly_markets:
        q = market["question"].upper()
        for ticker in catalog:
            if ticker in q or ticker.replace(".DE", "").replace(".PA", "").replace(".L", "") in q:
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

    # ── 1. Universe ──────────────────────────────────────────────────────────
    print("\n[1/13] Refreshing universe...")
    refresh_universe(validate=not fast)
    universe = load_universe()
    universe_map = build_universe_map(universe)
    tickers = [row["yf_ticker"] for row in universe]
    print(f"       {len(tickers)} assets loaded")

    # ── 2. FRED macro ────────────────────────────────────────────────────────
    print("[2/13] FRED macro indicators...")
    fred_data = fetch_fred_series()
    macro_regime = classify_regime(fred_data)

    # ── 3. Commodity futures ─────────────────────────────────────────────────
    print("[3/13] Commodity futures...")
    futures_signal = fetch_futures_signal()
    macro_signal = {
        "regime": macro_regime.get("regime", "growth"),
        "risk_level": macro_regime.get("risk_level", "medium"),
        "sector_tailwinds": list(set(futures_signal.get("sector_tailwinds", []))),
        "sector_headwinds": list(set(futures_signal.get("sector_headwinds", []))),
        "futures_summary": futures_signal.get("summary", ""),
        "fred_indicators": fred_data,
    }

    # ── 4. Polymarket ────────────────────────────────────────────────────────
    print("[4/13] Polymarket prediction markets...")
    poly_data = fetch_all_investment_markets()
    poly_geo     = poly_data["geo"]
    poly_macro   = poly_data["macro"]
    poly_company = poly_data["company"]
    all_poly     = poly_geo + poly_macro + poly_company
    print(f"       {len(poly_geo)} geo · {len(poly_macro)} macro · {len(poly_company)} company markets")

    # ── 5. GDELT ─────────────────────────────────────────────────────────────
    print("[5/13] GDELT regional conflict indices...")
    gdelt_data = fetch_regional_conflict_indices()

    # ── 6. News ──────────────────────────────────────────────────────────────
    print("[6/13] News RSS headlines...")
    news_data = fetch_news_headlines()
    headlines = news_data.get("headlines", [])
    print(f"       {len(headlines)} headlines collected")

    # ── 7. Price history + fundamentals ─────────────────────────────────────
    print("[7/13] Price history + fundamentals...")
    # Portfolio primary tickers always included so ratings show up in Portfolio tab
    PRIORITY = [
        "NVDA", "AMZN", "GOOGL", "TSM", "TSCO.L",
        "ISP.MI", "UCG.MI", "WBD.MI", "ASML.AS", "IWDA.AS",
    ]
    if fast:
        priority_set = [t for t in PRIORITY if t in tickers]
        remaining = [t for t in tickers if t not in priority_set]
        sample_tickers = priority_set + remaining[: max(0, 50 - len(priority_set))]
    else:
        sample_tickers = tickers
    price_data   = fetch_price_history(sample_tickers, period="1y")
    fundamentals = fetch_fundamentals(sample_tickers)
    price_stats  = compute_price_stats(price_data)
    earnings_catalog = build_earnings_catalog(fundamentals, all_poly)
    print(f"       {len(fundamentals)} tickers with fundamentals")

    # ── 8. Insider / options / short interest ────────────────────────────────
    print("[8/13] Insider flow, options, short interest...")
    rated_tickers = list({t for t in sample_tickers if t in price_data and t in fundamentals})[:200]
    insider_data = fetch_insider_transactions(rated_tickers)
    options_data = fetch_options_signal(rated_tickers)
    short_data   = fetch_short_interest(rated_tickers)
    print(f"       {len(rated_tickers)} tickers rated")

    # ── 9. WSB + BTC ─────────────────────────────────────────────────────────
    print("[9/13] WSB Reddit + BTC signal...")
    wsb_posts  = fetch_wsb_posts()
    wsb_signal = analyze_wsb_signals(wsb_posts)
    btc_signal = fetch_btc_signal()

    # ── 10. Market sentiment ─────────────────────────────────────────────────
    print("[10/13] Market sentiment (VIX, Fear/Greed, credit spreads)...")
    sentiment = fetch_market_sentiment()
    vix_val   = sentiment.get("vix", {}).get("vix", "?")
    fg_score  = sentiment.get("fear_greed", {}).get("score", "?")
    fg_rating = sentiment.get("fear_greed", {}).get("rating", "?")
    print(f"       VIX {vix_val} · Fear/Greed {fg_score} ({fg_rating})")

    # ── 11. Bond yields + yield curve ────────────────────────────────────────
    print("[11/13] Bond yields + yield curve...")
    bond_yields = fetch_bond_yields()
    print(f"       {bond_yields.get('summary', 'N/A')}")

    # ── 12. Currencies + sector ETFs ─────────────────────────────────────────
    print("[12/13] Currencies + sector ETF performance...")
    currencies  = fetch_currencies()
    sector_etfs = fetch_sector_etf_performance()
    leaders  = sector_etfs.get("leaders_1m", [])
    laggards = sector_etfs.get("laggards_1m", [])
    print(f"       ETF leaders 1M: {', '.join(leaders)} | laggards: {', '.join(laggards)}")

    # ── 13. Extended commodities ─────────────────────────────────────────────
    print("[13/13] Extended commodities...")
    commodities_ext = fetch_extended_commodities()
    print(f"       {len(commodities_ext)} commodity signals")

    # ── Post-processing ──────────────────────────────────────────────────────
    print("\n[post] News-market bridge...")
    news_market_summary = build_news_market_summary(headlines, all_poly)
    print(f"       {news_market_summary['coverage_pct']}% of markets have news support")

    print("[post] Retail trend (WSB)...")
    retail_trend = compute_retail_trend(wsb_signal, universe_map, short_data)
    print(f"       {retail_trend['trend_direction']} — {retail_trend['narrative'][:70]}")

    # ── Build signals dict ───────────────────────────────────────────────────
    signals = {
        "collected_at":      datetime.now().isoformat(),
        "universe_size":     len(tickers),
        "rated_ticker_count": len(rated_tickers),
        "universe_map": {
            k: {
                "sector": v.get("sector", ""),
                "region": v.get("region", ""),
                "type":   v.get("type", "stock"),
                "name":   v.get("name", ""),
            }
            for k, v in universe_map.items()
        },
        "macro":           macro_signal,
        "polymarket_geo":  poly_geo,
        "polymarket_macro": poly_macro,
        "polymarket_company": poly_company,
        "gdelt":           gdelt_data,
        "news":            news_data,
        "price_data":      price_data,
        "price_stats":     price_stats,
        "fundamentals":    fundamentals,
        "earnings":        earnings_catalog,
        "insider":         insider_data,
        "options":         options_data,
        "short_interest":  short_data,
        "wsb":             wsb_signal,
        "btc":             btc_signal,
        "sentiment":       sentiment,
        "bond_yields":     bond_yields,
        "currencies":      currencies,
        "sector_etfs":     sector_etfs,
        "commodities_ext": commodities_ext,
        "news_market_bridge": news_market_summary,
        "retail_trend":    retail_trend,
        "events":          [],
        "themes":          [],
    }

    # ── Theme scoring (needs signals dict) ───────────────────────────────────
    print("[post] Scoring cross-source themes...")
    themes = score_themes(signals)
    signals["themes"] = themes
    top3 = [f"{t['label']} ({t['composite']:.0f})" for t in themes[:3]]
    print(f"       Top themes: {' | '.join(top3)}")

    # ── Fast scorer ──────────────────────────────────────────────────────────
    print("\n[fast-scorer] Computing composite scores...")
    fast_scores = score_all_assets(signals)
    signals["fast_scores"] = fast_scores
    high_priority = [t for t, s in fast_scores.items() if s["score"] >= 70]
    print(f"[fast-scorer] {len(fast_scores)} scored · {len(high_priority)} flagged for deep rating")

    # ── Save signals ──────────────────────────────────────────────────────────
    out_path = "data/signals.json"
    with open(out_path, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    print(f"\n[collect_all] Saved {out_path}")

    # ── Pre-compute portfolio analytics (saves dashboard load time) ───────────
    portfolio_path = Path("input/portfolio.csv")
    analytics_path = Path("data/portfolio_analytics_cache.json")

    # In fast mode skip if cache is less than 6 hours old
    skip_analytics = False
    if fast and analytics_path.exists():
        try:
            age_hours = (datetime.now() - datetime.fromtimestamp(analytics_path.stat().st_mtime)).total_seconds() / 3600
            if age_hours < 6:
                print(f"[portfolio] Analytics cache {age_hours:.1f}h old — skipping recompute (fast mode)")
                skip_analytics = True
        except Exception:
            pass

    if portfolio_path.exists() and not skip_analytics:
        print("[portfolio] Pre-computing ROI series, benchmarks, quant metrics...")
        try:
            from tools.portfolio_tools import (
                parse_portfolio, fetch_current_prices, compute_portfolio_summary,
            )
            from tools.portfolio_analytics import (
                build_roi_timeseries, compute_quant_metrics,
                compute_position_technicals, compute_concentration_metrics,
                compute_correlation_matrix,
            )
            from tools.portfolio_tools import TICKER_MAP as _TMAP

            portfolio = parse_portfolio(portfolio_path)
            prices    = fetch_current_prices(portfolio["holdings"])
            summary   = compute_portfolio_summary(portfolio, prices)

            port_series, bm_series = build_roi_timeseries(portfolio["transactions"])
            sp500_s  = bm_series.get("S&P 500")
            metrics  = compute_quant_metrics(port_series, sp500_s)
            tech     = compute_position_technicals(portfolio["holdings"], _TMAP)
            conc     = compute_concentration_metrics(summary["positions"])
            corr     = compute_correlation_matrix(portfolio["holdings"], _TMAP)

            analytics = {
                "computed_at": datetime.now().isoformat(),
                "summary": {
                    "totals":          summary["totals"],
                    "positions":       summary["positions"],
                    "realized_detail": summary["realized_detail"],
                },
                "portfolio_roi":  {str(k.date()): v for k, v in port_series.items()},
                "benchmark_roi":  {
                    name: {str(k.date()): v for k, v in s.items()}
                    for name, s in bm_series.items()
                },
                "quant_metrics":  metrics,
                "technicals":     tech,
                "concentration":  conc,
                "correlation":    corr,
            }
            with open("data/portfolio_analytics_cache.json", "w") as f:
                json.dump(analytics, f, indent=2, default=str)
            print(f"[portfolio] Analytics cached. ROI={port_series.iloc[-1]:+.2f}% "
                  f"Sharpe={metrics.get('sharpe','?')}")
        except Exception as e:
            print(f"[portfolio] Analytics failed: {e}")

    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="50 tickers, skip validation (~5 min)")
    args = parser.parse_args()
    collect(fast=args.fast)
