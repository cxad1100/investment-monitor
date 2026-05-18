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
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from config import FRED_API_KEY
from tools.universe_manager import refresh_universe, load_universe
from tools.fred_tools import fetch_fred_series, classify_regime
from tools.futures_tools import fetch_futures_signal
from tools.polymarket_tools import fetch_all_investment_markets
from tools.gdelt_tools import fetch_regional_conflict_indices
from tools.news_tools import fetch_news_headlines, RSS_FEEDS
from tools.yfinance_tools import fetch_price_history, fetch_fundamentals
from tools.insider_tools import fetch_insider_transactions
from tools.options_tools import fetch_options_signal
from tools.short_interest_tools import fetch_short_interest
from tools.wsb_tools import fetch_wsb_posts, analyze_wsb_signals, SUBREDDIT_URLS
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

_T0 = time.time()


def _elapsed() -> str:
    return f"{time.time() - _T0:5.1f}s"


def _hdr(n: str, title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  [{n}] {title}")
    print(f"{'─'*55}")


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def _stat(label: str, val, total=None, unit: str = "") -> None:
    if total:
        pct = val / total * 100 if total > 0 else 0
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  {label:<28} {val:>5}/{total:<5} [{bar}] {pct:5.1f}%{' '+unit if unit else ''}")
    else:
        print(f"  {label:<28} {val}{' '+unit if unit else ''}")


def build_universe_map(universe: list[dict]) -> dict:
    return {row["yf_ticker"]: row for row in universe}


def build_earnings_catalog(fundamentals: dict, poly_markets: list[dict]) -> dict:
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
    mode = "FAST (50 tickers)" if fast else "FULL"
    print(f"\n{'═'*55}")
    print(f"  SIGNAL COLLECTION  [{mode}]  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*55}")

    # ── 1. Universe ──────────────────────────────────────────────────────────
    _hdr("1/13", "Universe")
    t = time.time()
    refresh_universe(validate=not fast)
    universe = load_universe()
    universe_map = build_universe_map(universe)
    tickers = [row["yf_ticker"] for row in universe]
    from collections import Counter
    regions = Counter(r.get("region", "?") for r in universe)
    types   = Counter(r.get("type", "stock") for r in universe)
    sectors = Counter(r.get("sector", "?") for r in universe if r.get("sector"))
    _stat("Total assets", len(tickers))
    _stat("Stocks", types.get("stock", 0), len(tickers))
    _stat("ETFs", types.get("etf", 0), len(tickers))
    top_regions = ", ".join(f"{k}:{v}" for k, v in regions.most_common(5))
    _ok(f"Regions: {top_regions}")
    top_sectors = ", ".join(f"{k[:12]}:{v}" for k, v in sectors.most_common(5))
    _ok(f"Top sectors: {top_sectors}")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 2. FRED macro ────────────────────────────────────────────────────────
    _hdr("2/13", "FRED Macro Indicators")
    t = time.time()
    fred_data = fetch_fred_series()
    macro_regime = classify_regime(fred_data)
    ok_fred  = sum(1 for v in fred_data.values() if v is not None and "error" not in str(v))
    _stat("FRED series fetched", ok_fred, len(fred_data))
    _ok(f"Regime: {macro_regime.get('regime','?').upper()}  |  Risk: {macro_regime.get('risk_level','?')}")
    for key, label in [("FEDFUNDS","Fed Rate"),("CPIAUCSL","CPI"),("T10Y2Y","Yield Spread"),("UNRATE","Unemployment")]:
        val = fred_data.get(key)
        if val is not None:
            _ok(f"{label}: {round(float(val),2) if isinstance(val,(int,float)) else val}")
    macro_signal = {
        "regime": macro_regime.get("regime", "growth"),
        "risk_level": macro_regime.get("risk_level", "medium"),
        "sector_tailwinds": list(set()),
        "sector_headwinds": list(set()),
        "futures_summary": "",
        "fred_indicators": fred_data,
    }
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 3. Commodity futures ─────────────────────────────────────────────────
    _hdr("3/13", "Commodity Futures")
    t = time.time()
    futures_signal = fetch_futures_signal()
    macro_signal["sector_tailwinds"] = list(set(futures_signal.get("sector_tailwinds", [])))
    macro_signal["sector_headwinds"] = list(set(futures_signal.get("sector_headwinds", [])))
    macro_signal["futures_summary"]  = futures_signal.get("summary", "")
    for item in futures_signal.get("signals", []):
        direction = "↑" if item.get("direction") == "bullish" else "↓"
        _ok(f"{direction} {item.get('commodity','?'):12} {item.get('price','?')}")
    tw = macro_signal["sector_tailwinds"]
    hw = macro_signal["sector_headwinds"]
    if tw: _ok(f"Tailwinds: {', '.join(tw)}")
    if hw: _warn(f"Headwinds: {', '.join(hw)}")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 4. Polymarket ────────────────────────────────────────────────────────
    _hdr("4/13", "Polymarket Prediction Markets")
    t = time.time()
    poly_data = fetch_all_investment_markets()
    poly_geo      = poly_data["geo"]
    poly_rates    = poly_data.get("rates", [])
    poly_economy  = poly_data.get("economy", [])
    poly_indices  = poly_data.get("indices", [])
    poly_earnings = poly_data.get("earnings", [])
    poly_stocks   = poly_data.get("stocks", [])
    poly_alpha    = poly_data.get("alpha", [])
    poly_macro    = poly_data.get("macro", [])   # legacy
    poly_company  = poly_data.get("company", []) # legacy
    all_poly      = poly_geo + poly_rates + poly_economy + poly_indices + poly_earnings + poly_stocks
    total_markets = len(all_poly)
    _stat("Total markets fetched", total_markets)
    _stat("  Geo",             len(poly_geo))
    _stat("  Rates (CB)",      len(poly_rates))
    _stat("  Economy",         len(poly_economy))
    _stat("  Indices",         len(poly_indices))
    _stat("  Earnings",        len(poly_earnings))
    _stat("  Stocks",          len(poly_stocks))
    conv_bull = [m for m in poly_alpha if m["alpha_signal"] == "conviction_bull"]
    conv_bear = [m for m in poly_alpha if m["alpha_signal"] == "conviction_bear"]
    uncertain = [m for m in poly_alpha if m["alpha_signal"] == "uncertainty_alpha"]
    _stat("Alpha: conviction-bull",  len(conv_bull))
    _stat("Alpha: conviction-bear",  len(conv_bear))
    _stat("Alpha: uncertainty",      len(uncertain))
    for m in conv_bull[:2]:
        _ok(f"  ↑ {m['question'][:65]}  ({m['probability']:.0%})")
    for m in conv_bear[:2]:
        _warn(f" ↓ {m['question'][:65]}  ({m['probability']:.0%})")
    for m in uncertain[:2]:
        _ok(f"  ? {m['question'][:60]}  ({m['probability']:.0%}, {m.get('days_to_resolution','?')}d)")
    if poly_earnings:
        _ok(f"Top earnings market: {poly_earnings[0]['question'][:65]}  ({poly_earnings[0]['probability']:.0%})")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 5. GDELT ─────────────────────────────────────────────────────────────
    _hdr("5/13", "GDELT Conflict Indices")
    t = time.time()
    gdelt_data = fetch_regional_conflict_indices()
    ok_gdelt = sum(1 for v in gdelt_data.values() if isinstance(v, dict) and "error" not in v)
    _stat("Regions fetched", ok_gdelt, len(gdelt_data))
    for region, data in sorted(gdelt_data.items(), key=lambda x: -(x[1].get("conflict_index",0) if isinstance(x[1],dict) else 0))[:5]:
        idx = data.get("conflict_index", 0) if isinstance(data, dict) else 0
        bar = "█" * int(idx / 10) + "░" * (10 - int(idx / 10))
        _ok(f"{region:<20} [{bar}] {idx:.0f}/100")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 6. News ──────────────────────────────────────────────────────────────
    _hdr("6/13", "News RSS Headlines")
    t = time.time()
    news_data = fetch_news_headlines()
    headlines = news_data.get("headlines", [])
    from collections import Counter as _C
    by_source = _C(h["source"] for h in headlines)
    total_feeds = len(RSS_FEEDS)
    ok_feeds = len(by_source)
    _stat("Feeds responding", ok_feeds, total_feeds)
    _stat("Total headlines",  len(headlines))
    for src, cnt in by_source.most_common(8):
        bar = "█" * min(cnt, 20) + "░" * max(0, 20 - cnt)
        print(f"    {src:<24} {cnt:>3} [{bar}]")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 7. Price history + fundamentals ─────────────────────────────────────
    _hdr("7/13", "Price History + Fundamentals")
    t = time.time()
    PRIORITY = [
        "NVDA", "AMZN", "GOOGL", "TSM", "TSCO.L",
        "ISP.MI", "UCG.MI", "WBD.MI", "ASML.AS", "IWDA.AS",
    ]
    if fast:
        priority_set   = [t2 for t2 in PRIORITY if t2 in tickers]
        remaining      = [t2 for t2 in tickers if t2 not in priority_set]
        sample_tickers = priority_set + remaining[: max(0, 50 - len(priority_set))]
    else:
        sample_tickers = tickers
    _stat("Tickers to fetch", len(sample_tickers))
    price_data   = fetch_price_history(sample_tickers, period="1y")
    fundamentals = fetch_fundamentals(sample_tickers)
    ok_price = sum(1 for v in price_data.values() if "error" not in v)
    ok_fund  = len(fundamentals)
    has_pe   = sum(1 for v in fundamentals.values() if v.get("pe_ratio"))
    has_analyst = sum(1 for v in fundamentals.values() if v.get("analyst_score") is not None)
    _stat("Price data OK",   ok_price,    len(sample_tickers))
    _stat("Fundamentals OK", ok_fund,     len(sample_tickers))
    _stat("Has P/E ratio",   has_pe,      ok_fund)
    _stat("Has analyst rating", has_analyst, ok_fund)
    price_stats = compute_price_stats(price_data)
    _ok(f"Avg 1Y return: {price_stats['avg_return_1y']:+.1f}%  |  Median: {price_stats['median_return_1y']:+.1f}%")
    earnings_catalog = build_earnings_catalog(fundamentals, all_poly)
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 8. Insider / options / short interest ────────────────────────────────
    _hdr("8/13", "Insider Flow · Options · Short Interest")
    t = time.time()
    rated_tickers = list({t2 for t2 in sample_tickers if t2 in price_data and t2 in fundamentals})[:200]
    _stat("Tickers rated", len(rated_tickers))
    insider_data = fetch_insider_transactions(rated_tickers)
    options_data = fetch_options_signal(rated_tickers)
    short_data   = fetch_short_interest(rated_tickers)
    ok_insider = sum(1 for v in insider_data.values() if v.get("net_buy_pct_mktcap", 0) != 0)
    ok_options = sum(1 for v in options_data.values() if v.get("put_call_ratio") is not None)
    ok_short   = sum(1 for v in short_data.values() if v.get("short_float_pct", 0) > 0)
    net_buyers  = sum(1 for v in insider_data.values() if v.get("net_buy_pct_mktcap", 0) > 0)
    net_sellers = sum(1 for v in insider_data.values() if v.get("net_buy_pct_mktcap", 0) < 0)
    high_short  = sum(1 for v in short_data.values() if v.get("short_float_pct", 0) > 10)
    _stat("Insider data",   ok_insider, len(rated_tickers))
    _stat("  Net buyers",   net_buyers,  max(ok_insider,1))
    _stat("  Net sellers",  net_sellers, max(ok_insider,1))
    _stat("Options data",   ok_options, len(rated_tickers))
    _stat("Short interest", ok_short,   len(rated_tickers))
    _stat("  Short >10%",   high_short, max(ok_short,1))
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 9. WSB + BTC ─────────────────────────────────────────────────────────
    _hdr("9/13", "Reddit Retail Sentiment + BTC")
    t = time.time()
    wsb_posts  = fetch_wsb_posts(fetch_comment_depth=not fast)
    wsb_signal = analyze_wsb_signals(wsb_posts)
    btc_signal = fetch_btc_signal()
    total_posts  = wsb_signal.get("total_posts_analyzed", 0)
    dd_posts_out = wsb_signal.get("dd_posts", [])
    opt_tickers  = wsb_signal.get("options_flow_tickers", [])
    squeeze_cand = wsb_signal.get("squeeze_candidates", [])
    trending     = wsb_signal.get("trending_tickers", [])
    subreddits   = wsb_signal.get("subreddits", [])
    _stat("Subreddits scraped", len(subreddits), len(SUBREDDIT_URLS))
    _stat("Posts analyzed",     total_posts)
    _stat("DD/Analysis posts",  len(dd_posts_out))
    _stat("Options-flow tickers", len(opt_tickers))
    _stat("Squeeze candidates",   len(squeeze_cand))
    if trending:
        top5 = [f"{t2['ticker']}({t2['mentions_7d']:.0f})" for t2 in trending[:5]]
        _ok(f"Top tickers: {', '.join(top5)}")
    if dd_posts_out:
        _ok(f"Top DD: \"{dd_posts_out[0]['title'][:55]}\"  score={dd_posts_out[0]['score']}")
    if squeeze_cand:
        _ok(f"Squeeze watch: {', '.join(squeeze_cand[:5])}")
    btc_price = btc_signal.get("price_usd", "?")
    btc_trend = btc_signal.get("trend", "?")
    _ok(f"BTC: ${btc_price:,.0f}  trend={btc_trend}" if isinstance(btc_price, (int,float)) else f"BTC: {btc_price}  trend={btc_trend}")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 10. Market sentiment ─────────────────────────────────────────────────
    _hdr("10/13", "Market Sentiment")
    t = time.time()
    sentiment = fetch_market_sentiment()
    vix_val   = sentiment.get("vix", {}).get("vix", "?")
    fg_score  = sentiment.get("fear_greed", {}).get("score", "?")
    fg_rating = sentiment.get("fear_greed", {}).get("rating", "?")
    hy_spread = sentiment.get("credit_spreads", {}).get("hy_spread", "?")
    _ok(f"VIX:            {vix_val}")
    _ok(f"Fear/Greed:     {fg_score} — {fg_rating}")
    _ok(f"HY Credit Spread: {hy_spread}")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 11. Bond yields + yield curve ────────────────────────────────────────
    _hdr("11/13", "Bond Yields + Yield Curve")
    t = time.time()
    bond_yields = fetch_bond_yields()
    _ok(f"Summary: {bond_yields.get('summary','N/A')}")
    for tenor, key in [("3M","3m"), ("2Y","2y"), ("10Y","10y"), ("30Y","30y")]:
        d = bond_yields.get(key, {})
        if isinstance(d, dict) and d.get("yield_pct"):
            chg = d.get("change_1m_bp","?")
            _ok(f"{tenor} yield: {d['yield_pct']}%  ({chg:+}bp 1M)" if isinstance(chg, (int,float)) else f"{tenor} yield: {d['yield_pct']}%")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 12. Currencies + sector ETFs ─────────────────────────────────────────
    _hdr("12/13", "Currencies + Sector ETFs")
    t = time.time()
    currencies  = fetch_currencies()
    sector_etfs = fetch_sector_etf_performance()
    ok_fx = sum(1 for v in currencies.values() if isinstance(v,dict) and v.get("price"))
    _stat("FX pairs fetched", ok_fx, len(currencies))
    for pair, data in list(currencies.items())[:6]:
        if isinstance(data, dict) and data.get("price"):
            chg = data.get("1M", "?")
            _ok(f"{pair:<10} {data['price']}  1M={chg:+.1f}%" if isinstance(chg,(int,float)) else f"{pair:<10} {data['price']}")
    leaders  = sector_etfs.get("leaders_1m", [])
    laggards = sector_etfs.get("laggards_1m", [])
    _ok(f"ETF leaders 1M:  {', '.join(leaders[:4])}")
    _warn(f"ETF laggards 1M: {', '.join(laggards[:4])}")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── 13. Extended commodities ─────────────────────────────────────────────
    _hdr("13/13", "Extended Commodities")
    t = time.time()
    commodities_ext = fetch_extended_commodities()
    ok_comm = sum(1 for v in commodities_ext.values() if isinstance(v,dict) and v.get("price"))
    _stat("Commodity signals", ok_comm, len(commodities_ext))
    for name, data in list(commodities_ext.items())[:8]:
        if isinstance(data, dict) and data.get("price"):
            chg = data.get("1M","?")
            direction = "↑" if isinstance(chg,(int,float)) and chg > 0 else ("↓" if isinstance(chg,(int,float)) and chg < 0 else " ")
            _ok(f"{direction} {name:<20} {data['price']}  1M={chg:+.1f}%" if isinstance(chg,(int,float)) else f"  {name:<20} {data['price']}")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── Post-processing ──────────────────────────────────────────────────────
    _hdr("POST", "Post-Processing")
    t = time.time()
    print("  → News-market bridge...")
    news_market_summary = build_news_market_summary(headlines, all_poly)
    _stat("Markets with news support", news_market_summary['coverage_pct'], 100, "%")

    print("  → Retail trend (WSB)...")
    retail_trend = compute_retail_trend(wsb_signal, universe_map, short_data)
    _ok(f"{retail_trend['trend_direction'].upper()} — {retail_trend['narrative'][:80]}")

    # ── Build signals dict ───────────────────────────────────────────────────
    signals = {
        "collected_at":       datetime.now().isoformat(),
        "universe_size":      len(tickers),
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
        "macro":              macro_signal,
        "polymarket_geo":      poly_geo,
        "polymarket_rates":    poly_rates,
        "polymarket_economy":  poly_economy,
        "polymarket_indices":  poly_indices,
        "polymarket_earnings": poly_earnings,
        "polymarket_stocks":   poly_stocks,
        "polymarket_alpha":    poly_alpha,
        "polymarket_macro":    poly_macro,
        "polymarket_company":  poly_company,
        "gdelt":              gdelt_data,
        "news":               news_data,
        "price_data":         price_data,
        "price_stats":        price_stats,
        "fundamentals":       fundamentals,
        "earnings":           earnings_catalog,
        "insider":            insider_data,
        "options":            options_data,
        "short_interest":     short_data,
        "wsb":                wsb_signal,
        "btc":                btc_signal,
        "sentiment":          sentiment,
        "bond_yields":        bond_yields,
        "currencies":         currencies,
        "sector_etfs":        sector_etfs,
        "commodities_ext":    commodities_ext,
        "news_market_bridge": news_market_summary,
        "retail_trend":       retail_trend,
        "events":             [],
        "themes":             [],
    }

    print("  → Scoring cross-source themes...")
    themes = score_themes(signals)
    signals["themes"] = themes
    _stat("Themes detected", len(themes))
    for th in themes[:5]:
        _ok(f"{th['label']:<35} score={th['composite']:.0f}")

    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── Fast scorer ──────────────────────────────────────────────────────────
    _hdr("SCORE", "Fast Scorer")
    t = time.time()
    fast_scores = score_all_assets(signals)
    signals["fast_scores"] = fast_scores

    grades = [s["grade"].replace(" ⚠","") for s in fast_scores.values()]
    no_sig = sum(1 for s in fast_scores.values() if s.get("no_signal"))
    buckets = {
        "AAA–AA-": sum(1 for g in grades if g in {"AAA","AA+","AA","AA-"}),
        "A+–A-":   sum(1 for g in grades if g in {"A+","A","A-"}),
        "BBB":     sum(1 for g in grades if "BBB" in g),
        "BB":      sum(1 for g in grades if g in {"BB+","BB","BB-"}),
        "B–CC":    sum(1 for g in grades if g in {"B","CCC","CC"}),
    }
    _stat("Total scored", len(fast_scores))
    _stat("⚠ No-signal flags", no_sig, len(fast_scores))
    for bucket, cnt in buckets.items():
        _stat(f"  Grade {bucket}", cnt, len(fast_scores))
    score_vals = [s["score"] for s in fast_scores.values()]
    _ok(f"Score range: {min(score_vals)}–{max(score_vals)}  avg={sum(score_vals)/len(score_vals):.1f}")
    high_priority = [t2 for t2, s in fast_scores.items() if s["score"] >= 70]
    _stat("Flagged for deep rating (≥70)", len(high_priority), len(fast_scores))
    if high_priority[:8]:
        _ok(f"Top picks: {', '.join(high_priority[:8])}")
    print(f"  ⏱  {time.time()-t:.1f}s")

    # ── Save signals ──────────────────────────────────────────────────────────
    out_path = "data/signals.json"
    with open(out_path, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    size_kb = Path(out_path).stat().st_size / 1024

    # ── Portfolio analytics ───────────────────────────────────────────────────
    portfolio_path  = Path("input/portfolio.csv")
    analytics_path  = Path("data/portfolio_analytics_cache.json")
    skip_analytics  = False
    if fast and analytics_path.exists():
        try:
            age_hours = (datetime.now() - datetime.fromtimestamp(analytics_path.stat().st_mtime)).total_seconds() / 3600
            if age_hours < 6:
                _warn(f"Portfolio analytics cache {age_hours:.1f}h old — skipping (fast mode)")
                skip_analytics = True
        except Exception:
            pass

    if portfolio_path.exists() and not skip_analytics:
        _hdr("PORT", "Portfolio Analytics")
        t = time.time()
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
            sp500_s   = bm_series.get("S&P 500")
            metrics   = compute_quant_metrics(port_series, sp500_s)
            tech      = compute_position_technicals(portfolio["holdings"], _TMAP)
            conc      = compute_concentration_metrics(summary["positions"])
            corr      = compute_correlation_matrix(portfolio["holdings"], _TMAP)

            analytics = {
                "computed_at":    datetime.now().isoformat(),
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
            tot = summary["totals"]
            _ok(f"Invested: €{tot['total_invested']:,.0f}  Value: €{tot['current_value']:,.0f}  Return: {tot['unrealized_pct']:+.1f}%")
            _ok(f"Sharpe={metrics.get('sharpe','?')}  Sortino={metrics.get('sortino','?')}  Max DD={metrics.get('max_drawdown','?')}")
            _ok(f"ROI series: {port_series.iloc[-1]:+.2f}%  |  {len(portfolio['holdings'])} open positions")
            _stat("Positions priced", sum(1 for v in prices.values() if v), len(portfolio["holdings"]))
        except Exception as e:
            _warn(f"Analytics failed: {e}")
        print(f"  ⏱  {time.time()-t:.1f}s")

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - _T0
    print(f"\n{'═'*55}")
    print(f"  COLLECTION COMPLETE  {total_elapsed:.1f}s total")
    print(f"  signals.json  {size_kb:.0f} KB  |  {len(fast_scores)} assets scored")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*55}\n")

    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="50 tickers, skip validation (~5 min)")
    args = parser.parse_args()
    collect(fast=args.fast)
