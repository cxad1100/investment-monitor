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
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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


# ── Live terminal tracker ─────────────────────────────────────────────────────

ALL_STEPS = [
    "1/13  Universe",
    "2/13  FRED Macro",
    "3/13  Commodity Futures",
    "4/13  Polymarket",
    "5/13  GDELT",
    "6/13  News RSS",
    "7/13  Price + Fundamentals",
    "8/13  Insider / Options / Short",
    "9/13  Reddit WSB + BTC",
    "10/13 Sentiment",
    "11/13 Bond Yields",
    "12/13 Currencies + ETFs",
    "13/13 Commodities",
    " *    Bridge + Themes",
    " *    Scoring",
    " *    Portfolio",
]
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Tracker:
    def __init__(self, fast: bool):
        self.fast   = fast
        self._t0    = time.time()
        self._st    = time.time()   # step start
        self._frame = 0
        self._idx   = -1
        # per-step state: (status, summary, elapsed)
        self._steps: list[tuple[str, str, float]] = []
        self.activity = ""          # sub-activity shown at bottom

    # ── public api ──────────────────────────────────────────────────────────

    def start(self, activity: str = "") -> None:
        self._idx += 1
        self._st = time.time()
        self.activity = activity
        self._steps.append(("running", "", 0.0))

    def update(self, activity: str) -> None:
        self.activity = activity
        if self._steps:
            s, sm, _ = self._steps[-1]
            self._steps[-1] = (s, sm, round(time.time() - self._st, 1))

    def done(self, summary: str = "") -> None:
        elapsed = round(time.time() - self._st, 1)
        self._steps[-1] = ("done", summary, elapsed)
        self.activity = ""

    def fail(self, summary: str = "") -> None:
        elapsed = round(time.time() - self._st, 1)
        self._steps[-1] = ("fail", summary, elapsed)
        self.activity = ""

    # ── renderer ─────────────────────────────────────────────────────────────

    def render(self) -> Panel:
        self._frame = (self._frame + 1) % len(_SPIN)
        spin = _SPIN[self._frame]

        total_elapsed = int(time.time() - self._t0)
        mm, ss = divmod(total_elapsed, 60)
        mode = "FAST" if self.fast else "FULL"
        title = f"[bold]MONITOR · SIGNAL COLLECTION[/bold]  [{mode}]  ⏱ {mm:02d}:{ss:02d}"

        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=2,  no_wrap=True)   # icon
        grid.add_column(width=22, no_wrap=True)   # step name
        grid.add_column(width=42, no_wrap=True)   # summary / activity
        grid.add_column(width=7,  no_wrap=True, justify="right")  # elapsed

        for i, step_name in enumerate(ALL_STEPS):
            if i < len(self._steps):
                status, summary, elapsed = self._steps[i]
                if status == "done":
                    icon    = Text("✓", style="bold green")
                    name_t  = Text(step_name, style="dim white")
                    sum_t   = Text(summary[:42], style="dim")
                    ela_t   = Text(f"{elapsed}s", style="dim")
                elif status == "running":
                    icon    = Text(spin, style="bold yellow")
                    name_t  = Text(step_name, style="bold yellow")
                    act     = self.activity[:42]
                    sum_t   = Text(act, style="yellow")
                    ela_t   = Text(f"{elapsed:.1f}s", style="yellow")
                else:  # fail
                    icon    = Text("✗", style="bold red")
                    name_t  = Text(step_name, style="red")
                    sum_t   = Text(summary[:42], style="red dim")
                    ela_t   = Text(f"{elapsed}s", style="red dim")
            elif i == len(self._steps):  # next up (highlighted)
                icon   = Text("›", style="bright_black")
                name_t = Text(step_name, style="bright_black")
                sum_t  = Text("waiting...", style="bright_black")
                ela_t  = Text("")
            else:
                icon   = Text("○", style="bright_black")
                name_t = Text(step_name, style="bright_black")
                sum_t  = Text("")
                ela_t  = Text("")

            grid.add_row(icon, name_t, sum_t, ela_t)

        return Panel(grid, title=title, border_style="bright_black",
                     subtitle=f"[dim]{self.activity[:70]}[/dim]" if self.activity else "")


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            if ticker in q or ticker.replace(".DE","").replace(".PA","").replace(".L","") in q:
                catalog[ticker]["beat_probability"] = market["probability"]
                catalog[ticker]["polymarket_question"] = market["question"]
    return catalog


def compute_price_stats(price_data: dict) -> dict:
    returns = [v.get("return_1y", 0) for v in price_data.values() if "error" not in v]
    return {
        "avg_return_1y":    round(float(np.mean(returns)),   2) if returns else 0.0,
        "median_return_1y": round(float(np.median(returns)), 2) if returns else 0.0,
    }


# ── Main collection ───────────────────────────────────────────────────────────

def collect(fast: bool = False) -> dict:
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        print("ERROR: FRED_API_KEY not set")
        sys.exit(1)

    Path("data").mkdir(exist_ok=True)
    console = Console()
    trk = Tracker(fast)

    with Live(trk.render(), console=console, refresh_per_second=10,
              vertical_overflow="visible") as live:

        def _refresh(activity: str = "") -> None:
            if activity:
                trk.update(activity)
            live.update(trk.render())

        # ── 1. Universe ──────────────────────────────────────────────────
        trk.start("loading universe CSV...")
        _refresh()
        refresh_universe(validate=not fast)
        _refresh("parsing assets...")
        universe     = load_universe()
        universe_map = build_universe_map(universe)
        tickers      = [row["yf_ticker"] for row in universe]
        regions  = Counter(r.get("region","?") for r in universe)
        top_r    = " ".join(f"{k}:{v}" for k,v in regions.most_common(4))
        trk.done(f"{len(tickers)} assets · {top_r}")
        _refresh()

        # ── 2. FRED macro ────────────────────────────────────────────────
        trk.start("fetching FRED series...")
        _refresh()
        fred_data    = fetch_fred_series()
        _refresh("classifying regime...")
        macro_regime = classify_regime(fred_data)
        ok_fred = sum(1 for v in fred_data.values() if v is not None)
        regime  = macro_regime.get("regime","?").upper()
        risk    = macro_regime.get("risk_level","?")
        macro_signal = {
            "regime": macro_regime.get("regime","growth"),
            "risk_level": risk,
            "sector_tailwinds": [],
            "sector_headwinds": [],
            "futures_summary": "",
            "fred_indicators": fred_data,
        }
        trk.done(f"{ok_fred}/16 series · regime={regime} · risk={risk}")
        _refresh()

        # ── 3. Commodity futures ─────────────────────────────────────────
        trk.start("fetching futures prices...")
        _refresh()
        futures_signal = fetch_futures_signal()
        macro_signal["sector_tailwinds"] = list(set(futures_signal.get("sector_tailwinds",[])))
        macro_signal["sector_headwinds"] = list(set(futures_signal.get("sector_headwinds",[])))
        macro_signal["futures_summary"]  = futures_signal.get("summary","")
        tw = ", ".join(macro_signal["sector_tailwinds"]) or "none"
        hw = ", ".join(macro_signal["sector_headwinds"]) or "none"
        trk.done(f"↑ {tw}  ↓ {hw}")
        _refresh()

        # ── 4. Polymarket ────────────────────────────────────────────────
        trk.start("paginating markets (up to 5000)...")
        _refresh()
        poly_data     = fetch_all_investment_markets()
        poly_geo      = poly_data["geo"]
        poly_rates    = poly_data.get("rates",[])
        poly_economy  = poly_data.get("economy",[])
        poly_indices  = poly_data.get("indices",[])
        poly_earnings = poly_data.get("earnings",[])
        poly_stocks   = poly_data.get("stocks",[])
        poly_alpha    = poly_data.get("alpha",[])
        poly_macro    = poly_data.get("macro",[])
        poly_company  = poly_data.get("company",[])
        all_poly      = poly_geo + poly_rates + poly_economy + poly_indices + poly_earnings + poly_stocks
        cb  = sum(1 for m in poly_alpha if m["alpha_signal"]=="conviction_bull")
        cbr = sum(1 for m in poly_alpha if m["alpha_signal"]=="conviction_bear")
        unc = sum(1 for m in poly_alpha if m["alpha_signal"]=="uncertainty_alpha")
        trk.done(
            f"{len(all_poly)} mkts · earn={len(poly_earnings)} "
            f"idx={len(poly_indices)} bull={cb} bear={cbr} unc={unc}"
        )
        _refresh()

        # ── 5. GDELT ─────────────────────────────────────────────────────
        trk.start("fetching conflict indices...")
        _refresh()
        gdelt_data = fetch_regional_conflict_indices()
        ok_g = sum(1 for v in gdelt_data.values() if isinstance(v,dict) and "error" not in v)
        top_conflict = sorted(
            [(k, v.get("conflict_index",0)) for k,v in gdelt_data.items() if isinstance(v,dict)],
            key=lambda x: -x[1]
        )[:2]
        top_str = " ".join(f"{k}:{v:.0f}" for k,v in top_conflict)
        trk.done(f"{ok_g}/{len(gdelt_data)} regions · hot: {top_str}")
        _refresh()

        # ── 6. News ──────────────────────────────────────────────────────
        trk.start(f"fetching {len(RSS_FEEDS)} RSS feeds + SEC 8-K...")
        _refresh()
        news_data  = fetch_news_headlines()
        headlines  = news_data.get("headlines",[])
        from collections import Counter as _C
        by_src = _C(h["source"] for h in headlines)
        trk.done(f"{len(headlines)} headlines · {len(by_src)}/{len(RSS_FEEDS)+1} feeds OK")
        _refresh()

        # ── 7. Price history + fundamentals ──────────────────────────────
        PRIORITY = ["NVDA","AMZN","GOOGL","TSM","TSCO.L","ISP.MI","UCG.MI","WBD.MI","ASML.AS","IWDA.AS"]
        if fast:
            prio_set       = [t for t in PRIORITY if t in tickers]
            sample_tickers = prio_set + [t for t in tickers if t not in prio_set][:max(0,50-len(prio_set))]
        else:
            sample_tickers = tickers

        trk.start(f"downloading prices for {len(sample_tickers)} tickers...")
        _refresh()
        price_data = fetch_price_history(sample_tickers, period="1y")
        _refresh(f"fetching fundamentals ({len(sample_tickers)} tickers)...")
        fundamentals     = fetch_fundamentals(sample_tickers)
        price_stats      = compute_price_stats(price_data)
        earnings_catalog = build_earnings_catalog(fundamentals, all_poly)
        ok_p = sum(1 for v in price_data.values() if "error" not in v)
        ok_f = len(fundamentals)
        has_a = sum(1 for v in fundamentals.values() if v.get("analyst_score") is not None)
        avg_r = price_stats["avg_return_1y"]
        trk.done(
            f"price {ok_p}/{len(sample_tickers)} "
            f"fund {ok_f}/{len(sample_tickers)} "
            f"analyst {has_a}/{ok_f} "
            f"avg1Y={avg_r:+.1f}%"
        )
        _refresh()

        # ── 8. Insider / options / short ──────────────────────────────────
        rated_tickers = list({t for t in sample_tickers if t in price_data and t in fundamentals})[:200]
        trk.start(f"insider flow for {len(rated_tickers)} tickers...")
        _refresh()
        insider_data = fetch_insider_transactions(rated_tickers)
        _refresh(f"options flow ({len(rated_tickers)} tickers)...")
        options_data = fetch_options_signal(rated_tickers)
        _refresh(f"short interest ({len(rated_tickers)} tickers)...")
        short_data   = fetch_short_interest(rated_tickers)
        ok_i  = sum(1 for v in insider_data.values() if v.get("net_buy_pct_mktcap",0)!=0)
        buyers = sum(1 for v in insider_data.values() if v.get("net_buy_pct_mktcap",0)>0)
        ok_o  = sum(1 for v in options_data.values() if v.get("put_call_ratio") is not None)
        hi_sh = sum(1 for v in short_data.values() if v.get("short_float_pct",0)>10)
        trk.done(f"insider {ok_i} ({buyers}↑) · options {ok_o} · short>10% {hi_sh}")
        _refresh()

        # ── 9. WSB + BTC ──────────────────────────────────────────────────
        sub_count = len(SUBREDDIT_URLS)
        trk.start(f"scraping {sub_count} subreddits...")
        _refresh()
        wsb_posts  = fetch_wsb_posts(fetch_comment_depth=not fast)
        _refresh("analysing mentions, sentiment, DD posts...")
        wsb_signal = analyze_wsb_signals(wsb_posts)
        _refresh("fetching BTC signal...")
        btc_signal  = fetch_btc_signal()
        tot_posts   = wsb_signal.get("total_posts_analyzed",0)
        dd_ct       = len(wsb_signal.get("dd_posts",[]))
        opt_ct      = len(wsb_signal.get("options_flow_tickers",[]))
        sq_ct       = len(wsb_signal.get("squeeze_candidates",[]))
        trending    = wsb_signal.get("trending_tickers",[])
        top5        = " ".join(t["ticker"] for t in trending[:5])
        btc_px      = btc_signal.get("price_usd","?")
        btc_px_str  = f"${btc_px:,.0f}" if isinstance(btc_px,(int,float)) else str(btc_px)
        trk.done(
            f"{tot_posts} posts · DD={dd_ct} opt={opt_ct} sq={sq_ct} "
            f"top: {top5} · BTC {btc_px_str}"
        )
        _refresh()

        # ── 10. Sentiment ─────────────────────────────────────────────────
        trk.start("VIX · Fear/Greed · credit spreads...")
        _refresh()
        sentiment  = fetch_market_sentiment()
        vix_val    = sentiment.get("vix",{}).get("vix","?")
        fg_score   = sentiment.get("fear_greed",{}).get("score","?")
        fg_rating  = sentiment.get("fear_greed",{}).get("rating","?")
        trk.done(f"VIX {vix_val} · F/G {fg_score} ({fg_rating})")
        _refresh()

        # ── 11. Bond yields ───────────────────────────────────────────────
        trk.start("fetching yields (2Y/10Y/30Y)...")
        _refresh()
        bond_yields = fetch_bond_yields()
        trk.done(bond_yields.get("summary","N/A"))
        _refresh()

        # ── 12. Currencies + sector ETFs ──────────────────────────────────
        trk.start("FX rates + sector ETF performance...")
        _refresh()
        currencies  = fetch_currencies()
        _refresh("sector ETF 1M performance...")
        sector_etfs = fetch_sector_etf_performance()
        ok_fx    = sum(1 for v in currencies.values() if isinstance(v,dict) and v.get("price"))
        leaders  = sector_etfs.get("leaders_1m",[])[:3]
        laggards = sector_etfs.get("laggards_1m",[])[:3]
        trk.done(f"{ok_fx} FX pairs · ↑{','.join(leaders)} ↓{','.join(laggards)}")
        _refresh()

        # ── 13. Extended commodities ──────────────────────────────────────
        trk.start("fetching extended commodity signals...")
        _refresh()
        commodities_ext = fetch_extended_commodities()
        ok_c = sum(1 for v in commodities_ext.values() if isinstance(v,dict) and v.get("price"))
        trk.done(f"{ok_c}/{len(commodities_ext)} commodity signals")
        _refresh()

        # ── Post: bridge + themes ─────────────────────────────────────────
        trk.start("news-market bridge · themes...")
        _refresh()
        news_market_summary = build_news_market_summary(headlines, all_poly)
        _refresh("computing retail trend...")
        retail_trend = compute_retail_trend(wsb_signal, universe_map, short_data)

        signals = {
            "collected_at":        datetime.now().isoformat(),
            "universe_size":       len(tickers),
            "rated_ticker_count":  len(rated_tickers),
            "universe_map": {
                k: {"sector":v.get("sector",""),"region":v.get("region",""),
                    "type":v.get("type","stock"),"name":v.get("name","")}
                for k,v in universe_map.items()
            },
            "macro":               macro_signal,
            "polymarket_geo":      poly_geo,
            "polymarket_rates":    poly_rates,
            "polymarket_economy":  poly_economy,
            "polymarket_indices":  poly_indices,
            "polymarket_earnings": poly_earnings,
            "polymarket_stocks":   poly_stocks,
            "polymarket_alpha":    poly_alpha,
            "polymarket_macro":    poly_macro,
            "polymarket_company":  poly_company,
            "gdelt":               gdelt_data,
            "news":                news_data,
            "price_data":          price_data,
            "price_stats":         price_stats,
            "fundamentals":        fundamentals,
            "earnings":            earnings_catalog,
            "insider":             insider_data,
            "options":             options_data,
            "short_interest":      short_data,
            "wsb":                 wsb_signal,
            "btc":                 btc_signal,
            "sentiment":           sentiment,
            "bond_yields":         bond_yields,
            "currencies":          currencies,
            "sector_etfs":         sector_etfs,
            "commodities_ext":     commodities_ext,
            "news_market_bridge":  news_market_summary,
            "retail_trend":        retail_trend,
            "events":              [],
            "themes":              [],
        }

        _refresh("scoring themes...")
        themes = score_themes(signals)
        signals["themes"] = themes
        top3   = " | ".join(f"{t['label']} ({t['composite']:.0f})" for t in themes[:3])
        cov    = news_market_summary.get("coverage_pct",0)
        trk.done(f"bridge {cov}% · {len(themes)} themes · {top3[:45]}")
        _refresh()

        # ── Scoring ───────────────────────────────────────────────────────
        trk.start(f"scoring {len(universe_map)} assets...")
        _refresh()
        fast_scores = score_all_assets(signals)
        signals["fast_scores"] = fast_scores
        score_vals   = [s["score"] for s in fast_scores.values()]
        no_sig       = sum(1 for s in fast_scores.values() if s.get("no_signal"))
        high_pri     = [t for t,s in fast_scores.items() if s["score"]>=70]
        top8         = " ".join(high_pri[:8])
        trk.done(
            f"{len(fast_scores)} scored · range {min(score_vals)}–{max(score_vals)} "
            f"avg {sum(score_vals)/len(score_vals):.0f} · ⚠{no_sig} · top: {top8[:30]}"
        )
        _refresh()

        # ── Save ─────────────────────────────────────────────────────────
        out_path = "data/signals.json"
        with open(out_path,"w") as f:
            json.dump(signals, f, indent=2, default=str)
        size_kb = Path(out_path).stat().st_size / 1024

        # ── Portfolio analytics ───────────────────────────────────────────
        portfolio_path = Path("input/portfolio.csv")
        analytics_path = Path("data/portfolio_analytics_cache.json")
        skip_analytics = False
        if fast and analytics_path.exists():
            try:
                age_h = (datetime.now()-datetime.fromtimestamp(analytics_path.stat().st_mtime)).total_seconds()/3600
                if age_h < 6:
                    trk.start(f"portfolio analytics cache {age_h:.1f}h old — skipping (fast mode)")
                    trk.done("skipped (cache fresh)")
                    _refresh()
                    skip_analytics = True
            except Exception:
                pass

        if portfolio_path.exists() and not skip_analytics:
            trk.start("portfolio: ROI series · benchmarks · quant metrics...")
            _refresh()
            try:
                from tools.portfolio_tools import (
                    parse_portfolio, fetch_current_prices, compute_portfolio_summary, TICKER_MAP as _TMAP,
                )
                from tools.portfolio_analytics import (
                    build_roi_timeseries, compute_quant_metrics,
                    compute_position_technicals, compute_concentration_metrics,
                    compute_correlation_matrix,
                )
                portfolio   = parse_portfolio(portfolio_path)
                _refresh("fetching live prices...")
                prices      = fetch_current_prices(portfolio["holdings"])
                summary     = compute_portfolio_summary(portfolio, prices)
                _refresh("building ROI timeseries + benchmarks...")
                port_series, bm_series = build_roi_timeseries(portfolio["transactions"])
                sp500_s     = bm_series.get("S&P 500")
                metrics     = compute_quant_metrics(port_series, sp500_s)
                _refresh("technicals · concentration · correlation...")
                tech        = compute_position_technicals(portfolio["holdings"], _TMAP)
                conc        = compute_concentration_metrics(summary["positions"])
                corr        = compute_correlation_matrix(portfolio["holdings"], _TMAP)
                analytics   = {
                    "computed_at":   datetime.now().isoformat(),
                    "summary": {"totals":summary["totals"],"positions":summary["positions"],"realized_detail":summary["realized_detail"]},
                    "portfolio_roi": {str(k.date()):v for k,v in port_series.items()},
                    "benchmark_roi": {name:{str(k.date()):v for k,v in s.items()} for name,s in bm_series.items()},
                    "quant_metrics": metrics,"technicals":tech,"concentration":conc,"correlation":corr,
                }
                with open("data/portfolio_analytics_cache.json","w") as f:
                    json.dump(analytics, f, indent=2, default=str)
                tot = summary["totals"]
                trk.done(
                    f"€{tot['current_value']:,.0f} ({tot['unrealized_pct']:+.1f}%) "
                    f"ROI={port_series.iloc[-1]:+.2f}% Sharpe={metrics.get('sharpe','?')}"
                )
            except Exception as e:
                trk.fail(f"failed: {e}")
            _refresh()

    # ── Final summary (outside Live) ──────────────────────────────────────────
    total_s = round(time.time() - trk._t0, 1)
    console.print(f"\n[bold green]✓ Collection complete[/bold green]  "
                  f"[dim]{total_s}s · {size_kb:.0f} KB · {len(fast_scores)} assets scored[/dim]")

    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="50 tickers, skip validation (~5 min)")
    args = parser.parse_args()
    collect(fast=args.fast)
