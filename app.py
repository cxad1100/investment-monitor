"""Streamlit dashboard for weekly investment ratings."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import re as _re

# ── Sector → relevant event group IDs ───────────────────────────────────────
SECTOR_GROUPS = {
    "Energy":                  ["iran_me", "russia_ukraine", "recession"],
    "Information Technology":  ["fomc_next", "fed_annual", "china_taiwan", "bitcoin", "ai_tech", "recession"],
    "Financials":              ["fomc_next", "fed_annual", "recession"],
    "Materials":               ["china_taiwan", "russia_ukraine", "recession"],
    "Industrials":             ["china_taiwan", "russia_ukraine", "recession"],
    "Consumer Discretionary":  ["fomc_next", "fed_annual", "recession"],
    "Consumer Staples":        ["recession"],
    "Utilities":               ["fomc_next", "fed_annual", "recession"],
    "Healthcare":              ["recession"],
    "Real Estate":             ["fomc_next", "fed_annual", "recession"],
    "Communication Services":  ["china_taiwan", "ai_tech", "bitcoin"],
}

# ── Impact descriptions: causal chain per sector ─────────────────────────────
IMPACT = {
    "fomc_next": {
        "Financials":             "Next-meeting rate decision directly reprices bank NIM expectations. A cut compresses NIMs immediately; hold = margin stability.",
        "Information Technology": "Rate cuts reduce the discount rate on future cash flows, re-rating growth multiples upward. Each 25bp cut adds ~2-4% to DCF-derived fair values for high-multiple tech.",
        "Utilities":              "Rate-sensitive sector: cut = immediate NAV expansion as dividend yield spread vs risk-free improves.",
        "Real Estate":            "25bp cut lowers mortgage rates ~20bp with 4-8 week lag, stimulating transaction volume and compressing cap rates.",
        "Consumer Discretionary": "Rate cut relieves consumer credit pressure (mortgages, auto, card rates), boosting spending capacity.",
        "default":                "Next FOMC decision reprices risk-free rate and equity discount rates across the board.",
    },
    "fed_annual": {
        "Financials":             "Full-year rate path determines NIM trajectory. 0 cuts = NIM stable but credit risk builds through year; 2+ cuts = spread compression, potential delinquency relief.",
        "Information Technology": "Prolonged high rates sustain multiple compression on growth stocks. Market pricing ~29% chance of any 2026 cut — status quo favors cash-flow-positive tech over high-multiple names.",
        "Utilities":              "Each cut adds 3-5% to utility NAV. With 71% probability of zero cuts, utilities remain structurally disadvantaged vs bonds.",
        "Real Estate":            "Rate path determines whether REIT valuations stabilize or continue compressing. Zero-cut scenario = continued headwind.",
        "default":                "Annual rate path shapes cost of capital and equity multiples for the full year.",
    },
    "iran_me": {
        "Energy":                 "Iran conflict risks Strait of Hormuz closure (~20% of global oil transit) → Brent spike $20-40+/bbl. Nuclear deal (59% probability) = 1-2M bbl/day supply addition, pressuring prices. Net: bullish on war, bearish on deal.",
        "Industrials":            "Defense contractors (LMT, RTX, NOC) benefit directly from escalation — emergency procurement, supplemental defense budgets.",
        "Consumer Discretionary": "Oil spike drives fuel cost surge, crushing logistics margins and reducing consumer discretionary spend.",
        "Consumer Staples":       "Input cost pressure via energy-driven logistics. Modest but persistent headwind on margins.",
        "default":                "Iran drives oil price volatility, propagating through energy-linked input costs across sectors.",
    },
    "china_taiwan": {
        "Information Technology": "Taiwan hosts ~90% of advanced semiconductor capacity (TSMC). Military clash = multi-year chip supply shock worse than 2021. NVDA, AMD, AAPL, AMAT, LRCX face direct exposure.",
        "Industrials":            "Precision component supply disruption hits industrial equipment; defense contractors benefit from NATO/US spending surge.",
        "Consumer Discretionary": "Electronics inventory depletion comparable to 2021 chip shortage — but structurally more severe and longer duration.",
        "Materials":              "China-linked rare earth and specialty chemical supply chains face secondary sanction risk.",
        "default":                "Taiwan conflict = most severe supply chain shock in decades. Systemic risk with limited hedge.",
    },
    "russia_ukraine": {
        "Energy":                 "Peace deal → European energy security improvement, gas prices normalize. Escalation → embargo risk, energy spike.",
        "Industrials":            "European defense rearmament (NATO 2% GDP target) is structural tailwind for aerospace & defense regardless of war outcome.",
        "Materials":              "Russia controls ~40% of global palladium, major nickel/aluminum supplier. Sanction changes on deal or escalation reprice these commodities.",
        "Financials":             "Deal = European bank reconstruction financing opportunity. Escalation = credit losses on Russian exposure.",
        "default":                "War resolution path shapes European growth outlook and commodity supply.",
    },
    "bitcoin": {
        "Information Technology": "BTC above $80k = institutional risk-on confirmed → supportive of high-multiple tech. Crash to $35-45k (32-50% probability) = broad risk-off signal, tech multiple compression.",
        "Financials":             "Crypto-adjacent fintechs (COIN, PYPL, SQ) directly correlated. Banks with crypto custody see fee uplift in bull cycles.",
        "Communication Services": "Crypto bull cycles drive user engagement and ad spend in gaming/social verticals.",
        "default":                "Bitcoin is the clearest real-time leading indicator of institutional risk appetite.",
    },
    "ai_tech": {
        "Information Technology": "AI model leadership (Google vs OpenAI vs DeepSeek) determines cloud/chip revenue share. Google maintaining #1 = GCP, TPU demand sustained. DeepSeek winning = inference cost collapse, bad for NVDA compute revenue.",
        "Communication Services": "AI search and advertising efficiency determines long-run ad revenue per query. Google/Meta AI leadership directly affects margin structure.",
        "default":                "AI competitive dynamics shape cloud revenue, chip demand, and productivity gains across the economy.",
    },
    "recession": {
        "Consumer Discretionary": "Recession = primary existential risk. Revenue drops 15-30% in severe downturns. First sector sold in risk-off.",
        "Financials":             "Loan loss provisions surge, deal flow collapses, credit card defaults rise. Banks typically fall 30-50% peak-to-trough.",
        "Industrials":            "Capex freezes early. Backlog-heavy businesses hold better than spot-order; services more resilient than equipment.",
        "Information Technology": "Enterprise IT budgets cut immediately. SaaS with multi-year contracts outperforms; hardware and ad-driven models hit hard.",
        "Consumer Staples":       "Recession-resistant — food, household products demand inelastic. Classic defensive rotation destination.",
        "Utilities":              "Recession-proof: electricity/gas demand inelastic. Rate cuts that accompany recessions boost utility valuations.",
        "Healthcare":             "Defensive: healthcare demand inelastic, government programs support pharma/hospital revenue.",
        "Materials":              "Highly cyclical — copper and steel crash with industrial production. Worst sector in recessions after Discretionary.",
        "default":                "22% recession probability is the key tail risk — cyclicals vulnerable, defensives outperform.",
    },
}


def _agg_fomc_next(markets: list[dict]) -> tuple[str, float]:
    """For next-meeting markets: probability distribution across outcomes."""
    cut50 = next((m["probability"] for m in markets if "decrease" in m["question"].lower() and "50" in m["question"]), 0)
    cut25 = next((m["probability"] for m in markets if "decrease" in m["question"].lower() and "25" in m["question"]), 0)
    hold  = next((m["probability"] for m in markets if "no change" in m["question"].lower()), 0)
    hike25 = next((m["probability"] for m in markets if "increase" in m["question"].lower() and "25" in m["question"]), 0)
    hike50 = next((m["probability"] for m in markets if "increase" in m["question"].lower() and "50" in m["question"]), 0)
    total_cut = round(cut25 + cut50, 2)
    if total_cut + hold + hike25 + hike50 == 0:
        return f"{len(markets)} meeting scenarios", 0.5
    parts = []
    if total_cut > 0.01: parts.append(f"cut {total_cut:.0%}")
    if hold > 0.01: parts.append(f"hold {hold:.0%}")
    if hike25 + hike50 > 0.01: parts.append(f"hike {hike25+hike50:.0%}")
    dominant = max(total_cut, hold, hike25 + hike50)
    return " · ".join(parts) if parts else f"{len(markets)} scenarios", dominant


def _agg_fed_annual(markets: list[dict]) -> tuple[str, float]:
    """Expected number of 2026 cuts from Polymarket probabilities."""
    no_cut = next((m["probability"] for m in markets if "no fed rate cut" in m["question"].lower()), None)
    hike   = next((m["probability"] for m in markets if "rate hike in 2026" in m["question"].lower()), 0)
    if no_cut is not None:
        p_any_cut = round(1 - no_cut, 2)
        signal = f"no cuts {no_cut:.0%} · ≥1 cut {p_any_cut:.0%}"
        if hike > 0.05:
            signal += f" · hike {hike:.0%}"
        return signal, max(no_cut, p_any_cut)
    return f"{len(markets)} annual rate scenarios", 0.5


def _agg_btc(markets: list[dict]) -> tuple[str, float]:
    """BTC implied price range from multiple price-level markets."""
    def _prob_for(keyword: str) -> float:
        m = next((m for m in markets if keyword in m["question"]), None)
        return m["probability"] if m else 0.0
    above_100 = _prob_for("100,000") or _prob_for("100k")
    above_150 = _prob_for("150k") or _prob_for("150,000")
    dip_55    = _prob_for("55,000")
    dip_45    = _prob_for("45,000")
    parts = [f"above 100k: {above_100:.0%}", f"dip below 55k: {dip_55:.0%}", f"dip below 45k: {dip_45:.0%}"]
    return " · ".join(parts), above_100


def _agg_conflict(markets: list[dict]) -> tuple[str, float]:
    probs = [m["probability"] for m in markets]
    peak = max(probs) if probs else 0
    top = sorted(markets, key=lambda m: m["probability"], reverse=True)[:2]
    desc = " · ".join(f"{m['question'][:50]}… {m['probability']:.0%}" for m in top)
    return desc, peak


def _agg_generic(markets: list[dict]) -> tuple[str, float]:
    if not markets:
        return "no markets", 0
    m = max(markets, key=lambda x: x["volume"])
    return f"{m['question'][:60]}… {m['probability']:.0%}", m["probability"]


def _group(gid: str, name: str, icon: str, markets: list[dict],
           relevant_sectors: set, agg_fn) -> dict | None:
    if not markets:
        return None
    signal_text, dominant_prob = agg_fn(markets)
    return {
        "id": gid,
        "name": name,
        "icon": icon,
        "signal_text": signal_text,
        "dominant_prob": dominant_prob,
        "relevant_sectors": relevant_sectors,
        "impact": IMPACT.get(gid, {}),
        "markets": sorted(markets, key=lambda x: x["volume"], reverse=True),
    }


def group_polymarket_events(signals: dict) -> list[dict]:
    """Group all Polymarket markets from signals dict into named thematic groups.

    Uses geo/macro/company sub-lists from collect_all to avoid re-fetching.
    Falls back to polymarket_geo only if new keys absent (backward compat).
    """
    geo     = signals.get("polymarket_geo", [])
    macro   = signals.get("polymarket_macro", [])
    company = signals.get("polymarket_company", [])
    all_m   = geo + macro + company

    def pick(fn) -> list[dict]:
        return [m for m in all_m if fn(m["question"].lower())]

    # FOMC next-meeting: "decrease interest rate … after the … meeting"
    fomc_next_m = pick(lambda q: ("decrease interest rate" in q or "increase interest rate" in q
                                   or "no change in fed interest rate" in q) and "after the" in q)
    # Fed annual path: "N fed rate cuts happen in 2026", "rate hike in 2026"
    fed_annual_m = pick(lambda q: ("fed rate cut" in q or "rate cut happen in 2026" in q
                                    or "rate hike in 2026" in q))
    # Iran/Middle East
    iran_m = pick(lambda q: any(k in q for k in ["iran", "invade iran", "iranian regime", "iranian"]))
    # China/Taiwan
    tw_m = pick(lambda q: any(k in q for k in ["taiwan", "china invade", "china x taiwan",
                                                  "china blockade", "china military clash"]))
    # Russia/Ukraine/NATO
    ru_m = pick(lambda q: any(k in q for k in ["russia invade nato", "zelenskyy", "ukraine peace",
                                                  "ukraine joins nato", "ukraine signs", "ukraine election"]))
    # Bitcoin price range
    btc_m = pick(lambda q: any(k in q for k in ["bitcoin hit", "bitcoin reach", "bitcoin dip",
                                                   "btc hit", "btc reach"]))
    # AI / tech leadership
    ai_m = pick(lambda q: any(k in q for k in ["best ai model", "top ai model", "ipo before",
                                                  "ipo by", "ipo day", "largest company",
                                                  "spacex", "databricks", "kraken ipo", "stripe ipo"]))
    # Recession
    rec_m = pick(lambda q: "recession" in q and "by end of" in q)

    groups = [
        _group("fomc_next",     "FOMC Next Meeting",       "", fomc_next_m,
               {"Financials","Information Technology","Utilities","Real Estate","Consumer Discretionary"}, _agg_fomc_next),
        _group("fed_annual",    "Fed Rate Path 2026",      "", fed_annual_m,
               {"Financials","Information Technology","Utilities","Real Estate","Consumer Discretionary","Consumer Staples"}, _agg_fed_annual),
        _group("iran_me",       "Iran / Middle East",       "", iran_m,
               {"Energy","Industrials","Consumer Discretionary","Consumer Staples","Materials"}, _agg_conflict),
        _group("china_taiwan",  "China / Taiwan",           "", tw_m,
               {"Information Technology","Industrials","Consumer Discretionary","Materials","Communication Services"}, _agg_conflict),
        _group("russia_ukraine","Russia / Ukraine / NATO",  "", ru_m,
               {"Energy","Industrials","Materials","Financials"}, _agg_conflict),
        _group("bitcoin",       "Bitcoin Range",            "", btc_m,
               {"Information Technology","Financials","Communication Services"}, _agg_btc),
        _group("ai_tech",       "AI / Tech / IPO Pipeline", "", ai_m,
               {"Information Technology","Communication Services"}, _agg_generic),
        _group("recession",     "US Recession Risk",        "", rec_m,
               {"Consumer Discretionary","Financials","Industrials","Materials",
                "Information Technology","Consumer Staples","Utilities","Healthcare"}, _agg_generic),
    ]
    return [g for g in groups if g is not None and g["markets"]]


def _fmt_pct(v, suffix="%") -> str:
    if v is None:
        return "N/A"
    return f"{v:+.1f}{suffix}" if v != 0 else f"0{suffix}"


def build_deep_dive(ticker: str, signals: dict) -> dict:
    """Synthesize all signals for one ticker into buy/hold rationale."""
    score_entry = signals["fast_scores"].get(ticker, {})
    score = score_entry.get("score", 50)
    sector = score_entry.get("sector", "")
    sub = score_entry.get("sub_scores", {})

    fund = signals.get("fundamentals", {}).get(ticker, {})
    insider = signals.get("insider", {}).get(ticker, {})
    options = signals.get("options", {}).get(ticker, {})
    short = signals.get("short_interest", {}).get(ticker, {})
    macro = signals.get("macro", {})
    btc = signals.get("btc", {})

    pros, cons = [], []

    # ── Fundamentals ────────────────────────────────────────────────────────
    pe = fund.get("pe_ratio")
    roe = fund.get("return_on_equity")
    rev_g = fund.get("revenue_growth_yoy")
    earn_g = fund.get("earnings_growth_yoy")
    de = fund.get("debt_to_equity")
    fcf = fund.get("free_cashflow_usd")

    if pe and pe < 15:
        pros.append(f"Cheap valuation — P/E {pe:.1f}x (below 15x threshold)")
    elif pe and pe > 40:
        cons.append(f"Rich valuation — P/E {pe:.1f}x demands continued growth execution")

    if roe and roe > 25:
        pros.append(f"High capital efficiency — ROE {roe:.1f}%")
    elif roe and roe < 8:
        cons.append(f"Low return on equity — ROE {roe:.1f}%")

    if rev_g is not None:
        if rev_g > 25:
            pros.append(f"Strong revenue growth — {rev_g:.1f}% YoY")
        elif rev_g > 10:
            pros.append(f"Solid revenue growth — {rev_g:.1f}% YoY")
        elif rev_g < 0:
            cons.append(f"Revenue declining — {rev_g:.1f}% YoY")

    if earn_g is not None and earn_g > 50:
        pros.append(f"Exceptional earnings growth — {earn_g:.1f}% YoY")
    elif earn_g is not None and earn_g < -20:
        cons.append(f"Earnings contracting — {earn_g:.1f}% YoY")

    if de is not None:
        if de < 30:
            pros.append(f"Conservative balance sheet — D/E {de:.0f}%")
        elif de > 250:
            cons.append(f"Highly leveraged — D/E {de:.0f}% (rate-sensitive)")

    if fcf and fcf > 0:
        fcf_b = fcf / 1e9
        pros.append(f"Positive free cash flow — ${fcf_b:.1f}B")

    # ── Analyst consensus ────────────────────────────────────────────────────
    analyst_score = fund.get("analyst_score")  # 1=strong buy, 5=strong sell
    analyst_rating = fund.get("analyst_rating", "")
    target_price = fund.get("target_price")
    n_analysts = fund.get("n_analysts", 0)
    upside = fund.get("upside_pct")
    next_earnings = fund.get("next_earnings")
    current_price = fund.get("current_price")

    if analyst_score is not None and n_analysts >= 3:
        if analyst_score <= 1.8:
            pros.append(f"Strong analyst consensus: {n_analysts} analysts rate '{analyst_rating}' (score {analyst_score:.1f}/5)")
        elif analyst_score <= 2.5:
            pros.append(f"Analyst consensus buy: {n_analysts} analysts rate '{analyst_rating}'")
        elif analyst_score >= 3.5:
            cons.append(f"Weak analyst consensus: {n_analysts} analysts rate '{analyst_rating}' (score {analyst_score:.1f}/5)")

    if upside is not None:
        if upside >= 20:
            pros.append(f"Significant upside to analyst consensus target: +{upside:.1f}% (target ${target_price} vs current ${current_price})")
        elif upside >= 10:
            pros.append(f"Upside to analyst target: +{upside:.1f}% (${target_price})")
        elif upside < -5:
            cons.append(f"Trading above analyst consensus target: {upside:.1f}% downside (target ${target_price})")

    if next_earnings:
        from datetime import date
        try:
            ed = date.fromisoformat(next_earnings)
            days_to = (ed - date.today()).days
            if 0 < days_to <= 30:
                cons.append(f"Earnings in {days_to} days ({next_earnings}) — binary event risk")
            elif 0 < days_to <= 60:
                pros.append(f"Earnings catalyst approaching: {next_earnings} ({days_to} days)")
        except Exception:
            pass

    # ── Insider flow ─────────────────────────────────────────────────────────
    buy_f = insider.get("buy_filings", 0)
    sell_f = insider.get("sell_filings", 0)
    ins_sig = insider.get("signal", "neutral")
    if ins_sig == "strong_buy" and buy_f > 0:
        pros.append(f"Insider buying — {buy_f} buy filing(s) vs {sell_f} sells in 90 days")
    elif sell_f > 3 and buy_f == 0:
        cons.append(f"Insider selling — {sell_f} sell filing(s), no buys in 90 days")
    elif ins_sig == "neutral":
        cons.append("No insider buy activity in last 90 days")

    # ── Options flow ─────────────────────────────────────────────────────────
    pcr = options.get("put_call_ratio")
    opt_sig = options.get("signal", "")
    if pcr is not None:
        if pcr < 0.4:
            pros.append(f"Heavily call-skewed options (PCR {pcr:.2f}) — market positioning strongly bullish")
        elif pcr < 0.7:
            pros.append(f"Bullish options positioning (PCR {pcr:.2f})")
        elif pcr > 1.5:
            cons.append(f"Put-heavy options flow (PCR {pcr:.2f}) — market hedging downside")
        elif pcr > 1.0:
            cons.append(f"Elevated put-call ratio (PCR {pcr:.2f}) — cautious market sentiment")

    # ── Short interest ───────────────────────────────────────────────────────
    sf = short.get("short_float_pct")
    short_sig = short.get("signal", "")
    if sf is not None:
        if sf < 2:
            pros.append(f"Very low short interest ({sf:.1f}%) — bears not positioned against this")
        elif sf > 15:
            cons.append(f"High short interest ({sf:.1f}%) — significant bearish conviction or squeeze risk")
        elif sf > 8:
            cons.append(f"Moderate-high short interest ({sf:.1f}%)")

    # ── Macro regime ─────────────────────────────────────────────────────────
    regime = macro.get("regime", "")
    tailwinds = macro.get("sector_tailwinds", [])
    headwinds = macro.get("sector_headwinds", [])
    if sector in tailwinds:
        pros.append(f"Sector tailwind — {sector} benefits from current {regime} macro regime")
    elif sector in headwinds:
        cons.append(f"Sector headwind — {sector} faces pressure in current {regime} regime")

    # ── BTC / liquidity ──────────────────────────────────────────────────────
    btc_regime = btc.get("regime", "")
    tech_sectors = {"Information Technology", "Communication Services", "Consumer Discretionary"}
    if btc_regime == "risk_on" and sector in tech_sectors:
        pros.append(f"BTC risk-on signal supports growth/tech positioning")
    elif btc_regime == "safe_haven" and sector in tech_sectors:
        cons.append(f"BTC/Gold divergence indicates risk-off — headwind for growth stocks")

    # ── Relevant event groups ────────────────────────────────────────────────
    all_groups = group_polymarket_events(signals)
    relevant_groups = [g for g in all_groups if sector in g["relevant_sectors"]]

    # Add event-group signals to pros/cons
    for grp in relevant_groups[:4]:
        prob = grp["dominant_prob"]
        gid = grp["id"]
        if gid in ("fomc_next", "fed_annual") and sector in {"Financials"}:
            no_cut = next((m["probability"] for m in grp["markets"] if "no fed rate cut" in m["question"].lower()
                           or "no change" in m["question"].lower()), 0)
            if no_cut > 0.5:
                pros.append(f"Rate hold (Polymarket: {no_cut:.0%}) — NIM stability for banks")
        elif gid in ("fomc_next", "fed_annual") and sector in {"Utilities", "Real Estate", "Information Technology"}:
            no_cut = next((m["probability"] for m in grp["markets"] if "no fed rate cut" in m["question"].lower()
                           or "no change" in m["question"].lower()), 0)
            if no_cut > 0.5:
                cons.append(f"Rate hold (Polymarket: {no_cut:.0%}) — valuation headwind for {sector}")
        elif gid == "iran_me" and sector == "Energy":
            conflict_p = max((m["probability"] for m in grp["markets"]
                              if "invade" in m["question"].lower() or "regime" in m["question"].lower()), default=0)
            if conflict_p > 0.15:
                pros.append(f"Iran conflict risk {conflict_p:.0%} (Polymarket) — Hormuz closure scenario = oil price spike")
        elif gid == "china_taiwan" and sector == "Information Technology":
            if prob > 0.05:
                cons.append(f"China/Taiwan risk {prob:.0%} (Polymarket) — TSMC supply chain tail risk for semiconductors")
        elif gid == "recession":
            if sector in {"Consumer Discretionary", "Financials", "Materials"} and prob > 0.15:
                cons.append(f"Recession risk {prob:.0%} (Polymarket) — cyclical sector most exposed")
            elif sector in {"Consumer Staples", "Utilities", "Healthcare"} and prob > 0.15:
                pros.append(f"Defensive positioning: outperforms in recession scenarios ({prob:.0%} market-implied risk)")

    # ── Verdict ──────────────────────────────────────────────────────────────
    if score >= 65:
        verdict = "STRONG BUY"
        verdict_color = "#00a854"
    elif score >= 60:
        verdict = "BUY"
        verdict_color = "#1db954"
    elif score >= 55:
        verdict = "WATCH"
        verdict_color = "#f39c12"
    else:
        verdict = "HOLD"
        verdict_color = "#888"

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "pros": pros,
        "cons": cons,
        "sub_scores": sub,
        "relevant_groups": [g for g in all_groups if sector in g["relevant_sectors"]],
        "macro_context": f"{regime.title()} regime · Risk: {macro.get('risk_level','?')} · BTC: {btc_regime.replace('_',' ')}",
        "fundamentals": fund,
        "insider": insider,
        "options": options,
        "short": short,
    }

def build_world_summary(signals: dict) -> dict:
    macro = signals.get("macro", {})
    btc = signals.get("btc", {})
    sentiment = signals.get("sentiment", {})
    groups = group_polymarket_events(signals)
    news = signals.get("news", {})
    gdelt = signals.get("gdelt", [])

    interactions = []
    for grp in groups:
        gid = grp["id"]
        mkts = grp["markets"]
        if not mkts:
            continue

        if gid == "iran_me":
            war_p = max((m["probability"] for m in mkts
                         if "invade" in m["question"].lower() or "regime" in m["question"].lower()), default=0)
            deal_p = next((m["probability"] for m in mkts if "deal" in m["question"].lower()), 0)
            if war_p > 0.1 or deal_p > 0.3:
                interactions.append({
                    "title": f"Iran — war {war_p:.0%}, nuclear deal {deal_p:.0%}",
                    "chains": [
                        f"War scenario ({war_p:.0%}): A US-Iran conflict would shut the Strait of Hormuz, "
                        f"which carries ~20% of global seaborne oil. Brent crude would spike USD 20-40/bbl. "
                        f"Energy producers gain directly on higher realized prices; airlines and consumer-facing "
                        f"companies face fuel and logistics cost surges that compress margins.",
                        f"Deal scenario ({deal_p:.0%}): A nuclear agreement re-admits ~1-2M bbl/day of Iranian "
                        f"supply to markets. This removes the current war risk premium from oil prices, "
                        f"likely softening crude USD 5-10/bbl. Bearish for oil producers; positive for "
                        f"transport, airlines, and consumer spending power.",
                    ]
                })

        elif gid == "fomc_next":
            hold_p = next((m["probability"] for m in mkts if "no change" in m["question"].lower()), 0)
            cut_p = sum(m["probability"] for m in mkts if "decrease" in m["question"].lower())
            interactions.append({
                "title": f"Fed June Meeting — hold {hold_p:.0%}, cut {cut_p:.0%}",
                "chains": [
                    f"Hold ({hold_p:.0%}): The Fed keeps the target rate unchanged. Banks maintain current "
                    f"net interest margins (NIM) — the spread between deposit costs and lending rates stays "
                    f"wide, supporting bank profitability. For growth stocks and REITs, the high discount "
                    f"rate persists, keeping forward earnings and property valuations compressed.",
                    f"Cut ({cut_p:.0%}): A rate reduction lowers the risk-free rate, mechanically expanding "
                    f"equity multiples for tech and REITs (each 25bp cut adds ~2-4% to DCF fair values). "
                    f"Bank NIMs compress immediately as deposit pricing lags lending rate declines.",
                ]
            })

        elif gid == "fed_annual":
            no_cut_p = next((m["probability"] for m in mkts if "no fed rate cut" in m["question"].lower()), 0)
            hike_p = next((m["probability"] for m in mkts if "rate hike in 2026" in m["question"].lower()), 0)
            if no_cut_p > 0.4:
                interactions.append({
                    "title": f"2026 Rate Path — no cuts {no_cut_p:.0%}, hike risk {hike_p:.0%}",
                    "chains": [
                        f"Higher-for-longer ({no_cut_p:.0%} probability): Elevated rates sustain discount "
                        f"rate pressure on growth stocks and REITs for the full year. Companies with "
                        f"near-term cash flows outperform those whose value is in distant earnings. "
                        f"Banks benefit from stable NIMs but face rising credit default risk in H2.",
                        f"Consumer impact: High mortgage, auto, and credit card rates reduce household "
                        f"disposable income. Discretionary spending is squeezed while staples demand "
                        f"holds. Watch consumer delinquency rates as a leading indicator of stress.",
                    ]
                })

        elif gid == "china_taiwan":
            clash_p = max((m["probability"] for m in mkts), default=0) if mkts else 0
            if clash_p > 0.04:
                interactions.append({
                    "title": f"China / Taiwan — military conflict {clash_p:.0%}",
                    "chains": [
                        f"Taiwan produces ~90% of the world's advanced semiconductors (TSMC). Even a "
                        f"partial conflict or blockade would create a multi-year supply shock for chips — "
                        f"worse than the 2021 shortage. NVIDIA, AMD, Apple, AMAT, and LRCX face direct "
                        f"exposure. Defense contractors benefit from accelerated NATO/US military spending.",
                        f"Secondary effect: A conflict would trigger broad China decoupling — tariffs, "
                        f"sanctions, and supply chain restructuring across electronics, industrials, and "
                        f"consumer goods. The market currently prices this at {clash_p:.0%}, "
                        f"but the tail impact if it occurs is systemic.",
                    ]
                })

        elif gid == "bitcoin":
            above_100 = next((m["probability"] for m in mkts
                              if "100,000" in m["question"] and "reach" in m["question"].lower()), 0)
            dip_55 = next((m["probability"] for m in mkts if "55,000" in m["question"]), 0)
            interactions.append({
                "title": f"Bitcoin — above USD 100k: {above_100:.0%}, below USD 55k: {dip_55:.0%}",
                "chains": [
                    f"Bitcoin above USD 100k ({above_100:.0%}): Sustained BTC strength signals broad "
                    f"institutional risk appetite and liquidity. Growth stocks, fintech, and crypto-adjacent "
                    f"equities benefit. Defensive names (utilities, gold miners) lag in this environment.",
                    f"Bitcoin below USD 55k ({dip_55:.0%}): A significant BTC decline is historically "
                    f"correlated with broader risk-off episodes — tightening liquidity, rising credit spreads, "
                    f"and multiple compression in high-growth equities. Defensives outperform in this scenario.",
                ]
            })

        elif gid == "recession":
            rec_p = mkts[0]["probability"] if mkts else 0
            if rec_p > 0.1:
                interactions.append({
                    "title": f"US Recession — {rec_p:.0%} probability by end of 2026",
                    "chains": [
                        f"Cyclical exposure ({rec_p:.0%} risk): Consumer Discretionary, Financials, and "
                        f"Materials historically fall 30-50% peak-to-trough in recessions. Capex freezes "
                        f"early, loan losses surge, and commodity demand collapses with industrial production.",
                        f"Defensive positioning: Consumer Staples, Utilities, and Healthcare are structurally "
                        f"insulated — demand for food, electricity, and healthcare is inelastic regardless "
                        f"of economic cycle. These sectors typically outperform cyclicals by 20-40% in recessions.",
                    ]
                })

    headlines = news.get("headlines", []) if isinstance(news, dict) else []
    theme_kw = {
        "AI & Technology": ["ai", "artificial intelligence", "nvidia", "chip", "semiconductor", "openai", "deepseek"],
        "Energy & Commodities": ["oil", "energy", "gas", "crude", "opec", "copper", "gold"],
        "Fed & Rates": ["fed", "rate", "interest", "inflation", "cpi", "powell"],
        "Geopolitics": ["iran", "china", "russia", "ukraine", "taiwan", "war", "sanctions"],
        "Earnings": ["earnings", "revenue", "profit", "beat", "miss", "guidance"],
        "IPO & M&A": ["ipo", "merger", "acquisition", "buyout", "deal"],
    }
    theme_counts = {t: 0 for t in theme_kw}
    for h in headlines:
        title_lower = h.get("title", "").lower()
        for theme, kws in theme_kw.items():
            if any(k in title_lower for k in kws):
                theme_counts[theme] += 1
    news_themes = [(t, c) for t, c in sorted(theme_counts.items(), key=lambda x: -x[1]) if c > 0]

    escalating = [r["region"] for r in gdelt if r.get("trend") == "escalating"]

    return {
        "regime": macro.get("regime", "unknown"),
        "risk_level": macro.get("risk_level", "medium"),
        "btc_regime": btc.get("regime", ""),
        "fg_score": sentiment.get("fear_greed", {}).get("score"),
        "fg_rating": sentiment.get("fear_greed", {}).get("rating", ""),
        "vix": sentiment.get("vix", {}).get("vix"),
        "vix_regime": sentiment.get("vix", {}).get("regime", ""),
        "vix_1y_low": sentiment.get("vix", {}).get("vix_1y_low"),
        "vix_1y_high": sentiment.get("vix", {}).get("vix_1y_high"),
        "spread_regime": sentiment.get("credit_spreads", {}).get("spread_regime", ""),
        "interactions": interactions,
        "news_themes": news_themes[:4],
        "escalating_regions": escalating,
        "tailwinds": macro.get("sector_tailwinds", []),
        "headwinds": macro.get("sector_headwinds", []),
    }


DATA_DIR = Path("data")
SIGNALS_FILE = DATA_DIR / "signals.json"
RATINGS_FILE = DATA_DIR / "ratings_report.json"

GRADE_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
               "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B", "CCC", "CC"]
GRADE_RANK = {g: len(GRADE_ORDER) - i for i, g in enumerate(GRADE_ORDER)}

GRADE_COLOR = {
    "AAA": "#00a854", "AA+": "#00a854", "AA": "#1db954", "AA-": "#52c27d",
    "A+": "#85d49f", "A": "#b3e5c4", "A-": "#d4f0e0",
    "BBB+": "#fff3cd", "BBB": "#ffe99a", "BBB-": "#ffd966",
    "BB+": "#ffb347", "BB": "#ff8c00", "BB-": "#ff6600",
    "B": "#e84118", "CCC": "#c0392b", "CC": "#922b21",
}


@st.cache_data(ttl=300)
def load_signals():
    if not SIGNALS_FILE.exists():
        return None
    with open(SIGNALS_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_ratings():
    if not RATINGS_FILE.exists():
        return None
    with open(RATINGS_FILE) as f:
        return json.load(f)


def grade_badge(grade: str) -> str:
    color = GRADE_COLOR.get(grade, "#888")
    return f'<span style="background:{color};color:#000;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em">{grade}</span>'


def scores_to_df(fast_scores: dict) -> pd.DataFrame:
    rows = []
    for ticker, v in fast_scores.items():
        rows.append({
            "Ticker": ticker,
            "Score": v.get("score", 0),
            "Grade": v.get("grade", "B"),
            "Sector": v.get("sector", ""),
            "Region": v.get("region", ""),
            "Type": v.get("type", "stock"),
        })
    df = pd.DataFrame(rows)
    df["GradeRank"] = df["Grade"].map(lambda g: GRADE_RANK.get(g, 0))
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TR Investment Ratings",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("Trade Republic — Weekly Investment Ratings")

signals = load_signals()
ratings = load_ratings()

if signals is None:
    st.warning("No signals.json found. Run `python collect_all.py` first.")
    st.stop()

collected_at = signals.get("collected_at", "unknown")
st.caption(f"Data collected: {collected_at}  ·  Universe: {signals.get('universe_size', '?')} assets")

# ── World View Summary ────────────────────────────────────────────────────────
world = build_world_summary(signals)

with st.container():
    st.markdown("## World View")

    _regime_desc = {
        "growth":      "GDP expanding, earnings growing, risk assets outperforming. Favours cyclicals: Technology, Consumer Discretionary, Financials.",
        "inflation":   "Prices rising faster than growth. Hard assets and commodities outperform. Favours Energy, Materials, short-duration assets.",
        "stagflation": "Slow growth + high inflation. Worst combo for equities. Defensives (Staples, Healthcare, Utilities) and commodities preferred.",
        "deflation":   "Falling prices, weak demand. Bonds outperform. Defensives and dividend stocks hold value; avoid cyclicals.",
        "recession":   "Economic contraction. Earnings falling broadly. Capital preservation priority; Staples, Utilities, Healthcare, cash.",
    }
    _risk_desc = {
        "low":    "Low volatility, tight credit spreads, strong momentum. Risk assets broadly supported.",
        "medium": "Mixed signals — some caution warranted. Selective positioning; favour quality over momentum.",
        "high":   "Elevated uncertainty. Credit spreads widening, volatility rising. Reduce cyclical exposure, increase defensives.",
    }

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Macro Regime", world["regime"].title(),
              help=_regime_desc.get(world["regime"], "Current macroeconomic regime derived from FRED indicators: Fed funds rate, CPI trend, yield curve, GDP growth."))
    m2.metric("Risk Level", world["risk_level"].upper(),
              help=_risk_desc.get(world["risk_level"], "Composite risk assessment combining regime uncertainty, credit spreads, and VIX level."))
    vix = world.get("vix")
    m3.metric("VIX", f"{vix:.1f}  {world['vix_regime'].replace('_',' ')}" if vix else "N/A",
              help=(
                  "CBOE Volatility Index — measures 30-day implied volatility of S&P 500 options. "
                  "Below 15: complacency. 15-20: calm. 20-25: elevated. 25-30: fear. Above 30: extreme fear. "
                  f"Current 1Y range: {world.get('vix_1y_low','?')} – {world.get('vix_1y_high','?')}."
              ))
    fg = world.get("fg_score")
    m4.metric("Fear / Greed", f"{fg:.0f}  {world['fg_rating'].replace('_',' ').title()}" if fg else "N/A",
              help=(
                  "CNN Fear & Greed Index (0–100). Aggregates 7 signals: market momentum, stock price strength, "
                  "stock price breadth, put/call ratio, junk bond demand, market volatility, safe haven demand. "
                  "0–25: Extreme Fear. 25–45: Fear. 45–55: Neutral. 55–75: Greed. 75–100: Extreme Greed. "
                  "Contrarian signal: extreme greed often precedes corrections."
              ))
    m5.metric("Credit Spreads", world["spread_regime"].title() or "N/A",
              help=(
                  "HYG/LQD spread proxy — measures the difference in returns between high-yield (HYG) and "
                  "investment-grade (LQD) bonds. Widening spreads signal rising default risk and credit stress, "
                  "typically a leading indicator of equity market weakness. Tightening = risk appetite improving."
              ))
    m6.metric("Liquidity Signal", world["btc_regime"].replace("_", " ").title() or "N/A",
              help=(
                  "Bitcoin vs Gold relative performance as a liquidity proxy. "
                  "Risk-On: BTC outperforming Gold — institutional risk appetite elevated, growth assets supported. "
                  "Safe-Haven: Gold outperforming BTC — flight to quality, defensive rotation likely. "
                  "Risk-Off: both falling — broad liquidity withdrawal."
              ))

    st.divider()

    col_ev, col_right = st.columns([3, 1])

    with col_ev:
        st.markdown("#### Key Events and Market Interactions")
        for item in world["interactions"]:
            st.markdown(f"**{item['title']}**")
            for chain in item["chains"]:
                st.markdown(chain)
            st.markdown("---")

    with col_right:
        # ── Cross-source market themes ─────────────────────────────────────
        themes = signals.get("themes", [])
        retail = signals.get("retail_trend", {})
        st.markdown("#### Market Themes")
        st.caption("Score = news 35% + WSB 25% + Polymarket 25% + macro 15%")

        if themes:
            for t in themes[:8]:
                comp = t["composite"]
                if comp < 5:
                    continue
                bar_n = int(min(comp, 50) / 50 * 10)
                bar = "█" * bar_n + "░" * (10 - bar_n)
                sources = []
                if t["news_score"] > 5:  sources.append(f"News {t['news_score']:.0f}")
                if t["wsb_score"] > 10:  sources.append(f"WSB {t['wsb_score']:.0f}")
                if t["poly_score"] > 5:  sources.append(f"Poly {t['poly_score']:.0f}")
                src_str = " · ".join(sources) if sources else "low signal"
                st.markdown(f"`{bar}` **{t['label']}** {comp:.0f}  \n<span style='color:#888;font-size:0.78em'>{src_str}</span>", unsafe_allow_html=True)

        st.markdown("")
        if retail:
            direction = retail.get("trend_direction", "mixed")
            dir_label = {"risk_on": "RISK ON", "defensive": "DEFENSIVE", "mixed": "MIXED"}.get(direction, direction.upper())
            st.markdown(f"**Retail: {dir_label}**")
            st.caption(retail.get("narrative", ""))

        st.markdown("")
        st.markdown("**Sector Outlook**")
        for s in world["tailwinds"][:3]:
            st.markdown(f"+ {s}")
        for s in world["headwinds"][:3]:
            st.markdown(f"- {s}")

    st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ratings, tab_signals, tab_events, tab_news = st.tabs(
    ["Ratings", "Macro Signals", "Events", "News"]
)

# ── Shared deep-dive renderer (used inside each rating expander) ──────────────
def _render_deep_dive(ticker: str, score: int, grade: str, sector: str, asset_type: str, signals: dict):
    dive = build_deep_dive(ticker, signals)
    company_name = signals.get("fundamentals", {}).get(ticker, {}).get("name", "")
    badge_color = GRADE_COLOR.get(grade, "#888")
    verdict = dive["verdict"]
    v_color = dive["verdict_color"]

    header = (
        f'<span style="font-size:1.1em;font-weight:bold">{ticker}</span>'
        + (f'&nbsp;&nbsp;<span style="color:#aaa">{company_name}</span>' if company_name and company_name != ticker else "")
        + f'&nbsp;&nbsp;<span style="background:{badge_color};color:#000;padding:2px 8px;border-radius:3px;font-weight:bold;font-size:0.82em">{grade}</span>'
        + f'&nbsp;&nbsp;<span style="background:{v_color};color:#fff;padding:2px 10px;border-radius:3px;font-weight:bold;font-size:0.82em">{verdict}</span>'
        + f'&nbsp;&nbsp;<span style="color:#888;font-size:0.82em">{sector} · {asset_type} · Score {score}/100</span>'
    )
    st.markdown(header, unsafe_allow_html=True)
    st.caption(dive["macro_context"])
    st.divider()

    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("**Pros**")
        for p in dive["pros"]:
            st.markdown(f"- {p}")
        if not dive["pros"]:
            st.caption("No strong positives identified")
    with col_con:
        st.markdown("**Cons**")
        for c in dive["cons"]:
            st.markdown(f"- {c}")
        if not dive["cons"]:
            st.caption("No major negatives identified")

    st.divider()

    sub = dive["sub_scores"]
    if sub:
        st.markdown("**Signal Breakdown**")
        sub_labels = {
            "earnings":    "Earnings / Analyst (22%)",
            "insider":     "Insider Flow (18%)",
            "macro":       "Macro Regime (12%)",
            "geo":         "Geopolitical Events (10%)",
            "fundamentals":"Fundamentals (15%)",
            "options":     "Options Flow (5%)",
            "wsb_short":   "WSB / Short (3%)",
            "momentum":    "Market Momentum (15%)",
        }
        sub_df = pd.DataFrame([
            {"Signal": sub_labels.get(k, k), "Score": v,
             "Color": "#2e7d32" if v >= 60 else ("#c62828" if v < 40 else "#e65100")}
            for k, v in sub.items()
        ])
        fig = px.bar(sub_df, x="Score", y="Signal", orientation="h",
                     color="Color", color_discrete_map="identity", range_x=[0, 100])
        fig.add_vline(x=50, line_dash="dash", line_color="#555")
        fig.update_layout(height=240, margin=dict(t=8, b=8, l=0, r=16),
                          showlegend=False, yaxis_title=None,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key=f"subchart_{ticker}")

    fund = dive["fundamentals"]
    ins  = dive["insider"]
    opt  = dive["options"]
    sht  = dive["short"]
    if any([fund, ins, opt, sht]):
        st.markdown("**Key Metrics**")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r1c1.metric("P/E", f"{fund.get('pe_ratio'):.1f}x" if fund.get("pe_ratio") else "N/A")
        r1c2.metric("Revenue Growth", f"{fund.get('revenue_growth_yoy'):+.1f}%" if fund.get("revenue_growth_yoy") is not None else "N/A")
        r1c3.metric("ROE", f"{fund.get('return_on_equity'):.1f}%" if fund.get("return_on_equity") else "N/A")
        r1c4.metric("Analyst Rating", fund.get("analyst_rating", "N/A").title())
        upside_val = fund.get("upside_pct")
        r2c1.metric("Upside to Target", f"{upside_val:+.1f}%" if upside_val is not None else "N/A",
                    delta=f"Target ${fund.get('target_price','?')}" if upside_val is not None else None)
        r2c2.metric("Put-Call Ratio", f"{opt.get('put_call_ratio'):.2f}" if opt.get("put_call_ratio") else "N/A")
        r2c3.metric("Short Float", f"{sht.get('short_float_pct'):.1f}%" if sht.get("short_float_pct") is not None else "N/A")
        r2c4.metric("Next Earnings", fund.get("next_earnings", "N/A"))

    rel_groups = dive["relevant_groups"]
    if rel_groups:
        st.divider()
        st.markdown("**Event Risk Analysis**")
        for grp in rel_groups:
            sector_impact = grp["impact"].get(sector, grp["impact"].get("default", ""))
            prob = grp["dominant_prob"]
            alert = "HIGH" if prob > 0.4 else ("MED" if prob > 0.15 else "LOW")
            with st.expander(f"[{alert}] {grp['name']} — {grp['signal_text']}", expanded=False):
                if sector_impact:
                    st.markdown(f"**Impact on {sector}:** {sector_impact}")
                st.markdown("---")
                for ev in grp["markets"]:
                    p = ev["probability"]
                    bar_fill = int(p * 14)
                    bar = "█" * bar_fill + "░" * (14 - bar_fill)
                    st.markdown(
                        f"`{bar}` **{p:.0%}** {ev['question']}  "
                        f"<span style='color:#888;font-size:0.75em'>vol ${ev['volume']/1e6:.1f}M · ends {ev.get('end_date','?')[:10]}</span>",
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════
# TAB 1 — RATINGS
# ═══════════════════════════════════════════════════════
with tab_ratings:
    fast_scores = signals.get("fast_scores", {})
    if not fast_scores:
        st.info("No scores yet. Run collect_all.py.")
    else:
        df = scores_to_df(fast_scores)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Assets", len(df))
        col2.metric("Investment Grade (≥BBB-)", int((df["GradeRank"] >= GRADE_RANK["BBB-"]).sum()))
        col3.metric("Top Score", int(df["Score"].max()))
        col4.metric("Mean Score", f"{df['Score'].mean():.1f}")

        st.divider()

        fc1, fc2, fc3 = st.columns(3)
        sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
        types   = ["All"] + sorted(df["Type"].dropna().unique().tolist())
        regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
        sel_sector = fc1.selectbox("Sector", sectors)
        sel_type   = fc2.selectbox("Type", types)
        sel_region = fc3.selectbox("Region", regions)

        filtered = df.copy()
        if sel_sector != "All":
            filtered = filtered[filtered["Sector"] == sel_sector]
        if sel_type != "All":
            filtered = filtered[filtered["Type"] == sel_type]
        if sel_region != "All":
            filtered = filtered[filtered["Region"] == sel_region]

        # Add company name column for display
        fundamentals = signals.get("fundamentals", {})
        filtered = filtered.copy()
        filtered["Company"] = filtered["Ticker"].map(
            lambda t: fundamentals.get(t, {}).get("name", "") or ""
        )

        display_df = filtered[["Ticker", "Company", "Grade", "Score", "Sector", "Type", "Region"]].copy()
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(
            display_df,
            width="stretch",
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            },
            height=480,
        )

        st.divider()

        # ── Sector chart ────────────────────────────────────────────────────────
        sector_avg = df.groupby("Sector")["Score"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(sector_avg, x="Sector", y="Score", color="Score",
                     color_continuous_scale="RdYlGn", range_color=[40, 70])
        fig.update_layout(height=300, margin=dict(t=10, b=40),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key="sector_chart")

        st.divider()

        # ── Ticker detail ────────────────────────────────────────────────────────
        st.markdown("#### Stock Analysis")
        all_tickers = filtered["Ticker"].tolist()
        default_idx = 0
        sel_ticker = st.selectbox(
            "Select ticker",
            options=all_tickers,
            index=default_idx,
            format_func=lambda t: f"{t}  —  {fundamentals.get(t, {}).get('name', '')}",
        )

        if sel_ticker:
            entry = fast_scores.get(sel_ticker, {})
            score = entry.get("score", 0)
            grade = entry.get("grade", "?")
            sector = entry.get("sector", "")
            asset_type = entry.get("type", "stock")
            st.divider()
            _render_deep_dive(sel_ticker, score, grade, sector, asset_type, signals)

# TAB 3 — MACRO SIGNALS
# ═══════════════════════════════════════════════════════
with tab_signals:
    macro = signals.get("macro", {})
    btc = signals.get("btc", {})
    sentiment = signals.get("sentiment", {})
    vix_data = sentiment.get("vix", {})
    fg_data = sentiment.get("fear_greed", {})
    spreads = sentiment.get("credit_spreads", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    regime = macro.get("regime", "unknown").upper()
    risk = macro.get("risk_level", "?").upper()
    c1.metric("Market Regime", regime)
    c2.metric("Risk Level", risk)
    c3.metric("BTC Regime", btc.get("regime", "?").replace("_", " ").title())
    vix_val = vix_data.get("vix")
    vix_1m = vix_data.get("vix_1m_ago")
    c4.metric("VIX", f"{vix_val:.1f}" if vix_val else "N/A",
              delta=f"{vix_val - vix_1m:+.1f} vs 1M" if vix_val and vix_1m else None,
              delta_color="inverse")
    fg_score = fg_data.get("score")
    fg_prev = fg_data.get("previous_close")
    c5.metric(f"Fear/Greed", f"{fg_score:.0f} — {fg_data.get('rating','?').replace('_',' ').title()}" if fg_score else "N/A",
              delta=f"{fg_score - fg_prev:+.1f}" if fg_score and fg_prev else None)

    if vix_data or fg_data or spreads:
        with st.expander("Market Sentiment Detail", expanded=False):
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown("**VIX (Volatility)**")
                st.markdown(f"Current: **{vix_data.get('vix', 'N/A')}** ({vix_data.get('regime', '?').replace('_',' ')})")
                st.markdown(f"1W ago: {vix_data.get('vix_1w_ago', '?')} · 1M ago: {vix_data.get('vix_1m_ago', '?')}")
                st.markdown(f"1Y range: {vix_data.get('vix_1y_low', '?')} – {vix_data.get('vix_1y_high', '?')}")
                st.markdown(f"Trend: *{vix_data.get('trend', '?')}*")
            with s2:
                st.markdown("**CNN Fear & Greed**")
                score = fg_data.get("score", 50)
                bar_fill = int(score / 5)
                bar = "█" * bar_fill + "░" * (20 - bar_fill)
                st.markdown(f"`{bar}` **{score:.0f}/100**")
                st.markdown(f"Rating: **{fg_data.get('rating','?').replace('_',' ').title()}**")
                st.markdown(f"1W ago: {fg_data.get('previous_1_week', '?')} · 1M ago: {fg_data.get('previous_1_month', '?')}")
            with s3:
                st.markdown("**Credit Spreads (HYG/LQD)**")
                spread_r = spreads.get("spread_regime", "?")
                spread_c = spreads.get("spread_change_1m")
                spread_label = "HIGH" if spread_r == "widening" else ("LOW" if spread_r == "tightening" else "STABLE")
                st.markdown(f"Spreads: **{spread_r.upper()}** ({spread_c:+.2f}% 1M)" if spread_c is not None else f"Regime: {spread_r}")
                hyg = spreads.get("hyg", {})
                lqd = spreads.get("lqd", {})
                tlt = spreads.get("tlt", {})
                if hyg:
                    st.markdown(f"HYG: {hyg.get('price', '?')} ({hyg.get('return_1m_pct', 0):+.1f}% 1M)")
                if lqd:
                    st.markdown(f"LQD: {lqd.get('price', '?')} ({lqd.get('return_1m_pct', 0):+.1f}% 1M)")
                if tlt:
                    st.markdown(f"TLT: {tlt.get('price', '?')} ({tlt.get('return_1m_pct', 0):+.1f}% 1M)")

    st.divider()

    # ── Bond yields ──────────────────────────────────────────────────────────
    bonds = signals.get("bond_yields", {})
    fx = signals.get("currencies", {})
    etfs = signals.get("sector_etfs", {})
    comm_ext = signals.get("commodities_ext", {})

    bc1, bc2, bc3, bc4 = st.columns(4)
    y10 = bonds.get("yields", {}).get("10Y", {})
    y3m = bonds.get("yields", {}).get("3M", {})
    curve = bonds.get("curve_10y_3m")
    curve_r = bonds.get("curve_regime", "unknown")
    bc1.metric("10Y Yield", f"{y10.get('yield_pct','?')}%", delta=f"{y10.get('change_1m_bp','?')}bp 1M")
    bc2.metric("3M Yield", f"{y3m.get('yield_pct','?')}%", delta=f"{y3m.get('change_1m_bp','?')}bp 1M")
    bc3.metric("Yield Curve (10Y-3M)", f"{curve:+.2f}%" if curve else "N/A")
    bc4.metric("Curve Regime", curve_r.replace("_", " ").title())

    # ── Currencies ───────────────────────────────────────────────────────────
    st.markdown("**Currencies**")
    fx_sig = fx.get("_signal", {})
    if fx_sig:
        st.caption(f"DXY {fx_sig.get('dxy_level','?')} ({fx_sig.get('dollar_trend','?')}) — {fx_sig.get('implication','')}")
    fxcols = st.columns(5)
    for i, (label, sym) in enumerate([("DXY", "DXY"), ("EUR/USD", "EURUSD"), ("USD/JPY", "USDJPY"), ("GBP/USD", "GBPUSD"), ("USD/CNY", "USDCNY")]):
        d = fx.get(sym, {})
        fxcols[i].metric(label, f"{d.get('price','?')}", delta=f"{d.get('1M','?')}% 1M" if d.get("1M") is not None else None)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sector ETF Performance (1M)")
        ranked = etfs.get("ranked_1m", [])
        if ranked:
            etf_df = pd.DataFrame(ranked, columns=["Sector", "1M Return %"])
            etf_df["1M Return %"] = etf_df["1M Return %"].round(2)
            etf_df["ETF"] = etf_df["Sector"].map(
                lambda s: signals.get("sector_etfs", {}).get("by_sector", {}).get(s, {}).get("ticker", "")
            )
            etf_df.index = range(1, len(etf_df) + 1)
            st.dataframe(etf_df[["ETF", "Sector", "1M Return %"]], width="stretch",
                         column_config={"1M Return %": st.column_config.NumberColumn(format="%.2f%%")},
                         height=360)

        st.divider()
        st.subheader("FRED Indicators")
        fred = macro.get("fred_indicators", {})
        for key, info in fred.items():
            trend_dir = "up" if info.get("trend") == "rising" else ("down" if info.get("trend") == "falling" else "flat")
            st.metric(
                label=f"{info.get('name', key)} ({trend_dir})",
                value=f"{info.get('latest', '?')} {info.get('unit', '')}",
                delta=f"vs yr ago: {info.get('prev_year', '?')}",
            )

    with col_b:
        st.subheader("Extended Commodities")
        if comm_ext:
            for name, d in comm_ext.items():
                ret_1m = d.get("1M")
                st.metric(name.title(), f"{d.get('price','?')}", delta=f"{ret_1m:+.1f}% 1M" if ret_1m is not None else None)

        st.divider()
        st.subheader("Sector Tailwinds / Headwinds")
        tailwinds = macro.get("sector_tailwinds", [])
        headwinds = macro.get("sector_headwinds", [])
        for s in tailwinds:
            st.markdown(f"+ {s}")
        for s in headwinds:
            st.markdown(f"- {s}")

        st.divider()
        st.subheader("Futures")
        st.caption(macro.get("futures_summary", "No data"))

        st.divider()
        st.subheader("GDELT Conflict Indices")
        gdelt = signals.get("gdelt", [])
        for region in gdelt:
            score = region.get("conflict_score", 0)
            trend = region.get("trend", "unknown")
            trend_label = "ESCALATING" if trend == "escalating" else ("DE-ESCALATING" if trend == "de-escalating" else "STABLE")
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            st.markdown(f"**{region['region']}** `{bar}` {score:.2f} — {trend_label}")

# ═══════════════════════════════════════════════════════
# TAB 3 — EVENTS (Polymarket)
# ═══════════════════════════════════════════════════════
with tab_events:
    st.subheader("Polymarket — Grouped Event Analysis")
    _total_pm = len(signals.get("polymarket_geo", [])) + len(signals.get("polymarket_macro", [])) + len(signals.get("polymarket_company", []))
    if _total_pm == 0:
        st.info("No Polymarket data.")
    else:
        st.caption(f"{_total_pm} markets across geo · macro · company categories")
        # Build a lookup: market_id → supporting headlines
        news_map = signals.get("news_market_bridge", {}).get("market_news_map", {})

        all_groups = group_polymarket_events(signals)
        for grp in all_groups:
            prob = grp["dominant_prob"]
            alert = "HIGH" if prob > 0.4 else ("MED" if prob > 0.15 else "LOW")

            # Collect supporting news across all markets in this group
            group_news = []
            seen_titles = set()
            for ev in grp["markets"]:
                for n in news_map.get(ev["id"], []):
                    if n["title"] not in seen_titles:
                        group_news.append(n)
                        seen_titles.add(n["title"])
            group_news = sorted(group_news, key=lambda x: -x["match_score"])[:4]

            news_badge = f"  |  {len(group_news)} news" if group_news else ""
            with st.expander(
                f"[{alert}] {grp['name']} — {grp['signal_text']}{news_badge}",
                expanded=prob > 0.2,
            ):
                default_impact = grp["impact"].get("default", "")
                if default_impact:
                    st.markdown(f"**Market Impact:** {default_impact}")

                # Supporting news
                if group_news:
                    st.markdown("**Supporting News**")
                    for n in group_news:
                        topics_str = ", ".join(n["shared_topics"])
                        st.markdown(
                            f"- [{n['source']}] **{n['title']}**  "
                            f"<span style='color:#888;font-size:0.75em'>topics: {topics_str}</span>",
                            unsafe_allow_html=True,
                        )

                st.divider()
                for ev in grp["markets"]:
                    p = ev["probability"]
                    bar_fill = int(p * 18)
                    bar = "█" * bar_fill + "░" * (18 - bar_fill)
                    ev_news = news_map.get(ev["id"], [])
                    news_indicator = f"  [{len(ev_news)} news]" if ev_news else ""
                    st.markdown(
                        f"`{bar}` **{p:.0%}** {ev['question']}{news_indicator}  "
                        f"<span style='color:#888;font-size:0.78em'>"
                        f"vol ${ev['volume']/1e6:.1f}M · ends {ev.get('end_date','?')[:10]}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

# ═══════════════════════════════════════════════════════
# TAB 4 — NEWS
# ═══════════════════════════════════════════════════════
with tab_news:
    st.subheader("Finance Headlines")
    news = signals.get("news", {})
    headlines = news.get("headlines", []) if isinstance(news, dict) else []
    if not headlines:
        st.info("No headlines. Run collect_all.py.")
    else:
        source_filter = st.selectbox(
            "Source",
            ["All"] + sorted({h["source"] for h in headlines}),
        )
        shown = [h for h in headlines if source_filter == "All" or h["source"] == source_filter]
        for h in shown:
            st.markdown(f"**{h['title']}**  \n*{h['source']}* · {h.get('published','')[:25]}")
            if h.get("summary"):
                st.caption(h["summary"])
            st.divider()
