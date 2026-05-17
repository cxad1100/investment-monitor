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
    parts = [f"above $100k: {above_100:.0%}", f"dip <$55k: {dip_55:.0%}", f"dip <$45k: {dip_45:.0%}"]
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
        _group("fomc_next",     "FOMC Next Meeting",       "🏦", fomc_next_m,
               {"Financials","Information Technology","Utilities","Real Estate","Consumer Discretionary"}, _agg_fomc_next),
        _group("fed_annual",    "Fed Rate Path 2026",      "📅", fed_annual_m,
               {"Financials","Information Technology","Utilities","Real Estate","Consumer Discretionary","Consumer Staples"}, _agg_fed_annual),
        _group("iran_me",       "Iran / Middle East",       "🛢️", iran_m,
               {"Energy","Industrials","Consumer Discretionary","Consumer Staples","Materials"}, _agg_conflict),
        _group("china_taiwan",  "China / Taiwan",           "🚢", tw_m,
               {"Information Technology","Industrials","Consumer Discretionary","Materials","Communication Services"}, _agg_conflict),
        _group("russia_ukraine","Russia / Ukraine / NATO",  "🪖", ru_m,
               {"Energy","Industrials","Materials","Financials"}, _agg_conflict),
        _group("bitcoin",       "Bitcoin Range",            "₿",  btc_m,
               {"Information Technology","Financials","Communication Services"}, _agg_btc),
        _group("ai_tech",       "AI / Tech / IPO Pipeline", "🤖", ai_m,
               {"Information Technology","Communication Services"}, _agg_generic),
        _group("recession",     "US Recession Risk",        "📉", rec_m,
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("📊 Trade Republic — Weekly Investment Ratings")

signals = load_signals()
ratings = load_ratings()

if signals is None:
    st.warning("No signals.json found. Run `python collect_all.py` first.")
    st.stop()

collected_at = signals.get("collected_at", "unknown")
st.caption(f"Data collected: {collected_at}  ·  Universe: {signals.get('universe_size', '?')} assets")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ratings, tab_deep, tab_signals, tab_events, tab_news = st.tabs(
    ["📈 Ratings", "💡 Deep Dive", "🌐 Macro Signals", "⚡ Events", "📰 News"]
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

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
        types = ["All"] + sorted(df["Type"].dropna().unique().tolist())
        regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
        sel_sector = fc1.selectbox("Sector", sectors)
        sel_type = fc2.selectbox("Type", types)
        sel_region = fc3.selectbox("Region", regions)

        filtered = df.copy()
        if sel_sector != "All":
            filtered = filtered[filtered["Sector"] == sel_sector]
        if sel_type != "All":
            filtered = filtered[filtered["Type"] == sel_type]
        if sel_region != "All":
            filtered = filtered[filtered["Region"] == sel_region]

        st.write(f"**{len(filtered)} assets**")

        # Render table with colored grades
        html_rows = []
        for _, row in filtered.head(200).iterrows():
            badge = grade_badge(row["Grade"])
            html_rows.append(
                f"<tr><td><b>{row['Ticker']}</b></td>"
                f"<td>{badge}</td>"
                f"<td>{row['Score']}</td>"
                f"<td>{row['Sector']}</td>"
                f"<td>{row['Type']}</td>"
                f"<td>{row['Region']}</td></tr>"
            )
        table_html = (
            "<table style='width:100%;border-collapse:collapse'>"
            "<thead><tr style='border-bottom:1px solid #444'>"
            "<th align='left'>Ticker</th><th align='left'>Grade</th>"
            "<th align='left'>Score</th><th align='left'>Sector</th>"
            "<th align='left'>Type</th><th align='left'>Region</th>"
            "</tr></thead><tbody>"
            + "".join(html_rows)
            + "</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        # Sector score chart
        st.divider()
        st.subheader("Average Score by Sector")
        sector_avg = df.groupby("Sector")["Score"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(sector_avg, x="Sector", y="Score", color="Score",
                     color_continuous_scale="RdYlGn", range_color=[40, 70])
        fig.update_layout(height=350, margin=dict(t=20, b=40))
        st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ═══════════════════════════════════════════════════════
with tab_deep:
    fast_scores = signals.get("fast_scores", {})
    DEEP_THRESHOLD = 60
    top_tickers = sorted(
        [(t, v) for t, v in fast_scores.items() if v.get("score", 0) >= DEEP_THRESHOLD],
        key=lambda x: x[1]["score"], reverse=True
    )

    if not top_tickers:
        st.info(f"No assets scored ≥{DEEP_THRESHOLD}. Run full collect_all.py (not --fast).")
    else:
        st.markdown(
            f"**{len(top_tickers)} assets** scored ≥{DEEP_THRESHOLD} — detailed signal analysis below."
        )
        st.caption("Sub-scores weighted: Earnings 25% · Insider 20% · Macro 15% · Geo 15% · Fundamentals 15% · Options 5% · WSB/Short 5%")

        for ticker, entry in top_tickers:
            score = entry["score"]
            grade = entry.get("grade", "?")
            sector = entry.get("sector", "")
            asset_type = entry.get("type", "stock")

            dive = build_deep_dive(ticker, signals)

            badge_color = GRADE_COLOR.get(grade, "#888")
            verdict = dive["verdict"]
            v_color = dive["verdict_color"]

            header = (
                f'<span style="font-size:1.15em;font-weight:bold">{ticker}</span>&nbsp;&nbsp;'
                f'<span style="background:{badge_color};color:#000;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em">{grade}</span>&nbsp;&nbsp;'
                f'<span style="background:{v_color};color:#fff;padding:2px 10px;border-radius:4px;font-weight:bold;font-size:0.85em">{verdict}</span>&nbsp;&nbsp;'
                f'<span style="color:#888;font-size:0.9em">{sector} · {asset_type} · Score {score}/100</span>'
            )

            with st.expander(f"{ticker}  {grade}  {verdict}  —  score {score}  ·  {sector}", expanded=(score >= 65)):
                st.markdown(header, unsafe_allow_html=True)
                st.caption(dive["macro_context"])
                st.divider()

                col_pro, col_con = st.columns(2)
                with col_pro:
                    st.markdown("**✅ Pros**")
                    if dive["pros"]:
                        for p in dive["pros"]:
                            st.markdown(f"- {p}")
                    else:
                        st.caption("No strong positives identified")

                with col_con:
                    st.markdown("**❌ Cons**")
                    if dive["cons"]:
                        for c in dive["cons"]:
                            st.markdown(f"- {c}")
                    else:
                        st.caption("No major negatives identified")

                st.divider()

                # Sub-score breakdown bar chart
                sub = dive["sub_scores"]
                if sub:
                    st.markdown("**Signal Breakdown**")
                    sub_labels = {
                        "earnings": "Earnings (25%)",
                        "insider": "Insider Flow (20%)",
                        "macro": "Macro Regime (15%)",
                        "geo": "Geopolitical (15%)",
                        "fundamentals": "Fundamentals (15%)",
                        "options": "Options Flow (5%)",
                        "wsb_short": "WSB/Short (5%)",
                    }
                    sub_df = pd.DataFrame([
                        {"Signal": sub_labels.get(k, k), "Score": v, "Color": "#27ae60" if v >= 60 else ("#e84118" if v < 40 else "#f39c12")}
                        for k, v in sub.items()
                    ])
                    fig = px.bar(sub_df, x="Score", y="Signal", orientation="h",
                                 color="Color", color_discrete_map="identity",
                                 range_x=[0, 100])
                    fig.add_vline(x=50, line_dash="dash", line_color="#666")
                    fig.update_layout(height=260, margin=dict(t=10, b=10, l=0, r=20),
                                      showlegend=False, yaxis_title=None)
                    st.plotly_chart(fig, width="stretch")

                # Key metrics table
                fund = dive["fundamentals"]
                ins = dive["insider"]
                opt = dive["options"]
                sht = dive["short"]
                if any([fund, ins, opt, sht]):
                    st.markdown("**Key Metrics**")
                    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
                    m1.metric("P/E", f"{fund.get('pe_ratio'):.1f}x" if fund.get("pe_ratio") else "N/A")
                    m2.metric("Rev Growth", f"{fund.get('revenue_growth_yoy'):+.1f}%" if fund.get("revenue_growth_yoy") is not None else "N/A")
                    m3.metric("ROE", f"{fund.get('return_on_equity'):.1f}%" if fund.get("return_on_equity") else "N/A")
                    m4.metric("Analyst", fund.get("analyst_rating", "N/A").title())
                    upside_val = fund.get("upside_pct")
                    m5.metric("Upside to Target", f"{upside_val:+.1f}%" if upside_val is not None else "N/A",
                              delta=f"${fund.get('target_price','?')}" if upside_val is not None else None)
                    m6.metric("Put-Call", f"{opt.get('put_call_ratio'):.2f}" if opt.get("put_call_ratio") else "N/A")
                    m7.metric("Short Float", f"{sht.get('short_float_pct'):.1f}%" if sht.get("short_float_pct") is not None else "N/A")
                    m8.metric("Next Earnings", fund.get("next_earnings", "N/A"))

                # Grouped event analysis
                rel_groups = dive["relevant_groups"]
                if rel_groups:
                    st.divider()
                    st.markdown("**Event Risk Analysis (Polymarket)**")
                    for grp in rel_groups:
                        sector_impact = grp["impact"].get(sector, grp["impact"].get("default", ""))
                        prob = grp["dominant_prob"]
                        alert = "🔴" if prob > 0.4 else ("🟡" if prob > 0.15 else "🟢")

                        with st.expander(f"{grp['icon']} {alert} **{grp['name']}** — {grp['signal_text']}", expanded=True):
                            if sector_impact:
                                st.markdown(f"**Impact on {sector}:** {sector_impact}")

                            st.markdown("---")
                            st.caption("Individual markets:")
                            for ev in grp["markets"]:
                                p = ev["probability"]
                                bar_fill = int(p * 14)
                                bar = "█" * bar_fill + "░" * (14 - bar_fill)
                                st.markdown(
                                    f"`{bar}` **{p:.0%}** {ev['question']}  "
                                    f"<span style='color:#888;font-size:0.75em'>"
                                    f"vol ${ev['volume']/1e6:.1f}M · ends {ev.get('end_date','?')[:10]}"
                                    f"</span>",
                                    unsafe_allow_html=True,
                                )

        st.divider()
        st.caption(f"Threshold: score ≥{DEEP_THRESHOLD}. Scores refresh every 5 min when data/signals.json updates.")

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
        with st.expander("📊 Market Sentiment Detail", expanded=False):
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
                color = "🔴" if spread_r == "widening" else ("🟢" if spread_r == "tightening" else "🟡")
                st.markdown(f"{color} Spreads: **{spread_r}** ({spread_c:+.2f}% 1M)" if spread_c is not None else f"Regime: {spread_r}")
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
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sector Tailwinds / Headwinds")
        tailwinds = macro.get("sector_tailwinds", [])
        headwinds = macro.get("sector_headwinds", [])
        for s in tailwinds:
            st.markdown(f"✅ **{s}**")
        for s in headwinds:
            st.markdown(f"❌ **{s}**")

        st.divider()
        st.subheader("Futures")
        st.caption(macro.get("futures_summary", "No data"))

        st.divider()
        st.subheader("BTC Signal")
        st.caption(btc.get("interpretation", "No data"))
        asset_impacts = btc.get("asset_impacts", {})
        for asset, signal in asset_impacts.items():
            icon = "✅" if signal == "tailwind" else "❌"
            st.markdown(f"{icon} {asset.replace('_', ' ').title()}: *{signal}*")

    with col_b:
        st.subheader("FRED Indicators")
        fred = macro.get("fred_indicators", {})
        for key, info in fred.items():
            trend_icon = "📈" if info.get("trend") == "rising" else ("📉" if info.get("trend") == "falling" else "➡️")
            st.metric(
                label=f"{trend_icon} {info.get('name', key)}",
                value=f"{info.get('latest', '?')} {info.get('unit', '')}",
                delta=f"vs yr ago: {info.get('prev_year', '?')}",
            )

        st.divider()
        st.subheader("GDELT Conflict Indices")
        gdelt = signals.get("gdelt", [])
        for region in gdelt:
            score = region.get("conflict_score", 0)
            trend = region.get("trend", "unknown")
            trend_icon = "🔴" if trend == "escalating" else ("🟢" if trend == "de-escalating" else "🟡")
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            st.markdown(f"{trend_icon} **{region['region']}** `{bar}` {score:.2f} — *{trend}*")

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
        all_groups = group_polymarket_events(signals)
        for grp in all_groups:
            prob = grp["dominant_prob"]
            alert = "🔴" if prob > 0.4 else ("🟡" if prob > 0.15 else "🟢")
            with st.expander(
                f"{grp['icon']} {alert} **{grp['name']}** — {grp['signal_text']}",
                expanded=prob > 0.2,
            ):
                default_impact = grp["impact"].get("default", "")
                if default_impact:
                    st.markdown(f"**Market Impact:** {default_impact}")
                st.divider()
                for ev in grp["markets"]:
                    p = ev["probability"]
                    bar_fill = int(p * 18)
                    bar = "█" * bar_fill + "░" * (18 - bar_fill)
                    st.markdown(
                        f"`{bar}` **{p:.0%}** {ev['question']}  "
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
