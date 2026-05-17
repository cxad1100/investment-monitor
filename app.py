"""Streamlit dashboard for weekly investment ratings."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Event group definitions ──────────────────────────────────────────────────
# Each group: keywords to match markets, sectors it's relevant to,
# and per-sector causal impact description.
EVENT_GROUPS = [
    {
        "id": "fed_rates",
        "name": "Fed Rate Policy",
        "icon": "🏦",
        "keywords": ["fed rate", "rate cut", "rate hike", "federal reserve", "interest rate"],
        "relevant_sectors": {"Financials", "Real Estate", "Utilities", "Information Technology",
                             "Consumer Discretionary", "Consumer Staples", "Industrials"},
        "aggregate": "implied_cuts",
        "impact": {
            "Financials": "Higher-for-longer rates widen net interest margins (NIM) for banks short-term, but slow loan growth and raise credit default risk. If cuts stay at 0-1 in 2026, bank profitability holds — but watch for credit quality deterioration in H2.",
            "Real Estate": "Rate persistence is the key headwind. REITs trade inversely to yields — elevated rates compress cap rates and property valuations. Each 25bp cut would add ~3-5% to REIT NAV.",
            "Utilities": "Utilities compete with bonds for yield-seeking capital. High rates make their dividends less attractive vs risk-free returns, compressing multiples. A sector to avoid until first cut.",
            "Information Technology": "High discount rates compress growth stock multiples. The longer rates stay elevated, the more near-term cash flow matters vs long-term growth projections — a headwind for high-multiple tech.",
            "Consumer Discretionary": "High rates squeeze consumer credit (mortgages, auto loans, credit cards), reducing discretionary spending capacity.",
            "default": "Rate environment shapes cost of capital and equity multiples across the market.",
        },
    },
    {
        "id": "iran",
        "name": "Iran / Middle East",
        "icon": "🛢️",
        "keywords": ["iran", "invade iran", "iranian regime", "iran nuke", "iran deal"],
        "relevant_sectors": {"Energy", "Industrials", "Consumer Discretionary", "Materials", "Consumer Staples"},
        "aggregate": "conflict_risk",
        "impact": {
            "Energy": "Iran conflict risks Strait of Hormuz closure (~20% of global oil transit). A US-Iran war would spike Brent crude $20-40+/barrel instantly. Conversely, nuclear deal (59% probability) could add 1-2M bbl/day of Iranian supply, pressuring oil prices. Net: asymmetric energy risk — bullish on conflict, bearish on deal.",
            "Industrials": "Defense contractors (aerospace & defense sub-sector) directly benefit from escalation — higher procurement, emergency spending authorizations. Iran scenario is a catalyst for LMT, RTX, NOC-type names.",
            "Consumer Discretionary": "Oil spike from Iran conflict raises fuel costs for consumers and logistics, compressing margins and reducing spending power.",
            "Consumer Staples": "Food and staples companies face input cost pressure from energy-driven logistics costs. Modest headwind.",
            "default": "Iran geopolitics drive oil price volatility which propagates through energy-linked input costs.",
        },
    },
    {
        "id": "china_taiwan",
        "name": "China / Taiwan",
        "icon": "🚢",
        "keywords": ["taiwan", "china invade", "china x taiwan", "china blockade", "china military"],
        "relevant_sectors": {"Information Technology", "Industrials", "Consumer Discretionary", "Materials"},
        "aggregate": "conflict_risk",
        "impact": {
            "Information Technology": "Taiwan produces ~90% of the world's advanced semiconductors (TSMC). A blockade or invasion would create a multi-year supply shock for chips — NVIDIA, AMD, Apple, AMAT, LRCX all critically dependent. Even a 10% conflict probability justifies diversification premium.",
            "Industrials": "Supply chain disruption risk from Taiwan conflict hits industrial equipment manufacturers dependent on precision components. Defense sector benefits from elevated NATO/US spending.",
            "Consumer Discretionary": "Electronics, appliances, and auto sectors face severe inventory shortfalls if Taiwan supply disrupted — comparable to 2020-2021 chip shortage but structurally worse.",
            "Materials": "Rare earth and specialty chemical supply chains routed through China face secondary sanction risk in a conflict scenario.",
            "default": "Taiwan conflict would be the most severe supply chain shock since WWII — systemic risk.",
        },
    },
    {
        "id": "russia_ukraine",
        "name": "Russia / Ukraine / NATO",
        "icon": "🪖",
        "keywords": ["russia invade nato", "zelenskyy", "ukraine peace", "ukraine joins nato", "ukraine election"],
        "relevant_sectors": {"Energy", "Industrials", "Materials", "Financials"},
        "aggregate": "conflict_risk",
        "impact": {
            "Energy": "Ukraine peace deal (positive for European energy security) vs NATO escalation (negative — energy embargo risk). Natural gas prices in Europe tied to war resolution.",
            "Industrials": "European defense rearmament (NATO 2% GDP pledge enforcement) is a structural tailwind for defense/aerospace regardless of war outcome.",
            "Materials": "Russia controls ~40% of global palladium, significant nickel and aluminum. Sanction changes on deal or escalation move materials prices.",
            "Financials": "European bank exposure to Russian assets and Ukraine reconstruction financing — deal = reconstruction opportunity, escalation = credit losses.",
            "default": "War resolution path drives European growth outlook and energy prices.",
        },
    },
    {
        "id": "bitcoin",
        "name": "Bitcoin / Crypto",
        "icon": "₿",
        "keywords": ["bitcoin", "btc", "crypto"],
        "relevant_sectors": {"Information Technology", "Financials", "Communication Services"},
        "aggregate": "btc_range",
        "impact": {
            "Information Technology": "Bitcoin functions as a risk-on/risk-off barometer for growth assets. BTC holding above $80k signals institutional risk appetite — supportive for high-multiple tech. A BTC crash to $35-45k range (32-50% probability per markets) would signal broad risk-off, hitting tech multiples.",
            "Financials": "Crypto-adjacent fintechs (Coinbase, PayPal, Block) tied directly. Traditional banks with crypto custody/trading exposure also affected. BTC rally = fee revenue uplift.",
            "Communication Services": "Advertising and platform companies see correlation with crypto sentiment — crypto bull cycles drive user engagement and ad spend in tech/gaming verticals.",
            "default": "Bitcoin is a leading indicator of risk appetite — direction matters more than absolute level.",
        },
    },
    {
        "id": "recession",
        "name": "US Recession",
        "icon": "📉",
        "keywords": ["recession", "gdp contraction"],
        "relevant_sectors": {"Consumer Discretionary", "Financials", "Industrials", "Materials",
                             "Information Technology", "Consumer Staples", "Utilities", "Healthcare"},
        "aggregate": "probability",
        "impact": {
            "Consumer Discretionary": "Recession is the primary tail risk — consumer spending contracts sharply. Discretionary is the most cyclically exposed sector; revenue can drop 15-30% in deep recessions.",
            "Financials": "Recession triggers loan loss provisions, credit card defaults, and deal flow collapse. Bank stocks typically fall 30-50% peak-to-trough in recessions.",
            "Industrials": "Capital expenditure freezes in recessions — industrial demand collapses. Companies with backlog visibility hold better than spot-order businesses.",
            "Information Technology": "Enterprise IT budgets are cut early in recessions. However, cloud/SaaS companies with sticky contracts prove more resilient than hardware or ad-driven businesses.",
            "Consumer Staples": "Recession-resistant — consumers still buy food, household products. Staples tend to outperform as defensive rotation destination.",
            "Utilities": "Recession-proof by nature — electricity/gas demand inelastic. Plus rate cuts likely accompany recession, benefiting utility valuations.",
            "Healthcare": "Defensive in recessions — healthcare demand inelastic, government spending supports hospitals and pharma.",
            "Materials": "Highly cyclical — demand collapses with industrial production. Copper and steel prices crash in recessions.",
            "default": "Recession (22% probability) is the key tail risk — review cyclical vs defensive positioning.",
        },
    },
]

# Sector → group IDs that are relevant (for ordering)
SECTOR_EVENT_KEYWORDS = {
    "Energy": ["iran", "oil", "opec", "russia", "ukraine"],
    "Information Technology": ["china", "taiwan", "bitcoin", "semiconductor"],
    "Financials": ["fed rate", "rate cut", "recession", "rate hike", "interest rate"],
    "Materials": ["china", "copper", "russia", "tariff"],
    "Industrials": ["tariff", "trade war", "china"],
    "Consumer Discretionary": ["recession", "tariff"],
    "Consumer Staples": ["recession"],
    "Utilities": ["fed rate", "rate cut", "recession", "interest rate"],
    "Healthcare": ["recession"],
    "Real Estate": ["fed rate", "recession", "interest rate"],
    "Communication Services": ["china"],
}


def group_polymarket_events(poly: list[dict]) -> list[dict]:
    """Assign each Polymarket market to a thematic group and compute aggregate signal."""
    grouped = {g["id"]: {"meta": g, "markets": []} for g in EVENT_GROUPS}
    unmatched = []

    for m in poly:
        q = m["question"].lower()
        matched = False
        for g in EVENT_GROUPS:
            if any(k in q for k in g["keywords"]):
                grouped[g["id"]]["markets"].append(m)
                matched = True
                break
        if not matched:
            unmatched.append(m)

    results = []
    for gid, data in grouped.items():
        markets = data["markets"]
        if not markets:
            continue
        meta = data["meta"]
        agg_type = meta["aggregate"]

        if agg_type == "implied_cuts":
            # Fed: compute implied cuts from probabilities
            no_cut_prob = next((m["probability"] for m in markets if "no fed rate cut" in m["question"].lower()), None)
            one_cut = next((m["probability"] for m in markets if "will 1 fed rate cut" in m["question"].lower()), 0)
            two_cut = next((m["probability"] for m in markets if "will 2 fed rate cut" in m["question"].lower()), 0)
            if no_cut_prob is not None:
                p_any_cut = round(1 - no_cut_prob, 2)
                signal_text = f"{no_cut_prob:.0%} chance of no cuts in 2026 · {p_any_cut:.0%} chance of ≥1 cut"
                dominant_prob = max(no_cut_prob, p_any_cut)
            else:
                signal_text = f"{len(markets)} rate scenarios priced"
                dominant_prob = 0.5

        elif agg_type == "btc_range":
            dip_55 = next((m["probability"] for m in markets if "55,000" in m["question"]), 0)
            dip_45 = next((m["probability"] for m in markets if "45,000" in m["question"]), 0)
            above_100 = next((m["probability"] for m in markets if "100,000" in m["question"] and "reach" in m["question"].lower()), 0)
            above_150 = next((m["probability"] for m in markets if "150k" in m["question"].lower() or "150,000" in m["question"]), 0)
            signal_text = (
                f"Below $55k: {dip_55:.0%} · Below $45k: {dip_45:.0%} · "
                f"Above $100k: {above_100:.0%} · Above $150k: {above_150:.0%}"
            )
            dominant_prob = above_100

        elif agg_type == "conflict_risk":
            # Take max probability across conflict markets
            probs = [m["probability"] for m in markets]
            dominant_prob = max(probs) if probs else 0
            avg_prob = sum(probs) / len(probs) if probs else 0
            signal_text = f"{len(markets)} scenarios · peak probability {dominant_prob:.0%}"

        else:  # probability
            dominant_prob = markets[0]["probability"] if markets else 0
            signal_text = f"{dominant_prob:.0%} probability"

        results.append({
            "id": gid,
            "name": meta["name"],
            "icon": meta["icon"],
            "signal_text": signal_text,
            "dominant_prob": dominant_prob,
            "relevant_sectors": meta["relevant_sectors"],
            "impact": meta["impact"],
            "markets": sorted(markets, key=lambda x: x["volume"], reverse=True),
        })

    return sorted(results, key=lambda x: x["dominant_prob"], reverse=True)


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
    poly = signals.get("polymarket_geo", [])

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
    all_groups = group_polymarket_events(poly)
    relevant_groups = [g for g in all_groups if sector in g["relevant_sectors"]]

    # Add event-group signals to pros/cons
    for grp in relevant_groups[:3]:
        prob = grp["dominant_prob"]
        gid = grp["id"]
        if gid == "fed_rates" and sector in {"Financials"}:
            no_cut = next((m["probability"] for m in grp["markets"] if "no fed rate cut" in m["question"].lower()), 0)
            if no_cut > 0.6:
                pros.append(f"Rates staying high (Polymarket: {no_cut:.0%} no cuts in 2026) — bank NIM benefits")
        elif gid == "fed_rates" and sector in {"Utilities", "Real Estate", "Information Technology"}:
            no_cut = next((m["probability"] for m in grp["markets"] if "no fed rate cut" in m["question"].lower()), 0)
            if no_cut > 0.5:
                cons.append(f"High-rate persistence (Polymarket: {no_cut:.0%} no cuts in 2026) — valuation headwind")
        elif gid == "iran" and sector == "Energy":
            conflict_prob = max((m["probability"] for m in grp["markets"] if "invade" in m["question"].lower() or "regime" in m["question"].lower()), default=0)
            if conflict_prob > 0.2:
                pros.append(f"Iran conflict risk (Polymarket: {conflict_prob:.0%}) — oil supply shock scenario = energy tailwind")
        elif gid == "china_taiwan" and sector == "Information Technology":
            if prob > 0.05:
                cons.append(f"China/Taiwan conflict risk (Polymarket: {prob:.0%}) — semiconductor supply chain tail risk")
        elif gid == "recession" and sector in {"Consumer Discretionary", "Financials", "Materials"}:
            if prob > 0.15:
                cons.append(f"Recession risk (Polymarket: {prob:.0%}) — cyclical sector exposure")
        elif gid == "recession" and sector in {"Consumer Staples", "Utilities", "Healthcare"}:
            if prob > 0.15:
                pros.append(f"Defensive sector — outperforms in recession scenarios (Polymarket: {prob:.0%} recession risk)")

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
        "relevant_groups": relevant_groups,
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
    poly = signals.get("polymarket_geo", [])
    if not poly:
        st.info("No Polymarket data.")
    else:
        all_groups = group_polymarket_events(poly)
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
