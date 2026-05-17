"""Streamlit dashboard for weekly investment ratings."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
tab_ratings, tab_signals, tab_events, tab_news = st.tabs(
    ["📈 Ratings", "🌐 Macro Signals", "⚡ Events", "📰 News"]
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
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — MACRO SIGNALS
# ═══════════════════════════════════════════════════════
with tab_signals:
    macro = signals.get("macro", {})
    btc = signals.get("btc", {})

    c1, c2, c3 = st.columns(3)
    regime = macro.get("regime", "unknown").upper()
    risk = macro.get("risk_level", "?").upper()
    c1.metric("Market Regime", regime)
    c2.metric("Risk Level", risk)
    c3.metric("BTC Regime", btc.get("regime", "?").replace("_", " ").title())

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
    st.subheader("Polymarket — Geo/Macro Events")
    poly = signals.get("polymarket_geo", [])
    if not poly:
        st.info("No Polymarket data.")
    else:
        for m in poly:
            prob = m.get("probability", 0.5)
            vol = m.get("volume", 0)
            color = "#e84118" if prob > 0.5 else ("#f39c12" if prob > 0.25 else "#27ae60")
            bar_filled = int(prob * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            st.markdown(
                f"**{m['question']}**  \n"
                f"`{bar}` **{prob:.0%}** yes  ·  vol ${vol:,.0f}  ·  ends {m.get('end_date','?')[:10]}",
                unsafe_allow_html=False,
            )
            st.divider()

    events = signals.get("events", [])
    if events:
        st.subheader("Extracted Events")
        for ev in events:
            st.json(ev)

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
