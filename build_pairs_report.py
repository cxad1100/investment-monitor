"""Generate the static pairs-trading HTML report.

  python build_pairs_report.py            # writes local/pairs.html + docs/pairs.html
  python build_pairs_report.py --refresh  # force re-download of price data

Statistical-arbitrage showcase: Engle-Granger cointegration scan over a
curated LS-Exchange universe, walk-forward z-score backtest with
Trade Republic-style costs. Paper simulation — shorting is simulated.
Public build shows percentages only.
"""

import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools import theme
from tools.report_html import fig_html, pct as _pct, card as _card, page
from tools.pairs_universe import UNIVERSE, candidate_pairs, fetch_prices
from tools.pairs_engine import select_pairs, pair_zscore
from tools.pairs_backtest import run_backtest

# ── Settings (one place) ──────────────────────────────────────────────────────
CAPITAL        = 10_000.0   # paper account, EUR
FORMATION_DAYS = 252        # 12 months
TRADING_DAYS   = 63         # 3 months
P_MAX          = 0.05
TOP_N          = 10
ENTRY_Z        = 2.0
STOP_Z         = 3.5
FEE_EUR        = 1.0        # Trade Republic per-order fee
COST_MULTS     = (0.0, 1.0, 2.0)
PAIR_MODE      = "country_sector"   # "country_sector" (tight) or "sector" (broad)

ROOT = Path(__file__).parent


def _broker(t: str) -> str:
    """Broker label for a yfinance ticker: 'WKN · Name' so the pair is findable
    in the app. Falls back to the ticker when the universe lacks broker IDs."""
    m = UNIVERSE.get(t, {})
    wkn, name = m.get("local_id", ""), m.get("name", t)
    return f"{wkn} · {name}" if wkn else name


# ── Data assembly ─────────────────────────────────────────────────────────────

def gather(force: bool = False, refresh: bool | None = None) -> dict:
    # `force` is the live-server convention; `refresh` kept for the CLI flag.
    refresh = force if refresh is None else refresh
    prices = fetch_prices(refresh=refresh)
    cands = candidate_pairs(mode=PAIR_MODE)
    slip = {t: UNIVERSE[t]["slippage_bps"] for t in UNIVERSE}

    bt = run_backtest(prices, cands, slip, capital=CAPITAL,
                      formation_days=FORMATION_DAYS, trading_days=TRADING_DAYS,
                      p_max=P_MAX, top_n=TOP_N, entry=ENTRY_Z, stop=STOP_Z,
                      fee_eur=FEE_EUR, cost_mults=COST_MULTS)

    # Live snapshot: latest 12 months as formation window
    form = prices.tail(FORMATION_DAYS)
    sel = select_pairs(form, cands, p_max=P_MAX, top_n=TOP_N)
    live = []
    for pr in sel["selected"]:
        pair_px = form[[pr["y"], pr["x"]]].dropna()
        z = pair_zscore(pair_px[pr["y"]], pair_px[pr["x"]], pr)
        live.append({**pr, "z_now": float(z.iloc[-1]), "z_series": z})
    return dict(prices=prices, bt=bt, live=live, n_tested=sel["n_tested"],
                n_candidates=len(cands))


# ── Sections ──────────────────────────────────────────────────────────────────

def sec_intro() -> str:
    return f"""
<div class="note">
<b>Statistical arbitrage demo</b> — scans {len(UNIVERSE)} broker-tradeable, sector-tagged
stocks (from the Lang &amp; Schwarz / Trade Republic list) for cointegrated pairs
({"same country + sector" if PAIR_MODE == "country_sector" else "same sector"},
Engle-Granger two-step), then trades the spread on z-score signals in a walk-forward
backtest with Trade Republic-style costs (€{FEE_EUR:.0f}/order + per-leg slippage from
the live bid/ask spread). Paper simulation: shorting is simulated — Trade Republic
offers no shorting. Not financial advice.
</div>"""


def sec_snapshot(d: dict) -> str:
    out = ["<h2>Current snapshot</h2>"]
    grouping = "same country + sector" if PAIR_MODE == "country_sector" else "same sector"
    out.append(f"""<p class='dim'>Latest {FORMATION_DAYS}-trading-day formation window.
{d['n_tested']} {grouping} candidates tested, {len(d['live'])}
cointegrated (p &lt; {P_MAX}) with a tradeable half-life. Multiple-testing caveat:
at p &lt; {P_MAX}, roughly {round(P_MAX * d['n_tested'])} of {d['n_tested']} tests
pass by pure chance. This snapshot is in-sample (μ/σ/β fit on this same window) —
the backtest below is strictly out-of-sample. Each leg shows its broker WKN · name
so you can find and trade the pair in the app.</p>""")
    n_signal = sum(1 for p in d["live"] if abs(p["z_now"]) >= ENTRY_Z)
    cards = [
        _card("Universe", str(len(UNIVERSE))),
        _card("Candidate pairs", str(d["n_candidates"])),
        _card("Tested", str(d["n_tested"])),
        _card("Cointegrated now", str(len(d["live"]))),
        _card("Signals now", str(n_signal)),
    ]
    out.append(f'<div class="cards">{"".join(cards)}</div>')
    rows = []
    for p in d["live"]:
        zcls = "neg" if abs(p["z_now"]) >= ENTRY_Z else ""
        sig = "—"
        if p["z_now"] <= -ENTRY_Z:
            sig = f"LONG {p['y']} / SHORT {p['x']}"
        elif p["z_now"] >= ENTRY_Z:
            sig = f"SHORT {p['y']} / LONG {p['x']}"
        m = UNIVERSE.get(p["y"], {})
        rows.append(
            f"<tr><td class='mono'>{p['y']}/{p['x']}</td>"
            f"<td class='dim' style='font-size:0.8rem'>{_broker(p['y'])}<br>{_broker(p['x'])}</td>"
            f"<td>{m.get('country','—')}</td>"
            f"<td>{m.get('sector','—')}</td>"
            f"<td class='num mono'>{p['pvalue']:.3f}</td>"
            f"<td class='num mono'>{p['beta']:.2f}</td>"
            f"<td class='num mono'>{p['half_life']:.0f}d</td>"
            f"<td class='num mono {zcls}'>{p['z_now']:+.2f}</td>"
            f"<td class='mono'>{sig}</td></tr>")
    out.append("<table><tr><th>Pair</th><th>Broker (WKN · name)</th><th>Country</th>"
               "<th>Sector</th><th class='num'>p-value</th>"
               "<th class='num'>β</th><th class='num'>Half-life</th>"
               "<th class='num'>z now</th><th>Signal</th></tr>"
               + "".join(rows) + "</table>")
    return "".join(out)


def sec_pair_charts(d: dict) -> str:
    live = d["live"][:3]
    if not live:
        return ""
    out = ["<h2>Spread z-scores (top pairs)</h2>",
           "<p class='dim'>z = (spread − μ) / σ with μ, σ, β frozen from the "
           f"formation window. Enter beyond ±{ENTRY_Z:.0f}, exit at 0, stop "
           f"beyond ±{STOP_Z:.1f}.</p>"]
    for p in live:
        z = p["z_series"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z.values, name="z",
                                 line=dict(color=theme.ACCENT, width=1.6)))
        for lvl, col in ((0.0, theme.FG_DIM), (ENTRY_Z, theme.YELLOW),
                         (-ENTRY_Z, theme.YELLOW), (STOP_Z, theme.RED),
                         (-STOP_Z, theme.RED)):
            fig.add_hline(y=lvl, line_color=col, line_width=1,
                          line_dash="dash" if lvl == 0 else "dot")
        fig.update_layout(
            height=260, yaxis_title="z-score", showlegend=False,
            margin=dict(t=34),
            title=dict(text=f"{p['y']} / {p['x']} · p={p['pvalue']:.3f} · "
                            f"β={p['beta']:.2f} · HL={p['half_life']:.0f}d",
                       font=dict(size=13)))
        out.append(f"<div class='chart'>{fig_html(fig)}</div>")
    return "".join(out)


def sec_backtest(d: dict, public: bool) -> str:
    bt = d["bt"]
    runs = bt["runs"]
    base = runs[1.0]
    st = base["stats"]
    out = [f"<h2>Walk-forward backtest (since {bt['start']})</h2>"]
    out.append(f"""<p class='dim'>Every {TRADING_DAYS} trading days: re-run
Engle-Granger on the trailing {FORMATION_DAYS} days (strictly before the trading
window — no look-ahead), select up to {TOP_N} pairs, freeze β/μ/σ, trade z-score
signals with t+1 execution. Capital is split equally across that window's pairs.</p>""")

    fig = go.Figure()
    colors = {0.0: theme.GREEN, 1.0: theme.ACCENT, 2.0: theme.RED}
    names = {0.0: "0× costs (frictionless)", 1.0: "1× costs (realistic)",
             2.0: "2× costs (pessimistic)"}
    for m in sorted(runs):
        roi = (runs[m]["equity"] / CAPITAL - 1.0) * 100
        fig.add_trace(go.Scatter(x=roi.index, y=roi.values,
                                 name=names.get(m, f"{m}× costs"),
                                 line=dict(color=colors.get(m, "#aaa"),
                                           width=2.4 if m == 1.0 else 1.4)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=425, yaxis=dict(title="Cumulative return (%)",
                                             ticksuffix="%"),
                      hovermode="x unified", margin=dict(t=58),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0, font=dict(size=11)))
    out.append(f"<div class='chart'>{fig_html(fig)}</div>")

    cards = [
        _card("Net return", _pct(st["net_return"] * 100)),
        _card("Sharpe", f"{st['sharpe']:.2f}"),
        _card("Max drawdown", _pct(st["max_drawdown"] * 100)),
        _card("Trades", str(st["n_trades"])),
        _card("Win rate", _pct(st["win_rate"] * 100, signed=False)
              if st["win_rate"] is not None else "—"),
        _card("Avg holding", f"{st['avg_days']:.0f}d"
              if st["avg_days"] is not None else "—"),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{st['net_return'] * CAPITAL:+,.0f}"))
        cards.append(_card("Costs paid", f"€{st['total_costs']:,.0f}"))
    else:
        cards.append(_card("Costs / capital",
                           _pct(st["total_costs"] / CAPITAL * 100, signed=False)))
    out.append(f'<div class="cards">{"".join(cards)}</div>')

    trades = sorted(base["trades"], key=lambda t: t["entry"])[-25:]
    rows = []
    for t in trades:
        ret_pct = t["net"] / t["capital"] * 100
        z_e = f"{t['z_entry']:+.2f}" if t["z_entry"] is not None else "—"
        legs = t["pair"].split("/")
        tip = " | ".join(_broker(s) for s in legs)     # WKN · name on hover
        rows.append(f"<tr><td class='mono' title='{tip}'>{t['pair']}</td>"
                    f"<td>{'long' if t['side'] == 1 else 'short'} spread</td>"
                    f"<td class='mono'>{t['entry'].date()}</td>"
                    f"<td class='mono'>{t['exit'].date()}</td>"
                    f"<td class='num mono'>{t['days']}</td>"
                    f"<td class='num mono'>{z_e}</td>"
                    f"<td class='num'>{_pct(ret_pct)}</td>"
                    + ("" if public else f"<td class='num mono'>€{t['net']:+,.0f}</td>")
                    + "</tr>")
    head = ("<tr><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th>"
            "<th class='num'>Days</th><th class='num'>z entry</th>"
            "<th class='num'>Return</th>"
            + ("" if public else "<th class='num'>Net P&amp;L</th>") + "</tr>")
    out.append(f"<h3>Trade ledger (last {len(trades)})</h3>"
               f"<table>{head}{''.join(rows)}</table>")

    wrows = "".join(
        f"<tr><td class='mono'>{w['trade_start'].date()}</td>"
        f"<td class='num mono'>{w['n_tested']}</td>"
        f"<td class='num mono'>{w['n_selected']}</td></tr>"
        for w in bt["windows"])
    out.append("<details><summary>Walk-forward windows: tested vs selected</summary>"
               "<table><tr><th>Trading window start</th><th class='num'>Tested</th>"
               f"<th class='num'>Selected</th></tr>{wrows}</table></details>")
    return "".join(out)


def sec_costs(d: dict) -> str:
    runs = d["bt"]["runs"]
    rows = []
    for m in sorted(runs):
        st = runs[m]["stats"]
        rows.append(f"<tr><td class='mono'>{m:.0f}×</td>"
                    f"<td class='num'>{_pct(st['net_return'] * 100)}</td>"
                    f"<td class='num mono'>{st['sharpe']:.2f}</td>"
                    f"<td class='num'>{_pct(st['max_drawdown'] * 100)}</td>"
                    f"<td class='num mono'>{st['n_trades']}</td>"
                    f"<td class='num'>{_pct(st['total_costs'] / CAPITAL * 100, signed=False)}</td></tr>")
    return ("<h2>Cost sensitivity</h2>"
            "<p class='dim'>Identical signals re-priced at 0×, 1× and 2× the assumed "
            f"frictions (€{FEE_EUR:.0f}/order + 5–15 bps slippage per leg). A strategy "
            "that only works at 0× has no edge; the gap between rows is the cost drag.</p>"
            "<table><tr><th>Costs</th><th class='num'>Net return</th>"
            "<th class='num'>Sharpe</th><th class='num'>Max DD</th>"
            "<th class='num'>Trades</th><th class='num'>Costs / capital</th></tr>"
            + "".join(rows) + "</table>"
            "<div class='note warn'><b>Honest caveats</b> — (1) Multiple testing: "
            "scanning ~50 pairs at p&lt;0.05 finds ~2-3 false positives per window by "
            "chance; the half-life filter and sector restriction mitigate but don't "
            "eliminate this. (2) Shorting is simulated — Trade Republic offers no "
            "shorting, so half of every pair trade is hypothetical. (3) Survivorship: "
            "the curated universe contains today's large caps. (4) Daily closes only — "
            "intraday spread behaviour and borrow costs are ignored.</div>")


def sec_method() -> str:
    return f"""
<h2>How it works</h2>
<details open><summary>Correlation is not cointegration (the core idea)</summary>
<p>Two stocks can be highly <i>correlated</i> (daily moves in the same direction)
while drifting apart forever — correlation says nothing about the <i>level</i> of
the spread. <b>Cointegration</b> is the property that some combination
log(P<sub>Y</sub>) − β·log(P<sub>X</sub>) is stationary: it oscillates around a
stable mean instead of wandering off. That stationary spread is the tradeable
object — divergence is expected to revert.</p></details>
<details><summary>Engle-Granger two-step</summary>
<p>Step 1: OLS log(P<sub>Y</sub>) = α + β·log(P<sub>X</sub>) gives the hedge ratio β;
the residual is the spread. Step 2: a unit-root test on those residuals. Subtlety:
because the residuals come from an estimated regression, plain ADF critical values
are wrong — this engine uses <code>statsmodels.tsa.stattools.coint</code>, which
applies the correct Engle-Granger distribution. Both orientations (Y on X, X on Y)
are tested; the lower p-value wins.</p></details>
<details><summary>Pair selection filters</summary>
<p>p &lt; {P_MAX}, hedge ratio β &gt; 0, and spread half-life between 2 and 60
trading days from an AR(1) fit (HL = −ln 2 / ρ). The half-life filter drops pairs
that revert too slowly to trade inside a {TRADING_DAYS}-day window. At most {TOP_N}
pairs per window, ranked by p-value.</p></details>
<details><summary>Signals &amp; execution</summary>
<p>z-score of the spread with μ, σ, β <b>frozen from the formation window</b>.
Enter when |z| ≥ {ENTRY_Z:.0f} (long the cheap leg, short the rich leg, β-weighted,
dollar-neutral); exit when z crosses 0; stop out when |z| ≥ {STOP_Z:.1f}
(cointegration treated as broken — no re-entry that window). A signal on day t
executes at the close of day t+1.</p></details>
<details><summary>No look-ahead, by construction</summary>
<p>Every parameter used in a trading window is estimated strictly before it; the
t+1 execution lag removes same-close bias; unit tests assert that truncating
future data leaves past signals and past equity unchanged.</p></details>
"""


# ── Assembly ──────────────────────────────────────────────────────────────────

def build(d: dict, public: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Pairs Trading Lab" + ("" if public else " — private")
    back = "index.html" if public else "report.html"
    badge = "public build — percentages only" if public else "private build"
    body = "".join([
        f"<h1>{title}</h1>",
        f"<p class='dim'>generated {now} · {badge} · "
        f"<a href='{back}'>← portfolio monitor</a></p>",
        sec_intro(),
        sec_snapshot(d),
        sec_pair_charts(d),
        sec_backtest(d, public),
        sec_costs(d),
        sec_method(),
    ])
    return page(title, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the local report")
    ap.add_argument("--refresh", action="store_true", help="force price re-download")
    args = ap.parse_args()

    print("gathering pairs data (yfinance)...")
    d = gather(refresh=args.refresh)

    local = ROOT / "local/pairs.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d, public=False))
    print(f"wrote {local}")

    pub = ROOT / "docs/pairs.html"
    pub.parent.mkdir(exist_ok=True)
    pub.write_text(build(d, public=True))
    print(f"wrote {pub}  (percentages only)")

    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
