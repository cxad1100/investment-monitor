"""Generate the static HTML dashboard.

  python build_report.py            # writes local/report.html (full, € amounts)
                                    #    and docs/index.html  (public, percentages only)

No server. Open local/report.html directly; docs/ deploys via GitHub Pages.
Public build hides: euro values, share counts, costs, transactions.
"""

import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools import theme
from tools.portfolio_tools import (
    COMPANY_NAMES as NAMES,
    parse_portfolio,
    fetch_current_prices,
    compute_portfolio_summary,
)
from tools.portfolio_analytics import (
    build_roi_timeseries,
    compute_quant_metrics,
    compute_correlation_matrix,
)
from tools.portfolio_tools import TICKER_MAP
from tools.optimizer import (
    fetch_price_history,
    to_returns,
    annualize,
    optimize,
    max_return_at_vol,
    efficient_frontier,
    random_portfolios,
    portfolio_perf,
    rolling_backtest,
    equity_to_roi,
)

# ── Optimizer settings (one place) ────────────────────────────────────────────
LOOKBACK_DAYS = 365      # estimation window for expected returns / covariance
RF            = 0.045    # risk-free rate (annual)
LONG_ONLY     = True
MAX_W         = 0.35     # max single-position weight
REB_FREQ      = "M"      # backtest rebalance frequency

ROOT = Path(__file__).parent


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def _pct(x, signed=True, nd=1):
    if x is None:
        return "—"
    s = f"{x:+.{nd}f}%" if signed else f"{x:.{nd}f}%"
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls} mono">{s}</span>'


def _card(label: str, value: str) -> str:
    return f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div></div>'


# ── Data assembly ─────────────────────────────────────────────────────────────

def gather() -> dict:
    portfolio = parse_portfolio(ROOT / "input/portfolio.csv")
    prices = fetch_current_prices(portfolio["holdings"])
    summary = compute_portfolio_summary(portfolio, prices)
    positions = [p for p in summary["positions"] if p["position_value"] > 0]
    txns = portfolio["transactions"]
    deployed = sum(t["price"] for t in txns if t["action"] == "buy")

    roi_series, bm_series = build_roi_timeseries(txns)
    metrics = compute_quant_metrics(roi_series, bm_series.get("S&P 500"))
    correlation = compute_correlation_matrix(portfolio["holdings"], TICKER_MAP)

    # Optimizer inputs
    tickers = [p["ticker"] for p in positions]
    hist = fetch_price_history(tickers, period="5y")
    rets = to_returns(hist)
    universe = [t for t in tickers if t in rets.columns]
    window = rets.loc[rets.index >= rets.index[-1] - pd.Timedelta(days=LOOKBACK_DAYS)]
    mean_ann, cov_ann = annualize(window[universe])
    mu, sig = mean_ann.values, cov_ann.values

    val = {p["ticker"]: p["position_value"] for p in positions}
    tot_val = sum(val[t] for t in universe) or 1.0
    cur_w = np.array([val[t] / tot_val for t in universe])
    cur_ret, cur_vol, cur_sharpe = portfolio_perf(cur_w, mu, sig, RF)

    kw = dict(long_only=LONG_ONLY, max_w=MAX_W)
    w_sharpe = optimize(mean_ann, cov_ann, objective="sharpe", rf=RF, **kw)
    w_same_risk = max_return_at_vol(mean_ann, cov_ann, cur_vol, **kw)
    frontier = efficient_frontier(mean_ann, cov_ann, n_points=40, **kw)
    cloud = random_portfolios(mean_ann, cov_ann, n=2500, rf=RF)

    backtest = rolling_backtest(hist[universe], lookback_days=LOOKBACK_DAYS,
                                rebalance_freq=REB_FREQ, objective="sharpe",
                                rf=RF, **kw)

    # YTD from ROI series
    ytd = None
    if not roi_series.empty:
        yr = datetime.now().year
        this_year = roi_series[roi_series.index.year == yr]
        prev = roi_series[roi_series.index.year < yr]
        if not this_year.empty:
            base = (1 + float(prev.iloc[-1]) / 100) if not prev.empty else 1.0
            ytd = ((1 + float(this_year.iloc[-1]) / 100) / base - 1) * 100

    return dict(positions=positions, summary=summary, deployed=deployed, txns=txns,
                roi_series=roi_series, bm_series=bm_series, metrics=metrics,
                correlation=correlation, universe=universe, mu=mu, sig=sig,
                mean_ann=mean_ann, cov_ann=cov_ann, cur_w=cur_w, tot_val=tot_val,
                cur_perf=(cur_ret, cur_vol, cur_sharpe), w_sharpe=w_sharpe,
                w_same_risk=w_same_risk, frontier=frontier, cloud=cloud,
                backtest=backtest, ytd=ytd)


# ── Sections ──────────────────────────────────────────────────────────────────

def sec_summary(d: dict, public: bool) -> str:
    tot = d["summary"]["totals"]
    ret = tot["total_pnl"] / d["deployed"] * 100 if d["deployed"] else 0
    cards = []
    if not public:
        cards.append(_card("Value", f"€{tot['current_value']:,.0f}"))
        pnl_cls = "pos" if tot["total_pnl"] >= 0 else "neg"
        cards.append(_card("Total P&L", f'<span class="{pnl_cls}">€{tot["total_pnl"]:+,.0f}</span>'))
        cards.append(_card("Deployed", f"€{d['deployed']:,.0f}"))
    cards.append(_card("Total Return", _pct(ret)))
    cards.append(_card("YTD", _pct(d["ytd"]) if d["ytd"] is not None else "—"))
    m = d["metrics"]
    if m:
        cards.append(_card("Sharpe", f'{m["sharpe"]:.2f}'))
        cards.append(_card("Max Drawdown", _pct(m["max_drawdown"])))
    cards.append(_card("Positions", str(len(d["positions"]))))
    return f'<div class="cards">{"".join(cards)}</div>'


def sec_weights_now(d: dict, public: bool) -> str:
    """The actionable section: what weights to use today, two options, and why."""
    universe, cur_w = d["universe"], d["cur_w"]
    mu, sig = d["mu"], d["sig"]
    cur_ret, cur_vol, cur_sharpe = d["cur_perf"]
    out = ["<h2>Weights to use now</h2>"]

    out.append(f"""
<div class="note">
<b>Why change anything?</b> Your current mix is a point <i>below</i> the efficient frontier:
for the risk you are already taking ({cur_vol*100:.1f}% annual volatility), a different mix of the
<i>same assets</i> has historically offered more expected return. Both options below only re-weight
what you already own — no new assets, long-only, max {MAX_W:.0%} per position.
</div>""")

    rows_head = ("<tr><th>Ticker</th><th>Name</th><th class='num'>Current</th>"
                 "<th class='num'>A · Max Sharpe</th><th class='num'>B · Same-risk max-return</th>"
                 + ("<th class='num'>A: € to shift</th>" if not public else "") + "</tr>")
    rows = []
    for i, t in enumerate(universe):
        wa = d["w_sharpe"][i] if d["w_sharpe"] is not None else None
        wb = d["w_same_risk"][i] if d["w_same_risk"] is not None else None
        shift = ""
        if not public and wa is not None:
            eur = (wa - cur_w[i]) * d["tot_val"]
            cls = "pos" if eur > 0 else "neg"
            shift = f'<td class="num"><span class="{cls} mono">€{eur:+,.0f}</span></td>'
        wa_cell = f"<td class='num mono'>{wa*100:.1f}%</td>" if wa is not None else "<td class='num'>—</td>"
        wb_cell = f"<td class='num mono'>{wb*100:.1f}%</td>" if wb is not None else "<td class='num'>—</td>"
        rows.append(
            f"<tr><td class='mono'>{t}</td><td>{NAMES.get(t, t)}</td>"
            f"<td class='num mono'>{cur_w[i]*100:.1f}%</td>{wa_cell}{wb_cell}{shift}</tr>"
        )
    out.append(f"<table>{rows_head}{''.join(rows)}</table>")

    def perf_cards(w, label):
        if w is None:
            return f"<div class='note warn'>{label}: no feasible solution under constraints.</div>"
        r, v, s = portfolio_perf(w, mu, sig, RF)
        return (f"<h3>{label}</h3><div class='cards'>"
                + _card("Expected return", _pct(r * 100))
                + _card("Volatility", _pct(v * 100, signed=False))
                + _card("Sharpe", f"{s:.2f}")
                + _card("vs current", _pct((r - cur_ret) * 100) + " return, "
                        + _pct((v - cur_vol) * 100) + " vol")
                + "</div>")

    out.append(perf_cards(np.array(cur_w), "Current portfolio"))
    out.append(perf_cards(d["w_sharpe"], "Option A — Max Sharpe (best return per unit of risk)"))
    out.append(perf_cards(d["w_same_risk"], "Option B — Same risk as now, maximum expected return"))

    out.append("""
<div class="note">
<b>Which to pick?</b>
<b>A (Max Sharpe)</b> is the textbook optimum — the mix with the highest expected
return <i>per unit of risk</i>. Its overall volatility may differ from yours today.
<b>B (Same-risk)</b> keeps your exact current volatility and just slides you up to
the frontier — a pure upgrade if you are comfortable with today's swings.
If A's volatility is close to yours, the two converge to the same point.
</div>
<div class="note warn">
<b>Honesty box.</b> "Expected return" = each asset's average daily return over the last
{LB} days, annualised (×252). That assumes recent winners keep winning — momentum, not prophecy.
The covariance (how assets move together) is far more stable than the return estimates.
These weights are a quantified suggestion, not financial advice.
</div>""".replace("{LB}", str(LOOKBACK_DAYS)))
    return "".join(out)


def sec_frontier(d: dict) -> str:
    mu, sig = d["mu"], d["sig"]
    cloud, frontier = d["cloud"], d["frontier"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cloud["vol"] * 100, y=cloud["ret"] * 100, mode="markers",
        marker=dict(size=4, color=cloud["sharpe"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Sharpe", thickness=12)),
        name="Random mixes", opacity=0.4,
        hovertemplate="vol %{x:.1f}%<br>ret %{y:.1f}%<extra></extra>"))
    if frontier:
        fig.add_trace(go.Scatter(x=[f["vol"] * 100 for f in frontier],
                                 y=[f["ret"] * 100 for f in frontier],
                                 mode="lines", line=dict(color=theme.FG, width=2), name="Frontier"))

    def pt(w, name, symbol, color):
        if w is None:
            return
        r, v, s = portfolio_perf(np.array(w), mu, sig, RF)
        fig.add_trace(go.Scatter(x=[v * 100], y=[r * 100], mode="markers+text",
                                 marker=dict(size=15, color=color, symbol=symbol,
                                             line=dict(width=1, color="#000")),
                                 text=[name], textposition="top center", name=name,
                                 hovertemplate=f"{name}: Sharpe {s:.2f}<extra></extra>"))

    pt(d["cur_w"], "Current", "circle", "#c586c0")
    pt(d["w_sharpe"], "A · Max Sharpe", "star", "#dcdcaa")
    pt(d["w_same_risk"], "B · Same-risk", "diamond", "#4ec9b0")
    fig.update_layout(height=440, xaxis_title="Volatility (annual %)",
                      yaxis_title="Expected return (annual %)",
                      legend=dict(x=0.01, y=0.99))
    return ("<h2>Efficient frontier</h2>"
            "<p class='dim'>Each grey dot is a random mix of your assets. The white line is the "
            "frontier — the best possible return at every risk level. Anything below it is "
            "leaving return on the table.</p>"
            f"<div class='chart'>{_fig_html(fig)}</div>")


def sec_roi(d: dict) -> str:
    roi, bms = d["roi_series"], d["bm_series"]
    if roi.empty:
        return ""
    BM_COLORS = {"Portfolio": "#569cd6", "S&P 500": "#d16969", "Gold": "#d7ba7d",
                 "Bitcoin": "#ce9178", "MSCI World": "#6a9955",
                 "Emerging Markets": "#c586c0", "Fixed Income": "#808080"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roi.index, y=roi.values, name="Portfolio",
                             line=dict(color=BM_COLORS["Portfolio"], width=2.5)))
    for nm, s in bms.items():
        if not s.empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=nm,
                                     line=dict(color=BM_COLORS.get(nm, "#aaa"), width=1.4),
                                     opacity=0.85))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=420, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", legend=dict(x=0.01, y=0.99))
    ranked = sorted([("Portfolio", float(roi.iloc[-1]))] +
                    [(n, float(s.iloc[-1])) for n, s in bms.items() if not s.empty],
                    key=lambda x: -x[1])
    chips = " · ".join(f"{n} {_pct(v)}" for n, v in ranked)
    return ("<h2>ROI vs benchmarks</h2>"
            "<p class='dim'>Cash-flow matched: the same money invested in each benchmark on each "
            "of your buy dates. Sale proceeds count as cash, so selling never looks like a loss.</p>"
            f"<div class='chart'>{_fig_html(fig)}</div><p>{chips}</p>")


def sec_risk(d: dict) -> str:
    m = d["metrics"]
    if not m:
        return ""
    items = [
        ("Sharpe", f'{m["sharpe"]:.2f}'), ("Sortino", f'{m["sortino"]:.2f}'),
        ("Volatility", _pct(m["volatility"], signed=False)), ("CAGR", _pct(m["cagr"])),
        ("Max drawdown", _pct(m["max_drawdown"])), ("Current drawdown", _pct(m["current_drawdown"])),
        ("VaR 95%", _pct(m["var_95"])), ("CVaR 95%", _pct(m["cvar_95"])),
        ("Win rate", _pct(m["win_rate"], signed=False)),
        ("Beta vs S&P", f'{m["beta"]:.2f}' if m.get("beta") is not None else "—"),
        ("Alpha (ann.)", _pct(m["alpha"]) if m.get("alpha") is not None else "—"),
        ("Best day", _pct(m["best_day"], nd=2)), ("Worst day", _pct(m["worst_day"], nd=2)),
    ]
    return "<h2>Risk &amp; efficiency</h2><div class='cards'>" + \
        "".join(_card(k, v) for k, v in items) + "</div>"


def sec_backtest(d: dict) -> str:
    bt = d["backtest"]
    if not bt or not bt.get("equity"):
        return ""
    roi = {k: equity_to_roi(v) for k, v in bt["equity"].items()}
    colors = {"Optimized": "#dcdcaa", "Equal-Weight": "#808080", "S&P 500": "#d16969"}
    fig = go.Figure()
    for name, s in roi.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name,
                                 line=dict(color=colors.get(name, "#aaa"),
                                           width=2.4 if name == "Optimized" else 1.5)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM)
    fig.update_layout(height=400, yaxis=dict(title="Cumulative ROI (%)", ticksuffix="%"),
                      hovermode="x unified", legend=dict(x=0.01, y=0.99))

    sp = roi.get("S&P 500")
    rows = []
    for name, s in roi.items():
        m = compute_quant_metrics(s, sp if name != "S&P 500" else None)
        if m:
            rows.append(f"<tr><td>{name}</td><td class='num mono'>{float(s.iloc[-1]):+.1f}%</td>"
                        f"<td class='num mono'>{m['cagr']:+.1f}%</td>"
                        f"<td class='num mono'>{m['sharpe']:.2f}</td>"
                        f"<td class='num mono'>{m['max_drawdown']:.1f}%</td>"
                        f"<td class='num mono'>{m['var_95']:.2f}%</td></tr>")
    table = ("<table><tr><th>Strategy</th><th class='num'>Total</th><th class='num'>CAGR</th>"
             "<th class='num'>Sharpe</th><th class='num'>Max DD</th><th class='num'>VaR 95</th></tr>"
             + "".join(rows) + "</table>")
    return (f"<h2>Rolling backtest (out-of-sample, since {bt['start']})</h2>"
            "<p class='dim'>Every month the optimizer re-estimates from the trailing year "
            "(strictly before the rebalance date — no look-ahead) and re-weights; the weights are "
            "then held out-of-sample. Equal-weight is the no-skill baseline on the same assets.</p>"
            f"<div class='chart'>{_fig_html(fig)}</div>{table}"
            "<div class='note warn'><b>Selection-bias caveat</b> — this universe is your "
            "<i>current</i> holdings, i.e. assets that already did well enough for you to own them "
            "today. The fair comparison is Optimized vs Equal-Weight, not vs the index.</div>")


def sec_positions(d: dict, public: bool) -> str:
    rows = []
    for p in d["positions"]:
        t = p["ticker"]
        w = p["position_value"] / d["tot_val"] * 100
        extra = ""
        if not public:
            extra = (f"<td class='num mono'>{p['shares']:.4f}</td>"
                     f"<td class='num mono'>€{p['avg_cost']:.2f}</td>"
                     f"<td class='num mono'>€{p['position_value']:,.0f}</td>"
                     f"<td class='num mono'>€{p['unrealized_pnl']:+,.0f}</td>")
        rows.append(f"<tr><td class='mono'>{t}</td><td>{NAMES.get(t, t)}</td>"
                    f"<td class='num mono'>{w:.1f}%</td>{extra}"
                    f"<td class='num'>{_pct(p['unrealized_pct'])}</td></tr>")
    head = ("<tr><th>Ticker</th><th>Name</th><th class='num'>Weight</th>"
            + ("" if public else "<th class='num'>Shares</th><th class='num'>Avg cost</th>"
               "<th class='num'>Value</th><th class='num'>P&amp;L</th>")
            + "<th class='num'>Return</th></tr>")
    return f"<h2>Positions</h2><table>{head}{''.join(rows)}</table>"


def sec_correlation(d: dict) -> str:
    corr = d["correlation"]
    if not corr:
        return ""
    tk = list(corr.keys())
    z = [[corr[r].get(c) for c in tk] for r in tk]
    text = [[f"{corr[r].get(c, 0):.2f}" for c in tk] for r in tk]
    fig = go.Figure(go.Heatmap(z=z, x=tk, y=tk, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#d16969"], [0.5, theme.BG], [1, "#569cd6"]],
                               zmin=-1, zmax=1, colorbar=dict(thickness=12, title="ρ")))
    fig.update_layout(height=max(320, len(tk) * 40),
                      xaxis=dict(tickangle=-35), margin=dict(l=80, b=60))
    return ("<h2>Return correlation (1Y)</h2>"
            "<p class='dim'>How your positions move together. Lots of deep blue = "
            "concentrated bets; diversification needs low or negative correlation.</p>"
            f"<div class='chart'>{_fig_html(fig)}</div>")


def sec_explainer() -> str:
    return f"""
<h2>How the numbers are computed</h2>
<details open><summary>Expected return &amp; volatility (the optimizer inputs)</summary>
<ul>
<li><b>Expected return</b> per asset = average daily price return over the last {LOOKBACK_DAYS}
calendar days × 252 trading days. Purely historical — it assumes the recent past continues.</li>
<li><b>Volatility</b> = standard deviation of those daily returns × √252. "How much it swings."</li>
<li><b>Portfolio volatility</b> is <i>not</i> the average of the assets' volatilities — it uses the
covariance matrix, so assets that zig when others zag cancel risk out. This is the entire point
of Markowitz: mixing imperfectly-correlated assets gives more return per unit of risk.</li>
</ul></details>
<details><summary>Sharpe ratio</summary>
<p>(portfolio return − {RF:.1%} risk-free) ÷ portfolio volatility. Return earned per unit of risk.
&gt;1 is good, &gt;2 excellent. The risk-free rate is what cash earns, so only the excess counts.</p>
</details>
<details><summary>The two weight options</summary>
<p><b>A · Max Sharpe</b>: scipy (SLSQP) searches all weight combinations (long-only, ≤{MAX_W:.0%}
per asset, weights sum to 100%) for the one with the highest Sharpe ratio.</p>
<p><b>B · Same-risk max-return</b>: maximises expected return under the same rules <i>plus</i> one
extra constraint — portfolio volatility may not exceed your current portfolio's volatility.
Same swings as today, more expected return.</p>
</details>
<details><summary>Portfolio ROI (one formula everywhere)</summary>
<p>ROI = (value of holdings + cash received from sells) ÷ total of all buys − 1.
Benchmarks are simulated with the same cash flows: every time you bought, the same euros buy the
benchmark instead. So the comparison answers: "what if every purchase had gone into the S&amp;P
instead?"</p>
</details>
<details><summary>Risk metrics</summary>
<ul>
<li><b>Max drawdown</b> — worst peak-to-trough fall of the portfolio value curve.</li>
<li><b>VaR 95%</b> — your daily loss was no worse than this on 95% of days.</li>
<li><b>CVaR 95%</b> — the average loss on the worst 5% of days (tail risk).</li>
<li><b>Beta</b> — sensitivity to the S&amp;P 500 (1 = moves one-for-one).
<b>Alpha</b> — annualised return beyond what beta alone would deliver.</li>
<li><b>Sortino</b> — Sharpe that only counts downside volatility.</li>
</ul></details>
"""


# ── Assembly ──────────────────────────────────────────────────────────────────

def build(d: dict, public: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Investment Monitor" + ("" if public else " — private")
    badge = ('<span class="dim">public build — euro amounts hidden</span>'
             if public else '<span class="dim">private build — full data</span>')
    body = "".join([
        f"<h1>{title}</h1><p class='dim'>generated {now} · {badge}</p>",
        sec_summary(d, public),
        sec_weights_now(d, public),
        sec_frontier(d),
        sec_roi(d),
        sec_risk(d),
        sec_backtest(d),
        sec_positions(d, public),
        sec_correlation(d),
        sec_explainer(),
    ])
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{theme.REPORT_CSS}</style>
</head><body><main>{body}</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the local report in the browser")
    args = ap.parse_args()

    print("gathering data (yfinance)...")
    d = gather()

    local = ROOT / "local/report.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d, public=False))
    print(f"wrote {local}")

    pub = ROOT / "docs/index.html"
    pub.parent.mkdir(exist_ok=True)
    pub.write_text(build(d, public=True))
    print(f"wrote {pub}  (no € amounts)")

    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
