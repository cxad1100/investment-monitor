"""Generate the static momentum HTML report.

  python build_momentum_report.py            # writes local/momentum.html + docs/momentum.html
  python build_momentum_report.py --refresh  # force re-download of price data

Long-only cross-sectional momentum (12-1, top-k, monthly rebalance) over the
broker-tradeable universe, walk-forward with Trade Republic-style costs.
Mirrors build_pairs_report.py. Public build shows percentages only (the €1
fee text excepted) — the universe is today's survivors, so the survivorship
caveat stays.
"""

import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go

from tools import theme
from tools.report_html import fig_html, pct as _pct, card as _card, page
from tools.momentum import run_momentum, benchmark_curves, equal_weight_curve
from tools.pairs_universe import UNIVERSE, fetch_prices
from tools.portfolio_tools import BENCHMARKS
from tools.data_buffer import cached_price_history

# ── Settings (one place) ──────────────────────────────────────────────────────
K            = 15
LOOKBACK     = 252
SKIP         = 21
REBAL        = "M"
LIQ_MAX      = 30
MIN_PRICE    = 1.0        # drop sub-€1 penny listings (12-1 momentum = tick noise there)
CAPITAL      = 10_000.0   # paper account, EUR
FEE_EUR      = 1.0        # Trade Republic per-order fee
COST_MULTS   = (0.0, 1.0, 2.0)

REBAL_LABEL = {"M": "monthly", "W": "weekly", "Q": "quarterly"}

ROOT = Path(__file__).parent


def _broker(t: str) -> str:
    """Broker label for a yfinance ticker: 'WKN · Name' so the pick is findable
    in the app. Falls back to the ticker when the universe lacks broker IDs."""
    m = UNIVERSE.get(t, {})
    wkn, name = m.get("local_id", ""), m.get("name", t)
    return f"{wkn} · {name}" if wkn else name


# ── Data assembly ─────────────────────────────────────────────────────────────

def gather(force: bool = False, refresh: bool | None = None) -> dict:
    # `force` is the live-server convention; `refresh` kept for the CLI flag.
    refresh = force if refresh is None else refresh
    prices = fetch_prices(refresh=refresh)
    slip = {t: UNIVERSE[t]["slippage_bps"] for t in UNIVERSE}

    res = run_momentum(prices, slip, k=K, lookback=LOOKBACK, skip=SKIP,
                       capital=CAPITAL, cost_mults=COST_MULTS, freq=REBAL,
                       liq_max=LIQ_MAX, fee_eur=FEE_EUR, min_price=MIN_PRICE)

    bench_tickers = [tk for _, (tk, _) in BENCHMARKS.items()]
    bench_raw = cached_price_history(bench_tickers, period="5y", force=refresh)
    bench = bench_raw.rename(columns={tk: name for name, (tk, _) in BENCHMARKS.items()})

    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL,
                meta={t: UNIVERSE[t] for t in UNIVERSE})


def _equity_window(res: dict):
    eq = res["runs"][1.0]["equity"]
    return eq.index[1:] if len(eq) > 1 else eq.index


# ── Sections ──────────────────────────────────────────────────────────────────

def sec_intro() -> str:
    return f"""
<div class="note">
<b>Cross-sectional momentum (long-only)</b> — ranks {len(UNIVERSE)} broker-tradeable
stocks by 12-1 momentum (trailing 12 months, skipping the most recent month), holds
the equal-weight top-{K}, rebalanced monthly. Walk-forward, no look-ahead, with
Trade Republic-style costs (€{FEE_EUR:.0f}/order + live half-spread). Unlike the
pairs page, this is long-only and <b>executable</b> on Trade Republic. Not financial
advice.
</div>"""


def sec_holdings(d: dict) -> str:
    log = d["res"]["holdings_log"]
    if not log:
        return ("<h2>Current top picks</h2>"
                "<p class='dim'>No rebalance with enough history yet.</p>")
    cur = log[-1]
    picks = cur["picks"]
    out = ["<h2>Current top picks</h2>"]
    out.append(f"<p class='dim'>Equal-weight top-{K} as of {cur['date'].date()}, "
               "ranked by 12-1 momentum score. Each leg shows its broker WKN · name "
               "so you can find and trade it in the app.</p>")
    if not picks:
        out.append("<p class='dim'>No eligible names at the latest rebalance.</p>")
        return "".join(out)
    rows = []
    w = 100.0 / len(picks)
    for t in picks:
        m = d["meta"].get(t, {})
        rows.append(
            f"<tr><td class='mono'>{t}</td>"
            f"<td class='dim' style='font-size:0.8rem'>{_broker(t)}</td>"
            f"<td>{m.get('country','—')}</td>"
            f"<td>{m.get('sector','—')}</td>"
            f"<td class='num mono'>{cur['scores'].get(t, float('nan')) * 100:+.1f}%</td>"
            f"<td class='num mono'>{w:.1f}%</td></tr>")
    out.append("<table><tr><th>Ticker</th><th>Broker (WKN · name)</th><th>Country</th>"
               "<th>Sector</th><th class='num'>12-1 momentum</th>"
               "<th class='num'>Weight</th></tr>"
               + "".join(rows) + "</table>")
    return "".join(out)


def sec_curve(d: dict) -> str:
    res = d["res"]
    log = res["holdings_log"]
    window = _equity_window(res)
    out = ["<h2>Walk-forward equity vs benchmarks</h2>"]
    out.append("<p class='dim'>Equity since the first rebalance with enough history, "
               "compared to a buy-hold equal-weight basket of today's top picks "
               "(survivorship-honest baseline) and the MSCI World / S&amp;P 500.</p>")

    fig = go.Figure()
    eq = res["runs"][1.0]["equity"].reindex(window).ffill()
    fig.add_trace(go.Scatter(x=eq.index, y=eq / d["capital"] * 100.0,
                             name=f"Momentum (top-{K})",
                             line=dict(color=theme.ACCENT, width=2.4)))

    # Use the first rebalance that actually has eligible picks — the very first
    # date(s) can have an empty pick list if there isn't yet `lookback+skip` days
    # of history for any name (equal_weight_curve can't build a basket from []).
    first_picks = next((h["picks"] for h in log if h["picks"]), [])
    if first_picks:
        ew = equal_weight_curve(d["prices"], first_picks, window, d["capital"])
        fig.add_trace(go.Scatter(x=ew.index, y=ew / d["capital"] * 100.0,
                                 name="Equal-weight (initial picks, buy-hold)",
                                 line=dict(color=theme.FG_DIM, width=1.4, dash="dot")))

    for name, curve in benchmark_curves(d["benchmarks"], window, d["capital"]).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve / d["capital"] * 100.0,
                                 name=name, line=dict(width=1.4)))

    fig.add_hline(y=100, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=500, yaxis_title="Index (start = 100)",
                      hovermode="x unified", margin=dict(t=96),   # room for wrapped legend rows
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0, font=dict(size=11),
                                  entrywidth=150, entrywidthmode="pixels"))
    out.append(f"<div class='chart'>{fig_html(fig)}</div>")
    return "".join(out)


def sec_stats(d: dict, public: bool) -> str:
    runs = d["res"]["runs"]
    base = runs[1.0]["stats"]
    out = ["<h2>Performance</h2>"]
    out.append(f"<p class='dim'>Walk-forward since {d['res']['start']}, "
               f"{REBAL_LABEL[REBAL]} rebalances, t+1 execution, costs as above.</p>")

    cards = [
        _card("Net return", _pct(base["net_return"] * 100)),
        _card("Sharpe", f"{base['sharpe']:.2f}"),
        _card("Max drawdown", _pct(base["max_drawdown"] * 100)),
        _card("Win rate", _pct(base["win_rate"] * 100, signed=False)
              if base["win_rate"] is not None else "—"),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{base['net_return'] * d['capital']:+,.0f}"))
        cards.append(_card("Costs paid", f"€{base['total_costs']:,.0f}"))
    else:
        cards.append(_card("Costs / capital",
                           _pct(base["total_costs"] / d["capital"] * 100, signed=False)))
    out.append(f'<div class="cards">{"".join(cards)}</div>')

    rows = "".join(
        f"<tr><td class='mono'>{m:.0f}×</td>"
        f"<td class='num'>{_pct(runs[m]['stats']['net_return'] * 100)}</td>"
        f"<td class='num mono'>{runs[m]['stats']['sharpe']:.2f}</td>"
        f"<td class='num'>{_pct(runs[m]['stats']['max_drawdown'] * 100)}</td></tr>"
        for m in sorted(runs))
    out.append("<h3>Cost sensitivity</h3>"
               "<p class='dim'>Identical holdings schedule, re-priced at 0×, 1× and 2× "
               f"the assumed €{FEE_EUR:.0f}/order + half-spread frictions.</p>"
               "<table><tr><th>Costs</th><th class='num'>Net return</th>"
               "<th class='num'>Sharpe</th><th class='num'>Max DD</th></tr>"
               + rows + "</table>")
    return "".join(out)


def sec_rebalance_log(d: dict) -> str:
    log = d["res"]["holdings_log"]
    if not log:
        return ""
    rows = []
    for h in reversed(log[-12:]):
        picks_label = ", ".join(h["picks"]) if h["picks"] else "—"
        rows.append(f"<tr><td class='mono'>{h['date'].date()}</td>"
                    f"<td class='num mono'>{len(h['picks'])}</td>"
                    f"<td class='mono' style='font-size:0.8rem'>{picks_label}</td></tr>")
    return ("<details><summary>Rebalance log (last 12)</summary>"
           "<table><tr><th>Date</th><th class='num'>Holdings</th><th>Picks</th></tr>"
           + "".join(rows) + "</table></details>")


def sec_caveat() -> str:
    return f"""
<div class="note warn">
<b>Survivorship bias — read the result as relative, not absolute.</b> The universe
is <i>today's</i> listed, liquid, history-verified names ({len(UNIVERSE)} tickers).
Backtesting on today's survivors overstates returns: names that delisted (and would
have been bought then dropped along the way) are absent from the data, and today's
membership is implicitly projected onto the past. The honest comparison is momentum
<i>vs an equal-weight buy-hold of the same starting universe</i> above, not the
headline return in isolation. Daily closes only — intraday execution and borrow
costs (for any future short overlay) are ignored.
</div>"""


def sec_method() -> str:
    return f"""
<h2>How it works</h2>
<details open><summary>12-1 momentum (the core idea)</summary>
<p>Momentum is the empirical tendency for assets that have performed well over the
past ~12 months to keep outperforming over the next month, and for recent losers to
keep lagging. The "12-1" convention skips the most recent month: trailing-month
returns show short-term <i>reversal</i> rather than momentum, so including it would
fight the signal. Score = price(t − {SKIP}d) / price(t − {LOOKBACK}d) − 1, using only
data available at the rebalance date.</p></details>
<details><summary>Eligibility &amp; selection</summary>
<p>At each {REBAL_LABEL[REBAL]} rebalance, a name is eligible if its assumed
half-spread is ≤ {LIQ_MAX} bps and it has ≥ {LOOKBACK + SKIP} trading days of
history with a positive last price. The top-{K} eligible names by 12-1 score are
held equal-weight until the next rebalance — turnover is whatever set difference
results from re-ranking, no forced full rotation.</p></details>
<details><summary>Costs &amp; execution</summary>
<p>Entries and exits at each rebalance are charged €{FEE_EUR:.0f}/order plus the
ticker's live half-spread (in bps of the position size); names held across a
rebalance are not re-charged. The cost-sensitivity table re-prices the identical
holdings schedule at 0×, 1× and 2× these frictions — selection never depends on the
multiplier.</p></details>
<details><summary>No look-ahead, by construction</summary>
<p>Scores at a rebalance date use only price history up to and including that date;
the resulting positions accrue returns strictly <i>after</i> it. Unit tests assert
that truncating future data leaves past scores and past holdings unchanged.</p>
</details>
"""


# ── Assembly ──────────────────────────────────────────────────────────────────

def build(d: dict, public: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Momentum Lab" + ("" if public else " — private")
    back = "index.html" if public else "report.html"
    badge = "public build — percentages only" if public else "private build"
    body = "".join([
        f"<h1>{title}</h1>",
        f"<p class='dim'>generated {now} · {badge} · "
        f"<a href='{back}'>← portfolio monitor</a></p>",
        sec_intro(),
        sec_holdings(d),
        sec_curve(d),
        sec_stats(d, public),
        sec_rebalance_log(d),
        sec_caveat(),
        sec_method(),
    ])
    return page(title, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the local report")
    ap.add_argument("--refresh", action="store_true", help="force price re-download")
    args = ap.parse_args()

    print("gathering momentum data (yfinance)...")
    d = gather(refresh=args.refresh)

    local = ROOT / "local/momentum.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d, public=False))
    print(f"wrote {local}")

    pub = ROOT / "docs/momentum.html"
    pub.parent.mkdir(exist_ok=True)
    pub.write_text(build(d, public=True))
    print(f"wrote {pub}  (percentages only)")

    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
