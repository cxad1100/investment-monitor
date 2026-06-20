"""The chosen production strategy — one config (the 'ultimate' extracted from the
32-config grid), on the survivorship-corrected universe.

  python build_strategy_report.py            # writes local/strategy.html + docs/strategy.html
  python build_strategy_report.py --open

Unlike the momentum *lab* (which renders the whole grid), this page commits to a
single config and shows it cleanly: which strategy + why, current picks, equity vs
benchmarks, train/validation/full performance, the PnL-colored monthly timeline,
and the (now small) honest caveats.
"""
import argparse
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools.report_html import pct as _pct, card as _card, page, fig_html
from tools import theme, significance as sig
from tools.momentum import (run_momentum, winsorize_prices, to_xetra_calendar,
                            precompute_eligibility)
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map
from tools.momentum_grid import MomentumConfig, _stats_slice, run_grid, ALL_CONFIGS
from tools.portfolio_tools import BENCHMARKS
from tools.data_buffer import cached_price_history
from build_momentum_report import (
    PRICES_CSV, META_CSV, ROOT, LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, CAPITAL,
    FEE_EUR, COST_MULTS, TRAIN_END, VAL_END, WINSOR_CAP, EXEC_LAG,
    _slip, _broker, _disp, _pnl_color, sec_holdings, sec_curve,
    sec_grid, sec_feasibility, sec_timelines, sec_survivorship, sec_method,
)

# The chosen strategy — A···EF = vol-adjusted, equal-weight top-15, quarterly, lazy.
# Picked from the 32-config grid (sector-neutral B is excluded — the global universe has no
# sector data, so it would be a silent no-op), run on the XETRA / Lang & Schwarz trading
# calendar (the only days you can actually fill), for the highest worst-case robustness
# min(train, validation) Sharpe among configs that pay for their own costs and are positive in
# both windows. A···EF leads on robustness (train 0.73 / val 0.62) and holds up out-of-sample
# (test 1.24); quarterly + lazy keep turnover (and the €1/order drag) low. Survivorship-clean by
# construction — momentum buys winners, dying names rank last, so it holds ~0 into death.
STRATEGY = MomentumConfig(vol_adjust=True, slots=15, freq="Q", lazy=True)


def _desc(cfg: MomentumConfig) -> str:
    parts = []
    if cfg.vol_adjust:
        parts.append("volatility-adjusted")
    if cfg.sector_neutral:
        parts.append("sector-neutral")
    if cfg.trend_filter:
        parts.append("trend-filtered (200d kill-switch)")
    parts.append(f"equal-weight top-{cfg.slots}")
    parts.append("quarterly" if cfg.freq == "Q" else "monthly")
    if cfg.lazy:
        parts.append("lazy-rebalanced")
    return ", ".join(parts)


def gather(force: bool = False, refresh: bool | None = None) -> dict:
    refresh = force if refresh is None else refresh
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = to_xetra_calendar(prices)                         # L&S/XETRA sessions only (tradeable days)
    prices = winsorize_prices(prices, cap=WINSOR_CAP)          # de-glitch the raw feed
    meta_df = pd.read_csv(META_CSV)                            # universe is pre-filtered at build time
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    pit = PITUniverse(prices, delisting_map(meta_df))

    benches = {n: v for n, v in BENCHMARKS.items() if n != "Bitcoin"}   # equities/bonds only
    bench_tickers = [tk for tk, _ in benches.values()]
    bench_raw = cached_price_history(bench_tickers, period="9y", force=refresh)
    bench = bench_raw.rename(columns={tk: name for name, (tk, _) in benches.items()})
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else bench.iloc[:, 0]

    # No sector data on the global universe → sectors=None (sector-neutral configs excluded).
    res = run_momentum(prices, slip, lookback=LOOKBACK, skip=SKIP, capital=CAPITAL,
                       cost_mults=COST_MULTS, start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                       min_price=MIN_PRICE, sectors=None, benchmark=spx, pit=pit,
                       execute_lag=EXEC_LAG, **STRATEGY.kwargs())
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    train = _stats_slice(eq, tr, eq.index[0], te, CAPITAL)
    val = _stats_slice(eq, tr, te + pd.Timedelta(days=1), ve, CAPITAL)
    test = _stats_slice(eq, tr, ve + pd.Timedelta(days=1), eq.index[-1], CAPITAL)
    hits = sum(len(h.get("dead", set())) for h in res["holdings_log"])
    grid = run_grid(prices, slip, sectors=None, benchmark=spx, pit=pit, start=START,
                    configs=[c for c in ALL_CONFIGS if not c.sector_neutral],
                    train_end=TRAIN_END, val_end=VAL_END, capital=CAPITAL,
                    lookback=LOOKBACK, skip=SKIP, execute_lag=EXEC_LAG)
    # ── Significance & robustness: random-selection null, deflated Sharpe, bootstrap CI ──
    hl = res["holdings_log"]
    rb_dates = [h["date"] for h in hl] + [hl[-1]["next"]]
    elig = precompute_eligibility(prices, slip, rb_dates, liq_max=LIQ_MAX,
                                  min_obs=LOOKBACK + SKIP, min_price=MIN_PRICE, pit=pit)
    pools = sig.period_pools(prices, rb_dates, elig, execute_lag=EXEC_LAG)
    strat_rets = sig.strategy_period_returns(hl)
    ppy = {"Q": 4.0, "M": 12.0, "W": 52.0}.get(STRATEGY.freq, 12.0)
    mc = sig.monte_carlo_null(pools, strat_rets, k=STRATEGY.slots, ppy=ppy, n_trials=1000, seed=0)
    dsr = sig.deflated_sharpe_ratio(strat_rets, [c["full"]["sharpe"] for c in grid["cells"]], ppy=ppy)
    ci = sig.bootstrap_sharpe_cagr_ci(strat_rets, ppy=ppy, seed=0)

    n_countries = len({m.get("country") for m in meta.values()} - {"—", None})
    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta,
                strategy=STRATEGY, train=train, val=val, test=test, graveyard_hits=hits,
                grid=grid, n_dead=int(meta_df["delisting_date"].notna().sum()),
                n_countries=n_countries,
                significance=dict(mc=mc, dsr=dsr, ci=ci, ppy=ppy))


def sec_intro(d: dict) -> str:
    cfg = d["strategy"]
    nc = d["n_countries"]
    return (f'<div class="note"><b>Chosen strategy — {cfg.code} ({_desc(cfg)}).</b> '
            "Picked from the <b>32-config grid</b> for the highest worst-case "
            "<b>min(train, validation) Sharpe</b> among configs that pay for their own trading "
            "costs and are positive in both windows — so the result rides robustness, not one "
            "lucky rally. (Sector-neutral is excluded: the global universe carries no sector "
            "data, so it would be a silent no-op.) The universe is the liquid, "
            f"<b>Trade-Republic-investable</b> names across <b>{nc} countries</b>, each priced off "
            "its <b>home exchange × EUR FX</b> — the way Lang &amp; Schwarz actually fills you "
            "(NVIDIA on NASDAQ, Samsung on KRX, Rheinmetall on XETRA, in their own currency, "
            "converted to EUR). Behind a <b>≥100k/day turnover</b> floor. Long-only, walk-forward, "
            "executable. Not advice.</div>")


def sec_perf(d: dict, public: bool) -> str:
    full = d["res"]["runs"][1.0]["stats"]
    test = d["test"]
    out = ["<h2>Performance</h2>",
           "<p class='dim'>Train = 2018–21 (used to pick the config), validation = 2022–23 "
           "(used to compare configs), <b>test = 2024→ (held out — never touched the "
           "choice)</b>. The <b>test</b> column is the only truly out-of-sample number; "
           "trust it over the eye-popping full-window total.</p>"]
    cards = [
        _card("Test net return", _pct(test["net_return"] * 100)),
        _card("Test Sharpe", f"{test['sharpe']:.2f}"),
        _card("Test max DD", _pct(test["max_drawdown"] * 100)),
        _card("Full net return", _pct(full["net_return"] * 100)),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{full['net_return'] * d['capital']:+,.0f}"))
    out.append(f'<div class="cards">{"".join(cards)}</div>')
    rows = "".join(
        f"<tr><td>{name}</td><td class='num'>{_pct(s['net_return'] * 100)}</td>"
        f"<td class='num mono'>{s['sharpe']:.2f}</td>"
        f"<td class='num'>{_pct(s['max_drawdown'] * 100)}</td></tr>"
        for name, s in (("Train 2018–21", d["train"]), ("Validation 2022–23", d["val"]),
                        ("Test 2024→ (held out)", test), ("Full 2018→", full)))
    out.append("<table><tr><th>Window</th><th class='num'>Net return</th>"
               "<th class='num'>Sharpe</th><th class='num'>Max DD</th></tr>" + rows + "</table>")
    return "".join(out)


def _yearly_returns(series: pd.Series) -> pd.Series:
    """Calendar-year returns keyed by year int: each year from the prior year's last
    close to this year's last close; the first year runs from inception."""
    s = series.dropna()
    last = s.groupby(s.index.year).last()
    prev = last.shift(1)
    if len(prev):
        prev.iloc[0] = s.iloc[0]          # first year measured from inception
    return (last / prev - 1.0).dropna()


def sec_yearly(d: dict, public: bool) -> str:
    eq = d["res"]["runs"][1.0]["equity"].dropna()
    if len(eq) < 2:
        return ""
    sret = _yearly_returns(eq)
    bench = d["benchmarks"]
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else None
    bret = _yearly_returns(spx.reindex(eq.index).ffill()) if spx is not None else pd.Series(dtype=float)
    # € P&L per year — the actual paper-account change (private builds only)
    last = eq.groupby(eq.index.year).last()
    prev = last.shift(1)
    prev.iloc[0] = eq.iloc[0]
    pnl = (last - prev).dropna()

    rows = []
    for y, r in sret.items():
        eur = f"<td class='num mono'>€{pnl.get(y, 0.0):+,.0f}</td>" if not public else ""
        b = (f"<td class='num'>{_pct(bret[y] * 100)}</td>"
             if y in bret.index else "<td class='num dim'>—</td>")
        rows.append(f"<tr><td class='mono'>{y}</td>"
                    f"<td class='num'>{_pct(r * 100)}</td>{eur}{b}</tr>")
    eur_hdr = "<th class='num'>P&amp;L (€10k)</th>" if not public else ""
    pnl_note = ("the €10k paper account's actual P&amp;L, and " if not public else "and ")
    return ("<h2>Yearly P&amp;L</h2>"
            "<p class='dim'>Calendar-year net return of the strategy (first year from "
            f"inception), {pnl_note}the S&amp;P 500 over the same year. 2018 and 2026 are "
            "part-years.</p>"
            "<table><tr><th>Year</th><th class='num'>Strategy</th>" + eur_hdr +
            "<th class='num'>S&amp;P 500</th></tr>" + "".join(rows) + "</table>")


def sec_significance(d: dict, public: bool) -> str:
    s = d["significance"]
    mc, dsr, ci = s["mc"], s["dsr"], s["ci"]
    pct_beat = 100.0 * (1.0 - mc["p_sharpe"])

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc["null_sharpe"], nbinsx=40, name="random books",
                               marker_color=theme.FG_DIM, opacity=0.85))
    fig.add_vline(x=mc["strat_sharpe"], line_color="#dcdcaa", line_width=2.5,
                  annotation_text="this strategy", annotation_position="top")
    fig.add_vline(x=mc["null_sharpe_median"], line_color="#569cd6", line_dash="dash",
                  annotation_text="random median", annotation_position="bottom")
    fig.update_layout(height=340, bargap=0.02, showlegend=False,
                      xaxis_title="Gross annualised Sharpe (per-rebalance)",
                      yaxis_title="random books", margin=dict(t=20))

    cards = [
        _card("p-value vs random", f"{mc['p_sharpe']:.3f}"),
        _card("Beats random books", f"{pct_beat:.1f}%"),
        _card("Deflated Sharpe (P real>0)", "—" if dsr['dsr'] != dsr['dsr'] else f"{dsr['dsr']:.0%}"),
        _card(f"Sharpe {ci['conf']}% CI", f"{ci['sharpe_lo']:.2f} – {ci['sharpe_hi']:.2f}"),
    ]
    verdict = ("clears" if mc["p_sharpe"] < 0.05 else "does <b>not</b> clear")
    dsr_txt = ("—" if dsr["dsr"] != dsr["dsr"] else
               f"After haircutting for the <b>{dsr['n_trials']} configs we scanned</b>, the return "
               f"skew/kurtosis and the {dsr['T']}-period sample length, the <b>Deflated Sharpe</b> "
               f"puts P(true Sharpe&gt;0) at <b>{dsr['dsr']:.0%}</b> (benchmark a lucky winner had to "
               f"clear: {dsr['sr_benchmark_annual']:.2f} annualised).")
    return ("<h2>Significance &amp; robustness</h2>"
            "<p class='dim'>Three desk-grade sanity checks, all on <b>gross per-rebalance</b> "
            "returns so the comparison is pure selection (costs hit a random book about the same). "
            f"<b>(1) Better than noise?</b> {mc['n_trials']:,} random books — same eligible pool, "
            "same dates, {k} names picked at <i>random</i> each rebalance — give the grey null "
            "below; momentum’s Sharpe (gold line) {verdict} the 5% bar with "
            "<b>p = {p:.3f}</b>. The random median (blue) is the universe’s own drift — beating it "
            "is the actual edge. <b>(2) Real after scanning configs?</b> {dsr} <b>(3) How wide the "
            "error bar?</b> a circular block-bootstrap puts the annualised Sharpe’s {conf}% CI at "
            "<b>{slo:.2f}–{shi:.2f}</b> and CAGR at <b>{clo:.0f}%–{chi:.0f}%</b>.</p>"
            .format(k=d["strategy"].slots, verdict=verdict, p=mc["p_sharpe"], dsr=dsr_txt,
                    conf=ci["conf"], slo=ci["sharpe_lo"], shi=ci["sharpe_hi"],
                    clo=ci["cagr_lo"] * 100, chi=ci["cagr_hi"] * 100) +
            f'<div class="cards">{"".join(cards)}</div>'
            f"<div class='chart'>{fig_html(fig)}</div>"
            "<p class='dim'>A low p-value says the <i>selection</i> adds value over drawing names "
            "at random from the same liquid pool; it does not promise the level repeats. Read it "
            "with the regime and capacity caveats below.</p>")


def sec_timeline(d: dict) -> str:
    lines = []
    for h in d["res"]["holdings_log"]:
        dead = h.get("dead", set())
        spans = " ".join(
            f"<span style='color:{_pnl_color(h['ret'].get(t, 0.0), t in dead)}' "
            f"title='{t} {h['ret'].get(t, 0.0):+.0%}'>{_disp(d['meta'], t)}</span>"
            for t in h["picks"])
        spans = spans or "<span class='dim'>cash</span>"
        rv = [v for v in h["ret"].values() if pd.notna(v)]
        mret = sum(rv) / len(rv) if rv else 0.0
        lines.append(f"<div><span class='mono dim'>{h['date'].date()}</span> "
                     f"<b style='color:{_pnl_color(mret, False)}'>{mret:+.1%}</b> {spans}</div>")
    return ("<h2>Every rebalance, colored by outcome</h2>"
            "<p class='dim'>Each line is one rebalance’s picks, colored by that holding "
            "period’s return — <span style='color:#0a6b00'>■</span> ≥+20% · "
            "<span style='color:#46c84e'>■</span> up · <span style='color:#ef4444'>■</span> down · "
            "<span style='color:#7a0000'>■</span> ≤−20% · <span style='color:#000'>■</span> "
            "defaulted (delisted/died). Hover for the %.</p>"
            f"<div style='font-size:0.78rem;line-height:1.7'>{''.join(lines)}</div>")


def sec_caveat(d: dict) -> str:
    hits = d.get("graveyard_hits", 0)
    test_ret = d["test"]["net_return"] * 100
    surv = (f"it held <b>0</b> of them into death" if hits == 0 else
            f"it held <b>{hits}</b> into delisting, liquidated by the graveyard at the last price")
    return (
        f'<div class="note"><b>What’s honest here — and what isn’t the problem.</b> '
        f'<b>Survivorship is corrected <i>and</i> immaterial.</b> The universe carries '
        f'{d["n_dead"]} names that delisted/collapsed 2018→now, yet {surv}: momentum buys '
        f'<i>winners</i>, and a dying name ranks last long before it goes — so this strategy '
        f'structurally never owns the corpses. The headline is therefore <i>not</i> a '
        f'survivorship artifact. <b>Capacity isn’t the problem either</b> — every name clears '
        f'a <b>≥100k/day turnover floor</b>, so a small account deploys without moving a price.'
        f'<br><br>The real caveats: <b>(1) Regime</b> — 2024→ was an exceptional momentum '
        f'tape (defence + AI: Rheinmetall, Siemens Energy, Palantir); even the held-out '
        f'{test_ret:+.0f}% test figure is regime-specific and will <b>not</b> repeat. '
        f'<b>(2) Concentration</b> — top-{d["strategy"].slots}, with <b>no sector or geographic '
        f'cap</b> (no sector data), so the book can pile into one country or theme (the universe '
        f'leans heavily to the US, plus large Korea/Taiwan pools); a few names drive the curve and '
        f'one blow-up hurts. <b>(3) Universe membership</b> — a name is included if it cleared the '
        f'turnover floor at <i>any</i> point 2018→now, so there is a mild liquidity look-ahead on '
        f'top of survivorship; and <b>TR-routability is assumed</b> from that liquidity, not '
        f'verified per name. Coverage is by home exchange × FX (the L&amp;S model), active '
        f'<i>and</i> delisted; the genuine gaps are <b>Tokyo (Japan) and Milan (Italy)</b>, whose '
        f'home venues the data source doesn’t serve. <b>(4) Mechanics</b> — daily closes, €1/order, '
        f'slippage modeled not measured, and <b>past performance is not future returns</b>. The '
        f'out-of-sample test is the guard against curve-fitting, not a promise.</div>')


def build(d: dict, public: bool = False) -> str:
    """One page: the chosen strategy up top (intro → picks → equity → perf → timeline →
    caveats), then a <hr> and the research lab below (the 32-config grid, feasibility, every
    variation's timeline, the survivorship note, and the method). Local-only output."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = d["strategy"]
    body = "".join([
        f"<h1>Momentum strategy — {cfg.code}</h1>",
        f"<p class='dim'>generated {now} · <a href='report.html'>← portfolio</a></p>",
        # ── the chosen strategy ──
        sec_intro(d),
        sec_holdings(d),
        sec_curve(d),
        sec_perf(d, public),
        sec_significance(d, public),
        sec_yearly(d, public),
        sec_timeline(d),
        sec_caveat(d),
        # ── the lab (private/live only): how this config was chosen + all the rest ──
        ("".join([
            "<hr style='margin:3rem 0;border:0;border-top:2px solid #333'>",
            "<h1>Research lab</h1><p class='dim'>How the config above was chosen — the whole "
            "32-config grid it was picked from, and the supporting data. Skip unless you "
            "want the workings.</p>",
            sec_survivorship(d), sec_grid(d), sec_feasibility(d), sec_timelines(d), sec_method(),
        ]) if not public else ""),
    ])
    return page(f"Strategy — {cfg.code}", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = gather(refresh=args.refresh)
    local = ROOT / "local/strategy.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d))                          # live/local only — no docs/ export
    print(f"wrote {local}  (strategy {STRATEGY.code} + lab)")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
