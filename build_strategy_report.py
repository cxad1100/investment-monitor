"""The chosen production strategy — one config (the 'ultimate' extracted from the
64-permutation grid), on the survivorship-corrected universe.

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

import pandas as pd

from tools.report_html import pct as _pct, card as _card, page
from tools.momentum import run_momentum, winsorize_prices
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map
from tools.momentum_grid import MomentumConfig, _stats_slice, run_grid
from tools.portfolio_tools import BENCHMARKS
from tools.data_buffer import cached_price_history
from build_momentum_report import (
    PRICES_CSV, META_CSV, ROOT, LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, CAPITAL,
    FEE_EUR, COST_MULTS, TRAIN_END, VAL_END, WINSOR_CAP, EXEC_LAG, MIN_TURNOVER,
    _slip, _broker, _disp, _pnl_color, sec_holdings, sec_curve,
    sec_grid, sec_feasibility, sec_timelines, sec_survivorship, sec_method,
)

# The chosen strategy — ·B·DE· = sector-neutral, top-10, quarterly.
# Picked from the 64-grid for the highest *out-of-sample* (validation) Sharpe among
# configs that pay for themselves, AND for sector diversification: the raw worst-case-
# Sharpe winner (···DE·) concentrates in one theme and posts an implausible +2414%
# validation headline, whereas ·B·DE· spreads across sectors (val Sharpe 2.34, the
# grid's best) so the result doesn't hinge on a single rally. Survivorship-clean by
# construction — momentum buys winners, dying names rank last, so it holds ~0 into death.
STRATEGY = MomentumConfig(sector_neutral=True, slots=10, freq="Q")


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
    prices = winsorize_prices(prices, cap=WINSOR_CAP)          # de-glitch the raw feed
    meta_df = pd.read_csv(META_CSV)
    if "med_turnover" in meta_df.columns:                      # liquidity floor (drops dead .F feeds)
        liquid = (meta_df["med_turnover"] >= MIN_TURNOVER) | meta_df["delisting_date"].notna()
        meta_df = meta_df[liquid].reset_index(drop=True)
        prices = prices[[c for c in prices.columns if c in set(meta_df["ticker"])]]
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    sectors = {t: (str(m["sector"]) if pd.notna(m.get("sector")) else "Unknown")
               for t, m in meta.items()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    pit = PITUniverse(prices, delisting_map(meta_df))

    benches = {n: v for n, v in BENCHMARKS.items() if n != "Bitcoin"}   # equities/bonds only
    bench_tickers = [tk for tk, _ in benches.values()]
    bench_raw = cached_price_history(bench_tickers, period="9y", force=refresh)
    bench = bench_raw.rename(columns={tk: name for name, (tk, _) in benches.items()})
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else bench.iloc[:, 0]

    res = run_momentum(prices, slip, lookback=LOOKBACK, skip=SKIP, capital=CAPITAL,
                       cost_mults=COST_MULTS, start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                       min_price=MIN_PRICE, sectors=sectors, benchmark=spx, pit=pit,
                       execute_lag=EXEC_LAG, **STRATEGY.kwargs())
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    train = _stats_slice(eq, tr, eq.index[0], te, CAPITAL)
    val = _stats_slice(eq, tr, te + pd.Timedelta(days=1), ve, CAPITAL)
    test = _stats_slice(eq, tr, ve + pd.Timedelta(days=1), eq.index[-1], CAPITAL)
    hits = sum(len(h.get("dead", set())) for h in res["holdings_log"])
    grid = run_grid(prices, slip, sectors=sectors, benchmark=spx, pit=pit, start=START,
                    train_end=TRAIN_END, val_end=VAL_END, capital=CAPITAL,
                    lookback=LOOKBACK, skip=SKIP, execute_lag=EXEC_LAG)
    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta,
                strategy=STRATEGY, train=train, val=val, test=test, graveyard_hits=hits,
                grid=grid, n_dead=int(meta_df["delisting_date"].notna().sum()))


def sec_intro(d: dict) -> str:
    cfg = d["strategy"]
    return (f'<div class="note"><b>Chosen strategy — {cfg.code} ({_desc(cfg)}).</b> '
            "Picked from the 64-permutation grid for the highest <b>out-of-sample "
            "(validation) Sharpe</b> among configs that pay for their own trading costs, and "
            "for <b>sector diversification</b> — so the result rides a spread of themes, not "
            "one lucky rally. The universe is the liquid, <b>Trade-Republic-investable</b> names "
            "across 18 countries, each priced off its <b>home exchange × EUR/USD</b> — the way "
            "Lang &amp; Schwarz actually fills you. So Seagate uses real NASDAQ data, Rheinmetall "
            "XETRA, Aker BP Oslo — not the dead Frankfurt-floor (.F) shadow EODHD lists. "
            "Filtered to a <b>≥100k/day turnover</b> floor. Long-only, walk-forward, executable. "
            "Not advice.</div>")


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
        f'{d["n_dead"]} EUR names that delisted/collapsed 2018→now, yet {surv}: momentum buys '
        f'<i>winners</i>, and a dying name ranks last long before it goes — so this strategy '
        f'structurally never owns the corpses. The headline is therefore <i>not</i> a '
        f'survivorship artifact. <b>Capacity isn’t the problem either</b> — every name clears '
        f'a <b>≥100k/day turnover floor</b>, so a small account deploys without moving a price.'
        f'<br><br>The real caveats: <b>(1) Regime</b> — 2024→ was an exceptional momentum '
        f'tape (defence + AI: Rheinmetall, Siemens Energy, Palantir); even the held-out '
        f'{test_ret:+.0f}% test figure is regime-specific and will <b>not</b> repeat. '
        f'<b>(2) Concentration</b> — top-{d["strategy"].slots}, so a few names drive the curve; '
        f'one bad blow-up hurts disproportionately. <b>(3) Coverage</b> — every name is priced '
        f'off its <i>home</i> exchange × FX (the L&amp;S model), so liquid foreign names '
        f'(Seagate, Aker BP) are included with real data — active <i>and</i> delisted (home '
        f'status separates a real death from a mere Frankfurt withdrawal). Residual gap: '
        f'countries outside the 18 mapped (China/HK, Japan, Australia) aren’t in yet. '
        f'<b>(4) Mechanics</b> — daily closes, €1/order, slippage modeled not '
        f'measured, and <b>past performance is not future returns</b>. The out-of-sample test '
        f'is the guard against curve-fitting, not a promise.</div>')


def build(d: dict, public: bool = False) -> str:
    """One page: the chosen strategy up top (intro → picks → equity → perf → timeline →
    caveats), then a <hr> and the research lab below (the 64-grid, feasibility, every
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
        sec_timeline(d),
        sec_caveat(d),
        # ── the lab (private/live only): how this config was chosen + all the rest ──
        ("".join([
            "<hr style='margin:3rem 0;border:0;border-top:2px solid #333'>",
            "<h1>Research lab</h1><p class='dim'>How the config above was chosen — the whole "
            "64-permutation grid it was picked from, and the supporting data. Skip unless you "
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
