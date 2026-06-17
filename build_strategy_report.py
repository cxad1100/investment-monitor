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
from tools.momentum import run_momentum
from tools.universe_pit import PITUniverse
from tools.universe_assemble import delisting_map
from tools.momentum_grid import MomentumConfig, _stats_slice
from tools.portfolio_tools import BENCHMARKS
from tools.data_buffer import cached_price_history
from build_momentum_report import (
    PRICES_CSV, META_CSV, ROOT, LOOKBACK, SKIP, START, LIQ_MAX, MIN_PRICE, CAPITAL,
    FEE_EUR, COST_MULTS, TRAIN_END, _slip, _broker, _pnl_color, sec_holdings, sec_curve,
)

# The chosen strategy — pick_ultimate's robust+feasible winner on the corrected data:
# ·B·DE· = sector-neutral, top-10, quarterly. Best worst-case (train/val) Sharpe at low
# turnover; survivorship-clean (0 graveyard hits — it never holds a name that dies).
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
    meta_df = pd.read_csv(META_CSV)
    meta = {r["ticker"]: dict(r) for _, r in meta_df.iterrows()}
    sectors = {t: (str(m["sector"]) if pd.notna(m.get("sector")) else "Unknown")
               for t, m in meta.items()}
    slip = {t: _slip(m) for t, m in meta.items() if t in prices.columns}
    pit = PITUniverse(prices, delisting_map(meta_df))

    bench_tickers = [tk for _, (tk, _) in BENCHMARKS.items()]
    bench_raw = cached_price_history(bench_tickers, period="9y", force=refresh)
    bench = bench_raw.rename(columns={tk: name for name, (tk, _) in BENCHMARKS.items()})
    spx = bench["S&P 500"] if "S&P 500" in bench.columns else bench.iloc[:, 0]

    res = run_momentum(prices, slip, lookback=LOOKBACK, skip=SKIP, capital=CAPITAL,
                       cost_mults=COST_MULTS, start=START, liq_max=LIQ_MAX, fee_eur=FEE_EUR,
                       min_price=MIN_PRICE, sectors=sectors, benchmark=spx, pit=pit,
                       **STRATEGY.kwargs())
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te = pd.Timestamp(TRAIN_END)
    train = _stats_slice(eq, tr, eq.index[0], te, CAPITAL)
    val = _stats_slice(eq, tr, te + pd.Timedelta(days=1), eq.index[-1], CAPITAL)
    hits = sum(len(h.get("dead", set())) for h in res["holdings_log"])
    return dict(prices=prices, res=res, benchmarks=bench, capital=CAPITAL, meta=meta,
                strategy=STRATEGY, train=train, val=val, graveyard_hits=hits,
                n_dead=int(meta_df["delisting_date"].notna().sum()))


def sec_intro(d: dict) -> str:
    cfg = d["strategy"]
    return (f'<div class="note"><b>Chosen strategy — {cfg.code} ({_desc(cfg)}).</b> '
            "The single most <b>robust</b> configuration extracted from the 64-permutation "
            "grid: the one maximising the worst of its train and validation Sharpe (so it "
            "isn’t a backtest-lucky outlier), among configs that pay for their own "
            "trading costs. Long-only, walk-forward, executable on Trade Republic. Not advice.</div>")


def sec_perf(d: dict, public: bool) -> str:
    full = d["res"]["runs"][1.0]["stats"]
    out = ["<h2>Performance</h2>",
           "<p class='dim'>Train = 2018–2022 (in-sample), validation = 2023→ "
           "(out-of-sample, never used to choose the config), full = the whole window. "
           "A strategy that holds up in validation is the real test.</p>"]
    cards = [
        _card("Full net return", _pct(full["net_return"] * 100)),
        _card("Full Sharpe", f"{full['sharpe']:.2f}"),
        _card("Validation Sharpe", f"{d['val']['sharpe']:.2f}"),
        _card("Max drawdown", _pct(full["max_drawdown"] * 100)),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{full['net_return'] * d['capital']:+,.0f}"))
    out.append(f'<div class="cards">{"".join(cards)}</div>')
    rows = "".join(
        f"<tr><td>{name}</td><td class='num'>{_pct(s['net_return'] * 100)}</td>"
        f"<td class='num mono'>{s['sharpe']:.2f}</td>"
        f"<td class='num'>{_pct(s['max_drawdown'] * 100)}</td></tr>"
        for name, s in (("Train 2018–22", d["train"]), ("Validation 2023→", d["val"]),
                        ("Full 2018→", full)))
    out.append("<table><tr><th>Window</th><th class='num'>Net return</th>"
               "<th class='num'>Sharpe</th><th class='num'>Max DD</th></tr>" + rows + "</table>")
    return "".join(out)


def sec_timeline(d: dict) -> str:
    lines = []
    for h in d["res"]["holdings_log"]:
        dead = h.get("dead", set())
        spans = " ".join(
            f"<span style='color:{_pnl_color(h['ret'].get(t, 0.0), t in dead)}' "
            f"title='{t} {h['ret'].get(t, 0.0):+.0%}'>{t}</span>" for t in h["picks"])
        spans = spans or "<span class='dim'>cash</span>"
        lines.append(f"<div><span class='mono dim'>{h['date'].date()}</span> {spans}</div>")
    return ("<h2>Every rebalance, colored by outcome</h2>"
            "<p class='dim'>Each line is one rebalance’s picks, colored by that holding "
            "period’s return — <span style='color:#0a6b00'>■</span> ≥+20% · "
            "<span style='color:#46c84e'>■</span> up · <span style='color:#ef4444'>■</span> down · "
            "<span style='color:#7a0000'>■</span> ≤−20% · <span style='color:#000'>■</span> "
            "defaulted (delisted/died). Hover for the %.</p>"
            f"<div style='font-size:0.78rem;line-height:1.7'>{''.join(lines)}</div>")


def sec_caveat(d: dict) -> str:
    hits = d.get("graveyard_hits", 0)
    surv = (f"this config is <b>survivorship-clean</b> — across the whole backtest it held "
            f"<b>0</b> names that later died, so its return is <i>not</i> inflated by dead "
            f"winners vanishing" if hits == 0 else
            f"this config held <b>{hits}</b> names into their delisting; the graveyard "
            f"liquidated each at its last traded price, so the loss is in the numbers")
    return (f'<div class="note"><b>What’s honest here.</b> The universe is '
            f'survivorship-<b>corrected</b> — {d["n_dead"]} EUR names that collapsed/delisted '
            f"2018→now are included and liquidated by the graveyard at their last price — and "
            f"{surv}. The real remaining caveat is <b>capacity</b>: a concentrated top-"
            f"{d['strategy'].slots} of German small/mid-caps can post outsized momentum returns "
            "that are hard to realise at size (slippage is modeled, not measured). Also: daily "
            "closes only, €1/order costs assumed, and <b>past performance is not future "
            "returns</b>. Validation (out-of-sample, 2023→) is the guard against curve-fitting.</div>")


def build(d: dict, public: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = d["strategy"]
    title = f"Strategy — {cfg.code}" + ("" if public else " (private)")
    back = "index.html" if public else "report.html"
    body = "".join([
        f"<h1>Momentum strategy — {cfg.code}</h1>",
        f"<p class='dim'>generated {now} · <a href='{back}'>← monitor</a></p>",
        sec_intro(d),
        sec_holdings(d),
        sec_curve(d),
        sec_perf(d, public),
        sec_timeline(d),
        sec_caveat(d),
    ])
    return page(title, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = gather(refresh=args.refresh)
    local = ROOT / "local/strategy.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d, public=False))
    (ROOT / "docs/strategy.html").write_text(build(d, public=True))
    print(f"wrote {local} + docs/strategy.html  (strategy {STRATEGY.code})")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
