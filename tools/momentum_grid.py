"""The 64-permutation momentum matrix: the six blueprint upgrades (A–F) as a
config, run walk-forward over the survivorship-corrected universe, split into
train/validation, with a small-account feasibility calc. Pure (data in, numbers
out); the build script supplies prices/sectors/benchmark/pit.
"""
from dataclasses import dataclass
from itertools import product

import pandas as pd

from tools.momentum import run_momentum
from tools.pairs_backtest import backtest_stats


@dataclass(frozen=True)
class MomentumConfig:
    vol_adjust: bool = False        # A
    sector_neutral: bool = False    # B
    trend_filter: bool = False      # C
    slots: int = 15                 # D: 15 off / 10 on
    freq: str = "M"                 # E: "M" off / "Q" on
    lazy: bool = False              # F

    @property
    def code(self) -> str:
        flags = [self.vol_adjust, self.sector_neutral, self.trend_filter,
                 self.slots == 10, self.freq == "Q", self.lazy]
        return "".join(c if on else "·" for c, on in zip("ABCDEF", flags))

    def kwargs(self) -> dict:
        return dict(vol_adjust=self.vol_adjust, sector_neutral=self.sector_neutral,
                    trend_filter=self.trend_filter, k=self.slots, freq=self.freq,
                    lazy=self.lazy)


ALL_CONFIGS = [MomentumConfig(va, sn, tf, slots, freq, lz)
               for va, sn, tf, slots, freq, lz in product(
                   (False, True), (False, True), (False, True),
                   (15, 10), ("M", "Q"), (False, True))]


def _stats_slice(equity: pd.Series, trades: list, lo, hi, capital: float) -> dict:
    """backtest_stats over the [lo, hi] window, re-based to `capital` at the slice
    start so net_return/sharpe/drawdown describe that sub-period in isolation."""
    eq = equity.loc[lo:hi]
    tr = [t for t in trades if lo <= pd.Timestamp(t["entry"]) <= hi]
    if len(eq) < 2:
        return dict(net_return=0.0, sharpe=0.0, max_drawdown=0.0, win_rate=None,
                    total_costs=0.0)
    return backtest_stats(eq / eq.iloc[0] * capital, tr, capital)


def run_grid(prices, slippage_bps, *, sectors=None, benchmark=None, pit=None,
             configs=None, start="2018-01-01", train_end="2022-12-31",
             capital=10_000.0, lookback=252, skip=21) -> dict:
    """Run each config walk-forward over [start, end]; partition each equity curve +
    trades into train (≤ train_end), validation (> train_end) and the full window.

    Returns {"cells": [{code, config, train, val, full, trades_per_year}], train_end}.
    One run per config — the picks already use only past data, so the val slice is
    genuine out-of-sample. Rank by validation Sharpe, not train, to dodge overfit.
    """
    configs = configs or ALL_CONFIGS
    te = pd.Timestamp(train_end)
    cells = []
    for cfg in configs:
        r = run_momentum(prices, slippage_bps, capital=capital, cost_mults=(1.0,),
                         lookback=lookback, skip=skip, start=start, sectors=sectors,
                         benchmark=benchmark, pit=pit, **cfg.kwargs())
        eq = r["runs"][1.0]["equity"]
        tr = r["runs"][1.0]["trades"]
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9) if len(eq) else 1.0
        cells.append(dict(
            code=cfg.code, config=cfg,
            train=_stats_slice(eq, tr, eq.index[0], te, capital),
            val=_stats_slice(eq, tr, te + pd.Timedelta(days=1), eq.index[-1], capital),
            full=r["runs"][1.0]["stats"],
            trades_per_year=len(tr) / years,
            timeline=[dict(date=str(h["date"].date()), ret=h["ret"], dead=list(h["dead"]))
                      for h in r["holdings_log"]]))
    return dict(cells=cells, train_end=train_end)


def feasibility(cell: dict, *, capital: float = 10_000.0, fee_eur: float = 1.0) -> dict:
    """Phase-6: fixed-fee drag on a €`capital` account. annual_fee = trades/yr × fee;
    fee_drag_pct = that as % of capital; pays_for_itself = net return clears the drag."""
    annual_fee = cell["trades_per_year"] * fee_eur
    drag_pct = annual_fee / capital * 100.0
    return dict(annual_fee_eur=annual_fee, fee_drag_pct=drag_pct,
                pays_for_itself=cell["full"]["net_return"] * 100.0 > drag_pct)
