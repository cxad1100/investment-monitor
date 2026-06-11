# Statistical Arbitrage (Pairs Trading) Engine — Design

**Date:** 2026-06-11
**Status:** Approved
**Goal:** Standalone algorithmic-trading showcase inside `investment-monitor`: find
cointegrated stock pairs, trade the spread on z-score signals, backtest walk-forward
with realistic costs and zero look-ahead bias. Output is a static HTML page in the
same style as the main report.

## Scope decisions (made with user)

| Decision | Choice |
|---|---|
| Universe | Curated ~30 large caps (extendable), all tradeable on Lang & Schwarz Exchange (Trade Republic) |
| Backtester | Custom vectorized pandas/numpy (no Backtrader/VectorBT) |
| Packaging | Separate page: `build_pairs_report.py` → `local/pairs.html` + `docs/pairs.html` |
| Structure | Walk-forward: 12-month formation → 3-month trading → roll 3 months |
| New dependency | `statsmodels` only |

## Architecture

```
investment-monitor/
├─ build_pairs_report.py        # orchestrator → local/pairs.html + docs/pairs.html
└─ tools/
   ├─ pairs_universe.py         # curated tickers, tagged {sector, currency}
   ├─ pairs_engine.py           # pure functions: Engle-Granger, hedge ratio, signals
   └─ pairs_backtest.py         # vectorized walk-forward backtester, costs + slippage
```

- Reuses `tools/theme.py` (vsdark plotly template, REPORT_CSS).
- Main report (`build_report.py`) gets a nav link to `pairs.html`.
- `public` flag plumbed through section functions for consistency with the main
  report: private page may show simulated € P&L, public page shows percentages.
  (No personal data is involved — this is a paper-trading simulation.)

## Data layer

- yfinance daily closes, ~5 years of history (12m formation + ≥4y of rolled
  trading windows).
- Prices in each ticker's **native currency** — no FX conversion.
- Cached to `data/pairs_prices.parquet`; rebuilds reuse cache, a `--refresh`
  flag forces re-download.

## Universe (`pairs_universe.py`)

Hardcoded dict (~30 names, easy to extend), every name confirmed Trade Republic /
LS-Exchange tradeable. Approximate groups:

- US banks: JPM, BAC, C, WFC, GS, MS
- US semis: NVDA, AMD, AVGO, TXN, INTC, MU
- US staples: KO, PEP, PG, CL
- US payments: V, MA, AXP
- US energy: XOM, CVX, COP
- DE/EU autos: BMW.DE, MBG.DE, VOW3.DE, P911.DE
- DE industrials/chemicals: SIE.DE, BAS.DE, BAYN.DE
- EU banks: DBK.DE, CBK.DE

**Candidate pairs = same sector AND same currency only.** Cross-currency pairs
leak the FX trend into the spread and produce spurious cointegration.

## Engine math (`pairs_engine.py`, pure functions)

### Engle-Granger two-step (per candidate pair, formation-window log prices)

1. OLS: `log(P_A) = α + β·log(P_B)` → hedge ratio β; residual series = spread.
2. Unit-root test on residuals via `statsmodels.tsa.stattools.coint`, which uses
   the correct Engle-Granger critical values (plain `adfuller` on estimated
   residuals uses the wrong distribution — this distinction is called out in the
   report's method explainer).
3. Run both orientations (A on B, B on A); keep the one with the lower p-value.

### Pair selection per formation window

- Cointegration p-value < 0.05, **and**
- Spread half-life ∈ [2, 60] trading days, computed from an AR(1) fit on the
  spread (filters pairs that revert too slowly to trade inside a 3-month window).
- Cap: top 10 pairs by p-value per window, to bound multiple-testing damage.
- Report displays "N candidates tested → k selected" so the multiple-comparisons
  caveat is visible, not hidden.

### Signals (trading window)

- `z = (spread − μ_formation) / σ_formation`; β, μ, σ are **frozen at formation
  end** and never re-estimated inside the trading window.
- Enter when |z| ≥ 2.0: short the rich leg, long the cheap leg, β-weighted,
  dollar-neutral.
- Exit when z crosses 0.
- Stop-loss when |z| ≥ 3.5 (treat cointegration as broken).
- Force-close all open positions at the end of each 3-month trading window.

### Look-ahead protections

- Formation/trading split: every parameter used in a trading window is estimated
  strictly before it.
- Execution lag: a signal computed on the close of day *t* executes at the close
  of day *t+1*.
- Test enforces it: truncating future data must not change past signals.

## Backtester (`pairs_backtest.py`, vectorized pandas)

- Paper account: €10,000 notional. Equal capital slice per pair selected in the
  window (as implemented: capital / n_selected, up to top-10 pairs → ≥€1,000 per
  pair; an earlier draft said "max ~5 → €2,000 per pair").
- **Costs per trade leg:** €1 fixed (Trade Republic order fee) + slippage as a
  half-spread estimate in bps, tiered: 5 bps US mega-caps, 10 bps DAX names,
  15 bps mid-liquidity names. Charged on entry and exit, both legs — a
  round-trip pair trade pays 4 × (fee + slippage).
- Shorting is simulated; Trade Republic offers no shorting. The report states
  this limitation plainly.
- Outputs: equity curve, per-pair trade ledger (entry/exit dates, z at entry,
  gross and net P&L), aggregate stats (net Sharpe, max drawdown, win rate,
  average holding days, total costs paid).
- **Cost sensitivity:** the full backtest is re-run at 0×, 1×, 2× the cost
  assumptions to show whether the edge survives frictions.

## Report (`build_pairs_report.py`)

Section functions mirroring `build_report.py` style:

1. **Method explainer** — Engle-Granger, why correlation ≠ cointegration.
2. **Current snapshot** — pairs cointegrated in the latest formation window,
   live z-scores, highlighted "signal now" pairs at |z| ≥ 2. Makes the page a
   live monitor, not only a backtest archive.
3. **Spread & z-score charts** for selected pairs (plotly, vsdark).
4. **Walk-forward backtest** — equity curve vs cash, stat cards, trade ledger.
5. **Cost sensitivity + honest-caveats box** — multiple testing, simulated
   shorting, survivorship bias in the curated universe.

## Tests (`tests/test_pairs_engine.py`)

- Synthetic cointegrated pair (shared random walk + stationary AR(1) noise)
  → must be detected; two independent random walks → must be rejected.
- Half-life computation on an AR(1) series with known coefficient.
- Signal transitions (enter / exit / stop) on a hand-built z-score series.
- Backtester P&L: small fabricated price frame with known trades → assert exact
  net P&L including fees and slippage.
- No-look-ahead: truncating future rows leaves past signals unchanged.

## Out of scope (future work)

- Kalman-filter dynamic hedge ratio.
- Johansen test / >2-asset baskets.
- Intraday data, live order execution.
