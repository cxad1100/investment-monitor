# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

A **static-HTML investment monitor**. `build_report.py` reads a Trade Republic
trade-history CSV, computes everything live from yfinance, runs a Markowitz optimizer,
and writes self-contained dark HTML reports. **No Streamlit, no server, no API keys.**

Two outputs per run:
- `local/report.html` — private, full € amounts (gitignored)
- `docs/index.html` — public for GitHub Pages: **percentages only — never € amounts,
  share counts, costs, or transactions**. Any new section must respect the `public`
  flag passed through `build()`/section functions.

> The old Streamlit multi-agent rating system lives in `../investment-monitor-full`.

## Setup / run

```bash
uv venv .venv --python 3.13 && uv pip install -r requirements.txt --python .venv
bash start.sh                       # build + open private report
.venv/bin/python build_report.py    # build only
.venv/bin/pytest tests/             # tests
```

Deploy = commit `docs/index.html` + push; GitHub Pages serves `/docs` on master.

## Architecture

```
input/portfolio.csv               ← trade history (gitignored, NEVER commit)
build_report.py                   ← data → optimizer → HTML (section functions, public flag)
 ├─ tools/portfolio_tools.py      ← parse CSV → holdings, avg cost, P&L; live prices
 ├─ tools/portfolio_analytics.py  ← ROI time-series, quant metrics, correlation
 ├─ tools/optimizer.py            ← Markowitz (scipy SLSQP), pure functions
 ├─ tools/portfolio_meta.py       ← sector map, ETF decomposition (sector_exposure_matrix)
 └─ tools/theme.py                ← VSCode Dark+ palette, plotly "vsdark" template, REPORT_CSS
```

Optimizer settings are constants at the top of `build_report.py`
(`LOOKBACK_DAYS=365`, `RF=0.045`, `LONG_ONLY=True`, `MAX_W=0.35`, `REB_FREQ="M"`).

## Calculation invariants (hard-won — don't regress)

- **One ROI formula everywhere**: `(holdings value + cash from sells) / sum of all buys − 1`.
  Sale proceeds count as cash; benchmarks are cash-flow matched (same EUR, same dates).
- `build_roi_timeseries` consumes transactions with a `<=` date pointer —
  weekend-dated trades (Tradegate Sundays) apply the next business day; exact-date
  lookup against `bdate_range` silently drops them (caused a 2pp ROI bug).
- `rolling_backtest` estimates weights from a **calendar-day** window strictly
  **before** each rebalance date — no look-ahead.
- Backtest universe = current holdings ⇒ selection bias; the report's honest comparison
  is Optimized vs Equal-Weight, and the caveat box must stay.
- Two weight recommendations: **A** `optimize(objective="sharpe")`,
  **B** `max_return_at_vol(vol_cap=current portfolio vol)`.

## Key maps (`tools/portfolio_tools.py`)

- `TICKER_MAP` — TR ticker → yfinance price ticker (e.g. `EUNL.F` → `IWDA.AS`)
- `COMPANY_NAMES` — display names
- `BENCHMARKS` — EUR-listed ETFs + USD ones FX-converted

## Adding a new holding

Append the row to `input/portfolio.csv` (`Date,Ticker,Action,Shares,Price,PricePerShare`)
with the exchange-suffixed yfinance ticker. New ticker → add to `TICKER_MAP` (if price
ticker differs), `COMPANY_NAMES`, and `PORTFOLIO_SECTOR_MAP` in `tools/portfolio_meta.py`.
Then rebuild: `bash start.sh`.
