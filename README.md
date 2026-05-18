# Monitor

Personal investment dashboard for Trade Republic portfolios. Collects signals from 13 data sources, scores every asset in the universe, and displays a live Streamlit dashboard with ratings, world view, and portfolio analytics.

## Architecture

```
collect_all.py          ← weekly data pipeline (no Claude API needed)
    │
    ▼
data/signals.json       ← compact signal snapshot (~700 KB)
    │
    ▼
[Claude Code session]   ← deep-rates assets scoring ≥ threshold
    │
    ▼
data/ratings_report.json
    │
    ▼
app.py (Streamlit)      ← dashboard: World View / Ratings / Portfolio
```

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/cxad1100/claudecode-a.git
cd claudecode-a
uv venv .venv --python 3.13 && uv pip install -r requirements.txt --python .venv
source .venv/bin/activate
```

**API keys** — create `input/.env`:
```
FRED_API_KEY=your_key_here     # free at fred.stlouisfed.org
ANTHROPIC_API_KEY=your_key     # optional, only needed for deep ratings
```

**Portfolio** — drop your Trade Republic CSV export at `input/portfolio.csv`:
```
Date,Ticker,Action,Shares,Price,PricePerShare
2024-11-18,ASML.AS,buy,0.25,160.30,641.20
...
```

## Usage

```bash
# Start dashboard (uses last collected data)
./start.sh

# Collect all signals — ~20-30 min, full universe
python collect_all.py

# Fast mode — ~5 min, 50 tickers, for testing
python collect_all.py --fast
```

## Signal Sources

| # | Source | Key Signal |
|---|--------|-----------|
| 1 | Wikipedia / seed lists | Full TR asset universe |
| 2 | FRED | Macro regime, rates, CPI, yield curve |
| 3 | yfinance futures | Sector tailwinds / headwinds |
| 4 | Polymarket | Event probabilities |
| 5 | GDELT | Regional conflict indices |
| 6 | Reuters RSS | Sector headlines |
| 7 | yfinance | Price history, fundamentals, analyst ratings |
| 8 | SEC EDGAR Form 4 | Insider buy/sell flow |
| 9 | yfinance options | Put/call ratio |
| 10 | yfinance | Short interest |
| 11 | yfinance | Bond yields + yield curve |
| 12 | yfinance | Currencies + sector ETF performance |
| 13 | yfinance | Extended commodities |
| + | Reddit WSB | Retail sentiment, squeeze flags |
| + | yfinance BTC | Liquidity / risk-on signal |

## Scoring

Fast scorer (`fast_scorer.py`) computes a composite 0–100 score per asset from earnings, insider flow, macro regime, geopolitics, fundamentals, options, WSB momentum, and cross-source themes. Scores are percentile-stretched across the universe so the full range is always used. Assets scoring above the threshold (`DEEP_RATING_THRESHOLD` in `config.py`) receive a Claude deep-analysis in Pass 2.

Grade scale: `AAA → AA+ → AA → AA- → A+ → A → A- → BBB+ → BBB → BBB- → BB+ → BB → BB- → B → CCC → CC`

A `⚠` suffix on a grade means fewer than 2 real data sources were found — the score is mostly defaults.

## Dashboard Tabs

- **World View** — macro regime, Polymarket events, themes, sentiment, currencies, bonds, commodities
- **Ratings** — full universe ranked by score, filterable by sector/region/type, deep-dive on click
- **Portfolio** — live prices (15 min delay), allocation bars by position and sector, P&L, ROI chart, quant metrics, correlation matrix

## Project Structure

```
app.py                  ← Streamlit dashboard
collect_all.py          ← 13-collector data pipeline
fast_scorer.py          ← composite scoring + grade assignment
config.py               ← model, thresholds, FRED series, risk rules
start.sh                ← one-click dashboard launcher
tools/
  universe_manager.py   ← asset universe (S&P500 + EU + ETFs)
  fred_tools.py         ← macro indicators
  yfinance_tools.py     ← price + fundamentals
  portfolio_tools.py    ← Trade Republic CSV parser + P&L
  portfolio_analytics.py← Sharpe, Sortino, Beta, drawdown, correlation
  polymarket_tools.py   ← prediction markets
  insider_tools.py      ← SEC Form 4
  options_tools.py      ← put/call ratio
  ...
input/                  ← personal data, gitignored
  .env                  ← API keys
  portfolio.csv         ← your trade history
data/                   ← generated outputs, gitignored
  universe.csv          ← auto-refreshed weekly
  signals.json          ← latest signal snapshot
  ratings_report.json   ← latest deep ratings
```

## Automated Weekly Run

A Claude Code remote agent runs every Sunday at 7pm Rome time via `collect_and_push.sh`, collecting fresh signals and pushing updated data to the repo.
