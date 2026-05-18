# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Install dependencies (requires uv: brew install uv)
uv venv .venv --python 3.13 && uv pip install -r requirements.txt --python .venv
source .venv/bin/activate

# .env required (copy and fill in):
# FRED_API_KEY=...   (free at https://fred.stlouisfed.org/docs/api/api_key.html)
# ANTHROPIC_API_KEY=...
```

## Running

```bash
# Collect all signals (full run, ~20-30 min)
python collect_all.py

# Fast test mode (50 tickers, skips validation, ~5 min)
python collect_all.py --fast

# Streamlit dashboard (reads data/signals.json + ratings_report.json)
streamlit run app.py

# Automated weekly collect + git push (runs via cron Sunday 7pm Rome)
bash collect_and_push.sh

# Run tests
.venv/bin/pytest tests/ -v

# Run a single test file
.venv/bin/pytest tests/test_fast_scorer.py -v
```

## Architecture

Two-pass weekly pipeline running every Sunday 7pm (Rome time) via Claude Code remote agent:

**Pass 1 (Python):** `collect_all.py` → 13 collectors → `data/signals.json`
**Pass 2 (Claude):** Reads `signals.json` → extracts events → rates all assets → `data/ratings_report.json`

```
data/universe.csv        ← all TR stocks + ETFs (auto-refreshed weekly)
     │
collect_all.py (13 collectors in sequence)
     │
data/signals.json        ← compact signal snapshot
     │
[Claude Code session]    ← extracts events A/B/C..., rates each asset
     │
data/ratings_report.json ← S&P-style grades + event portfolio matrix
     │
app.py (Streamlit)       ← dashboard: World View / Ratings / Portfolio tabs
```

## Signal Sources

| # | Collector | Source | Key Signal |
|---|---|---|---|
| 1 | UniverseManager | Wikipedia S&P 500 + seed European/ETF lists | Full TR asset list |
| 2 | Macro/FRED | fredapi | Regime + rates + CPI + yield curve |
| 3 | Futures | yfinance CL=F GC=F NG=F HG=F | Sector tailwinds/headwinds |
| 4 | Polymarket | gamma-api.polymarket.com | Event probabilities |
| 5 | GDELT | api.gdeltproject.org | Regional conflict indices |
| 6 | News | Reuters RSS | Sector headlines |
| 7 | Price + Fundamentals | yfinance | PE, ROE, earnings, analyst scores |
| 8 | InsiderFlow | SEC EDGAR Form 4 | Net insider buys/sells |
| 9 | OptionsFlow | yfinance options chain | Put-call ratio |
| 10 | ShortInterest | yfinance shortPercentOfFloat | Short float % |
| 11 | Bond yields + curve | yfinance | 2Y/10Y/30Y + spread |
| 12 | Currencies + sector ETFs | yfinance | FX, sector 1M performance leaders/laggards |
| 13 | Extended commodities | yfinance | Wheat, lumber, LNG, uranium, etc. |
| + | WSB | reddit.com/r/wallstreetbets | Ticker mentions, squeeze flags |
| + | BTC | yfinance BTC-USD | Liquidity/risk-on signal |

## Event Framework

Events A/B/C… extracted by Claude from Polymarket + GDELT + news. Each has `probability` + `complement_probability`. `event_extractor.py` validates coherence (probabilities in [0,1], mutually-exclusive pairs sum ≤ 1). Claude maps events to TR assets via multi-hop causal chains. `app.py` has hardcoded `IMPACT` causal-chain descriptions per event group × sector for the dashboard.

## Rating Scale

AAA / AA+ / AA / AA- / A+ / A / A- / BBB+ / BBB / BBB- / BB+ / BB / BB- / B / CCC / CC

Fast scorer (`fast_scorer.py`): composite 0-100 from earnings, insider, macro, momentum, geo, options, short-interest, fundamentals, price signals → grade. Assets scoring ≥70 (`DEEP_RATING_THRESHOLD` in `config.py`) get Claude deep analysis in Pass 2.

## Portfolio Tab

`tools/portfolio_tools.py` parses a Trade Republic trade-history CSV (`data/portfolio.csv`) into live holdings with average cost, unrealised/realised P&L, and per-position rating lookup. `TICKER_MAP` maps TR Frankfurt/Milan tickers (e.g. `NVD.F`) to yfinance price tickers; `RATING_LOOKUP` maps them to primary-exchange tickers for rating lookups (e.g. `NVD.F` → `NVDA`).

## Data Files

| File | Updated | Contents |
|---|---|---|
| `data/universe.csv` | Weekly | All TR assets (stocks + ETFs) |
| `data/signals.json` | Weekly | All 13 signal outputs |
| `data/ratings_report.json` | Weekly | Final S&P-style grades |
| `data/last_ratings.json` | Weekly | Previous week grades (for change detection) |
| `data/portfolio.csv` | Manual | TR trade-history export (for Portfolio tab) |

## Key Config (`config.py`)

- `MODEL = "claude-opus-4-7"` — model used by Pass 2 and legacy agents
- `DEEP_RATING_THRESHOLD = 70` — minimum fast-score for Claude deep rating
- `FRED_SERIES` — all FRED series IDs fetched in Pass 1
- `SECTOR_WEIGHTS_BY_REGIME` — target allocation weights by macro regime
- `RISK_RULES` — position/sector caps used by portfolio construction

## Legacy Pipeline

`pipeline.py` + `agents/` is an older 4-agent Claude-API system (data engineer → macro analyst → fundamental analyst → portfolio manager). Entry point: `python main.py`. This is separate from the current collect_all.py → signals.json workflow and is not part of the weekly run.
