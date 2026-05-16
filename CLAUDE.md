# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
# Install dependencies (requires uv: brew install uv)
uv venv .venv --python 3.13 && uv pip install -r requirements.txt --python .venv
source .venv/bin/activate

# Collect all signals (full run, ~20 min)
python collect_all.py

# Fast test mode (50 tickers, skips validation, ~5 min)
python collect_all.py --fast

# Run tests
.venv/bin/pytest tests/ -v
```

## Architecture

Two-pass weekly pipeline running every Sunday 7pm (Rome time) via Claude Code remote agent:

**Pass 1 (Python):** `collect_all.py` → 9 collectors → `data/signals.json`
**Pass 2 (Claude):** Reads `signals.json` → extracts events → rates all assets → `data/ratings_report.json`

```
data/universe.csv        ← all TR stocks + ETFs (auto-refreshed weekly)
     │
collect_all.py (9 collectors in sequence)
     │
data/signals.json        ← compact signal snapshot (~15KB)
     │
[Claude Code session]    ← extracts events A/B/C..., rates each asset
     │
data/ratings_report.json ← S&P-style grades + event portfolio matrix
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
| 7 | InsiderFlow | SEC EDGAR Form 4 | Net insider buys/sells |
| 8 | OptionsFlow | yfinance options chain | Put-call ratio |
| 9 | ShortInterest | yfinance shortPercentOfFloat | Short float % |
| + | WSB | reddit.com/r/wallstreetbets | Ticker mentions, squeeze flags |
| + | BTC | yfinance BTC-USD | Liquidity/risk-on signal |

## Event Framework

Events A/B/C… extracted by Claude from Polymarket + GDELT + news. Each has P(event) + P(¬event). `event_extractor.py` enforces coherence (no contradictions, mutual exclusivity). Claude maps events to TR assets via multi-hop causal chains.

## Rating Scale

AAA / AA+ / AA / AA- / A+ / A / A- / BBB+ / BBB / BBB- / BB+ / BB / BB- / B / CCC / CC

Fast scorer (Python): composite 0-100 → grade. Deep rating (Claude): rationale + event exposures for top movers.

## Data Files

| File | Updated | Contents |
|---|---|---|
| `data/universe.csv` | Weekly | All TR assets (stocks + ETFs) |
| `data/signals.json` | Weekly | All 9 signal outputs |
| `data/ratings_report.json` | Weekly | Final S&P-style grades |
| `data/last_ratings.json` | Weekly | Previous week grades (for change detection) |

## Tests

```bash
.venv/bin/pytest tests/ -v
```
