# Monitor

Investment dashboard for Trade Republic portfolios. Collects signals from 13 sources, scores every asset, and serves a live Streamlit dashboard.

## Setup

```bash
git clone https://github.com/cxad1100/claudecode-a.git
cd claudecode-a
uv venv .venv --python 3.13 && uv pip install -r requirements.txt --python .venv
source .venv/bin/activate
```

Create `input/.env`:
```
FRED_API_KEY=your_key    # free at fred.stlouisfed.org
```

Optionally add `input/portfolio.csv` (Trade Republic CSV export):
```
Date,Ticker,Action,Shares,Price,PricePerShare
2024-11-18,ASML.AS,buy,0.25,160.30,641.20
```

## Usage

```bash
./start.sh                   # launch dashboard (uses last collected data)
python collect_all.py        # collect all signals (~20-30 min)
python collect_all.py --fast # fast mode, 50 tickers (~5 min)
```

## Signal Sources

FRED macro, commodity futures, Polymarket, GDELT conflict indices, Reuters RSS, yfinance (prices, fundamentals, options, short interest, bonds, currencies, sector ETFs, commodities), SEC insider flow, Reddit WSB, BTC.

## Scoring

`fast_scorer.py` computes a 0–100 composite per asset (earnings, insider flow, macro, geopolitics, fundamentals, options, momentum). Scores are percentile-stretched across the full universe. `⚠` on a grade = fewer than 2 real data sources, mostly defaults.

## Dashboard

- **World View** — macro regime, events, themes, sentiment, bonds, FX, commodities  
- **Ratings** — full universe ranked by score, filterable, deep-dive on click  
- **Portfolio** — live prices, allocation bars, P&L, ROI vs benchmarks, quant metrics, correlation matrix
