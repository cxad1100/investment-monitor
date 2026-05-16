# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
pip install -r requirements.txt
# Set API keys in .env (copy from .env template)
python main.py
python main.py --universe data/tr_universe.csv --period 1y --output portfolio_report.json
```

## Architecture

4-agent sequential pipeline. Each agent uses Claude API tool-use loop and writes to a shared `state` dict.

```
main.py → pipeline.py → [data_engineer → macro_analyst → fundamental_analyst → portfolio_manager]
                                ↕              ↕                  ↕                  ↕
                          state["universe"] state["regime"]  state["ratings"]  state["portfolio"]
                          state["price_data"]
                          state["fundamentals"]
                          state["fred_data"]
```

**Agents** (`agents/`): Each agent has a system prompt, Claude API tool definitions, a tool executor (`_execute_tool`), and a `run(state) → state` function. Agents loop on `client.messages.create()` until `stop_reason == "end_turn"`.

**Tools** (`tools/`):
- `universe.py` — loads `data/tr_universe.csv`, maps ISINs to yfinance tickers
- `yfinance_tools.py` — price history, fundamentals, position sizing, portfolio stats
- `fred_tools.py` — FRED API fetch, quantitative regime classifier

**Config** (`config.py`): model ID, FRED series list, risk rules, sector weights by regime.

## Adding Custom Stocks

Edit `data/tr_universe.csv` or supply your own TR CSV export. Required columns: `isin, name, yf_ticker, sector, region`. The `yf_ticker` can be omitted — it's derived from the ISIN country prefix + region suffix.

## API Keys

Both keys are required. Add to `.env`:
- `ANTHROPIC_API_KEY` — from console.anthropic.com
- `FRED_API_KEY` — free at fred.stlouisfed.org/docs/api/api_key.html

## Model

Uses `claude-opus-4-7` with `thinking: {type: "adaptive"}`. Change in `config.py → MODEL`.
