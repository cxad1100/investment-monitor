# Alpha Rating System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a weekly multi-signal investment rating system that rates every TR stock and ETF with an S&P-style letter grade using Polymarket, GDELT, Reddit WSB, insider flow, options, BTC, and a probabilistic event framework.

**Architecture:** Python collectors compress 9 data sources into a compact `signals.json`. A Claude Code session reads signals, extracts probabilistic events (A/B/C… with P(event)), maps events to assets via causal chain reasoning, and rates all assets (Python fast-score all → Claude deep-rate top movers). Output is `ratings_report.json` committed weekly.

**Tech Stack:** yfinance, fredapi, requests, feedparser, pandas, numpy, python-dotenv. No Anthropic API key — Claude Pro only via scheduled remote agent.

---

## File Map

**New tools (pure Python collectors):**
- `tools/universe_manager.py` — build and weekly-refresh full TR asset list
- `tools/futures_tools.py` — commodity futures via yfinance (CL=F, GC=F, NG=F, HG=F)
- `tools/polymarket_tools.py` — Polymarket Gamma API (geo + earnings markets)
- `tools/gdelt_tools.py` — GDELT API v2 (country conflict indices)
- `tools/news_tools.py` — RSS news feeds (Reuters, FT sector headlines)
- `tools/insider_tools.py` — SEC EDGAR Form 4 (net insider buys/sells)
- `tools/options_tools.py` — yfinance options chain (put-call ratio, OI skew)
- `tools/short_interest_tools.py` — yfinance short float + FINRA
- `tools/wsb_tools.py` — Reddit r/wallstreetbets public JSON (tickers, squeezes, sector hype)
- `tools/btc_tools.py` — BTC-USD trend + liquidity/anti-establishment regime

**New core scripts:**
- `event_extractor.py` — reads geo+macro signals, produces events.json (called by collect_all.py as a shell step; Claude does one mini-analysis)
- `fast_scorer.py` — Python pass 1: compute composite score for every asset
- `collect_all.py` — orchestrates all 9 collectors → signals.json (replaces old collect_data.py)

**Data files:**
- `data/universe.csv` — full TR universe, stocks + ETFs (replaces data/tr_universe.csv)
- `data/signals.json` — weekly snapshot (input to Claude rating session)
- `data/events.json` — extracted probabilistic events (part of signals.json)
- `data/last_ratings.json` — previous week grades (for change detection)
- `data/ratings_report.json` — final weekly output

**Tests:**
- `tests/test_universe_manager.py`
- `tests/test_futures_tools.py`
- `tests/test_polymarket_tools.py`
- `tests/test_gdelt_tools.py`
- `tests/test_wsb_tools.py`
- `tests/test_btc_tools.py`
- `tests/test_event_coherence.py`
- `tests/test_fast_scorer.py`

**Modified:**
- `requirements.txt` — add feedparser
- `config.py` — add new constants
- `CLAUDE.md` — update docs

---

## Task 1: Dependencies and Test Infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add feedparser to requirements**

```
yfinance>=0.2.40
fredapi>=0.5.2
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
feedparser>=6.0.10
pytest>=7.0.0
```

- [ ] **Step 2: Install**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 3: Create test conftest with shared fixtures**

```python
# tests/conftest.py
import pytest

@pytest.fixture
def sample_asset():
    return {"isin": "US0378331005", "name": "Apple Inc", "yf_ticker": "AAPL",
            "type": "stock", "sector": "Technology", "region": "US"}

@pytest.fixture
def sample_signals():
    return {
        "macro": {"regime": "growth", "sector_tailwinds": ["Technology"], "sector_headwinds": ["Energy"]},
        "price_data": {"AAPL": {"return_1y": 25.0, "volatility_annualized": 22.0, "pct_from_52w_high": -3.0}},
        "fundamentals": {"AAPL": {"pe_ratio": 28.0, "return_on_equity": 35.0, "revenue_growth_yoy": 8.0}},
        "earnings": {"AAPL": {"beat_probability": 0.60, "next_earnings_date": "2026-07-31"}},
        "insider": {"AAPL": {"net_buy_pct_mktcap": 0.001}},
        "options": {"AAPL": {"put_call_ratio": 0.75}},
        "short_interest": {"AAPL": {"short_float_pct": 0.8}},
        "wsb": {"ticker_mentions": {"AAPL": {"mentions_7d": 45, "squeeze_flag": False}}},
        "events": [],
        "universe_map": {"AAPL": {"sector": "Technology", "region": "US", "type": "stock"}},
        "price_stats": {"avg_return_1y": 15.0},
    }

@pytest.fixture
def sample_events():
    return [
        {"id": "A", "description": "Ukraine ceasefire within 30 days", "probability": 0.31,
         "complement_probability": 0.69, "source": "polymarket", "resolution_date": "2026-06-15",
         "asset_impacts": [{"sector": "Defense", "direction": "negative", "magnitude": "strong"}]},
        {"id": "B", "description": "Iran-Israel conflict escalates", "probability": 0.44,
         "complement_probability": 0.56, "source": "polymarket", "resolution_date": "2026-06-01",
         "asset_impacts": [
             {"sector": "Energy", "direction": "positive", "magnitude": "strong"},
             {"sector": "Airlines", "direction": "negative", "magnitude": "strong"},
         ]},
    ]
```

- [ ] **Step 4: Create tests/__init__.py**

```python
# tests/__init__.py
```

- [ ] **Step 5: Verify pytest collects no errors**

```bash
pytest tests/ -v --collect-only
```

Expected: `no tests ran`, no import errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "feat: test infrastructure and updated dependencies"
```

---

## Task 2: Universe Manager — Build Full Universe

**Files:**
- Create: `tools/universe_manager.py`
- Create: `tests/test_universe_manager.py`
- Create: `data/universe.csv` (via script)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_universe_manager.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from tools.universe_manager import (
    download_sp500_tickers, fetch_seed_etfs, build_universe,
    validate_ticker, refresh_universe
)

def test_download_sp500_returns_list():
    result = download_sp500_tickers()
    assert isinstance(result, list)
    assert len(result) > 400
    first = result[0]
    assert "yf_ticker" in first
    assert "sector" in first
    assert first["type"] == "stock"
    assert first["region"] == "US"

def test_fetch_seed_etfs_returns_list():
    result = fetch_seed_etfs()
    assert isinstance(result, list)
    assert len(result) > 20
    first = result[0]
    assert first["type"] == "etf"

def test_validate_ticker_valid():
    with patch("tools.universe_manager.yf") as mock_yf:
        mock_info = MagicMock()
        mock_info.last_price = 150.0
        mock_yf.Ticker.return_value.fast_info = mock_info
        assert validate_ticker("AAPL") is True

def test_validate_ticker_invalid():
    with patch("tools.universe_manager.yf") as mock_yf:
        mock_info = MagicMock()
        mock_info.last_price = None
        mock_yf.Ticker.return_value.fast_info = mock_info
        assert validate_ticker("INVALID123") is False

def test_build_universe_deduplicates():
    stocks = [{"yf_ticker": "AAPL", "name": "Apple", "isin": "", "type": "stock", "sector": "Tech", "region": "US"}]
    etfs = [{"yf_ticker": "AAPL", "name": "Apple ETF", "isin": "", "type": "etf", "sector": "Blend", "region": "US"}]
    df = build_universe(stocks, etfs)
    # AAPL should appear only once (dedup on yf_ticker)
    assert len(df[df["yf_ticker"] == "AAPL"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_universe_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.universe_manager'`

- [ ] **Step 3: Implement universe_manager.py**

```python
# tools/universe_manager.py
"""Full TR asset universe: build, validate, and refresh weekly."""

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

UNIVERSE_PATH = "data/universe.csv"

SEED_EUROPEAN_STOCKS = [
    {"isin": "DE0007164600", "name": "SAP SE", "yf_ticker": "SAP.DE", "type": "stock", "sector": "Technology", "region": "DE"},
    {"isin": "NL0010273215", "name": "ASML Holding", "yf_ticker": "ASML.AS", "type": "stock", "sector": "Technology", "region": "NL"},
    {"isin": "FR0000131104", "name": "BNP Paribas", "yf_ticker": "BNP.PA", "type": "stock", "sector": "Financials", "region": "FR"},
    {"isin": "FR0000120321", "name": "L'Oreal", "yf_ticker": "OR.PA", "type": "stock", "sector": "Consumer Staples", "region": "FR"},
    {"isin": "DE0005140008", "name": "Deutsche Bank", "yf_ticker": "DBK.DE", "type": "stock", "sector": "Financials", "region": "DE"},
    {"isin": "DE0007100000", "name": "Mercedes-Benz", "yf_ticker": "MBG.DE", "type": "stock", "sector": "Consumer Discretionary", "region": "DE"},
    {"isin": "DE0005552004", "name": "Deutsche Telekom", "yf_ticker": "DTE.DE", "type": "stock", "sector": "Communication Services", "region": "DE"},
    {"isin": "FR0000121014", "name": "LVMH", "yf_ticker": "MC.PA", "type": "stock", "sector": "Consumer Discretionary", "region": "FR"},
    {"isin": "NL0000009165", "name": "Heineken", "yf_ticker": "HEIA.AS", "type": "stock", "sector": "Consumer Staples", "region": "NL"},
    {"isin": "GB0002374006", "name": "Diageo", "yf_ticker": "DGE.L", "type": "stock", "sector": "Consumer Staples", "region": "GB"},
    {"isin": "GB00B10RZP78", "name": "Unilever", "yf_ticker": "ULVR.L", "type": "stock", "sector": "Consumer Staples", "region": "GB"},
    {"isin": "GB0031348658", "name": "HSBC Holdings", "yf_ticker": "HSBA.L", "type": "stock", "sector": "Financials", "region": "GB"},
    {"isin": "GB0007188757", "name": "GSK", "yf_ticker": "GSK.L", "type": "stock", "sector": "Healthcare", "region": "GB"},
    {"isin": "CH0012221716", "name": "ABB Ltd", "yf_ticker": "ABBN.SW", "type": "stock", "sector": "Industrials", "region": "CH"},
    {"isin": "DE0005190003", "name": "BMW AG", "yf_ticker": "BMW.DE", "type": "stock", "sector": "Consumer Discretionary", "region": "DE"},
    {"isin": "DE0007037129", "name": "Rheinmetall", "yf_ticker": "RHM.DE", "type": "stock", "sector": "Industrials", "region": "DE"},
    {"isin": "SE0000115446", "name": "Ericsson", "yf_ticker": "ERIC-B.ST", "type": "stock", "sector": "Technology", "region": "SE"},
    {"isin": "FI0009000681", "name": "Nokia", "yf_ticker": "NOKIA.HE", "type": "stock", "sector": "Technology", "region": "FI"},
    {"isin": "ES0113211835", "name": "Banco Santander", "yf_ticker": "SAN.MC", "type": "stock", "sector": "Financials", "region": "ES"},
    {"isin": "IT0003128367", "name": "Enel", "yf_ticker": "ENEL.MI", "type": "stock", "sector": "Utilities", "region": "IT"},
    {"isin": "DE0006231004", "name": "Infineon Technologies", "yf_ticker": "IFX.DE", "type": "stock", "sector": "Technology", "region": "DE"},
    {"isin": "NL0009434992", "name": "Airbus SE", "yf_ticker": "AIR.PA", "type": "stock", "sector": "Industrials", "region": "FR"},
    {"isin": "GB00BH4HKS39", "name": "Ryanair", "yf_ticker": "RYA.L", "type": "stock", "sector": "Industrials", "region": "IE"},
    {"isin": "DE0008232125", "name": "Deutsche Lufthansa", "yf_ticker": "LHA.DE", "type": "stock", "sector": "Industrials", "region": "DE"},
]

SEED_ETFS = [
    {"isin": "IE00B4L5Y983", "name": "iShares Core MSCI World", "yf_ticker": "IWDA.AS", "type": "etf", "sector": "Blend", "region": "Global"},
    {"isin": "IE00B3XXRP09", "name": "Vanguard S&P 500", "yf_ticker": "VUSA.AS", "type": "etf", "sector": "Blend", "region": "US"},
    {"isin": "IE00B52MJY50", "name": "iShares Core MSCI EM", "yf_ticker": "IEMG", "type": "etf", "sector": "Blend", "region": "EM"},
    {"isin": "IE00BKM4GZ66", "name": "iShares Core MSCI Europe", "yf_ticker": "IEUA.AS", "type": "etf", "sector": "Blend", "region": "EU"},
    {"isin": "IE00B5BMR087", "name": "iShares Core S&P 500", "yf_ticker": "CSPX.L", "type": "etf", "sector": "Blend", "region": "US"},
    {"isin": "IE00B3RBWM25", "name": "Vanguard FTSE All-World", "yf_ticker": "VWRL.L", "type": "etf", "sector": "Blend", "region": "Global"},
    {"isin": "IE00B4L5Y983", "name": "iShares MSCI Germany", "yf_ticker": "EWG", "type": "etf", "sector": "Blend", "region": "DE"},
    {"isin": "US4642874659", "name": "iShares MSCI South Korea", "yf_ticker": "EWY", "type": "etf", "sector": "Blend", "region": "KR"},
    {"isin": "US4642873859", "name": "iShares MSCI Japan", "yf_ticker": "EWJ", "type": "etf", "sector": "Blend", "region": "JP"},
    {"isin": "US4642872349", "name": "iShares MSCI China", "yf_ticker": "MCHI", "type": "etf", "sector": "Blend", "region": "CN"},
    {"isin": "US46434G1031", "name": "iShares MSCI Brazil", "yf_ticker": "EWZ", "type": "etf", "sector": "Blend", "region": "BR"},
    {"isin": "US78378X2036", "name": "S&P Global Clean Energy ETF", "yf_ticker": "ICLN", "type": "etf", "sector": "Energy", "region": "Global"},
    {"isin": "US46432F3428", "name": "iShares Global Defense & Aerospace", "yf_ticker": "ITA", "type": "etf", "sector": "Industrials", "region": "US"},
    {"isin": "US46090E5030", "name": "iShares Semiconductor ETF", "yf_ticker": "SOXX", "type": "etf", "sector": "Technology", "region": "US"},
    {"isin": "US46137V4085", "name": "iShares Robotics and AI ETF", "yf_ticker": "IRBO", "type": "etf", "sector": "Technology", "region": "Global"},
    {"isin": "US33740Q1085", "name": "First Trust NASDAQ Cybersecurity", "yf_ticker": "CIBR", "type": "etf", "sector": "Technology", "region": "US"},
    {"isin": "US00214Q1040", "name": "ARK Innovation ETF", "yf_ticker": "ARKK", "type": "etf", "sector": "Technology", "region": "US"},
    {"isin": "US78462F1030", "name": "SPDR Gold Shares", "yf_ticker": "GLD", "type": "etf", "sector": "Commodities", "region": "Global"},
    {"isin": "US9229087690", "name": "Vanguard FTSE Europe ETF", "yf_ticker": "VGK", "type": "etf", "sector": "Blend", "region": "EU"},
    {"isin": "US46434G8473", "name": "iShares MSCI India ETF", "yf_ticker": "INDA", "type": "etf", "sector": "Blend", "region": "IN"},
]


def download_sp500_tickers() -> list[dict]:
    """Download S&P 500 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, header=0)
    df = tables[0]
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Symbol", "")).replace(".", "-")
        result.append({
            "isin": str(row.get("ISIN", "")),
            "name": str(row.get("Security", "")),
            "yf_ticker": ticker,
            "type": "stock",
            "sector": str(row.get("GICS Sector", "Unknown")),
            "region": "US",
            "added_date": datetime.now().isoformat(),
        })
    return result


def fetch_seed_etfs() -> list[dict]:
    """Return curated list of ETFs available on Trade Republic."""
    result = []
    for etf in SEED_ETFS:
        result.append({**etf, "added_date": datetime.now().isoformat()})
    return result


def fetch_seed_european_stocks() -> list[dict]:
    """Return curated list of European stocks available on TR."""
    return [{**s, "added_date": datetime.now().isoformat()} for s in SEED_EUROPEAN_STOCKS]


def validate_ticker(ticker: str) -> bool:
    """Return True if ticker has a valid live price in yfinance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None)
            return bool(price and price > 0)
        except Exception:
            return False


def build_universe(stocks: list[dict], etfs: list[dict]) -> pd.DataFrame:
    """Merge stocks and ETFs, deduplicate on yf_ticker."""
    all_assets = stocks + etfs
    df = pd.DataFrame(all_assets)
    df = df.drop_duplicates(subset=["yf_ticker"], keep="first")
    df = df.reset_index(drop=True)
    return df


def refresh_universe(universe_path: str = UNIVERSE_PATH, validate: bool = True) -> pd.DataFrame:
    """Weekly refresh: validate existing tickers, add new candidates."""
    Path("data").mkdir(exist_ok=True)

    try:
        existing = pd.read_csv(universe_path)
        print(f"[universe] Loaded {len(existing)} existing assets")
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["isin", "name", "yf_ticker", "type", "sector", "region", "added_date"])
        print("[universe] No existing universe found, building from scratch")

    if validate and not existing.empty:
        print("[universe] Validating existing tickers...")
        valid_mask = existing["yf_ticker"].apply(validate_ticker)
        removed_count = (~valid_mask).sum()
        existing = existing[valid_mask].copy()
        print(f"[universe] Removed {removed_count} invalid tickers")

    candidates = (
        download_sp500_tickers() +
        fetch_seed_european_stocks() +
        fetch_seed_etfs()
    )
    candidate_df = pd.DataFrame(candidates)
    existing_tickers = set(existing["yf_ticker"].tolist())
    new_candidates = candidate_df[~candidate_df["yf_ticker"].isin(existing_tickers)]

    if validate and not new_candidates.empty:
        print(f"[universe] Validating {len(new_candidates)} new candidates...")
        valid_new_mask = new_candidates["yf_ticker"].apply(validate_ticker)
        new_valid = new_candidates[valid_new_mask].copy()
    else:
        new_valid = new_candidates.copy()

    universe = pd.concat([existing, new_valid], ignore_index=True)
    universe.to_csv(universe_path, index=False)
    print(f"[universe] Universe: {len(universe)} total assets (+{len(new_valid)} added)")
    return universe


def load_universe(universe_path: str = UNIVERSE_PATH) -> list[dict]:
    """Load universe CSV and return list of dicts."""
    df = pd.read_csv(universe_path)
    return df.to_dict(orient="records")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_universe_manager.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Build initial universe (without full validation to save time)**

```bash
python -c "from tools.universe_manager import refresh_universe; refresh_universe(validate=False)"
```

Expected: `data/universe.csv` created with 500+ assets.

- [ ] **Step 6: Commit**

```bash
git add tools/universe_manager.py tests/test_universe_manager.py data/universe.csv
git commit -m "feat: universe manager with S&P 500 + European stocks + ETFs"
```

---

## Task 3: Commodity Futures Collector

**Files:**
- Create: `tools/futures_tools.py`
- Create: `tests/test_futures_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_futures_tools.py
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from tools.futures_tools import fetch_futures_signal, FUTURES_TICKERS

def test_futures_tickers_defined():
    assert "CL=F" in FUTURES_TICKERS
    assert "GC=F" in FUTURES_TICKERS
    assert "NG=F" in FUTURES_TICKERS
    assert "HG=F" in FUTURES_TICKERS

def test_fetch_futures_signal_structure():
    with patch("tools.futures_tools.yf") as mock_yf:
        closes = pd.Series([100.0, 105.0, 110.0, 108.0, 112.0] * 20)
        mock_hist = MagicMock()
        mock_hist.__getitem__ = lambda self, key: closes
        mock_yf.Ticker.return_value.history.return_value = mock_hist

        result = fetch_futures_signal()
        assert isinstance(result, dict)
        assert "sector_tailwinds" in result
        assert "sector_headwinds" in result
        assert isinstance(result["sector_tailwinds"], list)

def test_fetch_futures_signal_oil_rising_favors_energy():
    with patch("tools.futures_tools.yf") as mock_yf:
        def make_series(start, end):
            import numpy as np
            return pd.Series(list(np.linspace(start, end, 100)))

        futures_data = {
            "CL=F": make_series(70, 90),  # oil rising +29%
            "GC=F": make_series(1900, 1950),  # gold slight up
            "NG=F": make_series(3.0, 3.1),
            "HG=F": make_series(4.0, 4.0),
        }

        def ticker_side_effect(sym):
            m = MagicMock()
            series = futures_data.get(sym, make_series(100, 100))
            hist = MagicMock()
            hist.__getitem__ = lambda s, k: series
            m.history.return_value = hist
            return m

        mock_yf.Ticker.side_effect = ticker_side_effect
        result = fetch_futures_signal()
        assert "Energy" in result["sector_tailwinds"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_futures_tools.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# tools/futures_tools.py
"""Commodity futures signal via yfinance. CL=F crude, GC=F gold, NG=F gas, HG=F copper."""

import warnings
import yfinance as yf
import numpy as np

FUTURES_TICKERS = {
    "CL=F": {"name": "Crude Oil", "sector_positive": ["Energy"], "sector_negative": ["Airlines", "Consumer Discretionary"]},
    "GC=F": {"name": "Gold", "sector_positive": ["Materials"], "sector_negative": []},
    "NG=F": {"name": "Natural Gas", "sector_positive": ["Energy", "Utilities"], "sector_negative": ["Industrials"]},
    "HG=F": {"name": "Copper", "sector_positive": ["Materials", "Industrials"], "sector_negative": []},
}


def fetch_futures_signal(period: str = "3mo") -> dict:
    """Fetch commodity futures trends and derive sector tailwinds/headwinds."""
    futures_data = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker, meta in FUTURES_TICKERS.items():
            try:
                hist = yf.Ticker(ticker).history(period=period)["Close"].dropna()
                if len(hist) < 10:
                    continue
                ret_1m = float((hist.iloc[-1] / hist.iloc[-min(21, len(hist))] - 1) * 100)
                ret_3m = float((hist.iloc[-1] / hist.iloc[0] - 1) * 100)
                trend = "rising" if ret_1m > 3 else ("falling" if ret_1m < -3 else "flat")
                futures_data[ticker] = {
                    "name": meta["name"],
                    "price": round(float(hist.iloc[-1]), 2),
                    "return_1m_pct": round(ret_1m, 2),
                    "return_3m_pct": round(ret_3m, 2),
                    "trend": trend,
                    "sector_positive": meta["sector_positive"],
                    "sector_negative": meta["sector_negative"],
                }
            except Exception as e:
                futures_data[ticker] = {"name": meta["name"], "error": str(e)}

    tailwinds = set()
    headwinds = set()
    commodity_summary = []

    for ticker, data in futures_data.items():
        if "error" in data:
            continue
        trend = data["trend"]
        if trend == "rising":
            tailwinds.update(data.get("sector_positive", []))
            headwinds.update(data.get("sector_negative", []))
        elif trend == "falling":
            tailwinds.update(data.get("sector_negative", []))
            headwinds.update(data.get("sector_positive", []))
        commodity_summary.append(f"{data['name']}: {data['return_1m_pct']:+.1f}% 1M ({trend})")

    return {
        "futures": futures_data,
        "sector_tailwinds": list(tailwinds - headwinds),
        "sector_headwinds": list(headwinds - tailwinds),
        "summary": "; ".join(commodity_summary),
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_futures_tools.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/futures_tools.py tests/test_futures_tools.py
git commit -m "feat: commodity futures signal collector"
```

---

## Task 4: Polymarket Collector

**Files:**
- Create: `tools/polymarket_tools.py`
- Create: `tests/test_polymarket_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_polymarket_tools.py
from unittest.mock import patch
import pytest
from tools.polymarket_tools import (
    fetch_geopolitical_markets, fetch_earnings_markets,
    parse_probability, filter_by_volume
)

def test_parse_probability_from_outcome_prices():
    market = {"outcomePrices": ["0.67", "0.33"]}
    assert parse_probability(market) == pytest.approx(0.67, abs=0.01)

def test_parse_probability_missing_returns_half():
    market = {}
    assert parse_probability(market) == 0.5

def test_filter_by_volume_removes_low_volume():
    markets = [
        {"volume": "500", "question": "Will X happen?"},
        {"volume": "5000", "question": "Will Y happen?"},
        {"volume": "0", "question": "Will Z happen?"},
    ]
    result = filter_by_volume(markets, min_volume=1000)
    assert len(result) == 1
    assert result[0]["question"] == "Will Y happen?"

def test_fetch_geopolitical_markets_returns_list():
    mock_response = [
        {"id": "1", "question": "Will Ukraine war end?", "outcomePrices": ["0.31", "0.69"],
         "volume": "50000", "endDate": "2026-12-31", "category": "politics"},
    ]
    with patch("tools.polymarket_tools.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None
        result = fetch_geopolitical_markets()
        assert isinstance(result, list)
        # Should parse probability from outcomePrices
        if result:
            assert 0.0 <= result[0]["probability"] <= 1.0
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/test_polymarket_tools.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# tools/polymarket_tools.py
"""Polymarket Gamma API — extract prediction market probabilities."""

import requests

POLYMARKET_BASE = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

GEO_KEYWORDS = ["war", "attack", "invasion", "ceasefire", "peace", "sanctions",
                 "nuclear", "conflict", "military", "troops", "strike", "embargo",
                 "election", "coup", "treaty", "referendum", "crisis"]

EARNINGS_KEYWORDS = ["earnings", "revenue", "beat", "miss", "guidance",
                      "quarterly", "profit", "eps", "results"]


def parse_probability(market: dict) -> float:
    """Extract Yes probability from outcomePrices array."""
    prices = market.get("outcomePrices", [])
    if prices:
        try:
            return round(float(prices[0]), 4)
        except (ValueError, TypeError):
            pass
    return 0.5


def filter_by_volume(markets: list[dict], min_volume: float = 1000.0) -> list[dict]:
    """Keep only markets with sufficient trading volume (liquidity signal)."""
    result = []
    for m in markets:
        vol = float(m.get("volume", 0) or 0)
        if vol >= min_volume:
            result.append(m)
    return result


def _fetch_markets(params: dict) -> list[dict]:
    try:
        resp = requests.get(f"{POLYMARKET_BASE}/markets", params=params,
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else []
    except Exception:
        return []


def fetch_geopolitical_markets(min_volume: float = 2000.0) -> list[dict]:
    """Fetch active geopolitical prediction markets."""
    markets = _fetch_markets({"active": "true", "limit": "200"})
    geo = []
    for m in markets:
        q = m.get("question", "").lower()
        if any(k in q for k in GEO_KEYWORDS):
            geo.append({
                "id": m.get("id", ""),
                "question": m.get("question", ""),
                "probability": parse_probability(m),
                "volume": float(m.get("volume", 0) or 0),
                "end_date": m.get("endDate", ""),
                "source": "polymarket",
            })
    return filter_by_volume(geo, min_volume)


def fetch_earnings_markets(min_volume: float = 1000.0) -> list[dict]:
    """Fetch active earnings beat/miss prediction markets."""
    markets = _fetch_markets({"active": "true", "limit": "200"})
    earnings = []
    for m in markets:
        q = m.get("question", "").lower()
        if any(k in q for k in EARNINGS_KEYWORDS):
            earnings.append({
                "id": m.get("id", ""),
                "question": m.get("question", ""),
                "probability": parse_probability(m),
                "volume": float(m.get("volume", 0) or 0),
                "end_date": m.get("endDate", ""),
                "source": "polymarket",
            })
    return filter_by_volume(earnings, min_volume)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_polymarket_tools.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/polymarket_tools.py tests/test_polymarket_tools.py
git commit -m "feat: Polymarket prediction market collector"
```

---

## Task 5: GDELT Geopolitical Collector

**Files:**
- Create: `tools/gdelt_tools.py`

- [ ] **Step 1: Implement (GDELT API is brittle; integration test only)**

```python
# tools/gdelt_tools.py
"""GDELT API v2 — country conflict indices and recent event volumes."""

import requests
from datetime import datetime, timedelta

GDELT_TIMELINE = "https://api.gdeltproject.org/api/v2/timeline/timeline"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

MONITORED_REGIONS = [
    {"name": "Middle East", "query": "(Iran OR Israel OR Gaza OR Lebanon OR Syria OR Yemen) (war OR conflict OR attack OR military OR strike)"},
    {"name": "Eastern Europe", "query": "(Ukraine OR Russia OR NATO OR Donbas) (war OR attack OR offensive OR ceasefire OR troops)"},
    {"name": "Asia Pacific", "query": "(Taiwan OR China OR North Korea) (military OR strait OR invasion OR missile OR tension)"},
    {"name": "Sub-Saharan Africa", "query": "(Sudan OR Congo OR Mali OR Sahel) (coup OR conflict OR military OR civil war)"},
]


def _fetch_gdelt_volume(query: str, days: int = 14) -> list[float]:
    """Return daily event volume for query over last N days."""
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")
    end = datetime.now().strftime("%Y%m%d%H%M%S")
    params = {
        "query": query,
        "mode": "timelinevol",
        "format": "json",
        "STARTDATETIME": start,
        "ENDDATETIME": end,
        "SMOOTHING": "3",
    }
    try:
        resp = requests.get(GDELT_TIMELINE, params=params, headers=HEADERS, timeout=30)
        data = resp.json()
        timeline = data.get("timeline", [{}])[0].get("data", [])
        return [float(p.get("value", 0)) for p in timeline]
    except Exception:
        return []


def fetch_regional_conflict_indices() -> list[dict]:
    """Fetch conflict event volume for each monitored region and compute risk score + trend."""
    results = []
    for region in MONITORED_REGIONS:
        volumes = _fetch_gdelt_volume(region["query"])
        if not volumes or len(volumes) < 4:
            results.append({
                "region": region["name"],
                "conflict_score": 0.4,
                "trend": "unknown",
                "source": "gdelt",
            })
            continue

        recent = volumes[-7:] if len(volumes) >= 7 else volumes
        prev = volumes[-14:-7] if len(volumes) >= 14 else volumes[:len(volumes)//2]
        recent_avg = sum(recent) / len(recent)
        prev_avg = sum(prev) / len(prev) if prev else recent_avg

        conflict_score = min(1.0, recent_avg / 100.0)
        if recent_avg > prev_avg * 1.25:
            trend = "escalating"
        elif recent_avg < prev_avg * 0.75:
            trend = "de-escalating"
        else:
            trend = "stable"

        results.append({
            "region": region["name"],
            "conflict_score": round(conflict_score, 3),
            "trend": trend,
            "recent_avg_volume": round(recent_avg, 2),
            "source": "gdelt",
        })
    return results
```

- [ ] **Step 2: Quick smoke test**

```bash
python -c "from tools.gdelt_tools import fetch_regional_conflict_indices; r = fetch_regional_conflict_indices(); print(r[0])"
```

Expected: prints a dict with `region`, `conflict_score`, `trend` — values may be 0.4/unknown if GDELT is slow.

- [ ] **Step 3: Commit**

```bash
git add tools/gdelt_tools.py
git commit -m "feat: GDELT regional conflict index collector"
```

---

## Task 6: WSB Trend Collector

**Files:**
- Create: `tools/wsb_tools.py`
- Create: `tests/test_wsb_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_wsb_tools.py
from tools.wsb_tools import (
    extract_tickers_from_text, analyze_wsb_signals, SECTOR_KEYWORDS
)

def test_extract_tickers_dollar_sign():
    text = "Just bought $NVDA and $AAPL calls, not financial advice"
    tickers = extract_tickers_from_text(text)
    assert "NVDA" in tickers
    assert "AAPL" in tickers

def test_extract_tickers_excludes_common_words():
    text = "$I $AM $NOT $A ticker but $GME is"
    tickers = extract_tickers_from_text(text)
    assert "GME" in tickers
    assert "I" not in tickers
    assert "AM" not in tickers
    assert "NOT" not in tickers

def test_analyze_wsb_signals_structure():
    posts = [
        {"title": "$GME to the moon! Short squeeze incoming", "score": 1000, "text": "SI is 25%, gamma squeeze setup"},
        {"title": "Semiconductor stocks pumping, $NVDA $AMD on fire", "score": 500, "text": "AI demand driving chip stocks"},
        {"title": "$AAPL earnings play", "score": 200, "text": "Buying $AAPL calls"},
    ]
    result = analyze_wsb_signals(posts)
    assert "trending_tickers" in result
    assert "sector_hype" in result
    assert "squeeze_candidates" in result
    # GME should be trending with squeeze flag
    gme_entry = next((t for t in result["trending_tickers"] if t["ticker"] == "GME"), None)
    assert gme_entry is not None
    assert gme_entry["squeeze_flag"] is True

def test_sector_keywords_defined():
    assert "Semiconductors" in SECTOR_KEYWORDS
    assert "AI" in SECTOR_KEYWORDS
    assert "Defense" in SECTOR_KEYWORDS
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/test_wsb_tools.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# tools/wsb_tools.py
"""Reddit r/wallstreetbets public JSON — ticker mentions, squeeze detection, sector hype."""

import re
from collections import Counter
import requests

WSB_URLS = [
    "https://www.reddit.com/r/wallstreetbets/hot.json?limit=100",
    "https://www.reddit.com/r/wallstreetbets/top.json?t=week&limit=100",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

SQUEEZE_KEYWORDS = ["squeeze", "short interest", "gamma squeeze", "short float",
                     "days to cover", "si%", "ftd", "fail to deliver"]

SECTOR_KEYWORDS = {
    "Semiconductors": ["semiconductor", "chip", "foundry", "wafer", "tsmc", "amd", "nvidia", "intel", "qualcomm", "asml"],
    "Photonics": ["photonics", "laser", "lidar", "optical", "fiber optic", "photonic"],
    "Robotics": ["robot", "robotics", "automation", "autonomous", "humanoid"],
    "AI": ["artificial intelligence", "llm", "machine learning", "deep learning", "neural network", "ai stocks"],
    "Defense": ["defense", "military", "nato", "pentagon", "lockheed", "raytheon", "missile", "wartime"],
    "Biotech": ["biotech", "fda", "clinical trial", "drug approval", "biopharma", "oncology"],
    "Energy": ["oil", "energy stocks", "crude", "lng", "solar", "renewables", "drilling"],
    "Crypto": ["bitcoin", "btc", "ethereum", "defi", "crypto", "blockchain", "coinbase"],
}

EXCLUDE_TOKENS = {
    "A", "I", "AM", "PM", "US", "UK", "EU", "DD", "OP", "YOY", "IMO",
    "EOD", "CEO", "ETF", "EPS", "FYI", "SEC", "IPO", "ATH", "ATL",
    "TA", "TD", "IV", "OI", "WSB", "SPY", "SPX", "VIX", "GDP", "CPI",
}


def extract_tickers_from_text(text: str) -> list[str]:
    """Extract $TICKER patterns from text, filtering common false positives."""
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    return [t for t in dollar_tickers if t not in EXCLUDE_TOKENS]


def fetch_wsb_posts() -> list[dict]:
    """Fetch recent posts from r/wallstreetbets via public JSON endpoint."""
    posts = []
    for url in WSB_URLS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                p = child.get("data", {})
                posts.append({
                    "title": p.get("title", ""),
                    "score": int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "text": p.get("selftext", ""),
                })
        except Exception:
            pass
    return posts


def analyze_wsb_signals(posts: list[dict]) -> dict:
    """Extract trending tickers, squeeze candidates, and sector hype from posts."""
    all_tickers = []
    squeeze_posts: dict[str, list[str]] = {}

    for post in posts:
        full_text = post["title"] + " " + post.get("text", "")
        tickers = extract_tickers_from_text(full_text)
        all_tickers.extend(tickers)

        text_lower = full_text.lower()
        if any(k in text_lower for k in SQUEEZE_KEYWORDS):
            for t in tickers:
                if t not in squeeze_posts:
                    squeeze_posts[t] = []
                squeeze_posts[t].append(post["title"])

    ticker_counts = Counter(all_tickers)
    squeeze_candidates = list(squeeze_posts.keys())

    all_text_lower = " ".join(p["title"].lower() + " " + p.get("text", "").lower() for p in posts)
    sector_scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        count = sum(all_text_lower.count(k) for k in keywords)
        score = min(1.0, count / 30.0)
        if score > 0.05:
            sector_scores[sector] = round(score, 3)

    trending = [
        {
            "ticker": ticker,
            "mentions_7d": count,
            "squeeze_flag": ticker in squeeze_candidates,
        }
        for ticker, count in ticker_counts.most_common(30)
    ]

    return {
        "trending_tickers": trending,
        "sector_hype": [
            {"sector": s, "score": score}
            for s, score in sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        ],
        "squeeze_candidates": squeeze_candidates[:10],
        "total_posts_analyzed": len(posts),
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_wsb_tools.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/wsb_tools.py tests/test_wsb_tools.py
git commit -m "feat: WSB Reddit trend and squeeze signal collector"
```

---

## Task 7: BTC Signal Collector

**Files:**
- Create: `tools/btc_tools.py`
- Create: `tests/test_btc_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_btc_tools.py
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from tools.btc_tools import fetch_btc_signal, classify_btc_regime

def make_price_series(start, end, n=90):
    return pd.Series(list(np.linspace(start, end, n)))

def test_classify_btc_regime_risk_on():
    # BTC outperforms gold, both rising
    regime = classify_btc_regime(btc_4w=15.0, gold_4w=5.0)
    assert regime == "risk_on"

def test_classify_btc_regime_safe_haven():
    # Gold outperforms BTC
    regime = classify_btc_regime(btc_4w=2.0, gold_4w=8.0)
    assert regime == "safe_haven"

def test_classify_btc_regime_risk_off():
    # BTC falling sharply
    regime = classify_btc_regime(btc_4w=-15.0, gold_4w=3.0)
    assert regime == "risk_off"

def test_classify_btc_regime_neutral():
    regime = classify_btc_regime(btc_4w=1.0, gold_4w=1.5)
    assert regime == "neutral"

def test_fetch_btc_signal_structure():
    with patch("tools.btc_tools.yf") as mock_yf:
        def ticker_side(sym):
            m = MagicMock()
            closes = make_price_series(50000 if "BTC" in sym else 1900, 55000 if "BTC" in sym else 2000)
            hist = MagicMock()
            hist.__getitem__ = lambda s, k: closes
            hist.dropna.return_value = closes
            m.history.return_value = hist
            return m
        mock_yf.Ticker.side_effect = ticker_side

        result = fetch_btc_signal()
        assert "btc_price" in result
        assert "regime" in result
        assert "liquidity_signal" in result
        assert result["regime"] in ("risk_on", "safe_haven", "risk_off", "neutral")
```

- [ ] **Step 2: Run failing**

```bash
pytest tests/test_btc_tools.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# tools/btc_tools.py
"""BTC as macro signal: liquidity proxy and anti-establishment sentiment indicator."""

import warnings
import yfinance as yf


def classify_btc_regime(btc_4w: float, gold_4w: float) -> str:
    """Classify BTC macro regime from 4-week returns."""
    if btc_4w < -10:
        return "risk_off"
    if btc_4w > gold_4w + 5 and btc_4w > 0:
        return "risk_on"
    if gold_4w > btc_4w + 5 and gold_4w > 3:
        return "safe_haven"
    return "neutral"


def fetch_btc_signal(period: str = "3mo") -> dict:
    """Fetch BTC and gold price trends, classify regime, and derive asset impacts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            btc_hist = yf.Ticker("BTC-USD").history(period=period)["Close"].dropna()
            gold_hist = yf.Ticker("GC=F").history(period=period)["Close"].dropna()
            spy_hist = yf.Ticker("SPY").history(period=period)["Close"].dropna()
        except Exception as e:
            return {"error": str(e), "regime": "neutral", "liquidity_signal": "neutral",
                    "btc_price": 0, "interpretation": "Data fetch failed"}

        def pct_change(series, n):
            if len(series) < n + 1:
                return 0.0
            return float((series.iloc[-1] / series.iloc[-n] - 1) * 100)

        btc_1w = pct_change(btc_hist, 5)
        btc_4w = pct_change(btc_hist, 20)
        gold_4w = pct_change(gold_hist, 20)
        spy_4w = pct_change(spy_hist, 20)

        regime = classify_btc_regime(btc_4w, gold_4w)

        regime_descriptions = {
            "risk_on": (
                "positive",
                f"BTC {btc_4w:+.1f}% 4W outperforming gold {gold_4w:+.1f}%. "
                "Institutional risk appetite strong. Favors growth/tech over defensives."
            ),
            "safe_haven": (
                "negative",
                f"Gold {gold_4w:+.1f}% outperforming BTC {btc_4w:+.1f}%. "
                "Safe haven rotation: institutional distrust elevated. Favors gold ETFs, utilities, defensives."
            ),
            "risk_off": (
                "negative",
                f"BTC {btc_4w:+.1f}% 4W. Sharp de-risking: liquidity contracting. "
                "Expect pressure on growth and speculative assets."
            ),
            "neutral": (
                "neutral",
                f"BTC {btc_4w:+.1f}% 4W, gold {gold_4w:+.1f}% 4W. No strong macro signal from crypto."
            ),
        }

        liquidity_signal, interpretation = regime_descriptions[regime]

        impact_map = {
            "risk_on": {"growth_tech": "tailwind", "financials": "tailwind", "defensives": "headwind", "gold_etfs": "headwind"},
            "safe_haven": {"growth_tech": "headwind", "gold_etfs": "tailwind", "utilities": "tailwind", "defensives": "tailwind"},
            "risk_off": {"growth_tech": "headwind", "speculative": "headwind", "defensives": "tailwind"},
            "neutral": {},
        }

        return {
            "btc_price": round(float(btc_hist.iloc[-1]), 2),
            "btc_1w_return_pct": round(btc_1w, 2),
            "btc_4w_return_pct": round(btc_4w, 2),
            "gold_4w_return_pct": round(gold_4w, 2),
            "spy_4w_return_pct": round(spy_4w, 2),
            "regime": regime,
            "liquidity_signal": liquidity_signal,
            "interpretation": interpretation,
            "asset_impacts": impact_map[regime],
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_btc_tools.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/btc_tools.py tests/test_btc_tools.py
git commit -m "feat: BTC liquidity and sentiment signal collector"
```

---

## Task 8: Remaining Collectors (Insider, Options, Short Interest, News)

**Files:**
- Create: `tools/insider_tools.py`
- Create: `tools/options_tools.py`
- Create: `tools/short_interest_tools.py`
- Create: `tools/news_tools.py`

- [ ] **Step 1: Implement insider_tools.py**

```python
# tools/insider_tools.py
"""SEC EDGAR Form 4 — net insider buying/selling per company."""

import requests
from datetime import datetime, timedelta

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
HEADERS = {"User-Agent": "investment-research-bot admin@example.com"}


def fetch_insider_transactions(tickers: list[str], days_back: int = 90) -> dict[str, dict]:
    """
    Fetch Form 4 filings for each ticker from SEC EDGAR.
    Returns dict: ticker → {net_shares, net_value_usd, transaction_count, signal}
    """
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    results = {}

    for ticker in tickers:
        try:
            params = {
                "q": f'"{ticker}"',
                "dateRange": "custom",
                "startdt": start,
                "forms": "4",
                "_source": "filing",
            }
            resp = requests.get(EDGAR_SEARCH, params=params, headers=HEADERS, timeout=20)
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            buy_count = sum(1 for h in hits if "purchase" in str(h).lower() or "P" in str(h))
            sell_count = sum(1 for h in hits if "sale" in str(h).lower() or "S" in str(h))
            total = len(hits)

            if total == 0:
                signal = "neutral"
                net_buy_pct = 0.0
            elif buy_count > sell_count * 2:
                signal = "strong_buy"
                net_buy_pct = 0.8
            elif buy_count > sell_count:
                signal = "buy"
                net_buy_pct = 0.4
            elif sell_count > buy_count * 2:
                signal = "strong_sell"
                net_buy_pct = -0.8
            elif sell_count > buy_count:
                signal = "sell"
                net_buy_pct = -0.4
            else:
                signal = "neutral"
                net_buy_pct = 0.0

            results[ticker] = {
                "buy_filings": buy_count,
                "sell_filings": sell_count,
                "total_filings": total,
                "net_buy_pct_mktcap": net_buy_pct,
                "signal": signal,
                "days_back": days_back,
            }
        except Exception as e:
            results[ticker] = {"net_buy_pct_mktcap": 0.0, "signal": "neutral", "error": str(e)}

    return results
```

- [ ] **Step 2: Implement options_tools.py**

```python
# tools/options_tools.py
"""yfinance options chain — put-call ratio and open interest skew per ticker."""

import warnings
import yfinance as yf


def fetch_options_signal(tickers: list[str]) -> dict[str, dict]:
    """
    Compute put-call ratio and OI skew for each ticker.
    Returns dict: ticker → {put_call_ratio, oi_skew, signal}
    PCR < 0.7 = bullish, PCR > 1.3 = bearish
    """
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                expirations = t.options
                if not expirations:
                    results[ticker] = {"put_call_ratio": 1.0, "signal": "neutral"}
                    continue

                # Use nearest expiration with meaningful volume
                expiry = expirations[0] if len(expirations) <= 2 else expirations[1]
                chain = t.option_chain(expiry)
                call_oi = chain.calls["openInterest"].sum()
                put_oi = chain.puts["openInterest"].sum()
                call_vol = chain.calls["volume"].sum()
                put_vol = chain.puts["volume"].sum()

                pcr_oi = float(put_oi / call_oi) if call_oi > 0 else 1.0
                pcr_vol = float(put_vol / call_vol) if call_vol > 0 else 1.0
                pcr = round((pcr_oi + pcr_vol) / 2, 3)

                if pcr < 0.6:
                    signal = "strong_bullish"
                elif pcr < 0.8:
                    signal = "bullish"
                elif pcr > 1.4:
                    signal = "strong_bearish"
                elif pcr > 1.1:
                    signal = "bearish"
                else:
                    signal = "neutral"

                results[ticker] = {
                    "put_call_ratio": pcr,
                    "call_oi": int(call_oi),
                    "put_oi": int(put_oi),
                    "signal": signal,
                }
            except Exception as e:
                results[ticker] = {"put_call_ratio": 1.0, "signal": "neutral", "error": str(e)}
    return results
```

- [ ] **Step 3: Implement short_interest_tools.py**

```python
# tools/short_interest_tools.py
"""Short interest data via yfinance — short float % and days to cover."""

import warnings
import yfinance as yf


def fetch_short_interest(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch short interest metrics for each ticker.
    Returns dict: ticker → {short_float_pct, days_to_cover, signal}
    """
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                short_float = info.get("shortPercentOfFloat") or 0.0
                short_float_pct = round(float(short_float) * 100, 2)
                shares_short = info.get("sharesShort") or 0
                avg_volume = info.get("averageVolume") or 1
                days_to_cover = round(shares_short / avg_volume, 1) if avg_volume else 0.0

                if short_float_pct > 25:
                    signal = "very_high_short"
                elif short_float_pct > 15:
                    signal = "high_short"
                elif short_float_pct > 8:
                    signal = "moderate_short"
                else:
                    signal = "low_short"

                results[ticker] = {
                    "short_float_pct": short_float_pct,
                    "days_to_cover": days_to_cover,
                    "signal": signal,
                }
            except Exception as e:
                results[ticker] = {"short_float_pct": 0.0, "days_to_cover": 0.0,
                                   "signal": "unknown", "error": str(e)}
    return results
```

- [ ] **Step 4: Implement news_tools.py**

```python
# tools/news_tools.py
"""RSS news feeds — sector and company headline scraping (Reuters, FT)."""

import feedparser

RSS_FEEDS = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_tech": "https://feeds.reuters.com/reuters/technologyNews",
    "ft_markets": "https://www.ft.com/rss/home/uk",
}


def fetch_news_headlines(max_per_feed: int = 20) -> dict:
    """Fetch recent headlines from RSS feeds. Returns categorized headlines."""
    all_headlines = []
    for feed_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                all_headlines.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:200],
                    "published": str(entry.get("published", "")),
                    "source": feed_name,
                })
        except Exception:
            pass
    return {
        "headlines": all_headlines,
        "total": len(all_headlines),
    }
```

- [ ] **Step 5: Commit**

```bash
git add tools/insider_tools.py tools/options_tools.py tools/short_interest_tools.py tools/news_tools.py
git commit -m "feat: insider flow, options, short interest, and news collectors"
```

---

## Task 9: Event Coherence Framework

**Files:**
- Create: `event_extractor.py`
- Create: `tests/test_event_coherence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_event_coherence.py
import pytest
from event_extractor import (
    check_event_coherence, CoherenceError,
    normalize_complement_probabilities, flag_dependent_events
)

def test_complement_is_computed():
    events = [{"id": "A", "description": "War ends", "probability": 0.31, "asset_impacts": []}]
    result = normalize_complement_probabilities(events)
    assert result[0]["complement_probability"] == pytest.approx(0.69, abs=0.001)

def test_invalid_probability_raises():
    events = [{"id": "A", "description": "X", "probability": 1.5, "asset_impacts": []}]
    with pytest.raises(CoherenceError, match="probability must be 0.0-1.0"):
        check_event_coherence(events)

def test_mutually_exclusive_sum_over_one_raises():
    events = [
        {"id": "A", "description": "War starts", "probability": 0.7,
         "mutually_exclusive_with": ["B"], "asset_impacts": []},
        {"id": "B", "description": "War ends", "probability": 0.8,
         "mutually_exclusive_with": ["A"], "asset_impacts": []},
    ]
    with pytest.raises(CoherenceError, match="mutually exclusive"):
        check_event_coherence(events)

def test_valid_events_pass():
    events = [
        {"id": "A", "description": "Iran escalates", "probability": 0.44,
         "mutually_exclusive_with": ["B"], "asset_impacts": []},
        {"id": "B", "description": "Iran ceasefire", "probability": 0.31,
         "mutually_exclusive_with": ["A"], "asset_impacts": []},
    ]
    result = check_event_coherence(events)
    assert len(result) == 2

def test_flag_dependent_events():
    events = [
        {"id": "A", "description": "Fed cuts rates", "probability": 0.67, "asset_impacts": []},
        {"id": "B", "description": "Tech stocks rally on rate cut", "probability": 0.55,
         "depends_on": "A", "asset_impacts": []},
    ]
    result = flag_dependent_events(events)
    assert result[1]["dependency_note"] == "Depends on Event A"
```

- [ ] **Step 2: Run failing**

```bash
pytest tests/test_event_coherence.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement event_extractor.py**

```python
# event_extractor.py
"""
Event probability framework.
Extracts probabilistic events from signals, enforces coherence,
and produces events.json as part of signals.json.

The actual event extraction (interpreting geo/macro data into events) is done
by the Claude rating session reading signals.json — this module handles
the structural validation and JSON schema.
"""

import json


class CoherenceError(ValueError):
    pass


def normalize_complement_probabilities(events: list[dict]) -> list[dict]:
    """Set complement_probability = 1 - probability for each event."""
    for event in events:
        p = event.get("probability", 0.5)
        event["complement_probability"] = round(1.0 - p, 4)
    return events


def check_event_coherence(events: list[dict]) -> list[dict]:
    """
    Validate event probabilities:
    1. Each probability must be in [0.0, 1.0]
    2. Mutually exclusive event pairs must sum <= 1.0
    Raises CoherenceError on violation.
    """
    for event in events:
        p = event.get("probability", 0.5)
        if not (0.0 <= p <= 1.0):
            raise CoherenceError(
                f"Event {event['id']}: probability must be 0.0-1.0, got {p}"
            )

    event_map = {e["id"]: e for e in events}
    checked_pairs = set()
    for event in events:
        for excl_id in event.get("mutually_exclusive_with", []):
            pair = tuple(sorted([event["id"], excl_id]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            if excl_id in event_map:
                p_sum = event["probability"] + event_map[excl_id]["probability"]
                if p_sum > 1.0 + 1e-6:
                    raise CoherenceError(
                        f"Events {event['id']} and {excl_id} are mutually exclusive "
                        f"but probabilities sum to {p_sum:.3f} > 1.0"
                    )

    return events


def flag_dependent_events(events: list[dict]) -> list[dict]:
    """Add dependency notes so the rating agent avoids double-counting."""
    for event in events:
        dep = event.get("depends_on")
        if dep:
            event["dependency_note"] = f"Depends on Event {dep}"
    return events


def validate_and_normalize(events: list[dict]) -> list[dict]:
    """Full pipeline: check coherence, normalize complements, flag dependencies."""
    events = check_event_coherence(events)
    events = normalize_complement_probabilities(events)
    events = flag_dependent_events(events)
    return events


def build_empty_event_template() -> dict:
    """Return a blank event dict for the Claude session to fill in."""
    return {
        "id": "",
        "description": "",
        "probability": 0.5,
        "complement_description": "",
        "complement_probability": 0.5,
        "source": "polymarket|gdelt|inferred",
        "resolution_date": "",
        "mutually_exclusive_with": [],
        "depends_on": None,
        "asset_impacts": [
            {
                "sector": "",
                "region": "",
                "direction": "positive|negative",
                "magnitude": "strong|moderate|weak",
                "causal_chain": "event → mechanism → asset impact",
            }
        ],
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_event_coherence.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add event_extractor.py tests/test_event_coherence.py
git commit -m "feat: probabilistic event framework with coherence validation"
```

---

## Task 10: Fast Scorer — Python Pass 1

**Files:**
- Create: `fast_scorer.py`
- Create: `tests/test_fast_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fast_scorer.py
import pytest
from fast_scorer import (
    compute_earnings_score, compute_insider_score, compute_macro_score,
    compute_geo_score, compute_fundamentals_score, compute_options_score,
    compute_wsb_short_score, compute_composite_score, score_all_assets
)
from tests.conftest import *  # uses fixtures inline

def test_earnings_score_high_prob():
    assert compute_earnings_score({"beat_probability": 0.85}) == pytest.approx(85, abs=2)

def test_earnings_score_missing_defaults_to_50():
    assert compute_earnings_score({}) == pytest.approx(50, abs=2)

def test_insider_score_strong_buy():
    # net_buy_pct_mktcap = 0.8 → score near 90
    score = compute_insider_score({"net_buy_pct_mktcap": 0.8})
    assert score > 80

def test_insider_score_strong_sell():
    score = compute_insider_score({"net_buy_pct_mktcap": -0.8})
    assert score < 20

def test_macro_score_tailwind():
    macro = {"sector_tailwinds": ["Technology"], "sector_headwinds": []}
    assert compute_macro_score("Technology", macro) == pytest.approx(75, abs=5)

def test_macro_score_headwind():
    macro = {"sector_tailwinds": [], "sector_headwinds": ["Technology"]}
    assert compute_macro_score("Technology", macro) == pytest.approx(25, abs=5)

def test_geo_score_no_events():
    score = compute_geo_score("AAPL", {"sector": "Technology", "region": "US"}, [])
    assert score == pytest.approx(50, abs=5)

def test_geo_score_positive_event():
    events = [{
        "probability": 0.8,
        "asset_impacts": [{"sector": "Technology", "direction": "positive", "magnitude": "strong"}]
    }]
    score = compute_geo_score("AAPL", {"sector": "Technology", "region": "US"}, events)
    assert score > 50

def test_options_score_bullish_pcr():
    # PCR = 0.5 → bullish → score > 70
    assert compute_options_score({"put_call_ratio": 0.5}) > 70

def test_options_score_bearish_pcr():
    # PCR = 1.5 → bearish → score < 40
    assert compute_options_score({"put_call_ratio": 1.5}) < 40

def test_composite_score_bounds():
    earnings = {"beat_probability": 0.6}
    insider = {"net_buy_pct_mktcap": 0.2}
    macro = {"sector_tailwinds": ["Technology"], "sector_headwinds": []}
    fundamentals = {"pe_ratio": 25, "return_on_equity": 20, "revenue_growth_yoy": 10}
    price = {"return_1y": 20, "volatility_annualized": 18, "pct_from_52w_high": -5}
    options = {"put_call_ratio": 0.8}
    short = {"short_float_pct": 5}
    wsb = {}
    asset_meta = {"sector": "Technology", "region": "US"}
    price_stats = {"avg_return_1y": 12.0}

    score = compute_composite_score(
        "AAPL", asset_meta, earnings, insider, macro, [], fundamentals, price, price_stats, options, short, wsb
    )
    assert 0 <= score <= 100
```

- [ ] **Step 2: Run failing**

```bash
pytest tests/test_fast_scorer.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement fast_scorer.py**

```python
# fast_scorer.py
"""
Python Pass 1: compute composite signal score (0-100) for every asset.
No Claude API needed. Pure math from signals.json.
Assets scoring >=70 or grade-changed get Claude deep analysis (Pass 2).
"""

import json


def compute_earnings_score(earnings_data: dict) -> float:
    """Earnings catalyst score: beat_probability → 0-100."""
    prob = earnings_data.get("beat_probability", 0.5)
    return round(float(prob) * 100, 1)


def compute_insider_score(insider_data: dict) -> float:
    """Insider flow score: net_buy_pct_mktcap ∈ [-1,1] → 0-100."""
    net = float(insider_data.get("net_buy_pct_mktcap", 0.0))
    return round(max(0.0, min(100.0, 50.0 + net * 50.0)), 1)


def compute_macro_score(sector: str, macro_signal: dict) -> float:
    """Macro regime fit: tailwind=75, headwind=25, neutral=50."""
    tailwinds = macro_signal.get("sector_tailwinds", [])
    headwinds = macro_signal.get("sector_headwinds", [])
    if sector in tailwinds:
        return 75.0
    if sector in headwinds:
        return 25.0
    return 50.0


def compute_geo_score(ticker: str, asset_meta: dict, events: list[dict]) -> float:
    """Geopolitical exposure score: base 50, adjusted by event impacts."""
    sector = asset_meta.get("sector", "")
    region = asset_meta.get("region", "")
    score = 50.0
    magnitude_map = {"strong": 30.0, "moderate": 15.0, "weak": 5.0}

    for event in events:
        p = float(event.get("probability", 0.0))
        for impact in event.get("asset_impacts", []):
            # Match on ticker, sector, or region
            matches = (
                impact.get("ticker") == ticker or
                (impact.get("sector") and impact.get("sector") == sector) or
                (impact.get("region") and impact.get("region") == region)
            )
            if matches:
                direction = 1.0 if impact.get("direction") == "positive" else -1.0
                magnitude = magnitude_map.get(impact.get("magnitude", "weak"), 5.0)
                score += direction * p * magnitude

    return round(max(0.0, min(100.0, score)), 1)


def compute_fundamentals_score(fund: dict, price: dict, price_stats: dict) -> float:
    """Value + growth + momentum composite (existing algorithm)."""
    pe = float(fund.get("pe_ratio") or 25)
    roe = float(fund.get("return_on_equity") or 10)
    rev_growth = float(fund.get("revenue_growth_yoy") or 0)
    ret = float(price.get("return_1y", 0))
    avg_ret = float(price_stats.get("avg_return_1y", 0))
    pct_high = float(price.get("pct_from_52w_high", -20))

    value_score = max(0.0, min(100.0, 100.0 - (pe / 50.0) * 50.0 + (roe / 30.0) * 20.0))
    growth_score = max(0.0, min(100.0, 50.0 + rev_growth + (roe - 10.0)))
    momentum_score = max(0.0, min(100.0, 50.0 + (ret - avg_ret) / 2.0 + pct_high / 2.0))

    return round(0.4 * value_score + 0.3 * growth_score + 0.3 * momentum_score, 1)


def compute_options_score(options_data: dict) -> float:
    """Options flow: put-call ratio → 0-100. PCR<0.7 bullish, PCR>1.3 bearish."""
    pcr = float(options_data.get("put_call_ratio", 1.0))
    score = max(0.0, min(100.0, 100.0 - (pcr - 0.5) * 50.0))
    return round(score, 1)


def compute_wsb_short_score(short_data: dict, wsb_data: dict) -> float:
    """WSB + short interest composite squeeze/momentum signal."""
    short_float = float(short_data.get("short_float_pct", 0))
    mentions = wsb_data.get("mentions_7d", 0) if isinstance(wsb_data, dict) else 0
    squeeze_flag = wsb_data.get("squeeze_flag", False) if isinstance(wsb_data, dict) else False

    if squeeze_flag and short_float > 20:
        return 85.0
    if short_float > 25 and mentions > 100:
        return 75.0
    return 50.0


def compute_composite_score(
    ticker: str,
    asset_meta: dict,
    earnings_data: dict,
    insider_data: dict,
    macro_signal: dict,
    events: list[dict],
    fund_data: dict,
    price_data: dict,
    price_stats: dict,
    options_data: dict,
    short_data: dict,
    wsb_ticker_data: dict,
) -> int:
    """Weighted composite score (0-100) per spec weights."""
    sector = asset_meta.get("sector", "")

    s_earnings = compute_earnings_score(earnings_data)
    s_insider = compute_insider_score(insider_data)
    s_macro = compute_macro_score(sector, macro_signal)
    s_geo = compute_geo_score(ticker, asset_meta, events)
    s_fund = compute_fundamentals_score(fund_data, price_data, price_stats)
    s_options = compute_options_score(options_data)
    s_wsb = compute_wsb_short_score(short_data, wsb_ticker_data)

    composite = (
        0.25 * s_earnings +
        0.20 * s_insider +
        0.15 * s_macro +
        0.15 * s_geo +
        0.15 * s_fund +
        0.05 * s_options +
        0.05 * s_wsb
    )
    return max(0, min(100, round(composite)))


def score_all_assets(signals: dict) -> dict[str, dict]:
    """
    Run fast scorer over all assets in signals.json.
    Returns dict: ticker → {score, signal_scores, sector, region, type}
    """
    universe_map = signals.get("universe_map", {})
    earnings = signals.get("earnings", {})
    insider = signals.get("insider", {})
    macro = signals.get("macro", {})
    events = signals.get("events", [])
    fundamentals = signals.get("fundamentals", {})
    price_data = signals.get("price_data", {})
    price_stats = signals.get("price_stats", {"avg_return_1y": 0})
    options = signals.get("options", {})
    short_interest = signals.get("short_interest", {})
    wsb = signals.get("wsb", {}).get("ticker_mentions", {})

    results = {}
    for ticker, asset_meta in universe_map.items():
        score = compute_composite_score(
            ticker=ticker,
            asset_meta=asset_meta,
            earnings_data=earnings.get(ticker, {}),
            insider_data=insider.get(ticker, {}),
            macro_signal=macro,
            events=events,
            fund_data=fundamentals.get(ticker, {}),
            price_data=price_data.get(ticker, {}),
            price_stats=price_stats,
            options_data=options.get(ticker, {}),
            short_data=short_interest.get(ticker, {}),
            wsb_ticker_data=wsb.get(ticker, {}),
        )
        results[ticker] = {
            "score": score,
            "sector": asset_meta.get("sector", ""),
            "region": asset_meta.get("region", ""),
            "type": asset_meta.get("type", "stock"),
        }
    return results


def score_to_grade(score: int) -> str:
    """Convert composite score to S&P-style letter grade."""
    thresholds = [
        (90, "AAA"), (83, "AA+"), (77, "AA"), (71, "AA-"),
        (65, "A+"), (59, "A"), (53, "A-"), (47, "BBB+"),
        (41, "BBB"), (35, "BBB-"), (29, "BB+"), (23, "BB"),
        (17, "BB-"), (12, "B"), (6, "CCC"),
    ]
    for threshold, grade in thresholds:
        if score >= threshold:
            return grade
    return "CC"
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fast_scorer.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add fast_scorer.py tests/test_fast_scorer.py
git commit -m "feat: two-pass fast scorer with composite signal weighting"
```

---

## Task 11: collect_all.py — Master Orchestrator

**Files:**
- Create: `collect_all.py`

- [ ] **Step 1: Implement**

```python
# collect_all.py
"""
Master data collection script.
Runs all 9 collectors in sequence and produces data/signals.json.
No Claude API required. Runtime: ~10-20 minutes for full universe.

Usage:
  python collect_all.py
  python collect_all.py --fast  (skips per-ticker collectors for quick test)
"""

import argparse
import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

from config import FRED_API_KEY
from tools.universe_manager import refresh_universe, load_universe
from tools.fred_tools import fetch_fred_series, classify_regime
from tools.futures_tools import fetch_futures_signal
from tools.polymarket_tools import fetch_geopolitical_markets, fetch_earnings_markets
from tools.gdelt_tools import fetch_regional_conflict_indices
from tools.news_tools import fetch_news_headlines
from tools.yfinance_tools import fetch_price_history, fetch_fundamentals
from tools.insider_tools import fetch_insider_transactions
from tools.options_tools import fetch_options_signal
from tools.short_interest_tools import fetch_short_interest
from tools.wsb_tools import fetch_wsb_posts, analyze_wsb_signals
from tools.btc_tools import fetch_btc_signal
from fast_scorer import score_all_assets


def build_universe_map(universe: list[dict]) -> dict:
    return {row["yf_ticker"]: row for row in universe}


def build_earnings_catalog(fundamentals: dict, poly_earnings: list[dict]) -> dict:
    """Merge yfinance earnings data with Polymarket earnings markets."""
    catalog = {}
    # Base from yfinance
    for ticker, fund in fundamentals.items():
        catalog[ticker] = {
            "beat_probability": 0.5,  # default
            "next_earnings_date": fund.get("next_earnings_date"),
            "consensus_eps": fund.get("eps"),
        }
    # Overlay Polymarket probabilities where company name matches
    for market in poly_earnings:
        q = market["question"].upper()
        for ticker in catalog:
            if ticker in q or ticker.replace(".DE", "").replace(".PA", "") in q:
                catalog[ticker]["beat_probability"] = market["probability"]
                catalog[ticker]["polymarket_question"] = market["question"]
    return catalog


def compute_price_stats(price_data: dict) -> dict:
    returns = [v.get("return_1y", 0) for v in price_data.values() if "error" not in v]
    return {
        "avg_return_1y": round(float(np.mean(returns)), 2) if returns else 0.0,
        "median_return_1y": round(float(np.median(returns)), 2) if returns else 0.0,
    }


def collect(fast: bool = False) -> dict:
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        print("ERROR: FRED_API_KEY not set")
        sys.exit(1)

    Path("data").mkdir(exist_ok=True)
    print("\n" + "="*55)
    print("  SIGNAL COLLECTION")
    print("="*55)

    # 1. Universe refresh
    print("\n[1/9] Refreshing universe...")
    refresh_universe(validate=not fast)
    universe = load_universe()
    tickers = [row["yf_ticker"] for row in universe]
    print(f"      {len(tickers)} assets loaded")

    # 2. Macro + FRED
    print("[2/9] FRED macro indicators...")
    fred_data = fetch_fred_series()
    macro_regime = classify_regime(fred_data)

    # 3. Commodity futures
    print("[3/9] Commodity futures...")
    futures_signal = fetch_futures_signal()
    macro_signal = {
        "regime": macro_regime.get("regime", "growth"),
        "risk_level": macro_regime.get("risk_level", "medium"),
        "sector_tailwinds": list(set(futures_signal.get("sector_tailwinds", []))),
        "sector_headwinds": list(set(futures_signal.get("sector_headwinds", []))),
        "futures_summary": futures_signal.get("summary", ""),
        "fred_indicators": fred_data,
    }

    # 4. Polymarket (geo + earnings)
    print("[4/9] Polymarket prediction markets...")
    poly_geo = fetch_geopolitical_markets()
    poly_earnings = fetch_earnings_markets()
    print(f"      {len(poly_geo)} geo markets, {len(poly_earnings)} earnings markets")

    # 5. GDELT geopolitical
    print("[5/9] GDELT regional conflict indices...")
    gdelt_data = fetch_regional_conflict_indices()

    # 6. News
    print("[6/9] News RSS headlines...")
    news_data = fetch_news_headlines()

    # 7. yfinance fundamentals + price (batch tickers)
    print("[7/9] Price history + fundamentals (batching)...")
    if fast:
        sample_tickers = tickers[:50]
    else:
        sample_tickers = tickers

    price_data = fetch_price_history(sample_tickers, period="1y")
    fundamentals = fetch_fundamentals(sample_tickers)
    price_stats = compute_price_stats(price_data)
    earnings_catalog = build_earnings_catalog(fundamentals, poly_earnings)

    # 8. Insider / options / short (top 200 by market cap to manage rate limits)
    print("[8/9] Insider flow, options, short interest...")
    rated_tickers = list({t for t in sample_tickers if t in price_data and t in fundamentals})[:200]
    insider_data = fetch_insider_transactions(rated_tickers)
    options_data = fetch_options_signal(rated_tickers)
    short_data = fetch_short_interest(rated_tickers)

    # 9. WSB + BTC
    print("[9/9] WSB Reddit + BTC signal...")
    wsb_posts = fetch_wsb_posts()
    wsb_signal = analyze_wsb_signals(wsb_posts)
    btc_signal = fetch_btc_signal()

    universe_map = build_universe_map(universe)

    signals = {
        "collected_at": datetime.now().isoformat(),
        "universe_size": len(tickers),
        "rated_ticker_count": len(rated_tickers),
        "macro": macro_signal,
        "polymarket_geo": poly_geo,
        "gdelt": gdelt_data,
        "news": news_data,
        "price_data": price_data,
        "price_stats": price_stats,
        "fundamentals": fundamentals,
        "earnings": earnings_catalog,
        "insider": insider_data,
        "options": options_data,
        "short_interest": short_data,
        "wsb": wsb_signal,
        "btc": btc_signal,
        "universe_map": {k: {"sector": v.get("sector", ""), "region": v.get("region", ""),
                             "type": v.get("type", "stock"), "name": v.get("name", "")}
                         for k, v in universe_map.items()},
        "events": [],  # populated by Claude event-extraction step in rating session
    }

    # Run fast scorer over all assets
    print("\n[fast-scorer] Computing composite scores for all assets...")
    fast_scores = score_all_assets(signals)
    signals["fast_scores"] = fast_scores

    high_priority = [t for t, s in fast_scores.items() if s["score"] >= 70]
    print(f"[fast-scorer] {len(fast_scores)} scored, {len(high_priority)} flagged for deep rating")

    out_path = "data/signals.json"
    with open(out_path, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    print(f"\n[collect_all] signals.json saved ({len(str(signals)) // 1024}KB)")
    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all signals for weekly rating")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 50 tickers, skip validation")
    args = parser.parse_args()
    collect(fast=args.fast)
```

- [ ] **Step 2: Smoke test**

```bash
python collect_all.py --fast
```

Expected: runs all collectors, prints progress, writes `data/signals.json`.

- [ ] **Step 3: Commit**

```bash
git add collect_all.py
git commit -m "feat: collect_all.py — master signal orchestrator"
```

---

## Task 12: Update Config and CLAUDE.md

**Files:**
- Modify: `config.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update config.py**

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

MODEL = "claude-opus-4-7"

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "CPI (Inflation)",
    "T10Y2Y": "10Y-2Y Yield Spread",
    "UNRATE": "Unemployment Rate",
    "T10YIE": "10Y Breakeven Inflation",
    "GDP": "Real GDP Growth",
    "DGS10": "10-Year Treasury Yield",
}

RISK_RULES = {
    "max_position_weight": 0.10,
    "max_sector_weight": 0.30,
    "min_positions": 10,
    "max_positions": 20,
    "min_conviction": 50,
}

# Fast scorer: assets scoring >= this get deep Claude analysis
DEEP_RATING_THRESHOLD = 70

# Batch size for per-ticker collectors (insider, options, short interest)
TICKER_BATCH_SIZE = 200

# Minimum Polymarket volume to include a market as a signal
POLYMARKET_MIN_VOLUME = 2000.0

SECTOR_WEIGHTS_BY_REGIME = {
    "growth": {
        "Technology": 0.25, "Consumer Discretionary": 0.18,
        "Financials": 0.15, "Healthcare": 0.12,
        "Industrials": 0.10, "Consumer Staples": 0.08,
        "Communication Services": 0.05, "Utilities": 0.02, "Other": 0.05,
    },
    "inflation": {
        "Energy": 0.20, "Materials": 0.15, "Financials": 0.15,
        "Consumer Staples": 0.15, "Healthcare": 0.12,
        "Industrials": 0.10, "Technology": 0.08, "Other": 0.05,
    },
    "stagflation": {
        "Consumer Staples": 0.25, "Healthcare": 0.20, "Energy": 0.15,
        "Utilities": 0.12, "Financials": 0.10, "Materials": 0.08,
        "Technology": 0.05, "Other": 0.05,
    },
    "deflation": {
        "Consumer Staples": 0.22, "Utilities": 0.18, "Healthcare": 0.18,
        "Technology": 0.15, "Financials": 0.12,
        "Consumer Discretionary": 0.08, "Other": 0.07,
    },
    "recession": {
        "Consumer Staples": 0.25, "Utilities": 0.20, "Healthcare": 0.20,
        "Financials": 0.10, "Technology": 0.10,
        "Communication Services": 0.08, "Other": 0.07,
    },
}
```

- [ ] **Step 2: Update CLAUDE.md**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
pip install -r requirements.txt
# FRED_API_KEY must be set in .env
python collect_all.py          # collect all signals → data/signals.json
python collect_all.py --fast   # fast mode: 50 tickers, skip validation (~5 min)
pytest tests/ -v               # run all tests
```

## Architecture

Two-pass weekly pipeline:

**Pass 1 (Python):** `collect_all.py` fetches 9 data sources → `data/signals.json`
**Pass 2 (Claude):** Reads `signals.json`, extracts events, rates all assets → `data/ratings_report.json`

Pass 2 runs as a scheduled remote Claude Code agent (Sunday 7pm Rome). No Anthropic API key needed.

## Key Data Flow

```
data/universe.csv      ← all TR stocks + ETFs (auto-refreshed weekly)
     │
collect_all.py (9 collectors)
     │
data/signals.json      ← compact signal snapshot (~15KB)
     │
[Claude Code session]  ← reads signals, extracts events A/B/C..., rates each asset
     │
data/ratings_report.json ← S&P-style grades + event portfolio matrix
```

## Signal Sources

| Collector | Source | Key Signal |
|---|---|---|
| UniverseManager | Wikipedia S&P 500 + seed European/ETF lists | Full TR asset list |
| MacroCommodities | FRED + CL=F GC=F NG=F HG=F | Regime + sector tailwinds |
| Polymarket | gamma-api.polymarket.com | Event probabilities |
| GDELT | api.gdeltproject.org | Regional conflict indices |
| InsiderFlow | SEC EDGAR Form 4 | Net insider buys/sells |
| OptionsFlow | yfinance options chain | Put-call ratio |
| ShortInterest | yfinance shortPercentOfFloat | Short float % |
| WSB | reddit.com/r/wallstreetbets public JSON | Ticker mentions, squeeze flags |
| BTC | yfinance BTC-USD | Liquidity/risk-on signal |

## Event Framework

Events A/B/C… are extracted by Claude from Polymarket + GDELT + news. Each has P(event) and P(¬event). Python enforces coherence (no contradictions). Claude maps events to TR assets via multi-hop causal chains. See `event_extractor.py` for validation logic.

## Adding Custom Assets

Edit `data/universe.csv`. Columns: `isin, name, yf_ticker, type, sector, region, added_date`

## Test Suite

```bash
pytest tests/ -v                              # all tests
pytest tests/test_fast_scorer.py -v          # scoring logic
pytest tests/test_event_coherence.py -v      # event probability rules
```
```

- [ ] **Step 3: Commit**

```bash
git add config.py CLAUDE.md
git commit -m "feat: update config and docs for full alpha system"
```

---

## Task 13: Update Remote Schedule Routine

**Files:** (no local files — updates the Claude Code remote agent prompt via API)

- [ ] **Step 1: Note the new agent prompt**

The remote agent prompt needs to reference `collect_all.py` and include the full rating methodology. This is the prompt that runs every Sunday.

The updated prompt instructs Claude to:
1. `pip install -r requirements.txt`
2. `python collect_all.py` → produces `data/signals.json`
3. Read `data/signals.json`
4. Extract 5–10 probabilistic events from geo+macro data
5. Validate event coherence via `python -c "from event_extractor import validate_and_normalize; ..."`
6. Rate all assets (carry forward low scorers, deep-rate high scorers)
7. Write `data/ratings_report.json`
8. `git add data/ && git commit -m "Weekly ratings — $(date +%Y-%m-%d)" && git push`

- [ ] **Step 2: Update the routine via RemoteTrigger**

Run this in Claude Code to update the existing routine `trig_01RKW83FK8KGoMd9US74RYqB`:

The new prompt body is:

```
You are a quantitative investment analyst running a weekly rating of all Trade Republic stocks and ETFs.

SETUP:
pip install -r requirements.txt

STEP 1 — COLLECT DATA:
python collect_all.py
This fetches all 9 signals and writes data/signals.json. Runtime ~15-20 min.

STEP 2 — READ SIGNALS:
Read data/signals.json. Note: fast_scores already computed for all assets.

STEP 3 — EXTRACT EVENTS (A, B, C...):
From signals.json, using polymarket_geo, gdelt, news, and macro sections:
Extract 5-10 discrete world events. For each event:
- id: "A", "B", etc.
- description: plain English, specific
- probability: use Polymarket outcomePrices if available, infer from GDELT conflict trend otherwise
- complement_probability: 1.0 - probability
- source: "polymarket" or "gdelt_inferred"
- resolution_date: best estimate
- asset_impacts: for each event, reason through 2-3 hop causal chain to derive which TR sectors/regions/specific tickers benefit or suffer, with direction (positive/negative) and magnitude (strong/moderate/weak)

COHERENCE RULES (enforce strictly):
- P(event) + P(¬event) must equal 1.0 for every event
- Mutually exclusive events must have probabilities summing to <= 1.0
- Flag any event that depends on another event

STEP 4 — RATE ASSETS:
Assets with fast_score >= 70: Rate in FULL DETAIL with rationale citing specific numbers.
Assets with fast_score < 70: Carry forward last_ratings if available, otherwise assign preliminary grade from fast_score.

Grade scale (score → grade):
90-100: AAA | 83-89: AA+ | 77-82: AA | 71-76: AA- | 65-70: A+ | 59-64: A | 53-58: A- | 47-52: BBB+ | 41-46: BBB | 35-40: BBB- | 29-34: BB+ | 23-28: BB | 17-22: BB- | 12-16: B | 6-11: CCC | 0-5: CC

Outlook:
- Positive: score +5 vs last week, OR earnings beat prob >= 70% within 14 days, OR WSB squeeze flag + short_float > 20%
- Negative: score -5 vs last week, OR high-probability negative event directly hits sector within 14 days
- Watch: major Polymarket event (>50% prob) resolving within 14 days would flip grade tier
- Stable: no material catalyst

For each deeply-rated asset include:
- ticker, name, grade, outlook, score
- signal_scores: {earnings_catalyst, insider_flow, macro_fit, event_exposure, fundamentals, options_flow, wsb_short}
- event_exposures: which events affect this asset and why (cite causal chain)
- rationale: 2-3 sentences citing specific numbers (exact probabilities, exact insider $, exact PCR)
- key_catalysts: list of strings
- key_risks: list of strings

STEP 5 — WRITE OUTPUT:
Write data/ratings_report.json with structure:
{
  "generated_at": ISO date,
  "universe_size": N,
  "deeply_rated_count": N,
  "macro_regime": string,
  "btc_signal": string,
  "weekly_events": [event objects with asset_impacts filled in],
  "wsb_alerts": [squeeze candidates from signals],
  "ratings": [all assets, deep or carry-forward],
  "top_10_buys": [tickers with highest scores and Buy outlook],
  "top_5_sells": [tickers with lowest scores],
  "squeeze_watchlist": [tickers with squeeze flags and short_float > 20%],
  "changes_from_last_week": [grade changes vs last_ratings.json if it exists],
  "event_portfolio_matrix": {event_id: {strong_buy: [tickers], strong_sell: [tickers]}}
}

Save current ratings as data/last_ratings.json (for next week's change detection).

STEP 6 — COMMIT:
git add data/
git commit -m "Weekly ratings — $(date +%Y-%m-%d)"
git push
```

- [ ] **Step 3: Apply the update in Claude Code**

Ask Claude Code to update the remote routine using the RemoteTrigger tool with the new prompt above targeting routine ID `trig_01RKW83FK8KGoMd9US74RYqB`.

- [ ] **Step 4: Run now to test**

Use Claude Code's RemoteTrigger to run the routine immediately and verify:
- `data/signals.json` appears in the repo after the run
- `data/ratings_report.json` contains events A/B/C… with probabilities
- `data/ratings_report.json` contains ratings with grades and rationales

- [ ] **Step 5: Final commit**

```bash
git add docs/
git commit -m "feat: complete alpha rating system — plan and implementation"
git push
```

---

## Verification Checklist

After implementation, confirm:

- [ ] `pytest tests/ -v` → all tests green
- [ ] `python collect_all.py --fast` → completes in <10 minutes, writes `data/signals.json`
- [ ] `data/signals.json` contains non-empty `fast_scores`, `polymarket_geo`, `wsb`, `btc`
- [ ] Remote routine runs and writes `data/ratings_report.json` to GitHub
- [ ] `ratings_report.json` contains `weekly_events` with probabilities and `event_portfolio_matrix`
- [ ] At least one event has specific Polymarket probability (not just 0.5 default)
- [ ] WSB alerts section contains at least one ticker
- [ ] BTC signal section contains a `regime` field

---

## What Replaces What

| Old File | New File | Change |
|---|---|---|
| `collect_data.py` | `collect_all.py` | 9 collectors instead of 2 |
| `data/tr_universe.csv` | `data/universe.csv` | Full universe (500-2000+ vs 50) |
| `agents/*.py` | Claude Code session | No local Claude API calls |
| None | `fast_scorer.py` | New: Python pass 1 |
| None | `event_extractor.py` | New: event coherence framework |
