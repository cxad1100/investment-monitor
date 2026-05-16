"""Data collection script — no Claude API required.

Fetches all market data and saves a timestamped snapshot to data/snapshot.json.
Run this first, then ask Claude Code to analyze the snapshot.

Usage:
  python collect_data.py
  python collect_data.py --universe data/tr_universe.csv --period 1y
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tools.universe import get_universe
from tools.yfinance_tools import fetch_price_history, fetch_fundamentals, get_sector_median_pe
from tools.fred_tools import fetch_fred_series, classify_regime
from config import FRED_API_KEY


def collect(csv_path: str = "data/tr_universe.csv", period: str = "1y") -> dict:
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        print("ERROR: FRED_API_KEY not set in .env")
        sys.exit(1)

    print(f"[collect] Loading universe from {csv_path}...")
    universe = get_universe(csv_path, validate=False)
    tickers = [row["yf_ticker"] for row in universe]
    print(f"[collect] {len(tickers)} tickers loaded")

    print(f"[collect] Fetching {period} price history...")
    price_data = fetch_price_history(tickers, period)
    valid_price = {k: v for k, v in price_data.items() if "error" not in v}
    print(f"[collect] {len(valid_price)}/{len(tickers)} tickers have price data")

    print("[collect] Fetching fundamentals...")
    fundamentals = fetch_fundamentals(tickers)
    valid_fund = {k: v for k, v in fundamentals.items() if "error" not in v}
    print(f"[collect] {len(valid_fund)}/{len(tickers)} tickers have fundamental data")

    print("[collect] Fetching FRED macro data...")
    fred_data = fetch_fred_series()
    print(f"[collect] {len(fred_data)} FRED series fetched")

    regime = classify_regime(fred_data)

    sectors = list({row.get("sector", "Other") for row in universe})
    sector_median_pe = {}
    for sector in sectors:
        median_pe = get_sector_median_pe(valid_fund, sector)
        if median_pe:
            sector_median_pe[sector] = median_pe

    snapshot = {
        "fetched_at": datetime.now().isoformat(),
        "period": period,
        "universe": universe,
        "price_data": valid_price,
        "fundamentals": valid_fund,
        "fred_data": fred_data,
        "regime_quant": regime,
        "sector_median_pe": sector_median_pe,
        "summary": {
            "total_tickers": len(tickers),
            "valid_price_count": len(valid_price),
            "valid_fund_count": len(valid_fund),
            "valid_both_count": len(set(valid_price) & set(valid_fund)),
        },
    }

    Path("data").mkdir(exist_ok=True)
    out_path = "data/snapshot.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"[collect] Snapshot saved to {out_path}")
    print(f"[collect] {snapshot['summary']['valid_both_count']} stocks ready for analysis")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect market data for portfolio analysis")
    parser.add_argument("--universe", default="data/tr_universe.csv")
    parser.add_argument("--period", default="1y")
    args = parser.parse_args()
    collect(args.universe, args.period)
