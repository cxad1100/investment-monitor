"""Multi-Agent Investment System — Entry Point.

Usage:
  python main.py
  python main.py --universe data/tr_universe.csv --period 1y --output portfolio_report.json
"""

import argparse
import sys
from pipeline import run_pipeline, print_summary
from config import ANTHROPIC_API_KEY, FRED_API_KEY


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Investment System for Trade Republic universe"
    )
    parser.add_argument(
        "--universe",
        default="data/tr_universe.csv",
        help="Path to TR universe CSV (default: data/tr_universe.csv)",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Historical data period for yfinance (default: 1y). Options: 6mo, 1y, 2y",
    )
    parser.add_argument(
        "--output",
        default="portfolio_report.json",
        help="Output path for portfolio report JSON (default: portfolio_report.json)",
    )
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        print("ERROR: ANTHROPIC_API_KEY not set. Edit .env file.")
        sys.exit(1)
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        print("ERROR: FRED_API_KEY not set. Edit .env file.")
        print("  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    try:
        state = run_pipeline(
            csv_path=args.universe,
            period=args.period,
            output_path=args.output,
        )
        print_summary(state.get("portfolio", {}))
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
