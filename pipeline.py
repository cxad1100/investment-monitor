"""Orchestration pipeline: runs agents in sequence, passing shared state."""

import json
from datetime import datetime
from agents import data_engineer, macro_analyst, fundamental_analyst, portfolio_manager


def run_pipeline(
    csv_path: str = "data/tr_universe.csv",
    period: str = "1y",
    output_path: str = "portfolio_report.json",
) -> dict:
    """Run the full 4-agent pipeline and return the final state."""
    state: dict = {}
    start_time = datetime.now()

    print("\n" + "=" * 60)
    print("  MULTI-AGENT INVESTMENT SYSTEM")
    print("=" * 60)

    state = data_engineer.run(state, csv_path=csv_path, period=period)
    state = macro_analyst.run(state)
    state = fundamental_analyst.run(state)
    state = portfolio_manager.run(state)

    elapsed = (datetime.now() - start_time).seconds
    print(f"\n[Pipeline] Completed in {elapsed}s")

    save_report(state.get("portfolio", {}), output_path)
    return state


def save_report(portfolio_data: dict, output_path: str) -> None:
    """Write portfolio report to JSON file."""
    with open(output_path, "w") as f:
        json.dump(portfolio_data, f, indent=2, default=str)
    print(f"[Pipeline] Report saved to: {output_path}")


def print_summary(portfolio_data: dict) -> None:
    """Print a human-readable portfolio summary to stdout."""
    portfolio = portfolio_data.get("portfolio", [])
    stats = portfolio_data.get("stats", {})
    regime_ctx = portfolio_data.get("regime_context", "")
    thesis = portfolio_data.get("portfolio_thesis", "")

    print("\n" + "=" * 60)
    print("  PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"\nRegime: {regime_ctx}")
    print(f"Thesis: {thesis}")
    print(f"\nPositions ({len(portfolio)}):")
    print(f"  {'Ticker':<8} {'Sector':<25} {'Weight':>7} {'Rating':<6} {'Conv':>5}")
    print(f"  {'-'*7} {'-'*24} {'-'*7} {'-'*5} {'-'*5}")

    for p in sorted(portfolio, key=lambda x: x.get("weight", 0), reverse=True):
        ticker = p.get("ticker", "")
        sector = p.get("sector", "")[:24]
        weight = p.get("weight", 0)
        rating = p.get("rating", "")
        conviction = p.get("conviction", 0)
        print(f"  {ticker:<8} {sector:<25} {weight:>6.1%} {rating:<6} {conviction:>5}")

    print(f"\nPortfolio Statistics:")
    print(f"  Expected Return: {stats.get('expected_return', 0):.1%}")
    print(f"  Volatility:      {stats.get('volatility', 0):.1%}")
    print(f"  Sharpe Ratio:    {stats.get('sharpe', 0):.2f}")

    sector_breakdown = portfolio_data.get("sector_breakdown", {})
    if sector_breakdown:
        print(f"\nSector Breakdown:")
        for sector, weight in sorted(sector_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sector:<25} {weight:.1%}")

    print("\n" + "=" * 60)
