"""Curated stock universe for the pairs-trading engine.

Every name is tradeable on Lang & Schwarz Exchange (Trade Republic).
Candidate pairs are restricted to same sector AND same currency —
cross-currency pairs leak the FX trend into the spread and produce
spurious cointegration.

slippage_bps = assumed half-spread cost per traded leg:
5 bps US mega-caps, 10 bps DAX names, 15 bps mid-liquidity names.
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent

UNIVERSE: dict[str, dict] = {
    # US banks
    "JPM": dict(sector="US Banks", currency="USD", slippage_bps=5),
    "BAC": dict(sector="US Banks", currency="USD", slippage_bps=5),
    "C":   dict(sector="US Banks", currency="USD", slippage_bps=5),
    "WFC": dict(sector="US Banks", currency="USD", slippage_bps=5),
    "GS":  dict(sector="US Banks", currency="USD", slippage_bps=5),
    "MS":  dict(sector="US Banks", currency="USD", slippage_bps=5),
    # US semiconductors
    "NVDA": dict(sector="US Semis", currency="USD", slippage_bps=5),
    "AMD":  dict(sector="US Semis", currency="USD", slippage_bps=5),
    "AVGO": dict(sector="US Semis", currency="USD", slippage_bps=5),
    "TXN":  dict(sector="US Semis", currency="USD", slippage_bps=5),
    "INTC": dict(sector="US Semis", currency="USD", slippage_bps=5),
    "MU":   dict(sector="US Semis", currency="USD", slippage_bps=5),
    # US consumer staples
    "KO":  dict(sector="US Staples", currency="USD", slippage_bps=5),
    "PEP": dict(sector="US Staples", currency="USD", slippage_bps=5),
    "PG":  dict(sector="US Staples", currency="USD", slippage_bps=5),
    "CL":  dict(sector="US Staples", currency="USD", slippage_bps=5),
    # US payments
    "V":   dict(sector="US Payments", currency="USD", slippage_bps=5),
    "MA":  dict(sector="US Payments", currency="USD", slippage_bps=5),
    "AXP": dict(sector="US Payments", currency="USD", slippage_bps=5),
    # US energy
    "XOM": dict(sector="US Energy", currency="USD", slippage_bps=5),
    "CVX": dict(sector="US Energy", currency="USD", slippage_bps=5),
    "COP": dict(sector="US Energy", currency="USD", slippage_bps=5),
    # German autos
    "BMW.DE":  dict(sector="DE Autos", currency="EUR", slippage_bps=10),
    "MBG.DE":  dict(sector="DE Autos", currency="EUR", slippage_bps=10),
    "VOW3.DE": dict(sector="DE Autos", currency="EUR", slippage_bps=10),
    "P911.DE": dict(sector="DE Autos", currency="EUR", slippage_bps=15),
    # German industrials / chemicals
    "SIE.DE":  dict(sector="DE Industrial", currency="EUR", slippage_bps=10),
    "BAS.DE":  dict(sector="DE Industrial", currency="EUR", slippage_bps=10),
    "BAYN.DE": dict(sector="DE Industrial", currency="EUR", slippage_bps=10),
    # German banks
    "DBK.DE": dict(sector="DE Banks", currency="EUR", slippage_bps=10),
    "CBK.DE": dict(sector="DE Banks", currency="EUR", slippage_bps=10),
}


def candidate_pairs() -> list[tuple[str, str]]:
    """All ticker pairs sharing sector AND currency."""
    out = []
    for a, b in combinations(sorted(UNIVERSE), 2):
        ua, ub = UNIVERSE[a], UNIVERSE[b]
        if ua["sector"] == ub["sector"] and ua["currency"] == ub["currency"]:
            out.append((a, b))
    return out


def fetch_prices(tickers: list[str] | None = None, years: int = 5,
                 cache: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes for the universe, cached as CSV in data/.

    The cache is reused when it covers all requested tickers and its last
    row is at most 3 days old. Interior gaps are forward-filled; leading
    NaN (late IPOs like P911.DE) are kept — per-pair alignment happens in
    select_pairs / run_backtest.
    """
    tickers = list(tickers or UNIVERSE)
    cache = cache or ROOT / "data" / "pairs_prices.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        age = (pd.Timestamp.today().normalize() - df.index[-1]).days
        if set(tickers) <= set(df.columns) and age <= 3:
            return df[tickers]
    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.dropna(how="all").ffill()
    cache.parent.mkdir(exist_ok=True)
    close.to_csv(cache)
    return close[tickers]
