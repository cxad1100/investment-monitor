"""Synthetic Lang & Schwarz price proxy for foreign names.

Trade Republic routes equity orders through Lang & Schwarz, which prices a foreign
stock off its *home* exchange × the live FX rate — NOT the dead Frankfurt-floor (.F)
shadow that EODHD/yfinance serve for the German cross-listing (near-zero volume,
stale/fake-ramping prices). So to backtest a genuinely TR-investable foreign name
(Seagate, AMD, Nvidia) we recreate the L&S price: take the liquid home (US) close and
divide by the daily EUR/USD rate.

Pure conversion lives here (tested); the ISIN→home-ticker mapping + the fetch/merge
that injects these series back into the momentum universe run in the build scripts,
driven by data/proxy_map.csv (german_ticker, isin, us_ticker, route).
"""
import pandas as pd


def to_eur(home_close: pd.Series, fx_usd_per_eur: pd.Series) -> pd.Series:
    """Convert a USD close series to EUR with `fx_usd_per_eur` = USD per 1 EUR
    (EURUSD=X ≈ 1.08): eur = usd / fx. The FX series is reindexed onto the price
    dates and gap-filled (home market + FX calendars differ on holidays), so a
    missing rate never punches a NaN hole into the converted series."""
    fx = fx_usd_per_eur.reindex(home_close.index).ffill().bfill()
    return (home_close / fx).dropna()


def median_turnover_eur(volume: pd.Series, eur_close: pd.Series, *, tail: int = 500) -> float:
    """Median daily EUR turnover (home volume × the EUR price) over the last `tail`
    bars — the liquidity gate. A real US listing (Seagate) clears it easily; a thin
    OTC stub does not, so the proxy only re-admits names that are actually tradeable."""
    turn = (eur_close * volume.reindex(eur_close.index)).dropna()
    return float(turn.tail(tail).median()) if len(turn) else 0.0
