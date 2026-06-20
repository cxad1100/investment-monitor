"""Liquidity/criteria gate for the universe build — pure, shared, testable.

History: this module used to be the EODHD bulk universe builder (screen across major
exchanges → pre-screen turnover → full-fetch). That EODHD path is superseded by the
Trade-Republic-native pipeline (`tools.tr_tradeable --enumerate` → `tools.build_tr_universe`,
priced via yfinance). What survives — and is reused by the new builder — is the pure
criteria gate `apply_criteria` plus the country-name map and thresholds.

`apply_criteria(eur, vol, active)` decides whether an EUR-converted price series clears the
liquidity floor (≥€100k/day median turnover), the €1 price floor and the glitch checks; for
delisted names it additionally requires a real, once-liquid, glitch-free collapse.
"""
import pathlib

import pandas as pd

from tools.synthetic_proxy import median_turnover_eur
from tools.dead_stocks import is_clean_dead

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Country code (ISIN prefix / ISO) -> display name, used by the universe builders.
CC_NAME = {
    "US": "USA", "GB": "UK", "DE": "Germany", "FR": "France", "ES": "Spain",
    "NL": "Netherlands", "BE": "Belgium", "PT": "Portugal", "AT": "Austria",
    "CH": "Switzerland", "SE": "Sweden", "FI": "Finland", "NO": "Norway",
    "DK": "Denmark", "IE": "Ireland", "CA": "Canada", "HK": "Hong Kong",
    "AU": "Australia", "KR": "South Korea", "TW": "Taiwan", "IL": "Israel",
    "PL": "Poland", "JP": "Japan", "IT": "Italy",
    "BM": "Bermuda", "KY": "Cayman", "JE": "Jersey", "LU": "Luxembourg",
}

TURN_FLOOR = 100_000.0        # €/day median turnover — the liquidity gate
TURN_CEIL = 50_000_000_000.0  # €/day ceiling — above this is a price/volume glitch
PRICE_FLOOR = 1.0             # €  drop penny/tick noise
MIN_OBS = 378                 # ~1.5y of trading bars
DEAD_TURN_FLOOR = 50_000.0    # €/day while alive (deads run smaller but were tradeable)
DEAD_COLLAPSE = 0.5           # a real death ends ≤ ½ its peak


def apply_criteria(eur: pd.Series, vol: pd.Series, active: bool) -> dict | None:
    """Pure gate: an EUR price series + home volume -> {delisting_date, med_turnover}
    if it qualifies, else None. Live names need length + €1 + the turnover band; dead
    names additionally need a real, once-liquid, glitch-free collapse (≤ ½ peak)."""
    eur = eur.dropna()
    if len(eur) < MIN_OBS:
        return None
    if active:
        if float(eur.tail(60).median()) < PRICE_FLOOR:
            return None
        med = median_turnover_eur(vol, eur)
        if not (TURN_FLOOR <= med <= TURN_CEIL):
            return None
        return dict(delisting_date="", med_turnover=med)
    if float(eur.max()) < PRICE_FLOOR or float(eur.iloc[-1]) > DEAD_COLLAPSE * float(eur.max()):
        return None
    if not is_clean_dead(eur, min_obs=MIN_OBS):
        return None
    med = median_turnover_eur(vol, eur, tail=len(eur))
    if med < DEAD_TURN_FLOOR:
        return None
    return dict(delisting_date=str(eur.index[-1].date()), med_turnover=med)
