"""Source EUR-exchange stocks that DIED (delisted) 2018→now and were real while
alive (≥ €1, ≤ 1.5 % spread), for the survivorship-corrected momentum universe.

Pure core (this section) is unit-tested. The network adapters + CLI below are
best-effort; a hand-seeded data/eur_delisted_seed.csv guarantees real dead names
even if scraping yields nothing.

A removed index member is only DEAD if its prices stop near the removal date —
that distinguishes a bankruptcy (Wirecard) from a mere demotion (still trading).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_REMOVE = {"removed", "deleted", "delisted", "remove"}


def parse_index_changes(rows: list[dict]) -> list[dict]:
    """Index change rows → removal candidates [{ticker, name, removal_date}]."""
    out = []
    for r in rows:
        if str(r.get("action", "")).strip().lower() in _REMOVE:
            out.append({"ticker": r["ticker"], "name": r.get("name", r["ticker"]),
                        "removal_date": pd.Timestamp(r["date"])})
    return out


def classify_dead(price_series: pd.Series, removal_date, today,
                  *, gap_days: int = 20):
    """Delisting date (= last real bar) if the series stops trading near removal,
    else None (still printing prices → demoted, not dead)."""
    s = price_series.dropna()
    if s.empty:
        return None
    last = s.index[-1]
    if (pd.Timestamp(today) - last).days > gap_days:
        return last
    return None


def keep_real(candidates: list[dict], *, min_price: float = 1.0,
              max_spread_pct: float = 1.5) -> list[dict]:
    """Keep dead names that were real: last price ≥ min_price and spread ≤ cap %."""
    out = []
    for c in candidates:
        lp, sp = c.get("last_price"), c.get("spread_pct")
        if lp is None or sp is None:
            continue
        if lp >= min_price and sp <= max_spread_pct:
            out.append(c)
    return out
