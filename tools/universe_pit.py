"""Point-in-time universe view (pure): given a price frame and a map of
delisting dates, answer who was listed / tradeable / dead at any past date.

Survivors have no entry in `delisting` (delisting_date → None). A dead name's
price column ends at its delisting date (the assembly truncates it there), so
its last bar is its last traded price — what the graveyard liquidates at.
"""
import pandas as pd


class PITUniverse:
    def __init__(self, prices: pd.DataFrame,
                 delisting: dict[str, pd.Timestamp] | None = None):
        self.prices = prices
        self._first = {t: prices[t].first_valid_index() for t in prices.columns}
        self._delist = {t: pd.Timestamp(d)
                        for t, d in (delisting or {}).items() if pd.notna(d)}

    def first_trade_date(self, t):
        return self._first.get(t)

    def delisting_date(self, t):
        return self._delist.get(t)                       # None for survivors

    def listed(self, t, asof) -> bool:
        asof = pd.Timestamp(asof)
        ft = self._first.get(t)
        if ft is None or ft > asof:
            return False
        dl = self._delist.get(t)
        return dl is None or dl > asof

    def tradeable(self, asof, min_history_days: int = 273) -> set[str]:
        asof = pd.Timestamp(asof)
        hist = self.prices.loc[:asof]
        return {t for t in self.prices.columns
                if self.listed(t, asof)
                and hist[t].dropna().shape[0] >= min_history_days}

    def died_between(self, prev, asof) -> set[str]:
        prev, asof = pd.Timestamp(prev), pd.Timestamp(asof)
        return {t for t, d in self._delist.items() if prev < d <= asof}

    def last_price(self, t, asof=None):
        col = self.prices[t].dropna()
        if asof is not None:
            col = col.loc[:pd.Timestamp(asof)]
        return float(col.iloc[-1]) if len(col) else None
