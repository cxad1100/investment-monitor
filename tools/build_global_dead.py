"""Global survivorship set: re-source DELISTED names from their HOME exchange.

The German delisted master list (data/eur_delisted_list.csv) is dominated by foreign
names whose *Frankfurt* listing was pulled while the company kept trading at home — a
withdrawal, not a death. We separate the two using home-exchange status:

  german-delisted ISIN that is ACTIVE on its home exchange  → withdrawal  → skip
  german-delisted ISIN that is DELISTED on its home exchange → real death → include

For the real deaths we pull the home EOD up to delisting, convert to EUR, and keep only
clean, once-liquid true collapses (is_clean_dead + last ≤ ½ peak). delisting_date = last
home bar → the PIT graveyard liquidates there. Real home data, no broken .F shadows.

Run: .venv/bin/python -m tools.build_global_dead
Writes data/global_dead_prices.csv + data/global_dead_meta.csv.
"""
import concurrent.futures as cf
import json
import time

import pandas as pd

from tools import eodhd
from tools.synthetic_proxy import to_eur, median_turnover_eur
from tools.dead_stocks import is_clean_dead
from tools.build_global_proxy import COUNTRY_EXCH, fx_eur, fetch_cv, _get

ROOT = eodhd.ROOT
FLOOR = 50_000.0                 # €/day while alive (deads run smaller; still tradeable)
MAX_SURVIVAL = 0.5               # true collapse: last ≤ ½ peak


def exch_isin_codes(exch, key, delisted):
    flag = "&delisted=1" if delisted else ""
    rows = _get(f"https://eodhd.com/api/exchange-symbol-list/{exch}"
                f"?api_token={key}&fmt=json{flag}") or []
    return {str(r.get("Isin", "")): r["Code"] for r in rows
            if r.get("Isin") and r.get("Type") == "Common Stock"}


def main():
    key = eodhd.api_key()
    lst = pd.read_csv(ROOT / "data" / "eur_delisted_list.csv")
    lst = lst[lst["Type"] == "Common Stock"].dropna(subset=["Isin"])
    need = {v[0] for v in COUNTRY_EXCH.values()}
    print(f"fetching home active+delisted lists for {len(need)} exchanges…", flush=True)
    active = {e: exch_isin_codes(e, key, False) for e in need}
    dead = {e: exch_isin_codes(e, key, True) for e in need}

    targets, seen = [], set()
    for _, r in lst.iterrows():
        isin = str(r["Isin"])
        ce = COUNTRY_EXCH.get(isin[:2])
        if not ce or isin in seen:
            continue
        exch, ccy, pence = ce
        if isin in active.get(exch, {}):
            continue                                   # alive at home → withdrawal, not a death
        code = dead.get(exch, {}).get(isin)
        if not code:
            continue                                   # not delisted at home either → unknown, skip
        seen.add(isin)
        targets.append((f"{code}.{exch}", ccy, pence, str(r.get("Name", code))))
    print(f"{len(targets)} home-confirmed deaths to fetch", flush=True)

    fxs = {c: fx_eur(c, key) for c in {ccy for _, ccy, _, _ in targets}}

    def work(item):
        sym, ccy, pence, name = item
        close, vol = fetch_cv(sym, key)
        if close is None:
            return None
        if pence:
            close = close / 100.0
        fx = fxs.get(ccy)
        eur = close if fx is None else to_eur(close, fx)
        eur = eur.dropna()
        if len(eur) < 252 or float(eur.iloc[-1]) > MAX_SURVIVAL * float(eur.max()):
            return None                                # not a real collapse
        if not is_clean_dead(eur, min_obs=252):
            return None
        if median_turnover_eur(vol, eur, tail=len(eur)) < FLOOR:
            return None                                # never liquid while alive
        return sym, eur, dict(ticker=sym, name=name, sector="Unknown", country="—",
                              currency="EUR", slippage_bps=25, local_id="",
                              delisting_date=str(eur.index[-1].date()),
                              med_turnover=median_turnover_eur(vol, eur, tail=len(eur)), home=sym)

    rows, meta, t0 = {}, [], time.time()
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, r in enumerate(pool.map(work, targets), 1):
            if r:
                sym, eur, m = r
                rows[sym] = eur
                meta.append(m)
            if i % 250 == 0:
                print(f"  {i}/{len(targets)} kept={len(rows)} {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).sort_index().to_csv(ROOT / "data" / "global_dead_prices.csv")
    pd.DataFrame(meta).to_csv(ROOT / "data" / "global_dead_meta.csv", index=False)
    print(f"DONE {time.time()-t0:.0f}s: {len(rows)} home-sourced real deaths", flush=True)


if __name__ == "__main__":
    main()
