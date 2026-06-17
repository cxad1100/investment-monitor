"""Global Lang & Schwarz proxy: re-source EVERY universe name from its HOME exchange.

Trade Republic prices a stock off its home venue × live FX, not the dead German
Frankfurt-floor (.F) shadow. So for the whole universe we map each ISIN to its home
exchange (DE→XETRA, US→US, IT→MI, CH→SW, NO→OL, GB→LSE…), pull the liquid home EOD
from EODHD, convert to EUR (÷ FX, ÷100 first for London pence), and keep names that
clear a turnover floor. One consistent, real-priced, EUR universe — no broken shadows.

Run: .venv/bin/python -m tools.build_global_proxy
Writes data/global_proxy_prices.csv + data/global_proxy_meta.csv.
Reads data/proxy_map.csv (ticker, isin, country, …) for the ISINs.
"""
import concurrent.futures as cf
import json
import time

import pandas as pd

from tools import eodhd
from tools.pairs_universe import _load_universe
from tools.synthetic_proxy import to_eur, median_turnover_eur

ROOT = eodhd.ROOT
FLOOR = 100_000.0          # €/day median turnover

# ISIN country prefix -> (EODHD home exchange, home currency, prices_in_pence)
COUNTRY_EXCH = {
    "DE": ("XETRA", "EUR", 0), "US": ("US", "USD", 0), "CA": ("TO", "CAD", 0),
    "GB": ("LSE", "GBP", 1), "FR": ("PA", "EUR", 0), "CH": ("SW", "CHF", 0),
    "NL": ("AS", "EUR", 0), "ES": ("MC", "EUR", 0), "IT": ("MI", "EUR", 0),
    "SE": ("ST", "SEK", 0), "AT": ("VI", "EUR", 0), "BE": ("BR", "EUR", 0),
    "FI": ("HE", "EUR", 0), "NO": ("OL", "NOK", 0), "DK": ("CO", "DKK", 0),
    "IE": ("IR", "EUR", 0), "PT": ("LS", "EUR", 0), "LU": ("LU", "EUR", 0),
}


def _get(url):
    try:
        return json.loads(eodhd._http_get(url))
    except Exception:
        return None


def exchange_isin_map(exch, key):
    """{ISIN: Code} for active common stock on an EODHD exchange."""
    rows = _get(f"https://eodhd.com/api/exchange-symbol-list/{exch}?api_token={key}&fmt=json") or []
    return {str(r.get("Isin", "")): r["Code"] for r in rows
            if r.get("Isin") and r.get("Type") == "Common Stock"}


def fx_eur(ccy, key):
    """Daily series of `ccy` per 1 EUR (EUR{CCY}.FOREX). EUR → all-ones."""
    if ccy == "EUR":
        return None
    d = _get(f"https://eodhd.com/api/eod/EUR{ccy}.FOREX?api_token={key}&fmt=json&from=2017-01-01")
    if not isinstance(d, list) or not d:
        return None
    return pd.Series([r["adjusted_close"] for r in d],
                     index=pd.to_datetime([r["date"] for r in d])).dropna()


def fetch_cv(sym, key):
    """(close, volume) home series from EODHD, or (None, None)."""
    d = _get(f"https://eodhd.com/api/eod/{sym}?api_token={key}&fmt=json&from=2017-01-01")
    if not isinstance(d, list) or not d or "warning" in d[0]:
        return None, None
    idx = pd.to_datetime([r["date"] for r in d])
    close = pd.Series([r["adjusted_close"] for r in d], index=idx).dropna()
    vol = pd.Series([r.get("volume", 0) for r in d], index=idx)
    return (close, vol) if len(close) else (None, None)


SEED_EXCH = ("XETRA", "MU", "STU")   # Lang & Schwarz retail venues: XETRA + Munich/gettex + Stuttgart
PRICE_FLOOR = 1.0                    # drop sub-€1 names (penny/tick noise, the "no trash" guardrail)
ISIN_CC = {"US": "USA", "DE": "Germany", "FR": "France", "CA": "Canada", "CH": "Switzerland",
           "NL": "Netherlands", "GB": "UK", "IT": "Italy", "ES": "Spain", "SE": "Sweden",
           "NO": "Norway", "DK": "Denmark", "FI": "Finland", "BE": "Belgium", "AT": "Austria",
           "IE": "Ireland", "PT": "Portugal", "LU": "Luxembourg"}


def seed_companies(key):
    """{ISIN: (representative_german_ticker, name)} — every common stock on the L&S venues,
    deduped by ISIN (XETRA listing wins, so the broker line is the liquid one)."""
    out = {}
    for exch in reversed(SEED_EXCH):                 # STU, MU, then XETRA last → XETRA overwrites/wins
        rows = _get(f"https://eodhd.com/api/exchange-symbol-list/{exch}?api_token={key}&fmt=json") or []
        for r in rows:
            isin = str(r.get("Isin", ""))
            if isin and r.get("Type") == "Common Stock":
                out[isin] = (f"{r['Code']}.{exch}", str(r.get("Name", r["Code"])))
    return out


def main():
    key = eodhd.api_key()
    need_exch = {v[0] for v in COUNTRY_EXCH.values()}
    print(f"fetching {len(need_exch)} home exchange lists…", flush=True)
    isin2code = {e: exchange_isin_map(e, key) for e in need_exch}
    exch_info = {v[0]: (v[1], v[2]) for v in COUNTRY_EXCH.values()}
    companies = seed_companies(key)
    print(f"L&S seed: {len(companies)} unique companies (XETRA+MU+STU)", flush=True)

    def resolve(isin):
        """Home (sym, ccy, pence) for an ISIN: domicile exchange first, then US (many
        IE/BM/KY names are US-listed — Seagate is IE-domiciled, NASDAQ STX), then any."""
        dom = COUNTRY_EXCH.get(isin[:2])
        order = ([dom[0]] if dom else []) + (["US"] if (not dom or dom[0] != "US") else [])
        order += [e for e in isin2code if e not in order]
        for e in order:
            code = isin2code.get(e, {}).get(isin)
            if code:
                ccy, pence = exch_info[e]
                return f"{code}.{e}", ccy, pence
        return None

    targets = []
    for isin, (ger, name) in companies.items():
        if len(isin) < 2:
            continue
        home = resolve(isin)
        if home:
            sym, ccy, pence = home
            targets.append((ger, sym, ccy, pence, isin, name))
    print(f"mapped {len(targets)} to home tickers", flush=True)

    fxs = {c: fx_eur(c, key) for c in {ccy for _, _, ccy, _, _, _ in targets}}

    def work(item):
        ger, sym, ccy, pence, isin, name = item
        close, vol = fetch_cv(sym, key)
        if close is None:
            return None
        if pence:
            close = close / 100.0
        fx = fxs.get(ccy)
        eur = (close if fx is None else to_eur(close, fx)).dropna()
        if len(eur) < 300 or float(eur.tail(60).median()) < PRICE_FLOOR:
            return None                                  # too short, or sub-€1 penny
        med = median_turnover_eur(vol, eur)
        if med < FLOOR:
            return None                                  # illiquid
        return ger, eur, dict(ticker=ger, name=name, sector="Unknown",
                              country=ISIN_CC.get(isin[:2], "—"), currency="EUR",
                              slippage_bps=25, local_id="", delisting_date="",
                              med_turnover=med, isin=isin, home=sym)

    rows, meta, t0 = {}, [], time.time()
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        for i, r in enumerate(pool.map(work, targets), 1):
            if r:
                ger, eur, mrow = r
                rows[ger] = eur
                meta.append(mrow)
            if i % 500 == 0:
                print(f"  {i}/{len(targets)} kept={len(rows)} {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).sort_index().to_csv(ROOT / "data" / "global_proxy_prices.csv")
    pd.DataFrame(meta).to_csv(ROOT / "data" / "global_proxy_meta.csv", index=False)
    print(f"DONE {time.time()-t0:.0f}s: {len(rows)} liquid home-sourced names "
          f"-> global_proxy_prices.csv + global_proxy_meta.csv", flush=True)


if __name__ == "__main__":
    main()
