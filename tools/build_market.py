"""Fresh, source-of-truth universe: every liquid stock EODHD can serve, priced at home.

Start-fresh rebuild. Forget the Lang & Schwarz seed + ISIN→home resolve dance. We just
enumerate the major HOME exchanges EODHD actually serves (skipping the F/STU/MU German
cross-listing *shadows* — those are the broken near-zero-volume feeds the bias hunt
killed), pull every Common Stock (active AND delisted), convert each to EUR, and keep
whatever clears the criteria. The criteria ARE the filter: a name that clears the
liquidity floor (≥€100k/day median turnover), the €1 price floor and the glitch checks
is assumed Trade-Republic-tradeable — if it's that liquid it's almost certainly routable.

Fetching ~86k full histories would blow the API budget, so we pre-screen cheaply first:
EODHD's `eod-bulk-last-day/{EXCH}` returns a whole exchange's OHLCV in one call. We pull
it for ~17 semi-annual sample dates (2018→2026) — recent dates catch live names, older
dates catch delisted names *while they were still trading* — take each code's peak
close×volume as a turnover proxy, and only full-fetch the codes that clear a loose floor.

Run:
  .venv/bin/python -m tools.build_market screen     # -> data/universe/_candidates.csv
  .venv/bin/python -m tools.build_market fetch       # -> data/universe/universe_{prices,meta}.csv

Output lives in the NEW data/universe/ subdir (the single source of truth from now on).
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

from tools import eodhd
from tools.synthetic_proxy import to_eur, median_turnover_eur
from tools.dead_stocks import is_clean_dead

ROOT = eodhd.ROOT
OUT = ROOT / "data" / "universe"

# Major HOME exchanges EODHD serves on this plan -> (home currency, prices_in_pence).
# Deliberately NO F / STU / MU: those are the German cross-listing shadow venues whose
# thin, fake-ramping feeds we proved are garbage. XETRA stays — it's the German PRIMARY.
EXCHANGES = {
    "US": ("USD", 0), "LSE": ("GBP", 1), "XETRA": ("EUR", 0), "PA": ("EUR", 0),
    "MC": ("EUR", 0), "AS": ("EUR", 0), "BR": ("EUR", 0), "LS": ("EUR", 0),
    "VI": ("EUR", 0), "SW": ("CHF", 0), "ST": ("SEK", 0), "HE": ("EUR", 0),
    "OL": ("NOK", 0), "CO": ("DKK", 0), "IR": ("EUR", 0), "TO": ("CAD", 0),
    "V": ("CAD", 0), "HK": ("HKD", 0), "AU": ("AUD", 0), "KO": ("KRW", 0),
    "KQ": ("KRW", 0), "TW": ("TWD", 0), "TWO": ("TWD", 0), "TA": ("ILS", 1),
    "WAR": ("PLN", 0),
}

# ISIN country prefix -> the preferred home exchange (used to break dedup ties so a name
# is priced on its real home, not a thin cross-listing) and a display country name.
COUNTRY_PRIMARY = {
    "US": "US", "GB": "LSE", "DE": "XETRA", "FR": "PA", "ES": "MC", "NL": "AS",
    "BE": "BR", "PT": "LS", "AT": "VI", "CH": "SW", "SE": "ST", "FI": "HE",
    "NO": "OL", "DK": "CO", "IE": "IR", "CA": "TO", "HK": "HK", "AU": "AU",
    "KR": "KO", "TW": "TW", "IL": "TA", "PL": "WAR",
}
CC_NAME = {
    "US": "USA", "GB": "UK", "DE": "Germany", "FR": "France", "ES": "Spain",
    "NL": "Netherlands", "BE": "Belgium", "PT": "Portugal", "AT": "Austria",
    "CH": "Switzerland", "SE": "Sweden", "FI": "Finland", "NO": "Norway",
    "DK": "Denmark", "IE": "Ireland", "CA": "Canada", "HK": "Hong Kong",
    "AU": "Australia", "KR": "South Korea", "TW": "Taiwan", "IL": "Israel",
    "PL": "Poland", "BM": "Bermuda", "KY": "Cayman", "JE": "Jersey", "LU": "Luxembourg",
}

# Semi-annual probe dates: recent ones catch live names, older ones catch delisted names
# while they were still trading (so the pre-screen sees their once-alive liquidity).
SAMPLE_DATES = [
    "2018-06-15", "2018-12-14", "2019-06-17", "2019-12-16", "2020-06-15", "2020-12-15",
    "2021-06-15", "2021-12-15", "2022-06-15", "2022-12-15", "2023-06-15", "2023-12-15",
    "2024-06-17", "2024-12-16", "2025-06-16", "2025-12-15", "2026-06-17",
]

SCREEN_FLOOR = 80_000.0       # €/day loose pre-screen floor (real floor applied post-fetch)
TURN_FLOOR = 100_000.0        # €/day median turnover — the liquidity gate
TURN_CEIL = 50_000_000_000.0  # €/day ceiling — above this is a price/volume glitch
PRICE_FLOOR = 1.0             # €  drop penny/tick noise
MIN_OBS = 378                 # ~1.5y of trading bars
DEAD_TURN_FLOOR = 50_000.0    # €/day while alive (deads run smaller but were tradeable)
DEAD_COLLAPSE = 0.5           # a real death ends ≤ ½ its peak


def _get(url):
    try:
        return json.loads(eodhd._http_get(url))
    except Exception:
        return None


def _get_rl(url, tries=7):
    """Rate-limit-aware GET: EODHD's per-minute cap returns HTTP 429. The first fetch
    run silently dropped 22k names because a 429 was swallowed to None with no retry —
    16 workers bursting past the limit looked identical to 'no data'. Here a 429 / 5xx
    backs off and retries (so a rate-limit blip pauses instead of discarding a name);
    only a genuine empty/error response after exhausting retries returns None."""
    for attempt in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read()
                              .decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < tries - 1:
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            return None
        except Exception:
            if attempt < tries - 1:
                time.sleep(1.0 + attempt)
                continue
            return None
    return None


def symbol_rows(exch, key, delisted):
    flag = "&delisted=1" if delisted else ""
    rows = _get(f"https://eodhd.com/api/exchange-symbol-list/{exch}"
                f"?api_token={key}&fmt=json{flag}") or []
    return [r for r in rows if isinstance(r, dict)
            and r.get("Type") == "Common Stock" and r.get("Code")]


def bulk_turnover(exch, key, date):
    """{code: close*volume in LOCAL ccy} for one exchange on one date (one API call)."""
    url = f"https://eodhd.com/api/eod-bulk-last-day/{exch}?api_token={key}&fmt=json&date={date}"
    rows = _get(url)
    out = {}
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            c, px, vol = r.get("code"), r.get("adjusted_close") or r.get("close"), r.get("volume")
            if c and px and vol:
                out[c] = float(px) * float(vol)
    return out


def fx_latest(ccy, key):
    """Latest `ccy` per 1 EUR scalar (for the cheap proxy); EUR -> 1.0."""
    if ccy == "EUR":
        return 1.0
    d = _get(f"https://eodhd.com/api/eod/EUR{ccy}.FOREX?api_token={key}&fmt=json&from=2026-01-01")
    if isinstance(d, list) and d:
        return float(d[-1]["adjusted_close"])
    return None


def fx_series(ccy, key):
    """Daily `ccy` per 1 EUR series (accurate conversion for the full fetch); EUR -> None."""
    if ccy == "EUR":
        return None
    d = _get_rl(f"https://eodhd.com/api/eod/EUR{ccy}.FOREX?api_token={key}&fmt=json&from=2017-01-01")
    if not isinstance(d, list) or not d:
        return None
    return pd.Series([r["adjusted_close"] for r in d],
                     index=pd.to_datetime([r["date"] for r in d])).dropna()


def fetch_cv(sym, key):
    """(adjusted_close, volume) home series from EODHD, or (None, None)."""
    d = _get_rl(f"https://eodhd.com/api/eod/{sym}?api_token={key}&fmt=json&from=2017-01-01")
    if not isinstance(d, list) or not d or "warning" in d[0]:
        return None, None
    idx = pd.to_datetime([r["date"] for r in d])
    close = pd.Series([r["adjusted_close"] for r in d], index=idx).dropna()
    vol = pd.Series([r.get("volume", 0) for r in d], index=idx)
    return (close, vol) if len(close) else (None, None)


# ----------------------------------------------------------------------------- screen
def screen():
    """Cheap turnover pre-screen across all exchanges × sample dates -> _candidates.csv."""
    key = eodhd.api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    fxl = {c: fx_latest(c, key) for c in {c for c, _ in EXCHANGES.values()}}
    print(f"fx: {fxl}", flush=True)

    rows, t0 = [], time.time()
    for exch, (ccy, pence) in EXCHANGES.items():
        fx = fxl.get(ccy) or 1.0
        scale = (0.01 if pence else 1.0) / fx          # local close*vol -> EUR turnover
        active = {r["Code"]: r for r in symbol_rows(exch, key, False)}
        dead = {r["Code"]: r for r in symbol_rows(exch, key, True)}
        allrows = {**dead, **active}                   # active wins on overlap
        peak = {}
        for d in SAMPLE_DATES:
            for c, t in bulk_turnover(exch, key, d).items():
                if c in allrows:
                    e = t * scale
                    if e > peak.get(c, 0.0):
                        peak[c] = e
        kept = 0
        for c, e in peak.items():
            if e < SCREEN_FLOOR:
                continue
            r = allrows[c]
            rows.append(dict(code=c, exch=exch, isin=str(r.get("Isin", "") or ""),
                             name=str(r.get("Name", c)), ccy=ccy, pence=pence,
                             active=int(c in active), proxy_eur=round(e)))
            kept += 1
        print(f"  {exch:5} active={len(active)} dead={len(dead)} "
              f"screened={kept} {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    # dedup by ISIN: prefer the country-primary exchange, then highest proxy turnover.
    df["_pref"] = [int(COUNTRY_PRIMARY.get(i[:2]) == e)
                   for i, e in zip(df["isin"], df["exch"])]
    df = (df.sort_values(["_pref", "proxy_eur"], ascending=False)
            .drop_duplicates(subset="isin", keep="first")
            .drop(columns="_pref"))
    # rows with no ISIN can't be deduped — keep them all (one per code is already unique)
    df.to_csv(OUT / "_candidates.csv", index=False)
    print(f"SCREEN DONE {time.time()-t0:.0f}s: {len(df)} candidates -> {OUT/'_candidates.csv'}",
          flush=True)


# ------------------------------------------------------------------------------ fetch
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


def _process(item, fxs, key, fetch=fetch_cv):
    code, exch, isin, name, ccy, pence, active = item
    sym = f"{code}.{exch}"
    close, vol = fetch(sym, key)
    if close is None:
        return None
    if pence:
        close = close / 100.0
    eur = close if fxs.get(ccy) is None else to_eur(close, fxs[ccy])
    crit = apply_criteria(eur, vol, bool(active))
    if crit is None:
        return None
    country = CC_NAME.get(isin[:2], "—") if isin else "—"
    base = dict(ticker=sym, name=name, sector="Unknown", country=country, currency="EUR",
                slippage_bps=25, local_id="", isin=isin, home=sym, **crit)
    return sym, eur.dropna(), base


def fetch(workers=10):
    """Full-fetch screened candidates, apply criteria -> universe_{prices,meta}.csv.

    Resumable: any names already in universe_meta.csv (a prior, possibly rate-limited
    run) are kept and skipped, so a re-run only pulls the still-missing candidates and
    merges. Genuine criteria-rejects aren't saved, so they get re-tried on resume —
    cheap and correct."""
    key = eodhd.api_key()
    cand = pd.read_csv(OUT / "_candidates.csv")
    cand["isin"] = cand["isin"].fillna("").astype(str)

    done_rows, done_meta = {}, []
    mpath, ppath = OUT / "universe_meta.csv", OUT / "universe_prices.csv"
    if mpath.exists() and ppath.exists():
        dm = pd.read_csv(mpath)
        dp = pd.read_csv(ppath, index_col=0, parse_dates=True)
        done_meta = dm.fillna("").to_dict("records")
        done_rows = {c: dp[c].dropna() for c in dp.columns}
        print(f"resume: {len(done_meta)} names already captured, skipping them", flush=True)
    done = {m["ticker"] for m in done_meta}

    items = [(r.code, r.exch, r.isin, r.name, r.ccy, int(r.pence), int(r.active))
             for r in cand.itertuples(index=False) if f"{r.code}.{r.exch}" not in done]
    fxs = {c: fx_series(c, key) for c in {c for _, _, _, _, c, _, _ in items}}
    print(f"full-fetching {len(items)} remaining candidates with {workers} workers…", flush=True)

    rows, meta, t0 = dict(done_rows), list(done_meta), time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(lambda it: _process(it, fxs, key), items), 1):
            if r:
                sym, eur, m = r
                rows[sym] = eur
                meta.append(m)
            if i % 1000 == 0:
                print(f"  {i}/{len(items)} kept={len(rows)} {time.time()-t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_index().to_csv(OUT / "universe_prices.csv")
    pd.DataFrame(meta).to_csv(OUT / "universe_meta.csv", index=False)
    live = sum(1 for m in meta if not m["delisting_date"])
    print(f"FETCH DONE {time.time()-t0:.0f}s: {len(rows)} names "
          f"({live} live + {len(meta)-live} dead) -> {OUT}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "screen"
    {"screen": screen, "fetch": fetch}[cmd]()
