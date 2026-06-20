"""Exact Trade-Republic tradeability check, via your own TR login (second oracle).

The OpenFIGI/L&S filter (tools.tradeable) is a ~90% proxy. This queries Trade Republic
directly — the ground truth for *your* account — and writes a separate list so the two
can be merged/compared later (tools.tradeable -> tradeable_tickers.csv; this ->
tr_tradeable_tickers.csv). TR has no "list everything" endpoint, so we look up each
universe ISIN's instrument and keep the ones TR offers (non-empty exchange list / active).

  # one-time: verify the tradeable predicate on known ISINs (4DX yes, Elsight no)
  .venv/bin/python -m tools.tr_tradeable --probe

  # full run -> data/universe/tr_tradeable_tickers.csv (resumable)
  .venv/bin/python -m tools.tr_tradeable

EPHEMERAL BY DESIGN — leaves no reusable login behind:
  * Read-only: calls only login + instrument_details + recv. NEVER any order/transfer
    method. Cannot place a trade or move money.
  * The session lives in memory ONLY (save_cookies=False) — no token is written to disk,
    so every run needs a fresh 2FA. waf_token="awswaf" avoids launching a browser (no
    browser cache).
  * A finally-block ALWAYS runs on exit (success, error, or Ctrl-C): it closes the socket
    and shreds any pytr cookie/credential file (~/.pytr/…). After it finishes there is no
    credential, cookie or session token on disk — only the non-sensitive outputs (a list
    of tradeable tickers + an isin→bool cache).
  * PIN/phone come from a prompt or TR_PHONE / TR_PIN env only — never hard-coded, logged,
    or committed. Don't paste your PIN into chat. It's TR's unofficial API: own-account,
    read-only, but technically against their ToS.
"""
import argparse
import asyncio
import json
import os
import pathlib
import shutil
from getpass import getpass

import pandas as pd
from pytr.api import TradeRepublicApi, TradeRepublicError

ROOT = pathlib.Path(__file__).resolve().parent.parent
UNI_META = ROOT / "data" / "universe" / "universe_meta.csv"
OUT = ROOT / "data" / "universe" / "tr_tradeable_tickers.csv"
RAW = ROOT / "data" / "universe" / "tr_instrument_cache.json"     # {isin: tradeable bool} — not sensitive
PYTR_DIR = pathlib.Path.home() / ".pytr"                          # where pytr could drop cookies/creds

# ISINs to eyeball in --probe: known-tradeable vs known-untradeable from the manual check.
PROBE = {"AU0000095416": "4DMEDICAL (tradeable)", "AU000000ELS4": "Elsight (NOT)",
         "US29479A1088": "Erasca (NOT)", "FR0011341205": "Nanobiotix (tradeable)"}


def login(waf: str = "awswaf"):
    """Fresh interactive login. Nothing persisted (save_cookies=False); returns (api, phone)."""
    phone = os.environ.get("TR_PHONE") or input("TR phone (full, e.g. +39…/+49…): ").strip()
    pin = os.environ.get("TR_PIN") or getpass("TR PIN (hidden): ").strip()
    tr = TradeRepublicApi(phone_no=phone, pin=pin, save_cookies=False, waf_token=waf)
    tr.initiate_weblogin()
    code = input("2FA — approve in the app, or enter the SMS code, then Enter: ").strip()
    tr.complete_weblogin(code)
    print("logged in — session held in memory only (not written to disk)", flush=True)
    return tr, phone


def shred(tr, phone):
    """Close the socket and delete any pytr cookie/credential artifact. Best-effort, always."""
    try:
        if tr is not None:
            c = tr.close()                       # close() is a coroutine in pytr
            if asyncio.iscoroutine(c):
                try:
                    asyncio.run(c)
                except Exception:
                    c.close()
    except Exception:
        pass
    wiped = []
    targets = [PYTR_DIR / "credentials", PYTR_DIR / "cookies.txt"]
    if phone:
        targets.append(PYTR_DIR / f"cookies.{phone}.txt")
    targets += list(PYTR_DIR.glob("cookies*.txt")) if PYTR_DIR.exists() else []
    for t in dict.fromkeys(targets):
        try:
            if t.is_dir():
                shutil.rmtree(t)
                wiped.append(str(t))
            elif t.exists():
                t.unlink()
                wiped.append(str(t))
        except Exception:
            pass
    # if ~/.pytr is now empty, remove it too
    try:
        if PYTR_DIR.exists() and not any(PYTR_DIR.iterdir()):
            PYTR_DIR.rmdir()
            wiped.append(str(PYTR_DIR))
    except Exception:
        pass
    print(f"logged out — shredded: {wiped or 'nothing was persisted'}", flush=True)


def _tradeable(payload) -> bool:
    """TR instrument payload -> is it offered? Non-empty exchange list and not inactive.
    Verify against --probe output; adjust the keys here if TR's schema differs."""
    if not isinstance(payload, dict) or payload.get("errors"):
        return False
    exch = payload.get("exchangeIds") or payload.get("exchanges") or []
    return bool(exch) and payload.get("active", True) is not False


async def _lookup(tr, isins, *, window: int = 20, timeout: float = 8.0, raw=None,
                  cache=None) -> dict:
    out = {}
    for i in range(0, len(isins), window):
        batch = isins[i:i + window]
        subs = {}
        for isin in batch:
            try:
                subs[await tr.instrument_details(isin)] = isin
            except Exception:
                out[isin] = False
        pending = set(subs)
        while pending:
            try:
                sub_id, _sub, payload = await asyncio.wait_for(tr.recv(), timeout)
            except asyncio.TimeoutError:
                for sid in pending:                   # no answer → UNKNOWN (maybe rate-limited),
                    out[subs[sid]] = None             # not cached as untradeable; retried next run
                break
            except TradeRepublicError as e:           # NOT_FOUND etc → not tradeable on TR
                sid = e.subscription_id
                if sid in subs:
                    isin = subs[sid]
                    out[isin] = ({"errors": e.error} if raw is not None else False)
                    if raw is not None:
                        raw[isin] = {"errors": e.error}
                    pending.discard(sid)
                continue
            if sub_id in subs:
                isin = subs[sub_id]
                out[isin] = payload if raw is not None else _tradeable(payload)
                if raw is not None:
                    raw[isin] = payload
                pending.discard(sub_id)
                try:
                    await tr.unsubscribe(sub_id)
                except Exception:
                    pass
        if cache is not None:                         # flush definitive results per batch (resumable)
            cache.update({k: bool(v) for k, v in out.items() if v is not None})
            RAW.write_text(json.dumps(cache))
        print(f"  {min(i + window, len(isins))}/{len(isins)}", flush=True)
    return out


OUT_UNI = ROOT / "data" / "universe" / "tr_universe.csv"   # full TR stock universe (isin,name,country)
# TR market countries to enumerate (ISO codes); extend if a market you trade is missing.
COUNTRIES = ["IT", "DE", "FR", "US", "GB", "NL", "ES", "CH", "SE", "FI", "NO", "DK",
             "BE", "AT", "PT", "IE", "CA", "JP", "HK", "AU", "KR", "TW", "IL", "PL"]


async def _recv_payload(tr, sid, timeout=12.0):
    while True:
        s2, _sub, payload = await asyncio.wait_for(tr.recv(), timeout)
        if s2 == sid:
            try:
                await tr.unsubscribe(sid)
            except Exception:
                pass
            return payload


async def _enumerate(tr, countries, query="", page_size=100, debug=False):
    found = {}
    for cc in countries:
        page, got = 1, 0
        while True:
            try:
                sid = await tr.search(query, asset_type="stock", page=page,
                                      page_size=page_size, filter_country=cc)
                payload = await _recv_payload(tr, sid)
            except Exception as e:
                print(f"  {cc} page {page}: {repr(e)[:70]}", flush=True)
                break
            results = payload.get("results") if isinstance(payload, dict) else None
            if debug and page == 1:
                print(f"  [debug {cc}] payload keys={list(payload)[:6] if isinstance(payload,dict) else payload}; "
                      f"first result={json.dumps(results[0], indent=0)[:300] if results else results}", flush=True)
            if not results:
                break
            for r in results:
                iz = r.get("isin")
                if iz:
                    found[iz] = {"name": r.get("name") or r.get("shortName") or "", "country": cc}
            got += len(results)
            if len(results) < page_size or page > 80:
                break
            page += 1
        print(f"  {cc}: {got} stocks (running total {len(found)})", flush=True)
    return found


def enumerate_universe(waf, debug=False):
    tr = phone = None
    try:
        tr, phone = login(waf)
        found = asyncio.run(_enumerate(tr, COUNTRIES, debug=debug))
        df = pd.DataFrame([{"isin": k, "name": v["name"], "country": v["country"]}
                           for k, v in found.items()])
        df.to_csv(OUT_UNI, index=False)
        print(f"TR UNIVERSE {len(df)} unique stocks -> {OUT_UNI}", flush=True)
    finally:
        shred(tr, phone)


def probe(waf):
    tr = phone = None
    try:
        tr, phone = login(waf)
        raw = {}
        asyncio.run(_lookup(tr, list(PROBE), window=len(PROBE), timeout=12.0, raw=raw))
        for isin, label in PROBE.items():
            p = raw.get(isin)
            verdict = _tradeable(p) if isinstance(p, dict) else False
            print(f"\n=== {label} [{isin}] -> tradeable={verdict} ===")
            print(json.dumps(p, indent=2)[:1200] if isinstance(p, dict) else p)
    finally:
        shred(tr, phone)


def build(waf):
    tr = phone = None
    try:
        tr, phone = login(waf)
        uni = pd.read_csv(UNI_META)
        uni["isin"] = uni["isin"].fillna("").astype(str)
        live = uni[uni["delisting_date"].isna()]            # only LIVE names; dead are kept
        isins = [v for v in live["isin"].unique() if v]     # by keep_tradeable regardless (survivorship)
        cache = json.loads(RAW.read_text()) if RAW.exists() else {}
        todo = [v for v in isins if v not in cache]
        print(f"querying {len(todo)} live ISINs ({len(cache)} cached) of {len(isins)} live "
              f"(dead names skipped, kept as graveyard)…", flush=True)

        res = asyncio.run(_lookup(tr, todo, cache=cache))   # cache flushed per batch
        cache.update({k: bool(v) for k, v in res.items() if v is not None})  # skip UNKNOWNs
        RAW.write_text(json.dumps(cache))

        unknown = sum(1 for v in res.values() if v is None)
        keep = {r.ticker for r in live.itertuples(index=False) if cache.get(r.isin)}
        pd.Series([r.ticker for r in live.itertuples(index=False) if r.ticker in keep],
                  name="ticker").to_csv(OUT, index=False)
        print(f"TR-TRADEABLE {len(keep)}/{len(uni)} names -> {OUT}"
              + (f"  ({unknown} unknown — re-run to retry them)" if unknown else ""), flush=True)
    finally:
        shred(tr, phone)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="print raw payloads for known ISINs")
    ap.add_argument("--enumerate", action="store_true",
                    help="enumerate TR's full tradeable stock universe -> tr_universe.csv")
    ap.add_argument("--debug", action="store_true", help="print the first search payload (schema check)")
    ap.add_argument("--waf", default="awswaf", choices=["awswaf", "playwright"],
                    help="WAF token method; awswaf avoids launching a browser (default)")
    a = ap.parse_args()
    if a.enumerate:
        enumerate_universe(a.waf, debug=a.debug)
    elif a.probe:
        probe(a.waf)
    else:
        build(a.waf)
