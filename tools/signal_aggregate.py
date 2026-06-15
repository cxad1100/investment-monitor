"""Aggregate the raw corpus into a curated, quantified book the page renders.

Reality of the source: a market-analysis channel gives *direction* (persistently
bullish gold/Bitcoin, bearish the dollar) but almost no price targets. So:

  1. aggregate_assets  — PURE: consolidate per-asset signal across videos, keep
     only recurring + priceable names (asset_resolve drops junk/abstract/sponsor),
     with mention count, bull/bear lean, rationales, source videos.
  2. build_book        — an injected LLM pass ESTIMATES each asset's % move under
     two global scenarios + the scenario probabilities. Numbers are flagged
     `estimated` / `prob_source=llm_estimated` and a `_generated` note is set —
     they are model estimates of a non-rigorous source, never stated targets.

The output validates against tools.scenarios.validate_book, so the existing page
math (expected return → portfolio → scenario grid) renders it unchanged.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
from pathlib import Path

from tools.asset_resolve import resolve_asset, _norm
from tools.scenario_extract import _parse_json

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "local" / "scenarios" / "corpus.jsonl"
BOOK_PATH = ROOT / "local" / "scenarios" / "book.json"
MODEL = "gemma4-abliterated:Q4_K_M"

_DEFAULT_SCENARIOS = [
    {"id": "a", "label": "debasement / risk-off", "prob": 0.6},
    {"id": "b", "label": "normalization / disinflation", "prob": 0.4},
]


# ── Pure aggregation ──────────────────────────────────────────────────────────

def aggregate_assets(corpus_records, *, min_mentions: int = 2) -> list[dict]:
    """Consolidate per-asset directional signal across the corpus.

    Keeps only assets that (a) resolve to a yfinance ticker (drops abstract/junk/
    sponsor names) and (b) are mentioned in >= min_mentions distinct videos.
    """
    agg: dict = {}
    for r in corpus_records:
        vid = r.get("video_id")
        for a in r.get("assets", []):
            name = a.get("name") or ""
            ticker = resolve_asset(name, a.get("ticker_guess"))
            if not ticker:
                continue
            d = agg.setdefault(ticker, {"names": collections.Counter(), "videos": set(),
                                        "bull": 0, "bear": 0, "rationales": []})
            d["names"][name] += 1
            if vid:
                d["videos"].add(vid)
            for s in a.get("scenarios", []):
                lbl = (s.get("label") or "").lower()
                if "bull" in lbl:
                    d["bull"] += 1
                elif "bear" in lbl:
                    d["bear"] += 1
                if s.get("rationale"):
                    d["rationales"].append(s["rationale"])

    out = []
    for ticker, d in agg.items():
        mentions = len(d["videos"])
        if mentions < min_mentions:
            continue
        lean = ("bull" if d["bull"] > d["bear"]
                else "bear" if d["bear"] > d["bull"] else "neutral")
        out.append({"name": d["names"].most_common(1)[0][0], "ticker": ticker,
                    "mentions": mentions, "lean": lean,
                    "videos": sorted(d["videos"]), "rationales": d["rationales"][:5]})
    out.sort(key=lambda x: x["mentions"], reverse=True)
    return out


# ── LLM estimate → book ───────────────────────────────────────────────────────

def _estimate_prompt(assets: list[dict]) -> str:
    lines = []
    for a in assets:
        why = a["rationales"][0][:160] if a["rationales"] else ""
        lines.append(f'- {a["name"]} ({a["ticker"]}): channel lean={a["lean"]}, '
                     f'{a["mentions"]} videos. e.g. "{why}"')
    listing = "\n".join(lines)
    return (
        "You convert a market-analysis channel's qualitative stance into ESTIMATED "
        "scenario returns. Define two global macro scenarios:\n"
        "  a = debasement / risk-off (hard assets up, dollar down)\n"
        "  b = normalization / disinflation (the opposite)\n"
        "For EACH asset below, estimate its return under each scenario as a decimal "
        "(0.5 = +50%, -0.3 = -30%), consistent with the channel's lean. Also give "
        "each scenario a probability (the channel leans toward 'a'). These are "
        "rough estimates, not stated targets.\n\n"
        f"ASSETS:\n{listing}\n\n"
        'Return ONLY JSON of this shape:\n'
        '{"scenarios":[{"id":"a","label":"...","prob":0.6},{"id":"b","label":"...","prob":0.4}],'
        '"assets":[{"name":"<exact name>","a_pct":<dec>,"b_pct":<dec>}]}'
    )


def _find_estimate(g: dict, parsed_assets: list[dict]) -> dict | None:
    """Match an aggregated asset to the LLM's estimate, tolerating the model
    echoing the ticker in the name (e.g. 'Bitcoin (BTC-USD)')."""
    gname, gtick = _norm(g["name"]), _norm(g["ticker"])
    for est in parsed_assets:
        pn = _norm(est.get("name", ""))
        if not pn:
            continue
        if gname and (gname in pn or pn in gname):
            return est
        if gtick and gtick in pn:
            return est
    return None


def build_book(corpus_records, *, generate_fn, min_mentions: int = 2,
               as_of: str | None = None) -> dict:
    """Aggregate the corpus and have generate_fn estimate the numbers; assemble a
    page-ready book with provenance + estimated flags."""
    as_of = as_of or datetime.date.today().isoformat()
    note = (f"llm_estimated from corpus {as_of}; the % moves and probabilities are "
            "model estimates of a non-rigorous source, NOT targets stated in the videos")
    agg = aggregate_assets(corpus_records, min_mentions=min_mentions)
    if not agg:
        return {"_generated": note, "as_of": as_of, "scenarios": [], "assets": []}

    parsed = _parse_json(generate_fn(_estimate_prompt(agg))) or {}
    pscn = parsed.get("scenarios") or []
    if len(pscn) != 2:
        pscn = _DEFAULT_SCENARIOS
    scenarios = [{"id": s.get("id") or sid, "label": s.get("label", ""),
                  "prob": float(s.get("prob", 0.5)), "prob_source": "llm_estimated"}
                 for sid, s in zip(("a", "b"), pscn)]
    total = sum(s["prob"] for s in scenarios) or 1.0
    for s in scenarios:
        s["prob"] = round(s["prob"] / total, 4)
    sa, sb = scenarios[0]["id"], scenarios[1]["id"]

    parsed_assets = parsed.get("assets", [])
    assets = []
    for g in agg:
        est = _find_estimate(g, parsed_assets)
        if not est or est.get("a_pct") is None or est.get("b_pct") is None:
            continue
        assets.append({
            "name": g["name"], "ticker": g["ticker"], "weight": 0.0,
            "outcomes": {sa: {"target_pct": float(est["a_pct"]), "estimated": True},
                         sb: {"target_pct": float(est["b_pct"]), "estimated": True}},
            "source": {"mentions": g["mentions"], "lean": g["lean"], "videos": g["videos"]},
        })
    n = len(assets)
    if n:
        w = round(1.0 / n, 4)
        for a in assets:
            a["weight"] = w
        assets[-1]["weight"] = round(1.0 - w * (n - 1), 4)

    return {"_generated": note, "as_of": as_of, "scenarios": scenarios, "assets": assets}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") in ("ok", "extract_failed"):
            out.append(r)
    # latest record per video wins (retries append fresh rows)
    latest = {r.get("video_id"): r for r in out}
    return list(latest.values())


def main():
    ap = argparse.ArgumentParser(description="Aggregate corpus into an estimated book.json")
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true", help="print, don't write book.json")
    args = ap.parse_args()

    records = _load_corpus(CORPUS_PATH)
    print(f"corpus: {len(records)} usable records")
    agg = aggregate_assets(records, min_mentions=args.min_mentions)
    print(f"recurring priceable assets (>= {args.min_mentions} videos): "
          + ", ".join(f"{a['name']}({a['mentions']},{a['lean']})" for a in agg) or "(none)")

    from tools import ollama_client
    gen = lambda p: ollama_client.generate(p, args.model, temperature=0.2)
    book = build_book(records, generate_fn=gen, min_mentions=args.min_mentions)

    print(f"scenarios: {[(s['id'], s['label'], s['prob']) for s in book['scenarios']]}")
    for a in book["assets"]:
        o = a["outcomes"]
        print(f"  {a['name']:<10} w={a['weight']:.2f} "
              + " ".join(f"{k}={v['target_pct']:+.0%}" for k, v in o.items()))
    if args.dry_run:
        print("(dry-run; book.json not written)")
        return
    if BOOK_PATH.exists():
        BOOK_PATH.rename(BOOK_PATH.with_suffix(".prev.json"))
    BOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOK_PATH.write_text(json.dumps(book, indent=2, ensure_ascii=False))
    print(f"wrote {BOOK_PATH}")


if __name__ == "__main__":
    main()
