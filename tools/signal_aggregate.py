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
# Two hard-money assets on both sides isn't a complement (they rise/fall together).
_HARD = {"GC=F", "BTC-USD", "SI=F", "ETH-USD", "PL=F", "PA=F"}

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


# ── Barbell scenario-couples (a|b, c|d, …), one winner asset per scenario ──────

def _clip_words(text: str, n: int) -> str:
    return " ".join((text or "").split()[:n])


def _macro_theses(corpus_records, limit: int = 24) -> list[str]:
    out = []
    for r in corpus_records:
        for m in r.get("macro_theses", []):
            t = (m.get("thesis") or "").strip()
            if t:
                out.append(f'{m.get("direction", "?")}: {t[:160]}')
            if len(out) >= limit:
                return out
    return out


def _events_prompt(assets: list[dict], macro: list[str], today: str) -> str:
    """Stage 1: extract concrete, still-UNRESOLVED future-event predictions."""
    alist = ", ".join(f'{a["name"]}({a["lean"]})' for a in assets) or "(none recurring)"
    mlist = "\n".join(f"- {m}" for m in macro) or "(none)"
    return (
        f"Today is {today}. Extract the channel's concrete FUTURE-EVENT predictions — "
        "specific things it claims WILL (or won't) happen that are STILL UNRESOLVED as "
        "of today and could go either way (e.g. 'oil spikes above $100 on a Hormuz "
        "shock', 'the Fed cuts rates this year', 'Bitcoin breaks its all-time high').\n"
        "CRITICAL: the corpus includes OLD videos whose predicted events have ALREADY "
        "happened or are now in the past. Use those only as background context — do NOT "
        "return them as live predictions. Every event you return must be decidable in "
        "the FUTURE (after today) and not yet resolved. Each must be a CRISP, specific "
        "prediction with a clear yes/no outcome (a price level, a policy, a dated move) "
        "— NOT a worldview or state-of-the-world statement ('we live in a hard-money "
        "world'). No vague vibes.\n"
        "Give up to 6. For EACH: a one-line event, a probability it happens (0-1, the "
        'channel\'s implied conviction), a <=50-word meaning, and "future": true.\n\n'
        f"Recurring assets: {alist}\n"
        f"Claims / theses:\n{mlist}\n\n"
        'Return ONLY JSON: {"events":[{"event":"...","prob":0.6,"meaning":"<=50 words","future":true}]}'
    )


def _asset_prompt(event: dict) -> str:
    """Stage 2: for one event, pick the if-happens winner and the complement winner."""
    return (
        f'FUTURE EVENT: "{event["event"]}" (estimated probability {event["prob"]:.0%}).\n'
        "Pick TWO DIFFERENT, ideally negatively-correlated priceable assets:\n"
        "- if_happens: one asset (a single stock, industry/sector ETF, Bitcoin, or a "
        "commodity like gold) that GAINS if this event happens.\n"
        "- if_not: a DIFFERENT asset that GAINS if the event does NOT happen (its "
        "complement) — the natural beneficiary of the OPPOSITE outcome, negatively "
        "correlated with if_happens and from a DIFFERENT asset class. If if_happens is a "
        "hard asset (gold/Bitcoin/silver), if_not must NOT be another hard asset — pick "
        "equities, bonds, the dollar, or a sector that LOSES under the event's thesis. "
        "Never put two hard-money assets on both sides.\n"
        "For each give name, a real ticker, its estimated % gain in its own case "
        "(decimal, 0.3 = +30%), and its % move in the other case (off_pct, a loss).\n\n"
        'Return ONLY JSON: {"if_happens":{"name":"...","ticker":"...","gain_pct":0.3,'
        '"off_pct":-0.1},"if_not":{"name":"...","ticker":"...","gain_pct":0.2,"off_pct":-0.1}}'
    )


def extract_events(corpus_records, *, generate_fn, today: str | None = None,
                   max_events: int = 6) -> list[dict]:
    """Stage 1 — pull still-unresolved future-event predictions. Older videos feed
    context (recent first), but events already resolved as of `today` are dropped."""
    today = today or datetime.date.today().isoformat()
    recent = sorted(corpus_records, key=lambda r: r.get("upload_date") or "", reverse=True)
    agg = aggregate_assets(recent, min_mentions=1)
    macro = _macro_theses(recent, limit=30)
    parsed = _parse_json(generate_fn(_events_prompt(agg, macro, today))) or {}
    out = []
    for e in (parsed.get("events") or [])[:max_events]:
        if e.get("future") is False:           # resolved / past — context only, not a live event
            continue
        ev = (e.get("event") or "").strip()
        if not ev:
            continue
        prob = min(max(float(e.get("prob", 0.5)), 0.05), 0.95)
        out.append({"event": ev, "prob": prob, "meaning": _clip_words(e.get("meaning", ""), 50)})
    return out


def build_event_pairs(events, *, generate_fn) -> list[dict]:
    """Stage 2 — for each event, pick the if-happens winner and the complement
    (if-not) winner; assemble each as a valid mini-book with scenario a = event,
    scenario b = its complement (prob = 1 - P(event))."""
    pairs = []
    for i, e in enumerate(events, start=1):
        parsed = _parse_json(generate_fn(_asset_prompt(e))) or {}
        h, c = parsed.get("if_happens") or {}, parsed.get("if_not") or {}
        ht = resolve_asset(h.get("name", ""), h.get("ticker"))
        ct = resolve_asset(c.get("name", ""), c.get("ticker"))
        if not ht or not ct or ht == ct or (ht in _HARD and ct in _HARD):
            continue  # two different winners, and not both hard-money (no real complement)
        if h.get("gain_pct") is None or c.get("gain_pct") is None:
            continue
        h_gain = float(h["gain_pct"]); h_off = float(h.get("off_pct", -abs(h_gain) * 0.4))
        c_gain = float(c["gain_pct"]); c_off = float(c.get("off_pct", -abs(c_gain) * 0.4))
        pa = round(e["prob"], 4)
        scenarios = [
            {"id": "a", "label": "If it happens", "meaning": e["meaning"],
             "prob": pa, "prob_source": "llm_estimated"},
            {"id": "b", "label": "If it does NOT happen (complement)",
             "meaning": f"The event does not occur; {c.get('name', 'the complement asset')} benefits instead.",
             "prob": round(1.0 - pa, 4), "prob_source": "llm_estimated"},
        ]
        assets = [
            {"name": h.get("name"), "ticker": ht, "weight": 0.5, "winner": "a",
             "outcomes": {"a": {"target_pct": h_gain, "estimated": True},
                          "b": {"target_pct": h_off, "estimated": True}}},
            {"name": c.get("name"), "ticker": ct, "weight": 0.5, "winner": "b",
             "outcomes": {"a": {"target_pct": c_off, "estimated": True},
                          "b": {"target_pct": c_gain, "estimated": True}}},
        ]
        pairs.append({"id": str(i), "title": e["event"], "event": e["event"],
                      "scenarios": scenarios, "assets": assets})
    return pairs


def build_pairs(corpus_records, *, generate_fn, min_mentions: int = 1,
                as_of: str | None = None) -> dict:
    """Two-stage pipeline: extract future-event opinions, then pair each event with
    its complement and a winning asset on each side. Numbers/scenarios are flagged
    estimates; each pair is a valid mini-book the existing page math renders."""
    as_of = as_of or datetime.date.today().isoformat()
    note = (f"llm_estimated from corpus {as_of}; events, complements, % moves and "
            "probabilities are model estimates of a non-rigorous source, NOT stated by the videos")
    events = extract_events(corpus_records, generate_fn=generate_fn)
    return {"_generated": note, "as_of": as_of,
            "pairs": build_event_pairs(events, generate_fn=generate_fn)}


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
    ap.add_argument("--min-mentions", type=int, default=1)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true", help="print, don't write book.json")
    args = ap.parse_args()

    records = _load_corpus(CORPUS_PATH)
    print(f"corpus: {len(records)} usable records")
    agg = aggregate_assets(records, min_mentions=args.min_mentions)
    print("recurring priceable assets: "
          + (", ".join(f"{a['name']}({a['mentions']},{a['lean']})" for a in agg) or "(none)"))

    from tools import ollama_client
    gen = lambda p: ollama_client.generate(p, args.model, temperature=0.2)
    book = build_pairs(records, generate_fn=gen, min_mentions=args.min_mentions)

    print(f"scenario couples: {len(book['pairs'])}")
    for p in book["pairs"]:
        print(f"  [{p['id']}] {p['title']}")
        for s in p["scenarios"]:
            w = next(a for a in p["assets"] if a["winner"] == s["id"])
            print(f"     {s['id']} {s['label']} ({s['prob']:.0%}) -> "
                  f"{w['name']} {w['outcomes'][s['id']]['target_pct']:+.0%}")
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
