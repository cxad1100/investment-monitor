"""Generate the Scenarios HTML report — narrative-driven risk/return.

  python build_scenarios_report.py            # writes local/scenarios.html only

LOCAL-ONLY: third-party channel content, never deployed to public docs/. The
page renders local/scenarios/book.json (curated assets / scenarios / weights /
probabilities) through the pure engine in tools/scenarios.py, anchoring any
target-price outcomes to live current prices. The raw video corpus
(local/scenarios/corpus.jsonl) is the growing feed the book is curated from.

This mirrors the user's 'Quantifying Risk and Return' spreadsheet:
probability-weighted expected return per asset -> portfolio construction ->
per-scenario portfolio return (the risk view).
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

from tools.report_html import pct as _pct, card as _card, page
from tools.scenarios import (
    asset_expected_return, portfolio_expected_return, scenario_grid,
    resolve_returns, validate_book,
)

ROOT = Path(__file__).parent
BOOK_PATH = ROOT / "local" / "scenarios" / "book.json"
CORPUS_PATH = ROOT / "local" / "scenarios" / "corpus.jsonl"
CORPUS_TAIL = 15


# ── Data assembly ─────────────────────────────────────────────────────────────

def compute_view(book: dict, current_prices: dict, corpus: list | None = None) -> dict:
    """Pure: book + live current prices -> everything the page renders.

    Anchors target-price outcomes to current_prices; assets with no usable price
    are dropped from the math and reported in ``skipped``.
    """
    r = resolve_returns(book, current_prices)
    asset_exp = {a: asset_expected_return(r["asset_ret"][a], r["probs"]) for a in r["asset_ret"]}
    return {
        "book": book,
        "problems": validate_book(book),
        "current_prices": current_prices,
        "scenarios": book["scenarios"],
        "assets": book["assets"],
        "probs": r["probs"],
        "weights": r["weights"],
        "asset_ret": r["asset_ret"],
        "asset_exp": asset_exp,
        "portfolio_exp": portfolio_expected_return(r["weights"], asset_exp),
        "grid": scenario_grid(r["weights"], r["asset_ret"]),
        "skipped": r["skipped"],
        "corpus": corpus or [],
    }


def _price_tickers(book: dict) -> set:
    """Tickers needed for target-price outcomes (target_pct needs no price)."""
    out = set()
    for a in book.get("assets", []):
        for o in a.get("outcomes", {}).values():
            if o.get("target_pct") is None and o.get("target_price") is not None and a.get("ticker"):
                out.add(a["ticker"])
    return out


def _last_closes(tickers, force: bool = False) -> dict:
    if not tickers:
        return {}
    try:  # a fetch failure must not crash the page — assets just go unpriced/skipped
        from tools.data_buffer import cached_price_history
        df = cached_price_history(list(tickers), period="5d", force=force)
    except Exception:
        return {}
    closes = {}
    for t in tickers:
        try:
            s = df[t].dropna() if hasattr(df, "columns") else df.dropna()
        except Exception:
            continue
        if len(s):
            closes[t] = float(s.iloc[-1])
    return closes


def _load_corpus(path: Path, tail: int) -> list:
    if not path.exists():
        return []
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs[-tail:]


def gather(force: bool = False) -> dict:
    corpus = _load_corpus(CORPUS_PATH, CORPUS_TAIL)
    if not BOOK_PATH.exists():
        return {"book": None, "corpus": corpus,
                "as_of": datetime.now().strftime("%Y-%m-%d %H:%M")}
    book = json.loads(BOOK_PATH.read_text())
    prices = _last_closes(_price_tickers(book), force=force)
    d = compute_view(book, prices, corpus)
    d["as_of"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return d


# ── Sections ──────────────────────────────────────────────────────────────────

def sec_intro() -> str:
    return """
<div class="note">
<b>Scenarios — narrative-driven risk/return.</b> A market-analysis YouTube channel
supplies the <i>structure of possible outcomes</i> (an asset runs to X or falls to
Y, with a catalyst and a horizon); we anchor those to live prices and quantify them
the way the "Quantifying Risk and Return" sheet does: probability-weighted expected
return per asset → portfolio construction → portfolio return under each scenario.
Probabilities are the analyst's stated odds or a manual estimate today;
<b>prediction-market odds are planned</b>. Source is a single, non-rigorous
channel — directional framing, not financial advice.
</div>"""


def _no_book() -> str:
    return f"""
<div class="note warn">
<b>No book yet.</b> Ingest some videos, then create a curated
<code>local/scenarios/book.json</code> selecting the assets, scenarios,
probabilities and weights to quantify.
<pre>.venv/bin/python -m tools.video_ingest --limit 10   # fills local/scenarios/corpus.jsonl</pre>
The corpus is the raw feed; the book is the active view this page renders.
</div>"""


def sec_book_status(d: dict) -> str:
    if not d["problems"]:
        return ""
    items = "".join(f"<li>{p}</li>" for p in d["problems"])
    return ("<div class='note warn'><b>Book validation warnings</b>"
            f"<ul>{items}</ul>The numbers below ignore any dropped assets.</div>")


def _outcome_target(o: dict) -> str:
    if o.get("target_price") is not None:
        return f"{o['target_price']:g}"
    if o.get("target_pct") is not None:
        return f"{o['target_pct'] * 100:+.0f}% move"
    return "—"


def sec_assets(d: dict) -> str:
    out = ["<h2>Assets &amp; scenario outcomes</h2>"]
    scenarios = d["scenarios"]
    for a in d["assets"]:
        name = a["name"]
        if name not in d["asset_ret"]:
            continue
        cur = d["current_prices"].get(a.get("ticker"))
        cur_txt = f"{cur:,.2f}" if cur is not None else "n/a (pct-based)"
        src = a.get("source", {})
        src_txt = ""
        if src:
            src_txt = (f" · <span class='dim'>from {src.get('video_id','?')} "
                       f"({src.get('upload_date','?')})</span>")
        rows = []
        for s in scenarios:
            sid = s["id"]
            o = a["outcomes"].get(sid, {})
            ret = d["asset_ret"][name].get(sid)
            rows.append(
                f"<tr><td>{s.get('label', sid)} <span class='dim'>({sid})</span></td>"
                f"<td class='num mono'>{_outcome_target(o)}</td>"
                f"<td class='num'>{_pct(ret * 100) if ret is not None else '—'}</td>"
                f"<td class='dim' style='font-size:0.8rem'>{o.get('rationale','')}</td>"
                f"<td class='dim'>{o.get('horizon','')}</td></tr>")
        out.append(
            f"<h3>{name} <span class='dim mono' style='font-size:0.8rem'>"
            f"{a.get('ticker','')}</span></h3>"
            f"<p class='dim'>current {cur_txt} · weight {a['weight'] * 100:.0f}%{src_txt}</p>"
            "<table><tr><th>Scenario</th><th class='num'>Target</th>"
            "<th class='num'>Up/(down)side</th><th>Rationale</th><th>Horizon</th></tr>"
            + "".join(rows) + "</table>")
    if d["skipped"]:
        out.append("<p class='dim'>⚠ excluded (no live price): "
                   + ", ".join(d["skipped"]) + "</p>")
    return "".join(out)


def sec_probabilities(d: dict) -> str:
    rows = "".join(
        f"<tr><td class='mono'>{s['id']}</td><td>{s.get('label','')}</td>"
        f"<td class='num'>{_pct(s['prob'] * 100, signed=False)}</td>"
        f"<td class='dim'>{s.get('prob_source','manual')}</td></tr>"
        for s in d["scenarios"])
    return ("<h2>Scenario probabilities</h2>"
            "<table><tr><th>id</th><th>Scenario</th><th class='num'>Probability</th>"
            "<th>Source</th></tr>" + rows + "</table>"
            "<p class='dim'>Stated = the analyst's own odds; manual = your estimate. "
            "Prediction-market odds (Polymarket / Kalshi) are a planned swap-in.</p>")


def sec_expected(d: dict) -> str:
    scenarios = d["scenarios"]
    head = ("<tr><th>Asset</th>"
            + "".join(f"<th class='num'>{s['id']}</th>" for s in scenarios)
            + "<th class='num'>Expected</th></tr>")
    rows = []
    for name, exp in d["asset_exp"].items():
        cells = "".join(
            f"<td class='num'>{_pct(d['asset_ret'][name][s['id']] * 100)}</td>"
            for s in scenarios)
        rows.append(f"<tr><td>{name}</td>{cells}"
                    f"<td class='num'><b>{_pct(exp * 100)}</b></td></tr>")
    prob_row = ("<tr><td class='dim'>probability</td>"
                + "".join(f"<td class='num dim'>{_pct(d['probs'][s['id']] * 100, signed=False)}</td>"
                          for s in scenarios)
                + "<td></td></tr>")
    return ("<h2>Expected return (probability-weighted)</h2>"
            "<p class='dim'>Each asset's return under every scenario, weighted by the "
            "scenario probabilities: Σ p·r.</p>"
            "<table>" + head + prob_row + "".join(rows) + "</table>")


def sec_portfolio(d: dict) -> str:
    rows = []
    for name, w in d["weights"].items():
        exp = d["asset_exp"][name]
        rows.append(f"<tr><td>{name}</td>"
                    f"<td class='num mono'>{w * 100:.0f}%</td>"
                    f"<td class='num'>{_pct(exp * 100)}</td>"
                    f"<td class='num'>{_pct(w * exp * 100)}</td></tr>")
    total_w = sum(d["weights"].values())
    total = ("<tr><td><b>total</b></td>"
             f"<td class='num mono'>{total_w * 100:.0f}%</td><td></td>"
             f"<td class='num'><b>{_pct(d['portfolio_exp'] * 100)}</b></td></tr>")
    return ("<h2>Portfolio construction</h2>"
            "<table><tr><th>Asset</th><th class='num'>Weight</th>"
            "<th class='num'>Expected</th><th class='num'>Contribution</th></tr>"
            + "".join(rows) + total + "</table>")


def sec_grid(d: dict) -> str:
    cards = [_card(f"{s.get('label', s['id'])} ({s['id']})",
                   _pct(d["grid"].get(s["id"], 0.0) * 100))
             for s in d["scenarios"]]
    return ("<h2>Scenario risk view — portfolio return per scenario</h2>"
            "<p class='dim'>What the whole book returns if each scenario plays out "
            "(Σ weight·return). The spread is the risk: a tight, positive set of "
            "outcomes is a robust book; a big negative tail is the warning.</p>"
            f'<div class="cards">{"".join(cards)}</div>')


def sec_corpus(d: dict) -> str:
    corpus = d.get("corpus") or []
    if not corpus:
        return ("<details><summary>Source video corpus (0)</summary>"
                "<p class='dim'>Empty — run <code>python -m tools.video_ingest</code>.</p></details>")
    rows = []
    for r in reversed(corpus):
        rows.append(f"<tr><td class='mono'>{r.get('upload_date','?')}</td>"
                    f"<td>{(r.get('title') or '')[:80]}</td>"
                    f"<td class='num mono'>{len(r.get('assets', []))}</td>"
                    f"<td class='dim'>{r.get('status','?')}</td></tr>")
    return (f"<details><summary>Source video corpus ({len(corpus)} most recent)</summary>"
            "<table><tr><th>Uploaded</th><th>Title</th><th class='num'>Assets</th>"
            "<th>Status</th></tr>" + "".join(rows) + "</table></details>")


def sec_caveat() -> str:
    return """
<div class="note warn">
<b>Read this as a frame, not a forecast.</b> One non-rigorous channel; qualitative,
forward-looking opinions extracted by a local LLM (which can misread a summary).
Probabilities are subjective until prediction markets are wired in. There is no
backtest here — that's a deliberate later step, once the corpus is large enough to
test stated outcomes against what actually happened. Not financial advice.
</div>"""


def sec_method() -> str:
    return """
<h2>How it works</h2>
<details open><summary>Data flow</summary>
<p>A YouTube channel → <code>vidsum</code> (local transcription + summary) → a local
Ollama pass extracts each asset's scenarios into <code>corpus.jsonl</code> (the raw,
append-only feed). You curate the active <code>book.json</code> — the assets,
shared scenarios, probabilities and weights — from that corpus. This page renders the
book.</p></details>
<details><summary>The math (mirrors the spreadsheet)</summary>
<p>Target prices become returns versus the live current price; target-percent moves
are used directly. Asset expected return = Σ probability·return across scenarios.
Portfolio expected return = Σ weight·asset-expected. The scenario grid shows
Σ weight·return under each scenario — the risk view.</p></details>
<details><summary>Local-only, by design</summary>
<p>This page is never published to the public site: it summarises third-party content
and is experimental. It is served only by the local dashboard.</p></details>
"""


# ── Assembly ──────────────────────────────────────────────────────────────────

def build(d: dict, public: bool = False) -> str:
    now = d.get("as_of") or datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Scenarios" + ("" if public else " — private")
    head = [f"<h1>{title}</h1>",
            f"<p class='dim'>generated {now} · local-only · "
            f"<a href='report.html'>← portfolio monitor</a></p>",
            sec_intro()]
    if d.get("book") is None:
        body = "".join(head + [_no_book(), sec_corpus(d), sec_method()])
        return page(title, body)
    body = "".join(head + [
        sec_book_status(d),
        sec_assets(d),
        sec_probabilities(d),
        sec_expected(d),
        sec_portfolio(d),
        sec_grid(d),
        sec_corpus(d),
        sec_caveat(),
        sec_method(),
    ])
    return page(title, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the local report")
    ap.add_argument("--refresh", action="store_true", help="force price re-fetch")
    args = ap.parse_args()

    print("gathering scenario data…")
    d = gather(force=args.refresh)
    local = ROOT / "local" / "scenarios.html"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(build(d, public=False))
    print(f"wrote {local}  (local-only — no docs/ build)")
    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
