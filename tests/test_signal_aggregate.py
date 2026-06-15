"""Corpus → estimated book aggregation. Pure parts TDD'd; the LLM estimate step
uses an injected fake generate_fn (no network)."""
import json

from tools.signal_aggregate import (
    aggregate_assets, build_book, build_pairs, extract_events, build_event_pairs,
)
from tools.scenarios import validate_book


def _corpus():
    return [
        {"video_id": "v1", "upload_date": "20260101", "status": "ok", "assets": [
            {"name": "Bitcoin", "ticker_guess": "BTC",
             "scenarios": [{"label": "bull"}, {"label": "base"}]}]},
        {"video_id": "v2", "upload_date": "20260115", "status": "ok", "assets": [
            {"name": "Bitcoin", "ticker_guess": None, "scenarios": [{"label": "bull"}]},
            {"name": "Gold", "ticker_guess": None, "scenarios": [{"label": "bull"}]}]},
        {"video_id": "v3", "upload_date": "20260201", "status": "ok", "assets": [
            {"name": "Gold", "scenarios": [{"label": "bull"}]},
            {"name": "Seeking Alpha Premium", "scenarios": [{"label": "base"}]},
            {"name": "The Financial Industrial Complex", "scenarios": [{"label": "bear"}]}]},
    ]


def test_aggregate_keeps_recurring_priceable():
    ags = {a["name"]: a for a in aggregate_assets(_corpus(), min_mentions=2)}
    assert ags["Bitcoin"]["ticker"] == "BTC-USD"   # via asset_resolve alias
    assert ags["Gold"]["ticker"] == "GC=F"
    assert ags["Bitcoin"]["mentions"] == 2          # distinct videos, not scenario count
    assert ags["Bitcoin"]["lean"] == "bull"         # 2 bull vs 1 base
    assert set(ags["Bitcoin"]["videos"]) == {"v1", "v2"}


def test_aggregate_drops_junk_and_singletons():
    names = {a["name"] for a in aggregate_assets(_corpus(), min_mentions=2)}
    assert "Seeking Alpha Premium" not in names              # sponsor, single mention
    assert "The Financial Industrial Complex" not in names   # abstract, not priceable


def test_aggregate_min_mentions_filters_all():
    assert aggregate_assets(_corpus(), min_mentions=3) == []


def test_build_book_assembles_valid_estimated_book():
    fake = ('{"scenarios":[{"id":"a","label":"debasement","prob":0.7},'
            '{"id":"b","label":"normalization","prob":0.3}],'
            '"assets":[{"name":"Bitcoin","a_pct":0.6,"b_pct":-0.4},'
            '{"name":"Gold","a_pct":0.25,"b_pct":-0.1}]}')
    book = build_book(_corpus(), generate_fn=lambda p: fake, min_mentions=2)

    assert validate_book(book) == []                          # renders on the page
    assert book["scenarios"][0]["prob_source"] == "llm_estimated"
    assert "_generated" in book                               # provenance banner trigger
    btc = next(a for a in book["assets"] if a["name"] == "Bitcoin")
    assert btc["ticker"] == "BTC-USD"
    assert btc["outcomes"]["a"]["target_pct"] == 0.6
    assert btc["outcomes"]["a"]["estimated"] is True
    assert btc["source"]["mentions"] == 2
    assert abs(sum(a["weight"] for a in book["assets"]) - 1.0) < 1e-9


def test_build_book_matches_llm_name_with_ticker_suffix():
    # gemma tends to echo "Bitcoin (BTC-USD)"; must still match agg's "Bitcoin"
    fake = ('{"scenarios":[{"id":"a","label":"x","prob":0.6},{"id":"b","label":"y","prob":0.4}],'
            '"assets":[{"name":"Bitcoin (BTC-USD)","a_pct":1.5,"b_pct":-0.2},'
            '{"name":"Gold (GC=F)","a_pct":0.8,"b_pct":-0.4}]}')
    book = build_book(_corpus(), generate_fn=lambda p: fake, min_mentions=2)
    assert {a["name"] for a in book["assets"]} == {"Bitcoin", "Gold"}
    btc = next(a for a in book["assets"] if a["name"] == "Bitcoin")
    assert btc["outcomes"]["a"]["target_pct"] == 1.5


def test_build_book_empty_corpus_returns_empty_assets():
    book = build_book([], generate_fn=lambda p: '{"scenarios":[],"assets":[]}', min_mentions=2)
    assert book["assets"] == []


# ── Event pipeline: extract future-event opinions, pair each with its complement ──

_EVENTS = ('{"events":[{"event":"Oil spikes above $100 on a Hormuz shock",'
           '"prob":0.6,"meaning":"war disrupts the strait, a supply emergency, prices surge"}]}')
_ASSETS = ('{"if_happens":{"name":"ExxonMobil","ticker":"XOM","gain_pct":0.3,"off_pct":-0.1},'
           '"if_not":{"name":"Airlines ETF","ticker":"JETS","gain_pct":0.2,"off_pct":-0.1}}')


def _pipeline_fn(prompt):
    return _ASSETS if "Pick TWO DIFFERENT" in prompt else _EVENTS


def test_extract_events_returns_predictions():
    evs = extract_events(_corpus(), generate_fn=lambda p: _EVENTS)
    assert evs[0]["event"].startswith("Oil spikes")
    assert evs[0]["prob"] == 0.6
    assert len(evs[0]["meaning"].split()) <= 50


def test_extract_events_drops_resolved_past_events():
    j = ('{"events":['
         '{"event":"Oil already spiked in Q1 2026","prob":0.9,"meaning":"m","future":false},'
         '{"event":"Bitcoin breaks ATH later this year","prob":0.5,"meaning":"m","future":true}]}')
    evs = extract_events(_corpus(), generate_fn=lambda p: j)
    assert [e["event"] for e in evs] == ["Bitcoin breaks ATH later this year"]


def test_extract_events_clamps_probability():
    evs = extract_events(_corpus(),
                         generate_fn=lambda p: '{"events":[{"event":"x","prob":1.5,"meaning":"m"}]}')
    assert evs[0]["prob"] == 0.95   # clamped into [0.05, 0.95]


def test_build_event_pairs_complement_barbell():
    events = [{"event": "Oil spikes on a Hormuz shock", "prob": 0.6, "meaning": "supply emergency"}]
    p = build_event_pairs(events, generate_fn=lambda _p: _ASSETS)[0]
    assert p["event"].startswith("Oil spikes")
    assert [s["id"] for s in p["scenarios"]] == ["a", "b"]
    assert abs(p["scenarios"][0]["prob"] - 0.6) < 1e-9     # P(event)
    assert abs(p["scenarios"][1]["prob"] - 0.4) < 1e-9     # complement = 1 - P(event)
    xom = next(a for a in p["assets"] if a["ticker"] == "XOM")
    jets = next(a for a in p["assets"] if a["ticker"] == "JETS")
    assert xom["winner"] == "a" and xom["outcomes"]["a"]["target_pct"] == 0.3
    assert xom["outcomes"]["b"]["target_pct"] == -0.1      # if-happens asset loses on complement
    assert jets["winner"] == "b" and jets["outcomes"]["b"]["target_pct"] == 0.2
    assert validate_book(p) == []


def test_build_event_pairs_drops_same_and_unresolved():
    events = [{"event": "X", "prob": 0.5, "meaning": "m"}]
    same = ('{"if_happens":{"name":"Gold","ticker":"GC=F","gain_pct":0.3,"off_pct":-0.1},'
            '"if_not":{"name":"Gold bars","ticker":"GC=F","gain_pct":0.2,"off_pct":-0.1}}')
    assert build_event_pairs(events, generate_fn=lambda p: same) == []     # same ticker both sides
    unres = ('{"if_happens":{"name":"Abstract","ticker":null,"gain_pct":0.3,"off_pct":-0.1},'
             '"if_not":{"name":"Gold","ticker":"GC=F","gain_pct":0.2,"off_pct":-0.1}}')
    assert build_event_pairs(events, generate_fn=lambda p: unres) == []    # unresolvable winner


def test_build_event_pairs_drops_both_hard_money():
    events = [{"event": "X", "prob": 0.6, "meaning": "m"}]
    j = ('{"if_happens":{"name":"Bitcoin","ticker":"BTC-USD","gain_pct":0.5,"off_pct":-0.2},'
         '"if_not":{"name":"Gold","ticker":"GC=F","gain_pct":0.3,"off_pct":-0.1}}')
    assert build_event_pairs(events, generate_fn=lambda p: j) == []   # gold vs BTC is no complement


def test_build_pairs_pipeline_end_to_end():
    book = build_pairs(_corpus(), generate_fn=_pipeline_fn, min_mentions=1)
    assert "_generated" in book
    assert len(book["pairs"]) == 1
    p = book["pairs"][0]
    assert p["title"].startswith("Oil spikes")        # title = the future event
    assert validate_book(p) == []
