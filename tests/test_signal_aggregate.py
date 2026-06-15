"""Corpus → estimated book aggregation. Pure parts TDD'd; the LLM estimate step
uses an injected fake generate_fn (no network)."""
from tools.signal_aggregate import aggregate_assets, build_book
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
