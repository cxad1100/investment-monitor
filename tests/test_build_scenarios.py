"""Page-layer tests: pure compute_view (the spreadsheet through the page) and
that build() renders without I/O. No network — current prices are injected.
"""
import build_scenarios_report as S


# Spreadsheet-faithful: both assets use target *prices* anchored to current price.
def _book():
    return {
        "scenarios": [
            {"id": "a", "label": "soft landing", "prob": 0.9, "prob_source": "manual"},
            {"id": "b", "label": "recession", "prob": 0.1, "prob_source": "manual"},
        ],
        "assets": [
            {"name": "asset 1", "ticker": "AAA", "weight": 0.53,
             "outcomes": {"a": {"target_price": 200}, "b": {"target_price": 65}}},
            {"name": "asset 2", "ticker": "BBB", "weight": 0.47,
             "outcomes": {"a": {"target_price": 4400}, "b": {"target_price": 6900}}},
        ],
    }


PRICES = {"AAA": 95.0, "BBB": 5090.0}


def test_compute_view_matches_spreadsheet():
    d = S.compute_view(_book(), PRICES)
    assert abs(d["asset_exp"]["asset 1"] - 0.963158) < 1e-4
    assert abs(d["asset_exp"]["asset 2"] - (-0.086444)) < 1e-4
    assert abs(d["portfolio_exp"] - 0.469845) < 1e-4
    assert abs(d["grid"]["a"] - 0.522076) < 1e-4
    assert abs(d["grid"]["b"] - 0.0) < 1e-3
    assert d["problems"] == []
    assert d["skipped"] == []


def test_build_renders_sections_and_portfolio_number():
    d = S.compute_view(_book(), PRICES)
    d["as_of"] = "2026-06-14 12:00"
    d["corpus"] = []
    html = S.build(d, public=False)
    assert "<html" in html.lower()
    for token in ("Expected return", "Portfolio", "Scenario"):
        assert token in html
    assert "+47.0%" in html          # portfolio expected return, formatted by pct()


def test_build_handles_missing_book():
    d = {"book": None, "as_of": "now", "corpus": []}
    html = S.build(d, public=False)
    assert "No book" in html or "no book" in html


def _pairs_book():
    return {"_generated": "est", "as_of": "2026-06-15", "pairs": [
        {"id": "1", "title": "Monetary regime", "scenarios": [
            {"id": "a", "label": "Debasement", "meaning": "dollar down hard assets up",
             "prob": 0.6, "prob_source": "llm_estimated"},
            {"id": "b", "label": "Disinflation", "meaning": "inflation cools bonds win",
             "prob": 0.4, "prob_source": "llm_estimated"}],
         "assets": [
            {"name": "Gold", "ticker": "GC=F", "weight": 0.5, "winner": "a",
             "outcomes": {"a": {"target_pct": 0.8}, "b": {"target_pct": -0.15}}},
            {"name": "Long Bonds", "ticker": "TLT", "weight": 0.5, "winner": "b",
             "outcomes": {"a": {"target_pct": -0.15}, "b": {"target_pct": 0.25}}}]}]}


def test_build_renders_barbell_pairs():
    d = S.view_from_book(_pairs_book())
    d["as_of"] = "t"
    html = S.build(d, public=False)
    assert "Monetary regime" in html
    assert "Debasement" in html and "Disinflation" in html
    assert "dollar down hard assets up" in html          # 50-word meaning rendered
    assert "Gold" in html and "Long Bonds" in html        # one winner per scenario
    assert "+32.5%" in html                               # grid scenario a = .5*.8+.5*(-.15)
    assert "Numbers are LLM estimates" in html


def test_compute_view_flags_skipped_unpriced_asset():
    # asset 2 pct-based (no price needed); asset 1 price-based with no price -> skipped
    book = _book()
    book["assets"][1]["outcomes"] = {"a": {"target_pct": -0.14}, "b": {"target_pct": 0.36}}
    d = S.compute_view(book, {})        # no price for AAA
    assert "asset 1" in d["skipped"]
    assert "asset 1" not in d["asset_exp"]
    assert "asset 2" in d["asset_exp"]
