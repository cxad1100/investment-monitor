import pytest
from fast_scorer import (
    compute_earnings_score, compute_insider_score, compute_macro_score,
    compute_geo_score, compute_fundamentals_score, compute_options_score,
    compute_wsb_short_score, compute_composite_score, score_all_assets, score_to_grade
)

def test_earnings_score_high_prob():
    assert compute_earnings_score({"beat_probability": 0.85}) == pytest.approx(85, abs=2)

def test_earnings_score_missing_defaults_to_50():
    assert compute_earnings_score({}) == pytest.approx(50, abs=2)

def test_insider_score_strong_buy():
    score = compute_insider_score({"net_buy_pct_mktcap": 0.8})
    assert score > 80

def test_insider_score_strong_sell():
    score = compute_insider_score({"net_buy_pct_mktcap": -0.8})
    assert score < 20

def test_macro_score_tailwind():
    macro = {"sector_tailwinds": ["Technology"], "sector_headwinds": []}
    assert compute_macro_score("Technology", macro) == pytest.approx(75, abs=5)

def test_macro_score_headwind():
    macro = {"sector_tailwinds": [], "sector_headwinds": ["Technology"]}
    assert compute_macro_score("Technology", macro) == pytest.approx(25, abs=5)

def test_geo_score_no_events():
    score = compute_geo_score("AAPL", {"sector": "Technology", "region": "US"}, [])
    assert score == pytest.approx(50, abs=5)

def test_geo_score_positive_event():
    events = [{
        "probability": 0.8,
        "asset_impacts": [{"sector": "Technology", "direction": "positive", "magnitude": "strong"}]
    }]
    score = compute_geo_score("AAPL", {"sector": "Technology", "region": "US"}, events)
    assert score > 50

def test_options_score_bullish_pcr():
    assert compute_options_score({"put_call_ratio": 0.5}) > 70

def test_options_score_bearish_pcr():
    assert compute_options_score({"put_call_ratio": 1.5}) < 40

def test_composite_score_bounds():
    earnings = {"beat_probability": 0.6}
    insider = {"net_buy_pct_mktcap": 0.2}
    macro = {"sector_tailwinds": ["Technology"], "sector_headwinds": []}
    fundamentals = {"pe_ratio": 25, "return_on_equity": 20, "revenue_growth_yoy": 10}
    price = {"return_1y": 20, "volatility_annualized": 18, "pct_from_52w_high": -5}
    options = {"put_call_ratio": 0.8}
    short = {"short_float_pct": 5}
    wsb = {}
    asset_meta = {"sector": "Technology", "region": "US"}
    price_stats = {"avg_return_1y": 12.0}
    score = compute_composite_score(
        "AAPL", asset_meta, earnings, insider, macro, [], fundamentals, price, price_stats, options, short, wsb
    )
    assert 0 <= score <= 100

def test_score_to_grade_aaa():
    assert score_to_grade(95) == "AAA"

def test_score_to_grade_bbb():
    assert score_to_grade(43) == "BBB"

def test_score_to_grade_low():
    assert score_to_grade(3) == "CC"

def test_score_all_assets_returns_dict():
    signals = {
        "macro": {"sector_tailwinds": ["Technology"], "sector_headwinds": []},
        "price_data": {"AAPL": {"return_1y": 20.0, "volatility_annualized": 18.0, "pct_from_52w_high": -5.0}},
        "price_stats": {"avg_return_1y": 10.0},
        "fundamentals": {"AAPL": {"pe_ratio": 25.0, "return_on_equity": 20.0, "revenue_growth_yoy": 8.0}},
        "earnings": {"AAPL": {"beat_probability": 0.65}},
        "insider": {"AAPL": {"net_buy_pct_mktcap": 0.1}},
        "options": {"AAPL": {"put_call_ratio": 0.8}},
        "short_interest": {"AAPL": {"short_float_pct": 2.0}},
        "wsb": {"ticker_mentions": {}},
        "events": [],
        "universe_map": {"AAPL": {"sector": "Technology", "region": "US", "type": "stock"}},
    }
    result = score_all_assets(signals)
    assert "AAPL" in result
    assert 0 <= result["AAPL"]["score"] <= 100
