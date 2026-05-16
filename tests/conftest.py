import pytest

@pytest.fixture
def sample_asset():
    return {"isin": "US0378331005", "name": "Apple Inc", "yf_ticker": "AAPL",
            "type": "stock", "sector": "Technology", "region": "US"}

@pytest.fixture
def sample_signals():
    return {
        "macro": {"regime": "growth", "sector_tailwinds": ["Technology"], "sector_headwinds": ["Energy"]},
        "price_data": {"AAPL": {"return_1y": 25.0, "volatility_annualized": 22.0, "pct_from_52w_high": -3.0}},
        "fundamentals": {"AAPL": {"pe_ratio": 28.0, "return_on_equity": 35.0, "revenue_growth_yoy": 8.0}},
        "earnings": {"AAPL": {"beat_probability": 0.60, "next_earnings_date": "2026-07-31"}},
        "insider": {"AAPL": {"net_buy_pct_mktcap": 0.001}},
        "options": {"AAPL": {"put_call_ratio": 0.75}},
        "short_interest": {"AAPL": {"short_float_pct": 0.8}},
        "wsb": {"ticker_mentions": {"AAPL": {"mentions_7d": 45, "squeeze_flag": False}}},
        "events": [],
        "universe_map": {"AAPL": {"sector": "Technology", "region": "US", "type": "stock"}},
        "price_stats": {"avg_return_1y": 15.0},
    }

@pytest.fixture
def sample_events():
    return [
        {"id": "A", "description": "Ukraine ceasefire within 30 days", "probability": 0.31,
         "complement_probability": 0.69, "source": "polymarket", "resolution_date": "2026-06-15",
         "asset_impacts": [{"sector": "Defense", "direction": "negative", "magnitude": "strong"}]},
        {"id": "B", "description": "Iran-Israel conflict escalates", "probability": 0.44,
         "complement_probability": 0.56, "source": "polymarket", "resolution_date": "2026-06-01",
         "asset_impacts": [
             {"sector": "Energy", "direction": "positive", "magnitude": "strong"},
             {"sector": "Airlines", "direction": "negative", "magnitude": "strong"},
         ]},
    ]
