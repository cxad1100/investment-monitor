from unittest.mock import patch
import pytest
from tools.polymarket_tools import (
    fetch_geopolitical_markets, fetch_earnings_markets,
    parse_probability,
)

def test_parse_probability_from_outcome_prices():
    market = {"outcomePrices": ["0.67", "0.33"]}
    assert parse_probability(market) == pytest.approx(0.67, abs=0.01)

def test_parse_probability_missing_returns_half():
    market = {}
    assert parse_probability(market) == 0.5

def test_parse_probability_falls_back_to_last_trade_price():
    market = {"outcomePrices": [], "lastTradePrice": 0.42}
    assert parse_probability(market) == pytest.approx(0.42, abs=0.01)

def _make_mock_get(markets):
    def side_effect(url, params=None, headers=None, timeout=None):
        from unittest.mock import MagicMock
        m = MagicMock()
        offset = int((params or {}).get("offset", 0))
        m.json.return_value = markets if offset == 0 else []
        m.raise_for_status = lambda: None
        return m
    return side_effect

def test_fetch_geopolitical_markets_returns_list():
    mock_markets = [
        {"id": "1", "question": "Will the U.S. invade Iran before 2027?",
         "outcomePrices": ["0.31", "0.69"], "volume": "5000000", "endDate": "2027-01-01"},
    ]
    with patch("tools.polymarket_tools.requests.get", side_effect=_make_mock_get(mock_markets)):
        result = fetch_geopolitical_markets(min_volume=1000)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["probability"] == pytest.approx(0.31, abs=0.01)

def test_fetch_geopolitical_markets_excludes_noise():
    mock_markets = [
        {"id": "1", "question": "Will Raphael Warnock win 2028 nomination?",
         "outcomePrices": ["0.05", "0.95"], "volume": "10000000", "endDate": "2028-01-01"},
        {"id": "2", "question": "Will China invade Taiwan by end of 2026?",
         "outcomePrices": ["0.08", "0.92"], "volume": "5000000", "endDate": "2026-12-31"},
    ]
    with patch("tools.polymarket_tools.requests.get", side_effect=_make_mock_get(mock_markets)):
        result = fetch_geopolitical_markets(min_volume=1000)
        questions = [r["question"] for r in result]
        assert any("Taiwan" in q for q in questions)
        assert not any("Warnock" in q for q in questions)
