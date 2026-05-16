from unittest.mock import patch
import pytest
from tools.polymarket_tools import (
    fetch_geopolitical_markets, fetch_earnings_markets,
    parse_probability, filter_by_volume
)

def test_parse_probability_from_outcome_prices():
    market = {"outcomePrices": ["0.67", "0.33"]}
    assert parse_probability(market) == pytest.approx(0.67, abs=0.01)

def test_parse_probability_missing_returns_half():
    market = {}
    assert parse_probability(market) == 0.5

def test_filter_by_volume_removes_low_volume():
    markets = [
        {"volume": "500", "question": "Will X happen?"},
        {"volume": "5000", "question": "Will Y happen?"},
        {"volume": "0", "question": "Will Z happen?"},
    ]
    result = filter_by_volume(markets, min_volume=1000)
    assert len(result) == 1
    assert result[0]["question"] == "Will Y happen?"

def test_fetch_geopolitical_markets_returns_list():
    mock_response = [
        {"id": "1", "question": "Will Ukraine war end?", "outcomePrices": ["0.31", "0.69"],
         "volume": "50000", "endDate": "2026-12-31", "category": "politics"},
    ]
    with patch("tools.polymarket_tools.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None
        result = fetch_geopolitical_markets()
        assert isinstance(result, list)
        if result:
            assert 0.0 <= result[0]["probability"] <= 1.0
