import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from tools.universe_manager import (
    download_sp500_tickers, fetch_seed_etfs, build_universe,
    validate_ticker, refresh_universe
)

def test_download_sp500_returns_list():
    result = download_sp500_tickers()
    assert isinstance(result, list)
    assert len(result) > 400
    first = result[0]
    assert "yf_ticker" in first
    assert "sector" in first
    assert first["type"] == "stock"
    assert first["region"] == "US"

def test_fetch_seed_etfs_returns_list():
    result = fetch_seed_etfs()
    assert isinstance(result, list)
    assert len(result) > 20
    first = result[0]
    assert first["type"] == "etf"

def test_validate_ticker_valid():
    with patch("tools.universe_manager.yf") as mock_yf:
        mock_info = MagicMock()
        mock_info.last_price = 150.0
        mock_yf.Ticker.return_value.fast_info = mock_info
        assert validate_ticker("AAPL") is True

def test_validate_ticker_invalid():
    with patch("tools.universe_manager.yf") as mock_yf:
        mock_info = MagicMock()
        mock_info.last_price = None
        mock_yf.Ticker.return_value.fast_info = mock_info
        assert validate_ticker("INVALID123") is False

def test_build_universe_deduplicates():
    stocks = [{"yf_ticker": "AAPL", "name": "Apple", "isin": "", "type": "stock", "sector": "Tech", "region": "US"}]
    etfs = [{"yf_ticker": "AAPL", "name": "Apple ETF", "isin": "", "type": "etf", "sector": "Blend", "region": "US"}]
    df = build_universe(stocks, etfs)
    # AAPL should appear only once (dedup on yf_ticker)
    assert len(df[df["yf_ticker"] == "AAPL"]) == 1
