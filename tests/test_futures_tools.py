from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from tools.futures_tools import fetch_futures_signal, FUTURES_TICKERS

def test_futures_tickers_defined():
    assert "CL=F" in FUTURES_TICKERS
    assert "GC=F" in FUTURES_TICKERS
    assert "NG=F" in FUTURES_TICKERS
    assert "HG=F" in FUTURES_TICKERS

def test_fetch_futures_signal_structure():
    with patch("tools.futures_tools.yf") as mock_yf:
        closes = pd.Series([100.0, 105.0, 110.0, 108.0, 112.0] * 20)
        mock_hist = MagicMock()
        mock_hist.__getitem__ = lambda self, key: closes
        mock_yf.Ticker.return_value.history.return_value = mock_hist
        result = fetch_futures_signal()
        assert isinstance(result, dict)
        assert "sector_tailwinds" in result
        assert "sector_headwinds" in result
        assert isinstance(result["sector_tailwinds"], list)

def test_fetch_futures_signal_oil_rising_favors_energy():
    with patch("tools.futures_tools.yf") as mock_yf:
        def make_series(start, end):
            import numpy as np
            return pd.Series(list(np.linspace(start, end, 100)))
        futures_data = {
            "CL=F": make_series(70, 90),
            "GC=F": make_series(1900, 1950),
            "NG=F": make_series(3.0, 3.1),
            "HG=F": make_series(4.0, 4.0),
        }
        def ticker_side_effect(sym):
            m = MagicMock()
            series = futures_data.get(sym, make_series(100, 100))
            hist = MagicMock()
            hist.__getitem__ = lambda s, k: series
            m.history.return_value = hist
            return m
        mock_yf.Ticker.side_effect = ticker_side_effect
        result = fetch_futures_signal()
        assert "Energy" in result["sector_tailwinds"]
