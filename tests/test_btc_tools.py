from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from tools.btc_tools import fetch_btc_signal, classify_btc_regime

def make_price_series(start, end, n=90):
    return pd.Series(list(np.linspace(start, end, n)))

def test_classify_btc_regime_risk_on():
    regime = classify_btc_regime(btc_4w=15.0, gold_4w=5.0)
    assert regime == "risk_on"

def test_classify_btc_regime_safe_haven():
    regime = classify_btc_regime(btc_4w=2.0, gold_4w=8.0)
    assert regime == "safe_haven"

def test_classify_btc_regime_risk_off():
    regime = classify_btc_regime(btc_4w=-15.0, gold_4w=3.0)
    assert regime == "risk_off"

def test_classify_btc_regime_neutral():
    regime = classify_btc_regime(btc_4w=1.0, gold_4w=1.5)
    assert regime == "neutral"

def test_fetch_btc_signal_structure():
    with patch("tools.btc_tools.yf") as mock_yf:
        def ticker_side(sym):
            m = MagicMock()
            closes = make_price_series(50000 if "BTC" in sym else 1900, 55000 if "BTC" in sym else 2000)
            hist = MagicMock()
            hist.__getitem__ = lambda s, k: closes
            hist.dropna.return_value = closes
            m.history.return_value = hist
            return m
        mock_yf.Ticker.side_effect = ticker_side
        result = fetch_btc_signal()
        assert "btc_price" in result
        assert "regime" in result
        assert "liquidity_signal" in result
        assert result["regime"] in ("risk_on", "safe_haven", "risk_off", "neutral")
