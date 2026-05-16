"""yfinance options chain — put-call ratio and open interest skew per ticker."""

import warnings
import yfinance as yf


def fetch_options_signal(tickers: list[str]) -> dict[str, dict]:
    """Compute put-call ratio and OI skew for each ticker."""
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                expirations = t.options
                if not expirations:
                    results[ticker] = {"put_call_ratio": 1.0, "signal": "neutral"}
                    continue
                expiry = expirations[0] if len(expirations) <= 2 else expirations[1]
                chain = t.option_chain(expiry)
                call_oi = chain.calls["openInterest"].sum()
                put_oi = chain.puts["openInterest"].sum()
                call_vol = chain.calls["volume"].sum()
                put_vol = chain.puts["volume"].sum()
                pcr_oi = float(put_oi / call_oi) if call_oi > 0 else 1.0
                pcr_vol = float(put_vol / call_vol) if call_vol > 0 else 1.0
                pcr = round((pcr_oi + pcr_vol) / 2, 3)
                if pcr < 0.6:
                    signal = "strong_bullish"
                elif pcr < 0.8:
                    signal = "bullish"
                elif pcr > 1.4:
                    signal = "strong_bearish"
                elif pcr > 1.1:
                    signal = "bearish"
                else:
                    signal = "neutral"
                results[ticker] = {
                    "put_call_ratio": pcr,
                    "call_oi": int(call_oi),
                    "put_oi": int(put_oi),
                    "signal": signal,
                }
            except Exception as e:
                results[ticker] = {"put_call_ratio": 1.0, "signal": "neutral", "error": str(e)}
    return results
