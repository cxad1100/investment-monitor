"""Short interest data via yfinance — short float % and days to cover."""

import warnings
import yfinance as yf


def fetch_short_interest(tickers: list[str]) -> dict[str, dict]:
    """Fetch short interest metrics for each ticker."""
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                short_float = info.get("shortPercentOfFloat") or 0.0
                short_float_pct = round(float(short_float) * 100, 2)
                shares_short = info.get("sharesShort") or 0
                avg_volume = info.get("averageVolume") or 1
                days_to_cover = round(shares_short / avg_volume, 1) if avg_volume else 0.0
                if short_float_pct > 25:
                    signal = "very_high_short"
                elif short_float_pct > 15:
                    signal = "high_short"
                elif short_float_pct > 8:
                    signal = "moderate_short"
                else:
                    signal = "low_short"
                results[ticker] = {
                    "short_float_pct": short_float_pct,
                    "days_to_cover": days_to_cover,
                    "signal": signal,
                }
            except Exception as e:
                results[ticker] = {"short_float_pct": 0.0, "days_to_cover": 0.0,
                                   "signal": "unknown", "error": str(e)}
    return results
