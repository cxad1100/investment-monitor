"""Commodity futures signal via yfinance. CL=F crude, GC=F gold, NG=F gas, HG=F copper."""

import warnings
import yfinance as yf

FUTURES_TICKERS = {
    "CL=F": {"name": "Crude Oil", "sector_positive": ["Energy"], "sector_negative": ["Airlines", "Consumer Discretionary"]},
    "GC=F": {"name": "Gold", "sector_positive": ["Materials"], "sector_negative": []},
    "NG=F": {"name": "Natural Gas", "sector_positive": ["Energy", "Utilities"], "sector_negative": ["Industrials"]},
    "HG=F": {"name": "Copper", "sector_positive": ["Materials", "Industrials"], "sector_negative": []},
}


def fetch_futures_signal(period: str = "3mo") -> dict:
    """Fetch commodity futures trends and derive sector tailwinds/headwinds."""
    futures_data = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker, meta in FUTURES_TICKERS.items():
            try:
                hist = yf.Ticker(ticker).history(period=period)["Close"].dropna()
                if len(hist) < 10:
                    continue
                ret_1m = float((hist.iloc[-1] / hist.iloc[-min(21, len(hist))] - 1) * 100)
                ret_3m = float((hist.iloc[-1] / hist.iloc[0] - 1) * 100)
                trend = "rising" if ret_1m > 3 else ("falling" if ret_1m < -3 else "flat")
                futures_data[ticker] = {
                    "name": meta["name"],
                    "price": round(float(hist.iloc[-1]), 2),
                    "return_1m_pct": round(ret_1m, 2),
                    "return_3m_pct": round(ret_3m, 2),
                    "trend": trend,
                    "sector_positive": meta["sector_positive"],
                    "sector_negative": meta["sector_negative"],
                }
            except Exception as e:
                futures_data[ticker] = {"name": meta["name"], "error": str(e)}

    tailwinds = set()
    headwinds = set()
    commodity_summary = []

    for ticker, data in futures_data.items():
        if "error" in data:
            continue
        trend = data["trend"]
        if trend == "rising":
            tailwinds.update(data.get("sector_positive", []))
            headwinds.update(data.get("sector_negative", []))
        elif trend == "falling":
            tailwinds.update(data.get("sector_negative", []))
            headwinds.update(data.get("sector_positive", []))
        commodity_summary.append(f"{data['name']}: {data['return_1m_pct']:+.1f}% 1M ({trend})")

    return {
        "futures": futures_data,
        "sector_tailwinds": list(tailwinds - headwinds),
        "sector_headwinds": list(headwinds - tailwinds),
        "summary": "; ".join(commodity_summary),
    }
