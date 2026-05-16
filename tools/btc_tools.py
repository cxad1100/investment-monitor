"""BTC as macro signal: liquidity proxy and anti-establishment sentiment indicator."""

import warnings
import yfinance as yf


def classify_btc_regime(btc_4w: float, gold_4w: float) -> str:
    """Classify BTC macro regime from 4-week returns."""
    if btc_4w < -10:
        return "risk_off"
    if btc_4w > gold_4w + 5 and btc_4w > 0:
        return "risk_on"
    if gold_4w > btc_4w + 5 and gold_4w > 3:
        return "safe_haven"
    return "neutral"


def fetch_btc_signal(period: str = "3mo") -> dict:
    """Fetch BTC and gold price trends, classify regime, and derive asset impacts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            btc_hist = yf.Ticker("BTC-USD").history(period=period)["Close"].dropna()
            gold_hist = yf.Ticker("GC=F").history(period=period)["Close"].dropna()
            spy_hist = yf.Ticker("SPY").history(period=period)["Close"].dropna()
        except Exception as e:
            return {"error": str(e), "regime": "neutral", "liquidity_signal": "neutral",
                    "btc_price": 0, "interpretation": "Data fetch failed"}

        def pct_change(series, n):
            if len(series) < n + 1:
                return 0.0
            return float((series.iloc[-1] / series.iloc[-n] - 1) * 100)

        btc_1w = pct_change(btc_hist, 5)
        btc_4w = pct_change(btc_hist, 20)
        gold_4w = pct_change(gold_hist, 20)
        spy_4w = pct_change(spy_hist, 20)

        regime = classify_btc_regime(btc_4w, gold_4w)

        regime_descriptions = {
            "risk_on": (
                "positive",
                f"BTC {btc_4w:+.1f}% 4W outperforming gold {gold_4w:+.1f}%. Institutional risk appetite strong.",
            ),
            "safe_haven": (
                "negative",
                f"Gold {gold_4w:+.1f}% outperforming BTC {btc_4w:+.1f}%. Safe haven rotation.",
            ),
            "risk_off": (
                "negative",
                f"BTC {btc_4w:+.1f}% 4W. Sharp de-risking: liquidity contracting.",
            ),
            "neutral": (
                "neutral",
                f"BTC {btc_4w:+.1f}% 4W, gold {gold_4w:+.1f}% 4W. No strong macro signal.",
            ),
        }

        liquidity_signal, interpretation = regime_descriptions[regime]

        impact_map = {
            "risk_on": {"growth_tech": "tailwind", "financials": "tailwind", "defensives": "headwind", "gold_etfs": "headwind"},
            "safe_haven": {"growth_tech": "headwind", "gold_etfs": "tailwind", "utilities": "tailwind", "defensives": "tailwind"},
            "risk_off": {"growth_tech": "headwind", "speculative": "headwind", "defensives": "tailwind"},
            "neutral": {},
        }

        return {
            "btc_price": round(float(btc_hist.iloc[-1]), 2),
            "btc_1w_return_pct": round(btc_1w, 2),
            "btc_4w_return_pct": round(btc_4w, 2),
            "gold_4w_return_pct": round(gold_4w, 2),
            "spy_4w_return_pct": round(spy_4w, 2),
            "regime": regime,
            "liquidity_signal": liquidity_signal,
            "interpretation": interpretation,
            "asset_impacts": impact_map[regime],
        }
