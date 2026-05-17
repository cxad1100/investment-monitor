"""Market sentiment: VIX, CNN Fear & Greed, credit spreads."""

import requests
import yfinance as yf

CNN_FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CNN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
}

VIX_TICKER = "^VIX"
CREDIT_TICKERS = {"hyg": "HYG", "lqd": "LQD", "tlt": "TLT"}


def _classify_vix(level: float) -> str:
    if level < 15:
        return "complacency"
    if level < 20:
        return "calm"
    if level < 25:
        return "elevated"
    if level < 30:
        return "fear"
    return "extreme_fear"


def _classify_fg(score: float) -> str:
    if score >= 75:
        return "extreme_greed"
    if score >= 55:
        return "greed"
    if score >= 45:
        return "neutral"
    if score >= 25:
        return "fear"
    return "extreme_fear"


def fetch_vix() -> dict:
    try:
        hist = yf.Ticker(VIX_TICKER).history(period="1y")
        if hist.empty:
            return {}
        cur = float(hist["Close"].iloc[-1])
        week_ago = float(hist["Close"].iloc[-5]) if len(hist) >= 5 else cur
        month_ago = float(hist["Close"].iloc[-20]) if len(hist) >= 20 else cur
        low_1y = float(hist["Close"].min())
        high_1y = float(hist["Close"].max())
        trend = "rising" if cur > month_ago * 1.1 else ("falling" if cur < month_ago * 0.9 else "stable")
        return {
            "vix": round(cur, 2),
            "vix_1w_ago": round(week_ago, 2),
            "vix_1m_ago": round(month_ago, 2),
            "vix_1y_low": round(low_1y, 2),
            "vix_1y_high": round(high_1y, 2),
            "trend": trend,
            "regime": _classify_vix(cur),
        }
    except Exception:
        return {}


def fetch_fear_greed() -> dict:
    try:
        r = requests.get(CNN_FG_URL, headers=CNN_HEADERS, timeout=15)
        r.raise_for_status()
        fg = r.json().get("fear_and_greed", {})
        score = float(fg.get("score", 50))
        return {
            "score": round(score, 1),
            "rating": _classify_fg(score),
            "previous_close": round(float(fg.get("previous_close", score)), 1),
            "previous_1_week": round(float(fg.get("previous_1_week", score)), 1),
            "previous_1_month": round(float(fg.get("previous_1_month", score)), 1),
            "timestamp": fg.get("timestamp", ""),
        }
    except Exception:
        return {}


def fetch_credit_spreads() -> dict:
    """HYG/LQD spread proxy: widening = credit stress = risk-off."""
    try:
        prices = {}
        for key, sym in CREDIT_TICKERS.items():
            hist = yf.Ticker(sym).history(period="3mo")
            if hist.empty:
                continue
            cur = float(hist["Close"].iloc[-1])
            month_ago = float(hist["Close"].iloc[-20]) if len(hist) >= 20 else cur
            prices[key] = {"price": round(cur, 2), "return_1m_pct": round((cur - month_ago) / month_ago * 100, 2)}

        # HYG/LQD ratio: falling ratio = credit spread widening = stress
        hyg_p = prices.get("hyg", {}).get("price")
        lqd_p = prices.get("lqd", {}).get("price")
        spread_ratio = round(hyg_p / lqd_p, 4) if hyg_p and lqd_p else None

        hyg_ret = prices.get("hyg", {}).get("return_1m_pct", 0)
        lqd_ret = prices.get("lqd", {}).get("return_1m_pct", 0)
        spread_change = round(hyg_ret - lqd_ret, 2) if hyg_ret is not None else None

        if spread_change is not None:
            if spread_change < -1.5:
                spread_regime = "widening"  # stress
            elif spread_change > 1.5:
                spread_regime = "tightening"  # risk-on
            else:
                spread_regime = "stable"
        else:
            spread_regime = "unknown"

        return {
            "hyg": prices.get("hyg", {}),
            "lqd": prices.get("lqd", {}),
            "tlt": prices.get("tlt", {}),
            "hyg_lqd_ratio": spread_ratio,
            "spread_change_1m": spread_change,
            "spread_regime": spread_regime,
        }
    except Exception:
        return {}


def fetch_market_sentiment() -> dict:
    """Aggregate VIX + Fear/Greed + credit spreads into one sentiment block."""
    vix = fetch_vix()
    fg = fetch_fear_greed()
    spreads = fetch_credit_spreads()

    # Overall regime: combine signals
    vix_regime = vix.get("regime", "calm")
    fg_rating = fg.get("rating", "neutral")
    spread_regime = spreads.get("spread_regime", "stable")

    fear_count = sum([
        vix_regime in {"fear", "extreme_fear"},
        fg_rating in {"fear", "extreme_fear"},
        spread_regime == "widening",
    ])
    greed_count = sum([
        vix_regime == "complacency",
        fg_rating in {"greed", "extreme_greed"},
        spread_regime == "tightening",
    ])

    if fear_count >= 2:
        overall = "risk_off"
    elif greed_count >= 2:
        overall = "risk_on"
    else:
        overall = "neutral"

    return {
        "vix": vix,
        "fear_greed": fg,
        "credit_spreads": spreads,
        "overall_sentiment": overall,
    }
