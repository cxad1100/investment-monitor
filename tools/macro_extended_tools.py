"""Extended macro signals: bond yields, currencies, sector ETFs, commodities."""

import warnings
import yfinance as yf
import numpy as np

# ── Bond yield tickers ───────────────────────────────────────────────────────
BOND_TICKERS = {
    "3M":  "^IRX",
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

# ── Currency pairs ───────────────────────────────────────────────────────────
CURRENCY_TICKERS = {
    "DXY":    "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "USDCNY": "USDCNY=X",
    "USDCHF": "USDCHF=X",
}

# ── Sector ETFs ──────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Energy":                  "XLE",
    "Financials":              "XLF",
    "Healthcare":              "XLV",
    "Industrials":             "XLI",
    "Consumer Staples":        "XLP",
    "Utilities":               "XLU",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Communication Services":  "XLC",
    "Consumer Discretionary":  "XLY",
}

# ── Extended commodities ─────────────────────────────────────────────────────
COMMODITY_EXTENDED = {
    "silver":    "SI=F",
    "platinum":  "PL=F",
    "wheat":     "ZW=F",
    "corn":      "ZC=F",
    "soybeans":  "ZS=F",
    "lumber":    "LBS=F",
}


def _returns(hist, periods: dict[str, int]) -> dict[str, float | None]:
    """Compute percentage returns over multiple lookback windows."""
    closes = hist["Close"].dropna() if not hist.empty else None
    out = {}
    for label, days in periods.items():
        if closes is not None and len(closes) >= days:
            ret = (closes.iloc[-1] / closes.iloc[-days] - 1) * 100
            out[label] = round(float(ret), 2)
        else:
            out[label] = None
    return out


def _trend(val_now, val_month_ago) -> str:
    if val_now is None or val_month_ago is None:
        return "unknown"
    if val_now > val_month_ago * 1.01:
        return "rising"
    if val_now < val_month_ago * 0.99:
        return "falling"
    return "stable"


def fetch_bond_yields() -> dict:
    """Yield curve: 3M / 5Y / 10Y / 30Y levels + slope signals."""
    yields = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, sym in BOND_TICKERS.items():
            try:
                hist = yf.Ticker(sym).history(period="3mo")
                if hist.empty:
                    continue
                cur = float(hist["Close"].iloc[-1])
                month_ago = float(hist["Close"].iloc[-20]) if len(hist) >= 20 else cur
                yields[label] = {
                    "yield_pct": round(cur, 3),
                    "month_ago": round(month_ago, 3),
                    "change_1m_bp": round((cur - month_ago) * 100, 1),
                    "trend": _trend(cur, month_ago),
                }
            except Exception:
                pass

    # Yield curve analysis
    y3m  = (yields.get("3M")  or {}).get("yield_pct")
    y5y  = (yields.get("5Y")  or {}).get("yield_pct")
    y10y = (yields.get("10Y") or {}).get("yield_pct")
    y30y = (yields.get("30Y") or {}).get("yield_pct")

    curve_10y_3m = round(y10y - y3m, 3) if y10y and y3m else None
    curve_10y_5y = round(y10y - y5y, 3) if y10y and y5y else None

    if curve_10y_3m is not None:
        if curve_10y_3m < -0.5:
            curve_regime = "deeply_inverted"
        elif curve_10y_3m < 0:
            curve_regime = "inverted"
        elif curve_10y_3m < 0.5:
            curve_regime = "flat"
        else:
            curve_regime = "normal"
    else:
        curve_regime = "unknown"

    return {
        "yields": yields,
        "curve_10y_3m": curve_10y_3m,
        "curve_10y_5y": curve_10y_5y,
        "curve_regime": curve_regime,
        "summary": (
            f"10Y {y10y:.2f}% | 3M {y3m:.2f}% | spread {curve_10y_3m:+.2f}% ({curve_regime})"
            if y10y and y3m else "Bond data unavailable"
        ),
    }


def fetch_currencies() -> dict:
    """Dollar index + major pairs: levels, 1W/1M returns, trend."""
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, sym in CURRENCY_TICKERS.items():
            try:
                hist = yf.Ticker(sym).history(period="3mo")
                if hist.empty:
                    continue
                cur = float(hist["Close"].iloc[-1])
                ret = _returns(hist, {"1W": 5, "1M": 20, "3M": 60})
                result[label] = {
                    "price": round(cur, 4),
                    **ret,
                    "trend": _trend(cur, float(hist["Close"].iloc[-20]) if len(hist) >= 20 else cur),
                }
            except Exception:
                pass

    dxy = result.get("DXY", {})
    dxy_trend = dxy.get("trend", "unknown")
    dxy_1m = dxy.get("1M")

    # Dollar strength interpretation
    if dxy_trend == "rising" and dxy_1m and dxy_1m > 1:
        dollar_signal = "strengthening"
        implication = "Strong dollar headwind for EM equities, commodities, US multinational earnings"
    elif dxy_trend == "falling" and dxy_1m and dxy_1m < -1:
        dollar_signal = "weakening"
        implication = "Weak dollar tailwind for commodities, EM, US exporters"
    else:
        dollar_signal = "stable"
        implication = "Dollar range-bound — neutral cross-asset impact"

    result["_signal"] = {
        "dollar_trend": dollar_signal,
        "implication": implication,
        "dxy_level": dxy.get("price"),
        "dxy_1m_pct": dxy_1m,
    }
    return result


def fetch_sector_etf_performance() -> dict:
    """Sector ETF 1W/1M/3M returns + relative strength ranking."""
    performance = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sector, sym in SECTOR_ETFS.items():
            try:
                hist = yf.Ticker(sym).history(period="6mo")
                if hist.empty:
                    continue
                cur = float(hist["Close"].iloc[-1])
                ret = _returns(hist, {"1W": 5, "1M": 20, "3M": 60, "6M": 120})
                vol = float(hist["Close"].pct_change().dropna().std() * np.sqrt(252) * 100)
                performance[sector] = {
                    "ticker": sym,
                    "price": round(cur, 2),
                    **ret,
                    "volatility_ann": round(vol, 1),
                }
            except Exception:
                pass

    # Rank sectors by 1M return
    ranked_1m = sorted(
        [(s, d["1M"]) for s, d in performance.items() if d.get("1M") is not None],
        key=lambda x: x[1], reverse=True,
    )
    leaders    = [s for s, _ in ranked_1m[:3]]
    laggards   = [s for s, _ in ranked_1m[-3:]]

    return {
        "by_sector": performance,
        "leaders_1m": leaders,
        "laggards_1m": laggards,
        "ranked_1m": ranked_1m,
    }


def fetch_extended_commodities() -> dict:
    """Silver, platinum, wheat, corn, soybeans, lumber — price + returns."""
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, sym in COMMODITY_EXTENDED.items():
            try:
                hist = yf.Ticker(sym).history(period="3mo")
                if hist.empty:
                    continue
                cur = float(hist["Close"].iloc[-1])
                ret = _returns(hist, {"1W": 5, "1M": 20, "3M": 60})
                result[name] = {
                    "price": round(cur, 2),
                    **ret,
                    "trend": _trend(cur, float(hist["Close"].iloc[-20]) if len(hist) >= 20 else cur),
                }
            except Exception:
                pass
    return result
