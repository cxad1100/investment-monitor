"""FRED API data fetching tools."""

import json
from datetime import datetime, timedelta
from fredapi import Fred
import pandas as pd
from config import FRED_API_KEY, FRED_SERIES


def fetch_fred_series(series_ids: list[str] | None = None, lookback_years: int = 3) -> dict:
    """Fetch FRED macro time series. Returns dict of series_id → latest values."""
    if series_ids is None:
        series_ids = list(FRED_SERIES.keys())

    fred = Fred(api_key=FRED_API_KEY)
    start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

    result = {}
    for sid in series_ids:
        try:
            series = fred.get_series(sid, observation_start=start)
            series = series.dropna()
            if series.empty:
                continue
            result[sid] = {
                "name": FRED_SERIES.get(sid, sid),
                "latest": round(float(series.iloc[-1]), 4),
                "prev_year": round(float(series.iloc[-min(12, len(series))]), 4),
                "prev_month": round(float(series.iloc[-min(2, len(series))]), 4),
                "trend": "rising" if series.iloc[-1] > series.iloc[-min(6, len(series))] else "falling",
                "unit": "percent",
            }
        except Exception as e:
            result[sid] = {"error": str(e)}

    return result


def classify_regime(fred_data: dict) -> dict:
    """Rule-based regime classification from FRED indicators.

    Returns regime label and risk assessment.
    """
    try:
        fed_funds = fred_data.get("FEDFUNDS", {}).get("latest", 2.0)
        cpi = fred_data.get("CPIAUCSL", {}).get("latest", 2.0)
        cpi_prev_year = fred_data.get("CPIAUCSL", {}).get("prev_year", 2.0)
        yield_spread = fred_data.get("T10Y2Y", {}).get("latest", 0.5)
        unemployment = fred_data.get("UNRATE", {}).get("latest", 4.0)
        breakeven = fred_data.get("T10YIE", {}).get("latest", 2.5)

        inflation_high = cpi > 4.0 or breakeven > 3.0
        inflation_rising = cpi > cpi_prev_year * 1.1
        recession_risk = yield_spread < 0 or unemployment > 6.0
        growth_conditions = yield_spread > 0.5 and unemployment < 4.5 and fed_funds < 5.0

        if inflation_high and recession_risk:
            regime = "stagflation"
            risk_level = "high"
        elif inflation_high and inflation_rising:
            regime = "inflation"
            risk_level = "medium"
        elif recession_risk:
            regime = "recession"
            risk_level = "high"
        elif cpi < 1.0 and yield_spread < 0:
            regime = "deflation"
            risk_level = "medium"
        elif growth_conditions:
            regime = "growth"
            risk_level = "low"
        else:
            regime = "growth"
            risk_level = "medium"

        return {
            "regime": regime,
            "risk_level": risk_level,
            "indicators": {
                "fed_funds_rate": fed_funds,
                "cpi": cpi,
                "yield_spread_10y2y": yield_spread,
                "unemployment": unemployment,
                "breakeven_inflation": breakeven,
            },
        }
    except Exception as e:
        return {"regime": "growth", "risk_level": "medium", "error": str(e)}


def fred_data_summary(fred_data: dict) -> str:
    """Human-readable summary of FRED data for agent context."""
    lines = ["FRED Macro Indicators:"]
    for sid, data in fred_data.items():
        if "error" in data:
            lines.append(f"  {sid}: ERROR — {data['error']}")
        else:
            name = data.get("name", sid)
            latest = data.get("latest", "N/A")
            trend = data.get("trend", "")
            lines.append(f"  {name} ({sid}): {latest} ({trend})")
    return "\n".join(lines)
