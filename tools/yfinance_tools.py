"""yfinance data fetching tools: price history and fundamentals."""

import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def fetch_price_history(tickers: list[str], period: str = "1y") -> dict:
    """Fetch OHLCV price history for a list of tickers.

    Returns dict of ticker → {dates, closes, volumes, returns_1y, volatility}.
    """
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(period=period)
                if hist.empty or len(hist) < 20:
                    continue

                closes = hist["Close"].dropna()
                volumes = hist["Volume"].dropna()
                daily_returns = closes.pct_change().dropna()

                high_52w = float(closes.rolling(min(252, len(closes))).max().iloc[-1])
                low_52w = float(closes.rolling(min(252, len(closes))).min().iloc[-1])
                current_price = float(closes.iloc[-1])
                pct_from_high = (current_price - high_52w) / high_52w * 100

                result[ticker] = {
                    "current_price": round(current_price, 2),
                    "high_52w": round(high_52w, 2),
                    "low_52w": round(low_52w, 2),
                    "pct_from_52w_high": round(pct_from_high, 2),
                    "return_1y": round(float((closes.iloc[-1] / closes.iloc[0] - 1) * 100), 2),
                    "volatility_annualized": round(float(daily_returns.std() * np.sqrt(252) * 100), 2),
                    "avg_volume_30d": int(volumes.tail(30).mean()),
                    "data_points": len(closes),
                }
            except Exception as e:
                result[ticker] = {"error": str(e)}

    return result


def fetch_fundamentals(tickers: list[str]) -> dict:
    """Fetch fundamental data for a list of tickers.

    Returns dict of ticker → key financial metrics.
    """
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info

                eps = info.get("trailingEps") or info.get("forwardEps")
                pe = info.get("trailingPE") or info.get("forwardPE")
                revenue_growth = info.get("revenueGrowth")
                earnings_growth = info.get("earningsGrowth")
                debt_equity = info.get("debtToEquity")
                roe = info.get("returnOnEquity")
                market_cap = info.get("marketCap")
                dividend_yield = info.get("dividendYield")
                free_cashflow = info.get("freeCashflow")
                total_revenue = info.get("totalRevenue")

                # Analyst consensus
                rec_mean = info.get("recommendationMean")
                rec_key = info.get("recommendationKey", "hold")
                target_mean = info.get("targetMeanPrice")
                target_high = info.get("targetHighPrice")
                target_low = info.get("targetLowPrice")
                n_analysts = info.get("numberOfAnalystOpinions", 0)
                forward_eps = info.get("forwardEps")
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")

                upside_pct = None
                if target_mean and current_price and current_price > 0:
                    upside_pct = round((target_mean - current_price) / current_price * 100, 1)

                # Next earnings date from Unix timestamp
                earnings_ts = info.get("earningsTimestamp")
                next_earnings = None
                if earnings_ts:
                    try:
                        next_earnings = datetime.fromtimestamp(int(earnings_ts)).strftime("%Y-%m-%d")
                    except Exception:
                        pass

                result[ticker] = {
                    "pe_ratio": round(pe, 2) if pe else None,
                    "eps": round(eps, 4) if eps else None,
                    "revenue_growth_yoy": round(revenue_growth * 100, 2) if revenue_growth else None,
                    "earnings_growth_yoy": round(earnings_growth * 100, 2) if earnings_growth else None,
                    "debt_to_equity": round(debt_equity, 2) if debt_equity else None,
                    "return_on_equity": round(roe * 100, 2) if roe else None,
                    "market_cap_usd": market_cap,
                    "dividend_yield_pct": round(dividend_yield * 100, 2) if dividend_yield else None,
                    "free_cashflow_usd": free_cashflow,
                    "total_revenue_usd": total_revenue,
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "country": info.get("country", "Unknown"),
                    "currency": info.get("currency", "USD"),
                    "name": info.get("longName") or info.get("shortName", ticker),
                    # Analyst data
                    "analyst_score": round(rec_mean, 3) if rec_mean else None,
                    "analyst_rating": rec_key,
                    "target_price": round(target_mean, 2) if target_mean else None,
                    "target_high": round(target_high, 2) if target_high else None,
                    "target_low": round(target_low, 2) if target_low else None,
                    "n_analysts": n_analysts,
                    "upside_pct": upside_pct,
                    "forward_eps": round(forward_eps, 4) if forward_eps else None,
                    "next_earnings": next_earnings,
                    "current_price": round(current_price, 2) if current_price else None,
                }
            except Exception as e:
                result[ticker] = {"error": str(e)}

    return result


def get_sector_median_pe(fundamentals: dict, sector: str) -> float | None:
    """Compute median P/E ratio for stocks in a given sector."""
    pes = [
        v["pe_ratio"]
        for v in fundamentals.values()
        if isinstance(v, dict)
        and v.get("sector") == sector
        and v.get("pe_ratio") is not None
        and v["pe_ratio"] > 0
        and v["pe_ratio"] < 200
    ]
    return round(float(np.median(pes)), 2) if pes else None


def calculate_portfolio_stats(
    portfolio: list[dict], price_history: dict
) -> dict:
    """Estimate portfolio expected return, volatility, and Sharpe ratio.

    Uses 1Y historical returns and volatility for each position.
    """
    weighted_return = 0.0
    weighted_vol = 0.0
    total_weight = 0.0

    for position in portfolio:
        ticker = position["ticker"]
        weight = position["weight"]
        data = price_history.get(ticker, {})

        if "error" in data or not data:
            continue

        ret = data.get("return_1y", 0) / 100
        vol = data.get("volatility_annualized", 20) / 100

        weighted_return += weight * ret
        weighted_vol += weight * vol
        total_weight += weight

    if total_weight == 0:
        return {"expected_return": 0, "volatility": 0, "sharpe": 0}

    risk_free = 0.045
    sharpe = (weighted_return - risk_free) / weighted_vol if weighted_vol > 0 else 0

    return {
        "expected_return": round(weighted_return, 4),
        "volatility": round(weighted_vol, 4),
        "sharpe": round(sharpe, 3),
    }


def calculate_position_sizes(
    ratings: list[dict],
    sector_weights: dict,
    max_position: float = 0.10,
    max_sector: float = 0.30,
    min_conviction: int = 50,
    min_positions: int = 10,
    max_positions: int = 20,
) -> list[dict]:
    """Size positions using conviction-weighted allocation with risk rules.

    Returns list of {ticker, weight, sector} dicts that sum to ~1.0.
    """
    eligible = [r for r in ratings if r.get("conviction", 0) >= min_conviction and r.get("rating") in ("Buy", "Hold")]
    eligible.sort(key=lambda x: x.get("conviction", 0), reverse=True)
    eligible = eligible[:max_positions]

    if len(eligible) < min_positions:
        eligible = sorted(ratings, key=lambda x: x.get("conviction", 0), reverse=True)[:max_positions]

    if not eligible:
        return []

    conviction_sum = sum(r.get("conviction", 50) for r in eligible)
    raw_weights = {r["ticker"]: r.get("conviction", 50) / conviction_sum for r in eligible}

    sector_allocation: dict[str, float] = {}
    for r in eligible:
        sector = r.get("sector", "Other")
        target = sector_weights.get(sector, 0.05)
        sector_allocation[sector] = sector_allocation.get(sector, 0) + raw_weights[r["ticker"]]

    positions = []
    sector_used: dict[str, float] = {}

    for r in eligible:
        ticker = r["ticker"]
        sector = r.get("sector", "Other")
        raw_w = raw_weights[ticker]

        sector_cap = sector_weights.get(sector, max_sector)
        effective_max_sector = min(sector_cap, max_sector)
        sector_used_so_far = sector_used.get(sector, 0)

        weight = min(raw_w, max_position, effective_max_sector - sector_used_so_far)
        weight = max(weight, 0)

        if weight > 0:
            positions.append({"ticker": ticker, "weight": round(weight, 4), "sector": sector})
            sector_used[sector] = sector_used_so_far + weight

    total = sum(p["weight"] for p in positions)
    if total > 0:
        for p in positions:
            p["weight"] = round(p["weight"] / total, 4)

    return positions
