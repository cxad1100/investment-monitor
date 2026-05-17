"""Portfolio parser and P&L calculator for Trade Republic holdings."""

import warnings
import csv
from pathlib import Path
from datetime import datetime

import yfinance as yf

# Map TR tickers to yfinance-resolvable equivalents where needed
TICKER_MAP = {
    "NVD.F":  "NVD.F",    # NVIDIA on Xetra
    "AMZ.F":  "AMZ.F",    # Amazon on Xetra
    "ABEA.F": "ABEA.F",   # check at runtime
    "TSFA.F": "TL0.F",    # Tesla on Xetra (TR uses TL0.F)
    "LHL.F":  "LHL.F",
    "TCO0.F": "TCO0.F",   # Tencent on Xetra
    "WBD.MI": "WBD.MI",   # Warner Bros Discovery on Borsa Italiana
}


def parse_portfolio(csv_path: str | Path) -> dict:
    """
    Parse trade history CSV into current holdings with avg cost and realized P&L.

    Returns:
      holdings: {ticker: {shares, avg_cost_eur, total_invested, first_buy, last_activity}}
      realized: {ticker: {pnl_eur, shares_sold, proceeds}}
      transactions: list of all rows
    """
    holdings: dict[str, dict] = {}
    realized: dict[str, dict] = {}
    transactions = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker   = row["Ticker"].strip()
            action   = row["Action"].strip().lower()
            shares   = float(row["Shares"])
            price    = float(row["Price"])          # total EUR
            pps      = float(row["PricePerShare"])  # EUR per share
            date     = row["Date"].strip()

            transactions.append({
                "date": date, "ticker": ticker, "action": action,
                "shares": shares, "price": price, "pps": pps,
            })

            if action == "buy":
                if ticker not in holdings:
                    holdings[ticker] = {
                        "shares": 0.0, "avg_cost": 0.0,
                        "total_invested": 0.0, "first_buy": date,
                    }
                h = holdings[ticker]
                new_shares = h["shares"] + shares
                # Weighted average cost
                h["avg_cost"] = (h["shares"] * h["avg_cost"] + shares * pps) / new_shares if new_shares > 0 else pps
                h["shares"] = new_shares
                h["total_invested"] = h["total_invested"] + price
                h["last_activity"] = date

            elif action == "sell":
                if ticker not in realized:
                    realized[ticker] = {"pnl_eur": 0.0, "shares_sold": 0.0, "proceeds": 0.0}
                r = realized[ticker]
                avg_cost = holdings.get(ticker, {}).get("avg_cost", pps)
                r["pnl_eur"]    += (pps - avg_cost) * shares
                r["shares_sold"] += shares
                r["proceeds"]   += price

                if ticker in holdings:
                    holdings[ticker]["shares"] -= shares
                    holdings[ticker]["last_activity"] = date
                    if holdings[ticker]["shares"] <= 0.001:
                        del holdings[ticker]   # fully exited

    return {
        "holdings": holdings,
        "realized": realized,
        "transactions": transactions,
    }


def fetch_current_prices(holdings: dict) -> dict[str, float | None]:
    """Fetch current EUR prices for all held tickers."""
    prices = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in holdings:
            yf_ticker = TICKER_MAP.get(ticker, ticker)
            try:
                info = yf.Ticker(yf_ticker).fast_info
                price = getattr(info, "last_price", None)
                if price and price > 0:
                    prices[ticker] = round(float(price), 4)
                else:
                    # Fallback: history
                    hist = yf.Ticker(yf_ticker).history(period="5d")
                    if not hist.empty:
                        prices[ticker] = round(float(hist["Close"].iloc[-1]), 4)
                    else:
                        prices[ticker] = None
            except Exception:
                prices[ticker] = None
    return prices


def compute_portfolio_summary(portfolio: dict, current_prices: dict) -> dict:
    """
    Compute full P&L summary.

    Returns:
      positions: list of position dicts (sorted by value desc)
      totals: {total_invested, current_value, unrealized_pnl, unrealized_pct,
               realized_pnl, total_pnl}
    """
    holdings = portfolio["holdings"]
    realized = portfolio["realized"]

    positions = []
    total_invested   = 0.0
    current_value    = 0.0
    unrealized_pnl   = 0.0

    for ticker, h in holdings.items():
        shares    = h["shares"]
        avg_cost  = h["avg_cost"]
        invested  = h["total_invested"]
        cur_price = current_prices.get(ticker)

        if cur_price:
            pos_value   = shares * cur_price
            pos_pnl     = (cur_price - avg_cost) * shares
            pos_pnl_pct = (cur_price / avg_cost - 1) * 100 if avg_cost > 0 else 0.0
        else:
            pos_value   = shares * avg_cost  # fallback: cost basis
            pos_pnl     = 0.0
            pos_pnl_pct = 0.0

        positions.append({
            "ticker":       ticker,
            "shares":       round(shares, 6),
            "avg_cost":     round(avg_cost, 4),
            "current_price": cur_price,
            "position_value": round(pos_value, 2),
            "cost_basis":   round(shares * avg_cost, 2),
            "unrealized_pnl": round(pos_pnl, 2),
            "unrealized_pct": round(pos_pnl_pct, 2),
            "first_buy":    h.get("first_buy", ""),
            "last_activity": h.get("last_activity", ""),
        })

        total_invested += shares * avg_cost
        current_value  += pos_value
        unrealized_pnl += pos_pnl

    # Realized P&L across all closed positions
    total_realized = sum(r["pnl_eur"] for r in realized.values())

    positions.sort(key=lambda x: -x["position_value"])

    return {
        "positions": positions,
        "totals": {
            "total_invested":   round(total_invested, 2),
            "current_value":    round(current_value, 2),
            "unrealized_pnl":   round(unrealized_pnl, 2),
            "unrealized_pct":   round((current_value / total_invested - 1) * 100, 2) if total_invested > 0 else 0,
            "realized_pnl":     round(total_realized, 2),
            "total_pnl":        round(unrealized_pnl + total_realized, 2),
        },
        "realized_detail": realized,
    }
