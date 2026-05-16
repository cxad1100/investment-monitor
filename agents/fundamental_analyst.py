"""Agent 3: Fundamental Analyst — rates individual stocks Buy/Hold/Sell."""

import json
import re
import anthropic
from config import MODEL
from tools.yfinance_tools import get_sector_median_pe

SYSTEM_PROMPT = """You are a senior fundamental equity analyst at a quantitative investment firm.
Your job is to evaluate individual stocks and assign Buy/Hold/Sell ratings with conviction scores.

For each stock, evaluate three dimensions:

VALUE SCORE (0-100):
- Compare P/E to sector median (lower P/E = higher value score)
- Consider debt/equity ratio (lower = better)
- Dividend yield as a value indicator

GROWTH SCORE (0-100):
- Revenue growth year-over-year (higher = better)
- EPS growth rate
- Return on equity (ROE)

MOMENTUM SCORE (0-100):
- 1-year price return vs universe average
- Distance from 52-week high (closer = better momentum)
- Price volatility (lower vol = more stable momentum)

COMPOSITE CONVICTION SCORE = 0.4*value + 0.3*growth + 0.3*momentum (0-100)

RATING:
- Buy: conviction >= 65
- Hold: conviction 40-64
- Sell: conviction < 40

Use the tools to retrieve data for each ticker, compute scores, and build ratings.
Process stocks in batches for efficiency.

Output your final ratings as a JSON array with this structure:
[
  {
    "ticker": "AAPL",
    "rating": "Buy",
    "conviction": 78,
    "value_score": 65,
    "growth_score": 85,
    "momentum_score": 82,
    "sector": "Technology",
    "rationale": "Strong growth and momentum despite premium valuation"
  },
  ...
]
"""

TOOLS = [
    {
        "name": "get_stock_price_data",
        "description": "Get price history data for one or more tickers from the pre-fetched dataset. Returns 52w high/low, returns, volatility.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to retrieve",
                }
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_stock_fundamentals",
        "description": "Get fundamental data (P/E, EPS, revenue growth, ROE, debt/equity) for one or more tickers from the pre-fetched dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to retrieve",
                }
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_sector_median_pe",
        "description": "Get the median P/E ratio for a given sector from the universe, useful for relative valuation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "Sector name (e.g., 'Technology', 'Healthcare')",
                }
            },
            "required": ["sector"],
        },
    },
    {
        "name": "get_universe_stats",
        "description": "Get aggregate statistics for the universe (average returns, volatility) for relative comparison.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def _compute_universe_stats(price_data: dict) -> dict:
    import numpy as np
    returns = [v.get("return_1y", 0) for v in price_data.values() if "error" not in v]
    vols = [v.get("volatility_annualized", 20) for v in price_data.values() if "error" not in v]
    return {
        "avg_return_1y": round(float(np.mean(returns)), 2) if returns else 0,
        "median_return_1y": round(float(np.median(returns)), 2) if returns else 0,
        "avg_volatility": round(float(np.mean(vols)), 2) if vols else 20,
    }


def _execute_tool(name: str, inputs: dict, state: dict) -> str:
    price_data = state.get("price_data", {})
    fundamentals = state.get("fundamentals", {})

    if name == "get_stock_price_data":
        tickers = inputs.get("tickers", [])
        result = {t: price_data.get(t, {"error": "not found"}) for t in tickers}
        return json.dumps(result)

    elif name == "get_stock_fundamentals":
        tickers = inputs.get("tickers", [])
        result = {t: fundamentals.get(t, {"error": "not found"}) for t in tickers}
        return json.dumps(result)

    elif name == "get_sector_median_pe":
        sector = inputs.get("sector", "")
        median_pe = get_sector_median_pe(fundamentals, sector)
        return json.dumps({"sector": sector, "median_pe": median_pe})

    elif name == "get_universe_stats":
        stats = _compute_universe_stats(price_data)
        return json.dumps(stats)

    return json.dumps({"error": f"Unknown tool: {name}"})


def run(state: dict) -> dict:
    """Run the Fundamental Analyst agent. Returns updated state with stock ratings."""
    client = anthropic.Anthropic()

    price_data = state.get("price_data", {})
    fundamentals = state.get("fundamentals", {})
    valid_tickers = list(set(price_data.keys()) & set(fundamentals.keys()))

    print(f"[Fundamental Analyst] Evaluating {len(valid_tickers)} stocks...")

    messages = [
        {
            "role": "user",
            "content": (
                f"Evaluate all {len(valid_tickers)} stocks in the universe and assign "
                f"Buy/Hold/Sell ratings with conviction scores. Available tickers: "
                f"{', '.join(valid_tickers[:30])}{'...' if len(valid_tickers) > 30 else ''}. "
                f"Use the tools to retrieve data systematically. Process in batches of 10. "
                f"Output a complete JSON ratings array at the end."
            ),
        }
    ]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [Fundamental Analyst] → {block.name}")
                result = _execute_tool(block.name, block.input, state)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    final_text = next(
        (b.text for b in response.content if hasattr(b, "text")), "[]"
    )

    ratings = []
    try:
        json_match = re.search(r'\[[\s\S]*\]', final_text)
        if json_match:
            ratings = json.loads(json_match.group())
    except Exception:
        pass

    if not ratings:
        ratings = _fallback_ratings(valid_tickers, price_data, fundamentals, state.get("universe", []))

    state["ratings"] = ratings
    buy_count = sum(1 for r in ratings if r.get("rating") == "Buy")
    print(f"[Fundamental Analyst] Done. {len(ratings)} rated: {buy_count} Buy, "
          f"{sum(1 for r in ratings if r.get('rating') == 'Hold')} Hold, "
          f"{sum(1 for r in ratings if r.get('rating') == 'Sell')} Sell.")
    return state


def _fallback_ratings(tickers: list, price_data: dict, fundamentals: dict, universe: list) -> list:
    """Simple quantitative fallback if agent output can't be parsed."""
    import numpy as np

    sector_map = {row["yf_ticker"]: row.get("sector", "Other") for row in universe}
    ratings = []

    returns = [price_data[t].get("return_1y", 0) for t in tickers if t in price_data]
    avg_return = float(np.mean(returns)) if returns else 0

    for ticker in tickers:
        pd_data = price_data.get(ticker, {})
        fd_data = fundamentals.get(ticker, {})

        if "error" in pd_data or "error" in fd_data:
            continue

        ret = pd_data.get("return_1y", 0)
        vol = pd_data.get("volatility_annualized", 20)
        pe = fd_data.get("pe_ratio") or 25
        rev_growth = fd_data.get("revenue_growth_yoy") or 0
        roe = fd_data.get("return_on_equity") or 10
        pct_from_high = abs(pd_data.get("pct_from_52w_high", -20))

        value_score = max(0, min(100, 100 - (pe / 50 * 50) + (roe / 30 * 20)))
        growth_score = max(0, min(100, 50 + rev_growth + (roe - 10)))
        momentum_score = max(0, min(100, 50 + (ret - avg_return) / 2 - pct_from_high / 2))

        conviction = int(0.4 * value_score + 0.3 * growth_score + 0.3 * momentum_score)
        conviction = max(0, min(100, conviction))

        if conviction >= 65:
            rating = "Buy"
        elif conviction >= 40:
            rating = "Hold"
        else:
            rating = "Sell"

        ratings.append({
            "ticker": ticker,
            "rating": rating,
            "conviction": conviction,
            "value_score": round(value_score),
            "growth_score": round(growth_score),
            "momentum_score": round(momentum_score),
            "sector": sector_map.get(ticker, fd_data.get("sector", "Other")),
            "rationale": f"Quant fallback: value={round(value_score)}, growth={round(growth_score)}, momentum={round(momentum_score)}",
        })

    return sorted(ratings, key=lambda x: x["conviction"], reverse=True)
