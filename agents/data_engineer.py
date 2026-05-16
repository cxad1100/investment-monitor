"""Agent 1: Data Engineer — fetches, cleans, and structures all data."""

import json
import anthropic
from config import MODEL, FRED_SERIES
from tools.universe import get_universe
from tools.yfinance_tools import fetch_price_history, fetch_fundamentals
from tools.fred_tools import fetch_fred_series

SYSTEM_PROMPT = """You are a senior data engineer for a quantitative investment firm.
Your job is to fetch, validate, and structure all data required for portfolio analysis.

You have access to tools to:
1. Load the Trade Republic stock universe
2. Fetch historical price data from yfinance
3. Fetch fundamental financial data from yfinance
4. Fetch macroeconomic indicators from FRED

Workflow:
- First load the TR universe to get the list of valid tickers
- Fetch price history for all tickers
- Fetch fundamentals for all tickers
- Fetch all FRED macro series
- Report what data was successfully collected and flag any gaps

Be methodical. Fetch data in batches if needed. Report the final count of tickers
with valid price data and valid fundamental data. Do not proceed if fewer than
10 tickers have usable data.

Output a JSON summary at the end with keys:
- tickers_loaded: list of validated tickers
- price_data_count: number of tickers with price data
- fundamental_data_count: number of tickers with fundamental data
- fred_series_fetched: list of FRED series IDs fetched
- data_quality_flags: any warnings or issues
"""

TOOLS = [
    {
        "name": "load_tr_universe",
        "description": "Load and return the Trade Republic stock universe from the CSV file. Returns a list of stock records with isin, name, yf_ticker, sector, region.",
        "input_schema": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to the TR universe CSV file",
                }
            },
            "required": ["csv_path"],
        },
    },
    {
        "name": "fetch_price_history",
        "description": "Fetch historical OHLCV price data for a list of yfinance tickers. Returns price metrics including 52-week high/low, 1Y return, and volatility.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of yfinance ticker symbols",
                },
                "period": {
                    "type": "string",
                    "description": "Lookback period (e.g., '1y', '2y', '6mo')",
                    "default": "1y",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "fetch_fundamentals",
        "description": "Fetch fundamental financial data (P/E, EPS, revenue growth, debt/equity, ROE, etc.) for a list of tickers from yfinance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of yfinance ticker symbols",
                }
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "fetch_fred_series",
        "description": "Fetch macroeconomic time series from FRED (Federal Reserve Economic Data). Returns latest values, trends, and year-over-year changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "series_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "FRED series IDs to fetch. If empty, fetches all default series.",
                },
                "lookback_years": {
                    "type": "integer",
                    "description": "Number of years of history to fetch",
                    "default": 3,
                },
            },
            "required": [],
        },
    },
]


def _execute_tool(name: str, inputs: dict, csv_path: str) -> str:
    if name == "load_tr_universe":
        path = inputs.get("csv_path", csv_path)
        records = get_universe(path, validate=False)
        return json.dumps({"records": records, "count": len(records)})

    elif name == "fetch_price_history":
        tickers = inputs["tickers"]
        period = inputs.get("period", "1y")
        data = fetch_price_history(tickers, period)
        valid = {k: v for k, v in data.items() if "error" not in v}
        invalid = [k for k, v in data.items() if "error" in v]
        return json.dumps({"price_data": valid, "invalid_tickers": invalid, "valid_count": len(valid)})

    elif name == "fetch_fundamentals":
        tickers = inputs["tickers"]
        data = fetch_fundamentals(tickers)
        valid = {k: v for k, v in data.items() if "error" not in v}
        invalid = [k for k, v in data.items() if "error" in v]
        return json.dumps({"fundamentals": valid, "invalid_tickers": invalid, "valid_count": len(valid)})

    elif name == "fetch_fred_series":
        series_ids = inputs.get("series_ids") or list(FRED_SERIES.keys())
        lookback_years = inputs.get("lookback_years", 3)
        data = fetch_fred_series(series_ids, lookback_years)
        return json.dumps({"fred_data": data, "series_count": len(data)})

    return json.dumps({"error": f"Unknown tool: {name}"})


def run(state: dict, csv_path: str, period: str = "1y") -> dict:
    """Run the Data Engineer agent. Returns updated state with fetched data."""
    client = anthropic.Anthropic()

    print("[Data Engineer] Starting data collection...")

    messages = [
        {
            "role": "user",
            "content": (
                f"Load the Trade Republic universe from '{csv_path}', "
                f"fetch {period} price history and fundamentals for all tickers, "
                f"then fetch all FRED macro series. Validate the data quality and "
                f"report what was successfully collected."
            ),
        }
    ]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
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
                print(f"  [Data Engineer] → {block.name}({list(block.input.keys())})")
                result = _execute_tool(block.name, block.input, csv_path)
                result_data = json.loads(result)

                if block.name == "load_tr_universe":
                    state["universe"] = result_data.get("records", [])
                elif block.name == "fetch_price_history":
                    state["price_data"] = result_data.get("price_data", {})
                elif block.name == "fetch_fundamentals":
                    state["fundamentals"] = result_data.get("fundamentals", {})
                elif block.name == "fetch_fred_series":
                    state["fred_data"] = result_data.get("fred_data", {})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    final_text = next(
        (b.text for b in response.content if hasattr(b, "text")), ""
    )
    state["data_engineer_summary"] = final_text
    print(f"[Data Engineer] Done. {len(state.get('price_data', {}))} tickers with price data.")
    return state
