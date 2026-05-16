"""Agent 4: Portfolio Manager — constructs final portfolio with risk management."""

import json
import re
import anthropic
from config import MODEL, RISK_RULES
from tools.yfinance_tools import calculate_position_sizes, calculate_portfolio_stats

SYSTEM_PROMPT = """You are the Chief Portfolio Manager at a quantitative investment firm.
Your job is to construct a final investment portfolio using the macro regime context
and fundamental stock ratings, applying strict risk management rules.

Risk Management Rules (NON-NEGOTIABLE):
- Maximum single position: 10%
- Maximum single sector: 30%
- Minimum positions: 10
- Maximum positions: 20
- Minimum conviction score to include: 50
- Weights must sum to approximately 100%

Portfolio Construction Workflow:
1. Retrieve the current market regime and sector weight targets
2. Retrieve all stock ratings, filtered to Buy and Hold with conviction >= 50
3. Calculate position sizes using conviction-weighted allocation
4. Apply risk constraints (position caps, sector caps)
5. Calculate portfolio statistics (expected return, volatility, Sharpe ratio)
6. Generate the final portfolio with rationale

Prioritize:
- Regime-appropriate sector exposure
- High-conviction Buy-rated stocks
- Diversification across sectors and geographies
- Risk-adjusted returns (target Sharpe > 0.5)

Output the final portfolio as a JSON object with this exact structure:
{
  "portfolio": [
    {"ticker": "AAPL", "weight": 0.08, "sector": "Technology", "rating": "Buy", "conviction": 82, "rationale": "..."}
  ],
  "sector_breakdown": {"Technology": 0.25, "Healthcare": 0.15},
  "stats": {"expected_return": 0.12, "volatility": 0.18, "sharpe": 0.67},
  "regime_context": "growth environment with moderate risk",
  "portfolio_thesis": "Brief description of the overall portfolio strategy"
}
"""

TOOLS = [
    {
        "name": "get_regime_and_sector_weights",
        "description": "Get the current market regime classification and target sector weights for portfolio construction.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_stock_ratings",
        "description": "Get all fundamental stock ratings with conviction scores. Can filter by minimum conviction and rating type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_conviction": {
                    "type": "integer",
                    "description": "Minimum conviction score to include (0-100)",
                    "default": 50,
                },
                "ratings_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ratings to include, e.g. ['Buy', 'Hold']",
                },
            },
            "required": [],
        },
    },
    {
        "name": "calculate_position_sizes",
        "description": "Calculate portfolio position sizes using conviction-weighted allocation with risk constraints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_conviction": {
                    "type": "integer",
                    "description": "Minimum conviction score to include",
                    "default": 50,
                }
            },
            "required": [],
        },
    },
    {
        "name": "calculate_portfolio_stats",
        "description": "Calculate expected return, volatility, and Sharpe ratio for a given set of portfolio positions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "description": "List of {ticker, weight} dicts",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                    },
                }
            },
            "required": ["positions"],
        },
    },
]


def _execute_tool(name: str, inputs: dict, state: dict) -> str:
    regime = state.get("regime", {})
    ratings = state.get("ratings", [])
    price_data = state.get("price_data", {})

    if name == "get_regime_and_sector_weights":
        return json.dumps({
            "regime": regime.get("regime", "growth"),
            "risk_level": regime.get("risk_level", "medium"),
            "sector_weights": regime.get("sector_weights", {}),
            "key_signals": regime.get("key_signals", []),
            "rationale": regime.get("rationale", ""),
        })

    elif name == "get_stock_ratings":
        min_conv = inputs.get("min_conviction", 50)
        ratings_filter = inputs.get("ratings_filter") or ["Buy", "Hold"]
        filtered = [
            r for r in ratings
            if r.get("conviction", 0) >= min_conv
            and r.get("rating") in ratings_filter
        ]
        filtered_sorted = sorted(filtered, key=lambda x: x.get("conviction", 0), reverse=True)
        return json.dumps({
            "ratings": filtered_sorted,
            "count": len(filtered_sorted),
            "buy_count": sum(1 for r in filtered_sorted if r.get("rating") == "Buy"),
        })

    elif name == "calculate_position_sizes":
        min_conv = inputs.get("min_conviction", RISK_RULES["min_conviction"])
        sector_weights = regime.get("sector_weights", {})
        positions = calculate_position_sizes(
            ratings=ratings,
            sector_weights=sector_weights,
            max_position=RISK_RULES["max_position_weight"],
            max_sector=RISK_RULES["max_sector_weight"],
            min_conviction=min_conv,
            min_positions=RISK_RULES["min_positions"],
            max_positions=RISK_RULES["max_positions"],
        )
        return json.dumps({"positions": positions, "count": len(positions)})

    elif name == "calculate_portfolio_stats":
        positions = inputs.get("positions", [])
        stats = calculate_portfolio_stats(positions, price_data)
        return json.dumps(stats)

    return json.dumps({"error": f"Unknown tool: {name}"})


def run(state: dict) -> dict:
    """Run the Portfolio Manager agent. Returns updated state with final portfolio."""
    client = anthropic.Anthropic()

    print("[Portfolio Manager] Constructing portfolio...")

    messages = [
        {
            "role": "user",
            "content": (
                "Construct a final investment portfolio using the macro regime context "
                "and fundamental ratings. Apply all risk management rules. "
                "Use the tools to retrieve ratings, size positions, and calculate statistics. "
                "Output the complete portfolio JSON at the end."
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
                print(f"  [Portfolio Manager] → {block.name}")
                result = _execute_tool(block.name, block.input, state)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    final_text = next(
        (b.text for b in response.content if hasattr(b, "text")), "{}"
    )

    portfolio_data = {}
    try:
        json_match = re.search(r'\{[\s\S]*\}', final_text)
        if json_match:
            portfolio_data = json.loads(json_match.group())
    except Exception:
        pass

    if not portfolio_data or "portfolio" not in portfolio_data:
        portfolio_data = _fallback_portfolio(state)

    from datetime import date
    portfolio_data["generated_at"] = str(date.today())

    state["portfolio"] = portfolio_data
    n = len(portfolio_data.get("portfolio", []))
    sharpe = portfolio_data.get("stats", {}).get("sharpe", 0)
    print(f"[Portfolio Manager] Done. {n} positions. Sharpe: {sharpe:.2f}")
    return state


def _fallback_portfolio(state: dict) -> dict:
    """Quantitative fallback portfolio construction."""
    regime = state.get("regime", {})
    ratings = state.get("ratings", [])
    price_data = state.get("price_data", {})

    sector_weights = regime.get("sector_weights", {})
    positions = calculate_position_sizes(
        ratings=ratings,
        sector_weights=sector_weights,
        max_position=RISK_RULES["max_position_weight"],
        max_sector=RISK_RULES["max_sector_weight"],
        min_conviction=RISK_RULES["min_conviction"],
        min_positions=RISK_RULES["min_positions"],
        max_positions=RISK_RULES["max_positions"],
    )

    ratings_map = {r["ticker"]: r for r in ratings}
    portfolio = []
    for pos in positions:
        r = ratings_map.get(pos["ticker"], {})
        portfolio.append({
            "ticker": pos["ticker"],
            "weight": pos["weight"],
            "sector": pos["sector"],
            "rating": r.get("rating", "Hold"),
            "conviction": r.get("conviction", 50),
            "rationale": r.get("rationale", "Quantitative selection"),
        })

    stats = calculate_portfolio_stats(positions, price_data)

    sector_breakdown: dict[str, float] = {}
    for p in portfolio:
        sector_breakdown[p["sector"]] = round(
            sector_breakdown.get(p["sector"], 0) + p["weight"], 4
        )

    return {
        "portfolio": portfolio,
        "sector_breakdown": sector_breakdown,
        "stats": stats,
        "regime_context": f"{regime.get('regime', 'growth')} environment, {regime.get('risk_level', 'medium')} risk",
        "portfolio_thesis": "Conviction-weighted portfolio aligned with macro regime",
    }
