"""Agent 2: Macro-Economic Analyst — classifies market regime from FRED data."""

import json
import anthropic
from config import MODEL, SECTOR_WEIGHTS_BY_REGIME
from tools.fred_tools import classify_regime, fred_data_summary

SYSTEM_PROMPT = """You are a senior macro-economic analyst at a quantitative investment firm.
Your job is to analyze Federal Reserve Economic Data (FRED) indicators and determine
the current market regime, assess risks, and recommend sector tilts for portfolio construction.

You have access to tools to:
1. Read the FRED macro data that was already fetched
2. Run a quantitative regime classifier on the indicators

Workflow:
- Read the current FRED data summary
- Run the quantitative regime classifier
- Interpret the indicators with your expert judgment
- Provide sector weighting recommendations based on the regime

Regimes to classify: growth, inflation, stagflation, deflation, recession

For each regime, consider:
- Growth: rising GDP, low inflation, tight labor market, positive yield curve
- Inflation: high CPI, rising breakeven inflation, rising rates
- Stagflation: high inflation + rising unemployment or negative growth
- Deflation: falling prices, inverted yield curve, weak demand
- Recession: negative yield spread, high unemployment, falling GDP

Output your final analysis as a JSON object with this exact structure:
{
  "regime": "<regime_label>",
  "risk_level": "<low|medium|high>",
  "sector_weights": {"Technology": 0.25, ...},
  "key_signals": ["signal1", "signal2"],
  "rationale": "Brief explanation of the regime classification",
  "risk_factors": ["risk1", "risk2"]
}
"""

TOOLS = [
    {
        "name": "get_fred_data_summary",
        "description": "Get the current FRED macroeconomic data that was fetched by the Data Engineer agent. Returns a formatted summary of all indicators.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_regime_classifier",
        "description": "Run a quantitative regime classification algorithm on the FRED indicators. Returns a preliminary regime label and key indicator values.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_sector_weights_for_regime",
        "description": "Get the default sector weighting recommendations for a given economic regime.",
        "input_schema": {
            "type": "object",
            "properties": {
                "regime": {
                    "type": "string",
                    "enum": ["growth", "inflation", "stagflation", "deflation", "recession"],
                    "description": "The economic regime label",
                }
            },
            "required": ["regime"],
        },
    },
]


def _execute_tool(name: str, inputs: dict, fred_data: dict) -> str:
    if name == "get_fred_data_summary":
        return json.dumps({"summary": fred_data_summary(fred_data), "raw_data": fred_data})

    elif name == "run_regime_classifier":
        classification = classify_regime(fred_data)
        return json.dumps(classification)

    elif name == "get_sector_weights_for_regime":
        regime = inputs.get("regime", "growth")
        weights = SECTOR_WEIGHTS_BY_REGIME.get(regime, SECTOR_WEIGHTS_BY_REGIME["growth"])
        return json.dumps({"regime": regime, "sector_weights": weights})

    return json.dumps({"error": f"Unknown tool: {name}"})


def run(state: dict) -> dict:
    """Run the Macro Analyst agent. Returns updated state with regime analysis."""
    client = anthropic.Anthropic()
    fred_data = state.get("fred_data", {})

    print("[Macro Analyst] Analyzing macroeconomic regime...")

    messages = [
        {
            "role": "user",
            "content": (
                "Analyze the current macroeconomic data to classify the market regime. "
                "Use all available tools to gather information, then provide your "
                "expert assessment with sector weighting recommendations."
            ),
        }
    ]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
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
                print(f"  [Macro Analyst] → {block.name}")
                result = _execute_tool(block.name, block.input, fred_data)
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

    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', final_text)
        if json_match:
            regime_data = json.loads(json_match.group())
        else:
            regime_data = classify_regime(fred_data)
            regime_data["rationale"] = final_text[:500]
    except Exception:
        regime_data = classify_regime(fred_data)
        regime_data["rationale"] = "Quantitative classification (parse error)"

    if "sector_weights" not in regime_data:
        regime = regime_data.get("regime", "growth")
        regime_data["sector_weights"] = SECTOR_WEIGHTS_BY_REGIME.get(regime, SECTOR_WEIGHTS_BY_REGIME["growth"])

    state["regime"] = regime_data
    print(f"[Macro Analyst] Regime: {regime_data.get('regime')} | Risk: {regime_data.get('risk_level')}")
    return state
