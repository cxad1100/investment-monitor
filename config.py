import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

MODEL = "claude-opus-4-7"

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "CPI (Inflation)",
    "T10Y2Y": "10Y-2Y Yield Spread",
    "UNRATE": "Unemployment Rate",
    "T10YIE": "10Y Breakeven Inflation",
    "GDP": "Real GDP Growth",
    "DGS10": "10-Year Treasury Yield",
}

RISK_RULES = {
    "max_position_weight": 0.10,
    "max_sector_weight": 0.30,
    "min_positions": 10,
    "max_positions": 20,
    "min_conviction": 50,
}

SECTOR_WEIGHTS_BY_REGIME = {
    "growth": {
        "Technology": 0.28, "Consumer Discretionary": 0.18,
        "Financials": 0.15, "Healthcare": 0.12,
        "Industrials": 0.10, "Consumer Staples": 0.08,
        "Communication Services": 0.05, "Utilities": 0.02, "Other": 0.02,
    },
    "inflation": {
        "Energy": 0.20, "Materials": 0.15, "Financials": 0.15,
        "Consumer Staples": 0.15, "Healthcare": 0.12,
        "Industrials": 0.10, "Technology": 0.08, "Other": 0.05,
    },
    "stagflation": {
        "Consumer Staples": 0.25, "Healthcare": 0.20, "Energy": 0.15,
        "Utilities": 0.12, "Financials": 0.10, "Materials": 0.08,
        "Technology": 0.05, "Other": 0.05,
    },
    "deflation": {
        "Consumer Staples": 0.22, "Utilities": 0.18, "Healthcare": 0.18,
        "Technology": 0.15, "Financials": 0.12,
        "Consumer Discretionary": 0.08, "Other": 0.07,
    },
    "recession": {
        "Consumer Staples": 0.25, "Utilities": 0.20, "Healthcare": 0.20,
        "Financials": 0.10, "Technology": 0.10,
        "Communication Services": 0.08, "Other": 0.07,
    },
}
