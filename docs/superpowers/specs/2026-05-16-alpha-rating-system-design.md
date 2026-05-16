# TR Alpha Intelligence System — Design Spec
**Date:** 2026-05-16  
**Status:** Approved  
**Goal:** Rate every Trade Republic stock with an S&P-style letter grade + outlook using event-driven alpha signals from 7 data sources.

---

## Problem

The existing system uses fundamentals + macro (yfinance + FRED). This misses the strongest near-term alpha signals: earnings surprises, insider buying, geopolitical shocks, options positioning. We need an event-driven rating layer that captures what the market hasn't priced yet.

---

## Architecture

Sequential pipeline. Python collectors compress raw data into structured signals. Claude Code synthesizes signals into ratings.

```
Sunday 7pm → collect_all.py → signals.json → Claude rates stocks → ratings_report.json → git commit
```

### 7 Collectors

| # | Collector | Sources | Output Signal |
|---|---|---|---|
| 1 | MacroCommodities | FRED + yfinance futures (CL=F, GC=F, NG=F, HG=F) | Regime + sector tailwinds/headwinds + commodity trends |
| 2 | Geopolitical | GDELT API v2 + Polymarket geo markets + RSS news | Hotspot risk map + top event probabilities + sector exposures |
| 3 | EarningsCatalyst | yfinance estimates/calendar + Polymarket earnings markets | Per-company beat probability + upcoming dates + consensus |
| 4 | Fundamentals | yfinance: P/E, EPS, revenue growth, ROE, debt/equity | Per-company value/growth scores |
| 5 | InsiderFlow | SEC EDGAR Form 4 (free) | Net insider buy/sell per company, last 90 days |
| 6 | OptionsFlow | yfinance options chain: put-call ratio, open interest skew | Bullish/bearish positioning per company |
| 7 | ShortInterest | FINRA + yfinance | Short float %, days-to-cover |

All sources free/public. No additional API keys needed beyond existing FRED key.

### Data Flow

```
collect_all.py
  └── runs collectors 1–7 in sequence
  └── each collector emits a compact JSON object
  └── merges all signals into data/signals.json

[Claude Code scheduled agent]
  └── reads data/signals.json + data/tr_universe.csv
  └── for each stock: maps signals → composite score → letter grade + outlook + rationale
  └── writes ratings_report.json
  └── git commit + push
```

**signals.json total size:** ~8–12KB. Fits comfortably in one Claude context window.

---

## Rating Model

### Scale

S&P-compatible 21-point scale:

```
Investment Grade:  AAA  AA+  AA  AA-  A+  A  A-  BBB+  BBB  BBB-
Speculative Grade: BB+  BB  BB-  B+  B  B-
Distressed:        CCC  CC  C  D
```

Outlook: **Positive / Stable / Negative / Watch Positive / Watch Negative**

### Composite Score (0–100)

| Signal | Weight | Key Metric |
|---|---|---|
| Earnings catalyst | 25% | Polymarket beat probability (or yfinance surprise history if no market) |
| Insider flow | 20% | Net $ of insider buys vs sells, last 90 days, normalized by market cap |
| Macro + commodity fit | 15% | Does the stock's sector align with the current regime's tailwinds? |
| Geopolitical exposure | 15% | Company HQ region + revenue exposure to hotspot regions + commodity dependency |
| Fundamentals | 15% | 0.4×VALUE + 0.3×GROWTH + 0.3×MOMENTUM (existing scoring) |
| Options flow | 5% | Put-call ratio <0.7 = bullish, >1.3 = bearish |
| Short interest | 5% | High short float + positive catalyst = squeeze potential bonus |

### Score → Grade

| Score | Grade |
|---|---|
| 90–100 | AAA |
| 83–89 | AA+ |
| 77–82 | AA |
| 71–76 | AA- |
| 65–70 | A+ |
| 59–64 | A |
| 53–58 | A- |
| 47–52 | BBB+ |
| 41–46 | BBB |
| 35–40 | BBB- |
| 29–34 | BB+ |
| 23–28 | BB |
| 17–22 | BB- |
| 12–16 | B |
| 6–11 | CCC |
| 0–5 | CC / C / D |

### Outlook Logic

- **Positive:** score rising ≥5pts vs last week, OR earnings beat probability ≥70% within 14 days
- **Negative:** score falling ≥5pts vs last week, OR geopolitical event directly affects sector within 14 days
- **Watch:** major Polymarket event (>50% probability) resolving within 14 days that would materially change the rating
- **Stable:** no significant upcoming catalyst

---

## Output: ratings_report.json

```json
{
  "generated_at": "ISO date",
  "macro_regime": "string",
  "top_geopolitical_risks": [
    {"event": "string", "probability": 0.0, "sector_impact": "string"}
  ],
  "rating_summary": {
    "investment_grade": 0,
    "speculative": 0,
    "watch_negative": 0,
    "watch_positive": 0
  },
  "ratings": [
    {
      "ticker": "string",
      "name": "string",
      "grade": "AA+",
      "outlook": "Positive",
      "score": 0,
      "signal_scores": {
        "earnings_catalyst": 0,
        "insider_flow": 0,
        "macro_fit": 0,
        "geopolitical": 0,
        "fundamentals": 0,
        "options_flow": 0,
        "short_interest": 0
      },
      "rationale": "string — specific events, specific numbers",
      "key_catalysts": ["string"],
      "key_risks": ["string"],
      "sector": "string",
      "region": "string"
    }
  ],
  "top_10_buys": ["ticker"],
  "top_5_sells": ["ticker"],
  "changes_from_last_week": [
    {"ticker": "string", "old_grade": "string", "new_grade": "string", "reason": "string"}
  ]
}
```

---

## New Files to Build

### Collectors (pure Python, no Claude API)

| File | Collector | Key Libraries |
|---|---|---|
| `tools/futures_tools.py` | CL=F, GC=F, NG=F, HG=F via yfinance | yfinance |
| `tools/polymarket_tools.py` | Polymarket Gamma API — geo + earnings markets | requests |
| `tools/gdelt_tools.py` | GDELT API v2 — country risk indices + events | requests |
| `tools/news_tools.py` | RSS feeds (Reuters, FT) — sector/company headlines | feedparser |
| `tools/insider_tools.py` | SEC EDGAR Form 4 — net insider activity | requests, xml |
| `tools/options_tools.py` | yfinance options chain — put-call ratio | yfinance |
| `tools/short_interest_tools.py` | yfinance info.shortPercentOfFloat + FINRA | yfinance, requests |

### Orchestration

| File | Role |
|---|---|
| `collect_all.py` | Runs all 7 collectors, merges into signals.json |
| `signals_schema.py` | Pydantic models for signals.json (validation) |

### Updated Schedule Prompt

The remote agent prompt in the Claude Code routine needs updating to reference `collect_all.py` (not `collect_data.py`) and the new rating methodology.

---

## Geopolitical Exposure Mapping

Claude infers each company's exposure from:
1. **HQ region** (from yfinance `country` field)
2. **Sector** (e.g., Airlines → fuel cost exposure, European Banks → sanctions exposure)
3. **Commodity dependency** (manufacturers → copper/steel, energy companies → oil price)
4. **Polymarket markets that name specific companies** (e.g., "Will Boeing deliveries exceed X?")

Claude is explicitly instructed: "For each geopolitical hotspot, identify which TR universe stocks have material revenue or cost exposure, and adjust their geopolitical score accordingly."

---

## API Details

### Polymarket Gamma API (free, no key)
- Base: `https://gamma-api.polymarket.com`
- Geo/macro markets: `GET /markets?category=politics,economics&active=true`
- Earnings markets: `GET /markets?category=finance&active=true&keywords=earnings`
- Key fields: `question`, `outcomePrices` (probabilities), `volume24hr`, `endDate`

### GDELT API v2 (free, no key)
- Base: `https://api.gdeltproject.org/api/v2/`
- Country conflict index: `GET /summary/summary?d=lastmonth&k=<country>&fmt=json`
- Top events: `https://api.gdeltproject.org/api/v2/timeline/timeline?query=...&mode=timelinevolinfo`

### SEC EDGAR Form 4 (free, no key)
- Base: `https://efts.sec.gov/LATEST/search-index?q=%22form+4%22&dateRange=custom&startdt=<date>&enddt=<date>&entity=<company_name>`
- Or: `https://data.sec.gov/submissions/CIK{cik}.json` for company filings
- Key fields: issuer, reporter, transaction type (P=purchase, S=sale), shares, price

---

## Constraints

- **No Anthropic API key** — Claude Pro only. All data fetching is Python. Rating synthesis is one Claude Code session.
- **Weekly cadence** — runtime up to 30 minutes is acceptable.
- **Private repo** — FRED key committed to `.env`. No other secrets needed.
- **50 TR stocks** — universe size keeps signals.json compact enough for one context window.

---

## Success Criteria

1. Every TR stock gets a letter grade + outlook every Sunday
2. Rationale cites specific numbers (exact Polymarket probabilities, exact insider $ amounts, exact put-call ratios)
3. `changes_from_last_week` correctly identifies which stocks changed grade and why
4. `top_10_buys` consistently outperforms TR universe average over 3-month rolling window (measured retrospectively)
