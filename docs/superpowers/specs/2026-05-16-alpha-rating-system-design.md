# TR Alpha Intelligence System — Design Spec v2
**Date:** 2026-05-16  
**Status:** Approved  
**Goal:** Rate every Trade Republic stock and ETF with an S&P-style letter grade + outlook using event-driven alpha signals. Core innovation: extract probabilistic world events and map them to asset correlations via causal chain reasoning.

---

## What This System Does

Each Sunday it:
1. Maintains and refreshes the full TR asset universe (stocks + ETFs, 500–2000+ assets)
2. Collects 7 data signals from Polymarket, GDELT, Reddit, FRED, yfinance, SEC, and news
3. Extracts a weekly **Event List** — named events (A, B, C…) each with P(event) and asset impact
4. Assigns every TR asset an S&P-style letter grade + outlook + rationale citing specific events
5. Commits `ratings_report.json` to GitHub

---

## Core Innovation: Probabilistic Event Framework

### Concept

Instead of vague "macro risk," the system produces explicit named events:

```
Event A: "Ukraine ceasefire announced within 30 days"    P = 0.31
Event B: "Iran-Israel conflict escalates to air strikes" P = 0.44
Event C: "Fed cuts rates at September meeting"           P = 0.67
Event D: "Nvidia earnings beat consensus by >10%"        P = 0.78
Event E: "US recession begins within 6 months"          P = 0.22
```

Each event has:
- `id`: A, B, C...
- `description`: plain English
- `probability`: 0.0–1.0
- `complement`: P(¬event) = 1 - P(event)
- `source`: where the probability came from (Polymarket direct / GDELT-inferred / analyst consensus)
- `resolution_date`: when we'll know the outcome
- `asset_impacts`: list of {ticker, direction, magnitude, causal_chain}

### Coherence Rules

Before saving events to signals.json, Python enforces:
1. **Complement rule**: P(A) + P(¬A) = 1.0 for every event
2. **Mutual exclusivity**: if events are declared mutually exclusive, their probabilities must sum ≤ 1.0
3. **Dependency flag**: if Event A implies Event B, this is noted so the rating agent doesn't double-count

### Causal Chain Mapping (Claude's job)

For each event, Claude reasons through causal chains to derive asset impacts:

```
Event B: Iran-Israel escalates
  → Strait of Hormuz risk increases
    → Oil supply disruption risk rises
      → CL=F (crude futures) prices up → Energy sector tailwind
      → Airlines: fuel cost spike → headwind
      → European industrials: energy input cost → headwind
  → Defense spending sentiment rises
    → Defense ETFs, defense stocks: tailwind
  → Middle East stability falls
    → UAE/Saudi market exposure stocks: risk up
    → Suez Canal shipping: disruption risk
      → Shipping companies: risk
      → Asian exporters dependent on Suez: headwind
      → South Korea ETF (export economy): headwind

Asset impacts from Event B:
  XOM: +strong (energy, oil price)
  RHM.DE (Rheinmetall): +moderate (defense sentiment)
  LHA.DE (Lufthansa): -strong (fuel cost)
  HEIA.AS (Heineken): -weak (consumer confidence)
  iShares MSCI South Korea ETF: -moderate (shipping disruption)
```

This multi-hop reasoning is where the system generates genuine alpha — the market prices obvious first-order effects; causal chains reveal second and third-order impacts that are slower to price in.

---

## Universe Management

### Scale

Trade Republic offers approximately:
- **Stocks**: ~2000+ (US equities, European equities, small/mid caps)
- **ETFs**: ~1500+ (iShares, Vanguard, Xtrackers, Amundi, MSCI country ETFs)
- **Total**: potentially 3500+ assets

For weekly analysis at this scale, we use **two-pass architecture**.

### Weekly Universe Refresh

Every Sunday before analysis:

```python
# universe_manager.py
1. Load current data/universe.csv
2. Validate all existing tickers via yfinance.fast_info (remove if price = None)
3. Attempt to add new tickers from discovery sources:
   - TR community tracker (GitHub: nicholasgasior/tr-assets or equivalent)
   - XETRA listed securities with TR flag
   - Major ETF providers (iShares DE, Vanguard IE) ISINs known to be on TR
4. Write updated data/universe.csv with timestamp
5. Log: added N tickers, removed M tickers
```

Universe CSV columns: `isin, name, yf_ticker, type (stock|etf), sector, region, added_date`

### Two-Pass Rating (handles scale)

**Pass 1 — Python Fast Score (all assets)**
Pure Python computes a quick composite signal score (0–100) for every asset in the universe using pre-computed signals. No Claude. Takes ~2 minutes for 3000 assets.

**Pass 2 — Claude Deep Rating (top movers)**
- Assets scoring ≥70: rated in detail (expected ~200–400 assets)
- Assets with grade change from last week: rated in detail regardless of score
- Assets scoring <70 with no change: carry forward last week's grade with "Stable" outlook

This keeps the Claude session focused and the output high-quality.

---

## Architecture: 9 Collectors

```
collect_all.py
│
├── 1. UniverseManager           Refresh and validate full TR asset list
├── 2. MacroCommoditiesCollector FRED + futures: CL=F GC=F NG=F HG=F
├── 3. GeopoliticalCollector     GDELT + Polymarket geo markets + RSS news
├── 4. EventExtractor            Claude extracts Event A/B/C... with probabilities
├── 5. EarningsCatalystCollector yfinance estimates + Polymarket earnings markets
├── 6. FundamentalsCollector     yfinance: P/E EPS ROE revenue growth debt/equity
├── 7. InsiderFlowCollector      SEC EDGAR Form 4 — net insider activity
├── 8. OptionsFlowCollector      yfinance options: put-call ratio, OI skew
├── 9. ShortInterestCollector    yfinance shortPercentOfFloat + FINRA
├── 10. WSBCollector             Reddit r/wallstreetbets — trending tickers, squeezes
└── 11. BTCSignalCollector       BTC-USD: price trend, dominance, on-chain proxy
         │
         └── signals.json (compact, ~15–20KB total)
```

Note: EventExtractor (#4) uses the raw outputs of collectors #2 and #3 to produce the probabilistic event list. It's a Claude mini-session (one call) that reads geo + macro data and outputs structured events JSON.

---

## New Collectors (detail)

### 10. WSBCollector — Reddit r/wallstreetbets

**Purpose:** Detect fast-moving trends before they fully price in. Squeezes, sector pumps (semiconductors, photonics, robotics, AI), meme momentum.

**Source:** Reddit public JSON endpoint — no API key required.
```
GET https://www.reddit.com/r/wallstreetbets/hot.json?limit=100
GET https://www.reddit.com/r/wallstreetbets/top.json?t=week&limit=100
```

**Signal extraction (Python):**
- Regex scan all post titles + top comments for ticker mentions ($TICKER or TICKER pattern)
- Count mention frequency and velocity (mentions this week vs last week)
- Flag posts with keywords: "squeeze", "short interest", "gamma", "calls", "YOLO", "moon"
- Extract sector themes: semiconductor, AI, photonics, robotics, defense, biotech

**Output signal:**
```json
{
  "trending_tickers": [
    {"ticker": "GME", "mentions_7d": 847, "velocity": "+340%", "squeeze_flag": true},
    {"ticker": "NVDA", "mentions_7d": 623, "velocity": "+12%", "squeeze_flag": false}
  ],
  "sector_hype": [
    {"sector": "Semiconductors", "score": 0.82, "drivers": ["AI demand", "TSMC capex"]},
    {"sector": "Defense", "score": 0.71, "drivers": ["Ukraine aid", "NATO spending"]}
  ],
  "squeeze_candidates": ["GME", "AMC", "BBBY"],
  "short_squeeze_context": "High short interest + rising call volume pattern detected in GME"
}
```

### 11. BTCSignalCollector

**Purpose:** BTC as a dual signal:
1. **Liquidity proxy**: BTC rising = risk-on sentiment, institutional liquidity flowing into risk assets → growth stocks benefit
2. **Anti-establishment sentiment**: BTC rising with gold = distrust of institutions → defensive assets, hard assets
3. **Regulatory risk**: major regulatory news = crypto sector risk-off

**Source:** yfinance (BTC-USD), also ETH-USD as corroborating signal

**Output signal:**
```json
{
  "btc_price": 68420,
  "btc_1w_return": "+8.3%",
  "btc_4w_return": "+22.1%",
  "btc_vs_gold": "BTC outperforming gold → pure risk-on, not safe haven flight",
  "liquidity_signal": "positive",
  "regime": "risk_on",
  "interpretation": "BTC +8% WoW with equities also rising: institutional risk appetite strong. Favors growth/tech. Watch for rotation risk if BTC peaks.",
  "crypto_regulatory_news": "no major events this week",
  "impact_on_tr_assets": {
    "growth_tech": "tailwind",
    "financials": "neutral",
    "gold_etfs": "mild headwind (capital rotating away)"
  }
}
```

### 4. EventExtractor (mini Claude session)

Takes the raw outputs of MacroCommodities + Geopolitical collectors and calls Claude once to produce the structured event list:

**Input prompt to Claude:**
```
Given this macro data: [fred_signal] and geopolitical data: [geo_signal],
extract 5–10 discrete events for this week. For each event:
- Write a plain English description
- Assign probability 0.0–1.0 (use Polymarket if available, infer from GDELT trends otherwise)
- Ensure P(event) + P(¬event) = 1.0
- Flag any events that are logically dependent on each other
- Do NOT assign two mutually exclusive events probabilities that sum > 1.0

Output strict JSON array.
```

**Output: events.json (part of signals.json)**
```json
[
  {
    "id": "A",
    "description": "Ukraine ceasefire announced within 30 days",
    "probability": 0.31,
    "complement_description": "Ukraine war continues as-is",
    "complement_probability": 0.69,
    "source": "Polymarket market ID 0x3f2a...",
    "resolution_date": "2026-06-15",
    "depends_on": null,
    "asset_impacts": []
  }
]
```

The `asset_impacts` field is populated by the main rating agent, not by EventExtractor.

---

## Rating Model

### Weights (unchanged from v1)

| Signal | Weight |
|---|---|
| Earnings catalyst | 25% |
| Insider flow | 20% |
| Macro + commodity fit | 15% |
| Geopolitical / event exposure | 15% |
| Fundamentals | 15% |
| Options flow | 5% |
| Short interest / WSB | 5% |

WSB signal feeds into the short interest / momentum bucket. BTC signal feeds into macro fit.

### Grade Scale

```
AAA  AA+  AA  AA-  │  Investment grade, top tier
A+   A    A-        │  Investment grade, strong
BBB+ BBB  BBB-      │  Investment grade, adequate
BB+  BB   BB-       │  Speculative
B+   B    B-        │  Highly speculative
CCC  CC   C   D     │  Distressed
```

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
| 0–5 | CC/C/D |

### Outlook Logic

- **Positive**: score +5pts vs last week, OR earnings beat prob ≥70% within 14 days, OR WSB squeeze flag + short interest >20%
- **Negative**: score −5pts vs last week, OR event with >50% probability directly harms sector within 14 days
- **Watch Positive/Negative**: major Polymarket event resolving within 14 days that would flip grade tier
- **Stable**: no material catalyst

---

## Output: ratings_report.json

```json
{
  "generated_at": "ISO date",
  "universe_size": 3412,
  "deeply_rated_count": 287,
  "carry_forward_count": 3125,
  "macro_regime": "inflation",
  "btc_signal": "risk_on",
  "weekly_events": [
    {
      "id": "A",
      "description": "Ukraine ceasefire announced within 30 days",
      "probability": 0.31,
      "top_beneficiaries": ["reconstruction ETF", "NOKIA.HE (telecom infrastructure)"],
      "top_losers": ["defense ETFs", "RHM.DE"]
    }
  ],
  "wsb_alerts": [
    {"ticker": "GME", "type": "squeeze_risk", "detail": "847 mentions, +340% WoW velocity"}
  ],
  "ratings": [
    {
      "ticker": "NVDA",
      "name": "NVIDIA Corp",
      "grade": "AA+",
      "outlook": "Positive",
      "score": 87,
      "pass": "deep",
      "signal_scores": {
        "earnings_catalyst": 92,
        "insider_flow": 85,
        "macro_fit": 80,
        "event_exposure": 75,
        "fundamentals": 88,
        "options_flow": 75,
        "wsb_short": 60
      },
      "event_exposures": [
        {"event_id": "C", "direction": "positive", "magnitude": "moderate",
         "chain": "Fed rate cut → growth multiple expansion → high-PE tech benefits"}
      ],
      "rationale": "Polymarket: 78% earnings beat probability. Insider buy $12M (Form 4, 5d ago). Fed cut event C (P=0.67) is a direct tailwind for high-multiple tech. BTC risk-on regime confirms institutional appetite. Put-call 0.4 (bullish skew).",
      "key_catalysts": ["Earnings Aug 21 beat prob 78%", "Fed cut P=0.67"],
      "key_risks": ["Taiwan risk (Event F, P=0.28)", "P/E 42 → rate sensitivity"],
      "sector": "Technology",
      "region": "US",
      "type": "stock"
    }
  ],
  "top_10_buys": ["NVDA", "XOM", "RHM.DE"],
  "top_5_sells": ["LHA.DE", "DIS"],
  "squeeze_watchlist": ["GME"],
  "changes_from_last_week": [
    {"ticker": "TSLA", "old_grade": "BBB-", "new_grade": "BB+",
     "reason": "Earnings miss Polymarket 65%, BTC correlation declining"}
  ],
  "event_portfolio_matrix": {
    "A_ukraine_ceasefire": {
      "strong_buy": ["CAT (construction)", "NOKIA.HE"],
      "strong_sell": ["RHM.DE (defense)"]
    },
    "B_iran_escalation": {
      "strong_buy": ["XOM", "energy ETFs"],
      "strong_sell": ["LHA.DE (airlines)", "shipping ETFs"]
    }
  }
}
```

---

## File Structure (new files to build)

```
tools/
  universe_manager.py     Full TR universe: load, validate, refresh weekly
  futures_tools.py        CL=F GC=F NG=F HG=F via yfinance
  polymarket_tools.py     Polymarket Gamma API (geo + earnings markets)
  gdelt_tools.py          GDELT API v2 (country conflict indices)
  news_tools.py           RSS feeds: Reuters, FT
  insider_tools.py        SEC EDGAR Form 4
  options_tools.py        yfinance options chain: put-call ratio
  short_interest_tools.py yfinance shortPercentOfFloat + FINRA
  wsb_tools.py            Reddit public JSON: mention frequency, squeeze flags
  btc_tools.py            BTC-USD trend + regime classification

collect_all.py            Orchestrates all 9 collectors → signals.json
universe_manager.py       Weekly add/remove asset logic
data/
  universe.csv            Full TR asset universe (stocks + ETFs), refreshed weekly
  signals.json            Weekly signal snapshot (input to Claude)
  events.json             Extracted probabilistic events (subset of signals.json)
  ratings_report.json     Final output
  last_ratings.json       Previous week's grades (for change detection)
```

---

## API Details

### Polymarket Gamma API (free, no key)
- `GET https://gamma-api.polymarket.com/markets?active=true&category=politics`
- `GET https://gamma-api.polymarket.com/markets?active=true&category=finance`
- Key fields: `question`, `outcomePrices` (probability array), `volume`, `endDate`

### GDELT API v2 (free, no key)
- `GET https://api.gdeltproject.org/api/v2/summary/summary?d=lastmonth&k={country}&fmt=json`
- Conflict tone, event counts by CAMEO code

### Reddit public JSON (free, no key needed for read)
- `GET https://www.reddit.com/r/wallstreetbets/hot.json?limit=100`
- Requires `User-Agent: Mozilla/5.0` header

### SEC EDGAR Form 4 (free, no key)
- `GET https://efts.sec.gov/LATEST/search-index?q=%22form+4%22&dateRange=custom&startdt={30d_ago}&entity={name}`

### FINRA Short Interest (free)
- `GET https://regsho.finra.org/FNSQshvol{date}.txt` (daily short volume)

---

## Constraints

- **No Anthropic API key** — Claude Pro plan. Two mini-Claude calls per run: EventExtractor + main rating agent. Both run inside the scheduled Claude Code remote session.
- **Scale handling** — Python fast-scores all 3000+ assets; Claude deep-rates only top movers (~200–400)
- **Weekly cadence** — runtime budget: up to 45 minutes acceptable
- **Private repo** — FRED key in `.env`
- **ETFs** — rated on event exposure + momentum only (no P/E fundamentals); sector proxy from ETF name/ISIN

---

## Success Criteria

1. Every TR asset gets a letter grade each Sunday (deep or carry-forward)
2. Event list cites specific Polymarket probabilities or GDELT-inferred probabilities
3. Event portfolio matrix shows which TR assets benefit/suffer from each named event
4. WSB squeeze alerts surface ≥24 hours before Reddit-driven price move (measured retrospectively)
5. BTC regime signal correctly anticipates risk-on/risk-off rotations (measured retrospectively)
6. `top_10_buys` outperforms TR universe average over 3-month rolling window
