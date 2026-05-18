"""Polymarket Gamma API — comprehensive financial prediction market signals.

Categories collected in a single paginate pass (no repeat API calls):
  geo        — geopolitical / military risk
  rates      — Fed, ECB, BOJ, central bank rate decisions
  economy    — macro indicators: CPI, GDP, unemployment, recession
  indices    — S&P 500, Nasdaq, Dow, Russell, sector indices
  earnings   — per-company beat/miss, EPS, revenue thresholds
  stocks     — individual stock price targets, market cap milestones, IPOs
  alpha      — cross-category top 25 ranked by signal quality

Alpha signal classification:
  conviction_bull     prob>75%, vol>$500k → market strongly expects YES
  conviction_bear     prob<25%, vol>$500k → market strongly expects NO
  uncertainty_alpha   35-65%, resolving<30d, vol>$100k → genuine debate
  watch               moderate signal
  low_signal / noise  insufficient data
"""

import re
from datetime import datetime, timezone
import requests

POLYMARKET_BASE = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

# ── Noise exclusion ──────────────────────────────────────────────────────────
EXCLUDE_PHRASES = [
    "gta vi", "world cup", "nba", "nfl", "nhl", "stanley cup", "super bowl",
    "oscar", "grammy", "champions league", "europa league", "premier league",
    "formula 1", "f1 race", "wimbledon", "us open tennis", "tour de france",
    "miss universe", "eurovision", "american idol", "pop star", "celebrity",
    "fdv above", "fdv below",
    "metamask", "hyperbeat", "meme coin",
    "2028 us presidential", "2030 ",
]

# ── Geopolitical ──────────────────────────────────────────────────────────────
GEO_WORDS = [
    "iran", "russia", "ukraine", "taiwan", "china", "nato", "israel",
    "gaza", "opec", "north korea", "putin", "zelenskyy", "xi jinping",
    "saudi arabia", "venezuela", "iran nuclear", "south china sea",
]
GEO_PHRASES = [
    "invade", "invasion", "military clash", "ceasefire", "peace deal",
    "sanctions", "nuclear deal", "nuke", "regime fall", "blockade", "coup",
    "strait of hormuz", "oil embargo", "trade sanctions", "arms deal",
]

# ── Central bank rates ────────────────────────────────────────────────────────
RATES_PHRASES = [
    # Fed
    "fed decrease interest rate", "fed increase interest rate",
    "no change in fed interest rate", "fomc meeting", "federal reserve rate",
    "fed cut rates", "fed raise rates", "federal funds rate",
    "basis points after the", "rate cut after", "rate hike after",
    "fed rate in", "fed funds rate", "fed pivot",
    # ECB
    "ecb rate", "ecb cut", "ecb raise", "european central bank rate",
    "ecb deposit rate", "ecb meeting",
    # BOJ / BOE / other
    "bank of japan rate", "boj rate", "bank of england rate",
    "boe rate", "bank of canada rate", "rba rate", "reserve bank",
    # Annual path
    "rate cuts happen in 2026", "rate cuts happen in 2027",
    "rate cut happen in 2026", "rate hike in 2026",
    "interest rate by end of", "rates below", "rates above",
    "yield curve", "10-year yield", "10 year treasury",
]

# ── Economic indicators ───────────────────────────────────────────────────────
ECONOMY_PHRASES = [
    # Inflation
    "cpi above", "cpi below", "cpi at", "inflation above", "inflation below",
    "core cpi", "pce above", "pce below", "pce inflation",
    "inflation rate", "deflation", "hyperinflation",
    # Growth
    "gdp growth", "gdp above", "gdp below", "gdp contraction",
    "us recession", "recession by end of", "recession in 2026",
    "soft landing", "hard landing",
    # Labor
    "unemployment above", "unemployment below", "unemployment rate",
    "jobs report", "nonfarm payroll", "jobless claims",
    # Trade
    "tariff", "trade war", "trade deficit", "trade surplus",
    "import tax", "export ban", "trade deal",
    # Housing
    "housing starts", "home price", "mortgage rate",
    # Consumer
    "consumer confidence", "retail sales", "consumer spending",
    # Debt / fiscal
    "us debt ceiling", "government shutdown", "budget deficit",
    "national debt", "treasury default",
]

# ── Market indices ────────────────────────────────────────────────────────────
INDICES_PHRASES = [
    # S&P 500
    "s&p 500 above", "s&p 500 below", "s&p 500 reach", "s&p 500 hit",
    "s&p 500 end", "s&p 500 close", "s&p 500 by end",
    "spy above", "spy below", "spx above", "spx below",
    # Nasdaq
    "nasdaq above", "nasdaq below", "nasdaq reach", "nasdaq hit",
    "nasdaq 100", "qqq above", "qqq below",
    # Dow
    "dow jones above", "dow jones below", "dow above", "dow hit",
    "djia above", "djia below",
    # Russell / VIX / others
    "russell 2000", "vix above", "vix below", "volatility index",
    "ftse above", "dax above", "nikkei above", "hang seng",
    # General
    "stock market crash", "bull market", "bear market", "market correction",
    "all-time high", "new high", "index level",
]

# ── Company earnings & financial events ───────────────────────────────────────
EARNINGS_PHRASES = [
    # Beat/miss
    "beat earnings", "miss earnings", "beat on earnings", "earnings beat",
    "earnings miss", "beat estimates", "miss estimates",
    "beat expectations", "miss expectations",
    # EPS / revenue
    "eps above", "eps below", "earnings per share above", "earnings per share below",
    "revenue above", "revenue below", "revenue beat", "revenue miss",
    "quarterly revenue", "quarterly earnings", "quarterly profit",
    # Guidance
    "raise guidance", "cut guidance", "lower guidance", "guidance above",
    "full year guidance", "raised outlook", "lowered outlook",
    # Specific quarters
    "q1 earnings", "q2 earnings", "q3 earnings", "q4 earnings",
    "q1 revenue", "q2 revenue", "q3 revenue", "q4 revenue",
    "first quarter", "second quarter", "third quarter", "fourth quarter",
    # Dividends / buybacks
    "dividend increase", "dividend cut", "share buyback",
    "special dividend",
]

# ── Individual stocks & company events ───────────────────────────────────────
STOCKS_WORDS = [
    # Mega-cap / most-traded
    "nvidia", "apple", "microsoft", "google", "alphabet", "amazon", "meta",
    "tesla", "tsmc", "samsung", "asml", "broadcom", "netflix",
    "berkshire", "jpmorgan", "visa", "mastercard", "unitedhealth",
    "exxon", "chevron", "walmart", "costco", "home depot",
    # High-interest individual names
    "openai", "spacex", "stripe", "databricks", "anthropic", "palantir",
    "coinbase", "robinhood", "arm", "marvell", "amd", "intel", "qualcomm",
]
STOCKS_PHRASES = [
    "market cap be greater than", "market cap between", "market cap exceed",
    "largest company", "most valuable company",
    "ipo before", "ipo by", "ipo day", "go public",
    "will tesla", "will nvidia", "will apple", "will microsoft",
    "will amazon", "will google", "will meta", "will netflix",
    "stock price above", "stock price below", "stock price reach",
    "stock split", "acquisition of", "merge with", "acquire",
    "best ai model", "top ai model", "ai race",
    # Crypto price
    "bitcoin hit", "bitcoin reach", "bitcoin above", "bitcoin below",
    "btc hit", "btc reach", "btc above", "btc below", "bitcoin $",
    "ethereum above", "ethereum below", "eth above",
]


def parse_probability(market: dict) -> float:
    prices = market.get("outcomePrices", [])
    candidates = ([prices[0]] if prices else []) + [market.get("lastTradePrice")]
    for src in candidates:
        if src is None:
            continue
        try:
            p = float(src)
            if 0.0 < p < 1.0:
                return round(p, 4)
        except (ValueError, TypeError):
            pass
    return 0.5


def _fetch_all_active(max_markets: int = 5000) -> list[dict]:
    markets: list[dict] = []
    offset = 0
    while len(markets) < max_markets:
        try:
            r = requests.get(
                f"{POLYMARKET_BASE}/markets",
                params={"active": "true", "limit": "100", "offset": str(offset)},
                headers=HEADERS, timeout=30,
            )
            r.raise_for_status()
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            markets.extend(batch)
            if len(batch) < 100:
                break
            offset += 100
        except Exception:
            break
    return markets


def _is_noise(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in EXCLUDE_PHRASES)


def _days_to_resolution(end_date_str: str) -> float | None:
    if not end_date_str:
        return None
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = (end - now).total_seconds() / 86400
        return round(delta, 1) if delta > 0 else None
    except Exception:
        return None


def _alpha_signal(prob: float, volume: float, days: float | None) -> str:
    if volume < 10_000:
        return "noise"
    urgent = days is not None and days <= 45
    high_vol = volume >= 500_000
    if prob >= 0.75 and high_vol:
        return "conviction_bull"
    if prob <= 0.25 and high_vol:
        return "conviction_bear"
    if prob >= 0.75 and volume >= 100_000:
        return "conviction_bull"
    if prob <= 0.25 and volume >= 100_000:
        return "conviction_bear"
    if 0.35 <= prob <= 0.65 and urgent and volume >= 50_000:
        return "uncertainty_alpha"
    if (prob >= 0.65 or prob <= 0.35) and volume >= 25_000:
        return "watch"
    return "low_signal"


def _to_md(m: dict, category: str = "") -> dict:
    prob = parse_probability(m)
    volume = float(m.get("volume", 0) or 0)
    end_date = m.get("endDate", "")
    days = _days_to_resolution(end_date)
    try:
        best_bid = float(m.get("bestBid") or 0) or None
        best_ask = float(m.get("bestAsk") or 0) or None
    except Exception:
        best_bid = best_ask = None
    spread = round(best_ask - best_bid, 4) if best_bid and best_ask else None

    return {
        "id":                 m.get("id", ""),
        "question":           m.get("question", ""),
        "probability":        prob,
        "volume":             volume,
        "end_date":           end_date,
        "days_to_resolution": days,
        "spread":             spread,
        "alpha_signal":       _alpha_signal(prob, volume, days),
        "category":           category,
        "source":             "polymarket",
    }


def _word_match(q: str, words: list[str]) -> bool:
    ql = q.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", ql) for w in words)


def _phrase_match(q: str, phrases: list[str]) -> bool:
    ql = q.lower()
    return any(p in ql for p in phrases)


def fetch_all_investment_markets() -> dict:
    """
    Single paginate pass → 6 categories + alpha list.
    Prioritises earnings markets (lowest volume threshold = $5k) because
    individual company earnings odds carry the most direct alpha.
    """
    raw = _fetch_all_active(max_markets=5000)

    geo, rates, economy, indices, earnings, stocks = [], [], [], [], [], []
    seen: set[str] = set()

    # Volume thresholds per category
    T = {
        "earnings": 5_000,   # very low — even small earnings markets matter
        "stocks":  25_000,
        "indices": 50_000,
        "rates":  100_000,
        "economy":100_000,
        "geo":    100_000,
    }

    for m in raw:
        mid = m.get("id", "")
        if mid in seen:
            continue
        q = m.get("question", "")
        if _is_noise(q):
            continue
        vol = float(m.get("volume", 0) or 0)
        md = _to_md(m)

        matched = False

        # 1. Earnings — check first (highest alpha, lowest threshold)
        if not matched and _phrase_match(q, EARNINGS_PHRASES) and vol >= T["earnings"]:
            md["category"] = "earnings"
            earnings.append(md)
            seen.add(mid)
            matched = True

        # 2. Individual stock events
        if not matched and (
            _word_match(q, STOCKS_WORDS) or _phrase_match(q, STOCKS_PHRASES)
        ) and vol >= T["stocks"]:
            md["category"] = "stocks"
            stocks.append(md)
            seen.add(mid)
            matched = True

        # 3. Indices
        if not matched and _phrase_match(q, INDICES_PHRASES) and vol >= T["indices"]:
            md["category"] = "indices"
            indices.append(md)
            seen.add(mid)
            matched = True

        # 4. Central bank rates
        if not matched and _phrase_match(q, RATES_PHRASES) and vol >= T["rates"]:
            md["category"] = "rates"
            rates.append(md)
            seen.add(mid)
            matched = True

        # 5. Economy
        if not matched and _phrase_match(q, ECONOMY_PHRASES) and vol >= T["economy"]:
            md["category"] = "economy"
            economy.append(md)
            seen.add(mid)
            matched = True

        # 6. Geo
        if not matched and (
            _word_match(q, GEO_WORDS) or _phrase_match(q, GEO_PHRASES)
        ) and vol >= T["geo"]:
            md["category"] = "geo"
            geo.append(md)
            seen.add(mid)

    # Alpha: cross-category top 25 sorted by signal quality then volume
    all_markets = geo + rates + economy + indices + earnings + stocks
    alpha_order = {
        "conviction_bull": 0, "conviction_bear": 1,
        "uncertainty_alpha": 2, "watch": 3,
    }
    alpha = sorted(
        [m for m in all_markets if m["alpha_signal"] in alpha_order],
        key=lambda x: (alpha_order[x["alpha_signal"]], -x["volume"])
    )[:25]

    return {
        "geo":      sorted(geo,      key=lambda x: -x["volume"])[:40],
        "rates":    sorted(rates,    key=lambda x: -x["volume"])[:40],
        "economy":  sorted(economy,  key=lambda x: -x["volume"])[:40],
        "indices":  sorted(indices,  key=lambda x: -x["volume"])[:40],
        "earnings": sorted(earnings, key=lambda x: -x["volume"])[:60],
        "stocks":   sorted(stocks,   key=lambda x: -x["volume"])[:40],
        "alpha":    alpha,
        # legacy keys
        "macro":    sorted(rates + economy, key=lambda x: -x["volume"])[:50],
        "company":  sorted(stocks,  key=lambda x: -x["volume"])[:30],
    }


# ── Legacy aliases ───────────────────────────────────────────────────────────
def fetch_geo_conflict_markets(_min_volume: float = 100_000.0) -> list[dict]:
    return fetch_all_investment_markets()["geo"]

def fetch_macro_markets(_min_volume: float = 100_000.0) -> list[dict]:
    return fetch_all_investment_markets()["macro"]

def fetch_company_markets(_min_volume: float = 25_000.0) -> list[dict]:
    return fetch_all_investment_markets()["stocks"]

def fetch_geopolitical_markets(_min_volume: float = 100_000.0) -> list[dict]:
    return fetch_all_investment_markets()["geo"]

def fetch_earnings_markets(_min_volume: float = 5_000.0) -> list[dict]:
    return fetch_all_investment_markets()["earnings"]
