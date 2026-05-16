"""
Python Pass 1: compute composite signal score (0-100) for every asset.
No Claude API needed. Pure math from signals.json.
Assets scoring >=70 or grade-changed get Claude deep analysis (Pass 2).
"""


def compute_earnings_score(earnings_data: dict) -> float:
    """Earnings catalyst score: beat_probability → 0-100."""
    prob = earnings_data.get("beat_probability", 0.5)
    return round(float(prob) * 100, 1)


def compute_insider_score(insider_data: dict) -> float:
    """Insider flow score: net_buy_pct_mktcap ∈ [-1,1] → 0-100."""
    net = float(insider_data.get("net_buy_pct_mktcap", 0.0))
    return round(max(0.0, min(100.0, 50.0 + net * 50.0)), 1)


def compute_macro_score(sector: str, macro_signal: dict) -> float:
    """Macro regime fit: tailwind=75, headwind=25, neutral=50."""
    tailwinds = macro_signal.get("sector_tailwinds", [])
    headwinds = macro_signal.get("sector_headwinds", [])
    if sector in tailwinds:
        return 75.0
    if sector in headwinds:
        return 25.0
    return 50.0


def compute_geo_score(ticker: str, asset_meta: dict, events: list[dict]) -> float:
    """Geopolitical exposure score: base 50, adjusted by event impacts."""
    sector = asset_meta.get("sector", "")
    region = asset_meta.get("region", "")
    score = 50.0
    magnitude_map = {"strong": 30.0, "moderate": 15.0, "weak": 5.0}

    for event in events:
        p = float(event.get("probability", 0.0))
        for impact in event.get("asset_impacts", []):
            matches = (
                impact.get("ticker") == ticker or
                (impact.get("sector") and impact.get("sector") == sector) or
                (impact.get("region") and impact.get("region") == region)
            )
            if matches:
                direction = 1.0 if impact.get("direction") == "positive" else -1.0
                magnitude = magnitude_map.get(impact.get("magnitude", "weak"), 5.0)
                score += direction * p * magnitude

    return round(max(0.0, min(100.0, score)), 1)


def compute_fundamentals_score(fund: dict, price: dict, price_stats: dict) -> float:
    """Value + growth + momentum composite."""
    pe = float(fund.get("pe_ratio") or 25)
    roe = float(fund.get("return_on_equity") or 10)
    rev_growth = float(fund.get("revenue_growth_yoy") or 0)
    ret = float(price.get("return_1y", 0))
    avg_ret = float(price_stats.get("avg_return_1y", 0))
    pct_high = float(price.get("pct_from_52w_high", -20))

    value_score = max(0.0, min(100.0, 100.0 - (pe / 50.0) * 50.0 + (roe / 30.0) * 20.0))
    growth_score = max(0.0, min(100.0, 50.0 + rev_growth + (roe - 10.0)))
    momentum_score = max(0.0, min(100.0, 50.0 + (ret - avg_ret) / 2.0 + pct_high / 2.0))

    return round(0.4 * value_score + 0.3 * growth_score + 0.3 * momentum_score, 1)


def compute_options_score(options_data: dict) -> float:
    """Options flow: put-call ratio → 0-100. PCR<0.7 bullish, PCR>1.3 bearish."""
    pcr = float(options_data.get("put_call_ratio", 1.0))
    score = max(0.0, min(100.0, 75.0 - (pcr - 0.7) * 50.0))
    return round(score, 1)


def compute_wsb_short_score(short_data: dict, wsb_data: dict) -> float:
    """WSB + short interest composite squeeze/momentum signal."""
    short_float = float(short_data.get("short_float_pct", 0))
    mentions = wsb_data.get("mentions_7d", 0) if isinstance(wsb_data, dict) else 0
    squeeze_flag = wsb_data.get("squeeze_flag", False) if isinstance(wsb_data, dict) else False

    if squeeze_flag and short_float > 20:
        return 85.0
    if short_float > 25 and mentions > 100:
        return 75.0
    return 50.0


def compute_composite_score(
    ticker: str,
    asset_meta: dict,
    earnings_data: dict,
    insider_data: dict,
    macro_signal: dict,
    events: list[dict],
    fund_data: dict,
    price_data: dict,
    price_stats: dict,
    options_data: dict,
    short_data: dict,
    wsb_ticker_data: dict,
) -> int:
    """Weighted composite score (0-100) per spec weights."""
    sector = asset_meta.get("sector", "")

    s_earnings = compute_earnings_score(earnings_data)
    s_insider = compute_insider_score(insider_data)
    s_macro = compute_macro_score(sector, macro_signal)
    s_geo = compute_geo_score(ticker, asset_meta, events)
    s_fund = compute_fundamentals_score(fund_data, price_data, price_stats)
    s_options = compute_options_score(options_data)
    s_wsb = compute_wsb_short_score(short_data, wsb_ticker_data)

    composite = (
        0.25 * s_earnings +
        0.20 * s_insider +
        0.15 * s_macro +
        0.15 * s_geo +
        0.15 * s_fund +
        0.05 * s_options +
        0.05 * s_wsb
    )
    return max(0, min(100, round(composite)))


def score_all_assets(signals: dict) -> dict[str, dict]:
    """Run fast scorer over all assets in signals.json."""
    universe_map = signals.get("universe_map", {})
    earnings = signals.get("earnings", {})
    insider = signals.get("insider", {})
    macro = signals.get("macro", {})
    events = signals.get("events", [])
    fundamentals = signals.get("fundamentals", {})
    price_data = signals.get("price_data", {})
    price_stats = signals.get("price_stats", {"avg_return_1y": 0})
    options = signals.get("options", {})
    short_interest = signals.get("short_interest", {})
    wsb = signals.get("wsb", {}).get("ticker_mentions", {})

    results = {}
    for ticker, asset_meta in universe_map.items():
        score = compute_composite_score(
            ticker=ticker,
            asset_meta=asset_meta,
            earnings_data=earnings.get(ticker, {}),
            insider_data=insider.get(ticker, {}),
            macro_signal=macro,
            events=events,
            fund_data=fundamentals.get(ticker, {}),
            price_data=price_data.get(ticker, {}),
            price_stats=price_stats,
            options_data=options.get(ticker, {}),
            short_data=short_interest.get(ticker, {}),
            wsb_ticker_data=wsb.get(ticker, {}),
        )
        results[ticker] = {
            "score": score,
            "sector": asset_meta.get("sector", ""),
            "region": asset_meta.get("region", ""),
            "type": asset_meta.get("type", "stock"),
        }
    return results


def score_to_grade(score: int) -> str:
    """Convert composite score to S&P-style letter grade."""
    thresholds = [
        (90, "AAA"), (83, "AA+"), (77, "AA"), (71, "AA-"),
        (65, "A+"), (59, "A"), (53, "A-"), (47, "BBB+"),
        (41, "BBB"), (35, "BBB-"), (29, "BB+"), (23, "BB"),
        (17, "BB-"), (12, "B"), (6, "CCC"),
    ]
    for threshold, grade in thresholds:
        if score >= threshold:
            return grade
    return "CC"
