"""
Multi-factor stock scorer — 0 to 100 composite.

Design principles:
  - Individual stock quality first. Sector tailwinds are context, not a score driver.
  - Missing data = factor excluded from weight pool (never defaulted to 50).
  - Each factor scored on absolute thresholds grounded in investing logic.
  - No percentile stretch (it amplified sector bias into artificial cliffs).
  - Result: scores spread naturally from company fundamentals.

Factor weights (when all data present):
  A. Analyst consensus         30%
  B. Valuation                 22%
  C. Quality / profitability   20%
  D. Price momentum            15%
  E. Market signals             8%
  F. Earnings catalyst          5%
"""


# ── Factor A: Analyst consensus ──────────────────────────────────────────────

def _analyst_score(f: dict) -> float | None:
    """1-5 → 100-0, blended with price target upside."""
    raw = f.get("analyst_score")
    if raw is None:
        return None
    conviction = (5.0 - float(raw)) / 4.0 * 100.0   # 1=100, 5=0
    upside = float(f.get("upside_pct") or 0.0)
    upside_s = max(0.0, min(100.0, 50.0 + upside * 0.8))  # +60% upside → 98
    n = int(f.get("n_analysts") or 1)
    confidence = min(1.0, n / 10.0)   # 10+ analysts = full confidence
    base = 0.65 * conviction + 0.35 * upside_s
    # Pull toward 50 when very few analysts
    return round(base * confidence + 50.0 * (1 - confidence), 1)


# ── Factor B: Valuation (lower = cheaper = better) ───────────────────────────

def _pe_score(pe) -> float | None:
    if pe is None:
        return None
    pe = float(pe)
    if pe <= 0:  return 10.0   # loss-making
    if pe <= 10: return 90.0
    if pe <= 15: return 82.0
    if pe <= 20: return 72.0
    if pe <= 25: return 62.0
    if pe <= 35: return 50.0
    if pe <= 50: return 35.0
    if pe <= 75: return 22.0
    return 12.0   # extreme growth premium

def _peg_score(peg) -> float | None:
    if peg is None:
        return None
    peg = float(peg)
    if peg <= 0:    return 30.0   # negative growth or no meaning
    if peg <= 0.75: return 92.0
    if peg <= 1.0:  return 82.0
    if peg <= 1.5:  return 68.0
    if peg <= 2.0:  return 52.0
    if peg <= 3.0:  return 38.0
    return 20.0

def _ev_ebitda_score(ev) -> float | None:
    if ev is None:
        return None
    ev = float(ev)
    if ev <= 0:    return 20.0
    if ev <= 6:    return 88.0
    if ev <= 10:   return 78.0
    if ev <= 15:   return 65.0
    if ev <= 20:   return 50.0
    if ev <= 30:   return 35.0
    return 18.0

def _pb_score(pb) -> float | None:
    if pb is None:
        return None
    pb = float(pb)
    if pb <= 0:   return 10.0
    if pb <= 1:   return 88.0
    if pb <= 2:   return 75.0
    if pb <= 3:   return 62.0
    if pb <= 5:   return 48.0
    if pb <= 10:  return 30.0
    return 15.0

def _valuation_score(f: dict) -> float | None:
    subs = [s for s in [
        _pe_score(f.get("pe_ratio")),
        _peg_score(f.get("peg_ratio")),
        _ev_ebitda_score(f.get("ev_to_ebitda")),
        _pb_score(f.get("price_to_book")),
    ] if s is not None]
    if not subs:
        return None
    return round(sum(subs) / len(subs), 1)


# ── Factor C: Quality / profitability ────────────────────────────────────────

def _roe_score(roe) -> float | None:
    if roe is None:
        return None
    roe = float(roe)
    if roe < 0:   return 10.0
    if roe >= 35: return 95.0
    if roe >= 25: return 85.0
    if roe >= 15: return 72.0
    if roe >= 10: return 60.0
    if roe >= 5:  return 48.0
    return 30.0

def _op_margin_score(m) -> float | None:
    if m is None:
        return None
    m = float(m)
    if m < 0:    return 10.0
    if m >= 35:  return 95.0
    if m >= 25:  return 85.0
    if m >= 15:  return 72.0
    if m >= 8:   return 58.0
    if m >= 3:   return 42.0
    return 25.0

def _rev_growth_score(g) -> float | None:
    if g is None:
        return None
    g = float(g)
    if g >= 30:   return 92.0
    if g >= 20:   return 82.0
    if g >= 10:   return 70.0
    if g >= 5:    return 60.0
    if g >= 0:    return 52.0
    if g >= -5:   return 38.0
    if g >= -15:  return 25.0
    return 12.0

def _earnings_growth_score(g) -> float | None:
    if g is None:
        return None
    g = float(g)
    if g >= 30:   return 90.0
    if g >= 20:   return 78.0
    if g >= 10:   return 65.0
    if g >= 0:    return 55.0
    if g >= -10:  return 35.0
    return 18.0

def _quality_score(f: dict) -> float | None:
    subs_w = [
        (_roe_score(f.get("return_on_equity")),    0.35),
        (_op_margin_score(f.get("operating_margin")), 0.30),
        (_rev_growth_score(f.get("revenue_growth_yoy")), 0.20),
        (_earnings_growth_score(f.get("earnings_growth_yoy")), 0.15),
    ]
    available = [(s, w) for s, w in subs_w if s is not None]
    if not available:
        return None
    total_w = sum(w for _, w in available)
    return round(sum(s * w for s, w in available) / total_w, 1)


# ── Factor D: Price momentum ──────────────────────────────────────────────────

def _momentum_score(p: dict, avg_ret_1y: float) -> float | None:
    if not p or "error" in p:
        return None
    ret_1y = float(p.get("return_1y") or 0)
    pct_high = float(p.get("pct_from_52w_high") or -50)

    # Relative return vs universe (removes market beta)
    rel = ret_1y - avg_ret_1y
    if rel >= 40:   rel_s = 88.0
    elif rel >= 25: rel_s = 78.0
    elif rel >= 10: rel_s = 67.0
    elif rel >= 0:  rel_s = 58.0
    elif rel >= -10: rel_s = 46.0
    elif rel >= -25: rel_s = 33.0
    else:            rel_s = 18.0

    # Position vs 52W high — near high = strength, deep correction = weakness
    if pct_high >= -3:   h_s = 80.0
    elif pct_high >= -10: h_s = 70.0
    elif pct_high >= -20: h_s = 58.0
    elif pct_high >= -35: h_s = 44.0
    elif pct_high >= -50: h_s = 30.0
    else:                 h_s = 15.0

    return round(0.65 * rel_s + 0.35 * h_s, 1)


# ── Factor E: Market signals ──────────────────────────────────────────────────

def _market_signal_score(i: dict, s: dict, o: dict) -> float | None:
    parts = []

    net_buy = float(i.get("net_buy_pct_mktcap", 0))
    if net_buy != 0:
        if net_buy > 0.02:   ins = 85.0
        elif net_buy > 0.005: ins = 70.0
        elif net_buy > 0:     ins = 58.0
        elif net_buy > -0.005: ins = 45.0
        else:                  ins = 25.0
        parts.append((ins, 0.50))

    short_pct = float(s.get("short_float_pct", 0))
    if short_pct > 0:
        if short_pct > 30:   sh = 20.0
        elif short_pct > 20: sh = 30.0
        elif short_pct > 10: sh = 45.0
        elif short_pct > 5:  sh = 62.0
        else:                sh = 75.0
        parts.append((sh, 0.30))

    pcr = o.get("put_call_ratio")
    if pcr is not None:
        pcr = float(pcr)
        if pcr < 0.5:    opt = 80.0
        elif pcr < 0.8:  opt = 68.0
        elif pcr < 1.0:  opt = 55.0
        elif pcr < 1.3:  opt = 42.0
        else:            opt = 25.0
        parts.append((opt, 0.20))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return round(sum(v * w for v, w in parts) / total_w, 1)


# ── Factor F: Earnings catalyst ──────────────────────────────────────────────

def _earnings_catalyst_score(e: dict) -> float | None:
    prob = e.get("beat_probability")
    if prob is None or float(prob) == 0.5:
        return None  # default = no real signal
    prob = float(prob)
    if prob >= 0.80: return 85.0
    if prob >= 0.65: return 70.0
    if prob >= 0.55: return 60.0
    if prob >= 0.45: return 42.0
    if prob >= 0.30: return 28.0
    return 15.0


# ── Composite ─────────────────────────────────────────────────────────────────

def _compose(scores: dict[str, float | None], target_weights: dict[str, float]) -> int:
    """
    Weighted average of AVAILABLE factors only.
    Missing factors have their weight redistributed proportionally to present factors.
    """
    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        return 50  # truly no data at all

    total_target = sum(target_weights[k] for k in available)
    if total_target == 0:
        return 50

    composite = sum(available[k] * target_weights[k] for k in available) / total_target
    return max(0, min(100, round(composite)))


# ── Signal count for confidence flag ─────────────────────────────────────────

def _count_signals(f: dict, p: dict, i: dict, s: dict, e: dict) -> int:
    return sum([
        f.get("analyst_score") is not None,
        any(f.get(k) for k in ("pe_ratio", "peg_ratio", "ev_to_ebitda", "price_to_book")),
        any(f.get(k) for k in ("return_on_equity", "operating_margin", "revenue_growth_yoy")),
        bool(p and "error" not in p and p.get("return_1y") is not None),
        float(i.get("net_buy_pct_mktcap", 0)) != 0,
        float(s.get("short_float_pct", 0)) > 0,
        e.get("beat_probability") not in (None, 0.5),
    ])


# ── Main scorer ───────────────────────────────────────────────────────────────

WEIGHTS = {
    "analyst":   30.0,
    "valuation": 22.0,
    "quality":   20.0,
    "momentum":  15.0,
    "market":     8.0,
    "earnings":   5.0,
}

# Mild sector adjustment (max ±5 pts). Not a scoring factor — applied after composite.
SECTOR_ADJUST = {
    # tailwinds: +3 to +5
    "Energy":     +3,
    "Financials": +2,
    "Information Technology": +2,
    # headwinds: -2 to -4
    "Real Estate":            -4,
    "Consumer Staples":       -2,
    "Consumer Discretionary": -2,
}


def score_all_assets(signals: dict) -> dict[str, dict]:
    universe_map  = signals.get("universe_map", {})
    earnings      = signals.get("earnings", {})
    insider       = signals.get("insider", {})
    fundamentals  = signals.get("fundamentals", {})
    price_data    = signals.get("price_data", {})
    price_stats   = signals.get("price_stats", {"avg_return_1y": 0})
    options       = signals.get("options", {})
    short_interest= signals.get("short_interest", {})

    avg_ret = float(price_stats.get("avg_return_1y", 0))

    results = {}
    for ticker, asset_meta in universe_map.items():
        sector = asset_meta.get("sector", "")
        f  = fundamentals.get(ticker, {})
        p  = price_data.get(ticker, {})
        i  = insider.get(ticker, {})
        o  = options.get(ticker, {})
        s  = short_interest.get(ticker, {})
        e  = earnings.get(ticker, {})

        factor_scores = {
            "analyst":   _analyst_score(f),
            "valuation": _valuation_score(f),
            "quality":   _quality_score(f),
            "momentum":  _momentum_score(p, avg_ret),
            "market":    _market_signal_score(i, s, o),
            "earnings":  _earnings_catalyst_score(e),
        }

        composite = _compose(factor_scores, WEIGHTS)

        # Apply mild sector context (not a scoring signal, just a nudge)
        adj = SECTOR_ADJUST.get(sector, 0)
        composite = max(0, min(100, composite + adj))

        signal_count = _count_signals(f, p, i, s, e)
        no_signal = signal_count <= 1

        results[ticker] = {
            "score":        composite,
            "grade":        score_to_grade(composite),
            "sector":       sector,
            "region":       asset_meta.get("region", ""),
            "type":         asset_meta.get("type", "stock"),
            "no_signal":    no_signal,
            "signal_count": signal_count,
            "sub_scores":   {k: v for k, v in factor_scores.items() if v is not None},
        }

    return results


def score_to_grade(score: int) -> str:
    thresholds = [
        (88, "AAA"), (82, "AA+"), (76, "AA"), (70, "AA-"),
        (65, "A+"),  (60, "A"),   (55, "A-"), (50, "BBB+"),
        (45, "BBB"), (40, "BBB-"),(35, "BB+"), (30, "BB"),
        (25, "BB-"), (18, "B"),   (10, "CCC"),
    ]
    for threshold, grade in thresholds:
        if score >= threshold:
            return grade
    return "CC"
