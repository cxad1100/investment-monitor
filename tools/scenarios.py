"""Pure expected-value math for the Scenarios page — the user's
'Quantifying Risk and Return' spreadsheet, as testable functions.

No I/O, no network. The page's gather() resolves target prices to returns
(using live current prices) and hands plain numbers to these functions.

Model (book.json):
  scenarios: [{"id","label","prob","prob_source"}]          # global, shared by all assets
  assets:    [{"name","ticker","weight","outcomes": {sid: {"target_price"|"target_pct"}}}]
"""

from __future__ import annotations

_SUM_TOL = 1e-3  # probs / weights "sum to ~1" tolerance


def pct_move(current: float, target: float) -> float:
    """Fractional move from current to target price (0.10 == +10%)."""
    return target / current - 1.0


def asset_expected_return(returns_by_scenario: dict, probs: dict) -> float:
    """Probability-weighted return for one asset: Σ prob[s]·ret[s]."""
    return sum(probs[s] * returns_by_scenario[s] for s in probs)


def portfolio_expected_return(weights: dict, asset_exp: dict) -> float:
    """Blended portfolio expected return: Σ weight[a]·expected[a]."""
    return sum(weights[a] * asset_exp[a] for a in weights)


def scenario_grid(weights: dict, asset_ret: dict) -> dict:
    """Portfolio return under each scenario (the risk view).

    asset_ret: {asset -> {scenario_id -> return}}. Returns {scenario_id -> return},
    each = Σ over assets weight[a]·asset_ret[a][s].
    """
    scenarios = set().union(*(r.keys() for r in asset_ret.values())) if asset_ret else set()
    return {
        s: sum(weights[a] * asset_ret[a][s] for a in asset_ret if s in asset_ret[a])
        for s in scenarios
    }


def resolve_returns(book: dict, current_prices: dict) -> dict:
    """Turn a book + live current prices into per-scenario returns.

    Each outcome is either an explicit ``target_pct`` (used directly) or a
    ``target_price`` (anchored to the live current price for that ticker). An
    asset whose outcome needs a price we don't have is dropped from the math and
    reported in ``skipped`` (the page flags it) — never silently valued.

    Returns {probs, weights, asset_ret, skipped}.
    """
    probs = {s["id"]: s["prob"] for s in book["scenarios"]}
    asset_ret: dict = {}
    weights: dict = {}
    skipped: list = []

    for a in book["assets"]:
        name, ticker = a["name"], a.get("ticker")
        rets: dict = {}
        ok = True
        for sid, outcome in a["outcomes"].items():
            if "target_pct" in outcome and outcome["target_pct"] is not None:
                rets[sid] = outcome["target_pct"]
            else:
                cur = current_prices.get(ticker)
                if cur is None:
                    ok = False
                    break
                rets[sid] = pct_move(cur, outcome["target_price"])
        if ok:
            asset_ret[name] = rets
            weights[name] = a["weight"]
        else:
            skipped.append(name)

    return {"probs": probs, "weights": weights, "asset_ret": asset_ret, "skipped": skipped}


def validate_book(book: dict) -> list[str]:
    """Human-readable problems with a book; empty list means OK."""
    problems: list[str] = []

    scenarios = book.get("scenarios", [])
    sids = [s["id"] for s in scenarios]
    prob_sum = sum(s.get("prob", 0.0) for s in scenarios)
    if abs(prob_sum - 1.0) > _SUM_TOL:
        problems.append(f"scenario probabilities sum to {prob_sum:.3f}, expected 1.000")

    assets = book.get("assets", [])
    weight_sum = sum(a.get("weight", 0.0) for a in assets)
    if abs(weight_sum - 1.0) > _SUM_TOL:
        problems.append(f"asset weights sum to {weight_sum:.3f}, expected 1.000")

    for a in assets:
        name = a.get("name", "?")
        outcomes = a.get("outcomes", {})
        for sid in sids:
            if sid not in outcomes:
                problems.append(f"asset '{name}' missing outcome for scenario '{sid}'")
        for sid in outcomes:
            if sid not in sids:
                problems.append(f"asset '{name}' has outcome for unknown scenario '{sid}'")
            else:
                o = outcomes[sid]
                has_price = o.get("target_price") is not None
                has_pct = o.get("target_pct") is not None
                if has_price == has_pct:  # neither or both
                    problems.append(
                        f"asset '{name}' scenario '{sid}' needs exactly one of "
                        "target_price / target_pct")

    return problems
