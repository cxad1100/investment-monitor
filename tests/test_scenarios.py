"""Pure scenario-engine math, pinned to the user's 'Quantifying Risk and Return'
spreadsheet. The displayed sheet values are rounded (111%, -32%, 96%, 47%, 52%, 0%);
these tests assert the exact underlying numbers the sheet rounds from.

Spreadsheet inputs:
  asset 1: current 95,   scenario a -> 200,  scenario b -> 65
  asset 2: current 5090, scenario a -> 4400, scenario b -> 6900
  scenario probabilities: a 90%, b 10%
  weights: asset 1 53%, asset 2 47%
"""

from tools.scenarios import (
    pct_move,
    asset_expected_return,
    portfolio_expected_return,
    scenario_grid,
    resolve_returns,
    validate_book,
)

PROBS = {"a": 0.9, "b": 0.1}
WEIGHTS = {"asset 1": 0.53, "asset 2": 0.47}

# exact per-scenario returns from the sheet's prices
RET1 = {"a": 200 / 95 - 1, "b": 65 / 95 - 1}        # +1.10526, -0.31579
RET2 = {"a": 4400 / 5090 - 1, "b": 6900 / 5090 - 1}  # -0.13556, +0.35560


def test_pct_move():
    assert abs(pct_move(95, 200) - 1.1052631) < 1e-6
    assert abs(pct_move(95, 65) - (-0.3157894)) < 1e-6
    assert pct_move(100, 100) == 0.0


def test_asset_expected_return_matches_sheet():
    # asset 1 -> 96.3% (sheet shows 96%); asset 2 -> -8.6% (sheet shows -9%)
    assert abs(asset_expected_return(RET1, PROBS) - 0.963158) < 1e-4
    assert abs(asset_expected_return(RET2, PROBS) - (-0.086444)) < 1e-4


def test_portfolio_expected_return_matches_sheet():
    asset_exp = {
        "asset 1": asset_expected_return(RET1, PROBS),
        "asset 2": asset_expected_return(RET2, PROBS),
    }
    # 0.53*0.963158 + 0.47*(-0.086444) = 0.469845 -> sheet 47%
    assert abs(portfolio_expected_return(WEIGHTS, asset_exp) - 0.469845) < 1e-4


def test_scenario_grid_matches_sheet():
    grid = scenario_grid(WEIGHTS, {"asset 1": RET1, "asset 2": RET2})
    # scenario a -> 52.2% (sheet 52%); scenario b -> ~0% (sheet 0%)
    assert abs(grid["a"] - 0.522076) < 1e-4
    assert abs(grid["b"] - 0.0) < 1e-3


# ── resolve_returns: book + live current prices -> per-scenario returns ──────────

def _book():
    return {
        "scenarios": [
            {"id": "a", "label": "soft landing", "prob": 0.9, "prob_source": "manual"},
            {"id": "b", "label": "recession", "prob": 0.1, "prob_source": "manual"},
        ],
        "assets": [
            {"name": "asset 1", "ticker": "AAA", "weight": 0.53,
             "outcomes": {"a": {"target_price": 200}, "b": {"target_price": 65}}},
            {"name": "asset 2", "ticker": "BBB", "weight": 0.47,
             "outcomes": {"a": {"target_pct": -0.14}, "b": {"target_pct": 0.36}}},
        ],
    }


def test_resolve_returns_anchors_target_price_to_live_current():
    r = resolve_returns(_book(), {"AAA": 95.0})  # BBB uses target_pct, needs no price
    assert abs(r["asset_ret"]["asset 1"]["a"] - 1.1052631) < 1e-6
    assert abs(r["asset_ret"]["asset 1"]["b"] - (-0.3157894)) < 1e-6
    assert r["asset_ret"]["asset 2"]["a"] == -0.14   # target_pct passthrough
    assert r["asset_ret"]["asset 2"]["b"] == 0.36
    assert r["probs"] == {"a": 0.9, "b": 0.1}
    assert r["weights"] == {"asset 1": 0.53, "asset 2": 0.47}
    assert r["skipped"] == []


def test_resolve_returns_skips_target_price_asset_without_price():
    r = resolve_returns(_book(), {})  # no live price for AAA -> asset 1 unpriceable
    assert "asset 1" in r["skipped"]
    assert "asset 1" not in r["asset_ret"]
    assert "asset 1" not in r["weights"]
    assert "asset 2" in r["asset_ret"]      # pct-based asset still resolves


# ── validate_book ───────────────────────────────────────────────────────────────

def test_validate_book_ok():
    assert validate_book(_book()) == []


def test_validate_book_flags_bad_probs_and_weights():
    bad = _book()
    bad["scenarios"][0]["prob"] = 0.5   # probs now sum to 0.6
    bad["assets"][0]["weight"] = 0.10   # weights now sum to 0.57
    problems = validate_book(bad)
    assert any("probab" in p.lower() for p in problems)
    assert any("weight" in p.lower() for p in problems)


def test_validate_book_flags_missing_outcome():
    bad = _book()
    del bad["assets"][0]["outcomes"]["b"]   # asset 1 missing scenario b
    problems = validate_book(bad)
    assert any("asset 1" in p and "b" in p for p in problems)
