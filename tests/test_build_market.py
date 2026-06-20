"""Criteria gate for the fresh bulk universe build (tools.build_market.apply_criteria).

Pure, no-network: feed synthetic EUR price + volume series and assert the liquidity /
price / collapse rules admit the right names. The network plumbing (bulk pre-screen,
EOD fetch, FX) is thin glue over already-tested primitives, so it isn't unit-tested here.
"""
import numpy as np
import pandas as pd

from tools.build_market import apply_criteria, MIN_OBS, PRICE_FLOOR


def _series(n=400, price=50.0, vol=100_000):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(price, index=idx, dtype=float), pd.Series(vol, index=idx, dtype=float)


def test_active_liquid_passes():
    px, vol = _series(price=50.0, vol=100_000)          # €5M/day turnover
    out = apply_criteria(px, vol, active=True)
    assert out is not None and out["delisting_date"] == ""
    assert out["med_turnover"] > 1_000_000


def test_active_too_short_rejected():
    px, vol = _series(n=MIN_OBS - 5)
    assert apply_criteria(px, vol, active=True) is None


def test_active_sub_euro_rejected():
    px, vol = _series(price=0.40, vol=10_000_000)        # liquid but penny
    assert px.tail(60).median() < PRICE_FLOOR
    assert apply_criteria(px, vol, active=True) is None


def test_active_illiquid_rejected():
    px, vol = _series(price=50.0, vol=200)               # €10k/day < €100k floor
    assert apply_criteria(px, vol, active=True) is None


def test_active_turnover_glitch_ceiling_rejected():
    px, vol = _series(price=50.0, vol=5_000_000_000)     # €250B/day > ceiling glitch
    assert apply_criteria(px, vol, active=True) is None


def test_dead_clean_collapse_passes():
    px, vol = _series(n=400, price=50.0, vol=100_000)
    px.iloc[-30:] = np.linspace(50, 5, 30)               # real collapse to ≤ ½ peak
    out = apply_criteria(px, vol, active=False)
    assert out is not None and out["delisting_date"] == str(px.index[-1].date())


def test_dead_no_collapse_rejected():
    px, vol = _series(price=50.0, vol=100_000)            # still at peak → not a death
    assert apply_criteria(px, vol, active=False) is None


def test_dead_up_glitch_rejected():
    px, vol = _series(n=400, price=50.0, vol=100_000)
    px.iloc[200] = 500.0                                  # +900% one-day glitch
    px.iloc[-30:] = np.linspace(50, 5, 30)               # then collapse
    assert apply_criteria(px, vol, active=False) is None
