"""Unit tests for the pure WKN/ISIN/slippage helpers (no network)."""

import pytest

from tools.wkn_resolve import isin_from_wkn, spread_to_slippage


@pytest.mark.parametrize("wkn,isin", [
    ("555750", "DE0005557508"),   # Deutsche Telekom
    ("A1EWWW", "DE000A1EWWW0"),   # Adidas
    ("716460", "DE0007164600"),   # SAP
    ("BASF11", "DE000BASF111"),   # BASF
])
def test_isin_from_wkn(wkn, isin):
    assert isin_from_wkn(wkn) == isin


def test_isin_from_wkn_lowercase_and_whitespace():
    assert isin_from_wkn(" a1ewww ") == "DE000A1EWWW0"


def test_spread_to_slippage_monotonic_and_clamped():
    tight = spread_to_slippage(100.0, 100.1)     # 0.05% half-spread ≈ 5 bps
    wide = spread_to_slippage(100.0, 110.0)      # ~4.5% half-spread → clamp hi
    assert tight < wide
    assert spread_to_slippage(100.0, 100.0) == 2          # zero spread → lo clamp
    assert spread_to_slippage(100.0, 200.0) == 50         # huge spread → hi clamp


def test_spread_to_slippage_bad_input():
    assert spread_to_slippage(None, None) == 50           # unparseable → hi
    assert spread_to_slippage(0.0, 0.0) == 50             # mid<=0 → hi
    assert spread_to_slippage(101.0, 100.0) == 50         # crossed → hi
