"""Unit tests for the pairs-trading universe and engine math."""

import numpy as np
import pandas as pd
import pytest

from tools.pairs_universe import UNIVERSE, candidate_pairs


def test_universe_entries_complete():
    for tk, meta in UNIVERSE.items():
        assert set(meta) == {"sector", "currency", "slippage_bps"}, tk
        assert meta["currency"] in ("USD", "EUR")
        assert meta["slippage_bps"] in (5, 10, 15)


def test_candidate_pairs_same_sector_and_currency():
    pairs = candidate_pairs()
    assert ("BAC", "JPM") in pairs            # same sector, same currency
    for a, b in pairs:
        assert UNIVERSE[a]["sector"] == UNIVERSE[b]["sector"]
        assert UNIVERSE[a]["currency"] == UNIVERSE[b]["currency"]
    # never cross-sector or cross-currency
    assert ("NVDA", "SIE.DE") not in pairs
    assert ("JPM", "DBK.DE") not in pairs


def test_candidate_pairs_count_reasonable():
    n = len(candidate_pairs())
    assert 30 <= n <= 80                      # ~52 with the curated universe
