"""Unit tests for the on-disk data buffer (TTL cache + last-good fallback)."""

import pandas as pd
import pytest

from tools.data_buffer import (
    cached_current_prices,
    cached_market_caps,
    cached_price_history,
)


@pytest.fixture
def bufdir(tmp_path):
    return tmp_path / "buffer"


class Counter:
    """Fake fetch that records call count and returns a scripted value."""

    def __init__(self, value):
        self.calls = 0
        self.value = value

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.value() if callable(self.value) else self.value


def test_history_ttl_hit_then_force(bufdir):
    frame = pd.DataFrame({"AAA.F": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    fetch = Counter(frame)
    a = cached_price_history(["AAA.F"], buffer_dir=bufdir, _fetch=fetch)
    b = cached_price_history(["AAA.F"], buffer_dir=bufdir, _fetch=fetch)  # within TTL
    assert fetch.calls == 1                      # second call served from cache
    pd.testing.assert_frame_equal(a, b)
    cached_price_history(["AAA.F"], buffer_dir=bufdir, _fetch=fetch, force=True)
    assert fetch.calls == 2                       # force bypasses cache


def test_history_ttl_expired_refetches(bufdir):
    frame = pd.DataFrame({"AAA.F": [1.0]}, index=pd.bdate_range("2024-01-01", periods=1))
    fetch = Counter(frame)
    cached_price_history(["AAA.F"], buffer_dir=bufdir, _fetch=fetch, ttl_hours=12)
    cached_price_history(["AAA.F"], buffer_dir=bufdir, _fetch=fetch, ttl_hours=0)  # expired
    assert fetch.calls == 2


def test_market_caps_keep_last_good_on_partial_fetch(bufdir):
    full = Counter({"JPM": 5e11, "BAC": 3e11})
    cached_market_caps(["JPM", "BAC"], buffer_dir=bufdir, _fetch=full)
    # later fetch fails for BAC (returns only JPM); force to bypass TTL
    partial = Counter({"JPM": 6e11})
    out = cached_market_caps(["JPM", "BAC"], buffer_dir=bufdir, _fetch=partial, force=True)
    assert out["JPM"] == 6e11                     # new value wins
    assert out["BAC"] == 3e11                     # last-good retained


def test_current_prices_force_falls_back_to_buffer_not_none(bufdir):
    holdings = {"JPM": {}, "BAC": {}}
    good = Counter({"JPM": 100.0, "BAC": 50.0})
    cached_current_prices(holdings, buffer_dir=bufdir, _fetch=good, force=True)
    # BAC now fails its live fetch
    flaky = Counter({"JPM": 110.0, "BAC": None})
    prices, stale, as_of = cached_current_prices(holdings, buffer_dir=bufdir,
                                                 _fetch=flaky, force=True)
    assert prices["JPM"] == 110.0
    assert prices["BAC"] == 50.0                  # last-good, never None/avg-cost
    assert "BAC" in stale and "JPM" not in stale
    assert as_of is not None


def test_current_prices_nonforce_serves_buffer_without_fetch(bufdir):
    holdings = {"JPM": {}}
    seed = Counter({"JPM": 100.0})
    cached_current_prices(holdings, buffer_dir=bufdir, _fetch=seed, force=True)
    later = Counter({"JPM": 999.0})
    prices, stale, _ = cached_current_prices(holdings, buffer_dir=bufdir, _fetch=later)
    assert later.calls == 0                        # warm buffer → no network
    assert prices["JPM"] == 100.0
    assert stale == {}


def test_current_prices_cold_buffer_fetches_even_nonforce(bufdir):
    holdings = {"JPM": {}}
    seed = Counter({"JPM": 100.0})
    prices, _, _ = cached_current_prices(holdings, buffer_dir=bufdir, _fetch=seed)
    assert seed.calls == 1                          # cold buffer must fetch
    assert prices["JPM"] == 100.0
