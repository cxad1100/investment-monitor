"""asset name / model ticker-guess -> yfinance ticker (no network)."""
from tools import asset_resolve
from tools.asset_resolve import resolve_asset, _norm


def test_norm_strips_case_and_punctuation():
    assert _norm("S&P 500") == "sp500"
    assert _norm("s&p500") == "sp500"
    assert _norm("  Bitcoin ") == "bitcoin"


def test_resolve_index_alias():
    assert resolve_asset("S&P 500") == "^GSPC"
    assert resolve_asset("the S&P") == "^GSPC"


def test_resolve_crypto_and_commodity_alias():
    assert resolve_asset("Bitcoin") == "BTC-USD"
    assert resolve_asset("gold") == "GC=F"


def test_resolve_megacap_name():
    assert resolve_asset("Nvidia") == "NVDA"


def test_resolve_falls_back_to_ticker_guess():
    assert resolve_asset("Some Obscure Co", "OBSC") == "OBSC"


def test_resolve_prefers_known_alias_over_guess():
    assert resolve_asset("Bitcoin", "WRONG") == "BTC-USD"


def test_resolve_ignores_null_guess():
    assert resolve_asset("totally unknown widget", "null") is None
    assert resolve_asset("totally unknown widget", None) is None


def test_resolve_uses_known_company_names(monkeypatch):
    # reversed COMPANY_NAMES / UNIVERSE path, faked to stay deterministic
    monkeypatch.setattr(asset_resolve, "_known_names", lambda: {"someobscureco": "SOC.DE"})
    assert resolve_asset("Some Obscure Co") == "SOC.DE"
