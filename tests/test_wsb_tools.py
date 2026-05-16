from tools.wsb_tools import (
    extract_tickers_from_text, analyze_wsb_signals, SECTOR_KEYWORDS
)

def test_extract_tickers_dollar_sign():
    text = "Just bought $NVDA and $AAPL calls, not financial advice"
    tickers = extract_tickers_from_text(text)
    assert "NVDA" in tickers
    assert "AAPL" in tickers

def test_extract_tickers_excludes_common_words():
    text = "$I $AM $NOT $A ticker but $GME is"
    tickers = extract_tickers_from_text(text)
    assert "GME" in tickers
    assert "I" not in tickers
    assert "AM" not in tickers
    assert "NOT" not in tickers

def test_analyze_wsb_signals_structure():
    posts = [
        {"title": "$GME to the moon! Short squeeze incoming", "score": 1000, "text": "SI is 25%, gamma squeeze setup"},
        {"title": "Semiconductor stocks pumping, $NVDA $AMD on fire", "score": 500, "text": "AI demand driving chip stocks"},
        {"title": "$AAPL earnings play", "score": 200, "text": "Buying $AAPL calls"},
    ]
    result = analyze_wsb_signals(posts)
    assert "trending_tickers" in result
    assert "sector_hype" in result
    assert "squeeze_candidates" in result
    gme_entry = next((t for t in result["trending_tickers"] if t["ticker"] == "GME"), None)
    assert gme_entry is not None
    assert gme_entry["squeeze_flag"] is True

def test_sector_keywords_defined():
    assert "Semiconductors" in SECTOR_KEYWORDS
    assert "AI" in SECTOR_KEYWORDS
    assert "Defense" in SECTOR_KEYWORDS
