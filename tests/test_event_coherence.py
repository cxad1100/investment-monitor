import pytest
from event_extractor import (
    check_event_coherence, CoherenceError,
    normalize_complement_probabilities, flag_dependent_events
)

def test_complement_is_computed():
    events = [{"id": "A", "description": "War ends", "probability": 0.31, "asset_impacts": []}]
    result = normalize_complement_probabilities(events)
    assert result[0]["complement_probability"] == pytest.approx(0.69, abs=0.001)

def test_invalid_probability_raises():
    events = [{"id": "A", "description": "X", "probability": 1.5, "asset_impacts": []}]
    with pytest.raises(CoherenceError, match="probability must be 0.0-1.0"):
        check_event_coherence(events)

def test_mutually_exclusive_sum_over_one_raises():
    events = [
        {"id": "A", "description": "War starts", "probability": 0.7,
         "mutually_exclusive_with": ["B"], "asset_impacts": []},
        {"id": "B", "description": "War ends", "probability": 0.8,
         "mutually_exclusive_with": ["A"], "asset_impacts": []},
    ]
    with pytest.raises(CoherenceError, match="mutually exclusive"):
        check_event_coherence(events)

def test_valid_events_pass():
    events = [
        {"id": "A", "description": "Iran escalates", "probability": 0.44,
         "mutually_exclusive_with": ["B"], "asset_impacts": []},
        {"id": "B", "description": "Iran ceasefire", "probability": 0.31,
         "mutually_exclusive_with": ["A"], "asset_impacts": []},
    ]
    result = check_event_coherence(events)
    assert len(result) == 2

def test_flag_dependent_events():
    events = [
        {"id": "A", "description": "Fed cuts rates", "probability": 0.67, "asset_impacts": []},
        {"id": "B", "description": "Tech stocks rally on rate cut", "probability": 0.55,
         "depends_on": "A", "asset_impacts": []},
    ]
    result = flag_dependent_events(events)
    assert result[1]["dependency_note"] == "Depends on Event A"
