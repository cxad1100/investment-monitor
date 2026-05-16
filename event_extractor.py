"""
Event probability framework.
Structural validation for probabilistic events extracted from signals.
The actual event extraction is done by the Claude rating session reading signals.json.
"""


class CoherenceError(ValueError):
    pass


def normalize_complement_probabilities(events: list[dict]) -> list[dict]:
    """Set complement_probability = 1 - probability for each event."""
    for event in events:
        p = event.get("probability", 0.5)
        event["complement_probability"] = round(1.0 - p, 4)
    return events


def check_event_coherence(events: list[dict]) -> list[dict]:
    """
    Validate event probabilities:
    1. Each probability must be in [0.0, 1.0]
    2. Mutually exclusive event pairs must sum <= 1.0
    Raises CoherenceError on violation.
    """
    for event in events:
        p = event.get("probability", 0.5)
        if not (0.0 <= p <= 1.0):
            raise CoherenceError(
                f"Event {event['id']}: probability must be 0.0-1.0, got {p}"
            )

    event_map = {e["id"]: e for e in events}
    checked_pairs = set()
    for event in events:
        for excl_id in event.get("mutually_exclusive_with", []):
            pair = tuple(sorted([event["id"], excl_id]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            if excl_id in event_map:
                p_sum = event["probability"] + event_map[excl_id]["probability"]
                if p_sum > 1.0 + 1e-6:
                    raise CoherenceError(
                        f"Events {event['id']} and {excl_id} are mutually exclusive "
                        f"but probabilities sum to {p_sum:.3f} > 1.0"
                    )

    return events


def flag_dependent_events(events: list[dict]) -> list[dict]:
    """Add dependency notes so the rating agent avoids double-counting."""
    for event in events:
        dep = event.get("depends_on")
        if dep:
            event["dependency_note"] = f"Depends on Event {dep}"
    return events


def validate_and_normalize(events: list[dict]) -> list[dict]:
    """Full pipeline: check coherence, normalize complements, flag dependencies."""
    events = check_event_coherence(events)
    events = normalize_complement_probabilities(events)
    events = flag_dependent_events(events)
    return events


def build_empty_event_template() -> dict:
    """Return a blank event dict for the Claude session to fill in."""
    return {
        "id": "",
        "description": "",
        "probability": 0.5,
        "complement_description": "",
        "complement_probability": 0.5,
        "source": "polymarket|gdelt|inferred",
        "resolution_date": "",
        "mutually_exclusive_with": [],
        "depends_on": None,
        "asset_impacts": [
            {
                "sector": "",
                "region": "",
                "direction": "positive|negative",
                "magnitude": "strong|moderate|weak",
                "causal_chain": "event → mechanism → asset impact",
            }
        ],
    }
