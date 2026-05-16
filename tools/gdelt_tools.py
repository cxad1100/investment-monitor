"""GDELT API v2 — country conflict indices and recent event volumes."""

import requests
from datetime import datetime, timedelta

GDELT_TIMELINE = "https://api.gdeltproject.org/api/v2/timeline/timeline"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

MONITORED_REGIONS = [
    {"name": "Middle East", "query": "(Iran OR Israel OR Gaza OR Lebanon OR Syria OR Yemen) (war OR conflict OR attack OR military OR strike)"},
    {"name": "Eastern Europe", "query": "(Ukraine OR Russia OR NATO OR Donbas) (war OR attack OR offensive OR ceasefire OR troops)"},
    {"name": "Asia Pacific", "query": "(Taiwan OR China OR North Korea) (military OR strait OR invasion OR missile OR tension)"},
    {"name": "Sub-Saharan Africa", "query": "(Sudan OR Congo OR Mali OR Sahel) (coup OR conflict OR military OR civil war)"},
]


def _fetch_gdelt_volume(query: str, days: int = 14) -> list[float]:
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")
    end = datetime.now().strftime("%Y%m%d%H%M%S")
    params = {
        "query": query,
        "mode": "timelinevol",
        "format": "json",
        "STARTDATETIME": start,
        "ENDDATETIME": end,
        "SMOOTHING": "3",
    }
    try:
        resp = requests.get(GDELT_TIMELINE, params=params, headers=HEADERS, timeout=30)
        data = resp.json()
        timeline = data.get("timeline", [{}])[0].get("data", [])
        return [float(p.get("value", 0)) for p in timeline]
    except Exception:
        return []


def fetch_regional_conflict_indices() -> list[dict]:
    """Fetch conflict event volume for each monitored region."""
    results = []
    for region in MONITORED_REGIONS:
        volumes = _fetch_gdelt_volume(region["query"])
        if not volumes or len(volumes) < 4:
            results.append({
                "region": region["name"],
                "conflict_score": 0.4,
                "trend": "unknown",
                "source": "gdelt",
            })
            continue
        recent = volumes[-7:] if len(volumes) >= 7 else volumes
        prev = volumes[-14:-7] if len(volumes) >= 14 else volumes[:len(volumes)//2]
        recent_avg = sum(recent) / len(recent)
        prev_avg = sum(prev) / len(prev) if prev else recent_avg
        conflict_score = min(1.0, recent_avg / 100.0)
        if recent_avg > prev_avg * 1.25:
            trend = "escalating"
        elif recent_avg < prev_avg * 0.75:
            trend = "de-escalating"
        else:
            trend = "stable"
        results.append({
            "region": region["name"],
            "conflict_score": round(conflict_score, 3),
            "trend": trend,
            "recent_avg_volume": round(recent_avg, 2),
            "source": "gdelt",
        })
    return results
