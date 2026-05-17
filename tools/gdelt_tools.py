"""GDELT Doc API v2 — regional conflict intensity from article volume and tone."""

import time
import requests

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-research/1.0)"}

MONITORED_REGIONS = [
    {
        "name": "Middle East",
        "query": "(Iran OR Israel OR Gaza OR Lebanon OR Syria OR Yemen) (war OR conflict OR attack OR military OR strike)",
    },
    {
        "name": "Eastern Europe",
        "query": "(Ukraine OR Russia OR NATO OR Donbas) (war OR attack OR offensive OR ceasefire OR troops)",
    },
    {
        "name": "Asia Pacific",
        "query": "(Taiwan OR China OR North Korea) (military OR strait OR invasion OR missile OR tension)",
    },
    {
        "name": "Sub-Saharan Africa",
        "query": "(Sudan OR Congo OR Mali OR Sahel) (coup OR conflict OR military)",
    },
]

# Baseline: ~50 articles/region/week = conflict_score ~0.5
_ARTICLE_SCALE = 100.0


def _fetch_articles(query: str, max_records: int = 100) -> list[dict]:
    try:
        resp = requests.get(
            GDELT_DOC,
            params={"query": query, "mode": "artlist", "format": "json",
                    "maxrecords": str(max_records)},
            headers=HEADERS,
            timeout=20,
        )
        if resp.status_code == 429:
            time.sleep(15)
            resp = requests.get(
                GDELT_DOC,
                params={"query": query, "mode": "artlist", "format": "json",
                        "maxrecords": str(max_records)},
                headers=HEADERS,
                timeout=20,
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception:
        return []


def fetch_regional_conflict_indices() -> list[dict]:
    """Return conflict intensity for each monitored region.

    conflict_score: 0.0 (quiet) to 1.0 (intense). Derived from article count
    and average tone (negative tone = more conflict). Returns 0.3 on API failure.
    """
    results = []
    for i, region in enumerate(MONITORED_REGIONS):
        if i > 0:
            time.sleep(6)  # GDELT limit: 1 req/5s

        articles = _fetch_articles(region["query"])
        if not articles:
            results.append({
                "region": region["name"],
                "conflict_score": 0.3,
                "trend": "unknown",
                "article_count": 0,
                "source": "gdelt",
            })
            continue

        count = len(articles)
        tones = []
        for a in articles:
            t = a.get("tone", "")
            try:
                tones.append(float(t))
            except (ValueError, TypeError):
                pass

        avg_tone = sum(tones) / len(tones) if tones else 0.0
        # tone < 0 → negative/conflict; normalize to 0-1
        tone_score = max(0.0, min(1.0, (-avg_tone + 5) / 20.0))
        count_score = min(1.0, count / _ARTICLE_SCALE)
        conflict_score = round(0.6 * count_score + 0.4 * tone_score, 3)

        if avg_tone < -3:
            trend = "escalating"
        elif avg_tone > 3:
            trend = "de-escalating"
        else:
            trend = "stable"

        results.append({
            "region": region["name"],
            "conflict_score": conflict_score,
            "trend": trend,
            "article_count": count,
            "avg_tone": round(avg_tone, 2),
            "source": "gdelt",
        })

    return results
