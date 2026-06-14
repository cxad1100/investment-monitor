"""Turn a vidsum prose summary into structured scenarios JSON via a local LLM.

The LLM call is injected as ``generate_fn(prompt) -> str`` (mirrors vidsum's
``summarize(segments, generate_fn)``) so this is testable with no network. Local
models fumble strict JSON, so parsing is defensive: strip code fences, fall back
to the outermost {...}, retry once with a reminder, and on total failure return a
flagged record rather than raising — one bad video must never abort a batch.
"""
from __future__ import annotations

import json
import re

# Per-video schema we ask the model to emit. Returns asset-level scenarios (each
# asset's own bull/base/bear), plus macro theses that have no single price.
_SCHEMA = (
    '{\n'
    '  "assets": [\n'
    '    {"name": "<asset/company>", "ticker_guess": "<best-guess ticker or null>",\n'
    '     "scenarios": [\n'
    '       {"label": "<bull|base|bear|...>", "target_price": <number or null>,\n'
    '        "target_pct": <decimal move e.g. 0.25 for +25%, or null>,\n'
    '        "rationale": "<short why>", "horizon": "<e.g. 6-12m / 2026-Q4 / null>",\n'
    '        "stated_probability": <0..1 if the speaker gives odds, else null>}\n'
    '     ]}\n'
    '  ],\n'
    '  "macro_theses": [\n'
    '    {"thesis": "<claim with no single price>", "direction": "<bullish|bearish|neutral>",\n'
    '     "horizon": "<...>", "stated_probability": <0..1 or null>}\n'
    '  ],\n'
    '  "overall_stance": "<one phrase>"\n'
    '}'
)

_JSON_REMINDER = (
    "\n\nIMPORTANT: respond with ONLY the JSON object, no prose, no code fences."
)


def _prompt(summary: str, upload_date: str | None) -> str:
    when = f" The video was uploaded on {upload_date}." if upload_date else ""
    return (
        "You extract a structured map of possible market outcomes from a summary of "
        f"a market-analysis video.{when}\n"
        "Use ONLY claims present in the summary — do not invent assets or numbers. "
        "Give price targets as target_price when an absolute level is stated, or "
        "target_pct (a decimal, +0.25 = +25%) when a move is stated; otherwise null. "
        "Set stated_probability only if the speaker actually gives odds.\n\n"
        "Return JSON of exactly this shape:\n"
        f"{_SCHEMA}\n\n"
        f"SUMMARY:\n{summary}"
    )


def _parse_json(text: str):
    """Best-effort extract a JSON object from a possibly-chatty LLM reply."""
    if not text:
        return None
    candidates = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    if "{" in text and "}" in text:                       # outermost braces span
        candidates.append(text[text.index("{"): text.rindex("}") + 1])
    candidates.append(text.strip())
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_scenarios(summary: str, generate_fn, *, retries: int = 1,
                      upload_date: str | None = None) -> dict:
    """Extract scenarios; never raises on a bad reply.

    On success returns the parsed object with ``extraction_failed=False`` and
    ``assets``/``macro_theses`` guaranteed to be lists. On total parse failure
    returns ``{"assets": [], "macro_theses": [], "extraction_failed": True,
    "raw": <last reply>}``.
    """
    prompt = _prompt(summary, upload_date)
    text = generate_fn(prompt)
    obj = _parse_json(text)
    attempts = 0
    while obj is None and attempts < retries:
        attempts += 1
        text = generate_fn(prompt + _JSON_REMINDER)
        obj = _parse_json(text)

    if obj is None:
        return {"assets": [], "macro_theses": [], "extraction_failed": True, "raw": text}

    obj.setdefault("assets", [])
    obj.setdefault("macro_theses", [])
    if not isinstance(obj["assets"], list):
        obj["assets"] = []
    if not isinstance(obj["macro_theses"], list):
        obj["macro_theses"] = []
    obj["extraction_failed"] = False
    return obj
