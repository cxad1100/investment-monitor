"""No-network tests for prose-summary -> scenarios JSON extraction.

The Ollama call is injected as ``generate_fn`` (like vidsum's summarize), so the
robust-parse + retry behaviour is tested with canned strings.
"""
from tools.scenario_extract import extract_scenarios, _parse_json


def _seq(*responses):
    """generate_fn that returns each response in turn and records call count."""
    calls = {"n": 0}
    it = iter(responses)

    def fn(_prompt):
        calls["n"] += 1
        return next(it)

    fn.calls = calls
    return fn


VALID = '{"assets": [{"name": "Nvidia", "ticker_guess": "NVDA", "scenarios": ' \
        '[{"label": "bull", "target_pct": 0.4}]}], "macro_theses": []}'


def test_parse_plain_json():
    obj = _parse_json(VALID)
    assert obj["assets"][0]["name"] == "Nvidia"


def test_parse_fenced_json_with_prose():
    text = "Sure, here is the data:\n```json\n" + VALID + "\n```\nHope that helps!"
    obj = _parse_json(text)
    assert obj["assets"][0]["ticker_guess"] == "NVDA"


def test_parse_json_with_leading_prose_no_fence():
    text = "Here you go: " + VALID + " -- done"
    obj = _parse_json(text)
    assert obj is not None
    assert obj["assets"][0]["name"] == "Nvidia"


def test_parse_returns_none_on_garbage():
    assert _parse_json("I cannot help with that.") is None


def test_extract_parses_first_try():
    fn = _seq(VALID)
    out = extract_scenarios("summary text", fn)
    assert out["extraction_failed"] is False
    assert out["assets"][0]["name"] == "Nvidia"
    assert fn.calls["n"] == 1


def test_extract_retries_once_then_succeeds():
    fn = _seq("no json here", VALID)
    out = extract_scenarios("summary text", fn)
    assert out["extraction_failed"] is False
    assert fn.calls["n"] == 2          # retried exactly once


def test_extract_flags_failure_after_retries():
    fn = _seq("nope", "still nope")
    out = extract_scenarios("summary text", fn)
    assert out["extraction_failed"] is True
    assert out["assets"] == []
    assert out["macro_theses"] == []
    assert out["raw"] == "still nope"  # last response preserved for debugging


def test_extract_defaults_missing_keys():
    fn = _seq('{"assets": []}')          # model omitted macro_theses
    out = extract_scenarios("summary text", fn)
    assert out["macro_theses"] == []
    assert out["extraction_failed"] is False
