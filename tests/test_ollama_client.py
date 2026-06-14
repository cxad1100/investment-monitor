"""No-network tests for the local Ollama client (urlopen monkeypatched)."""
import io
import json

import pytest

from tools import ollama_client
from tools.ollama_client import generate, OllamaError


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, payload: dict, capture: dict | None = None):
    def fake_urlopen(req, timeout=None):
        if capture is not None:
            capture["req"] = req
            capture["body"] = json.loads(req.data)
        return _FakeResp(json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)


def test_generate_returns_response_text(monkeypatch):
    cap: dict = {}
    _patch_urlopen(monkeypatch, {"response": "hello world"}, cap)
    out = generate("my prompt", "some-model")
    assert out == "hello world"
    # request is shaped for ollama's /api/generate
    assert cap["req"].full_url.endswith("/api/generate")
    assert cap["body"]["model"] == "some-model"
    assert cap["body"]["prompt"] == "my prompt"
    assert cap["body"]["stream"] is False


def test_generate_raises_on_error_field(monkeypatch):
    _patch_urlopen(monkeypatch, {"error": "model not found"})
    with pytest.raises(OllamaError):
        generate("p", "missing-model")
