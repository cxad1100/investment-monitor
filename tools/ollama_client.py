"""Minimal local Ollama HTTP client (standard library only).

Ported from /code/vidsum (vidsum/ollama_client.py) so the monitor stays
self-contained — no cross-repo import, no API keys, no server. Used by
scenario_extract for the prose-summary -> structured-JSON pass.
"""
import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when ollama cannot be reached or returns an error."""


def _payload(prompt, model, num_ctx, temperature):
    return json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature},
        }
    ).encode("utf-8")


def generate(prompt, model, host="http://localhost:11434",
             num_ctx=8192, temperature=0.3, timeout=600):
    """Call ollama /api/generate and return the generated text."""
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=_payload(prompt, model, num_ctx, temperature),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace").strip()
        raise OllamaError(
            f"ollama returned HTTP {exc.code}: {body}. "
            f"If the model is missing, run: ollama pull {model}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"cannot reach ollama at {host} ({exc.reason}); start it with: ollama serve"
        ) from exc
    if "error" in data:
        raise OllamaError(str(data["error"]))
    return data.get("response", "")
