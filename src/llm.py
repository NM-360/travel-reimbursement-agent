"""
Plain Ollama client — no agent framework.

Talks to a local Ollama server (https://ollama.com) over its REST API using only
`requests`. Exposes a single `chat()` function that supports tool calling, which
is everything the agent loop in `agent.py` needs.

Env vars:
  OLLAMA_HOST   default http://localhost:11434
  OLLAMA_MODEL  default qwen3:8b
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class LLMError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def is_available() -> bool:
    """Cheap health check used by the UI/CLI to fail fast with a clear message."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    format_json: bool = False,
    timeout: int = 180,
) -> dict[str, Any]:
    """
    Send one chat turn to Ollama and return the assistant `message` dict.

    The returned message may contain `tool_calls` (a list of requested tool
    invocations) instead of, or in addition to, `content`. The caller is
    responsible for executing tools and appending results.
    """
    payload: dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
        # qwen3 is a hybrid reasoning model; disable <think> to keep tool calls clean.
        "think": False,
    }
    if tools:
        payload["tools"] = tools
    if format_json:
        payload["format"] = "json"

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running? ({exc})"
        ) from exc

    if resp.status_code != 200:
        raise LLMError(f"Ollama returned {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    if "message" not in data:
        raise LLMError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")
    return data["message"]


def embed(text: str, *, model: str | None = None, timeout: int = 60) -> list[float]:
    """
    Return the embedding vector for `text` using Ollama's embeddings endpoint
    (default model: nomic-embed-text). Used by the policy_lookup tool for
    semantic retrieval.
    """
    payload = {"model": model or OLLAMA_EMBED_MODEL, "prompt": text}
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach Ollama embeddings at {OLLAMA_HOST} ({exc})") from exc

    if resp.status_code != 200:
        raise LLMError(f"Ollama embeddings returned {resp.status_code}: {resp.text[:300]}")

    vec = resp.json().get("embedding")
    if not vec:
        raise LLMError(f"No embedding in response: {resp.text[:300]}")
    return vec
