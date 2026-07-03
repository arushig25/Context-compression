from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class OpenAIResponse:
    text: str
    response_id: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


def _extract_text(resp: Dict[str, Any]) -> str:
    """
    Responses API returns an `output` array containing message/tool items.
    We aggregate all assistant `output_text` chunks.
    """
    out = []
    for item in resp.get("output", []) or []:
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                out.append(c["text"])
    return "".join(out).strip()


def create_response(
    *,
    instructions: str,
    user_text: str,
    model: str,
    previous_response_id: Optional[str] = None,
    temperature: float = 0.6,
    max_output_tokens: int = 500,
    store: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAIResponse:
    """
    Minimal stdlib client for OpenAI's Responses API.

    Docs: https://platform.openai.com/docs/api-reference/responses
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/responses"

    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_text,
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "store": bool(store),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8", errors="replace")
        except Exception:
            err = str(e)
        raise RuntimeError(f"OpenAI HTTP {e.code}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI connection error: {e}") from e

    resp = json.loads(body)
    return OpenAIResponse(text=_extract_text(resp), response_id=resp.get("id"), raw=resp)

