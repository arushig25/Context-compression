from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .env import load_dotenv


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
_RETRY_IN_RE = re.compile(r"retry in\s+([0-9.]+)s", re.IGNORECASE)


@dataclass
class GeminiResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None


def _extract_text(resp: Dict[str, Any]) -> str:
    # Response shape: candidates[0].content.parts[*].text
    cands = resp.get("candidates") or []
    if not cands:
        return ""
    content = (cands[0].get("content") or {})
    parts = content.get("parts") or []
    out = []
    for p in parts:
        t = p.get("text")
        if isinstance(t, str):
            out.append(t)
    return "".join(out).strip()


def generate_content(
    *,
    system_instruction: str,
    user_text: str,
    model: str,
    temperature: float = 0.6,
    max_output_tokens: int = 500,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    response_json_schema: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> GeminiResponse:
    """
    Minimal stdlib client for Gemini `generateContent`.

    Official REST endpoint:
      POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent

    Auth: `x-goog-api-key: $GEMINI_API_KEY`
    """
    # Allow storing secrets in a local `.env` file without extra dependencies.
    load_dotenv()
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    base_url = (base_url or os.environ.get("GEMINI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/v1beta/models/{model}:generateContent"

    generation_config: Dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_output_tokens),
    }
    is_gemma = model.startswith("gemma")
    # Gemma on AI Studio supports neither responseMimeType nor responseJsonSchema.
    if response_mime_type and not is_gemma:
        generation_config["responseMimeType"] = response_mime_type
    if response_schema is not None and not is_gemma:
        generation_config["responseSchema"] = response_schema
    if response_json_schema is not None and not is_gemma:
        # Structured outputs via responseJsonSchema only supported on Gemini models.
        generation_config["responseJsonSchema"] = response_json_schema

    # Gemma models on AI Studio don't support system_instruction — fold it into the user turn.
    if is_gemma:
        combined_user_text = f"{system_instruction}\n\n{user_text}"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": combined_user_text}]}],
            "generationConfig": generation_config,
        }
    else:
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": generation_config,
        }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
    attempt = 0
    last_err: Optional[str] = None
    while True:
        attempt += 1
        req = urllib.request.Request(url=url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            # Try to surface a helpful message and optionally auto-retry on 429.
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(e)
            last_err = err_body

            if e.code == 429 and attempt <= max_retries:
                # Honor Retry-After header if present, otherwise parse the JSON message.
                retry_after = e.headers.get("Retry-After")
                sleep_s: Optional[float] = None
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except ValueError:
                        sleep_s = None
                if sleep_s is None:
                    try:
                        j = json.loads(err_body)
                        msg = (j.get("error") or {}).get("message") or ""
                        m = _RETRY_IN_RE.search(msg)
                        if m:
                            sleep_s = float(m.group(1))
                    except Exception:
                        sleep_s = None
                if sleep_s is None:
                    sleep_s = min(2.0 * attempt, 10.0)

                # Small buffer to avoid hammering the limit boundary.
                time.sleep(max(0.0, float(sleep_s) + 0.25))
                continue

            raise RuntimeError(f"Gemini HTTP {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini connection error: {e}") from e

    resp = json.loads(body)
    return GeminiResponse(text=_extract_text(resp), raw=resp)
