from __future__ import annotations

import re
from typing import Dict, List


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """
    Rough tokenizer-agnostic estimate.

    For hackathon-style metrics, it's usually enough to compare relative reductions.
    Swap this out for a model tokenizer if you need exact counts.
    """
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def measure_compression(raw_history: List[str], compressed_ctx: str) -> Dict[str, float]:
    raw_tokens = sum(estimate_tokens(t) for t in raw_history)
    compressed_tokens = estimate_tokens(compressed_ctx)
    ratio = (raw_tokens / compressed_tokens) if compressed_tokens else float("inf")
    return {
        "raw_tokens": float(raw_tokens),
        "compressed_tokens": float(compressed_tokens),
        "ratio": float(ratio),
        "turns": float(len(raw_history)),
    }

