from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _find_dotenv(start: Path) -> Optional[Path]:
    for p in (start, *start.parents):
        cand = p / ".env"
        if cand.is_file():
            return cand
    return None


def load_dotenv(path: str | Path | None = None) -> Optional[Path]:
    """
    Tiny dotenv loader (stdlib-only).
    - If path is None, searches for `.env` from CWD upwards.
    - Sets variables only if they are not already set in os.environ.

    Returns the loaded `.env` path or None if not found.
    """
    p = Path(path).expanduser().resolve() if path else _find_dotenv(Path.cwd())
    if not p or not p.is_file():
        return None

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_quotes(v)
        if not k:
            continue
        if k not in os.environ:
            os.environ[k] = v
    return p

