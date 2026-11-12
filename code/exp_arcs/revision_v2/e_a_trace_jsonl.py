"""Read stage traces using only literal newlines as JSONL delimiters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def trace_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # str.splitlines() also splits on Unicode U+0085/U+2028 embedded in valid
    # JSON strings; only an actual newline delimits these JSONL records.
    return [json.loads(row) for row in path.read_text().split("\n") if row.strip()]
