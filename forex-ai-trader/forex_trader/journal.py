"""Diario JSONL: ogni analisi, decisione, ordine ed esito con il relativo ragionamento."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


def _plain(obj: object) -> object:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class Journal:
    def __init__(self, path: str | Path | None = None, memory: int = 200) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: deque[dict[str, object]] = deque(maxlen=memory)

    def record(self, kind: str, at: datetime | None = None, **data: object) -> dict[str, object]:
        entry: dict[str, object] = {
            "at": (at or datetime.now(timezone.utc)).isoformat(),
            "kind": kind,
            **json.loads(json.dumps(data, default=_plain, ensure_ascii=False)),
        }
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def recent(self, kind: str, instrument: str | None = None, limit: int = 5) -> list[dict[str, object]]:
        out = [
            e for e in reversed(self.entries)
            if e["kind"] == kind and (instrument is None or e.get("instrument") == instrument)
        ]
        return out[:limit]
