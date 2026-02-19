from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def audit_log_path(sessions_root: Path) -> Path:
    return sessions_root / "audit.ndjson"


def write_audit_event(sessions_root: Path, event: dict[str, Any]) -> None:
    p = audit_log_path(sessions_root)
    ev = dict(event)
    ev.setdefault("ts", float(time.time()))
    line = json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n"

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
