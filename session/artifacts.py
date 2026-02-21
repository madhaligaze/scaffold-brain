from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dirs(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Any) -> None:
    ensure_dirs(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_bytes(path: Path, data: bytes) -> None:
    ensure_dirs(path.parent)
    path.write_bytes(data)


def save_numpy(path: Path, arr: np.ndarray) -> None:
    ensure_dirs(path.parent)
    np.save(path, arr)


def list_revisions(session_root: Path) -> list[str]:
    world_root = session_root / "world"
    if not world_root.exists():
        return []
    return sorted([p.name for p in world_root.iterdir() if p.is_dir()])


def prune_ndjson_tail(path: Path, *, max_bytes: int = 5_000_000, max_lines: int = 20_000) -> dict[str, int]:
    """Keep only the tail of an NDJSON file to enforce size/line limits."""
    if max_bytes <= 0 and max_lines <= 0:
        return {"kept_lines": 0, "dropped_lines": 0}

    if not path.exists() or not path.is_file():
        return {"kept_lines": 0, "dropped_lines": 0}

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"kept_lines": 0, "dropped_lines": 0}

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return {"kept_lines": 0, "dropped_lines": 0}

    keep = lines
    if max_lines > 0 and len(keep) > max_lines:
        keep = keep[-max_lines:]

    if max_bytes > 0:
        while keep and (sum(len(x) for x in keep) + (len(keep) - 1)) > max_bytes:
            keep.pop(0)

    dropped = max(0, len(lines) - len(keep))
    try:
        ensure_dirs(path.parent)
        path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    except Exception:
        pass
    return {"kept_lines": int(len(keep)), "dropped_lines": int(dropped)}


def prune_dir_by_age(dir_path: Path, *, max_age_days: int = 14, include_files: bool = True) -> dict[str, int]:
    """Delete files (and empty directories) older than max_age_days by mtime."""
    import os
    import time

    if max_age_days <= 0:
        return {"deleted": 0, "kept": 0}

    cutoff = float(time.time()) - float(max_age_days) * 86400.0
    deleted = 0
    kept = 0

    if not dir_path.exists():
        return {"deleted": 0, "kept": 0}

    for root_dir, dirs, files in os.walk(dir_path, topdown=False):
        rp = Path(root_dir)
        if include_files:
            for fn in files:
                p = rp / fn
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink(missing_ok=True)
                        deleted += 1
                    else:
                        kept += 1
                except Exception:
                    kept += 1
        for dn in dirs:
            dp = rp / dn
            try:
                if dp.is_dir() and not any(dp.iterdir()) and dp.stat().st_mtime < cutoff:
                    dp.rmdir()
                    deleted += 1
            except Exception:
                pass

    return {"deleted": int(deleted), "kept": int(kept)}
