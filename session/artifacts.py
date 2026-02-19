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
