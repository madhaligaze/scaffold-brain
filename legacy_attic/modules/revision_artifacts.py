"""Revision-bound artifacts (Stage 10)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from modules.world_snapshot import DEFAULT_SNAPSHOT_DIR, SnapshotRef


def _artifact_dir(session_id: str, revision: str, snapshot_root: str = DEFAULT_SNAPSHOT_DIR) -> Path:
    ref = SnapshotRef(session_id=session_id, revision=revision, root_dir=snapshot_root)
    ref.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return ref.artifacts_dir


def save_json_artifact(
    *,
    session_id: str,
    revision: str,
    name: str,
    payload: Dict[str, Any],
    snapshot_root: str = DEFAULT_SNAPSHOT_DIR,
) -> str:
    adir = _artifact_dir(session_id, revision, snapshot_root)
    ts = int(time.time())
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    path = adir / f"{safe_name}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return str(path)


def save_bytes_artifact(
    *,
    session_id: str,
    revision: str,
    filename: str,
    data: bytes,
    snapshot_root: str = DEFAULT_SNAPSHOT_DIR,
) -> str:
    adir = _artifact_dir(session_id, revision, snapshot_root)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in filename)
    path = adir / safe_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def save_file_artifact(
    *,
    session_id: str,
    revision: str,
    src_path: str,
    dst_filename: Optional[str] = None,
    snapshot_root: str = DEFAULT_SNAPSHOT_DIR,
) -> str:
    adir = _artifact_dir(session_id, revision, snapshot_root)
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(src_path)
    dst = adir / (dst_filename or src.name)
    dst.write_bytes(src.read_bytes())
    return str(dst)


def artifact_path(
    *,
    session_id: str,
    revision: str,
    filename: str,
    snapshot_root: str = DEFAULT_SNAPSHOT_DIR,
) -> str:
    adir = _artifact_dir(session_id, revision, snapshot_root)
    return str(adir / filename)
