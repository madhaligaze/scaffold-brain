"""Stage 12: Revision-bound export manifest.

Keeps an auditable, append-only-ish manifest of artifacts produced for a given
world snapshot revision (BOM, mesh, debug dumps, quality reports, etc.).

The goal is that *any* exported/generated file can be traced back to:
  - session_id
  - snapshot revision
  - generation/export parameters
  - creating endpoint
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from modules.world_snapshot import DEFAULT_SNAPSHOT_DIR, SnapshotRef


MANIFEST_FILENAME = "manifest.json"


def _manifest_path(session_id: str, revision: str, snapshot_root: str = DEFAULT_SNAPSHOT_DIR) -> Path:
    ref = SnapshotRef(session_id=session_id, revision=revision, root_dir=snapshot_root)
    ref.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return ref.artifacts_dir / MANIFEST_FILENAME


def load_manifest(session_id: str, revision: str, snapshot_root: str = DEFAULT_SNAPSHOT_DIR) -> Dict[str, Any]:
    p = _manifest_path(session_id, revision, snapshot_root)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # fall through to new manifest
            pass
    return {
        "session_id": session_id,
        "revision": revision,
        "created_at": time.time(),
        "updated_at": time.time(),
        "artifacts": [],
        "events": [],
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def add_artifact(
    *,
    session_id: str,
    revision: str,
    kind: str,
    filename: str,
    meta: Optional[Dict[str, Any]] = None,
    snapshot_root: str = DEFAULT_SNAPSHOT_DIR,
) -> Dict[str, Any]:
    """Add/merge artifact record by filename."""
    manifest = load_manifest(session_id, revision, snapshot_root)
    artifacts = list(manifest.get("artifacts") or [])
    now = time.time()
    meta = dict(meta or {})
    rec = {
        "kind": str(kind),
        "filename": str(filename),
        "created_at": now,
        "meta": meta,
    }
    # de-dup by filename (latest wins)
    out = [a for a in artifacts if str(a.get("filename")) != str(filename)]
    out.append(rec)
    manifest["artifacts"] = out
    manifest["updated_at"] = now
    p = _manifest_path(session_id, revision, snapshot_root)
    _atomic_write_json(p, manifest)
    return rec


def add_event(
    *,
    session_id: str,
    revision: str,
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    snapshot_root: str = DEFAULT_SNAPSHOT_DIR,
) -> Dict[str, Any]:
    """Append an event record (never de-dups)."""
    manifest = load_manifest(session_id, revision, snapshot_root)
    events = list(manifest.get("events") or [])
    now = time.time()
    ev = {"name": str(name), "at": now, "payload": dict(payload or {})}
    events.append(ev)
    manifest["events"] = events
    manifest["updated_at"] = now
    p = _manifest_path(session_id, revision, snapshot_root)
    _atomic_write_json(p, manifest)
    return ev
