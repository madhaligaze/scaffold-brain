"""Raw input capture for auditability (Stage 13).

Persist raw client inputs (frames + measurements) so any structure/BOM can be
audited back to the raw observations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from modules.world_snapshot import DEFAULT_SNAPSHOT_DIR, SnapshotRef


INCOMING_DIRNAME = "incoming_raw"


def _incoming_root(session_id: str, root_dir: str = DEFAULT_SNAPSHOT_DIR, *, create: bool = True) -> Path:
    p = Path(root_dir) / session_id / INCOMING_DIRNAME
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _gzip_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


@dataclass
class SavedFrameRaw:
    dir_path: str
    meta_gz: str
    rgb_path: str
    depth_path: Optional[str]
    conf_path: Optional[str]
    rgb_sha256: str
    depth_sha256: Optional[str]


def save_incoming_frame(
    *,
    session_id: str,
    frame_id: str,
    rgb_bytes: bytes,
    depth_bytes: Optional[bytes],
    conf_bytes: Optional[bytes],
    meta: Dict[str, Any],
    root_dir: str = DEFAULT_SNAPSHOT_DIR,
) -> SavedFrameRaw:
    ts = int(time.time() * 1000)
    safe_frame = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (frame_id or "frame"))
    d = _incoming_root(session_id, root_dir) / "frames" / f"{safe_frame}_{ts}"
    d.mkdir(parents=True, exist_ok=True)

    rgb_path = d / "rgb.bin"
    rgb_path.write_bytes(rgb_bytes)

    depth_path = None
    conf_path = None
    if depth_bytes is not None:
        depth_path = d / "depth.bin"
        depth_path.write_bytes(depth_bytes)
    if conf_bytes is not None:
        conf_path = d / "confidence.bin"
        conf_path.write_bytes(conf_bytes)

    meta_payload = dict(meta or {})
    meta_payload.update(
        {
            "session_id": session_id,
            "frame_id": frame_id,
            "saved_at_ms": ts,
            "rgb_sha256": _sha256(rgb_bytes),
            "depth_sha256": _sha256(depth_bytes) if depth_bytes else None,
            "confidence_sha256": _sha256(conf_bytes) if conf_bytes else None,
            "paths": {
                "rgb": str(rgb_path),
                "depth": str(depth_path) if depth_path else None,
                "confidence": str(conf_path) if conf_path else None,
            },
        }
    )
    meta_gz = d / "meta.json.gz"
    _gzip_write_json(meta_gz, meta_payload)

    return SavedFrameRaw(
        dir_path=str(d),
        meta_gz=str(meta_gz),
        rgb_path=str(rgb_path),
        depth_path=str(depth_path) if depth_path else None,
        conf_path=str(conf_path) if conf_path else None,
        rgb_sha256=meta_payload["rgb_sha256"],
        depth_sha256=meta_payload["depth_sha256"],
    )


def save_incoming_measurements(*, session_id: str, payload: Dict[str, Any], root_dir: str = DEFAULT_SNAPSHOT_DIR) -> str:
    ts = int(time.time() * 1000)
    d = _incoming_root(session_id, root_dir) / "measurements"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"measurements_{ts}.json.gz"
    _gzip_write_json(path, payload)
    return str(path)


def finalize_incoming_raw_to_revision(*, session_id: str, revision: str, root_dir: str = DEFAULT_SNAPSHOT_DIR) -> Dict[str, Any]:
    incoming = _incoming_root(session_id, root_dir, create=False)
    if not incoming.exists():
        return {"session_id": session_id, "revision": revision, "moved": 0, "paths": []}

    ref = SnapshotRef(session_id=session_id, revision=revision, root_dir=root_dir)
    target = ref.artifacts_dir / "raw_inputs"
    target.mkdir(parents=True, exist_ok=True)

    moved = 0
    moved_paths = []

    frames_dir = incoming / "frames"
    if frames_dir.exists():
        for sub in sorted(frames_dir.iterdir()):
            if not sub.is_dir():
                continue
            dst = target / "frames" / sub.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = target / "frames" / f"{sub.name}_{int(time.time())}"
            shutil.move(str(sub), str(dst))
            moved += 1
            moved_paths.append(str(dst))

    meas_dir = incoming / "measurements"
    if meas_dir.exists():
        for f in sorted(meas_dir.iterdir()):
            if not f.is_file():
                continue
            dst = target / "measurements" / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = target / "measurements" / f"{f.stem}_{int(time.time())}{f.suffix}"
            shutil.move(str(f), str(dst))
            moved += 1
            moved_paths.append(str(dst))

    try:
        for p in sorted(incoming.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except Exception:
                    pass
        try:
            incoming.rmdir()
        except Exception:
            pass
    except Exception:
        pass

    index = {
        "session_id": session_id,
        "revision": revision,
        "moved": moved,
        "paths": moved_paths,
        "created_at": time.time(),
    }
    (target / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def list_revision_raw_inputs(*, session_id: str, revision: str, root_dir: str = DEFAULT_SNAPSHOT_DIR) -> Dict[str, Any]:
    ref = SnapshotRef(session_id=session_id, revision=revision, root_dir=root_dir)
    base = ref.artifacts_dir / "raw_inputs"
    if not base.exists():
        return {"session_id": session_id, "revision": revision, "items": []}

    items = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            items.append(
                {
                    "rel": str(p.relative_to(ref.artifacts_dir)),
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                }
            )
    return {"session_id": session_id, "revision": revision, "items": items}
