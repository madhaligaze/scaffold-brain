"""World snapshots (Stage 9/10).

This module makes generation/export reproducible and auditable by committing
an immutable snapshot of the scene.

Why this exists
---------------
Without snapshots, the backend can silently change the world model between
"generate" and "export" calls (more frames arrive, voxel carving changes, etc.).
That leads to non-repeatable results and makes it impossible to audit.

Snapshot layout
---------------
{root}/{session_id}/{revision}/
  meta.json
  session.json.gz            (Session.to_dict() sans large blobs)
  voxel_world.json.gz        (VoxelWorld: occupied + free + params)
  artifacts/                 (BOM, mesh, debug dumps, etc)

The `revision` is content-based, derived from voxel occupancy/free + core
scene metadata. It is stable across restarts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SNAPSHOT_DIR = "/tmp/ai_brain_snapshots"


def _compute_revision(
    session_id: str,
    voxel_payload: Dict[str, Any],
    session_payload: Dict[str, Any],
) -> str:
    """Compute a stable revision from key snapshot payload pieces."""
    h = hashlib.sha1()
    h.update(session_id.encode("utf-8"))

    occ = voxel_payload.get("occupied", []) or []
    free = voxel_payload.get("free", []) or []
    h.update(json.dumps(occ, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(free, sort_keys=True).encode("utf-8"))

    core = {
        "resolution": voxel_payload.get("resolution"),
        "bounds_min": voxel_payload.get("bounds_min"),
        "bounds_max": voxel_payload.get("bounds_max"),
        "scene_summary": (session_payload.get("scene_context") or {}),
    }
    h.update(json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8"))

    return h.hexdigest()[:16]


def _gzip_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _gzip_json_read(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class SnapshotRef:
    session_id: str
    revision: str
    root_dir: str = DEFAULT_SNAPSHOT_DIR

    @property
    def dir(self) -> Path:
        return Path(self.root_dir) / self.session_id / self.revision

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def session_path(self) -> Path:
        return self.dir / "session.json.gz"

    @property
    def voxel_path(self) -> Path:
        return self.dir / "voxel_world.json.gz"

    @property
    def artifacts_dir(self) -> Path:
        return self.dir / "artifacts"


def serialize_voxel_world(voxel_world: Any) -> Dict[str, Any]:
    """Serialize VoxelWorld in a stable, JSON-safe format."""
    try:
        occ = sorted([list(v) for v in getattr(voxel_world, "occupied", set())])
        free = sorted([list(v) for v in getattr(voxel_world, "free", set())])
        return {
            "resolution": float(getattr(voxel_world, "resolution", 0.05)),
            "bounds_min": list(getattr(voxel_world, "bounds_min", (-4.0, -4.0, -1.0))),
            "bounds_max": list(getattr(voxel_world, "bounds_max", (4.0, 4.0, 3.0))),
            "occupied": occ,
            "free": free,
            "last_depth_stats": getattr(voxel_world, "get_last_depth_stats", lambda: {})(),
        }
    except Exception as e:
        return {"error": f"serialize_voxel_world failed: {e}"}


def deserialize_voxel_world(payload: Dict[str, Any]) -> Any:
    """Build a new VoxelWorld instance from snapshot payload."""
    from modules.voxel_world import VoxelWorld

    vw = VoxelWorld(
        resolution=float(payload.get("resolution", 0.05)),
        bounds_min=tuple(payload.get("bounds_min", (-4.0, -4.0, -1.0))),
        bounds_max=tuple(payload.get("bounds_max", (4.0, 4.0, 3.0))),
    )

    occ = payload.get("occupied", []) or []
    free = payload.get("free", []) or []
    vw.occupied = set(tuple(map(int, v)) for v in occ)
    vw.free = set(tuple(map(int, v)) for v in free)

    return vw


def commit_snapshot(
    session_id: str,
    session_dict: Dict[str, Any],
    voxel_world: Any,
    root_dir: str = DEFAULT_SNAPSHOT_DIR,
    reason: str = "manual",
) -> SnapshotRef:
    """Commit a snapshot and return its reference."""

    voxel_payload = serialize_voxel_world(voxel_world)

    slim_session = dict(session_dict or {})
    if "frames" in slim_session and isinstance(slim_session["frames"], list):
        slim_session["frames"] = slim_session["frames"][-50:]
    if "generated_variants" in slim_session and isinstance(slim_session["generated_variants"], list):
        slim_session["generated_variants"] = slim_session["generated_variants"][-10:]

    revision = _compute_revision(session_id=session_id, voxel_payload=voxel_payload, session_payload=slim_session)
    ref = SnapshotRef(session_id=session_id, revision=revision, root_dir=root_dir)

    ref.dir.mkdir(parents=True, exist_ok=True)
    ref.artifacts_dir.mkdir(parents=True, exist_ok=True)

    _gzip_json_write(ref.session_path, slim_session)
    _gzip_json_write(ref.voxel_path, voxel_payload)

    meta = {
        "session_id": session_id,
        "revision": revision,
        "reason": reason,
        "created_at": __import__("time").time(),
        "voxel_counts": {
            "occupied": len(voxel_payload.get("occupied", []) or []),
            "free": len(voxel_payload.get("free", []) or []),
        },
    }
    ref.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return ref


def list_snapshots(session_id: str, root_dir: str = DEFAULT_SNAPSHOT_DIR) -> List[Dict[str, Any]]:
    base = Path(root_dir) / session_id
    if not base.exists():
        return []

    out: List[Dict[str, Any]] = []
    for rev_dir in base.iterdir():
        if not rev_dir.is_dir():
            continue
        meta_path = rev_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out.append(
            {
                "revision": rev_dir.name,
                "meta": meta,
                "path": str(rev_dir),
            }
        )

    out.sort(key=lambda x: (x.get("meta") or {}).get("created_at", 0), reverse=True)
    return out


def load_snapshot(session_id: str, revision: str, root_dir: str = DEFAULT_SNAPSHOT_DIR) -> Dict[str, Any]:
    """Load snapshot payloads and return a dict.

    Returns:
      {"session": dict, "voxel_payload": dict, "voxel_world": VoxelWorld, "ref": SnapshotRef}
    """
    ref = SnapshotRef(session_id=session_id, revision=revision, root_dir=root_dir)
    if not ref.dir.exists():
        raise FileNotFoundError(f"snapshot not found: {ref.dir}")

    session_payload = _gzip_json_read(ref.session_path)
    voxel_payload = _gzip_json_read(ref.voxel_path)
    voxel_world = deserialize_voxel_world(voxel_payload)

    return {"session": session_payload, "voxel_payload": voxel_payload, "voxel_world": voxel_world, "ref": ref}


def restore_voxel_world(target_voxel_world: Any, payload: Dict[str, Any]) -> None:
    """Overwrite an existing VoxelWorld from snapshot payload."""
    if not payload:
        return
    try:
        target_voxel_world.resolution = float(payload.get("resolution", getattr(target_voxel_world, "resolution", 0.05)))
        target_voxel_world.bounds_min = tuple(payload.get("bounds_min", getattr(target_voxel_world, "bounds_min", (-4.0, -4.0, -1.0))))
        target_voxel_world.bounds_max = tuple(payload.get("bounds_max", getattr(target_voxel_world, "bounds_max", (4.0, 4.0, 3.0))))
        occ = payload.get("occupied", []) or []
        free = payload.get("free", []) or []
        target_voxel_world.occupied = set(tuple(map(int, v)) for v in occ)
        target_voxel_world.free = set(tuple(map(int, v)) for v in free)
        if hasattr(target_voxel_world, "_grid"):
            try:
                target_voxel_world._grid = {tuple(map(int, v)): getattr(target_voxel_world, "OCCUPIED", 1) for v in occ}
                for v in free:
                    target_voxel_world._grid.setdefault(tuple(map(int, v)), getattr(target_voxel_world, "FREE", 0))
            except Exception:
                pass
    except Exception:
        return
