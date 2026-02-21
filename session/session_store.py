from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from session.artifacts import ensure_dirs, save_bytes, save_json, prune_dir_by_age, prune_ndjson_tail
from trace.decision_trace import trace_to_ndjson_bytes


class SessionStore:
    def __init__(self, sessions_root: str = "sessions") -> None:
        self.root = Path(sessions_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        session_id = str(uuid4())
        ensure_dirs(self.root / session_id)
        ensure_dirs(self.root / session_id / "frames")
        ensure_dirs(self.root / session_id / "world")
        ensure_dirs(self.root / session_id / "exports")
        ensure_dirs(self.root / session_id / "anchors")
        return session_id

    def session_root(self, session_id: str) -> Path:
        return self.root / session_id

    def prune_sessions(self, *, max_age_days: int) -> dict[str, int]:
        """Delete whole session directories older than max_age_days by mtime."""
        try:
            max_age_days = int(max_age_days)
        except Exception:
            max_age_days = 14

        if max_age_days <= 0:
            return {"deleted": 0, "kept": 0}

        cutoff = float(time.time()) - float(max_age_days) * 86400.0
        deleted = 0
        kept = 0
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            if d.name.startswith("_"):
                kept += 1
                continue
            try:
                mtime = float(d.stat().st_mtime)
            except Exception:
                kept += 1
                continue

            if mtime < cutoff:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    deleted += 1
                except Exception:
                    kept += 1
            else:
                kept += 1

        return {"deleted": int(deleted), "kept": int(kept)}

    def save_frame(
        self,
        session_id: str,
        frame_id: str,
        meta: dict[str, Any],
        rgb_bytes: bytes,
        depth_bytes: bytes | None = None,
        pointcloud_bytes: bytes | None = None,
        *,
        validated_meta: dict[str, Any] | None = None,
    ) -> None:
        base = ensure_dirs(self.session_root(session_id) / "frames" / frame_id)
        save_json(base / "meta.json", meta)
        if validated_meta is not None:
            save_json(base / "validated_meta.json", validated_meta)
        save_bytes(base / "rgb.jpg", rgb_bytes)
        if depth_bytes is not None:
            save_bytes(base / "depth.u16", depth_bytes)
        if pointcloud_bytes is not None:
            save_bytes(base / "pointcloud.bin", pointcloud_bytes)

    def save_anchors(self, session_id: str, anchors: list[dict[str, Any]]) -> None:
        save_json(self.session_root(session_id) / "anchors" / "anchors.json", anchors)

    def lock_revision(
        self,
        session_id: str,
        world_model_state: dict[str, Any],
        overlays: dict[str, Any],
        trace: list[dict[str, Any]],
        env_mesh: str | None = None,
        env_mesh_bytes: bytes | None = None,
    ) -> str:
        rev_id = str(uuid4())
        base = ensure_dirs(self.session_root(session_id) / "world" / rev_id)
        save_json(base / "world_state.json", world_model_state)
        save_json(base / "overlays.json", overlays)
        save_json(base / "trace.json", trace)
        save_bytes(base / "trace.ndjson", trace_to_ndjson_bytes(trace))
        if env_mesh_bytes is not None:
            save_bytes(base / "env_mesh.obj", env_mesh_bytes)
        elif env_mesh:
            save_bytes(base / "env_mesh.obj", env_mesh.encode("utf-8"))
        return rev_id

    def save_export(self, session_id: str, rev_id: str, bundle: dict[str, Any]) -> None:
        base = ensure_dirs(self.session_root(session_id) / "exports" / rev_id)
        save_json(base / "scene_bundle.json", bundle)
        # Backward compatibility for legacy code paths.
        save_json(self.session_root(session_id) / "exports" / "latest.json", {"rev_id": rev_id})

    def load_export(self, session_id: str, rev_id: str) -> dict[str, Any] | None:
        path = self.session_root(session_id) / "exports" / rev_id / "scene_bundle.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_exports(self, session_id: str) -> list[str]:
        root = self.session_root(session_id) / "exports"
        if not root.exists():
            return []
        out: list[str] = []
        for d in root.iterdir():
            if d.is_dir():
                out.append(d.name)
        out.sort()
        return out


    def global_root(self) -> Path:
        return ensure_dirs(self.root / "_global")

    def global_telemetry_dir(self) -> Path:
        return ensure_dirs(self.global_root() / "telemetry")

    def prune_telemetry(
        self,
        *,
        max_age_days: int = 14,
        max_file_bytes: int = 5_000_000,
        max_lines: int = 20_000,
    ) -> dict[str, int]:
        deleted = 0
        kept = 0

        gdir = self.global_telemetry_dir()
        res = prune_dir_by_age(gdir, max_age_days=max_age_days, include_files=True)
        deleted += res.get("deleted", 0)
        kept += res.get("kept", 0)
        for p in [gdir / "crash_reports.ndjson", gdir / "client_reports.ndjson"]:
            out = prune_ndjson_tail(p, max_bytes=max_file_bytes, max_lines=max_lines)
            kept += out.get("kept_lines", 0)
            deleted += out.get("dropped_lines", 0)

        for d in self.root.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            tdir = d / "telemetry"
            if not tdir.exists():
                continue
            res = prune_dir_by_age(tdir, max_age_days=max_age_days, include_files=True)
            deleted += res.get("deleted", 0)
            kept += res.get("kept", 0)
            out = prune_ndjson_tail(tdir / "events.ndjson", max_bytes=max_file_bytes, max_lines=max_lines)
            kept += out.get("kept_lines", 0)
            deleted += out.get("dropped_lines", 0)

        return {"deleted": int(deleted), "kept": int(kept)}
