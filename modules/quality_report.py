"""Stage 12: Quality report.

This report is meant to be *human-reviewable* and *revision-bound*.
It answers:
  - is the world model ready/locked at this revision?
  - what reprojection metrics were observed?
  - where is the model weak (unknown, misses, mismatches)?
  - what policy was used for unknown space in planning (forbid/buffer)?

The report is stored as an artifact under the snapshot revision and is also
registered in the revision manifest.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from modules.revision_artifacts import artifact_path
from modules.export_manifest import add_artifact, add_event


def build_quality_report(
    *,
    session_id: str,
    revision: str,
    session: Any,
    planned_elements_count: int = 0,
    unknown_policy: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = getattr(session, "scene_context", None)
    lifecycle_state = getattr(session, "lifecycle_state", None)
    readiness = getattr(ctx, "last_readiness", None) if ctx else None
    reprojection = getattr(ctx, "last_reprojection", None) if ctx else None
    scan_plan = getattr(ctx, "last_scan_plan", None) if ctx else None

    report: Dict[str, Any] = {
        "session_id": session_id,
        "revision": revision,
        "created_at": time.time(),
        "lifecycle_state": getattr(lifecycle_state, "value", str(lifecycle_state)) if lifecycle_state else getattr(session, "status", "UNKNOWN"),
        "world_locked": bool(getattr(session, "world_locked", False)),
        "mesh_version": getattr(session, "locked_mesh_version", None) or getattr(session, "mesh_version", None),
        "planned_elements_count": int(planned_elements_count or 0),
        "unknown_policy": (unknown_policy or "").lower() or None,
        "readiness": readiness or {},
        "reprojection": reprojection or {},
        "scan_plan": scan_plan or {},
    }

    # Quick summary flags for dashboards.
    rep = report["reprojection"] or {}
    report["summary"] = {
        "ready_to_lock": bool((readiness or {}).get("ready_to_lock")) if readiness else None,
        "samples": rep.get("samples"),
        "hit_rate": rep.get("hit_rate"),
        "miss_rate": rep.get("miss_rate"),
        "mismatch_rate": rep.get("mismatch_rate"),
        "median_abs_error_m": rep.get("median_abs_error_m"),
    }
    return report


def save_quality_report(
    *,
    session_id: str,
    revision: str,
    session: Any,
    planned_elements_count: int = 0,
    unknown_policy: Optional[str] = None,
) -> str:
    report = build_quality_report(
        session_id=session_id,
        revision=revision,
        session=session,
        planned_elements_count=planned_elements_count,
        unknown_policy=unknown_policy,
    )

    filename = f"quality_report_{revision}.json"
    out_path = artifact_path(session_id=session_id, revision=revision, filename=filename)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    add_artifact(session_id=session_id, revision=revision, kind="quality_report", filename=filename, meta={"planned_elements_count": planned_elements_count})
    add_event(session_id=session_id, revision=revision, name="quality_report_saved", payload={"file": filename})
    return str(out_path)
