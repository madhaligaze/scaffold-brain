import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional


class SessionState(str, Enum):
    """Session lifecycle state."""

    SCANNING = "SCANNING"
    LOCKED = "LOCKED"
    PLANNING = "PLANNING"
    EXPORTED = "EXPORTED"


@dataclass
class LockInfo:
    locked_at: float
    reason: str
    world_revision: str
    readiness: Dict[str, Any]
    thresholds: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["LockInfo"]:
        if not isinstance(data, dict):
            return None
        return cls(
            locked_at=float(data.get("locked_at", time.time())),
            reason=str(data.get("reason", "unknown")),
            world_revision=str(data.get("world_revision", "")),
            readiness=data.get("readiness", {}) or {},
            thresholds=data.get("thresholds", {}) or {},
        )


def compute_world_revision(session_dict: Dict[str, Any]) -> str:
    """Compute compact deterministic world revision hash for cache/UI."""
    scene = (session_dict.get("scene_context") or {}) if isinstance(session_dict, dict) else {}
    payload = {
        "frames": len(session_dict.get("frames", []) if isinstance(session_dict, dict) else []),
        "anchors": len(scene.get("anchor_points") or []),
        "ar_points": len(scene.get("all_ar_points") or []),
        "detected_objects": len(scene.get("all_detected_objects") or []),
        "point_cloud": len(scene.get("point_cloud") or []),
        "updated_at": session_dict.get("updated_at") if isinstance(session_dict, dict) else None,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]
