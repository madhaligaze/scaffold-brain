"""Android -> Backend measurement protocol (Stage 13).

The goal is to make user measurements (AR anchors, ruler/tape distances,
fiducial markers) explicit, versioned, validated, and auditable.

Design notes
------------
We keep the protocol conservative:
  - all metric units are meters
  - explicit coordinate frames (world coordinates only)
  - stable IDs

Compatibility
-------------
The backend accepts protocol versions <= SUPPORTED_PROTOCOL_VERSION.
If a client sends a higher protocol version, reject the request with 409.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


SUPPORTED_PROTOCOL_VERSION = 1


def is_finite_number(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except Exception:
        return False


def validate_protocol_version(protocol_version: int) -> Tuple[bool, str]:
    try:
        pv = int(protocol_version)
    except Exception:
        return False, "protocol_version must be int"

    if pv <= 0:
        return False, "protocol_version must be positive"
    if pv > SUPPORTED_PROTOCOL_VERSION:
        return (
            False,
            f"Unsupported protocol_version={pv}. Server supports <= {SUPPORTED_PROTOCOL_VERSION}.",
        )
    return True, ""


def validate_world_point(p: Dict[str, Any], *, name: str = "point") -> Tuple[bool, str]:
    if not isinstance(p, dict):
        return False, f"{name} must be object with x,y,z"
    for k in ("x", "y", "z"):
        if k not in p or not is_finite_number(p.get(k)):
            return False, f"{name}.{k} must be finite number"
    return True, ""


def validate_distance_m(distance_m: Any, *, name: str = "distance_m") -> Tuple[bool, str]:
    if not is_finite_number(distance_m):
        return False, f"{name} must be finite number"
    d = float(distance_m)
    if d < 0:
        return False, f"{name} must be >= 0"
    if d > 200.0:
        return False, f"{name} is unrealistically large (>200m)"
    return True, ""


def validate_anchor_id(anchor_id: Any) -> Tuple[bool, str]:
    if not anchor_id or not isinstance(anchor_id, str):
        return False, "anchor_id must be non-empty string"
    if len(anchor_id) > 64:
        return False, "anchor_id too long"
    return True, ""


def validate_marker_id(marker_id: Any) -> Tuple[bool, str]:
    if not marker_id or not isinstance(marker_id, str):
        return False, "marker_id must be non-empty string"
    if len(marker_id) > 64:
        return False, "marker_id too long"
    return True, ""


def point_tuple(p: Dict[str, Any]) -> Tuple[float, float, float]:
    return (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))


def distance_between(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax, ay, az = point_tuple(a)
    bx, by, bz = point_tuple(b)
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))
