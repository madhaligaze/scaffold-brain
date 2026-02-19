from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])


def _quat_angle_deg(q1_xyzw: list[float], q2_xyzw: list[float]) -> float:
    # angle between rotations via quaternion dot (abs for double-cover)
    d = abs(_dot(q1_xyzw, q2_xyzw))
    d = max(-1.0, min(1.0, d))
    # cos(theta/2) = d
    theta = 2.0 * math.acos(d)
    return float(theta * (180.0 / math.pi))


def evaluate_pose_step(
    prev_pose: Dict[str, Any] | None,
    pose: Dict[str, Any] | None,
    *,
    max_trans_good_m: float = 0.25,
    max_trans_bad_m: float = 0.75,
    max_rot_good_deg: float = 10.0,
    max_rot_bad_deg: float = 35.0,
) -> Tuple[str, List[dict]]:
    """
    STAGE C: drift/jump detection using consecutive poses.
    Returns (quality, reasons) where quality is GOOD/WARN/BAD.
    """
    if prev_pose is None or pose is None:
        return "UNKNOWN", []
    p1 = prev_pose.get("position")
    p2 = pose.get("position")
    q1 = prev_pose.get("quaternion")
    q2 = pose.get("quaternion")
    if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) == 3 and len(p2) == 3):
        return "UNKNOWN", [{"type": "POSE_MISSING_POSITION"}]
    if not (isinstance(q1, (list, tuple)) and isinstance(q2, (list, tuple)) and len(q1) == 4 and len(q2) == 4):
        return "UNKNOWN", [{"type": "POSE_MISSING_QUAT"}]

    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    dz = float(p2[2]) - float(p1[2])
    trans = math.sqrt(dx * dx + dy * dy + dz * dz)
    rot_deg = _quat_angle_deg([float(x) for x in q1], [float(x) for x in q2])

    reasons: List[dict] = []
    quality = "GOOD"

    if trans > max_trans_good_m:
        quality = "WARN"
        reasons.append({"type": "TRANS_JITTER", "trans_m": float(trans), "thr_good_m": float(max_trans_good_m)})
    if rot_deg > max_rot_good_deg:
        quality = "WARN"
        reasons.append({"type": "ROT_JITTER", "rot_deg": float(rot_deg), "thr_good_deg": float(max_rot_good_deg)})

    if trans > max_trans_bad_m or rot_deg > max_rot_bad_deg:
        quality = "BAD"
        reasons.append(
            {
                "type": "POSE_JUMP",
                "trans_m": float(trans),
                "rot_deg": float(rot_deg),
                "thr_bad_m": float(max_trans_bad_m),
                "thr_bad_deg": float(max_rot_bad_deg),
            }
        )

    return quality, reasons
