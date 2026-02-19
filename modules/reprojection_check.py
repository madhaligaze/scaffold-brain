"""Stage 5: Reprojection consistency check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class ScanSuggestion:
    x: float
    y: float
    z: float
    reason: str
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z), "reason": self.reason, "weight": float(self.weight)}


@dataclass
class ReprojectionResult:
    ok: bool
    sampled: int
    hits: int
    misses: int
    mismatches: int
    median_abs_error_m: float
    p90_abs_error_m: float
    miss_rate: float
    mismatch_rate: float
    suggestions: List[Dict[str, Any]]


class ReprojectionAPIResult:
    """Return type that works for both API and tests.

    `main.py` expects a dict-like object with `.get('reprojection')` and
    `.get('scan_suggestions')`.

    Some unit tests (and legacy callers) expect attribute access like
    `.samples`, `.hit_rate`, `.median_abs_error_m`.
    """

    def __init__(self, reprojection: Dict[str, Any], scan_suggestions: List[Dict[str, Any]]):
        self.reprojection = reprojection
        self.scan_suggestions = scan_suggestions

    def get(self, key: str, default=None):
        if key == "reprojection":
            return self.reprojection
        if key == "scan_suggestions":
            return self.scan_suggestions
        return default

    @property
    def samples(self) -> int:
        return int(self.reprojection.get("samples", 0))

    @property
    def hit_rate(self) -> float:
        return float(self.reprojection.get("hit_rate", 0.0))

    @property
    def miss_rate(self) -> float:
        return float(self.reprojection.get("miss_rate", 0.0))

    @property
    def mismatch_rate(self) -> float:
        return float(self.reprojection.get("mismatch_rate", 0.0))

    @property
    def median_abs_error_m(self) -> float:
        return float(self.reprojection.get("median_abs_error_m", 0.0))


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n <= 1e-8:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def check_reprojection(
    *,
    voxel_world,
    # Support both raw uint16 depth buffer and pre-decoded meters array.
    # Tests (and some integrations) pass `depth_m` directly.
    depth_bytes: bytes | None = None,
    depth_m: np.ndarray | None = None,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx_px: float | None = None,
    cy_px: float | None = None,
    # Back-compat aliases
    cx: float | None = None,
    cy: float | None = None,
    pose7: List[float],
    depth_scale: float = 1000.0,
    pixel_step: int = 12,
    max_range: float = 8.0,
    mismatch_abs_thresh_m: float = 0.12,
    # Legacy aliases
    max_depth: float | None = None,
    error_threshold_m: float | None = None,
    miss_rate_threshold: float | None = None,
    max_suggestions: int = 120,
) -> ReprojectionAPIResult:
    if max_depth is not None:
        max_range = float(max_depth)
    if error_threshold_m is not None:
        mismatch_abs_thresh_m = float(error_threshold_m)
    # miss_rate_threshold is used by legacy callers/tests for ok/not ok; we keep it for compatibility.
    if cx_px is None:
        cx_px = cx
    if cy_px is None:
        cy_px = cy

    if voxel_world is None or (depth_bytes is None and depth_m is None):
        return ReprojectionAPIResult({"enabled": False}, [])
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0 or len(pose7) < 7:
        return ReprojectionAPIResult({"enabled": False, "error": "bad_inputs"}, [])

    if depth_m is None:
        if depth_bytes is None or len(depth_bytes) < width * height * 2:
            return ReprojectionAPIResult({"enabled": False, "error": "bad_depth"}, [])
        depth_u16 = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(height, width)
        depth_m = depth_u16.astype(np.float32) / float(depth_scale)
    else:
        depth_m = np.asarray(depth_m, dtype=np.float32)
        if depth_m.shape != (height, width):
            return ReprojectionAPIResult({"enabled": False, "error": "bad_depth_shape"}, [])

    rot = _quat_to_rot(float(pose7[3]), float(pose7[4]), float(pose7[5]), float(pose7[6]))
    cam_t = np.array([float(pose7[0]), float(pose7[1]), float(pose7[2])], dtype=np.float32)

    abs_errors: List[float] = []
    hits = misses = mismatches = unknown_rays = samples = 0
    suggestions: List[ScanSuggestion] = []
    step = max(1, int(pixel_step))

    for v in range(0, height, step):
        for u in range(0, width, step):
            d_obs = float(depth_m[v, u])
            if d_obs <= 0.0 or d_obs > float(max_range):
                continue
            if cx_px is None or cy_px is None:
                return ReprojectionAPIResult({"enabled": False, "error": "missing_intrinsics"}, [])
            xc = (float(u) - float(cx_px)) * d_obs / float(fx)
            yc = (float(v) - float(cy_px)) * d_obs / float(fy)
            dir_c = np.array([xc, yc, d_obs], dtype=np.float32)
            norm = float(np.linalg.norm(dir_c))
            if norm <= 1e-6:
                continue
            dir_c /= norm
            dir_w = rot @ dir_c
            dir_w_norm = float(np.linalg.norm(dir_w))
            if dir_w_norm <= 1e-6:
                continue
            dir_w /= dir_w_norm

            samples += 1
            exp = voxel_world.raycast_distance(
                origin_world=(float(cam_t[0]), float(cam_t[1]), float(cam_t[2])),
                direction_world=(float(dir_w[0]), float(dir_w[1]), float(dir_w[2])),
                max_dist=float(max_range),
                unknown_is_blocked=False,
                return_unknown_hit=True,
            )

            if exp is None:
                misses += 1
                if len(suggestions) < int(max_suggestions):
                    p_hit = cam_t + (dir_w * float(d_obs))
                    suggestions.append(ScanSuggestion(float(p_hit[0]), float(p_hit[1]), float(p_hit[2]), reason="model_missing", weight=1.0))
                continue

            d_exp, hit_is_unknown = exp
            if hit_is_unknown:
                unknown_rays += 1

            hits += 1
            err = abs(float(d_exp) - float(d_obs))
            abs_errors.append(err)

            if err > float(mismatch_abs_thresh_m):
                mismatches += 1
                if len(suggestions) < int(max_suggestions):
                    p_hit = cam_t + (dir_w * float(d_obs))
                    suggestions.append(ScanSuggestion(float(p_hit[0]), float(p_hit[1]), float(p_hit[2]), reason="depth_mismatch", weight=min(3.0, 1.0 + err / 0.2)))

    reproj: Dict[str, Any] = {
        "enabled": True,
        "samples": int(samples),
        "hit_rate": float(hits / samples) if samples else 0.0,
        "miss_rate": float(misses / samples) if samples else 0.0,
        "mismatch_rate": float(mismatches / hits) if hits else 0.0,
        "unknown_ray_ratio": float(unknown_rays / hits) if hits else 0.0,
    }
    if abs_errors:
        arr = np.array(abs_errors, dtype=np.float32)
        reproj.update({
            "median_abs_error_m": float(np.percentile(arr, 50)),
            "p90_abs_error_m": float(np.percentile(arr, 90)),
            "p95_abs_error_m": float(np.percentile(arr, 95)),
        })
    else:
        reproj.update({"median_abs_error_m": None, "p90_abs_error_m": None, "p95_abs_error_m": None})

    uniq = {}
    for s in suggestions:
        key = (round(s.x, 2), round(s.y, 2), round(s.z, 2), s.reason)
        if key in uniq:
            uniq[key].weight = max(uniq[key].weight, s.weight)
        else:
            uniq[key] = s

    out_suggestions = [s.to_dict() for s in sorted(uniq.values(), key=lambda x: -x.weight)[: int(max_suggestions)]]
    api = ReprojectionAPIResult(reprojection=reproj, scan_suggestions=out_suggestions)
    if miss_rate_threshold is not None:
        api.reprojection["ok"] = bool(api.miss_rate <= float(miss_rate_threshold))
    return api


def run_reprojection_check(**kwargs) -> ReprojectionResult:
    camera_pose = kwargs.get("camera_pose") or kwargs.get("pose7") or [0, 0, 0, 0, 0, 0, 1]
    res = check_reprojection(
        voxel_world=kwargs.get("voxel_world"),
        depth_bytes=kwargs.get("depth_bytes"),
        width=int(kwargs.get("width", 0)),
        height=int(kwargs.get("height", 0)),
        fx=float(kwargs.get("fx", 0)),
        fy=float(kwargs.get("fy", 0)),
        cx_px=float(kwargs.get("cx_px", 0)),
        cy_px=float(kwargs.get("cy_px", 0)),
        pose7=camera_pose,
        depth_scale=float(kwargs.get("depth_scale", 1000.0)),
        pixel_step=int(kwargs.get("sample_step_px", 12)),
        max_range=float(kwargs.get("max_range_m", 8.0)),
        mismatch_abs_thresh_m=float(kwargs.get("mismatch_thresh_m", 0.12)),
        max_suggestions=int(kwargs.get("max_suggestions", 120)),
    )
    r = res.get("reprojection", {})
    s = res.get("scan_suggestions", [])
    return ReprojectionResult(
        ok=not bool(r.get("error")),
        sampled=int(r.get("samples", 0)),
        hits=int(round(float(r.get("hit_rate", 0.0)) * max(1, int(r.get("samples", 0))))),
        misses=int(round(float(r.get("miss_rate", 0.0)) * max(1, int(r.get("samples", 0))))),
        mismatches=int(round(float(r.get("mismatch_rate", 0.0)) * max(1, int(r.get("samples", 0))))),
        median_abs_error_m=float(r.get("median_abs_error_m") or 0.0),
        p90_abs_error_m=float(r.get("p90_abs_error_m") or 0.0),
        miss_rate=float(r.get("miss_rate", 0.0)),
        mismatch_rate=float(r.get("mismatch_rate", 0.0)),
        suggestions=[{"point": {"x": x.get("x"), "y": x.get("y"), "z": x.get("z")}, "reason": x.get("reason"), "severity": x.get("weight", 1.0)} for x in s],
    )
