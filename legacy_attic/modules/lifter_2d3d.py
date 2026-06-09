# modules/lifter_2d3d.py
"""
2D -> 3D lifting (Stage 2)
--------------------------

Given Det2D + (intrinsics, pose, depth or point cloud), estimate a Det3D:

- Prefer depth_map: sample depths inside bbox/mask, robust median, backproject pixels.
- Fallback point_cloud: select world points that project into bbox.

This module is intentionally conservative:
- If valid depth coverage is too low, returns None.
- Dimensions are estimated from 3D points (AABB in world); orientation optional and can
  be refined later (primitive fitting).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def pose7_to_matrix(pose7: List[float]) -> np.ndarray:
    # pose7: [tx,ty,tz,qx,qy,qz,qw] world_from_camera
    tx, ty, tz, qx, qy, qz, qw = pose7
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def decode_depth_bytes(depth_bytes: bytes, width: int, height: int) -> np.ndarray:
    # Support uint16 and float32
    n = width * height
    if len(depth_bytes) == n * 2:
        arr = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((height, width))
        return arr.astype(np.float32)
    if len(depth_bytes) == n * 4:
        arr = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))
        return arr.astype(np.float32)
    # Fallback: try uint16
    arr = np.frombuffer(depth_bytes, dtype=np.uint16)
    if arr.size >= n:
        return arr[:n].astype(np.float32).reshape((height, width))
    raise ValueError("Unexpected depth buffer size")


def decode_confidence_bytes(conf_bytes: bytes, width: int, height: int) -> np.ndarray:
    n = width * height
    if len(conf_bytes) < n:
        # pad with 255
        out = np.full((height, width), 255, dtype=np.uint8)
        out.flat[: len(conf_bytes)] = np.frombuffer(conf_bytes, dtype=np.uint8)
        return out
    return np.frombuffer(conf_bytes, dtype=np.uint8)[:n].reshape((height, width))


def _robust_depth(depths_m: np.ndarray) -> Optional[float]:
    if depths_m.size < 20:
        return None
    med = float(np.median(depths_m))
    mad = float(np.median(np.abs(depths_m - med))) + 1e-6
    z = np.abs(depths_m - med) / (1.4826 * mad)
    keep = depths_m[z < 3.5]
    if keep.size < 20:
        return None
    return float(np.median(keep))


def lift_det2d_to_3d(
    det2d: Dict[str, Any],
    frame_id: str,
    intr: Intrinsics,
    pose_world_from_camera: List[float],
    depth_m: Optional[np.ndarray] = None,
    conf: Optional[np.ndarray] = None,
    depth_scale: float = 1000.0,
    point_cloud_world: Optional[np.ndarray] = None,
    min_valid_fraction: float = 0.05,
    pixel_step: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Returns Det3D-like dict or None.
    """
    bbox = det2d.get("bbox_xyxy") or det2d.get("bbox")  # tolerate legacy
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1i = max(0, int(np.floor(x1)))
    y1i = max(0, int(np.floor(y1)))
    x2i = min(intr.width - 1, int(np.ceil(x2)))
    y2i = min(intr.height - 1, int(np.ceil(y2)))
    if x2i <= x1i or y2i <= y1i:
        return None

    Twc = pose7_to_matrix(pose_world_from_camera)

    points_w: List[List[float]] = []

    if depth_m is not None:
        # depth_m raw units (uint16 or float); convert to meters
        patch = depth_m[y1i:y2i:pixel_step, x1i:x2i:pixel_step]
        if patch.size == 0:
            return None
        z_raw = patch.flatten()
        if conf is not None:
            cpatch = conf[y1i:y2i:pixel_step, x1i:x2i:pixel_step].flatten()
            mask = cpatch > 0
            z_raw = z_raw[mask]
        z_raw = z_raw[z_raw > 0]
        if z_raw.size == 0:
            return None

        # If depth seems to be in millimeters, depth_scale=1000; if already meters, use 1.
        z_m = z_raw.astype(np.float32) / float(depth_scale)

        rep_z = _robust_depth(z_m)
        if rep_z is None:
            return None

        # Build sparse point cloud from valid pixels (not just representative Z)
        # Use trimming around rep_z
        keep = np.abs(z_m - rep_z) < max(0.15, 3.0 * float(np.std(z_m)) + 1e-3)
        z_m = z_m[keep]
        if z_m.size < 20:
            return None

        # Reconstruct points for those pixels: need their u,v coordinates
        us = np.arange(x1i, x2i, pixel_step)
        vs = np.arange(y1i, y2i, pixel_step)
        U, V = np.meshgrid(us, vs)
        U = U.flatten()
        V = V.flatten()

        # apply confidence and keep masks consistent
        z_raw_full = patch.flatten()
        valid_full = z_raw_full > 0
        if conf is not None:
            valid_full = valid_full & (conf[y1i:y2i:pixel_step, x1i:x2i:pixel_step].flatten() > 0)

        U = U[valid_full]
        V = V[valid_full]
        z_full_m = z_raw_full[valid_full].astype(np.float32) / float(depth_scale)
        # trim to rep_z band
        band = np.abs(z_full_m - rep_z) < 0.25
        U = U[band]
        V = V[band]
        z_full_m = z_full_m[band]
        if z_full_m.size < 20:
            return None

        X = (U - intr.cx) / intr.fx * z_full_m
        Y = (V - intr.cy) / intr.fy * z_full_m
        Z = z_full_m
        pts_c = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # N,4
        pts_w = (Twc @ pts_c.T).T[:, :3]
        points_w = pts_w.tolist()

        valid_fraction = float(z_raw.size) / float(patch.size)
        if valid_fraction < min_valid_fraction:
            return None

        score = float(det2d.get("score", 0.0)) * min(1.0, valid_fraction / 0.25)

    elif point_cloud_world is not None:
        # Select points that project into bbox
        # Transform points to camera: Tcw = inv(Twc)
        Tcw = np.linalg.inv(Twc)
        pts_w = point_cloud_world
        # homogeneous
        pts_wh = np.hstack([pts_w, np.ones((pts_w.shape[0], 1), dtype=np.float64)])
        pts_c = (Tcw @ pts_wh.T).T[:, :3]
        z = pts_c[:, 2]
        keep = z > 0.1
        pts_c = pts_c[keep]
        pts_w = pts_w[keep]
        if pts_c.shape[0] < 30:
            return None
        u = (pts_c[:, 0] * intr.fx / z[keep]) + intr.cx
        v = (pts_c[:, 1] * intr.fy / z[keep]) + intr.cy
        inside = (u >= x1i) & (u <= x2i) & (v >= y1i) & (v <= y2i)
        pts_sel = pts_w[inside]
        if pts_sel.shape[0] < 30:
            return None
        points_w = pts_sel.tolist()
        score = float(det2d.get("score", 0.0)) * min(1.0, pts_sel.shape[0] / 200.0)

    else:
        return None

    pts = np.array(points_w, dtype=np.float64)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) / 2.0
    dims = (mx - mn)

    # Guard against degenerate boxes
    if float(np.max(dims)) < 1e-3:
        return None

    return {
        "class_label": str(det2d.get("class_label", det2d.get("type", "unknown"))),
        "position_world": center.tolist(),
        "dimensions_world": dims.tolist(),
        "orientation_world": None,
        "score": float(score),
        "frame_id": frame_id,
        "points_world": points_w[:2000],  # cap for payload
    }
