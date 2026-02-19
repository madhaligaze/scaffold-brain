from __future__ import annotations

from typing import Any

import numpy as np


def lift_mask_to_world_points(
    *,
    mask_u8: np.ndarray,
    depth_u16: np.ndarray,
    intrinsics: dict[str, Any],
    pose: dict[str, Any],
    depth_scale_m_per_unit: float,
    max_points: int = 5000,
) -> np.ndarray:
    if mask_u8.ndim != 2 or depth_u16.ndim != 2:
        raise ValueError("mask and depth must be 2D arrays")
    if mask_u8.shape != depth_u16.shape:
        raise ValueError("mask and depth shapes must match")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    if fx <= 0 or fy <= 0:
        raise ValueError("invalid intrinsics")

    pos = pose.get("position") or pose.get("pos")
    quat = pose.get("quaternion") or pose.get("quat")
    if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
        raise ValueError("pose.position missing")
    if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
        raise ValueError("pose.quaternion missing")

    t = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
    q = np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n <= 1e-8:
        raise ValueError("invalid quaternion")
    q = q / n

    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )

    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if ys.size > max_points:
        idx = np.random.choice(ys.size, size=max_points, replace=False)
        ys = ys[idx]
        xs = xs[idx]

    d = depth_u16[ys, xs].astype(np.float32) * float(depth_scale_m_per_unit)
    valid = d > 0
    ys = ys[valid]
    xs = xs[valid]
    d = d[valid]
    if d.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    X = (xs.astype(np.float32) - cx) * d / fx
    Y = (ys.astype(np.float32) - cy) * d / fy
    Z = d
    pts_c = np.stack([X, Y, Z], axis=1)
    pts_w = (pts_c @ R.T) + t[None, :]
    return pts_w.astype(np.float32)
