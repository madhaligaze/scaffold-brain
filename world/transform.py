from __future__ import annotations

import numpy as np


def quat_to_rot(q: list[float] | np.ndarray) -> np.ndarray:
    """
    q = [x,y,z,w]
    returns 3x3 rotation matrix
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    x, y, z, w = q.tolist()
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose: dict) -> np.ndarray:
    """
    Contract: pose is camera->world.
      pose["position"] = [tx,ty,tz]
      pose["quaternion"] = [x,y,z,w]
    """
    t = np.asarray(pose["position"], dtype=np.float64).reshape(3)
    q = pose["quaternion"]
    R = quat_to_rot(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_to_quat_xyzw(R: np.ndarray) -> list[float]:
    """Convert 3x3 rotation matrix to quaternion [x,y,z,w] (xyzw)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = (tr + 1.0) ** 0.5 * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.asarray([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    q = (q / n).tolist()
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def matrix_to_pose(T: np.ndarray) -> dict:
    """Convert 4x4 transform (camera->world) to pose dict with position+quaternion (xyzw)."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    return {"position": [float(t[0]), float(t[1]), float(t[2])], "quaternion": rot_to_quat_xyzw(R)}
