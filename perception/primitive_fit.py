from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FitResult:
    kind: str
    params: dict[str, Any]
    inliers: int
    rmse: float


def _rmse(dist: np.ndarray) -> float:
    if dist.size == 0:
        return 1e9
    return float(np.sqrt(np.mean(np.square(dist.astype(np.float32)))))


def _plane_from_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    # returns (n, d) where plane is n·x + d = 0, ||n|| = 1
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = (n / nn).astype(np.float32)
    d = -float(np.dot(n, a))
    return n, d


def fit_plane_ransac(
    points: np.ndarray,
    *,
    n_iter: int = 200,
    dist_thresh: float = 0.02,
    min_inliers: int = 200,
    rng_seed: int = 0,
):
    """
    points: (N,3) float32 world coords
    returns dict or None:
      {"normal":[3], "d":float, "inliers":int, "rmse":float, "inlier_ratio":float}
    """
    if points is None or points.shape[0] < max(3, min_inliers):
        return None
    pts = points.astype(np.float32, copy=False)
    N = int(pts.shape[0])
    rng = np.random.default_rng(rng_seed)

    best = None
    best_inl = 0
    best_rmse = 1e9

    for _ in range(int(n_iter)):
        idx = rng.choice(N, size=3, replace=False)
        model = _plane_from_3pts(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if model is None:
            continue
        n, d = model
        dist = np.abs(pts @ n + d)
        inliers = dist <= float(dist_thresh)
        inl = int(np.sum(inliers))
        if inl < int(min_inliers):
            continue
        rmse = float(math.sqrt(float(np.mean((dist[inliers]) ** 2))))
        if inl > best_inl or (inl == best_inl and rmse < best_rmse):
            best_inl = inl
            best_rmse = rmse
            best = (n, d, inl, rmse)

    if best is None:
        return None

    n, d, inl, rmse = best
    # Orient normal to have positive Y if possible (assume ARCore-like world: +Y up).
    if float(n[1]) < 0.0:
        n = (-n).astype(np.float32)
        d = -float(d)

    return {
        "normal": [float(n[0]), float(n[1]), float(n[2])],
        "d": float(d),
        "inliers": int(inl),
        "rmse": float(rmse),
        "inlier_ratio": float(inl) / float(N),
    }


def fit_cylinder_pca(points_w: np.ndarray) -> FitResult:
    pts = np.asarray(points_w, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 10:
        return FitResult("cylinder", {"ok": False}, 0, 1e9)
    mu = np.mean(pts, axis=0)
    X = pts - mu[None, :]
    C = (X.T @ X) / max(1, pts.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    axis = V[:, int(np.argmax(w))].astype(np.float32)
    na = float(np.linalg.norm(axis))
    if na < 1e-8:
        return FitResult("cylinder", {"ok": False}, 0, 1e9)
    axis = axis / na
    tt = X @ axis
    proj = mu[None, :] + (tt[:, None] * axis[None, :])
    radial = np.linalg.norm(pts - proj, axis=1)
    radius = float(np.median(radial))
    height = float(np.max(tt) - np.min(tt))
    origin = (mu + axis * float(np.min(tt))).astype(np.float32)
    return FitResult("cylinder", {"ok": True, "axis_origin": [float(origin[0]), float(origin[1]), float(origin[2])], "axis_dir": [float(axis[0]), float(axis[1]), float(axis[2])], "radius_m": float(max(0.0, radius)), "height_m": float(max(0.0, height))}, int(pts.shape[0]), _rmse(radial - radius))


def fit_box_pca(points_w: np.ndarray) -> FitResult:
    pts = np.asarray(points_w, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 10:
        return FitResult("box", {"ok": False}, 0, 1e9)
    mu = np.mean(pts, axis=0)
    X = pts - mu[None, :]
    C = (X.T @ X) / max(1, pts.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    R = V[:, order].astype(np.float32)
    local = X @ R
    lo = np.min(local, axis=0)
    hi = np.max(local, axis=0)
    center_l = (lo + hi) * 0.5
    extents = (hi - lo) * 0.5
    center_w = mu + (R @ center_l)
    d = np.maximum(0.0, np.abs(local - center_l[None, :]) - extents[None, :])
    return FitResult("box", {"ok": True, "center": [float(center_w[0]), float(center_w[1]), float(center_w[2])], "axes": [[float(R[0,0]), float(R[1,0]), float(R[2,0])],[float(R[0,1]), float(R[1,1]), float(R[2,1])],[float(R[0,2]), float(R[1,2]), float(R[2,2])]], "extents": [float(extents[0]), float(extents[1]), float(extents[2])]}, int(pts.shape[0]), _rmse(np.linalg.norm(d, axis=1)))
