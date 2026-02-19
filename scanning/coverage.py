from __future__ import annotations

import numpy as np


def compute_work_aabb(anchors: list[dict], *, padding_m: float = 1.0) -> tuple[list[float], list[float]] | None:
    pts = [a.get("position") for a in anchors if isinstance(a.get("position"), (list, tuple)) and len(a.get("position")) == 3]
    if not pts:
        return None
    P = np.asarray(pts, dtype=np.float32)
    lo = np.min(P, axis=0) - float(padding_m)
    hi = np.max(P, axis=0) + float(padding_m)
    return lo.tolist(), hi.tolist()


def compute_unknown_hotspots(world_model, box_min: list[float], box_max: list[float], *, max_points: int = 24) -> list[list[float]]:
    """
    Returns a few representative points (world coords) in UNKNOWN areas inside box.
    Intended for next-best-view guidance. Conservative and fast.
    """
    bmin = np.asarray(box_min, dtype=np.float32).reshape(3)
    bmax = np.asarray(box_max, dtype=np.float32).reshape(3)
    lo = np.minimum(bmin, bmax)
    hi = np.maximum(bmin, bmax)

    # Convert to voxel index range
    occ = world_model.occupancy
    i0 = ((lo - occ.origin) / occ.voxel_size).astype(np.int32)
    i1 = ((hi - occ.origin) / occ.voxel_size).astype(np.int32) + 1
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, np.asarray(occ.grid.shape, dtype=np.int32))
    if np.any(i1 <= i0):
        return []

    sub = occ.grid[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]]
    unknown_idx = np.argwhere(sub == 0)  # UNKNOWN
    if unknown_idx.size == 0:
        return []

    # Sample a handful of unknown voxels
    if unknown_idx.shape[0] > max_points:
        rng = np.random.default_rng(int(world_model.metrics.get("frames", 0) or 0) + 7)
        sel = rng.choice(unknown_idx.shape[0], size=max_points, replace=False)
        unknown_idx = unknown_idx[sel]

    # Convert to world coords (centers)
    idx = unknown_idx.astype(np.float32) + i0[None, :].astype(np.float32)
    pts_w = occ.origin[None, :] + (idx + 0.5) * float(occ.voxel_size)
    return pts_w.astype(np.float32).tolist()
