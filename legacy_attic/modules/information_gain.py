"""Stage 7 - Approximate information gain for active scanning.

This module estimates how valuable a candidate view (camera position) is for
reducing UNKNOWN space inside a target region (target_box).

We do NOT attempt a full sensor model. Instead we approximate:

- Sample a set of points in the target_box.
- For points that are currently UNKNOWN in VoxelWorld:
    * If line-of-sight from camera to point is not blocked by OCCUPIED, count it.
- The ratio is the 'gain' proxy (0..1).

This makes the system behave more like an engineering tool:
it prefers views that are likely to confirm missing geometry in the *relevant*
region (where scaffolding will be placed), rather than arbitrary clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import random

import numpy as np


@dataclass
class TargetBox:
    center: Tuple[float, float, float]
    half_extents: Tuple[float, float, float]


def _rand_uniform(a: float, b: float) -> float:
    return float(a + (b - a) * random.random())


def sample_points_in_box(
    box: TargetBox,
    max_samples: int = 220,
    min_step_m: float = 0.35,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return (N,3) samples inside box."""
    if seed is not None:
        random.seed(int(seed))

    cx, cy, cz = map(float, box.center)
    hx, hy, hz = map(float, box.half_extents)

    step = max(float(min_step_m), 0.05)
    nx = max(1, int((2 * hx) / step))
    ny = max(1, int((2 * hy) / step))
    nz = max(1, int((2 * hz) / step))

    grid_count = nx * ny * nz
    stride = 1
    if grid_count > max_samples:
        stride = int(math.ceil((grid_count / max_samples) ** (1 / 3)))

    pts = []
    for ix in range(0, nx + 1, stride):
        x = (cx - hx) + (2 * hx) * (ix / max(nx, 1))
        for iy in range(0, ny + 1, stride):
            y = (cy - hy) + (2 * hy) * (iy / max(ny, 1))
            for iz in range(0, nz + 1, stride):
                z = (cz - hz) + (2 * hz) * (iz / max(nz, 1))
                pts.append((float(x), float(y), float(z)))
                if len(pts) >= max_samples:
                    break
            if len(pts) >= max_samples:
                break
        if len(pts) >= max_samples:
            break

    extra = max(0, int(0.1 * max_samples) - 1)
    for _ in range(extra):
        pts.append(
            (
                _rand_uniform(cx - hx, cx + hx),
                _rand_uniform(cy - hy, cy + hy),
                _rand_uniform(cz - hz, cz + hz),
            )
        )

    if not pts:
        pts = [(cx, cy, cz)]

    return np.array(pts, dtype=float)


def estimate_information_gain(
    voxel_world: Any,
    camera_pos: Tuple[float, float, float],
    box: TargetBox,
    *,
    max_samples: int = 220,
    min_step_m: float = 0.35,
    los_clearance_m: float = 0.03,
) -> Tuple[float, Dict[str, float]]:
    """Estimate how many UNKNOWN samples in the target box are visible.

    Returns:
    - gain: float in [0,1]
    - diagnostics: counts and ratios
    """
    if voxel_world is None:
        return 0.0, {"has_world": 0.0}

    pts = sample_points_in_box(box, max_samples=max_samples, min_step_m=min_step_m)

    unknown = 0
    visible_unknown = 0

    cam = (float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]))

    for p in pts:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        try:
            st = int(voxel_world.get_state(x, y, z))
        except Exception:
            st = getattr(voxel_world, "UNKNOWN", -1)

        if st != getattr(voxel_world, "UNKNOWN", -1):
            continue

        unknown += 1

        try:
            blocked = bool(
                voxel_world.is_blocked(
                    cam,
                    (x, y, z),
                    clearance=float(los_clearance_m),
                    unknown_is_blocked=False,
                )
            )
        except Exception:
            blocked = False

        if not blocked:
            visible_unknown += 1

    denom = max(1, unknown)
    gain = float(visible_unknown) / float(denom)

    return gain, {
        "has_world": 1.0,
        "unknown_samples": float(unknown),
        "visible_unknown_samples": float(visible_unknown),
        "gain": float(gain),
    }
