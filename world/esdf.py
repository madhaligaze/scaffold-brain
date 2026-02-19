from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from world.occupancy import OCCUPIED, UNKNOWN


class ESDF:
    """
    Euclidean Signed Distance (unsigned distance here) over the voxel grid.
    Conservative: UNKNOWN treated as OCCUPIED for clearance queries.
    """

    def __init__(self) -> None:
        self._dist_m: np.ndarray | None = None
        self._origin: np.ndarray | None = None
        self._voxel_size: float | None = None
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def build_from_occupancy(self, grid: np.ndarray, origin: np.ndarray, voxel_size: float) -> None:
        occ_mask = (grid == OCCUPIED) | (grid == UNKNOWN)
        free_mask = ~occ_mask
        dist_vox = distance_transform_edt(free_mask)
        self._dist_m = dist_vox.astype(np.float32) * float(voxel_size)
        self._origin = origin.astype(np.float32)
        self._voxel_size = float(voxel_size)
        self._dirty = False

    def query_distance(self, points: list[list[float]], grid: np.ndarray, origin: np.ndarray, voxel_size: float) -> list[float]:
        if self._dirty or self._dist_m is None:
            self.build_from_occupancy(grid, origin, voxel_size)
        assert self._dist_m is not None and self._origin is not None and self._voxel_size is not None
        pts = np.asarray(points, dtype=np.float32)
        idx = ((pts - self._origin) / self._voxel_size).astype(np.int32)
        shp = np.asarray(self._dist_m.shape, dtype=np.int32)
        out: list[float] = []
        for i in idx:
            if np.any(i < 0) or np.any(i >= shp):
                out.append(0.0)
            else:
                out.append(float(self._dist_m[i[0], i[1], i[2]]))
        return out
