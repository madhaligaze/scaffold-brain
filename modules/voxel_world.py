"""
VoxelWorld — Воксельная карта рабочего пространства.
=====================================================
ИИ перестаёт быть слепым. Каждый кубик 10x10x10 см —
либо свободен (0), либо занят твёрдым объектом (1).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

VoxelCoord = Tuple[int, int, int]


class VoxelWorld:
    """Трёхмерная сетка занятости пространства."""

    FREE = 0
    OCCUPIED = 1
    PIPE = 2
    WALL = 3
    FLOOR = 4
    BEAM_SLOT = 5

    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self._grid: Dict[VoxelCoord, int] = {}

    def mark_point(self, x: float, y: float, z: float, vtype: int = OCCUPIED) -> None:
        self._grid[self._to_voxel(x, y, z)] = vtype

    def mark_box(self, center: Dict, dims: Dict, vtype: int = OCCUPIED) -> None:
        cx, cy, cz = center["x"], center["y"], center["z"]
        hw = dims["width"] / 2
        hd = dims["depth"] / 2
        hh = dims["height"] / 2

        r = self.resolution
        xs = np.arange(cx - hw, cx + hw + r, r)
        ys = np.arange(cy - hd, cy + hd + r, r)
        zs = np.arange(cz - hh, cz + hh + r, r)

        for x in xs:
            for y in ys:
                for z in zs:
                    self._grid[self._to_voxel(x, y, z)] = vtype

    def ingest_depth_map(
        self,
        depth_bytes: bytes,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx_px: float,
        cy_px: float,
        camera_pose: List[float],
        max_range: float = 8.0,
    ) -> int:
        if len(depth_bytes) < width * height * 2:
            return 0

        depth_mm = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(height, width)
        depth_m = depth_mm.astype(np.float32) / 1000.0

        rotation = self._quat_to_rotation(*camera_pose[3:7])
        translation = np.array(camera_pose[:3])

        count = 0
        step = 4

        for v in range(0, height, step):
            for u in range(0, width, step):
                d = depth_m[v, u]
                if d <= 0 or d > max_range:
                    continue

                xc = (u - cx_px) * d / fx
                yc = (v - cy_px) * d / fy
                zc = d

                point_world = rotation @ np.array([xc, yc, zc]) + translation
                self.mark_point(*point_world, vtype=self.OCCUPIED)
                count += 1

        return count

    def ingest_yolo_detections(self, detections: List[Dict]) -> None:
        type_map = {
            "pipe_obstacle": self.PIPE,
            "wall": self.WALL,
            "floor_slab": self.FLOOR,
            "column": self.OCCUPIED,
            "cable_tray": self.PIPE,
        }
        for det in detections:
            vtype = type_map.get(det.get("type", ""), self.OCCUPIED)
            pos = det.get("position", {})
            dims = det.get("dimensions", {"width": 0.2, "depth": 0.2, "height": 0.5})
            if pos:
                self.mark_box(pos, dims, vtype=vtype)

    def is_blocked(self, start: Dict, end: Dict, clearance: float = 0.05) -> bool:
        if not self._grid:
            return False

        sx, sy, sz = start["x"], start["y"], start["z"]
        ex, ey, ez = end["x"], end["y"], end["z"]

        length = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2)
        if length < self.resolution:
            return False

        steps = max(int(length / self.resolution), 2)
        extra = max(1, int(clearance / self.resolution))

        for i in range(steps + 1):
            t = i / steps
            wx = sx + t * (ex - sx)
            wy = sy + t * (ey - sy)
            wz = sz + t * (ez - sz)

            vx, vy, vz = self._to_voxel(wx, wy, wz)
            for dx in range(-extra, extra + 1):
                for dz in range(-extra, extra + 1):
                    vtype = self._grid.get((vx + dx, vy, vz + dz), self.FREE)
                    if vtype not in (self.FREE, self.FLOOR, self.BEAM_SLOT):
                        return True
        return False

    def get_type(self, x: float, y: float, z: float) -> int:
        return self._grid.get(self._to_voxel(x, y, z), self.FREE)

    def get_floor_z(self, x: float, y: float, search_below: float = 5.0) -> Optional[float]:
        vx, vy, vz = self._to_voxel(x, y, 0)
        end_vi = vz - int(search_below / self.resolution)

        for vi in range(vz, end_vi, -1):
            if self._grid.get((vx, vy, vi), self.FREE) == self.FLOOR:
                return vi * self.resolution
        return None

    def to_ar_mesh(self) -> Dict:
        result = []
        for (vx, vy, vz), vtype in self._grid.items():
            if vtype != self.FREE:
                result.append(
                    {
                        "x": vx * self.resolution,
                        "y": vy * self.resolution,
                        "z": vz * self.resolution,
                        "type": vtype,
                    }
                )
        return {"voxels": result, "resolution": self.resolution}

    def _to_voxel(self, x: float, y: float, z: float) -> VoxelCoord:
        r = self.resolution
        return (
            int(math.floor(x / r)),
            int(math.floor(y / r)),
            int(math.floor(z / r)),
        )

    @staticmethod
    def _quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        return np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]
        )
