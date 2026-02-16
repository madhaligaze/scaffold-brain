"""
VoxelWorld — Воксельная карта рабочего пространства.
=====================================================
ИСПРАВЛЕНИЕ v3.1 (Аудит):
  - ГЛАВНЫЙ источник данных: point_cloud от ARCore (список [x,y,z])
  - YOLO-детекции: вспомогательный источник (только bbox, без глубины)
  - is_blocked: raymarching с шагом resolution/2 для точности
  - add_point_cloud: фильтрация шума (z < -1 или z > 30 отбрасываем)

Архитектура заполнения:
  Android ARCore Point Cloud → POST /session/stream → add_point_cloud()
  Android ARCore Depth Map   → POST /session/depth_stream → ingest_depth_map()
  YOLO bounding boxes        → ingest_yolo_boxes() (ПРИБЛИЗИТЕЛЬНО, без Z!)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


class VoxelWorld:
    """Трёхмерная сетка занятости пространства."""

    # Типы вокселей
    FREE = 0
    OCCUPIED = 1
    PIPE = 2
    WALL = 3
    FLOOR = 4

    def __init__(self, resolution: float = 0.1):
        """
        Args:
            resolution: размер одного вокселя, метры (0.1 = 10 см)
        """
        self.resolution = resolution
        self.occupied: Set[Tuple[int, int, int]] = set()
        # Тип каждого занятого вокселя (опционально)
        self._types: Dict[Tuple[int, int, int], int] = {}
        # Backward-compatible map view used by legacy code.
        self._grid: Dict[Tuple[int, int, int], int] = {}

    # ── Заполнение ────────────────────────────────────────────────────────────

    def add_point_cloud(self, points: List[List[float]], vtype: int = OCCUPIED) -> int:
        """
        ГЛАВНЫЙ метод заполнения. Принимает облако точек от ARCore.

        ARCore Point Cloud уже в мировых координатах (метры).
        Не требует матриц камеры — просто список [x, y, z].

        Args:
            points: [[x, y, z], ...] — мировые координаты
            vtype:  тип занятости (OCCUPIED по умолчанию)

        Returns:
            Количество добавленных вокселей
        """
        count = 0
        for p in points:
            if len(p) < 3:
                continue
            x, y, z = p[0], p[1], p[2]

            # Фильтр шума: слишком низко под полом или слишком высоко
            if z < -1.0 or z > 30.0:
                continue

            coord = self._to_grid(x, y, z)
            if coord not in self.occupied:
                self.occupied.add(coord)
                self._types[coord] = vtype
                self._grid[coord] = vtype
                count += 1

        return count

    def mark_box(self, center: Dict, dims: Dict, vtype: int = OCCUPIED) -> int:
        """
        Заполнить прямоугольную зону (из YOLO bbox + оценка глубины).

        ВНИМАНИЕ: Без точной глубины Z центр берётся из camera_z.
        Используй только как приблизительную разметку!

        Args:
            center: {"x": f, "y": f, "z": f}
            dims:   {"width": f, "depth": f, "height": f}
        """
        cx, cy, cz = center["x"], center["y"], center["z"]
        hw = dims.get("width", 0.2) / 2
        hd = dims.get("depth", 0.2) / 2
        hh = dims.get("height", 0.5) / 2
        r = self.resolution

        count = 0
        for x in np.arange(cx - hw, cx + hw + r, r):
            for y in np.arange(cy - hd, cy + hd + r, r):
                for z in np.arange(cz - hh, cz + hh + r, r):
                    coord = self._to_grid(x, y, z)
                    if coord not in self.occupied:
                        self.occupied.add(coord)
                        self._types[coord] = vtype
                        self._grid[coord] = vtype
                        count += 1
        return count

    def ingest_yolo_detections(self, detections: List[Dict], fallback_depth: float = 2.0) -> None:
        """
        ВСПОМОГАТЕЛЬНЫЙ метод: заполнение из YOLO без реальной глубины.

        YOLO даёт только 2D bbox на картинке, не даёт Z!
        Используем fallback_depth как оценку расстояния от камеры.
        Это ПРИБЛИЗИТЕЛЬНО — только если нет point_cloud.

        Args:
            detections:     список объектов из vision.py
            fallback_depth: предполагаемое расстояние до объекта (метры)
        """
        type_map = {
            "pipe_obstacle": self.PIPE,
            "wall": self.WALL,
            "floor_slab": self.FLOOR,
            "cable_tray": self.PIPE,
            "column": self.OCCUPIED,
        }
        for det in detections:
            pos = det.get("position", {})
            dims = det.get("dimensions", {"width": 0.3, "depth": 0.3, "height": 0.5})
            if not pos:
                continue
            # Если Z не задан (YOLO не знает глубину) — используем fallback
            if pos.get("z", 0) == 0:
                pos = {**pos, "z": fallback_depth}
            vtype = type_map.get(det.get("type", ""), self.OCCUPIED)
            self.mark_box(pos, dims, vtype=vtype)

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
        """
        Конвертирует ARCore Depth Map (uint16, мм) в воксели.
        БОНУС: работает только на устройствах с Depth API.

        Args:
            depth_bytes:  raw bytes uint16 little-endian
            camera_pose:  [tx, ty, tz, qx, qy, qz, qw]
        """
        if len(depth_bytes) < width * height * 2:
            return 0

        depth_mm = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(height, width)
        depth_m = depth_mm.astype(np.float32) / 1000.0

        R = self._quat_to_rotation(*camera_pose[3:7])
        t = np.array(camera_pose[:3])

        count = 0
        step = 4  # каждый 4-й пиксель

        for v in range(0, height, step):
            for u in range(0, width, step):
                d = depth_m[v, u]
                if d <= 0 or d > max_range:
                    continue
                xc = (u - cx_px) * d / fx
                yc = (v - cy_px) * d / fy
                pw = R @ np.array([xc, yc, d]) + t
                coord = self._to_grid(*pw)
                if coord not in self.occupied:
                    self.occupied.add(coord)
                    self._types[coord] = self.OCCUPIED
                    self._grid[coord] = self.OCCUPIED
                    count += 1
        return count

    # ── Запросы ───────────────────────────────────────────────────────────────

    def is_blocked(
        self,
        start: Union[Tuple[float, float, float], Dict[str, float]],
        end: Union[Tuple[float, float, float], Dict[str, float]],
    ) -> bool:
        """
        Raymarching: проверяет, пересекает ли отрезок start→end занятые воксели.

        Принимает кортежи (x, y, z) — совместимо с форматом AStarPathfinder.

        Args:
            start, end: (x, y, z) мировые координаты

        Returns:
            True если путь заблокирован хотя бы одним занятым вокселем
        """
        if not self.occupied:
            return False

        p1 = np.array(self._as_tuple(start), dtype=float)
        p2 = np.array(self._as_tuple(end), dtype=float)
        dist = np.linalg.norm(p2 - p1)

        if dist < self.resolution:
            return False

        # Шаг = resolution/2 для надёжного обнаружения
        steps = max(int(dist / (self.resolution / 2)), 2)

        for i in range(steps + 1):
            t = i / steps
            p = p1 + (p2 - p1) * t
            if self._to_grid(*p) in self.occupied:
                return True

        return False

    def get_type(self, x: float, y: float, z: float) -> int:
        coord = self._to_grid(x, y, z)
        return self._types.get(coord, self.FREE)

    def get_floor_z(self, x: float, y: float, search_below: float = 5.0) -> Optional[float]:
        vx, vy, vz = self._to_grid(x, y, 0)
        end_vi = vz - int(search_below / self.resolution)

        for vi in range(vz, end_vi, -1):
            if self._types.get((vx, vy, vi), self.FREE) == self.FLOOR:
                return vi * self.resolution
        return None

    def clear(self) -> None:
        """Очищает мир (при старте новой сессии)."""
        self.occupied.clear()
        self._types.clear()
        self._grid.clear()

    @property
    def total_voxels(self) -> int:
        return len(self.occupied)

    def to_ar_mesh(self) -> Dict:
        """Воксели для AR-оверлея."""
        return {
            "voxels": [
                {
                    "x": vx * self.resolution,
                    "y": vy * self.resolution,
                    "z": vz * self.resolution,
                    "type": self._types.get((vx, vy, vz), self.OCCUPIED),
                }
                for (vx, vy, vz) in self.occupied
            ],
            "resolution": self.resolution,
        }

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def _to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        r = self.resolution
        return (int(math.floor(x / r)), int(math.floor(y / r)), int(math.floor(z / r)))

    @staticmethod
    def _as_tuple(p: Union[Tuple[float, float, float], Dict[str, float]]) -> Tuple[float, float, float]:
        if isinstance(p, dict):
            return (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
        return (float(p[0]), float(p[1]), float(p[2]))

    @staticmethod
    def _quat_to_rotation(qx, qy, qz, qw) -> np.ndarray:
        return np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]
        )
