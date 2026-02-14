"""Модуль пространственной геометрии: ROI-фильтр, коллизии и стандарты Layher."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh


class WorldGeometry:
    """Управление 3D-сценой: препятствия, проверки пересечений и стандартизация длин."""

    LAYHER_STANDARDS = [0.73, 1.09, 1.57, 2.07, 2.57, 3.07]

    def __init__(self, padding: float = 3.0):
        self.scene = trimesh.Scene()
        self.padding = padding
        self.obstacles: List[trimesh.Trimesh] = []

    def create_roi_filter(self, user_points: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Строит ROI bbox вокруг пользовательских точек + паддинг (по умолчанию 3м)."""
        if not user_points:
            return np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 5.0])

        pts = np.array([[p["x"], p["y"], p["z"]] for p in user_points], dtype=float)
        min_bound = pts.min(axis=0) - self.padding
        max_bound = pts.max(axis=0) + self.padding
        min_bound[2] = max(0.0, min_bound[2])
        return min_bound, max_bound

    def crop_scene(self, points: List[Dict[str, float]], padding: float | None = None) -> List[Dict[str, float]]:
        """Оставляет только точки внутри ROI. Полезно для фильтрации входного облака."""
        if padding is not None:
            original = self.padding
            self.padding = padding
            min_bound, max_bound = self.create_roi_filter(points)
            self.padding = original
        else:
            min_bound, max_bound = self.create_roi_filter(points)

        cropped: List[Dict[str, float]] = []
        for point in points:
            coords = np.array([point["x"], point["y"], point["z"]], dtype=float)
            if np.all(coords >= min_bound) and np.all(coords <= max_bound):
                cropped.append(point)
        return cropped

    def add_obstacle(self, obs_type: str, start: List[float], end: List[float], radius: float):
        """Добавляет препятствие (труба/цилиндр) в сцену между двумя точками."""
        if obs_type.lower() != "pipe":
            raise ValueError("WorldGeometry.add_obstacle currently supports only obs_type='pipe'")

        start_np = np.array(start, dtype=float)
        end_np = np.array(end, dtype=float)
        direction = end_np - start_np
        height = float(np.linalg.norm(direction))
        if height == 0:
            raise ValueError("Obstacle start and end points must be different")

        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=24)

        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        target = direction / height
        transform = trimesh.geometry.align_vectors(z_axis, target)
        if transform is None:
            transform = np.eye(4)
        cylinder.apply_transform(transform)

        midpoint = (start_np + end_np) / 2.0
        cylinder.apply_translation(midpoint)

        self.obstacles.append(cylinder)
        self.scene.add_geometry(cylinder)

    def _extract_beam_points(self, beam: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
        beam_id = str(beam.get("id", "unknown"))

        if "start_node" in beam and "end_node" in beam:
            p1 = np.array(
                [beam["start_node"]["x"], beam["start_node"]["y"], beam["start_node"]["z"]], dtype=float
            )
            p2 = np.array(
                [beam["end_node"]["x"], beam["end_node"]["y"], beam["end_node"]["z"]], dtype=float
            )
            return p1, p2, beam_id

        if "start" in beam and "end" in beam and isinstance(beam["start"], dict):
            p1 = np.array([beam["start"]["x"], beam["start"]["y"], beam["start"]["z"]], dtype=float)
            p2 = np.array([beam["end"]["x"], beam["end"]["y"], beam["end"]["z"]], dtype=float)
            return p1, p2, beam_id

        raise ValueError("Beam must contain start/end points in start_node/end_node or start/end format")

    def check_collisions(self, scaffold_beams: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Проверяет пересечения балок с загруженными препятствиями."""
        if not self.obstacles:
            return []

        collisions: List[Dict[str, str]] = []
        use_manager = True
        try:
            manager = trimesh.collision.CollisionManager()
            for i, obs in enumerate(self.obstacles):
                manager.add_object(f"obstacle_{i}", obs)
        except BaseException:
            # Fallback для окружений без python-fcl: используем пересечение AABB
            use_manager = False
            manager = None

        for beam in scaffold_beams:
            p1, p2, beam_id = self._extract_beam_points(beam)
            length = float(np.linalg.norm(p2 - p1))
            if length == 0:
                continue

            beam_mesh = trimesh.creation.cylinder(radius=0.024, height=length, sections=18)
            transform = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), (p2 - p1) / length)
            if transform is None:
                transform = np.eye(4)
            beam_mesh.apply_transform(transform)
            beam_mesh.apply_translation((p1 + p2) / 2.0)

            is_collision = False
            if use_manager and manager is not None:
                is_collision = manager.in_collision_single(beam_mesh)
            else:
                beam_bounds = beam_mesh.bounds
                for obstacle in self.obstacles:
                    obs_bounds = obstacle.bounds
                    overlap_min = np.maximum(beam_bounds[0], obs_bounds[0])
                    overlap_max = np.minimum(beam_bounds[1], obs_bounds[1])
                    if np.all(overlap_min <= overlap_max):
                        is_collision = True
                        break

            if is_collision:
                collisions.append(
                    {
                        "beam_id": beam_id,
                        "type": "OBSTACLE_INTERSECTION",
                        "message": f"Балка {beam_id} проходит сквозь препятствие!",
                    }
                )

        return collisions

    @staticmethod
    def get_distance(start: Dict[str, float], end: Dict[str, float]) -> float:
        """Расстояние между двумя точками {x,y,z}."""
        p1 = np.array([start["x"], start["y"], start["z"]], dtype=float)
        p2 = np.array([end["x"], end["y"], end["z"]], dtype=float)
        return float(np.linalg.norm(p2 - p1))

    @staticmethod
    def get_midpoint(start: Dict[str, float], end: Dict[str, float]) -> Dict[str, float]:
        """Середина между двумя точками {x,y,z}."""
        return {
            "x": float((start["x"] + end["x"]) / 2.0),
            "y": float((start["y"] + end["y"]) / 2.0),
            "z": float((start["z"] + end["z"]) / 2.0),
        }

    def align_to_layher(self, length: float) -> float:
        """Округляет произвольную длину к ближайшему стандарту Layher."""
        return min(self.LAYHER_STANDARDS, key=lambda x: abs(x - length))


# --- Backward-compatible adapters ---
class GeometryUtils:
    """Совместимость со старым API геометрических утилит."""

    @staticmethod
    def distance_3d(p1: Dict[str, float], p2: Dict[str, float]) -> float:
        return WorldGeometry.get_distance(p1, p2)

    @staticmethod
    def calculate_bounding_box(nodes: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if not nodes:
            return ({"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 0.0})
        xs = [float(n.get("x", 0.0)) for n in nodes]
        ys = [float(n.get("y", 0.0)) for n in nodes]
        zs = [float(n.get("z", 0.0)) for n in nodes]
        return (
            {"x": min(xs), "y": min(ys), "z": min(zs)},
            {"x": max(xs), "y": max(ys), "z": max(zs)},
        )


class CollisionDetector:
    """Совместимость со старым API детекции препятствий."""

    def __init__(self):
        self._world = WorldGeometry()

    def add_obstacle(self, obstacle: Dict[str, Any]):
        obs_type = obstacle.get("type", "").lower()
        if obs_type == "pipe":
            start = obstacle.get("start", {"x": 0.0, "y": 0.0, "z": 0.0})
            end = obstacle.get("end", {"x": 0.0, "y": 0.0, "z": 1.0})
            radius = float(obstacle.get("radius", 0.1))
            self._world.add_obstacle(
                "pipe",
                [float(start.get("x", 0.0)), float(start.get("y", 0.0)), float(start.get("z", 0.0))],
                [float(end.get("x", 0.0)), float(end.get("y", 0.0)), float(end.get("z", 0.0))],
                radius,
            )

    def check_beam_collision(self, beam_start: Dict[str, float], beam_end: Dict[str, float], clearance: float = 0.2) -> List[Dict[str, str]]:
        _ = clearance
        collisions = self._world.check_collisions(
            [{"id": "legacy_beam", "start": beam_start, "end": beam_end}]
        )
        return collisions
