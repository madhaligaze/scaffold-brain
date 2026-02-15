# modules/geometry.py
"""
Геометрические утилиты для работы с 3D-пространством лесов.
WorldGeometry — фасад, который использует main.py.
"""
import numpy as np
import math
from typing import List, Dict, Tuple, Optional


class GeometryUtils:
    """Набор геометрических функций для работы с 3D-моделями"""

    @staticmethod
    def distance_3d(p1: Dict, p2: Dict) -> float:
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        dz = p1['z'] - p2['z']
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    @staticmethod
    def point_to_line_distance(point: Dict, line_start: Dict, line_end: Dict) -> float:
        line_vec = np.array([
            line_end['x'] - line_start['x'],
            line_end['y'] - line_start['y'],
            line_end['z'] - line_start['z']
        ])
        point_vec = np.array([
            point['x'] - line_start['x'],
            point['y'] - line_start['y'],
            point['z'] - line_start['z']
        ])
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return float(np.linalg.norm(point_vec))
        line_unit = line_vec / line_length
        projection = np.dot(point_vec, line_unit)
        projection = max(0, min(line_length, projection))
        closest = (
            line_start['x'] + line_unit[0] * projection,
            line_start['y'] + line_unit[1] * projection,
            line_start['z'] + line_unit[2] * projection,
        )
        dist_vec = np.array([
            point['x'] - closest[0],
            point['y'] - closest[1],
            point['z'] - closest[2]
        ])
        return float(np.linalg.norm(dist_vec))

    @staticmethod
    def angle_between_beams(b1s: Dict, b1e: Dict, b2s: Dict, b2e: Dict) -> float:
        v1 = np.array([b1e['x'] - b1s['x'], b1e['y'] - b1s['y'], b1e['z'] - b1s['z']])
        v2 = np.array([b2e['x'] - b2s['x'], b2e['y'] - b2s['y'], b2e['z'] - b2s['z']])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_a = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    @staticmethod
    def is_point_inside_box(point: Dict, box_min: Dict, box_max: Dict) -> bool:
        return (box_min['x'] <= point['x'] <= box_max['x'] and
                box_min['y'] <= point['y'] <= box_max['y'] and
                box_min['z'] <= point['z'] <= box_max['z'])

    @staticmethod
    def calculate_bounding_box(nodes: List[Dict]) -> Tuple[Dict, Dict]:
        if not nodes:
            return {'x': 0, 'y': 0, 'z': 0}, {'x': 0, 'y': 0, 'z': 0}
        xs = [n['x'] for n in nodes]
        ys = [n['y'] for n in nodes]
        zs = [n['z'] for n in nodes]
        return {'x': min(xs), 'y': min(ys), 'z': min(zs)}, {'x': max(xs), 'y': max(ys), 'z': max(zs)}

    @staticmethod
    def calculate_center_of_mass(nodes: List[Dict]) -> Dict:
        if not nodes:
            return {'x': 0, 'y': 0, 'z': 0}
        return {
            'x': sum(n['x'] for n in nodes) / len(nodes),
            'y': sum(n['y'] for n in nodes) / len(nodes),
            'z': sum(n['z'] for n in nodes) / len(nodes),
        }

    @staticmethod
    def check_collision(b1s: Dict, b1e: Dict, b2s: Dict, b2e: Dict, tolerance: float = 0.1) -> bool:
        distances = [
            GeometryUtils.point_to_line_distance(b1s, b2s, b2e),
            GeometryUtils.point_to_line_distance(b1e, b2s, b2e),
            GeometryUtils.point_to_line_distance(b2s, b1s, b1e),
            GeometryUtils.point_to_line_distance(b2e, b1s, b1e),
        ]
        return min(distances) < tolerance

    @staticmethod
    def interpolate_points_along_beam(start: Dict, end: Dict, num_points: int = 10) -> List[Dict]:
        return [
            {
                'x': start['x'] + (i / num_points) * (end['x'] - start['x']),
                'y': start['y'] + (i / num_points) * (end['y'] - start['y']),
                'z': start['z'] + (i / num_points) * (end['z'] - start['z']),
            }
            for i in range(num_points + 1)
        ]


class CollisionDetector:
    """Детектор коллизий между элементами конструкции и препятствиями"""

    def __init__(self):
        self.obstacles: List[Dict] = []

    def add_obstacle(self, obstacle: Dict):
        self.obstacles.append(obstacle)

    def check_beam_collision(self, beam_start: Dict, beam_end: Dict, clearance: float = 0.2) -> List[Dict]:
        return [
            obs for obs in self.obstacles
            if self._beam_intersects_obstacle(beam_start, beam_end, obs, clearance)
        ]

    def _beam_intersects_obstacle(self, bs: Dict, be: Dict, obstacle: Dict, clearance: float) -> bool:
        obs_type = obstacle.get('type', 'box')
        obs_pos = obstacle['position']
        if obs_type == 'box':
            dims = obstacle['dimensions']
            box_min = {k: obs_pos[k] - clearance for k in 'xyz'}
            box_max = {
                'x': obs_pos['x'] + dims['width'] + clearance,
                'y': obs_pos['y'] + dims['depth'] + clearance,
                'z': obs_pos['z'] + dims['height'] + clearance,
            }
            mid = {k: (bs[k] + be[k]) / 2 for k in 'xyz'}
            return (GeometryUtils.is_point_inside_box(bs, box_min, box_max) or
                    GeometryUtils.is_point_inside_box(be, box_min, box_max) or
                    GeometryUtils.is_point_inside_box(mid, box_min, box_max))
        elif obs_type == 'sphere':
            radius = obstacle['dimensions']['radius'] + clearance
            return GeometryUtils.point_to_line_distance(obs_pos, bs, be) < radius
        return False


# ════════════════════════════════════════════════════════════════
#  WorldGeometry — ФАСАД для main.py
#  ИСПРАВЛЕНИЕ: этот класс отсутствовал → main.py падал с ImportError
# ════════════════════════════════════════════════════════════════

class WorldGeometry:
    """
    Фасад геометрических операций для main.py.
    Объединяет GeometryUtils и CollisionDetector в единый интерфейс.

    Используется в /session/model для:
    - check_collisions(beams) → список пар балок с пересечением
    - register_obstacles(objects) → добавление детектированных препятствий
    - get_scene_bounds(nodes) → bounding box всей сцены
    """

    def __init__(self):
        self.detector = CollisionDetector()
        self._obstacles: List[Dict] = []

    def register_obstacle(self, obstacle: Dict) -> None:
        """
        Регистрирует препятствие из Vision-детекции.

        Args:
            obstacle: {
                "type": "box"|"sphere",
                "position": {"x": f, "y": f, "z": f},
                "dimensions": {"width": f, "depth": f, "height": f}  # для box
                             {"radius": f}                             # для sphere
            }
        """
        self._obstacles.append(obstacle)
        self.detector.add_obstacle(obstacle)

    def check_collisions(
        self,
        beams: List[Dict],
        nodes: Optional[List[Dict]] = None,
        clearance: float = 0.15,
    ) -> List[Dict]:
        """
        Проверяет попарные коллизии балок и коллизии с зарегистрированными препятствиями.

        Args:
            beams: список балок {"id": str, "start": str, "end": str}
            nodes: список узлов {"id": str, "x": f, "y": f, "z": f}
                   Если None — проверяются только препятствия, не попарные пересечения
            clearance: зазор безопасности в метрах

        Returns:
            Список коллизий:
            [{"beam_id": str, "conflict_id": str, "type": "beam_beam"|"beam_obstacle"}, ...]
        """
        collisions: List[Dict] = []

        if not beams:
            return collisions

        # Строим индекс узлов для быстрого доступа
        node_map: Dict[str, Dict] = {}
        if nodes:
            node_map = {n['id']: n for n in nodes}

        # 1. Проверяем коллизии балок с препятствиями
        if self._obstacles and node_map:
            for beam in beams:
                bs = node_map.get(beam.get('start', ''))
                be = node_map.get(beam.get('end', ''))
                if bs and be:
                    hits = self.detector.check_beam_collision(bs, be, clearance)
                    for hit in hits:
                        collisions.append({
                            "beam_id": beam['id'],
                            "conflict_id": hit.get('id', 'obstacle'),
                            "type": "beam_obstacle",
                        })

        # 2. Проверяем попарные пересечения балок
        if node_map and len(beams) > 1:
            for i in range(len(beams)):
                for j in range(i + 1, len(beams)):
                    b1, b2 = beams[i], beams[j]
                    b1s = node_map.get(b1.get('start', ''))
                    b1e = node_map.get(b1.get('end', ''))
                    b2s = node_map.get(b2.get('start', ''))
                    b2e = node_map.get(b2.get('end', ''))
                    if b1s and b1e and b2s and b2e:
                        # Пропускаем балки, у которых общий узел (они должны пересекаться в нём)
                        shared = {b1.get('start'), b1.get('end')} & {b2.get('start'), b2.get('end')}
                        if not shared and GeometryUtils.check_collision(b1s, b1e, b2s, b2e, clearance):
                            collisions.append({
                                "beam_id": b1['id'],
                                "conflict_id": b2['id'],
                                "type": "beam_beam",
                            })

        return collisions

    def get_scene_bounds(self, nodes: List[Dict]) -> Dict:
        """
        Возвращает bounding box сцены.

        Returns:
            {"min": {x,y,z}, "max": {x,y,z}, "size": {w,h,d}, "center": {x,y,z}}
        """
        if not nodes:
            return {"min": {}, "max": {}, "size": {"w": 0, "h": 0, "d": 0}, "center": {}}
        mn, mx = GeometryUtils.calculate_bounding_box(nodes)
        center = GeometryUtils.calculate_center_of_mass(nodes)
        return {
            "min": mn,
            "max": mx,
            "size": {
                "w": round(mx['x'] - mn['x'], 3),
                "h": round(mx['z'] - mn['z'], 3),
                "d": round(mx['y'] - mn['y'], 3),
            },
            "center": center,
        }