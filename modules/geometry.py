# modules/geometry.py
"""
Геометрические утилиты для работы с 3D-пространством лесов.
"""
import numpy as np
import math
from typing import List, Dict, Tuple

class GeometryUtils:
    """Набор геометрических функций для работы с 3D-моделями"""
    
    @staticmethod
    def distance_3d(p1: Dict, p2: Dict) -> float:
        """
        Евклидово расстояние между двумя точками в 3D.
        
        Args:
            p1, p2: точки {x, y, z}
        
        Returns:
            Расстояние в метрах
        """
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        dz = p1['z'] - p2['z']
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    @staticmethod
    def point_to_line_distance(point: Dict, line_start: Dict, line_end: Dict) -> float:
        """
        Расстояние от точки до линии (балки).
        Полезно для проверки зазоров при обходе препятствий.
        """
        # Вектор линии
        line_vec = np.array([
            line_end['x'] - line_start['x'],
            line_end['y'] - line_start['y'],
            line_end['z'] - line_start['z']
        ])
        
        # Вектор от начала линии до точки
        point_vec = np.array([
            point['x'] - line_start['x'],
            point['y'] - line_start['y'],
            point['z'] - line_start['z']
        ])
        
        # Длина линии
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            # Линия вырождена в точку
            return np.linalg.norm(point_vec)
        
        # Проекция point_vec на line_vec
        line_unit = line_vec / line_length
        projection = np.dot(point_vec, line_unit)
        
        # Ограничиваем проекцию отрезком [0, line_length]
        projection = max(0, min(line_length, projection))
        
        # Ближайшая точка на линии
        closest = line_start['x'] + line_unit[0] * projection, \
                  line_start['y'] + line_unit[1] * projection, \
                  line_start['z'] + line_unit[2] * projection
        
        # Расстояние от point до closest
        dist_vec = np.array([
            point['x'] - closest[0],
            point['y'] - closest[1],
            point['z'] - closest[2]
        ])
        
        return np.linalg.norm(dist_vec)
    
    @staticmethod
    def angle_between_beams(beam1_start: Dict, beam1_end: Dict,
                           beam2_start: Dict, beam2_end: Dict) -> float:
        """
        Угол между двумя балками в градусах.
        
        Returns:
            Угол от 0 до 180 градусов
        """
        # Вектора балок
        v1 = np.array([
            beam1_end['x'] - beam1_start['x'],
            beam1_end['y'] - beam1_start['y'],
            beam1_end['z'] - beam1_start['z']
        ])
        
        v2 = np.array([
            beam2_end['x'] - beam2_start['x'],
            beam2_end['y'] - beam2_start['y'],
            beam2_end['z'] - beam2_start['z']
        ])
        
        # Нормализация
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0
        
        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm
        
        # Косинус угла
        cos_angle = np.dot(v1_unit, v2_unit)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Защита от погрешностей
        
        # Угол в радианах, затем в градусах
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def is_point_inside_box(point: Dict, box_min: Dict, box_max: Dict) -> bool:
        """
        Проверяет, находится ли точка внутри прямоугольного параллелепипеда.
        
        Args:
            point: {x, y, z}
            box_min: {x, y, z} минимальные координаты
            box_max: {x, y, z} максимальные координаты
        """
        return (box_min['x'] <= point['x'] <= box_max['x'] and
                box_min['y'] <= point['y'] <= box_max['y'] and
                box_min['z'] <= point['z'] <= box_max['z'])
    
    @staticmethod
    def calculate_bounding_box(nodes: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Вычисляет ограничивающий прямоугольник (bounding box) для набора узлов.
        
        Returns:
            (min_point, max_point)
        """
        if not nodes:
            return ({'x': 0, 'y': 0, 'z': 0}, {'x': 0, 'y': 0, 'z': 0})
        
        xs = [n['x'] for n in nodes]
        ys = [n['y'] for n in nodes]
        zs = [n['z'] for n in nodes]
        
        min_point = {'x': min(xs), 'y': min(ys), 'z': min(zs)}
        max_point = {'x': max(xs), 'y': max(ys), 'z': max(zs)}
        
        return min_point, max_point
    
    @staticmethod
    def calculate_center_of_mass(nodes: List[Dict]) -> Dict:
        """
        Вычисляет центр масс конструкции (среднее положение узлов).
        """
        if not nodes:
            return {'x': 0, 'y': 0, 'z': 0}
        
        avg_x = sum(n['x'] for n in nodes) / len(nodes)
        avg_y = sum(n['y'] for n in nodes) / len(nodes)
        avg_z = sum(n['z'] for n in nodes) / len(nodes)
        
        return {'x': avg_x, 'y': avg_y, 'z': avg_z}
    
    @staticmethod
    def check_collision(beam1_start: Dict, beam1_end: Dict,
                       beam2_start: Dict, beam2_end: Dict,
                       tolerance: float = 0.1) -> bool:
        """
        Проверяет пересечение двух балок в 3D.
        
        Args:
            tolerance: минимальное расстояние для считывания пересечения (м)
        
        Returns:
            True, если балки пересекаются или находятся ближе tolerance
        """
        # Упрощенная проверка: расстояние между отрезками
        # Реализация алгоритма ближайших точек на двух отрезках
        
        # Это сложный алгоритм, упрощенная версия:
        # Проверяем расстояние от концов одной балки до другой балки
        
        min_dist = float('inf')
        
        # Проверяем 4 комбинации
        distances = [
            GeometryUtils.point_to_line_distance(beam1_start, beam2_start, beam2_end),
            GeometryUtils.point_to_line_distance(beam1_end, beam2_start, beam2_end),
            GeometryUtils.point_to_line_distance(beam2_start, beam1_start, beam1_end),
            GeometryUtils.point_to_line_distance(beam2_end, beam1_start, beam1_end)
        ]
        
        min_dist = min(distances)
        
        return min_dist < tolerance
    
    @staticmethod
    def interpolate_points_along_beam(start: Dict, end: Dict, num_points: int = 10) -> List[Dict]:
        """
        Создает промежуточные точки вдоль балки.
        Полезно для визуализации или детального анализа.
        
        Args:
            start, end: концы балки
            num_points: количество промежуточных точек
        
        Returns:
            Список точек от start до end
        """
        points = []
        
        for i in range(num_points + 1):
            t = i / num_points  # Параметр от 0 до 1
            
            x = start['x'] + t * (end['x'] - start['x'])
            y = start['y'] + t * (end['y'] - start['y'])
            z = start['z'] + t * (end['z'] - start['z'])
            
            points.append({'x': x, 'y': y, 'z': z})
        
        return points


class CollisionDetector:
    """Детектор коллизий между элементами конструкции и препятствиями"""
    
    def __init__(self):
        self.obstacles = []
    
    def add_obstacle(self, obstacle: Dict):
        """
        Добавляет препятствие.
        
        Args:
            obstacle: {
                "type": "box"|"cylinder"|"sphere",
                "position": {x, y, z},
                "dimensions": {...}  # зависит от типа
            }
        """
        self.obstacles.append(obstacle)
    
    def check_beam_collision(self, beam_start: Dict, beam_end: Dict, 
                            clearance: float = 0.2) -> List[Dict]:
        """
        Проверяет, пересекается ли балка с препятствиями.
        
        Args:
            clearance: зазор безопасности (метры)
        
        Returns:
            Список препятствий, с которыми есть коллизия
        """
        collisions = []
        
        for obstacle in self.obstacles:
            if self._beam_intersects_obstacle(beam_start, beam_end, obstacle, clearance):
                collisions.append(obstacle)
        
        return collisions
    
    def _beam_intersects_obstacle(self, beam_start: Dict, beam_end: Dict,
                                  obstacle: Dict, clearance: float) -> bool:
        """Проверяет пересечение балки с конкретным препятствием"""
        
        obs_type = obstacle.get('type', 'box')
        obs_pos = obstacle['position']
        
        if obs_type == 'box':
            # Препятствие = прямоугольный параллелепипед
            dims = obstacle['dimensions']
            
            # Расширяем bounding box на clearance
            box_min = {
                'x': obs_pos['x'] - clearance,
                'y': obs_pos['y'] - clearance,
                'z': obs_pos['z'] - clearance
            }
            box_max = {
                'x': obs_pos['x'] + dims['width'] + clearance,
                'y': obs_pos['y'] + dims['depth'] + clearance,
                'z': obs_pos['z'] + dims['height'] + clearance
            }
            
            # Проверяем, пересекает ли балка этот box
            # Упрощенно: проверяем концы и середину балки
            mid_point = {
                'x': (beam_start['x'] + beam_end['x']) / 2,
                'y': (beam_start['y'] + beam_end['y']) / 2,
                'z': (beam_start['z'] + beam_end['z']) / 2
            }
            
            return (GeometryUtils.is_point_inside_box(beam_start, box_min, box_max) or
                    GeometryUtils.is_point_inside_box(beam_end, box_min, box_max) or
                    GeometryUtils.is_point_inside_box(mid_point, box_min, box_max))
        
        elif obs_type == 'sphere':
            # Препятствие = сфера
            radius = obstacle['dimensions']['radius'] + clearance
            
            # Расстояние от центра сферы до балки
            dist = GeometryUtils.point_to_line_distance(obs_pos, beam_start, beam_end)
            
            return dist < radius
        
        else:
            # Неизвестный тип, не проверяем
            return False
