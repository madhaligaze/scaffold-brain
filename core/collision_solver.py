"""
Intelligent Collision Solver - "Вода обтекает камень"
=======================================================
ПРАВИЛО: ИИ должен обходить препятствия, а НЕ удалять элементы.

Если на фото есть труба, а ИИ ставит стойку сквозь неё — это провал.
Используем trimesh для жесткой детекции коллизий.
ИИ должен обходить препятствия, как вода обтекает камень.
"""
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import copy

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None


@dataclass
class Obstacle:
    """Препятствие в сцене"""
    id: str
    type: str  # "pipe", "wall", "column", "window", etc.
    position: Tuple[float, float, float]  # (x, y, z) центра
    dimensions: Tuple[float, float, float]  # (width, height, depth)
    rotation: float = 0.0  # Угол поворота в радианах
    mesh: Optional['trimesh.Trimesh'] = None  # 3D меш для точной коллизии
    
    def get_bbox(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Возвращает bounding box: (min_point, max_point)"""
        w, h, d = self.dimensions
        x, y, z = self.position
        
        return (
            (x - w/2, y - d/2, z - h/2),
            (x + w/2, y + d/2, z + h/2)
        )
    
    def contains_point(self, point: Tuple[float, float, float], 
                      clearance: float = 0.1) -> bool:
        """
        Проверяет, находится ли точка внутри препятствия (с учетом зазора).
        
        Args:
            point: (x, y, z) координаты точки
            clearance: Дополнительный зазор в метрах
            
        Returns:
            True если точка внутри препятствия
        """
        min_pt, max_pt = self.get_bbox()
        x, y, z = point
        
        return (
            min_pt[0] - clearance <= x <= max_pt[0] + clearance and
            min_pt[1] - clearance <= y <= max_pt[1] + clearance and
            min_pt[2] - clearance <= z <= max_pt[2] + clearance
        )


@dataclass
class CollisionResult:
    """Результат проверки коллизий"""
    has_collision: bool
    collisions: List[Dict]  # Список коллизий
    
    def __repr__(self):
        if not self.has_collision:
            return "✓ Коллизий нет"
        return f"✗ Найдено коллизий: {len(self.collisions)}"


class CollisionSolver:
    """
    Решатель коллизий.
    
    Стратегия "Воды":
    1. Обнаружить коллизию
    2. Сдвинуть узлы на минимальное безопасное расстояние
    3. Если сдвиг невозможен — предложить альтернативный путь
    """
    
    def __init__(self, clearance: float = 0.15):
        """
        Args:
            clearance: Минимальный зазор от препятствий (метры)
        """
        self.clearance = clearance
        self.collision_cache: Dict[str, bool] = {}
        self.non_removable_types = {"standard", "ledger", "transom"}
    
    def check_beam_obstacle_collision(self, beam_start: Tuple[float, float, float],
                                     beam_end: Tuple[float, float, float],
                                     obstacle: Obstacle) -> bool:
        """
        Проверка пересечения балки с препятствием.
        
        Использует алгоритм пересечения отрезка с AABB (Axis-Aligned Bounding Box).
        
        Args:
            beam_start: (x, y, z) начала балки
            beam_end: (x, y, z) конца балки
            obstacle: Препятствие
            
        Returns:
            True если есть пересечение
        """
        # Если есть trimesh меш — используем точную проверку
        if TRIMESH_AVAILABLE and obstacle.mesh is not None:
            return self._check_beam_mesh_collision(beam_start, beam_end, obstacle.mesh)
        
        # Иначе используем AABB
        min_pt, max_pt = obstacle.get_bbox()
        
        # Расширяем AABB на clearance
        min_pt = tuple(m - self.clearance for m in min_pt)
        max_pt = tuple(m + self.clearance for m in max_pt)
        
        return self._line_intersects_aabb(beam_start, beam_end, min_pt, max_pt)
    
    def _line_intersects_aabb(self, p1: Tuple[float, float, float],
                             p2: Tuple[float, float, float],
                             aabb_min: Tuple[float, float, float],
                             aabb_max: Tuple[float, float, float]) -> bool:
        """
        Проверка пересечения отрезка с AABB.
        
        Использует алгоритм slab method.
        """
        # Направление луча
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        
        # Избегаем деления на ноль
        if abs(dx) < 1e-6:
            dx = 1e-6
        if abs(dy) < 1e-6:
            dy = 1e-6
        if abs(dz) < 1e-6:
            dz = 1e-6
        
        # Вычисляем t для пересечений
        t1 = (aabb_min[0] - p1[0]) / dx
        t2 = (aabb_max[0] - p1[0]) / dx
        t3 = (aabb_min[1] - p1[1]) / dy
        t4 = (aabb_max[1] - p1[1]) / dy
        t5 = (aabb_min[2] - p1[2]) / dz
        t6 = (aabb_max[2] - p1[2]) / dz
        
        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
        
        # Нет пересечения если tmax < 0 или tmin > tmax
        if tmax < 0 or tmin > tmax:
            return False
        
        # Нет пересечения если tmin > 1 (пересечение за концом отрезка)
        if tmin > 1.0:
            return False
        
        return True
    
    def _check_beam_mesh_collision(self, beam_start: Tuple[float, float, float],
                                   beam_end: Tuple[float, float, float],
                                   mesh: 'trimesh.Trimesh') -> bool:
        """Точная проверка пересечения с trimesh мешем"""
        if not TRIMESH_AVAILABLE or mesh is None:
            return False
        
        # Создаем луч
        ray_origins = [beam_start]
        ray_directions = [
            (beam_end[0] - beam_start[0],
             beam_end[1] - beam_start[1],
             beam_end[2] - beam_start[2])
        ]
        
        # Проверяем пересечение
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )
        
        return len(locations) > 0
    
    def detect_all_collisions(self, nodes: List[Dict], beams: List[Dict],
                             obstacles: List[Obstacle]) -> CollisionResult:
        """
        Обнаружение всех коллизий в конструкции.
        
        Args:
            nodes: Список узлов
            beams: Список балок
            obstacles: Список препятствий
            
        Returns:
            CollisionResult с детальным списком коллизий
        """
        collisions = []
        node_map = {n['id']: n for n in nodes}
        
        # Проверяем каждую балку с каждым препятствием
        for beam in beams:
            start_node = node_map.get(beam['start'])
            end_node = node_map.get(beam['end'])
            
            if not start_node or not end_node:
                continue
            
            beam_start = (start_node['x'], start_node['y'], start_node['z'])
            beam_end = (end_node['x'], end_node['y'], end_node['z'])
            
            for obstacle in obstacles:
                if self.check_beam_obstacle_collision(beam_start, beam_end, obstacle):
                    collisions.append({
                        "type": "beam_obstacle",
                        "beam_id": beam['id'],
                        "obstacle_id": obstacle.id,
                        "obstacle_type": obstacle.type,
                        "severity": "high"
                    })
        
        # Проверяем узлы внутри препятствий
        for node in nodes:
            point = (node['x'], node['y'], node['z'])
            
            for obstacle in obstacles:
                if obstacle.contains_point(point, self.clearance):
                    collisions.append({
                        "type": "node_obstacle",
                        "node_id": node['id'],
                        "obstacle_id": obstacle.id,
                        "obstacle_type": obstacle.type,
                        "severity": "critical"
                    })
        
        return CollisionResult(
            has_collision=len(collisions) > 0,
            collisions=collisions
        )
    
    def resolve_collisions(self, nodes: List[Dict], beams: List[Dict],
                          obstacles: List[Obstacle],
                          max_iterations: int = 10) -> Dict:
        """
        УМНОЕ РЕШЕНИЕ КОЛЛИЗИЙ - автоматическое исправление.
        
        Стратегия "Воды":
        1. Находим все коллизии
        2. Сдвигаем узлы на минимальное безопасное расстояние
        3. Если узел нельзя сдвинуть — ищем альтернативный путь
        4. Если путь не найден — удаляем балку (крайняя мера)
        
        Args:
            nodes: Узлы конструкции
            beams: Балки конструкции
            obstacles: Препятствия
            max_iterations: Максимум итераций
            
        Returns:
            {
                "nodes": [...],  # Исправленные узлы
                "beams": [...],  # Исправленные балки
                "iterations": int,
                "removed_beams": List[str],
                "moved_nodes": List[str],
                "success": bool
            }
        """
        resolved_nodes = copy.deepcopy(nodes)
        resolved_beams = copy.deepcopy(beams)
        
        removed_beams = []
        moved_nodes = set()
        
        for iteration in range(max_iterations):
            # Проверяем коллизии
            result = self.detect_all_collisions(resolved_nodes, resolved_beams, obstacles)
            
            if not result.has_collision:
                # Все коллизии решены!
                return {
                    "nodes": resolved_nodes,
                    "beams": resolved_beams,
                    "iterations": iteration,
                    "removed_beams": removed_beams,
                    "moved_nodes": list(moved_nodes),
                    "success": True
                }
            
            # Обрабатываем каждую коллизию
            for collision in result.collisions:
                if collision['type'] == 'node_obstacle':
                    # Пытаемся сдвинуть узел
                    node_id = collision['node_id']
                    obstacle_id = collision['obstacle_id']
                    obstacle = next((o for o in obstacles if o.id == obstacle_id), None)
                    
                    if obstacle:
                        success = self._move_node_away_from_obstacle(
                            resolved_nodes, node_id, obstacle
                        )
                        if success:
                            moved_nodes.add(node_id)
                
                elif collision['type'] == 'beam_obstacle':
                    # Пытаемся сдвинуть оба конца балки
                    beam_id = collision['beam_id']
                    obstacle_id = collision['obstacle_id']
                    obstacle = next((o for o in obstacles if o.id == obstacle_id), None)
                    
                    if obstacle:
                        beam = next((b for b in resolved_beams if b['id'] == beam_id), None)
                        if beam:
                            # Пытаемся сдвинуть начало и конец
                            moved_start = self._move_node_away_from_obstacle(
                                resolved_nodes, beam['start'], obstacle
                            )
                            moved_end = self._move_node_away_from_obstacle(
                                resolved_nodes, beam['end'], obstacle
                            )
                            
                            if moved_start:
                                moved_nodes.add(beam['start'])
                            if moved_end:
                                moved_nodes.add(beam['end'])
                            
                            # Если не помогло: несущие элементы удалять ЗАПРЕЩЕНО
                            if not (moved_start or moved_end):
                                beam_type = str(beam.get("type", "")).lower()
                                if beam_type in self.non_removable_types:
                                    return {
                                        "nodes": resolved_nodes,
                                        "beams": resolved_beams,
                                        "iterations": iteration + 1,
                                        "removed_beams": removed_beams,
                                        "moved_nodes": list(moved_nodes),
                                        "success": False,
                                        "error": "Невозможно построить: препятствие в несущей зоне"
                                    }

                                # Для второстепенных элементов допускается удаление как крайняя мера
                                resolved_beams = [b for b in resolved_beams if b["id"] != beam_id]
                                removed_beams.append(beam_id)
        
        # Достигли максимума итераций
        return {
            "nodes": resolved_nodes,
            "beams": resolved_beams,
            "iterations": max_iterations,
            "removed_beams": removed_beams,
            "moved_nodes": list(moved_nodes),
            "success": False  # Не все коллизии решены
        }
    
    def _move_node_away_from_obstacle(self, nodes: List[Dict], node_id: str,
                                     obstacle: Obstacle) -> bool:
        """
        Сдвигает узел от препятствия на безопасное расстояние.
        
        Args:
            nodes: Список узлов (модифицируется in-place)
            node_id: ID узла для сдвига
            obstacle: Препятствие
            
        Returns:
            True если узел успешно сдвинут
        """
        node = next((n for n in nodes if n['id'] == node_id), None)
        if not node:
            return False
        
        # Если узел на земле (закреплен) — не сдвигаем
        if node['z'] <= 0.05:
            return False
        
        # Вычисляем направление от центра препятствия к узлу
        ox, oy, oz = obstacle.position
        nx, ny, nz = node['x'], node['y'], node['z']
        
        dx = nx - ox
        dy = ny - oy
        dz = nz - oz
        
        # Нормализуем вектор
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        if length < 1e-6:
            # Узел точно в центре — сдвигаем в случайном направлении
            dx, dy, dz = 1.0, 0.0, 0.0
            length = 1.0
        
        dx /= length
        dy /= length
        dz /= length
        
        # Сдвигаем узел на (радиус препятствия + clearance)
        w, h, d = obstacle.dimensions
        safe_distance = max(w, d) / 2 + self.clearance + 0.2  # Дополнительный запас
        
        node['x'] = ox + dx * safe_distance
        node['y'] = oy + dy * safe_distance
        node['z'] = max(nz, 0.05)  # Не уходим под землю
        
        return True
    
    def suggest_alternative_path(self, start: Tuple[float, float, float],
                                end: Tuple[float, float, float],
                                obstacles: List[Obstacle]) -> Optional[List[Tuple[float, float, float]]]:
        """
        Предлагает альтернативный путь в обход препятствий.
        
        Используется упрощенный A* алгоритм.
        
        Args:
            start: Начальная точка
            end: Конечная точка
            obstacles: Препятствия
            
        Returns:
            Список точек пути или None если путь не найден
        """
        # Упрощенная версия: пытаемся обойти по дуге
        # В полной версии нужен A* с сеткой
        
        # Вычисляем центральную точку
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        mid_z = (start[2] + end[2]) / 2
        
        # Пробуем точки вокруг центра
        offsets = [
            (0.5, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, -0.5, 0.0),
        ]
        
        for offset in offsets:
            waypoint = (
                mid_x + offset[0],
                mid_y + offset[1],
                mid_z + offset[2]
            )
            
            # Проверяем, свободен ли путь через эту точку
            path_clear = True
            for obstacle in obstacles:
                if (self.check_beam_obstacle_collision(start, waypoint, obstacle) or
                    self.check_beam_obstacle_collision(waypoint, end, obstacle)):
                    path_clear = False
                    break
            
            if path_clear:
                return [start, waypoint, end]
        
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════

def create_obstacle_from_detection(detection: Dict) -> Obstacle:
    """
    Создает объект Obstacle из результата детекции YOLO.
    
    Args:
        detection: {type, position, dimensions, confidence, ...}
        
    Returns:
        Obstacle объект
    """
    return Obstacle(
        id=f"obs_{detection.get('type', 'unknown')}_{id(detection)}",
        type=detection.get('type', 'unknown'),
        position=tuple(detection.get('position', (0, 0, 0))),
        dimensions=tuple(detection.get('dimensions', (0.5, 2.0, 0.5))),
        rotation=detection.get('rotation', 0.0)
    )


if __name__ == "__main__":
    print("🧪 ТЕСТИРОВАНИЕ COLLISION SOLVER")
    print("=" * 70)
    
    # Тест 1: Создание препятствия
    print("\n1. Создание препятствия:")
    pipe = Obstacle(
        id="pipe_1",
        type="pipe",
        position=(2.0, 1.0, 1.5),
        dimensions=(0.3, 2.0, 0.3)  # Вертикальная труба
    )
    print(f"   {pipe.type} в позиции {pipe.position}")
    
    # Тест 2: Проверка точки внутри препятствия
    print("\n2. Проверка точек:")
    test_points = [
        (2.0, 1.0, 1.5),  # Внутри
        (3.0, 1.0, 1.5),  # Снаружи
    ]
    for point in test_points:
        inside = pipe.contains_point(point)
        print(f"   {point}: {'ВНУТРИ' if inside else 'снаружи'}")
    
    # Тест 3: Коллизия балки
    print("\n3. Проверка коллизии балки:")
    solver = CollisionSolver(clearance=0.15)
    
    beam_safe = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    beam_collision = ((1.5, 1.0, 1.0), (2.5, 1.0, 2.0))
    
    safe_result = solver.check_beam_obstacle_collision(beam_safe[0], beam_safe[1], pipe)
    collision_result = solver.check_beam_obstacle_collision(
        beam_collision[0], beam_collision[1], pipe
    )
    
    print(f"   Балка 1 (безопасная): {'КОЛЛИЗИЯ!' if safe_result else 'OK'}")
    print(f"   Балка 2 (через трубу): {'КОЛЛИЗИЯ!' if collision_result else 'OK'}")
    
    # Тест 4: Решение коллизий
    print("\n4. Автоматическое решение коллизий:")
    test_nodes = [
        {"id": "n1", "x": 1.5, "y": 1.0, "z": 0.0},
        {"id": "n2", "x": 2.5, "y": 1.0, "z": 2.0},
    ]
    test_beams = [
        {"id": "b1", "start": "n1", "end": "n2"}
    ]
    
    result = solver.resolve_collisions(test_nodes, test_beams, [pipe])
    print(f"   Успех: {result['success']}")
    print(f"   Итераций: {result['iterations']}")
    print(f"   Узлов сдвинуто: {len(result['moved_nodes'])}")
    print(f"   Балок удалено: {len(result['removed_beams'])}")
    
    print("\n" + "=" * 70)
    print("✓ Тесты завершены!")