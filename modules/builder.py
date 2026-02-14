# modules/builder.py
import numpy as np
from typing import List, Dict, Tuple, Set
import heapq
from collections import defaultdict

class ScaffoldGenerator:
    """Генератор вариантов строительных лесов с учетом складских остатков"""
    
    def __init__(self):
        # Доступные наборы материалов (Склад)
        self.inventory_presets = [
            {"name": "Стандарт 3м", "stands": [3.0, 2.0], "ledgers": [2.0, 1.5], "weight_factor": 1.0},
            {"name": "Складской запас (2.5м)", "stands": [2.5, 1.0], "ledgers": [2.13, 1.09], "weight_factor": 1.1},
            {"name": "Усиленный (короткий шаг)", "stands": [2.0], "ledgers": [1.0, 1.2], "weight_factor": 1.5}
        ]

    def generate_options(self, target_width, target_height, target_depth, obstacles=None):
        """
        Генерирует 3 варианта конструкции на основе размеров объекта
        
        Args:
            target_width: ширина (X)
            target_height: высота (Z)
            target_depth: глубина (Y)
            obstacles: список препятствий [{x, y, z, width, height, depth}, ...]
        """
        options = []
        
        # 1. Вариант: "Максимальная надежность" (обычно мелкий шаг)
        options.append(self._create_variant(
            target_width, target_height, target_depth, 
            stand_len=2.0, ledger_len=1.0, 
            label="Надежный (усиленный)",
            obstacles=obstacles
        ))

        # 2. Вариант: "Минимум материала" (максимально длинные пролеты)
        options.append(self._create_variant(
            target_width, target_height, target_depth, 
            stand_len=3.0, ledger_len=2.0, 
            label="Экономичный (минимум деталей)",
            obstacles=obstacles
        ))

        # 3. Вариант: "Наличие на складе" (нестандартные размеры)
        options.append(self._create_variant(
            target_width, target_height, target_depth, 
            stand_len=2.5, ledger_len=2.13, 
            label="Из наличия (Склад: 2.5м x 2.13м)",
            obstacles=obstacles
        ))

        return options

    def _create_variant(self, W, H, D, stand_len, ledger_len, label, obstacles=None):
        """
        Вспомогательная функция для построения сетки узлов и балок с обходом препятствий
        
        Args:
            W, H, D: габариты
            stand_len: длина стоек
            ledger_len: длина ригелей/ledgers
            label: название варианта
            obstacles: препятствия для обхода
        """
        nodes = []
        beams = []
        
        # Расчет количества секций
        num_x = int(np.ceil(W / ledger_len)) + 1
        num_z = int(np.ceil(H / stand_len)) + 1
        num_y = int(np.ceil(D / ledger_len)) + 1
        
        # Сетка для проверки занятости (для обхода препятствий)
        occupied_grid = self._create_obstacle_grid(obstacles, ledger_len, stand_len) if obstacles else set()

        # Генерация узлов (сетка с обходом препятствий)
        node_map = {}  # (i, j, k) -> node_id
        
        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z):
                    x = i * ledger_len
                    y = j * ledger_len
                    z = k * stand_len
                    
                    # Проверка: не попадает ли узел в зону препятствия
                    if self._is_occupied(x, y, z, occupied_grid):
                        continue
                    
                    node_id = f"n_{i}_{j}_{k}"
                    nodes.append({
                        "id": node_id,
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "z": round(z, 3)
                    })
                    node_map[(i, j, k)] = node_id

        # Генерация балок (соединения между узлами)
        beam_id = 0
        
        # 1. Вертикальные стойки (по Z)
        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z - 1):
                    start_key = (i, j, k)
                    end_key = (i, j, k + 1)
                    
                    if start_key in node_map and end_key in node_map:
                        beams.append({
                            "id": f"b_v_{beam_id}",
                            "start": node_map[start_key],
                            "end": node_map[end_key],
                            "type": "vertical"
                        })
                        beam_id += 1
        
        # 2. Горизонтальные ригели (по X)
        for j in range(num_y):
            for k in range(num_z):
                for i in range(num_x - 1):
                    start_key = (i, j, k)
                    end_key = (i + 1, j, k)
                    
                    if start_key in node_map and end_key in node_map:
                        beams.append({
                            "id": f"b_x_{beam_id}",
                            "start": node_map[start_key],
                            "end": node_map[end_key],
                            "type": "horizontal_x"
                        })
                        beam_id += 1
        
        # 3. Горизонтальные ригели (по Y)
        for i in range(num_x):
            for k in range(num_z):
                for j in range(num_y - 1):
                    start_key = (i, j, k)
                    end_key = (i, j + 1, k)
                    
                    if start_key in node_map and end_key in node_map:
                        beams.append({
                            "id": f"b_y_{beam_id}",
                            "start": node_map[start_key],
                            "end": node_map[end_key],
                            "type": "horizontal_y"
                        })
                        beam_id += 1
        
        # 4. Диагональные связи (для жесткости)
        for i in range(num_x - 1):
            for j in range(num_y - 1):
                for k in range(0, num_z, 2):  # Диагонали через уровень
                    # Диагональ в плоскости XY
                    if (i, j, k) in node_map and (i + 1, j + 1, k) in node_map:
                        beams.append({
                            "id": f"b_diag_{beam_id}",
                            "start": node_map[(i, j, k)],
                            "end": node_map[(i + 1, j + 1, k)],
                            "type": "diagonal"
                        })
                        beam_id += 1
        
        return {
            "variant_name": label,
            "material_info": f"Стойки: {stand_len}м, Ригели: {ledger_len}м",
            "nodes": nodes,
            "beams": beams,
            "stats": {
                "total_nodes": len(nodes),
                "total_beams": len(beams),
                "total_weight_kg": len(beams) * 15  # примерно 15 кг на балку
            }
        }
    
    def _create_obstacle_grid(self, obstacles, grid_size_xy, grid_size_z):
        """Создает набор занятых ячеек сетки"""
        occupied = set()
        
        for obs in obstacles:
            x_min = int(obs['x'] / grid_size_xy)
            x_max = int((obs['x'] + obs['width']) / grid_size_xy) + 1
            y_min = int(obs['y'] / grid_size_xy)
            y_max = int((obs['y'] + obs['depth']) / grid_size_xy) + 1
            z_min = int(obs['z'] / grid_size_z)
            z_max = int((obs['z'] + obs['height']) / grid_size_z) + 1
            
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    for k in range(z_min, z_max):
                        occupied.add((i, j, k))
        
        return occupied
    
    def _is_occupied(self, x, y, z, occupied_grid):
        """Проверяет, занята ли точка препятствием"""
        # Упрощенная проверка - можно улучшить
        return False  # Пока отключено, т.к. нет точных координат препятствий


class ScaffoldExpert:
    """
    Экспертная система для валидации правил безопасности и демонтажа.
    Реализует логику "можно ли снять эту балку", основанную на здравом смысле.
    """
    
    def __init__(self):
        pass
    
    def validate_dismantle(self, element_id: str, nodes: List[Dict], beams: List[Dict]) -> Dict:
        """
        Проверяет логическую безопасность удаления элемента.
        Правила демонтажа:
        1. Сверху вниз (не снимать то, на чем что-то стоит)
        2. Не снимать единственную опору
        3. Не снимать балку, если над ней есть нагруженные узлы
        
        Returns:
            {
                "can_remove": bool,
                "reason": str
            }
        """
        # Найти балку
        target_beam = None
        for b in beams:
            if b['id'] == element_id:
                target_beam = b
                break
        
        if not target_beam:
            return {"can_remove": False, "reason": "Элемент не найден"}
        
        # Получить узлы балки
        start_node = self._find_node(target_beam['start'], nodes)
        end_node = self._find_node(target_beam['end'], nodes)
        
        if not start_node or not end_node:
            return {"can_remove": False, "reason": "Узлы балки не найдены"}
        
        # Правило 1: Проверка на вертикальные стойки (не снимать опорные)
        if self._is_vertical(start_node, end_node):
            # Проверяем, есть ли что-то над этой стойкой
            max_z = max(start_node['z'], end_node['z'])
            
            # Ищем узлы выше
            nodes_above = [n for n in nodes if n['z'] > max_z + 0.1]
            
            if nodes_above:
                # Проверяем, есть ли другие опоры в радиусе 2м
                same_xy_nodes = [
                    n for n in nodes 
                    if abs(n['x'] - start_node['x']) < 2.0 
                    and abs(n['y'] - start_node['y']) < 2.0
                    and n['z'] <= max_z
                ]
                
                # Считаем вертикальные стойки в этой зоне
                vertical_supports = 0
                for b in beams:
                    b_start = self._find_node(b['start'], nodes)
                    b_end = self._find_node(b['end'], nodes)
                    if b_start and b_end and self._is_vertical(b_start, b_end):
                        if b_start in same_xy_nodes or b_end in same_xy_nodes:
                            vertical_supports += 1
                
                if vertical_supports <= 2:
                    return {
                        "can_remove": False,
                        "reason": "⚠️ Это опорная стойка! Над ней есть конструкция. Снимайте сверху вниз."
                    }
        
        # Правило 2: Проверка на земляные опоры
        if start_node['z'] <= 0.05 or end_node['z'] <= 0.05:
            # Проверяем, сколько балок на земле
            ground_beams = [
                b for b in beams 
                if self._is_ground_level(b, nodes)
            ]
            
            if len(ground_beams) <= 4:
                return {
                    "can_remove": False,
                    "reason": "⚠️ Это одна из последних опор на земле! Демонтаж опасен."
                }
        
        # Правило 3: Проверка связности конструкции после удаления
        if not self._check_connectivity_after_removal(element_id, nodes, beams):
            return {
                "can_remove": False,
                "reason": "⚠️ Удаление этой балки разделит конструкцию на части!"
            }
        
        # Если все проверки пройдены
        return {
            "can_remove": True,
            "reason": "✓ Логически безопасно. Проверьте расчет нагрузок."
        }
    
    def suggest_order(self, nodes: List[Dict], beams: List[Dict]) -> List[str]:
        """
        Предлагает правильную последовательность демонтажа.
        Стратегия: сверху вниз, от периферии к центру.
        
        Returns:
            Список ID балок в порядке безопасного демонтажа
        """
        # Сортируем балки по высоте (сверху вниз)
        beam_heights = []
        
        for beam in beams:
            start_node = self._find_node(beam['start'], nodes)
            end_node = self._find_node(beam['end'], nodes)
            
            if start_node and end_node:
                avg_z = (start_node['z'] + end_node['z']) / 2
                beam_heights.append((beam['id'], avg_z))
        
        # Сортируем по убыванию высоты
        beam_heights.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем только ID
        order = [beam_id for beam_id, _ in beam_heights]
        
        return order
    
    def _find_node(self, node_id: str, nodes: List[Dict]) -> Dict:
        """Поиск узла по ID"""
        for n in nodes:
            if n['id'] == node_id:
                return n
        return None
    
    def _is_vertical(self, node1: Dict, node2: Dict) -> bool:
        """Проверка: является ли балка вертикальной"""
        dx = abs(node1['x'] - node2['x'])
        dy = abs(node1['y'] - node2['y'])
        dz = abs(node1['z'] - node2['z'])
        
        # Вертикальная, если движение только по Z
        return dx < 0.01 and dy < 0.01 and dz > 0.1
    
    def _is_ground_level(self, beam: Dict, nodes: List[Dict]) -> bool:
        """Проверка: балка на уровне земли?"""
        start_node = self._find_node(beam['start'], nodes)
        end_node = self._find_node(beam['end'], nodes)
        
        if not start_node or not end_node:
            return False
        
        return start_node['z'] <= 0.05 or end_node['z'] <= 0.05
    
    def _check_connectivity_after_removal(self, remove_id: str, nodes: List[Dict], beams: List[Dict]) -> bool:
        """
        Проверяет, останется ли конструкция связной после удаления балки.
        Использует поиск в ширину (BFS).
        """
        # Создаем граф смежности без удаляемой балки
        graph = defaultdict(set)
        
        for beam in beams:
            if beam['id'] == remove_id:
                continue
            graph[beam['start']].add(beam['end'])
            graph[beam['end']].add(beam['start'])
        
        if not graph:
            return False
        
        # Проверяем связность через BFS
        visited = set()
        start_node = next(iter(graph.keys()))
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Все узлы с балками должны быть достижимы
        all_connected_nodes = set()
        for beam in beams:
            if beam['id'] != remove_id:
                all_connected_nodes.add(beam['start'])
                all_connected_nodes.add(beam['end'])
        
        return len(visited) == len(all_connected_nodes)


class PathFinder:
    """
    Алгоритм поиска пути для обхода препятствий при генерации лесов.
    Использует A* для нахождения оптимального маршрута балки.
    """
    
    def __init__(self, grid_size=0.5):
        self.grid_size = grid_size
    
    def find_path_around_obstacle(self, start: Tuple[float, float, float], 
                                  end: Tuple[float, float, float],
                                  obstacles: List[Dict]) -> List[Tuple[float, float, float]]:
        """
        Находит путь от start до end, обходя препятствия.
        
        Returns:
            Список промежуточных точек маршрута
        """
        # Дискретизация пространства
        start_grid = self._to_grid(start)
        end_grid = self._to_grid(end)
        
        # Создаем набор занятых ячеек
        blocked = self._create_blocked_set(obstacles)
        
        # A* поиск
        path = self._astar(start_grid, end_grid, blocked)
        
        # Конвертируем обратно в метры
        return [self._from_grid(p) for p in path]
    
    def _to_grid(self, point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Конвертирует координаты в сетку"""
        return (
            int(point[0] / self.grid_size),
            int(point[1] / self.grid_size),
            int(point[2] / self.grid_size)
        )
    
    def _from_grid(self, grid_point: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Конвертирует из сетки обратно в метры"""
        return (
            grid_point[0] * self.grid_size,
            grid_point[1] * self.grid_size,
            grid_point[2] * self.grid_size
        )
    
    def _create_blocked_set(self, obstacles: List[Dict]) -> Set[Tuple[int, int, int]]:
        """Создает набор заблокированных ячеек"""
        blocked = set()
        
        for obs in obstacles:
            x_min = int(obs['x'] / self.grid_size)
            x_max = int((obs['x'] + obs.get('width', 1.0)) / self.grid_size) + 1
            y_min = int(obs['y'] / self.grid_size)
            y_max = int((obs['y'] + obs.get('depth', 1.0)) / self.grid_size) + 1
            z_min = int(obs['z'] / self.grid_size)
            z_max = int((obs['z'] + obs.get('height', 1.0)) / self.grid_size) + 1
            
            # Добавляем зазор 20 см
            margin = int(0.2 / self.grid_size)
            
            for i in range(x_min - margin, x_max + margin):
                for j in range(y_min - margin, y_max + margin):
                    for k in range(z_min - margin, z_max + margin):
                        blocked.add((i, j, k))
        
        return blocked
    
    def _astar(self, start: Tuple[int, int, int], 
              end: Tuple[int, int, int], 
              blocked: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        A* алгоритм поиска кратчайшего пути
        """
        def heuristic(a, b):
            # Евклидово расстояние
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
        
        def neighbors(point):
            # 6 направлений (без диагоналей для упрощения)
            directions = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1)
            ]
            
            result = []
            for dx, dy, dz in directions:
                next_point = (point[0] + dx, point[1] + dy, point[2] + dz)
                if next_point not in blocked and next_point[2] >= 0:  # Не уходим под землю
                    result.append(next_point)
            return result
        
        # Приоритетная очередь: (приоритет, точка)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end:
                # Восстанавливаем путь
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Путь не найден
        return [start, end]
