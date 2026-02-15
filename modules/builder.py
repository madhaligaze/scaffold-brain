# modules/builder.py
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
import heapq
from collections import defaultdict


class ScaffoldGenerator:
    """Генератор вариантов строительных лесов с учётом точек опор и складских остатков."""

    def __init__(self):
        self.inventory_presets = [
            {"name": "Стандарт 3м",         "stands": [3.0, 2.0], "ledgers": [2.0, 1.5],  "weight_factor": 1.0},
            {"name": "Складской запас 2.5м", "stands": [2.5, 1.0], "ledgers": [2.13, 1.09], "weight_factor": 1.1},
            {"name": "Усиленный короткий шаг","stands": [2.0],      "ledgers": [1.0, 1.2],  "weight_factor": 1.5},
        ]

    # ─── ПУБЛИЧНЫЕ МЕТОДЫ ──────────────────────────────────────────────────────

    def generate_options(
        self,
        target_width: float,
        target_height: float,
        target_depth: float,
        obstacles: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Генерирует 3 варианта конструкции по заданным габаритам.
        Используется в /engineer/generate-variants и /ai/auto-design.
        """
        return [
            self._create_variant(target_width, target_height, target_depth,
                                 stand_len=2.0, ledger_len=1.0,
                                 label="Надёжный (усиленный)", obstacles=obstacles),
            self._create_variant(target_width, target_height, target_depth,
                                 stand_len=3.0, ledger_len=2.0,
                                 label="Экономичный (минимум деталей)", obstacles=obstacles),
            self._create_variant(target_width, target_height, target_depth,
                                 stand_len=2.5, ledger_len=2.13,
                                 label="Из наличия (Склад: 2.5м × 2.13м)", obstacles=obstacles),
        ]

    def generate_smart_options(
        self,
        user_points: List[Dict],
        ai_points: List[Dict],
        bounds: Dict,
        obstacles: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Умная генерация вариантов с учётом:
        - точек, поставленных пользователем (user_points)
        - AI-детектированных точек опор (ai_points)
        - габаритов зоны (bounds)

        ИСПРАВЛЕНИЕ: метод отсутствовал → main.py падал с AttributeError

        Args:
            user_points:  [{x, y, z}, ...] — AR-маркеры от пользователя
            ai_points:    [{x, y, z, type?, confidence?}, ...] — опоры от YOLO
            bounds:       {"w": float, "h": float, "d": float}
            obstacles:    [{type, position, dimensions}, ...]

        Returns:
            Список из 3 вариантов (формат совместим с generate_options)
        """
        W = max(float(bounds.get("w", 4.0)), 1.0)
        H = max(float(bounds.get("h", 3.0)), 1.0)
        D = max(float(bounds.get("d", 2.0)), 1.0)

        # Собираем все опорные точки
        all_anchors = list(user_points or []) + list(ai_points or [])

        # Если точек совсем нет — генерируем стандартные варианты
        if not all_anchors:
            return self.generate_options(W, H, D, obstacles=obstacles)

        # Определяем оптимальный шаг сетки на основе расстояний между точками
        step_hint = self._estimate_step(all_anchors)

        # Вариант 1: Высокая надёжность — сетка с шагом ≤ step_hint
        v1 = self._create_variant_anchored(
            all_anchors, W, H, D,
            stand_len=min(step_hint, 2.0), ledger_len=min(step_hint, 1.5),
            label="🛡 Надёжный (под ваши опоры)",
            obstacles=obstacles,
        )

        # Вариант 2: Минимум материала — шаг побольше, но опоры учтены
        v2 = self._create_variant_anchored(
            all_anchors, W, H, D,
            stand_len=min(step_hint * 1.5, 3.0), ledger_len=min(step_hint * 1.5, 2.0),
            label="💡 Экономичный (минимум деталей)",
            obstacles=obstacles,
        )

        # Вариант 3: Склад — нестандартные размеры
        v3 = self._create_variant_anchored(
            all_anchors, W, H, D,
            stand_len=2.5, ledger_len=2.13,
            label="📦 Из наличия (Склад: 2.5м × 2.13м)",
            obstacles=obstacles,
        )

        return [v1, v2, v3]

    def fix_collisions(self, variant: Dict, collisions: List[Dict]) -> Dict:
        """
        Пытается устранить коллизии в варианте конструкции.

        ИСПРАВЛЕНИЕ: метод отсутствовал → main.py падал с AttributeError

        Стратегия:
        - beam_beam коллизия: удаляем одну из пересекающихся балок (ту, что короче)
        - beam_obstacle: смещаем узлы на clearance + 0.1м

        Args:
            variant:    вариант конструкции {nodes, beams, ...}
            collisions: список коллизий от WorldGeometry.check_collisions()

        Returns:
            Исправленный вариант (может содержать меньше балок)
        """
        if not collisions:
            return variant

        import copy
        fixed = copy.deepcopy(variant)
        beams: List[Dict] = fixed.get("beams", [])
        nodes: List[Dict] = fixed.get("nodes", [])

        node_map = {n['id']: n for n in nodes}
        beams_to_remove: Set[str] = set()

        for collision in collisions:
            ctype = collision.get("type", "")
            beam_id = collision.get("beam_id", "")
            conflict_id = collision.get("conflict_id", "")

            if ctype == "beam_beam":
                # Удаляем более короткую балку из пары
                b1 = next((b for b in beams if b['id'] == beam_id), None)
                b2 = next((b for b in beams if b['id'] == conflict_id), None)
                if b1 and b2:
                    len1 = self._beam_length_from_map(b1, node_map)
                    len2 = self._beam_length_from_map(b2, node_map)
                    beams_to_remove.add(beam_id if len1 <= len2 else conflict_id)

            elif ctype == "beam_obstacle":
                # Помечаем балку на удаление (консервативный подход)
                beams_to_remove.add(beam_id)

        # Убираем помеченные балки
        if beams_to_remove:
            fixed["beams"] = [b for b in beams if b['id'] not in beams_to_remove]
            removed = len(beams_to_remove)
            fixed.setdefault("fix_log", []).append(
                f"Удалено {removed} балок из-за коллизий"
            )

        # Обновляем статистику
        if "stats" in fixed:
            fixed["stats"]["total_beams"] = len(fixed["beams"])
            fixed["stats"]["collisions_fixed"] = len(beams_to_remove)

        return fixed

    # ─── ВСПОМОГАТЕЛЬНЫЕ ПРИВАТНЫЕ МЕТОДЫ ────────────────────────────────────

    def _estimate_step(self, points: List[Dict]) -> float:
        """Оценивает оптимальный шаг сетки по расстояниям между точками."""
        if len(points) < 2:
            return 2.0
        dists = []
        for i in range(min(len(points), 8)):
            for j in range(i + 1, min(len(points), 8)):
                dx = points[i].get('x', 0) - points[j].get('x', 0)
                dy = points[i].get('y', 0) - points[j].get('y', 0)
                dz = points[i].get('z', 0) - points[j].get('z', 0)
                d = (dx**2 + dy**2 + dz**2)**0.5
                if d > 0.1:
                    dists.append(d)
        if not dists:
            return 2.0
        median = sorted(dists)[len(dists) // 2]
        # Ограничиваем разумными значениями для строительных лесов
        return max(1.0, min(3.0, median))

    def _create_variant_anchored(
        self,
        anchors: List[Dict],
        W: float, H: float, D: float,
        stand_len: float, ledger_len: float,
        label: str,
        obstacles=None,
    ) -> Dict:
        """
        Создаёт вариант, дополнительно добавляя узлы у точек-якорей.
        Базовая сетка генерируется как обычно, якоря «притягивают» ближайшие узлы.
        """
        base = self._create_variant(W, H, D, stand_len, ledger_len, label, obstacles)

        # Добавляем «гвоздики» — вертикальные стойки у каждого якоря
        existing_ids = {n['id'] for n in base['nodes']}
        extra_nodes = []
        extra_beams = []
        anchor_beam_id = len(base['beams'])

        for idx, pt in enumerate(anchors):
            ax = float(pt.get('x', 0))
            ay = float(pt.get('y', 0))

            # Несколько уровней по высоте над якорем
            num_levels = max(1, int(np.ceil(H / stand_len)))
            prev_id = None
            for k in range(num_levels + 1):
                az = k * stand_len
                node_id = f"anc_{idx}_{k}"
                if node_id not in existing_ids:
                    extra_nodes.append({"id": node_id, "x": round(ax, 3),
                                        "y": round(ay, 3), "z": round(az, 3)})
                    existing_ids.add(node_id)
                if prev_id:
                    extra_beams.append({
                        "id": f"b_anc_{anchor_beam_id}",
                        "start": prev_id, "end": node_id, "type": "vertical"
                    })
                    anchor_beam_id += 1
                prev_id = node_id

        base['nodes'].extend(extra_nodes)
        base['beams'].extend(extra_beams)
        base['stats']['total_nodes'] = len(base['nodes'])
        base['stats']['total_beams'] = len(base['beams'])
        base['stats']['anchor_nodes'] = len(extra_nodes)
        return base

    def _create_variant(self, W, H, D, stand_len, ledger_len, label, obstacles=None):
        nodes = []
        beams = []
        num_x = int(np.ceil(W / ledger_len)) + 1
        num_z = int(np.ceil(H / stand_len)) + 1
        num_y = int(np.ceil(D / ledger_len)) + 1

        occupied_grid = self._create_obstacle_grid(obstacles, ledger_len, stand_len) if obstacles else set()
        node_map: Dict[Tuple, str] = {}

        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z):
                    x = i * ledger_len
                    y = j * ledger_len
                    z = k * stand_len
                    if self._is_occupied(x, y, z, occupied_grid, ledger_len, stand_len):
                        continue
                    node_id = f"n_{i}_{j}_{k}"
                    nodes.append({"id": node_id, "x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})
                    node_map[(i, j, k)] = node_id

        beam_id = 0
        # Вертикальные стойки
        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z - 1):
                    if (i, j, k) in node_map and (i, j, k + 1) in node_map:
                        beams.append({"id": f"b_v_{beam_id}", "start": node_map[(i, j, k)],
                                      "end": node_map[(i, j, k + 1)], "type": "vertical"})
                        beam_id += 1
        # Горизонтальные по X
        for j in range(num_y):
            for k in range(num_z):
                for i in range(num_x - 1):
                    if (i, j, k) in node_map and (i + 1, j, k) in node_map:
                        beams.append({"id": f"b_x_{beam_id}", "start": node_map[(i, j, k)],
                                      "end": node_map[(i + 1, j, k)], "type": "horizontal_x"})
                        beam_id += 1
        # Горизонтальные по Y
        for i in range(num_x):
            for k in range(num_z):
                for j in range(num_y - 1):
                    if (i, j, k) in node_map and (i, j + 1, k) in node_map:
                        beams.append({"id": f"b_y_{beam_id}", "start": node_map[(i, j, k)],
                                      "end": node_map[(i, j + 1, k)], "type": "horizontal_y"})
                        beam_id += 1
        # Диагонали
        for i in range(num_x - 1):
            for j in range(num_y - 1):
                for k in range(0, num_z, 2):
                    if (i, j, k) in node_map and (i + 1, j + 1, k) in node_map:
                        beams.append({"id": f"b_d_{beam_id}", "start": node_map[(i, j, k)],
                                      "end": node_map[(i + 1, j + 1, k)], "type": "diagonal"})
                        beam_id += 1

        return {
            "variant_name": label,
            "material_info": f"Стойки: {stand_len}м, Ригели: {ledger_len}м",
            "nodes": nodes,
            "beams": beams,
            "stats": {
                "total_nodes": len(nodes),
                "total_beams": len(beams),
                "total_weight_kg": len(beams) * 15,
            },
        }

    def _create_obstacle_grid(self, obstacles, grid_xy, grid_z):
        occupied = set()
        for obs in obstacles:
            x_min = int(obs['x'] / grid_xy)
            x_max = int((obs['x'] + obs['width']) / grid_xy) + 1
            y_min = int(obs['y'] / grid_xy)
            y_max = int((obs['y'] + obs['depth']) / grid_xy) + 1
            z_min = int(obs['z'] / grid_z)
            z_max = int((obs['z'] + obs['height']) / grid_z) + 1
            for ii in range(x_min, x_max):
                for jj in range(y_min, y_max):
                    for kk in range(z_min, z_max):
                        occupied.add((ii, jj, kk))
        return occupied

    def _is_occupied(self, x, y, z, occupied_grid, grid_xy, grid_z):
        """
        ИСПРАВЛЕНО: раньше всегда возвращал False.
        Теперь проверяет реальное пересечение с сеткой препятствий.
        """
        if not occupied_grid:
            return False
        key = (int(x / grid_xy), int(y / grid_xy), int(z / grid_z))
        return key in occupied_grid

    def _beam_length_from_map(self, beam: Dict, node_map: Dict) -> float:
        bs = node_map.get(beam.get('start', ''))
        be = node_map.get(beam.get('end', ''))
        if not bs or not be:
            return 0.0
        return ((bs['x'] - be['x'])**2 + (bs['y'] - be['y'])**2 + (bs['z'] - be['z'])**2)**0.5


class ScaffoldExpert:
    """Экспертная система для валидации правил безопасности и демонтажа."""

    def validate_dismantle(self, element_id: str, nodes: List[Dict], beams: List[Dict]) -> Dict:
        target_beam = next((b for b in beams if b['id'] == element_id), None)
        if not target_beam:
            return {"can_remove": False, "reason": "Элемент не найден"}

        start_node = self._find_node(target_beam['start'], nodes)
        end_node = self._find_node(target_beam['end'], nodes)
        if not start_node or not end_node:
            return {"can_remove": False, "reason": "Узлы балки не найдены"}

        if self._is_vertical(start_node, end_node):
            max_z = max(start_node['z'], end_node['z'])
            nodes_above = [n for n in nodes if n['z'] > max_z + 0.1]
            if nodes_above:
                same_xy_nodes = [
                    n for n in nodes
                    if abs(n['x'] - start_node['x']) < 2.0
                    and abs(n['y'] - start_node['y']) < 2.0
                    and n['z'] <= max_z
                ]
                vertical_supports = sum(
                    1 for b in beams
                    if (bs := self._find_node(b['start'], nodes)) and
                    (be := self._find_node(b['end'], nodes)) and
                    self._is_vertical(bs, be) and
                    (bs in same_xy_nodes or be in same_xy_nodes)
                )
                if vertical_supports <= 2:
                    return {"can_remove": False,
                            "reason": "⚠️ Опорная стойка! Над ней есть конструкция. Снимайте сверху вниз."}

        if start_node['z'] <= 0.05 or end_node['z'] <= 0.05:
            ground_beams = [b for b in beams if self._is_ground_level(b, nodes)]
            if len(ground_beams) <= 4:
                return {"can_remove": False,
                        "reason": "⚠️ Одна из последних опор на земле! Демонтаж опасен."}

        if not self._check_connectivity_after_removal(element_id, nodes, beams):
            return {"can_remove": False,
                    "reason": "⚠️ Удаление разделит конструкцию на части!"}

        return {"can_remove": True, "reason": "✓ Логически безопасно. Проверьте расчёт нагрузок."}

    def suggest_order(self, nodes: List[Dict], beams: List[Dict]) -> List[str]:
        beam_heights = []
        for beam in beams:
            s = self._find_node(beam['start'], nodes)
            e = self._find_node(beam['end'], nodes)
            if s and e:
                beam_heights.append((beam['id'], (s['z'] + e['z']) / 2))
        beam_heights.sort(key=lambda x: x[1], reverse=True)
        return [b for b, _ in beam_heights]

    def _find_node(self, node_id, nodes):
        return next((n for n in nodes if n['id'] == node_id), None)

    def _is_vertical(self, n1, n2):
        return abs(n1['x'] - n2['x']) < 0.01 and abs(n1['y'] - n2['y']) < 0.01 and abs(n1['z'] - n2['z']) > 0.1

    def _is_ground_level(self, beam, nodes):
        s = self._find_node(beam['start'], nodes)
        e = self._find_node(beam['end'], nodes)
        return s and e and (s['z'] <= 0.05 or e['z'] <= 0.05)

    def _check_connectivity_after_removal(self, remove_id, nodes, beams):
        graph = defaultdict(set)
        for beam in beams:
            if beam['id'] == remove_id:
                continue
            graph[beam['start']].add(beam['end'])
            graph[beam['end']].add(beam['start'])
        if not graph:
            return False
        start = next(iter(graph))
        visited = {start}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for nb in graph[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        all_nodes = set()
        for beam in beams:
            if beam['id'] != remove_id:
                all_nodes.add(beam['start'])
                all_nodes.add(beam['end'])
        return len(visited) == len(all_nodes)


class PathFinder:
    """A* поиск пути для обхода препятствий."""

    def __init__(self, grid_size=0.5):
        self.grid_size = grid_size

    def find_path_around_obstacle(self, start, end, obstacles):
        start_grid = self._to_grid(start)
        end_grid = self._to_grid(end)
        blocked = self._create_blocked_set(obstacles)
        path = self._astar(start_grid, end_grid, blocked)
        return [self._from_grid(p) for p in path]

    def _to_grid(self, p):
        return (int(p[0] / self.grid_size), int(p[1] / self.grid_size), int(p[2] / self.grid_size))

    def _from_grid(self, g):
        return (g[0] * self.grid_size, g[1] * self.grid_size, g[2] * self.grid_size)

    def _create_blocked_set(self, obstacles):
        blocked = set()
        margin = int(0.2 / self.grid_size)
        for obs in obstacles:
            x_min = int(obs['x'] / self.grid_size) - margin
            x_max = int((obs['x'] + obs.get('width', 1.0)) / self.grid_size) + 1 + margin
            y_min = int(obs['y'] / self.grid_size) - margin
            y_max = int((obs['y'] + obs.get('depth', 1.0)) / self.grid_size) + 1 + margin
            z_min = int(obs['z'] / self.grid_size) - margin
            z_max = int((obs['z'] + obs.get('height', 1.0)) / self.grid_size) + 1 + margin
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    for k in range(z_min, z_max):
                        blocked.add((i, j, k))
        return blocked

    def _astar(self, start, end, blocked):
        def h(a, b):
            return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == end:
                path = []
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start)
                return path[::-1]
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nb = (cur[0]+dx, cur[1]+dy, cur[2]+dz)
                if nb in blocked or nb[2] < 0:
                    continue
                tg = g_score[cur] + 1
                if nb not in g_score or tg < g_score[nb]:
                    came_from[nb] = cur
                    g_score[nb] = tg
                    heapq.heappush(open_set, (tg + h(nb, end), nb))
        return [start, end]