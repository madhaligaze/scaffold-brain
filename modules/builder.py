# modules/builder.py
import numpy as np
from typing import List, Dict, Tuple, Set
import heapq
from collections import defaultdict

class ScaffoldGenerator:
    """Генератор вариантов строительных лесов с учетом складских остатков"""

    LAYHER_LEDGER_STANDARDS = [0.73, 1.09, 1.57, 2.07, 2.57, 3.07]
    LAYHER_LEDGER_WEIGHT_KG = {
        0.73: 5.8,
        1.09: 7.2,
        1.57: 9.1,
        2.07: 11.4,
        2.57: 13.8,
        3.07: 16.6,
    }
    LAYHER_STAND_WEIGHT_KG = {
        1.0: 6.2,
        2.0: 11.9,
        2.5: 14.3,
        3.0: 16.8,
    }
    LAYHER_DECK_WEIGHT_KG = {1.57: 15.5, 2.07: 18.2, 2.57: 21.0, 3.07: 23.4}
    LAYHER_ACCESS_DECK_WEIGHT_KG = {1.57: 20.6, 2.07: 24.5, 2.57: 28.7, 3.07: 32.8}
    LAYHER_COUPLER_WEIGHT_KG = 1.25
    
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
            stand_len=2.0, ledger_len=1.09, 
            label="Надежный (усиленный)",
            obstacles=obstacles
        ))

        # 2. Вариант: "Минимум материала" (максимально длинные пролеты)
        options.append(self._create_variant(
            target_width, target_height, target_depth, 
            stand_len=3.0, ledger_len=2.07, 
            label="Экономичный (минимум деталей)",
            obstacles=obstacles
        ))

        # 3. Вариант: "Наличие на складе" (нестандартные размеры)
        options.append(self._create_variant(
            target_width, target_height, target_depth, 
            stand_len=2.5, ledger_len=2.57, 
            label="Из наличия (Склад: 2.5м x 2.57м)",
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
        
        node_lookup = {node["id"]: node for node in nodes}
        nomenclature = self._generate_nomenclature(beams, node_lookup)
        beam_weight = self._calculate_beam_weight(beams, node_lookup)
        extra_weight = self._calculate_nomenclature_weight(nomenclature)

        return {
            "variant_name": label,
            "material_info": f"Стойки: {stand_len}м, Ригели: {ledger_len}м",
            "nodes": nodes,
            "beams": beams,
            "nomenclature": nomenclature,
            "stats": {
                "total_nodes": len(nodes),
                "total_beams": len(beams),
                "total_decks": len(nomenclature["decks"]),
                "total_access_decks": len(nomenclature["access_decks"]),
                "total_couplers": len(nomenclature["couplers"]),
                "total_weight_kg": round(beam_weight + extra_weight, 2)
            }
        }

    def _generate_nomenclature(self, beams: List[Dict], node_lookup: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        decks: List[Dict] = []
        access_decks: List[Dict] = []
        couplers: List[Dict] = []

        horizontal_beams = [b for b in beams if b.get("type") in {"horizontal_x", "horizontal_y"}]
        for idx, beam in enumerate(horizontal_beams):
            start_node = node_lookup.get(beam["start"])
            end_node = node_lookup.get(beam["end"])
            if not start_node or not end_node:
                continue
            if min(start_node["z"], end_node["z"]) < 0.5:
                continue

            span = self._beam_length(start_node, end_node)
            layher_size = self._closest_standard(span, self.LAYHER_DECK_WEIGHT_KG)
            deck_item = {
                "id": f"deck_{beam['id']}",
                "beam_id": beam["id"],
                "article": f"SteelDeck-{layher_size:.2f}m",
                "length_m": layher_size,
                "level_z": round((start_node["z"] + end_node["z"]) / 2.0, 3),
            }
            decks.append(deck_item)

            if idx % 6 == 0:
                access_decks.append(
                    {
                        "id": f"access_{beam['id']}",
                        "beam_id": beam["id"],
                        "article": f"AccessDeck-{layher_size:.2f}m",
                        "length_m": layher_size,
                        "level_z": deck_item["level_z"],
                    }
                )

        for beam in beams:
            start_node = node_lookup.get(beam["start"])
            end_node = node_lookup.get(beam["end"])
            if not start_node or not end_node:
                continue
            length = self._beam_length(start_node, end_node)
            closest = min(self.LAYHER_LEDGER_STANDARDS, key=lambda v: abs(v - length))
            if beam.get("type") == "diagonal" or abs(closest - length) > 0.05:
                couplers.append(
                    {
                        "id": f"coupler_{beam['id']}",
                        "beam_id": beam["id"],
                        "article": "SwivelCoupler",
                        "weight_kg": self.LAYHER_COUPLER_WEIGHT_KG,
                    }
                )

        return {"decks": decks, "access_decks": access_decks, "couplers": couplers}

    def _beam_length(self, start_node: Dict, end_node: Dict) -> float:
        return float(
            np.sqrt(
                (float(start_node["x"]) - float(end_node["x"])) ** 2
                + (float(start_node["y"]) - float(end_node["y"])) ** 2
                + (float(start_node["z"]) - float(end_node["z"])) ** 2
            )
        )

    def _closest_standard(self, value: float, reference: Dict[float, float]) -> float:
        return min(reference.keys(), key=lambda candidate: abs(candidate - value))

    def _calculate_beam_weight(self, beams: List[Dict], node_lookup: Dict[str, Dict]) -> float:
        total = 0.0
        for beam in beams:
            start_node = node_lookup.get(beam["start"])
            end_node = node_lookup.get(beam["end"])
            if not start_node or not end_node:
                continue
            length = self._beam_length(start_node, end_node)
            if beam.get("type") == "vertical":
                total += self.LAYHER_STAND_WEIGHT_KG[self._closest_standard(length, self.LAYHER_STAND_WEIGHT_KG)]
            else:
                total += self.LAYHER_LEDGER_WEIGHT_KG[self._closest_standard(length, self.LAYHER_LEDGER_WEIGHT_KG)]
        return total

    def _calculate_nomenclature_weight(self, nomenclature: Dict[str, List[Dict]]) -> float:
        deck_weight = sum(self.LAYHER_DECK_WEIGHT_KG[self._closest_standard(d["length_m"], self.LAYHER_DECK_WEIGHT_KG)] for d in nomenclature["decks"])
        access_weight = sum(self.LAYHER_ACCESS_DECK_WEIGHT_KG[self._closest_standard(d["length_m"], self.LAYHER_ACCESS_DECK_WEIGHT_KG)] for d in nomenclature["access_decks"])
        coupler_weight = len(nomenclature["couplers"]) * self.LAYHER_COUPLER_WEIGHT_KG
        return deck_weight + access_weight + coupler_weight
    

    def generate_smart_options(self, user_points: List[Dict], ai_points: List[Dict], bounds: Dict) -> List[Dict]:
        """Генерирует 3 стратегии на основе пользовательских и AI-опор."""
        normalized_user = [self._normalize_anchor(p, default_type="USER_ANCHOR", default_weight=1.0) for p in user_points]
        normalized_ai = [self._normalize_anchor(p, default_type=p.get("type", "AI_BEAM"), default_weight=self._default_weight(p.get("type"))) for p in ai_points]

        width = float(bounds.get("w", 2.0))
        height = float(bounds.get("h", 2.0))
        depth = float(bounds.get("d", 1.0))

        variants = []

        manual_anchors = normalized_user or normalized_ai
        manual = self._create_variant(width, height, depth, stand_len=2.0, ledger_len=1.57, label="По вашим отметкам")
        manual["strategy"] = "MANUAL"
        manual["anchors"] = manual_anchors
        manual["support_summary"] = self._support_summary(manual_anchors)
        variants.append(manual)

        hybrid_anchors = self._add_floor_supports(normalized_user, normalized_ai)
        hybrid = self._create_variant(width, height, depth, stand_len=2.5, ledger_len=2.07, label="Безопасный гибрид")
        hybrid["strategy"] = "HYBRID"
        hybrid["anchors"] = hybrid_anchors
        hybrid["support_summary"] = self._support_summary(hybrid_anchors)
        variants.append(hybrid)

        efficiency = self._create_variant(width, height, depth, stand_len=3.0, ledger_len=2.57, label="Экономия (Склад)")
        efficiency["strategy"] = "EFFICIENCY"
        efficiency["anchors"] = hybrid_anchors
        efficiency["support_summary"] = self._support_summary(hybrid_anchors)
        variants.append(efficiency)

        return variants

    def _add_floor_supports(self, user_points: List[Dict], ai_points: List[Dict]) -> List[Dict]:
        """Добавляет опоры пола под пользовательскими точками и учитывает найденный AI пол."""
        final = list(user_points)
        floor_points = [p for p in ai_points if p.get("type") in {"AI_FLOOR", "FLOOR"}]

        for up in user_points:
            if up.get("z", 0.0) <= 0.2:
                continue
            if not any(self._is_under(fp, up) for fp in floor_points):
                floor_points.append({
                    "x": up["x"],
                    "y": up["y"],
                    "z": 0.0,
                    "type": "AI_FLOOR",
                    "weight": 0.9,
                    "source": "ai",
                })

        final.extend(floor_points)

        seen = set()
        deduped = []
        for p in final:
            key = (round(float(p.get("x", 0.0)), 2), round(float(p.get("y", 0.0)), 2), round(float(p.get("z", 0.0)), 2), p.get("type"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        return deduped

    def _normalize_anchor(self, point: Dict, default_type: str, default_weight: float) -> Dict:
        return {
            "x": float(point.get("x", 0.0)),
            "y": float(point.get("y", 0.0)),
            "z": float(point.get("z", 0.0)),
            "type": point.get("type", default_type),
            "weight": float(point.get("weight", default_weight)),
            "source": point.get("source", "user" if default_type.startswith("USER") else "ai"),
        }

    def _default_weight(self, support_type: str) -> float:
        if support_type in {"AI_FLOOR", "FLOOR"}:
            return 0.9
        if support_type in {"AI_BEAM", "BEAM"}:
            return 0.8
        return 0.7

    def _is_under(self, low: Dict, high: Dict) -> bool:
        if float(low.get("z", 0.0)) > float(high.get("z", 0.0)):
            return False
        dist_xy = np.sqrt((float(low.get("x", 0.0)) - float(high.get("x", 0.0))) ** 2 + (float(low.get("y", 0.0)) - float(high.get("y", 0.0))) ** 2)
        return dist_xy <= 0.75

    def _support_summary(self, anchors: List[Dict]) -> Dict:
        if not anchors:
            return {"count": 0, "avg_weight": 0.0}
        weights = [float(a.get("weight", 0.0)) for a in anchors]
        return {"count": len(anchors), "avg_weight": round(float(np.mean(weights)), 3)}

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
