"""
AStarPathfinder — Поиск пути по стандартам Layher.
===================================================
ИСПРАВЛЕНИЕ v3.1 (Аудит):
  - _get_layher_neighbors: прыгает на LEDGERS=[0.73, 1.09, ...], НЕ на 0.1 м
  - Соседи = реально построимые узлы (длина = стандарт из каталога Layher)
  - Вертикаль: только ВВЕРХ (LIFT_HEIGHTS=[2.00, 2.07, 2.57, 3.07])
  - is_blocked: теперь принимает tuple (x,y,z), а не dict
  - Ограничение итераций: 2000 (сервер не виснет)
  - Возвращает список сегментов с type и length для BOM

ВНИМАНИЕ: Узлы хранятся как tuple (x, y, z) с округлением до 2 знаков.
Это нужно, чтобы (1.09 + 1.09) == (2.18) без float-мусора.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Set, Tuple, Union

# ── Стандарты Layher (горизонталь и вертикаль) ───────────────────────────────
LAYHER_LEDGERS: List[float] = [0.73, 1.09, 1.57, 2.07, 2.57, 3.07]
LAYHER_TRANSOMS: List[float] = [0.73, 1.09]
LAYHER_LIFTS: List[float] = [2.00, 2.07, 2.57, 3.07]

# 4 направления в горизонтальной плоскости (XY)
_DIRS_XY = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class AStarPathfinder:
    """
    A*-планировщик, который строит только реально существующие балки Layher.
    """

    def __init__(
        self,
        voxel_world,
        ledger_lengths: List[float] = None,
        lift_heights: List[float] = None,
        max_iterations: int = 2000,
    ):
        self.world = voxel_world
        self.LEDGERS = sorted(ledger_lengths or LAYHER_LEDGERS)
        self.LIFTS = sorted(lift_heights or LAYHER_LIFTS)
        self.MAX_ITER = max_iterations

    def find_path(
        self,
        start: Union[Tuple[float, float, float], Dict[str, float]],
        target: Union[Tuple[float, float, float], Dict[str, float]],
    ) -> List[Dict]:
        """Ищет маршрут от стартовой точки до цели."""
        s = self._snap(self._as_tuple(start))
        g = self._snap(self._as_tuple(target))

        open_heap: List = []
        visited: Set = set()
        came_from: Dict = {}
        g_cost: Dict = {s: 0.0}

        heapq.heappush(open_heap, (self._h(s, g), 0.0, s))

        best_node = s
        best_dist = self._h(s, g)
        iterations = 0

        while open_heap and iterations < self.MAX_ITER:
            iterations += 1
            _, cost, current = heapq.heappop(open_heap)

            if current in visited:
                continue
            visited.add(current)

            dist = self._h(current, g)
            if dist < best_dist:
                best_dist = dist
                best_node = current

            if dist < 0.5:
                return self._reconstruct(came_from, current)

            for nxt, length, btype in self._get_layher_neighbors(current):
                if nxt in visited:
                    continue
                if self.world.is_blocked(current, nxt):
                    continue
                new_cost = cost + length
                if new_cost < g_cost.get(nxt, float("inf")):
                    g_cost[nxt] = new_cost
                    came_from[nxt] = (current, length, btype)
                    f = new_cost + self._h(nxt, g)
                    heapq.heappush(open_heap, (f, new_cost, nxt))

        print(f"⚠️  A* reached limit: iter={iterations}, best_dist={best_dist:.2f}m to target")
        if best_node != s:
            return self._reconstruct(came_from, best_node)
        return []

    def _get_layher_neighbors(self, node: Tuple[float, float, float]) -> List[Tuple[Tuple[float, float, float], float, str]]:
        x, y, z = node
        result = []

        for L in self.LEDGERS:
            for dx, dy in _DIRS_XY:
                nxt = self._snap((x + dx * L, y + dy * L, z))
                btype = "ledger" if dx != 0 else "transom"
                result.append((nxt, L, btype))

        for H in self.LIFTS:
            nxt = self._snap((x, y, z + H))
            result.append((nxt, H, "standard"))

        return result

    def _reconstruct(self, came_from: Dict, end: Tuple[float, float, float]) -> List[Dict]:
        path = []
        current = end

        while current in came_from:
            parent, length, btype = came_from[current]
            path.append({"start": parent, "end": current, "type": btype, "length": round(length, 4)})
            current = parent

        return list(reversed(path))

    @staticmethod
    def _snap(p: Tuple[float, ...]) -> Tuple[float, float, float]:
        return (round(p[0], 2), round(p[1], 2), round(p[2], 2))

    @staticmethod
    def _h(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def _as_tuple(p: Union[Tuple[float, float, float], Dict[str, float]]) -> Tuple[float, float, float]:
        if isinstance(p, dict):
            return (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
        return (float(p[0]), float(p[1]), float(p[2]))


class ScaffoldPathfinder:
    """Backward-compatible wrapper over AStarPathfinder for legacy callers."""

    def __init__(self, voxel_world, step_h: float = 1.09, step_v: float = 2.07, max_detour: float = 6.0):
        # NOTE(v3.1): even legacy wrapper must navigate using real Layher jumps,
        # otherwise AutoScaffolder remains artificially constrained to a single
        # beam length and cannot detour around obstacles in production scenes.
        ledgers = sorted(set(LAYHER_LEDGERS + [step_h]))
        lifts = sorted(set(LAYHER_LIFTS + [step_v]))
        self._astar = AStarPathfinder(
            voxel_world,
            ledger_lengths=ledgers,
            lift_heights=lifts,
            max_iterations=2000,
        )
        self.world = voxel_world

    def is_direct_possible(self, start: Dict, end: Dict) -> bool:
        return not self.world.is_blocked(start, end)

    def find_path(self, start: Dict, end: Dict) -> List[Dict]:
        if not self.world.is_blocked(start, end):
            return [start, end]

        segments = self._astar.find_path(start, end)
        if not segments:
            return [dict(**start, _blocked=True), dict(**end, _blocked=True)]

        points = [dict(start)]
        for seg in segments:
            ex, ey, ez = seg["end"]
            points.append({"x": ex, "y": ey, "z": ez})
        return points
