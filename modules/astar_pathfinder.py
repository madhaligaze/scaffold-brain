"""A* Pathfinder для строительных лесов."""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

try:
    from modules.voxel_world import VoxelWorld
except ImportError:
    from voxel_world import VoxelWorld

LAYHER_STEP_X = [0.73, 1.09, 1.40, 1.57, 2.07, 2.57, 3.07]
LAYHER_STEP_Z = [1.00, 2.00, 2.07, 2.57, 3.07]
DEFAULT_STEP = 1.09


class AStarNode:
    __slots__ = ("coord", "g", "h", "f", "parent")

    def __init__(self, coord: Tuple[int, int, int], g: float, h: float, parent: Optional["AStarNode"] = None):
        self.coord = coord
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other: "AStarNode") -> bool:
        return self.f < other.f


class ScaffoldPathfinder:
    def __init__(
        self,
        voxel_world: VoxelWorld,
        step_h: float = DEFAULT_STEP,
        step_v: float = 2.07,
        max_detour: float = 6.0,
    ):
        self.world = voxel_world
        self.step_h = self._snap_layher_h(step_h)
        self.step_v = self._snap_layher_v(step_v)
        self.max_iter = int((max_detour / min(step_h, step_v)) ** 3)

    def find_path(self, start: Dict, end: Dict) -> List[Dict]:
        if not self.world.is_blocked(start, end):
            return [start, end]

        gs = self._world_to_grid(start)
        ge = self._world_to_grid(end)

        path_coords = self._astar(gs, ge)
        if path_coords is None:
            return [dict(**start, _blocked=True), dict(**end, _blocked=True)]

        return [self._grid_to_world(c) for c in path_coords]

    def is_direct_possible(self, start: Dict, end: Dict) -> bool:
        return not self.world.is_blocked(start, end)

    def _astar(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        open_heap: List[AStarNode] = []
        open_set: Dict[Tuple[int, int, int], float] = {}
        closed: set = set()

        start_node = AStarNode(start, 0.0, self._heuristic(start, goal))
        heapq.heappush(open_heap, start_node)
        open_set[start] = 0.0

        iterations = 0
        while open_heap and iterations < self.max_iter:
            iterations += 1
            current = heapq.heappop(open_heap)

            if current.coord == goal:
                return self._reconstruct(current)

            if current.coord in closed:
                continue
            closed.add(current.coord)

            for neighbor_coord, cost in self._get_neighbors(current.coord):
                if neighbor_coord in closed:
                    continue

                tentative_g = current.g + cost
                if tentative_g >= open_set.get(neighbor_coord, float("inf")):
                    continue

                node = AStarNode(neighbor_coord, tentative_g, self._heuristic(neighbor_coord, goal), parent=current)
                heapq.heappush(open_heap, node)
                open_set[neighbor_coord] = tentative_g

        return None

    def _get_neighbors(self, coord: Tuple[int, int, int]):
        cx, cy, cz = coord
        candidates = [
            ((cx + 1, cy, cz), self.step_h),
            ((cx - 1, cy, cz), self.step_h),
            ((cx, cy + 1, cz), self.step_h),
            ((cx, cy - 1, cz), self.step_h),
            ((cx, cy, cz + 1), self.step_v * 1.5),
            ((cx, cy, cz - 1), self.step_v * 1.5),
        ]
        result = []
        for nc, cost in candidates:
            wp = self._grid_to_world(nc)
            wc = self._grid_to_world(coord)
            if not self.world.is_blocked(wc, wp):
                result.append((nc, cost))
        return result

    @staticmethod
    def _heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def _reconstruct(node: AStarNode) -> List[Tuple[int, int, int]]:
        path = []
        current: Optional[AStarNode] = node
        while current:
            path.append(current.coord)
            current = current.parent
        return list(reversed(path))

    def _world_to_grid(self, p: Dict) -> Tuple[int, int, int]:
        return (
            round(p["x"] / self.step_h),
            round(p["y"] / self.step_h),
            round(p["z"] / self.step_v),
        )

    def _grid_to_world(self, c: Tuple[int, int, int]) -> Dict:
        return {"x": c[0] * self.step_h, "y": c[1] * self.step_h, "z": c[2] * self.step_v}

    @staticmethod
    def _snap_layher_h(val: float) -> float:
        return min(LAYHER_STEP_X, key=lambda x: abs(x - val))

    @staticmethod
    def _snap_layher_v(val: float) -> float:
        return min(LAYHER_STEP_Z, key=lambda x: abs(x - val))
