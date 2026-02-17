"""AutoScaffolder — генеративный сборщик лесов от целевой зоны."""
from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional, Tuple

try:
    from modules.voxel_world import VoxelWorld
    from modules.astar_pathfinder import ScaffoldPathfinder
    from core.layher_standards import snap_to_layher_grid
except ImportError:
    from voxel_world import VoxelWorld
    from astar_pathfinder import ScaffoldPathfinder
    from core.layher_standards import snap_to_layher_grid


class AutoScaffolder:
    MIN_PLATFORM_WIDTH = 1.09
    SAFETY_DECK_MARGIN = 0.5

    def __init__(self, voxel_world: VoxelWorld, ledger_len: float = 1.09, standard_h: float = 2.07):
        self.world = voxel_world
        self.ledger_len = snap_to_layher_grid(ledger_len, "ledger")
        self.standard_h = snap_to_layher_grid(standard_h, "standard")
        self.pathfinder = ScaffoldPathfinder(voxel_world, step_h=self.ledger_len, step_v=self.standard_h)

    def build_to_target(
        self,
        target: Dict,
        clearance_box: Optional[Dict] = None,
        floor_z: float = 0.0,
    ) -> Dict:
        if clearance_box is None:
            clearance_box = {"width": self.ledger_len, "depth": self.ledger_len}

        detected_floor = self.world.get_floor_z(target["x"], target["y"])
        actual_floor = detected_floor if detected_floor is not None else floor_z

        tower_height = target["z"] - actual_floor
        if tower_height <= 0:
            raise ValueError(f"Цель ниже пола: target.z={target['z']}, floor_z={actual_floor}")

        num_floors = math.ceil(tower_height / self.standard_h)

        w = snap_to_layher_grid(max(clearance_box["width"], self.MIN_PLATFORM_WIDTH), "ledger")
        d = snap_to_layher_grid(max(clearance_box["depth"], self.MIN_PLATFORM_WIDTH), "ledger")

        half_w, half_d = w / 2, d / 2
        cx, cy = target["x"], target["y"]
        foot_positions = [
            (cx - half_w, cy - half_d),
            (cx + half_w, cy - half_d),
            (cx + half_w, cy + half_d),
            (cx - half_w, cy + half_d),
        ]

        nodes: List[Dict] = []
        beams: List[Dict] = []
        node_map: Dict[Tuple, str] = {}

        for floor_idx in range(num_floors + 1):
            z = min(actual_floor + floor_idx * self.standard_h, target["z"])
            for col_idx, (fx, fy) in enumerate(foot_positions):
                nid = f"n_{col_idx}_{floor_idx}"
                node = {"id": nid, "x": round(fx, 4), "y": round(fy, 4), "z": round(z, 4), "is_fixed": floor_idx == 0}
                nodes.append(node)
                node_map[(col_idx, floor_idx)] = nid

        for col_idx in range(4):
            for floor_idx in range(num_floors):
                start = node_map[(col_idx, floor_idx)]
                end = node_map[(col_idx, floor_idx + 1)]
                sn = next(n for n in nodes if n["id"] == start)
                en = next(n for n in nodes if n["id"] == end)
                path = self.pathfinder.find_path(sn, en)
                blocked = any(p.get("_blocked") for p in path)
                beams.append({"id": f"std_{col_idx}_{floor_idx}", "type": "standard", "start": start, "end": end, "length": round(self.standard_h, 4), "blocked": blocked})

        ledger_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        transom_pairs = [(0, 3), (1, 2)]

        for floor_idx in range(num_floors + 1):
            for ci, cj in ledger_pairs:
                start = node_map[(ci, floor_idx)]
                end = node_map[(cj, floor_idx)]
                sn = next(n for n in nodes if n["id"] == start)
                en = next(n for n in nodes if n["id"] == end)
                length = math.dist([sn["x"], sn["y"]], [en["x"], en["y"]])
                path = self.pathfinder.find_path(sn, en)
                blocked = any(p.get("_blocked") for p in path)
                if blocked and len(path) > 2:
                    nodes, beams = self._insert_detour(nodes, beams, path, start, end)
                else:
                    beams.append({"id": f"led_{ci}_{cj}_{floor_idx}", "type": "ledger", "start": start, "end": end, "length": round(length, 4), "blocked": blocked})

            for ci, cj in transom_pairs:
                start = node_map[(ci, floor_idx)]
                end = node_map[(cj, floor_idx)]
                sn = next(n for n in nodes if n["id"] == start)
                en = next(n for n in nodes if n["id"] == end)
                beams.append({"id": f"trans_{ci}_{cj}_{floor_idx}", "type": "transom", "start": start, "end": end, "length": round(math.dist([sn['x'], sn['y'], sn['z']], [en['x'], en['y'], en['z']]), 4)})

        for floor_idx in range(0, num_floors, 2):
            next_floor = min(floor_idx + 1, num_floors)
            for ci, cj in [(0, 1), (2, 3)]:
                start_id = node_map[(ci, floor_idx)]
                end_id = node_map[(cj, next_floor)]
                sn = next(n for n in nodes if n["id"] == start_id)
                en = next(n for n in nodes if n["id"] == end_id)
                beams.append({"id": f"diag_{ci}_{floor_idx}", "type": "diagonal", "start": start_id, "end": end_id, "length": round(math.dist([sn['x'], sn['y'], sn['z']], [en['x'], en['y'], en['z']]), 4)})

        return {
            "nodes": nodes,
            "beams": beams,
            "label": f"AutoScaffold → цель ({target['x']:.1f}, {target['y']:.1f}, {target['z']:.1f})",
            "target": target,
            "floors": num_floors,
            "floor_z": actual_floor,
        }

    def _insert_detour(self, nodes: List[Dict], beams: List[Dict], path: List[Dict], start_id: str, end_id: str) -> Tuple[List[Dict], List[Dict]]:
        prev_id = start_id
        for i, wp in enumerate(path[1:], start=1):
            if i == len(path) - 1:
                next_id = end_id
            else:
                next_id = f"detour_{uuid.uuid4().hex[:6]}"
                nodes.append({"id": next_id, "x": wp["x"], "y": wp["y"], "z": wp["z"]})

            beams.append(
                {
                    "id": f"detour_beam_{uuid.uuid4().hex[:6]}",
                    "type": "ledger",
                    "start": prev_id,
                    "end": next_id,
                    "length": round(
                        math.dist(
                            [wp["x"], wp["y"], wp["z"]],
                            [path[i - 1]["x"], path[i - 1]["y"], path[i - 1]["z"]],
                        ),
                        4,
                    ),
                }
            )
            prev_id = next_id
        return nodes, beams
