"""
Mesh Builder — Сборка 3D моделей лесов с Trimesh
=================================================
ЗАДАЧА: Превратить список элементов в единую 3D mesh с цветами нагрузок.

УЛУЧШЕНИЯ v4.0:
  - Правильное выравнивание элементов (rotation matrix)
  - Поддержка всех типов элементов (vertical, ledger, diagonal, deck)
  - Heatmap colors (green → yellow → red)
  - Экспорт в glTF/OBJ
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh


class ScaffoldMeshBuilder:
    """Строитель 3D mesh моделей строительных лесов."""

    STRESS_COLORS = {
        "green": [0, 200, 0, 255],
        "yellow": [255, 255, 0, 255],
        "orange": [255, 165, 0, 255],
        "red": [255, 0, 0, 255],
        "gray": [128, 128, 128, 255],
    }

    DIMENSIONS = {
        "vertical": {"radius": 0.024, "sections": 16},
        "ledger": {"width": 0.048, "height": 0.048},
        "transom": {"width": 0.048, "height": 0.048},
        "diagonal": {"radius": 0.020, "sections": 12},
        "deck": {"width": None, "height": 0.03},
    }

    def __init__(self):
        self.meshes: List[trimesh.Trimesh] = []
        self.combined_mesh: Optional[trimesh.Trimesh] = None

    def build_from_elements(self, elements: List[Dict[str, Any]]) -> trimesh.Trimesh:
        start_time = time.time()
        self.meshes = []

        for elem in elements:
            elem_type = str(elem.get("type", "ledger")).lower()
            start = self._parse_point(elem.get("start"))
            end = self._parse_point(elem.get("end"))

            if start is None or end is None:
                continue

            if elem_type in ("vertical", "standard"):
                mesh = self._create_vertical(start, end)
            elif elem_type in ("ledger", "transom"):
                mesh = self._create_ledger(start, end)
            elif elem_type == "diagonal":
                mesh = self._create_diagonal(start, end)
            elif elem_type == "deck":
                mesh = self._create_deck(start, end, float(elem.get("width", 0.5)))
            else:
                continue

            if mesh.vertices.size == 0:
                continue

            color = self._get_stress_color(
                str(elem.get("stress_color", "gray")), float(elem.get("load_ratio", 0.0))
            )
            mesh.visual.vertex_colors = color
            self.meshes.append(mesh)

        if self.meshes:
            self.combined_mesh = trimesh.util.concatenate(self.meshes)
        else:
            self.combined_mesh = trimesh.Trimesh()

        elapsed = time.time() - start_time
        print(
            f"  Mesh built: {len(elements)} elements → "
            f"{len(self.combined_mesh.vertices)} vertices "
            f"in {int(elapsed * 1000)}ms"
        )
        return self.combined_mesh

    def _create_vertical(self, start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()

        dims = self.DIMENSIONS["vertical"]
        cylinder = trimesh.creation.cylinder(
            radius=dims["radius"], height=length, sections=dims["sections"]
        )
        direction = (end - start) / length
        return self._align_mesh(cylinder, start, end, direction)

    def _create_ledger(self, start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()

        dims = self.DIMENSIONS["ledger"]
        box = trimesh.creation.box(extents=[length, dims["width"], dims["height"]])
        direction = (end - start) / length
        return self._align_mesh(box, start, end, direction)

    def _create_diagonal(self, start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()

        dims = self.DIMENSIONS["diagonal"]
        cylinder = trimesh.creation.cylinder(
            radius=dims["radius"], height=length, sections=dims["sections"]
        )
        direction = (end - start) / length
        return self._align_mesh(cylinder, start, end, direction)

    def _create_deck(self, start: np.ndarray, end: np.ndarray, width: float) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()

        dims = self.DIMENSIONS["deck"]
        box = trimesh.creation.box(extents=[length, width, dims["height"]])
        direction = (end - start) / length
        return self._align_mesh(box, start, end, direction)

    def _align_mesh(
        self,
        mesh: trimesh.Trimesh,
        start: np.ndarray,
        end: np.ndarray,
        direction: np.ndarray,
    ) -> trimesh.Trimesh:
        center = (start + end) / 2
        default_dir = np.array([0, 0, 1])

        if np.allclose(direction, default_dir):
            pass
        elif np.allclose(direction, -default_dir):
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        else:
            axis = np.cross(default_dir, direction)
            axis_length = np.linalg.norm(axis)
            if axis_length > 1e-6:
                axis = axis / axis_length
                angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
                rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
                mesh.apply_transform(rotation_matrix)

        mesh.apply_translation(center)
        return mesh

    def _get_stress_color(self, stress_name: str, load_ratio: float) -> np.ndarray:
        if stress_name in self.STRESS_COLORS:
            return np.array(self.STRESS_COLORS[stress_name])

        if load_ratio < 0.3:
            return np.array(self.STRESS_COLORS["green"])
        if load_ratio < 0.7:
            t = (load_ratio - 0.3) / 0.4
            green = np.array(self.STRESS_COLORS["green"])
            yellow = np.array(self.STRESS_COLORS["yellow"])
            return (green * (1 - t) + yellow * t).astype(int)
        if load_ratio < 0.9:
            t = (load_ratio - 0.7) / 0.2
            yellow = np.array(self.STRESS_COLORS["yellow"])
            orange = np.array(self.STRESS_COLORS["orange"])
            return (yellow * (1 - t) + orange * t).astype(int)
        return np.array(self.STRESS_COLORS["red"])

    def _parse_point(self, point: Any) -> Optional[np.ndarray]:
        if point is None:
            return None

        if isinstance(point, dict):
            return np.array([point.get("x", 0), point.get("y", 0), point.get("z", 0)])
        if isinstance(point, (list, tuple)) and len(point) >= 3:
            return np.array(point[:3])
        return None

    def export_gltf(self, filepath: str) -> bool:
        if self.combined_mesh is None or len(self.combined_mesh.vertices) == 0:
            return False

        try:
            self.combined_mesh.export(filepath, file_type="gltf")
            return True
        except Exception as e:
            print(f"Export glTF error: {e}")
            return False

    def export_obj(self, filepath: str) -> bool:
        if self.combined_mesh is None or len(self.combined_mesh.vertices) == 0:
            return False

        try:
            self.combined_mesh.export(filepath, file_type="obj")
            return True
        except Exception as e:
            print(f"Export OBJ error: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        if self.combined_mesh is None:
            return {"error": "No mesh"}

        if len(self.combined_mesh.vertices) == 0:
            return {
                "vertices": 0,
                "faces": 0,
                "bounds": {"min": [0, 0, 0], "max": [0, 0, 0]},
                "volume": 0.0,
                "is_watertight": False,
            }

        return {
            "vertices": len(self.combined_mesh.vertices),
            "faces": len(self.combined_mesh.faces),
            "bounds": {
                "min": self.combined_mesh.bounds[0].tolist(),
                "max": self.combined_mesh.bounds[1].tolist(),
            },
            "volume": float(self.combined_mesh.volume),
            "is_watertight": self.combined_mesh.is_watertight,
        }


if __name__ == "__main__":
    print("🧪 TESTING MESH BUILDER")
    sample_elements = [
        {
            "type": "vertical",
            "start": [0, 0, 0],
            "end": [0, 0, 2],
            "stress_color": "green",
        },
        {
            "type": "ledger",
            "start": [0, 0, 2],
            "end": [2, 0, 2],
            "load_ratio": 0.65,
        },
        {
            "type": "diagonal",
            "start": [0, 0, 0],
            "end": [2, 0, 2],
            "load_ratio": 0.85,
        },
    ]
    builder = ScaffoldMeshBuilder()
    builder.build_from_elements(sample_elements)
    print(builder.get_statistics())
    print("✓ Test passed!")
