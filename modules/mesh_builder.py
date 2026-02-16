"""Mesh Builder — Trimesh scaffold mesh creation."""
from __future__ import annotations

from typing import List, Dict, Optional, Any
import time
import numpy as np
import trimesh


class ScaffoldMeshBuilder:
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
        "deck": {"height": 0.03},
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
            color = self._get_stress_color(str(elem.get("stress_color", "gray")), float(elem.get("load_ratio", 0.0)))
            mesh.visual.vertex_colors = color
            self.meshes.append(mesh)

        self.combined_mesh = trimesh.util.concatenate(self.meshes) if self.meshes else trimesh.Trimesh()
        print(f"Mesh built: {len(elements)} elements -> {len(self.combined_mesh.vertices)} vertices in {int((time.time()-start_time)*1000)}ms")
        return self.combined_mesh

    def _create_vertical(self, start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()
        dims = self.DIMENSIONS["vertical"]
        mesh = trimesh.creation.cylinder(radius=dims["radius"], height=length, sections=dims["sections"])
        return self._align_mesh(mesh, start, end)

    def _create_ledger(self, start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()
        dims = self.DIMENSIONS["ledger"]
        mesh = trimesh.creation.box(extents=[length, dims["width"], dims["height"]])
        return self._align_mesh(mesh, start, end)

    def _create_diagonal(self, start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()
        dims = self.DIMENSIONS["diagonal"]
        mesh = trimesh.creation.cylinder(radius=dims["radius"], height=length, sections=dims["sections"])
        return self._align_mesh(mesh, start, end)

    def _create_deck(self, start: np.ndarray, end: np.ndarray, width: float) -> trimesh.Trimesh:
        length = np.linalg.norm(end - start)
        if length < 0.01:
            return trimesh.Trimesh()
        mesh = trimesh.creation.box(extents=[length, width, self.DIMENSIONS["deck"]["height"]])
        return self._align_mesh(mesh, start, end)

    def _align_mesh(self, mesh: trimesh.Trimesh, start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-9:
            return mesh
        direction = direction / length

        default_dir = np.array([0, 0, 1])
        center = (start + end) / 2

        if np.allclose(direction, -default_dir):
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        elif not np.allclose(direction, default_dir):
            axis = np.cross(default_dir, direction)
            axis_length = np.linalg.norm(axis)
            if axis_length > 1e-6:
                axis /= axis_length
                angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
                mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))

        mesh.apply_translation(center)
        return mesh

    def _get_stress_color(self, stress_name: str, load_ratio: float) -> np.ndarray:
        if stress_name in self.STRESS_COLORS:
            return np.array(self.STRESS_COLORS[stress_name])
        if load_ratio < 0.3:
            return np.array(self.STRESS_COLORS["green"])
        if load_ratio < 0.7:
            t = (load_ratio - 0.3) / 0.4
            return (np.array(self.STRESS_COLORS["green"]) * (1 - t) + np.array(self.STRESS_COLORS["yellow"]) * t).astype(int)
        if load_ratio < 0.9:
            t = (load_ratio - 0.7) / 0.2
            return (np.array(self.STRESS_COLORS["yellow"]) * (1 - t) + np.array(self.STRESS_COLORS["orange"]) * t).astype(int)
        return np.array(self.STRESS_COLORS["red"])

    def _parse_point(self, point: Any) -> Optional[np.ndarray]:
        if point is None:
            return None
        if isinstance(point, dict):
            return np.array([point.get("x", 0), point.get("y", 0), point.get("z", 0)], dtype=float)
        if isinstance(point, (list, tuple)) and len(point) >= 3:
            return np.array(point[:3], dtype=float)
        return None

    def export_gltf(self, filepath: str) -> bool:
        if self.combined_mesh is None or len(self.combined_mesh.vertices) == 0:
            return False
        try:
            self.combined_mesh.export(filepath, file_type="gltf")
            return True
        except Exception:
            return False

    def export_obj(self, filepath: str) -> bool:
        if self.combined_mesh is None or len(self.combined_mesh.vertices) == 0:
            return False
        try:
            self.combined_mesh.export(filepath, file_type="obj")
            return True
        except Exception:
            return False

    def get_statistics(self) -> Dict[str, Any]:
        if self.combined_mesh is None:
            return {"error": "No mesh"}
        return {
            "vertices": len(self.combined_mesh.vertices),
            "faces": len(self.combined_mesh.faces),
            "bounds": {
                "min": self.combined_mesh.bounds[0].tolist() if len(self.combined_mesh.vertices) else [0, 0, 0],
                "max": self.combined_mesh.bounds[1].tolist() if len(self.combined_mesh.vertices) else [0, 0, 0],
            },
            "volume": float(self.combined_mesh.volume) if len(self.combined_mesh.vertices) else 0.0,
            "is_watertight": bool(self.combined_mesh.is_watertight) if len(self.combined_mesh.vertices) else False,
        }
