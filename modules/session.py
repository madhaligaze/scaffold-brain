"""Session management for incremental AR-assisted measuring workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from modules.vision import VisionSystem


@dataclass
class DesignSession:
    """Accumulates keyframes, user anchors and AI supports during measuring."""

    session_id: str
    vision_system: VisionSystem
    status: str = "MEASURING"
    keyframes: List[Dict[str, Any]] = field(default_factory=list)
    user_anchors: List[Dict[str, Any]] = field(default_factory=list)
    detected_supports: List[Dict[str, Any]] = field(default_factory=list)

    def update_world_model(
        self,
        image_bytes: bytes,
        pose_matrix: List[float],
        markers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Processes incoming stream packet and updates world model state."""
        analysis = self.vision_system.process_scene(
            image_bytes=image_bytes,
            distance=self._estimate_distance(pose_matrix),
            ar_points=markers,
        )

        if analysis["ready_for_design"]:
            self.keyframes.append(
                {
                    "pose": pose_matrix,
                    "objects": analysis["objects"],
                }
            )

        for marker in markers or []:
            normalized = {
                "x": float(marker.get("x", 0.0)),
                "y": float(marker.get("y", 0.0)),
                "z": float(marker.get("z", 0.0)),
                "type": marker.get("type", "USER_ANCHOR"),
                "weight": 1.0,
                "source": "user",
            }
            self.user_anchors.append(normalized)

        self.detected_supports.extend(self._extract_ai_supports(analysis.get("objects", [])))

        quality = analysis.get("quality", {})
        return {
            "instructions": quality.get("instructions", []),
            "warnings": quality.get("warnings", []),
            "ready_for_design": analysis.get("ready_for_design", False),
            "keyframes_collected": len(self.keyframes),
        }

    def get_bounds(self) -> Dict[str, float]:
        """Returns approximate workspace bounds from known anchors/supports."""
        points = [
            [p["x"], p["y"], p["z"]]
            for p in (self.user_anchors + self.detected_supports)
            if {"x", "y", "z"}.issubset(p.keys())
        ]
        if not points:
            return {"w": 2.0, "h": 2.0, "d": 1.0}

        arr = np.array(points, dtype=float)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        return {
            "w": float(max(0.5, maxs[0] - mins[0])),
            "d": float(max(0.5, maxs[1] - mins[1])),
            "h": float(max(0.5, maxs[2] - mins[2])),
        }

    def _estimate_distance(self, pose_matrix: List[float]) -> float:
        """Extracts rough camera distance from pose matrix translation component."""
        if len(pose_matrix) >= 16:
            tx, ty, tz = pose_matrix[12], pose_matrix[13], pose_matrix[14]
            distance = float(np.sqrt(tx * tx + ty * ty + tz * tz))
            return max(0.5, distance)
        return 3.0

    def _extract_ai_supports(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Builds support candidates from detected vision objects."""
        supports: List[Dict[str, Any]] = []
        for obj in objects:
            obj_type = obj.get("type")
            if obj_type not in {"beam", "column", "floor_slab"}:
                continue

            center = obj.get("center", [0.0, 0.0])
            x = float(center[0]) / 100.0
            y = float(center[1]) / 100.0
            support_type = "AI_FLOOR" if obj_type == "floor_slab" else "AI_BEAM"
            weight = 0.9 if support_type == "AI_FLOOR" else 0.8
            supports.append({"x": x, "y": y, "z": 0.0, "type": support_type, "weight": weight, "source": "ai"})
        return supports

