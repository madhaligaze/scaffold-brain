# modules/object_tracker.py
"""
Track & fuse 3D detections into stable WorldObjects (Stage 2)
------------------------------------------------------------

Minimal implementation:
- Data association by class + distance + (optional) 3D IoU (AABB).
- EMA smoothing for position/dimensions.
- CONFIRMED after N observations.

This is intentionally conservative and keeps UNKNOWN-space in mind by allowing
external confidence penalties.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


def aabb_iou(center_a: np.ndarray, dims_a: np.ndarray, center_b: np.ndarray, dims_b: np.ndarray) -> float:
    # Axis-aligned boxes. dims are full extents.
    min_a = center_a - dims_a / 2.0
    max_a = center_a + dims_a / 2.0
    min_b = center_b - dims_b / 2.0
    max_b = center_b + dims_b / 2.0

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter))
    if inter_vol <= 0:
        return 0.0
    vol_a = float(np.prod(np.maximum(1e-6, dims_a)))
    vol_b = float(np.prod(np.maximum(1e-6, dims_b)))
    return inter_vol / (vol_a + vol_b - inter_vol + 1e-9)


@dataclass
class TrackerConfig:
    max_assoc_dist_m: float = 0.6
    min_iou: float = 0.05
    ema_alpha: float = 0.35
    confirm_after: int = 5


class ObjectTracker:
    def __init__(self, cfg: Optional[TrackerConfig] = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.objects: List[Dict[str, Any]] = []

    def seed(self, existing: List[Dict[str, Any]]) -> None:
        self.objects = list(existing or [])

    def _match(self, det: Dict[str, Any]) -> Optional[int]:
        cls = det.get("class_label", "unknown")
        c = np.array(det.get("position_world", [0, 0, 0]), dtype=np.float64)
        d = np.array(det.get("dimensions_world", [0.1, 0.1, 0.1]), dtype=np.float64)

        best_i = None
        best_score = -1.0
        for i, obj in enumerate(self.objects):
            if obj.get("class_label") != cls:
                continue
            oc = np.array(obj.get("pose", {}).get("position", [0, 0, 0]), dtype=np.float64)
            od = self._dims_from_obj(obj)

            dist = float(np.linalg.norm(c - oc))
            if dist > self.cfg.max_assoc_dist_m:
                continue
            iou = aabb_iou(c, d, oc, od)
            score = (1.0 - dist / self.cfg.max_assoc_dist_m) + 2.0 * iou
            if iou < self.cfg.min_iou and dist > (self.cfg.max_assoc_dist_m * 0.5):
                continue
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def _dims_from_obj(self, obj: Dict[str, Any]) -> np.ndarray:
        dims = obj.get("dimensions", {})
        if "dx" in dims:
            return np.array([dims.get("dx", 0.1), dims.get("dy", 0.1), dims.get("dz", 0.1)], dtype=np.float64)
        # cylinder bounding box approx
        r = float(dims.get("radius", 0.05))
        L = float(dims.get("length", 0.2))
        return np.array([2*r, 2*r, L], dtype=np.float64)

    def update(self, dets3d: List[Dict[str, Any]], frame_id: str) -> List[Dict[str, Any]]:
        for det in dets3d:
            mi = self._match(det)
            if mi is None:
                obj = {
                    "id": str(uuid4()),
                    "class_label": det.get("class_label", "unknown"),
                    "geometry_type": "MESH_PROXY",
                    "pose": {"position": det.get("position_world", [0, 0, 0]), "orientation": det.get("orientation_world")},
                    "dimensions": {"dx": det.get("dimensions_world", [0.1, 0.1, 0.1])[0],
                                   "dy": det.get("dimensions_world", [0.1, 0.1, 0.1])[1],
                                   "dz": det.get("dimensions_world", [0.1, 0.1, 0.1])[2]},
                    "confidence": float(det.get("score", 0.0)),
                    "observed_frames": [frame_id],
                    "status": "TENTATIVE",
                }
                self.objects.append(obj)
                continue

            obj = self.objects[mi]
            alpha = self.cfg.ema_alpha
            # Update pose
            old_p = np.array(obj.get("pose", {}).get("position", [0, 0, 0]), dtype=np.float64)
            new_p = np.array(det.get("position_world", old_p.tolist()), dtype=np.float64)
            p = (1 - alpha) * old_p + alpha * new_p
            obj.setdefault("pose", {})["position"] = p.tolist()

            # Update dims
            old_d = self._dims_from_obj(obj)
            new_d = np.array(det.get("dimensions_world", old_d.tolist()), dtype=np.float64)
            d = (1 - alpha) * old_d + alpha * new_d
            obj.setdefault("dimensions", {})["dx"] = float(d[0])
            obj.setdefault("dimensions", {})["dy"] = float(d[1])
            obj.setdefault("dimensions", {})["dz"] = float(d[2])

            # Confidence fusion (conservative)
            score = float(det.get("score", 0.0))
            obj["confidence"] = float(max(obj.get("confidence", 0.0), score * 0.9))

            # Frames & status
            frs = obj.setdefault("observed_frames", [])
            if frame_id not in frs:
                frs.append(frame_id)
            if len(frs) >= self.cfg.confirm_after:
                obj["status"] = "CONFIRMED"

        return self.objects
