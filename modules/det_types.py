# modules/det_types.py
"""
Unified detection / world object formats (Stage 2)
--------------------------------------------------

Det2D: pure image-space detection.
Det3D: lifted detection in world space for a given frame.
WorldObject: fused, stable object in the WorldModel.

All structures are dict-friendly to keep session save/load simple.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Det2D:
    class_label: str
    bbox_xyxy: List[float]  # [x1,y1,x2,y2] in pixels
    score: float
    mask_rle: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_label": self.class_label,
            "bbox_xyxy": self.bbox_xyxy,
            "score": self.score,
            "mask_rle": self.mask_rle,
        }


@dataclass
class Det3D:
    class_label: str
    position_world: List[float]  # [x,y,z]
    dimensions_world: List[float]  # [dx,dy,dz] (axis-aligned unless orientation provided)
    score: float
    frame_id: str
    orientation_world: Optional[List[float]] = None  # [qx,qy,qz,qw]
    points_world: Optional[List[List[float]]] = None  # optional support points for fitting

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_label": self.class_label,
            "position_world": self.position_world,
            "dimensions_world": self.dimensions_world,
            "orientation_world": self.orientation_world,
            "score": self.score,
            "frame_id": self.frame_id,
            "points_world": self.points_world,
        }


@dataclass
class WorldObject:
    id: str
    class_label: str
    geometry_type: str  # CYLINDER / BOX / PLANE / MESH_PROXY
    pose: Dict[str, Any]  # {"position":[x,y,z], "orientation":[qx,qy,qz,qw] optional}
    dimensions: Dict[str, Any]  # {"dx":..,"dy":..,"dz":..} or {"radius":..,"length":..}
    confidence: float
    observed_frames: List[str] = field(default_factory=list)
    status: str = "TENTATIVE"  # CONFIRMED / TENTATIVE / HYPOTHESIS

    # Stage 3: engineering reconstruction (not fantasy)
    # Visible part in parametric space (for cylinders/boxes): usually {"t0":..,"t1":..,"p0":[...],"p1":[...]}
    observable_segment: Optional[Dict[str, Any]] = None
    # Candidate extensions beyond visible segment. Each item holds stop_reason and confidence.
    extension_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    # Evidence that an end really terminates (cap/flange/abrupt density drop)
    termination_evidence: Dict[str, Any] = field(default_factory=dict)
    # If any hypothesis hits UNKNOWN: planning should be conservative and request re-scan.
    needs_scan: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_label": self.class_label,
            "geometry_type": self.geometry_type,
            "pose": self.pose,
            "dimensions": self.dimensions,
            "confidence": self.confidence,
            "observed_frames": list(self.observed_frames),
            "status": self.status,
            "observable_segment": self.observable_segment,
            "extension_hypotheses": list(self.extension_hypotheses),
            "termination_evidence": dict(self.termination_evidence or {}),
            "needs_scan": bool(self.needs_scan),
        }
