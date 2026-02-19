# modules/perception_backend.py
"""
Perception backend (Stage 2)
----------------------------

Pipeline per frame:
1) Det2D: run optional 2D detector.
2) Lift: Det2D -> Det3D using intrinsics + depth/point_cloud + pose.
3) Confidence penalty if object lies in UNKNOWN voxels.
4) Track&Fuse: Det3D -> stable world_objects.
5) Primitive fitting (pipes/beams): refine geometry_type + dimensions.

This module is designed to be called from /session/frame.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from modules.detector_2d import Detector2D
from modules.lifter_2d3d import Intrinsics, decode_confidence_bytes, decode_depth_bytes, lift_det2d_to_3d
from modules.object_tracker import ObjectTracker
from modules.primitive_fitting import classify_for_fitting, fit_cylinder, fit_oriented_box
from modules.extension_engine import (
    observable_segment_from_points,
    termination_evidence_cylinder,
    propose_axis_extension,
)

logger = logging.getLogger(__name__)


class PerceptionBackend:
    def __init__(self) -> None:
        self.detector = Detector2D()
        self.tracker = ObjectTracker()

    def seed_from_scene(self, scene_context) -> None:
        # Seed tracker state from stored objects
        existing = getattr(scene_context, "world_objects", None) or scene_context.__dict__.get("world_objects", [])
        if existing:
            self.tracker.seed(existing)

    def process_frame(
        self,
        *,
        frame_id: str,
        rgb_bytes: Optional[bytes],
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        pose7: List[float],
        depth_bytes: Optional[bytes],
        depth_scale: float,
        conf_bytes: Optional[bytes],
        point_cloud_world: Optional[List[List[float]]],
        voxel_world=None,
        enable_vision: bool = True,
    ) -> Dict[str, Any]:
        # Step 1: 2D detections
        det2d_list: List[Dict[str, Any]] = []
        if enable_vision and rgb_bytes is not None:
            det2d_list = self.detector.infer(rgb_bytes)

        # Step 2: decode depth/conf and lift
        intr = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
        depth_m = None
        conf = None
        if depth_bytes is not None:
            try:
                depth_m = decode_depth_bytes(depth_bytes, width, height)
            except Exception as e:
                logger.warning("Depth decode failed: %s", e)
                depth_m = None
        if conf_bytes is not None:
            try:
                conf = decode_confidence_bytes(conf_bytes, width, height)
            except Exception:
                conf = None

        pc_np = None
        if point_cloud_world:
            try:
                pc_np = np.array([p[:3] for p in point_cloud_world], dtype=np.float64)
            except Exception:
                pc_np = None

        det3d_list: List[Dict[str, Any]] = []
        for d in det2d_list:
            det3d = lift_det2d_to_3d(
                d,
                frame_id=frame_id,
                intr=intr,
                pose_world_from_camera=pose7,
                depth_m=depth_m,
                conf=conf,
                depth_scale=depth_scale,
                point_cloud_world=pc_np,
                pixel_step=4,
            )
            if det3d is None:
                continue

            # Step 3: unknown-space penalty (conservative)
            if voxel_world is not None:
                det3d["score"] = float(det3d.get("score", 0.0)) * self._unknown_penalty(det3d, voxel_world)

            det3d_list.append(det3d)

        # Step 4: Track&Fuse
        world_objects = self.tracker.update(det3d_list, frame_id=frame_id)

        # Step 5: Primitive fitting (optional refinement)
        # We use points from the latest det3d if available and class matches.
        refined = []
        scan_suggestions: List[Dict[str, Any]] = []
        for obj in world_objects:
            ref = dict(obj)
            refined.append(ref)
        # Update in-place for objects that were just observed
        observed_ids = set()
        for d in det3d_list:
            # find best match by class+distance
            cid = self._nearest_object_id(d, refined)
            if cid:
                observed_ids.add(cid)
                self._refine_geometry(cid, d, refined)
                self._stage3_reconstruct(cid, d, refined, voxel_world, scan_suggestions)

        self.tracker.objects = refined

        return {
            "det2d": det2d_list,
            "det3d": det3d_list,
            "world_objects": refined,
            "scan_suggestions": scan_suggestions,
            "vision_enabled": bool(enable_vision),
            "detector_available": bool(self.detector.available),
        }

    def _unknown_penalty(self, det3d: Dict[str, Any], voxel_world) -> float:
        pts = det3d.get("points_world")
        if pts:
            # sample up to 200 points
            sample = pts[:: max(1, len(pts) // 200)]
            states = [voxel_world.get_state(p[0], p[1], p[2]) for p in sample]
            unknown = sum(1 for s in states if s == voxel_world.UNKNOWN)
            frac = unknown / max(1, len(states))
        else:
            p = det3d.get("position_world", [0, 0, 0])
            frac = 1.0 if voxel_world.get_state(p[0], p[1], p[2]) == voxel_world.UNKNOWN else 0.0

        # Penalty: up to -50% score at 100% unknown.
        return float(max(0.3, 1.0 - 0.5 * frac))

    def _nearest_object_id(self, det3d: Dict[str, Any], objects: List[Dict[str, Any]]) -> Optional[str]:
        cls = det3d.get("class_label")
        p = np.array(det3d.get("position_world", [0, 0, 0]), dtype=np.float64)
        best = None
        best_d = 1e9
        for o in objects:
            if o.get("class_label") != cls:
                continue
            op = np.array(o.get("pose", {}).get("position", [0, 0, 0]), dtype=np.float64)
            d = float(np.linalg.norm(p - op))
            if d < best_d:
                best_d = d
                best = o.get("id")
        if best_d < 0.8:
            return best
        return None

    def _refine_geometry(self, obj_id: str, det3d: Dict[str, Any], objects: List[Dict[str, Any]]) -> None:
        fit_kind = classify_for_fitting(det3d.get("class_label", ""))
        pts = det3d.get("points_world") or []
        if fit_kind == "NONE" or len(pts) < 50:
            return
        P = np.array(pts, dtype=np.float64)
        fit = None
        if fit_kind == "CYLINDER":
            fit = fit_cylinder(P)
        elif fit_kind == "BOX":
            fit = fit_oriented_box(P)
        if fit is None:
            return

        for o in objects:
            if o.get("id") != obj_id:
                continue
            o["geometry_type"] = fit.get("geometry_type", o.get("geometry_type", "MESH_PROXY"))
            # Store refined dimensions in unified dict
            dims = fit.get("dimensions", {})
            if "radius" in dims:
                o["dimensions"] = {"radius": dims["radius"], "length": dims.get("length", 0.0)}
            else:
                o["dimensions"] = {"dx": dims.get("dx", 0.0), "dy": dims.get("dy", 0.0), "dz": dims.get("dz", 0.0)}
            # Update pose: keep position; for cylinder store axis; for box store orientation_matrix.
            pose = o.setdefault("pose", {})
            fpose = fit.get("pose", {})
            if "position" in fpose:
                pose["position"] = fpose["position"]
            if "axis" in fpose:
                pose["axis"] = fpose["axis"]
            if "orientation_matrix" in fpose:
                pose["orientation_matrix"] = fpose["orientation_matrix"]
            break

    def _stage3_reconstruct(
        self,
        obj_id: str,
        det3d: Dict[str, Any],
        objects: List[Dict[str, Any]],
        voxel_world,
        scan_suggestions: List[Dict[str, Any]],
    ) -> None:
        pts = det3d.get("points_world") or []
        if len(pts) < 30:
            return

        for o in objects:
            if o.get("id") != obj_id:
                continue

            g = (o.get("geometry_type") or "").upper()
            if g not in ("CYLINDER", "BOX"):
                return

            pose = o.get("pose", {}) or {}
            dims = o.get("dimensions", {}) or {}
            origin = np.array(pose.get("position", det3d.get("position_world", [0, 0, 0])), dtype=np.float64)

            if g == "CYLINDER":
                axis = np.array(pose.get("axis", [0, 0, 1]), dtype=np.float64)
            else:
                R = np.array(pose.get("orientation_matrix", np.eye(3).tolist()), dtype=np.float64)
                axis = R[:, 2] if R.shape == (3, 3) else np.array([0, 0, 1], dtype=np.float64)

            axis_n = float(np.linalg.norm(axis))
            if axis_n < 1e-9:
                axis = np.array([0, 0, 1], dtype=np.float64)
            else:
                axis = axis / axis_n

            P = np.array(pts, dtype=np.float64)
            seg = observable_segment_from_points(points_world=P, axis_origin=origin, axis_dir=axis, trim_pct=5.0)
            if not seg:
                return

            o["observable_segment"] = seg
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", 0.0))

            evidence = {}
            if g == "CYLINDER":
                radius = float(dims.get("radius", 0.05))
                nearby = [
                    (x.get("class_label"), (x.get("pose", {}) or {}).get("position", [0, 0, 0]))
                    for x in objects
                    if x.get("id") != obj_id
                ]
                evidence = termination_evidence_cylinder(
                    points_world=P,
                    axis_origin=origin,
                    axis_dir=axis,
                    t0=t0,
                    t1=t1,
                    radius=radius,
                    nearby_labels=nearby,
                )

            hyp, needs_scan, scan_hints = propose_axis_extension(
                obj=o,
                voxel_world=voxel_world,
                axis_origin=origin,
                axis_dir=axis,
                t0=t0,
                t1=t1,
                base_confidence=float(o.get("confidence", det3d.get("score", 0.5))),
                termination_evidence=evidence,
                objects=objects,
            )
            o["termination_evidence"] = evidence
            o["extension_hypotheses"] = hyp
            o["needs_scan"] = bool(needs_scan)
            scan_suggestions.extend(scan_hints)
            return
