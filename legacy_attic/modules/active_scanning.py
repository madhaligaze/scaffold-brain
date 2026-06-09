"""Stage 6: Active scanning.

Transforms raw scan suggestions into a small number of actionable view prompts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Cluster:
    id: int
    centroid: np.ndarray
    count: int
    reasons: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "centroid": {"x": float(self.centroid[0]), "y": float(self.centroid[1]), "z": float(self.centroid[2])},
            "count": int(self.count),
            "reasons": dict(self.reasons),
        }


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cluster_suggestions(
    suggestions: List[Dict[str, Any]],
    radius_m: float = 0.6,
    # Keep small clusters: in industrial scenes a single inconsistency point can be valuable.
    # Larger clustering/selection happens later via scoring + max_views.
    min_cluster_size: int = 1,
    max_clusters: int = 12,
) -> List[Cluster]:
    pts: List[np.ndarray] = []
    reasons: List[str] = []
    for s in suggestions:
        try:
            p = s.get("point") if isinstance(s, dict) else None
            if isinstance(p, dict):
                pts.append(np.array([float(p.get("x")), float(p.get("y")), float(p.get("z"))], dtype=np.float32))
            else:
                pts.append(np.array([float(s.get("x")), float(s.get("y")), float(s.get("z"))], dtype=np.float32))
            reasons.append(str(s.get("reason") or "unknown"))
        except Exception:
            continue

    if not pts:
        return []

    assigned = [False] * len(pts)
    clusters: List[Cluster] = []
    cid = 0

    for i, p in enumerate(pts):
        if assigned[i]:
            continue
        members = [i]
        assigned[i] = True
        changed = True
        while changed:
            changed = False
            c = np.mean([pts[j] for j in members], axis=0)
            for k, pk in enumerate(pts):
                if assigned[k]:
                    continue
                if _dist(pk, c) <= float(radius_m):
                    assigned[k] = True
                    members.append(k)
                    changed = True

        if len(members) < int(min_cluster_size):
            for j in members:
                assigned[j] = False
            continue

        centroid = np.mean([pts[j] for j in members], axis=0)
        rmap: Dict[str, int] = {}
        for j in members:
            rmap[reasons[j]] = rmap.get(reasons[j], 0) + 1

        clusters.append(Cluster(id=cid, centroid=centroid, count=len(members), reasons=rmap))
        cid += 1
        if len(clusters) >= int(max_clusters):
            break

    if not clusters:
        centroid = np.mean(pts, axis=0)
        rmap: Dict[str, int] = {}
        for r in reasons:
            rmap[r] = rmap.get(r, 0) + 1
        clusters = [Cluster(id=0, centroid=centroid, count=len(pts), reasons=rmap)]

    return clusters


def _yaw_pitch_from_vec(v: np.ndarray) -> Tuple[float, float]:
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw = math.degrees(math.atan2(vy, vx))
    horiz = math.sqrt(vx * vx + vy * vy)
    pitch = math.degrees(math.atan2(vz, max(1e-6, horiz)))
    return yaw, pitch


def _score_view(voxel_world, cam_pos: np.ndarray, target: np.ndarray) -> float:
    try:
        if voxel_world.is_blocked(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), unknown_is_blocked=False):
            return -1e9
    except Exception:
        pass

    ray = target - cam_pos
    dist = float(np.linalg.norm(ray))
    if dist <= 1e-3:
        return -1e9
    dirn = ray / dist

    unknown_ratio = 0.0
    try:
        unknown_ratio = float(voxel_world.unknown_ratio_along_ray(tuple(cam_pos), tuple(dirn), max_dist=dist))
    except Exception:
        unknown_ratio = 0.5

    hit_d = None
    try:
        hit_d = voxel_world.raycast_distance(tuple(cam_pos), tuple(dirn), max_dist=dist, unknown_is_blocked=False)
    except Exception:
        hit_d = None

    occluded = 1.0 if (hit_d is not None and float(hit_d) < dist - voxel_world.resolution * 2) else 0.0

    score = 0.0
    score += max(0.0, 4.0 - abs(dist - 1.6))
    score -= 6.0 * unknown_ratio
    score -= 8.0 * occluded

    return float(score)


def build_scan_plan(
    *,
    voxel_world,
    suggestions: Optional[List[Dict[str, Any]]] = None,
    scan_suggestions: Optional[List[Dict[str, Any]]] = None,
    max_views: int = 3,
    cluster_radius_m: float = 0.6,
    distance_m: float = 1.6,
    angles_deg: Optional[List[float]] = None,
    height_offset_m: float = 0.2,
) -> Dict[str, Any]:
    if angles_deg is None:
        angles_deg = [0.0, 30.0, -30.0, 60.0, -60.0]

    src = suggestions if suggestions is not None else (scan_suggestions or [])
    clusters = cluster_suggestions(src, radius_m=cluster_radius_m)

    candidates: List[Dict[str, Any]] = []
    for c in clusters:
        target = c.centroid
        for ang in angles_deg:
            theta = math.radians(float(ang))
            offset = np.array([math.cos(theta) * distance_m, math.sin(theta) * distance_m, 0.0], dtype=np.float32)
            cam_pos = target + offset
            cam_pos[2] = target[2] + float(height_offset_m)

            score = _score_view(voxel_world, cam_pos, target)
            if score < -1e8:
                continue

            yaw, pitch = _yaw_pitch_from_vec(target - cam_pos)

            candidates.append(
                {
                    "position": {"x": float(cam_pos[0]), "y": float(cam_pos[1]), "z": float(cam_pos[2])},
                    "look_at": {"x": float(target[0]), "y": float(target[1]), "z": float(target[2])},
                    "yaw_deg": float(yaw),
                    "pitch_deg": float(pitch),
                    "score": float(score),
                    "target_cluster_id": int(c.id),
                }
            )

    candidates.sort(key=lambda x: x.get("score", -1e9), reverse=True)

    chosen: List[Dict[str, Any]] = []
    used_clusters: set[int] = set()
    for cand in candidates:
        if len(chosen) >= int(max_views):
            break
        cid = int(cand.get("target_cluster_id", -1))
        if cid in used_clusters and len(used_clusters) < len(clusters):
            continue
        chosen.append(cand)
        used_clusters.add(cid)

    return {"clusters": [c.to_dict() for c in clusters], "next_best_views": chosen}


def propose_views(
    scan_suggestions: List[Dict[str, Any]],
    current_pose: Optional[List[float]],
    voxel_world: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    # current_pose reserved for future (e.g., prefer closer candidate views)
    del current_pose
    return build_scan_plan(voxel_world=voxel_world, scan_suggestions=scan_suggestions, **kwargs)
