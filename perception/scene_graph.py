from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.object_tracker import ObjectTracker
from perception.primitive_fit import fit_box_pca, fit_cylinder_pca, fit_plane_ransac


@dataclass
class SceneGraph:
    """Structured scene entities: dominant plane + local primitives near anchors."""

    objects: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=lambda: {"updated_ts": float(time.time())})
    _tracker: ObjectTracker = field(default_factory=ObjectTracker, repr=False)

    def add_or_update(self, obj: dict[str, Any]) -> None:
        oid = str(obj.get("id") or "")
        if not oid:
            return

        for i, existing in enumerate(self.objects):
            if str(existing.get("id")) == oid:
                self.objects[i] = obj
                self.meta["updated_ts"] = float(time.time())
                return

        self.objects.append(obj)
        self.meta["updated_ts"] = float(time.time())

    def to_dict(self) -> dict[str, Any]:
        return {"objects": self.objects, "meta": self.meta}

    def serialize(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _supports(anchors: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in anchors or []:
        if a.get("kind") in ("support", "anchor", "opora") and isinstance(
            a.get("position"), (list, tuple)
        ):
            pos = list(a.get("position"))
            if len(pos) == 3:
                out.append(a)
    return out


def _classify_pca(points: np.ndarray) -> str:
    if points.shape[0] < 30:
        return "unknown"

    cov = np.cov(points.T)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 1e-12))[::-1]
    if w.size != 3:
        return "unknown"

    ratio = float(w[1] / max(1e-12, w[2]))
    return "cylinder" if ratio < 1.4 else "box"


def _extract_near_anchor(
    world_model, anchor_pos: list[float], radius_m: float = 0.45, max_points: int = 2000
) -> np.ndarray | None:
    pts = world_model.occupancy.occupied_points(max_points=max(12000, max_points * 6))
    if pts is None or len(pts) < 200:
        return None

    cloud = np.asarray(pts, dtype=np.float32)
    anchor = np.asarray(anchor_pos, dtype=np.float32).reshape(1, 3)
    dist2 = np.sum((cloud - anchor) ** 2, axis=1)
    keep = dist2 <= float(radius_m * radius_m)
    near = cloud[keep]
    if near.shape[0] < 80:
        return None

    if near.shape[0] > max_points:
        idx = np.random.choice(near.shape[0], size=max_points, replace=False)
        near = near[idx]
    return near


def update_scene_graph_from_world(
    scene_graph: SceneGraph,
    world_model,
    *,
    anchors: list[dict[str, Any]] | None = None,
    every_n_frames: int = 5,
) -> dict[str, Any] | None:
    frames = int(world_model.metrics.get("frames", 0))
    if frames % max(1, int(every_n_frames)) != 0:
        return None

    out: dict[str, Any] = {}

    pts = world_model.occupancy.occupied_points(max_points=6000)
    if pts is not None and len(pts) >= 100:
        points = np.asarray(pts, dtype=np.float32)
        plane = fit_plane_ransac(
            points, n_iter=250, dist_thresh=0.03, min_inliers=400, rng_seed=frames
        )
        if plane is not None:
            obj = {
                "id": "plane_dominant_0",
                "type": "plane",
                "normal": [float(x) for x in plane["normal"]],
                "d": float(plane["d"]),
                "inliers": int(plane["inliers"]),
                "confidence": float(min(1.0, max(0.0, plane["inliers"] / 1200.0))),
                "updated_ts": float(time.time()),
            }
            scene_graph.add_or_update(obj)
            out["plane_inliers"] = int(plane["inliers"])

    supports = _supports(anchors)
    detections: list[dict[str, Any]] = []
    for s in supports:
        sid = str(s.get("id") or "")
        pos = list(s.get("position"))
        local = _extract_near_anchor(world_model, pos, radius_m=0.45, max_points=1800)
        if local is None:
            continue

        kind = _classify_pca(local)
        if kind == "cylinder":
            cyl = fit_cylinder_pca(local)
            if not cyl.params.get("ok"):
                continue
            detections.append(
                {
                    "kind": "cylinder",
                    "params": {**cyl.params, "anchor_id": sid},
                    "score": float(max(0.0, min(1.0, 1.0 - (cyl.rmse / 0.06)))),
                }
            )
        elif kind == "box":
            box = fit_box_pca(local)
            if not box.params.get("ok"):
                continue
            detections.append(
                {
                    "kind": "box",
                    "params": {**box.params, "anchor_id": sid},
                    "score": float(max(0.0, min(1.0, 1.0 - (box.rmse / 0.06)))),
                }
            )

    if detections:
        tracked = scene_graph._tracker.update(detections)
        for t in tracked:
            scene_graph.add_or_update(
                {
                    "id": t.object_id,
                    "type": t.kind,
                    "params": t.params,
                    "confidence": t.score,
                    "updated_ts": float(time.time()),
                }
            )
        out["primitives"] = int(len(tracked))

    return out or None
