from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackedObject:
    object_id: str
    kind: str
    params: dict[str, Any]
    score: float


def _center(params: dict[str, Any]) -> np.ndarray:
    if "center" in params and isinstance(params["center"], list) and len(params["center"]) == 3:
        return np.array(params["center"], dtype=np.float32)
    if "axis_origin" in params and isinstance(params["axis_origin"], list) and len(params["axis_origin"]) == 3:
        return np.array(params["axis_origin"], dtype=np.float32)
    if all(k in params for k in ("a", "b", "c", "d")):
        n = np.array([params["a"], params["b"], params["c"]], dtype=np.float32)
        d = float(params["d"])
        nn = float(np.linalg.norm(n))
        if nn > 1e-8:
            n = n / nn
            return (-d * n).astype(np.float32)
    return np.zeros((3,), dtype=np.float32)


class ObjectTracker:
    def __init__(self, match_dist_m: float = 0.30) -> None:
        self.match_dist_m = float(match_dist_m)
        self._objects: list[TrackedObject] = []

    def update(self, detections: list[dict[str, Any]]) -> list[TrackedObject]:
        updated: list[TrackedObject] = []
        used_prev: set[str] = set()

        for det in detections:
            kind = str(det.get("kind"))
            params = dict(det.get("params") or {})
            score = float(det.get("score", 1.0))
            c = _center(params)

            best = None
            best_d = 1e9
            for obj in self._objects:
                if obj.kind != kind or obj.object_id in used_prev:
                    continue
                d = float(np.linalg.norm(_center(obj.params) - c))
                if d < best_d:
                    best_d = d
                    best = obj

            if best is not None and best_d <= self.match_dist_m:
                used_prev.add(best.object_id)
                best.params = params
                best.score = score
                updated.append(best)
            else:
                updated.append(TrackedObject(object_id=str(uuid.uuid4()), kind=kind, params=params, score=score))

        self._objects = updated
        return list(self._objects)
