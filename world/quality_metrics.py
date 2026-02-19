from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QualityMetrics:
    observed_ratio: float
    unknown_ratio: float
    conflict_ratio: float
    voxel_weight_histogram: dict[str, Any]
    unknown_ratio_near_anchors: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_ratio": float(self.observed_ratio),
            "unknown_ratio": float(self.unknown_ratio),
            "conflict_ratio": float(self.conflict_ratio),
            "unknown_ratio_near_anchors": None if self.unknown_ratio_near_anchors is None else float(self.unknown_ratio_near_anchors),
            "voxel_weight_histogram": self.voxel_weight_histogram,
        }


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def compute_quality_metrics(world_model, anchors: list[dict] | None = None) -> QualityMetrics:
    stats = world_model.occupancy.stats() if hasattr(world_model, "occupancy") else {}
    total = _safe_float(stats.get("total", 0.0), 0.0)
    unknown = _safe_float(stats.get("unknown", 0.0), 0.0)
    conflict = _safe_float(stats.get("conflict", stats.get("conflicts", 0.0)), 0.0)

    if total <= 0:
        observed_ratio = 0.0
        unknown_ratio = 1.0
        conflict_ratio = 0.0
    else:
        unknown_ratio = max(0.0, min(1.0, unknown / total))
        observed_ratio = max(0.0, min(1.0, 1.0 - unknown_ratio))
        conflict_ratio = max(0.0, min(1.0, conflict / total))

    hist = {"bins": [0, 1, 2, 4, 8, 16, 32, 64, 255], "counts": []}
    w = getattr(world_model.occupancy, "weights", None) if hasattr(world_model, "occupancy") else None
    if isinstance(w, np.ndarray) and w.size > 0:
        flat = w.reshape(-1)
        bins = np.array(hist["bins"], dtype=np.int64)
        counts = []
        for i in range(len(bins) - 1):
            lo = int(bins[i])
            hi = int(bins[i + 1])
            counts.append(int(np.sum((flat >= lo) & (flat < hi))))
        counts.append(int(np.sum(flat >= int(bins[-1]))))
        hist["counts"] = counts

    unknown_near = None
    if anchors and hasattr(world_model.occupancy, "query"):
        pts = []
        for a in anchors:
            p = a.get("position")
            if isinstance(p, list) and len(p) == 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
        if pts:
            try:
                occ = world_model.occupancy.query(pts)
                occ = list(occ)
                unknown_near = float(sum(1 for v in occ if int(v) == 0)) / float(len(occ))
            except Exception:
                unknown_near = None

    return QualityMetrics(
        observed_ratio=observed_ratio,
        unknown_ratio=unknown_ratio,
        conflict_ratio=conflict_ratio,
        voxel_weight_histogram=hist,
        unknown_ratio_near_anchors=unknown_near,
    )
