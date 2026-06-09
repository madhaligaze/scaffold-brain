"""Stage 7: Readiness gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReadinessProfile:
    max_miss_rate: float = 0.20
    max_mismatch_rate: float = 0.12
    max_median_abs_error_m: float = 0.07
    max_unknown_ratio: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_miss_rate": float(self.max_miss_rate),
            "max_mismatch_rate": float(self.max_mismatch_rate),
            "max_median_abs_error_m": float(self.max_median_abs_error_m),
            "max_unknown_ratio": float(self.max_unknown_ratio),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReadinessProfile":
        data = data or {}
        return cls(
            max_miss_rate=float(data.get("max_miss_rate", cls.max_miss_rate)),
            max_mismatch_rate=float(data.get("max_mismatch_rate", cls.max_mismatch_rate)),
            max_median_abs_error_m=float(data.get("max_median_abs_error_m", cls.max_median_abs_error_m)),
            max_unknown_ratio=float(data.get("max_unknown_ratio", cls.max_unknown_ratio)),
        )


ReadinessThresholds = ReadinessProfile


def compute_readiness(
    *,
    reprojection: Optional[Dict[str, Any]] = None,
    unknown_ratio: Optional[float] = None,
    profile: Optional[ReadinessProfile] = None,
    voxel_world: Any = None,
    target_center: Optional[Tuple[float, float, float]] = None,
    target_half_extents: Optional[Tuple[float, float, float]] = None,
    thresholds: Optional[ReadinessProfile] = None,
) -> Dict[str, Any]:
    profile = profile or thresholds or ReadinessProfile()
    reasons: List[str] = []

    if unknown_ratio is None and voxel_world is not None and target_center is not None and target_half_extents is not None:
        try:
            unknown_ratio = float(voxel_world.unknown_fraction_in_box(target_center, target_half_extents))
        except Exception:
            unknown_ratio = None

    miss_rate = None
    mismatch_rate = None
    median_abs = None

    if reprojection:
        miss_rate = reprojection.get("miss_rate")
        mismatch_rate = reprojection.get("mismatch_rate")
        median_abs = reprojection.get("median_abs_error_m")

        if miss_rate is not None and float(miss_rate) > profile.max_miss_rate:
            reasons.append("miss_rate")
        if mismatch_rate is not None and float(mismatch_rate) > profile.max_mismatch_rate:
            reasons.append("mismatch_rate")
        if median_abs is not None and float(median_abs) > profile.max_median_abs_error_m:
            reasons.append("median_abs_error")

    if unknown_ratio is not None and float(unknown_ratio) > profile.max_unknown_ratio:
        reasons.append("unknown_ratio")

    ready = len(reasons) == 0

    return {
        "ready_to_lock": bool(ready),
        "reasons": reasons,
        "profile": profile.to_dict(),
        "observed": {
            "miss_rate": miss_rate,
            "mismatch_rate": mismatch_rate,
            "median_abs_error_m": median_abs,
            "unknown_ratio": unknown_ratio,
        },
        "unknown_ratio": unknown_ratio,
        "thresholds": profile.to_dict(),
        "reprojection": {
            "miss_rate": miss_rate,
            "mismatch_rate": mismatch_rate,
            "median_abs_error_m": median_abs,
        }
        if reprojection
        else None,
    }
