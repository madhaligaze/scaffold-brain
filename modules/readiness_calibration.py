"""Stage 7b: Readiness calibration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from modules.readiness import ReadinessProfile


def _p(values: List[float], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.percentile(np.array(vals, dtype=np.float32), q))


def extract_reprojection_metrics_from_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for fr in frames or []:
        qm = fr.get("quality_metrics") or {}
        rep = None
        gs = qm.get("geometry_stats")
        if isinstance(gs, dict):
            rep = gs.get("reprojection")
        if rep is None:
            rep = qm.get("reprojection")
        if not isinstance(rep, dict):
            continue
        item: Dict[str, float] = {}
        for key in ("miss_rate", "mismatch_rate", "median_abs_error_m", "unknown_ratio"):
            if key in rep and rep.get(key) is not None:
                try:
                    item[key] = float(rep[key])
                except Exception:
                    pass
        if item:
            out.append(item)
    return out


def calibrate_profile(
    *,
    history: List[Dict[str, Any]],
    default: Optional[ReadinessProfile] = None,
    window: int = 80,
    safety_margin: float = 1.25,
) -> Dict[str, Any]:
    default = default or ReadinessProfile()
    hist = list(history or [])[-int(window):]

    miss = [h.get("miss_rate") for h in hist if h.get("miss_rate") is not None]
    mismatch = [h.get("mismatch_rate") for h in hist if h.get("mismatch_rate") is not None]
    median_err = [h.get("median_abs_error_m") for h in hist if h.get("median_abs_error_m") is not None]
    unknown = [h.get("unknown_ratio") for h in hist if h.get("unknown_ratio") is not None]

    p75_miss = _p(miss, 75)
    p75_mismatch = _p(mismatch, 75)
    p75_median = _p(median_err, 75)
    p75_unknown = _p(unknown, 75)

    def clamp(v: Optional[float], lo: float, hi: float, fallback: float) -> float:
        if v is None:
            return float(fallback)
        return float(max(lo, min(hi, v)))

    suggested = ReadinessProfile(
        max_miss_rate=clamp((p75_miss or default.max_miss_rate) * safety_margin, 0.05, 0.60, default.max_miss_rate),
        max_mismatch_rate=clamp((p75_mismatch or default.max_mismatch_rate) * safety_margin, 0.03, 0.45, default.max_mismatch_rate),
        max_median_abs_error_m=clamp((p75_median or default.max_median_abs_error_m) * safety_margin, 0.02, 0.25, default.max_median_abs_error_m),
        max_unknown_ratio=clamp((p75_unknown or default.max_unknown_ratio) * safety_margin, 0.10, 0.90, default.max_unknown_ratio),
    )

    return {
        "suggested_profile": suggested.to_dict(),
        "window": int(window),
        "samples": len(hist),
        "percentiles": {
            "p75_miss_rate": p75_miss,
            "p75_mismatch_rate": p75_mismatch,
            "p75_median_abs_error_m": p75_median,
            "p75_unknown_ratio": p75_unknown,
        },
        "safety_margin": float(safety_margin),
    }


def suggest_thresholds(metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    return calibrate_profile(history=metrics, default=ReadinessProfile()).get("suggested_profile", ReadinessProfile().to_dict())
