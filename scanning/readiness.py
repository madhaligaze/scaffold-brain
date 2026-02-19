from __future__ import annotations

import math
from typing import Any

import numpy as np

from scanning.coverage import compute_work_aabb


def _views_near_point(world_model, p: list[float], *, radius_m: float = 2.5) -> int:
    # Approximate using quantized viewpoints list (stored only as count), so we use camera_history if available.
    hist = world_model.metrics.get("camera_positions")  # optional future field
    if not isinstance(hist, list):
        return 0
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    r2 = float(radius_m * radius_m)
    cnt = 0
    for it in hist:
        if not (isinstance(it, (list, tuple)) and len(it) == 3):
            continue
        dx = float(it[0]) - px
        dy = float(it[1]) - py
        dz = float(it[2]) - pz
        if dx * dx + dy * dy + dz * dz <= r2:
            cnt += 1
    return int(cnt)


def compute_readiness(world_model, anchors: list[dict], policy) -> tuple[bool, float, list[str]]:
    """
    STAGE D: readiness is a gate, not a vibe.
    - coverage in the work AABB
    - min viewpoints overall
    - min views around each support anchor (if we have camera history)
    """
    aabb = compute_work_aabb(anchors, padding_m=1.0)
    reasons: list[str] = []
    if aabb is None:
        # Legacy compatibility path: allow readiness without anchors based on global occupancy.
        stats = world_model.occupancy.stats()
        total = float(stats.get("total", 0) or 0)
        if total <= 0:
            return False, 0.0, ["NO_ANCHORS"]
        unknown = float(stats.get("unknown", 0) or 0)
        observed = 1.0 - (unknown / max(1.0, total))
        min_obs = float(getattr(policy, "readiness_observed_ratio_min", 0.1))
        vp = int(world_model.metrics.get("viewpoints", 0))
        min_vp = int(getattr(policy, "min_viewpoints", 1) or 1)
        if observed < min_obs:
            reasons.append(f"LOW_COVERAGE:{observed:.3f}<{min_obs:.3f}")
        if vp < min_vp:
            reasons.append(f"LOW_VIEWPOINTS:{vp}<{min_vp}")
        score = 0.75 * max(0.0, min(1.0, observed)) + 0.25 * max(
            0.0,
            min(1.0, float(vp) / float(max(1, min_vp))),
        )
        return (observed >= min_obs) and (vp >= min_vp), float(score), reasons

    bmin, bmax = aabb
    stats = world_model.occupancy.stats_aabb(bmin, bmax)
    if int(stats.get("total", 0)) <= 0:
        return False, 0.0, ["EMPTY_AABB"]

    unknown = float(stats.get("unknown", 0))
    total = float(stats.get("total", 1))
    observed = 1.0 - (unknown / max(1.0, total))

    min_obs = float(getattr(policy, "readiness_observed_ratio_min", 0.1))
    if observed < min_obs:
        reasons.append(f"LOW_COVERAGE:{observed:.3f}<{min_obs:.3f}")

    # Viewpoints
    vp = int(world_model.metrics.get("viewpoints", 0))
    min_vp = int(getattr(policy, "min_viewpoints", 1) or 1)
    if vp < min_vp:
        reasons.append(f"LOW_VIEWPOINTS:{vp}<{min_vp}")

    # Per-support view requirement (best-effort)
    supports = [a for a in anchors if a.get("kind") == "support" and isinstance(a.get("position"), (list, tuple))]
    min_views_per_support = int(getattr(policy, "min_views_per_support", 2) or 2)
    if supports:
        bad = 0
        for s in supports:
            p = list(s.get("position"))
            n = _views_near_point(world_model, p, radius_m=2.5)
            if n < min_views_per_support:
                bad += 1
        if bad > 0:
            reasons.append(f"SUPPORTS_NEED_MORE_VIEWS:{bad}/{len(supports)}")

    # Score: weighted blend
    score = 0.75 * max(0.0, min(1.0, observed)) + 0.25 * max(0.0, min(1.0, float(vp) / float(max(1, min_vp))))
    ready = (observed >= min_obs) and (vp >= min_vp) and (len(reasons) == 0)
    return bool(ready), float(score), reasons


_compute_readiness_core = compute_readiness


def _supports(anchors: list[dict]) -> list[dict]:
    out = []
    for a in anchors or []:
        if a.get("kind") in ("support", "anchor", "opora"):
            out.append(a)
    return out


def compute_readiness(world_model, anchors: list[dict], policy) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []
    ready, score, core_reasons = _compute_readiness_core(world_model, anchors, policy)
    reasons.extend(list(core_reasons or []))
    min_views = int(getattr(policy, "min_views_per_anchor", 0) or 0)
    if min_views > 0 and hasattr(world_model, "anchor_view_count"):
        bad = []
        for s in _supports(anchors):
            pos = s.get("position")
            if isinstance(pos, list) and len(pos) == 3:
                n = int(world_model.anchor_view_count(pos))
                if n < min_views:
                    bad.append({"id": s.get("id"), "views": n})
        if bad:
            ready = False
            reasons.append(f"min_views_per_anchor_not_met:{bad}")
    return bool(ready), float(max(0.0, min(1.0, score))), reasons
