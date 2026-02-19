from __future__ import annotations

import math

import numpy as np

from scanning.coverage import compute_unknown_hotspots, compute_work_aabb
from scanning.scan_hints import make_scan_hint


def _orbit(center: list[float], radius: float, height: float, k: int, n: int) -> dict:
    angle = (2.0 * math.pi * float(k)) / max(1.0, float(n))
    return {
        "position": [
            float(center[0] + radius * math.cos(angle)),
            float(center[1] + radius * math.sin(angle)),
            float(center[2] + height),
        ],
        "look_at": [float(center[0]), float(center[1]), float(center[2])],
        "distance_m": float(radius),
        "note": "Orbit for coverage",
        "kind": "scan_hint",
    }


def _generate_scan_plan_core(world_model, anchors: list[dict], n_hints: int = 7) -> list[dict]:
    supports = [a for a in anchors if a.get("kind") == "support" and a.get("position") is not None]
    work = compute_work_aabb(anchors, padding_m=1.25)

    if not supports and not work:
        return [
            make_scan_hint(
                [1.5, 1.5, 1.6],
                look_at=[0.0, 0.0, 1.0],
                note="Capture broad context around work area",
            )
        ]

    out: list[dict] = []
    if supports:
        points = np.asarray([s["position"] for s in supports], dtype=np.float32)
        center = np.mean(points, axis=0).tolist()
        radius = 1.8
        orbit_count = min(4, max(1, n_hints // 2))
        for k in range(orbit_count):
            out.append(_orbit(center, radius=radius, height=1.2, k=k, n=orbit_count))

    if work:
        box_min, box_max = work
        hotspots = compute_unknown_hotspots(world_model, box_min, box_max, max_points=12)
        for hp in hotspots[: max(1, n_hints - len(out))]:
            look = hp
            pos = [float(hp[0] + 0.8), float(hp[1] + 0.8), float(hp[2] + 1.2)]
            out.append(
                make_scan_hint(
                    pos, look_at=look, note="Scan UNKNOWN hotspot to increase confidence"
                )
            )

    dedup: list[dict] = []
    for h in out:
        p = np.asarray(h["position"], dtype=np.float32)
        if any(
            float(np.linalg.norm(p - np.asarray(x["position"], dtype=np.float32))) < 0.35
            for x in dedup
        ):
            continue
        dedup.append(h)
        if len(dedup) >= int(n_hints):
            break
    return dedup


def _esdf_distance_ok(world_model, pos: list[float], min_clearance_m: float) -> bool:
    if not pos or len(pos) != 3:
        return False
    esdf = getattr(world_model, "esdf", None)
    if esdf is None:
        return True
    fn = getattr(esdf, "query_distance", None) or getattr(esdf, "distance", None)
    if not callable(fn):
        return True
    try:
        d = float(fn([pos])[0])
        return bool(d >= float(min_clearance_m))
    except Exception:
        return True


def generate_scan_plan(world_model, anchors: list[dict], policy=None, N: int = 7):
    # keep backward compatibility with the old 3rd positional argument (N: int)
    if isinstance(policy, (int, float)):
        plan = list(_generate_scan_plan_core(world_model, anchors, int(policy)))
        min_clearance = 0.2
    else:
        plan = list(_generate_scan_plan_core(world_model, anchors, int(N)))
        min_clearance = (
            float(getattr(policy, "min_clearance_m", 0.2)) if policy is not None else 0.2
        )

    filtered = []
    for h in plan:
        pos = h.get("pos") or h.get("position")
        if isinstance(pos, list) and len(pos) == 3:
            if _esdf_distance_ok(world_model, pos, min_clearance):
                filtered.append(h)
            else:
                continue
        else:
            filtered.append(h)
    return filtered
