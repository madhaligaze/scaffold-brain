from __future__ import annotations

from typing import Any
import numpy as np

from scanning.scan_hints import make_scan_hint
from world.occupancy import UNKNOWN


def propose_extension(
    *,
    p0: list[float],
    p1: list[float],
    world_model,
    policy,
    step_m: float | None = None,
    max_extend_m: float = 6.0,
) -> dict[str, Any]:
    a = np.asarray(p0, dtype=np.float32).reshape(3)
    b = np.asarray(p1, dtype=np.float32).reshape(3)
    v = b - a
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return {"status": "invalid", "stop_reason": "DEGENERATE", "confidence": 0.0}
    d = v / L

    step = float(step_m) if step_m is not None else float(world_model.occupancy.voxel_size) * 0.75
    n = max(1, int(max_extend_m / max(step, 1e-3)))
    end = b.copy()
    stop_reason = "MAX_LEN"
    needs_scan = None
    constraint_trace: list[dict[str, Any]] = []

    for i in range(1, n + 1):
        q = b + d * (i * step)
        if not world_model.occupancy.in_bounds(q):
            stop_reason = "OUT_OF_SCOPE"
            constraint_trace.append({"step": i, "point": q.tolist(), "stop": True, "reason": "OUT_OF_SCOPE"})
            break
        occ = world_model.occupancy.query([q.tolist()])[0]
        if occ == int(UNKNOWN) and getattr(policy, "unknown_mode", "forbid") != "allow":
            stop_reason = "UNKNOWN"
            needs_scan = make_scan_hint(q.tolist(), look_at=b.tolist(), note="Scan missing area to confirm extension")
            end = q
            constraint_trace.append({"step": i, "point": q.tolist(), "occ": int(occ), "stop": True, "reason": "UNKNOWN"})
            break
        dist = world_model.query_distance([q.tolist()])[0]
        if dist < float(policy.min_clearance_m):
            stop_reason = "COLLISION"
            constraint_trace.append(
                {"step": i, "point": q.tolist(), "occ": int(occ), "dist_m": float(dist), "stop": True, "reason": "COLLISION"}
            )
            break
        end = q
        if len(constraint_trace) < 40:
            constraint_trace.append({"step": i, "point": q.tolist(), "occ": int(occ), "dist_m": float(dist), "stop": False})

    conf = 0.6
    if stop_reason == "COLLISION":
        conf = 0.8
    if stop_reason == "OUT_OF_SCOPE":
        conf = 0.4
    if stop_reason == "UNKNOWN":
        conf = 0.2

    return {
        "status": "ok",
        "axis": d.tolist(),
        "start": b.tolist(),
        "end": end.tolist(),
        "confidence": float(conf),
        "stop_reason": stop_reason,
        "needs_scan": needs_scan,
        "constraint_trace": constraint_trace,
    }


def propose_anchor_linear_hypotheses(world_model, anchors: list[dict], policy) -> tuple[list[dict], list[dict]]:
    pts = [a for a in anchors if a.get("kind") in {"support", "boundary", "target"} and a.get("position") is not None]
    if len(pts) < 2:
        return [], []

    P = np.asarray([p["position"] for p in pts], dtype=np.float32)
    used = set()
    hyps: list[dict] = []
    needs: list[dict] = []
    for i in range(len(pts)):
        if i in used:
            continue
        di = np.linalg.norm(P - P[i : i + 1], axis=1)
        di[i] = 1e9
        j = int(np.argmin(di))
        if j in used:
            continue
        if float(di[j]) < 0.25 or float(di[j]) > 6.0:
            continue
        used.add(i)
        used.add(j)
        a = pts[i]
        b = pts[j]
        ext = propose_extension(p0=a["position"], p1=b["position"], world_model=world_model, policy=policy)
        hyp = {
            "id": f"hyp_line_{a.get('id','a')}_{b.get('id','b')}",
            "type": "linear_hypothesis",
            "seed": {"a": a, "b": b},
            "proposal": ext,
        }
        hyps.append(hyp)
        if ext.get("needs_scan"):
            needs.append(ext["needs_scan"])
    return hyps, needs
