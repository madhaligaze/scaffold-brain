# modules/extension_engine.py
"""
Stage 3 - Engineering reconstruction (not fantasy)
--------------------------------------------------

Given a fitted primitive (cylinder/box) and voxel world with UNKNOWN/FREE/OCCUPIED,
produce:
- observable_segment: the actually observed part from 3D points
- termination_evidence: signals that an end really terminates
- extension_hypotheses: conservative extensions that stop at:
    * OCCUPIED surface (wall/column/beam)
    * another object (primitive overlap)
    * UNKNOWN (then needs_scan=True + scan hint)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def observable_segment_from_points(
    *,
    points_world: np.ndarray,
    axis_origin: np.ndarray,
    axis_dir: np.ndarray,
    trim_pct: float = 5.0,
) -> Optional[Dict[str, Any]]:
    if points_world is None or points_world.shape[0] < 30:
        return None

    a = _norm(axis_dir)
    X = points_world - axis_origin.reshape(1, 3)
    t = (X @ a).astype(np.float64)

    lo = np.percentile(t, trim_pct)
    hi = np.percentile(t, 100.0 - trim_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-3:
        lo = float(np.min(t))
        hi = float(np.max(t))

    p0 = (axis_origin + a * float(lo)).astype(np.float64)
    p1 = (axis_origin + a * float(hi)).astype(np.float64)
    return {
        "t0": float(lo),
        "t1": float(hi),
        "p0": p0.tolist(),
        "p1": p1.tolist(),
    }


def _cap_plane_score(points_slab: np.ndarray, axis_dir: np.ndarray) -> float:
    if points_slab.shape[0] < 20:
        return 0.0
    c = points_slab.mean(axis=0)
    X = points_slab - c
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)
    w = w[order]
    V = V[:, order]
    n = _norm(V[:, 0])
    a = _norm(axis_dir)
    align = abs(float(np.dot(n, a)))
    thickness = float(np.sqrt(max(w[0], 0.0)))
    thick_score = float(np.clip((0.025 - thickness) / 0.019, 0.0, 1.0))
    return float(np.clip(0.65 * align + 0.35 * thick_score, 0.0, 1.0))


def termination_evidence_cylinder(
    *,
    points_world: np.ndarray,
    axis_origin: np.ndarray,
    axis_dir: np.ndarray,
    t0: float,
    t1: float,
    radius: float,
    nearby_labels: Optional[List[Tuple[str, List[float]]]] = None,
) -> Dict[str, Any]:
    a = _norm(axis_dir)
    X = points_world - axis_origin.reshape(1, 3)
    t = (X @ a).astype(np.float64)

    behind = max(0.05, 2.0 * radius)
    ahead = max(0.06, 2.2 * radius)

    def _end(end_t: float, sign: int) -> Dict[str, Any]:
        if sign > 0:
            in_behind = (t >= (end_t - behind)) & (t <= end_t)
            in_ahead = (t > end_t) & (t <= (end_t + ahead))
        else:
            in_behind = (t <= (end_t + behind)) & (t >= end_t)
            in_ahead = (t < end_t) & (t >= (end_t - ahead))

        nb = int(np.sum(in_behind))
        na = int(np.sum(in_ahead))
        density_drop = 0.0
        if nb >= 25:
            ratio = na / max(1, nb)
            density_drop = float(np.clip((0.25 - ratio) / 0.25, 0.0, 1.0))

        slab = 0.03
        if sign > 0:
            slab_sel = (t >= (end_t - slab)) & (t <= (end_t + slab * 0.5))
        else:
            slab_sel = (t <= (end_t + slab)) & (t >= (end_t - slab * 0.5))
        cap_score = _cap_plane_score(points_world[slab_sel], a)

        semantic = 0.0
        if nearby_labels:
            end_p = axis_origin + a * float(end_t)
            for lbl, pos in nearby_labels:
                label_lower = (lbl or "").lower()
                if not any(k in label_lower for k in ["flange", "valve", "endcap", "end-cap", "cap"]):
                    continue
                p = np.array(pos[:3], dtype=np.float64)
                if float(np.linalg.norm(p - end_p)) < max(0.15, 2.0 * radius):
                    semantic = 1.0
                    break

        score = float(np.clip(0.45 * cap_score + 0.35 * density_drop + 0.20 * semantic, 0.0, 1.0))
        return {
            "cap_score": float(cap_score),
            "density_drop": float(density_drop),
            "semantic": float(semantic),
            "termination_score": float(score),
            "terminated": bool(score >= 0.7),
        }

    return {
        "neg_end": _end(float(t0), -1),
        "pos_end": _end(float(t1), +1),
    }


def _aabb_from_obj(obj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array(obj.get("pose", {}).get("position", [0, 0, 0]), dtype=np.float64)
    dims = obj.get("dimensions", {})
    if "dx" in dims:
        d = np.array([_safe_float(dims.get("dx", 0.1)), _safe_float(dims.get("dy", 0.1)), _safe_float(dims.get("dz", 0.1))], dtype=np.float64)
    else:
        r = _safe_float(dims.get("radius", 0.05))
        L = _safe_float(dims.get("length", 0.2))
        d = np.array([2 * r, 2 * r, max(1e-3, L)], dtype=np.float64)
    mn = pos - d / 2.0
    mx = pos + d / 2.0
    return mn, mx


def point_in_other_object(
    p: np.ndarray,
    *,
    objects: List[Dict[str, Any]],
    exclude_id: Optional[str],
    padding: float = 0.02,
) -> bool:
    for o in objects:
        if exclude_id and o.get("id") == exclude_id:
            continue
        mn, mx = _aabb_from_obj(o)
        mn = mn - padding
        mx = mx + padding
        if np.all(p >= mn) and np.all(p <= mx):
            return True
    return False


def propose_axis_extension(
    *,
    obj: Dict[str, Any],
    voxel_world,
    axis_origin: np.ndarray,
    axis_dir: np.ndarray,
    t0: float,
    t1: float,
    base_confidence: float,
    termination_evidence: Optional[Dict[str, Any]],
    objects: List[Dict[str, Any]],
    max_extension_m: float = 8.0,
) -> Tuple[List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    a = _norm(axis_dir)
    step = float(getattr(voxel_world, "resolution", 0.05)) if voxel_world is not None else 0.05
    step = max(0.01, min(step, 0.08))
    max_steps = int(max_extension_m / step)

    scan_suggestions: List[Dict[str, Any]] = []
    needs_scan = False
    hypotheses: List[Dict[str, Any]] = []

    def _extend(end_name: str, start_t: float, sign: int) -> Dict[str, Any]:
        nonlocal needs_scan
        te = (termination_evidence or {}).get(end_name, {})
        if te.get("terminated") is True:
            return {
                "end": end_name,
                "t_start": float(start_t),
                "t_end": float(start_t),
                "length": 0.0,
                "confidence": float(base_confidence) * 0.9,
                "stop_reason": "TERMINATED",
            }

        last_t = float(start_t)
        stop_reason = "MAX_LENGTH"
        scan_hint = None

        for k in range(1, max_steps + 1):
            t = float(start_t + sign * k * step)
            p = (axis_origin + a * t).astype(np.float64)

            if voxel_world is not None:
                st = int(voxel_world.get_state(float(p[0]), float(p[1]), float(p[2])))
                if st == int(getattr(voxel_world, "OCCUPIED", 1)):
                    stop_reason = "OCCUPIED"
                    break
                if st == int(getattr(voxel_world, "UNKNOWN", -1)):
                    stop_reason = "UNKNOWN"
                    needs_scan = True
                    scan_hint = p.tolist()
                    break

            if point_in_other_object(p, objects=objects, exclude_id=obj.get("id"), padding=0.03):
                stop_reason = "OTHER_OBJECT"
                break

            last_t = t

        length = float(abs(last_t - start_t))
        conf = float(base_confidence)
        conf *= float(np.exp(-length / 5.0))
        if stop_reason == "UNKNOWN":
            conf *= 0.6
        elif stop_reason == "OTHER_OBJECT":
            conf *= 0.8
        elif stop_reason == "OCCUPIED":
            conf *= 0.9

        h = {
            "end": end_name,
            "t_start": float(start_t),
            "t_end": float(last_t),
            "length": float(length),
            "confidence": float(np.clip(conf, 0.0, 1.0)),
            "stop_reason": stop_reason,
        }
        if scan_hint is not None:
            h["scan_hint_world"] = scan_hint
            scan_suggestions.append(
                {
                    "reason": "needs_scan_unknown",
                    "world_point": scan_hint,
                    "object_id": obj.get("id"),
                    "object_label": obj.get("class_label"),
                    "end": end_name,
                }
            )
        return h

    hypotheses.append(_extend("neg_end", float(t0), -1))
    hypotheses.append(_extend("pos_end", float(t1), +1))

    return hypotheses, needs_scan, scan_suggestions
