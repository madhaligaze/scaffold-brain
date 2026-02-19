from __future__ import annotations

from typing import Any

import numpy as np

from scaffold.spec import DEFAULT_SPEC
from scaffold.trace import trace_candidate_grid, trace_element_added, trace_solver_start


def _work_bounds(anchors: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    pts = [a.get("position") for a in anchors if a.get("kind") in ("support", "boundary") and a.get("position") is not None]
    pts = [p for p in pts if isinstance(p, (list, tuple)) and len(p) == 3]
    if not pts:
        return None
    P = np.asarray(pts, dtype=np.float32)
    lo = np.min(P, axis=0)
    hi = np.max(P, axis=0)
    return lo, hi


def _snap_to_catalog(length_m: float, catalog: tuple[float, ...]) -> float:
    arr = np.asarray(list(catalog), dtype=np.float32)
    j = int(np.argmin(np.abs(arr - float(length_m))))
    return float(arr[j])


def generate_scaffold(
    world_model,
    anchors: list[dict],
    policy,
    *,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    del world_model
    event_trace = trace if trace is not None else []
    trace_solver_start(
        event_trace,
        {
            "grid_step_m": float(getattr(policy, "scaffold_grid_step_m", 2.0)),
            "default_height_m": float(getattr(policy, "scaffold_default_height_m", DEFAULT_SPEC.default_height_m)),
        },
    )

    bounds = _work_bounds(anchors)
    if bounds is None:
        return [], {"solver": "grid_v1", "supports_used": 0, "elements": 0}

    lo, hi = bounds
    lo = lo - np.asarray([0.6, 0.6, 0.0], dtype=np.float32)
    hi = hi + np.asarray([0.6, 0.6, 0.0], dtype=np.float32)

    step = float(getattr(policy, "scaffold_grid_step_m", 2.0))
    step = float(
        np.clip(
            step,
            float(getattr(policy, "scaffold_min_bay_m", DEFAULT_SPEC.min_bay_m)),
            float(getattr(policy, "scaffold_max_bay_m", DEFAULT_SPEC.max_bay_m)),
        )
    )

    xs = np.arange(float(lo[0]), float(hi[0]) + 1e-6, step, dtype=np.float32)
    ys = np.arange(float(lo[1]), float(hi[1]) + 1e-6, step, dtype=np.float32)

    height = float(getattr(policy, "scaffold_default_height_m", DEFAULT_SPEC.default_height_m))
    deck_levels = list(getattr(policy, "scaffold_deck_levels_m", [2.0, height]))
    deck_levels = [float(z) for z in deck_levels if float(z) > 0.1]
    if not deck_levels:
        deck_levels = [min(2.0, height)]
    top_z = float(min(max(deck_levels), height))

    trace_candidate_grid(event_trace, {"nx": int(xs.size), "ny": int(ys.size), "step_m": float(step), "top_z_m": float(top_z)})

    elements: list[dict] = []

    for x in xs:
        for y in ys:
            p = [float(x), float(y), float(lo[2])]
            e = {
                "type": "post",
                "pose": {"position": p, "quaternion": [0.0, 0.0, 0.0, 1.0]},
                "dims": {"height_m": float(height), "radius_m": float(DEFAULT_SPEC.post_radius_m)},
            }
            elements.append(e)
    for i in range(min(3, len(elements))):
        trace_element_added(event_trace, elements[i], "grid_post")

    def P(ix: int, iy: int) -> list[float]:
        return [float(xs[ix]), float(ys[iy]), float(lo[2])]

    nx = int(xs.size)
    ny = int(ys.size)

    def add_ledger(a: list[float], b: list[float], z: float, reason: str) -> None:
        aa = [a[0], a[1], float(z)]
        bb = [b[0], b[1], float(z)]
        L = float(np.linalg.norm(np.asarray(bb, dtype=np.float32) - np.asarray(aa, dtype=np.float32)))
        Ls = _snap_to_catalog(L, DEFAULT_SPEC.ledger_lengths_m)
        e = {
            "type": "ledger",
            "pose": {"position": aa, "quaternion": [0.0, 0.0, 0.0, 1.0]},
            "dims": {"length_m": float(Ls), "a": aa, "b": bb, "radius_m": float(DEFAULT_SPEC.ledger_radius_m)},
        }
        elements.append(e)
        if len(elements) < 12:
            trace_element_added(event_trace, e, reason)

    base_z = float(lo[2] + 0.3)
    for ix in range(nx):
        for iy in range(ny):
            if ix + 1 < nx:
                add_ledger(P(ix, iy), P(ix + 1, iy), base_z, "ledger_base_x")
                add_ledger(P(ix, iy), P(ix + 1, iy), top_z, "ledger_top_x")
            if iy + 1 < ny:
                add_ledger(P(ix, iy), P(ix, iy + 1), base_z, "ledger_base_y")
                add_ledger(P(ix, iy), P(ix, iy + 1), top_z, "ledger_top_y")

    def add_brace(a: list[float], b: list[float], z0: float, z1: float, reason: str) -> None:
        aa = [a[0], a[1], float(z0)]
        bb = [b[0], b[1], float(z1)]
        e = {
            "type": "brace",
            "pose": {"position": aa, "quaternion": [0.0, 0.0, 0.0, 1.0]},
            "dims": {"a": aa, "b": bb, "radius_m": float(DEFAULT_SPEC.brace_radius_m)},
        }
        elements.append(e)
        if len(elements) < 20:
            trace_element_added(event_trace, e, reason)

    if bool(getattr(policy, "stability_require_diagonals", True)) and nx >= 2 and ny >= 2:
        add_brace(P(0, 0), P(nx - 1, 0), base_z, top_z, "brace_face_ymin")
        add_brace(P(0, ny - 1), P(nx - 1, ny - 1), base_z, top_z, "brace_face_ymax")

    deck = {
        "type": "deck",
        "pose": {
            "position": [float((lo[0] + hi[0]) * 0.5), float((lo[1] + hi[1]) * 0.5), float(top_z)],
            "quaternion": [0.0, 0.0, 0.0, 1.0],
        },
        "dims": {"box_min": [float(lo[0]), float(lo[1]), float(top_z)], "box_max": [float(hi[0]), float(hi[1]), float(top_z)]},
    }
    elements.append(deck)
    trace_element_added(event_trace, deck, "deck_top")

    stair = {
        "type": "stair",
        "pose": {"position": [float(lo[0]), float(lo[1]), float(base_z)], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "dims": {"to_z_m": float(top_z)},
    }
    elements.append(stair)
    trace_element_added(event_trace, stair, "stair_marker")

    solver_meta = {
        "solver": "grid_v1",
        "supports_used": int(len(anchors)),
        "elements": int(len(elements)),
        "step_m": float(step),
        "top_z_m": float(top_z),
    }
    return elements, solver_meta
