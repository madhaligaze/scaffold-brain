from __future__ import annotations

from typing import Any, Iterable
import uuid

import numpy as np

from policy.unknown_space import apply_unknown_policy
from trace.decision_trace import add_constraint_eval
from world.occupancy import UNKNOWN


def _pos(e: dict[str, Any]) -> np.ndarray | None:
    pose = e.get("pose") or {}
    p = pose.get("position", pose.get("pos"))
    if isinstance(p, (list, tuple)) and len(p) == 3:
        return np.asarray(p, dtype=np.float32).reshape(3)
    return None


def _type(e: dict[str, Any]) -> str:
    return str(e.get("type") or e.get("kind") or "unknown")


def _iter_segments(elements: list[dict[str, Any]]) -> Iterable[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]]:
    for e in elements:
        t = _type(e)
        p = _pos(e)
        if p is None:
            continue
        dims = e.get("dims") or {}
        if t == "post":
            h = float(dims.get("height_m") or dims.get("height") or dims.get("z") or 0.0)
            if h <= 0:
                continue
            a = p
            b = p + np.asarray([0.0, 0.0, h], dtype=np.float32)
            yield t, a, b, e
        elif t in ("ledger", "brace"):
            a0 = dims.get("a")
            b0 = dims.get("b")
            if isinstance(a0, (list, tuple)) and isinstance(b0, (list, tuple)) and len(a0) == 3 and len(b0) == 3:
                a = np.asarray(a0, dtype=np.float32).reshape(3)
                b = np.asarray(b0, dtype=np.float32).reshape(3)
                yield t, a, b, e
        else:
            yield t, p, p, e


def collision_check(
    elements: list[dict[str, Any]],
    world_model,
    policy,
    *,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    violations: list[dict[str, Any]] = []
    if not elements:
        return False, [{"type": "NO_SCAFFOLD", "msg": "No scaffold elements generated"}]

    min_clear = float(policy.min_clearance_m)
    sample_pts: list[list[float]] = []
    meta: list[dict[str, Any]] = []

    for t, a, b, e in _iter_segments(elements):
        pa = a.tolist()
        pb = b.tolist()
        pm = ((a + b) * 0.5).tolist()
        sample_pts.extend([pa, pm, pb])
        ref = e.get("id") or e.get("name")
        meta.extend(
            [
                {"elem_type": t, "ref": ref, "where": "a"},
                {"elem_type": t, "ref": ref, "where": "m"},
                {"elem_type": t, "ref": ref, "where": "b"},
            ]
        )

    if not sample_pts:
        return False, [{"type": "NO_POINTS", "msg": "No positions to validate"}]

    d = world_model.query_distance(sample_pts)
    decision_id = f"collision:{uuid.uuid4()}"
    for i, dist in enumerate(d):
        ok = float(dist) >= min_clear
        if trace is not None:
            add_constraint_eval(
                trace,
                decision_id=decision_id,
                constraint_id="collision_clearance",
                ok=ok,
                reason="clearance_ok" if ok else "clearance_too_low",
                metrics={"dist_m": float(dist), "min_clearance_m": float(min_clear)},
                element_id=str(meta[i].get("ref") or ""),
            )
        if not ok:
            violations.append(
                {
                    "type": "COLLISION",
                    "at": meta[i],
                    "dist_m": float(dist),
                    "min_clearance_m": min_clear,
                }
            )
    return len(violations) == 0, violations


def stability_rules(elements: list[dict[str, Any]], policy) -> tuple[bool, list[dict[str, Any]]]:
    violations: list[dict[str, Any]] = []
    posts = [e for e in elements if _type(e) == "post"]
    braces = [e for e in elements if _type(e) == "brace"]
    decks = [e for e in elements if _type(e) == "deck"]

    if len(posts) < 4:
        violations.append({"type": "STABILITY_TOO_FEW_POSTS", "count": len(posts), "min": 4})

    if bool(getattr(policy, "stability_require_diagonals", True)) and len(braces) < 2:
        violations.append({"type": "STABILITY_MISSING_BRACES", "count": len(braces), "min": 2})

    for deck in decks:
        pos = _pos(deck)
        z = float((pos if pos is not None else np.zeros(3))[2])
        if z < 0.2:
            violations.append({"type": "DECK_TOO_LOW", "z_m": z})

    return len(violations) == 0, violations


def access_rules(elements: list[dict[str, Any]], policy) -> tuple[bool, list[dict[str, Any]]]:
    del policy
    violations: list[dict[str, Any]] = []
    decks = [e for e in elements if _type(e) == "deck"]
    access = [e for e in elements if _type(e) in ("stair", "ladder")]

    if decks and not access:
        violations.append({"type": "ACCESS_MISSING_STAIRS", "msg": "Deck exists but no stair/ladder element present"})

    return len(violations) == 0, violations




def unknown_space_check(
    elements: list[dict[str, Any]],
    world_model,
    policy,
    *,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Enforces unknown_mode=forbid strictly; unknown_mode=buffer is advisory (traced).
    """
    decision_id = f"unknown:{uuid.uuid4()}"
    report = apply_unknown_policy(world_model, [], policy)
    mode = str(report.get("mode", "forbid"))
    violations: list[dict[str, Any]] = []

    for e in elements:
        p = _pos(e)
        if p is None:
            continue
        occ = int(world_model.occupancy.query([p.tolist()])[0])
        in_unknown = occ == int(UNKNOWN)

        if mode == "forbid":
            ok = not in_unknown
            if trace is not None:
                add_constraint_eval(
                    trace,
                    decision_id=decision_id,
                    constraint_id="no_unknown_at_pose",
                    ok=ok,
                    reason="observed" if ok else "unknown_voxel",
                    metrics={"occ": occ, "mode": mode},
                    element_id=str(e.get("id") or ""),
                )
            if not ok:
                violations.append({"type": "UNKNOWN_AT_POSE", "element": e.get("type"), "pose": p.tolist()})

    if mode != "forbid" and trace is not None:
        add_constraint_eval(
            trace,
            decision_id=decision_id,
            constraint_id="unknown_buffer_mode",
            ok=True,
            reason="buffer_or_allow_mode_no_hard_fail",
            metrics={"mode": mode, "unknown_buffer_m": float(getattr(policy, "unknown_buffer_m", 0.5))},
            element_id=None,
        )

    return len(violations) == 0, violations

def validate_all(
    elements: list[dict[str, Any]],
    world_model,
    policy,
    *,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    all_violations: list[dict[str, Any]] = []

    ok1, v1 = collision_check(elements, world_model, policy, trace=trace)
    all_violations.extend(v1)

    ok2, v2 = stability_rules(elements, policy)
    all_violations.extend(v2)

    ok3, v3 = access_rules(elements, policy)
    all_violations.extend(v3)

    ok4, v4 = unknown_space_check(elements, world_model, policy, trace=trace)
    all_violations.extend(v4)

    return (ok1 and ok2 and ok3 and ok4 and len(all_violations) == 0), all_violations
