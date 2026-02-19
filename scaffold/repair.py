from __future__ import annotations

import math
import uuid
from typing import Any

from trace.decision_trace import add_constraint_eval, add_trace_event
from world.occupancy import OCCUPIED, UNKNOWN


def _nudge(pos: list[float], step: float, k: int) -> list[float]:
    angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
    a = angles[k % len(angles)]
    return [float(pos[0]) + step * math.cos(a), float(pos[1]) + step * math.sin(a), float(pos[2])]


def repair_elements(
    elements: list[dict],
    world_model,
    policy,
    *,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[list[dict], dict]:
    rounds = int(getattr(policy, "planner_repair_rounds", 8))
    max_shift = float(getattr(policy, "planner_max_shift_m", 0.60))
    step = max(0.05, min(0.20, max_shift / max(1, rounds)))

    decision_id = f"repair:{uuid.uuid4()}"
    forbid_unknown = str(getattr(policy, "unknown_mode", "forbid")) == "forbid"

    repaired = [dict(e) for e in elements]
    moved = 0

    if trace is not None:
        add_trace_event(
            trace,
            "planner_repair_start",
            {"rounds": rounds, "max_shift_m": max_shift, "step_m": step, "forbid_unknown": forbid_unknown},
        )

    rounds_used = 0
    for r in range(rounds):
        rounds_used = r + 1
        any_change = False
        for i, e in enumerate(repaired):
            pose = dict(e.get("pose") or {})
            pos = pose.get("position", pose.get("pos", [0.0, 0.0, 0.0]))
            pos = list(pos if isinstance(pos, list) else [0.0, 0.0, 0.0])
            if len(pos) != 3:
                continue

            occ = int(world_model.occupancy.query([pos])[0])
            bad = (occ == int(OCCUPIED)) or (forbid_unknown and occ == int(UNKNOWN))
            if not bad:
                continue

            ok_found = False
            for k in range(8):
                cand = _nudge(pos, step, k + r * 8)
                occ2 = int(world_model.occupancy.query([cand])[0])
                bad2 = (occ2 == int(OCCUPIED)) or (forbid_unknown and occ2 == int(UNKNOWN))
                if trace is not None:
                    add_constraint_eval(
                        trace,
                        decision_id=decision_id,
                        constraint_id="repair_nudge_try",
                        ok=(not bad2),
                        reason="occupied_or_unknown" if bad2 else "clear",
                        metrics={"round": r, "try": k, "occ": occ2},
                        element_id=e.get("id"),
                    )
                if not bad2:
                    pos = cand
                    ok_found = True
                    break

            if ok_found:
                if "position" in pose:
                    pose["position"] = pos
                else:
                    pose["pos"] = pos
                e["pose"] = pose
                repaired[i] = e
                any_change = True
                moved += 1

        if not any_change:
            break

    meta = {"moved_elements": moved, "rounds_used": rounds_used if rounds > 0 else 0}
    if trace is not None:
        add_trace_event(trace, "planner_repair_done", meta)
    return repaired, meta
