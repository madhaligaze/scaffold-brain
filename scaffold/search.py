from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

from scaffold.solver import generate_scaffold
from scaffold.validators import validate_all
from trace.decision_trace import add_trace_event


@dataclass
class Candidate:
    candidate_id: str
    params: dict[str, Any]
    elements: list[dict]
    valid: bool
    violations: list[dict]
    score: float


def _support_positions(anchors: list[dict]) -> list[list[float]]:
    out: list[list[float]] = []
    for a in anchors:
        if a.get("kind") == "support" and isinstance(a.get("position"), list) and len(a["position"]) == 3:
            out.append([float(a["position"][0]), float(a["position"][1]), float(a["position"][2])])
    return out


def _shift_elements(elements: list[dict], dx: float, dy: float) -> list[dict]:
    shifted: list[dict] = []
    for e in elements:
        ee = dict(e)
        pose = dict(ee.get("pose") or {})
        pos = pose.get("position", pose.get("pos", [0.0, 0.0, 0.0]))
        pos = list(pos if isinstance(pos, list) else [0.0, 0.0, 0.0])
        if len(pos) != 3:
            pos = [0.0, 0.0, 0.0]
        pos[0] = float(pos[0]) + float(dx)
        pos[1] = float(pos[1]) + float(dy)
        if "position" in pose:
            pose["position"] = pos
        else:
            pose["pos"] = pos
        ee["pose"] = pose
        shifted.append(ee)
    return shifted


def _score_candidate(policy, violations: list[dict], unknown_ratio: float, shift_m: float) -> float:
    v = float(len(violations))
    return (
        float(getattr(policy, "planner_score_w_violations", 10.0)) * v
        + float(getattr(policy, "planner_score_w_unknown", 2.0)) * float(max(0.0, min(1.0, unknown_ratio)))
        + float(getattr(policy, "planner_score_w_shift", 0.5)) * float(max(0.0, shift_m))
    )


def search_scaffolds(
    world_model,
    anchors: list[dict],
    policy,
    *,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[Candidate, list[Candidate]]:
    candidate_limit = int(getattr(policy, "planner_max_candidates", 24))
    max_shift = float(getattr(policy, "planner_max_shift_m", 0.60))

    support_pts = _support_positions(anchors)
    if not support_pts:
        support_pts = [[0.0, 0.0, 0.0]]

    deck_levels = list(getattr(policy, "scaffold_deck_levels_m", [2.0, 4.0]))
    heights = sorted(set([float(getattr(policy, "scaffold_default_height_m", 4.0))] + [float(x) for x in deck_levels]))
    heights = [h for h in heights if h > 0.5]
    if not heights:
        heights = [4.0]

    shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    r = max_shift
    shifts += [(-r, 0.0), (r, 0.0), (0.0, -r), (0.0, r)]
    shifts += [(-r, -r), (-r, r), (r, -r), (r, r)]

    candidates: list[Candidate] = []
    cid_base = str(uuid.uuid4())[:8]

    if trace is not None:
        add_trace_event(
            trace,
            "planner_search_start",
            {"candidate_limit": candidate_limit, "max_shift_m": max_shift, "heights": heights, "shifts": len(shifts)},
        )

    original_height = float(getattr(policy, "scaffold_default_height_m", 4.0))
    try:
        for h in heights:
            for (dx, dy) in shifts:
                if len(candidates) >= candidate_limit:
                    break

                setattr(policy, "scaffold_default_height_m", float(h))
                elements, solver_meta = generate_scaffold(world_model, anchors, policy, trace=trace)
                elements = _shift_elements(elements, dx, dy)

                valid, violations = validate_all(elements, world_model, policy, trace=trace)
                unknown_stats = world_model.occupancy.stats()
                total = float(unknown_stats.get("total", 0.0) or 0.0)
                unknown = float(unknown_stats.get("unknown", 0.0) or 0.0)
                unknown_ratio = (unknown / total) if total > 0 else 1.0

                shift_m = float(math.hypot(dx, dy))
                score = _score_candidate(policy, violations, unknown_ratio, shift_m)

                cand = Candidate(
                    candidate_id=f"{cid_base}-{len(candidates)}",
                    params={"height_m": float(h), "dx": float(dx), "dy": float(dy), "solver_meta": solver_meta},
                    elements=elements,
                    valid=bool(valid),
                    violations=violations,
                    score=float(score),
                )
                candidates.append(cand)

                if trace is not None:
                    add_trace_event(
                        trace,
                        "planner_candidate_evaluated",
                        {
                            "candidate_id": cand.candidate_id,
                            "score": cand.score,
                            "valid": cand.valid,
                            "violations": len(cand.violations),
                            "params": {"height_m": h, "dx": dx, "dy": dy},
                        },
                    )

            if len(candidates) >= candidate_limit:
                break
    finally:
        setattr(policy, "scaffold_default_height_m", original_height)

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (c.score, 0 if c.valid else 1, len(c.violations)),
    )
    best = candidates_sorted[0] if candidates_sorted else Candidate("none", {}, [], False, [{"code": "NO_CANDIDATES"}], 1e9)

    if trace is not None:
        add_trace_event(
            trace,
            "planner_search_done",
            {
                "candidates": len(candidates),
                "best_id": best.candidate_id,
                "best_score": best.score,
                "best_valid": best.valid,
                "best_violations": len(best.violations),
            },
        )

    return best, candidates_sorted
