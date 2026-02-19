from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from policy.unknown_space import apply_unknown_policy
from scanning.next_best_view import generate_scan_plan
from scanning.readiness import compute_readiness
from scaffold.bom import bom_from_elements
from scaffold.repair import repair_elements
from scaffold.search import search_scaffolds
from scaffold.solver import generate_scaffold
from scaffold.trace import trace_candidate_grid, trace_solver_start, trace_validator_result
from scaffold.validators import validate_all
from trace.decision_trace import add_trace_event


router = APIRouter(tags=["planning"])


class ScaffoldRequest(BaseModel):
    session_id: str


@router.post("/planning/request_scaffold")
def request_scaffold(request: Request, payload: ScaffoldRequest):
    state = request.app.state.runtime
    sid = payload.session_id

    world = state.get_world(sid)
    anchors = state.anchors.get(sid, [])

    # STAGE C gate: never generate on BAD tracking
    tq = str(world.metrics.get("tracking_quality", "UNKNOWN"))
    if tq == "BAD":
        scan_plan = generate_scan_plan(world, anchors, state.policy)
        raise HTTPException(
            status_code=409,
            detail={
                "status": "TRACKING_BAD",
                "tracking_reasons": world.metrics.get("tracking_reasons", []),
                "scan_plan": scan_plan,
            },
        )

    ready, score, reasons = compute_readiness(world, anchors, state.policy)
    if not ready:
        scan_plan = generate_scan_plan(world, anchors, state.policy)
        raise HTTPException(
            status_code=409,
            detail={"status": "NOT_READY", "score": score, "reasons": reasons, "scan_plan": scan_plan},
        )

    unknown_report = apply_unknown_policy(world, anchors, state.policy)
    if unknown_report.get("violations"):
        raise HTTPException(status_code=409, detail={"status": "UNKNOWN_VIOLATION", "unknown": unknown_report})

    trace = state.traces.setdefault(sid, [])
    trace_solver_start(trace, {"session_id": sid})

    elements, solver_meta = generate_scaffold(world, anchors, state.policy, trace=trace)
    trace_candidate_grid(trace, solver_meta)

    valid, violations = validate_all(elements, world, state.policy, trace=trace)
    trace_validator_result(trace, valid, violations)
    if not valid:
        raise HTTPException(status_code=409, detail={"status": "VALIDATION_FAILED", "violations": violations})

    bom = bom_from_elements(elements)
    return {"status": "ok", "elements": elements, "bom": bom, "rev_hint": state.last_rev.get(sid)}


class PlanPayload(BaseModel):
    session_id: str


@router.post("/plan/scaffold")
def plan_scaffold(request: Request, payload: PlanPayload) -> dict[str, object]:
    state = request.app.state.runtime
    session_id = payload.session_id

    world = state.get_world(session_id)
    anchors = state.anchors.get(session_id, [])
    trace = state.traces.setdefault(session_id, [])

    add_trace_event(trace, "plan_start", {"session_id": session_id})

    best, ranked = search_scaffolds(world, anchors, state.policy, trace=trace)
    elements = best.elements

    elements, repair_meta = repair_elements(elements, world, state.policy, trace=trace)
    add_trace_event(trace, "plan_repair_meta", repair_meta)

    valid, violations = validate_all(elements, world, state.policy, trace=trace)
    add_trace_event(trace, "plan_validated_final", {"valid": bool(valid), "violations": violations})

    if not valid and bool(getattr(state.policy, "enforce_validators_strict", True)):
        raise HTTPException(
            status_code=409,
            detail={
                "status": "VALIDATION_FAILED",
                "best_candidate": {"id": best.candidate_id, "score": best.score, "violations": best.violations},
                "violations": violations,
            },
        )

    overlays = world.compute_overlays(state.policy.__dict__)
    env_mesh_bytes = world.export_env_mesh_obj_bytes() if hasattr(world, "export_env_mesh_obj_bytes") else world.export_env_mesh_obj()

    rev_id = state.store.lock_revision(
        session_id,
        world.serialize_state(),
        overlays,
        trace,
        env_mesh_bytes=env_mesh_bytes,
    )
    state.last_rev[session_id] = rev_id

    base = f"/sessions/{session_id}/world/{rev_id}"
    scene_bundle = {
        "session_id": session_id,
        "rev_id": rev_id,
        "env": {
            "mesh_obj_url": f"{base}/env_mesh.obj",
            "overlays_url": f"{base}/overlays.json",
            "world_state_url": f"{base}/world_state.json",
            "trace_url": f"{base}/trace.json",
        },
        "scaffold": {
            "elements": elements,
            "valid": bool(valid),
            "violations": violations,
            "best_candidate": {"id": best.candidate_id, "score": best.score, "params": best.params},
            "ranked_candidates_top5": [
                {"id": c.candidate_id, "score": c.score, "valid": c.valid, "violations": len(c.violations), "params": c.params}
                for c in ranked[:5]
            ],
        },
        "meta": {"policy": state.policy.__dict__},
    }

    state.store.save_export(session_id, rev_id, scene_bundle)
    add_trace_event(trace, "plan_export_saved", {"rev_id": rev_id})

    return {"status": "ok", "session_id": session_id, "rev_id": rev_id, "scene_bundle": scene_bundle}
