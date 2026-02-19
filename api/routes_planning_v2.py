from __future__ import annotations

from typing import Any

import trimesh
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from export.overlays_export import export_clearance_violations_glb, export_unknown_heatmap_glb
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
from world.mesh_export import env_mesh_obj_bytes

router = APIRouter(tags=["planning"])


class ScaffoldRequest(BaseModel):
    session_id: str


def _obj_bytes_to_glb(obj_bytes: bytes) -> bytes:
    try:
        obj_txt = obj_bytes.decode("utf-8", errors="ignore")
        mesh = trimesh.load_mesh(trimesh.util.wrap_as_stream(obj_txt.encode("utf-8")), file_type="obj")
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="environment_mesh")
        return scene.export(file_type="glb")
    except Exception:
        return trimesh.Scene().export(file_type="glb")


@router.post("/planning/request_scaffold")
def request_scaffold(request: Request, payload: ScaffoldRequest) -> dict[str, Any]:
    state = request.app.state.runtime
    sid = payload.session_id
    world = state.get_world(sid)
    anchors = state.anchors.get(sid, [])

    # Gate on tracking quality if present
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
def plan_scaffold(request: Request, payload: PlanPayload) -> dict[str, Any]:
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
    env_mesh_bytes = env_mesh_obj_bytes(world)
    env_mesh_glb = _obj_bytes_to_glb(env_mesh_bytes)

    rev_id = state.store.lock_revision(
        session_id,
        world.serialize_state(),
        overlays,
        trace,
        env_mesh_bytes=env_mesh_bytes,
    )
    state.last_rev[session_id] = rev_id

    world_dir = state.store.session_root(session_id) / "world" / rev_id
    try:
        world_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    (world_dir / "env_mesh.glb").write_bytes(env_mesh_glb)

    min_clearance = float(getattr(state.policy, "min_clearance_m", 0.2))
    unknown_glb = export_unknown_heatmap_glb(world, world_dir / "unknown_heatmap.glb", stride=2, max_voxels=25000)
    clearance_glb = export_clearance_violations_glb(
        world,
        world_dir / "clearance_violations.glb",
        min_clearance_m=min_clearance,
        stride=2,
        max_voxels=25000,
    )

    base_world = f"/sessions/{session_id}/world/{rev_id}"
    base_exports = f"/sessions/{session_id}/exports/{rev_id}"
    scene_bundle = {
        "session_id": session_id,
        "rev_id": rev_id,
        "env": {
            "mesh_obj_url": f"{base_world}/env_mesh.obj",
            "mesh_glb_url": f"{base_world}/env_mesh.glb",
            "overlays_url": f"{base_world}/overlays.json",
            "world_state_url": f"{base_world}/world_state.json",
            "trace_url": f"{base_world}/trace.json",
            "trace_ndjson_url": f"{base_world}/trace.ndjson",
        },
        "ui": {
            "bundle_version": "1.2",
            "layers": [
                {
                    "id": "environment_mesh",
                    "label": "Environment mesh",
                    "kind": "mesh",
                    "default_on": True,
                    "file": {"glb": {"path": f"{base_world}/env_mesh.glb"}},
                },
                {"id": "scaffold", "label": "Scaffold", "kind": "model", "default_on": True},
                {
                    "id": "unknown_heatmap",
                    "label": "Unknown space",
                    "kind": "overlay",
                    "default_on": False,
                    "file": {"glb": {"path": f"{base_world}/unknown_heatmap.glb"}},
                },
                {
                    "id": "clearance_violations",
                    "label": "Clearance violations",
                    "kind": "overlay",
                    "default_on": True,
                    "file": {"glb": {"path": f"{base_world}/clearance_violations.glb"}},
                },
                {
                    "id": "trace",
                    "label": "Decision trace",
                    "kind": "debug",
                    "default_on": False,
                    "file": {"ndjson": {"path": f"{base_world}/trace.ndjson"}},
                },
            ],
        },
        "scaffold": {
            "elements": elements,
            "valid": bool(valid),
            "violations": violations,
            "best_candidate": {"id": best.candidate_id, "score": best.score, "params": best.params},
            "ranked_candidates_top5": [
                {
                    "id": c.candidate_id,
                    "score": c.score,
                    "valid": c.valid,
                    "violations": len(c.violations),
                    "params": c.params,
                }
                for c in ranked[:5]
            ],
        },
        "overlay_files": {
            "unknown_heatmap": {
                "glb": {
                    "path": f"{base_world}/unknown_heatmap.glb",
                    "meta": {"count": unknown_glb.get("count"), "stride": unknown_glb.get("stride")},
                }
            },
            "clearance_violations": {
                "glb": {
                    "path": f"{base_world}/clearance_violations.glb",
                    "meta": {
                        "count": clearance_glb.get("count"),
                        "stride": clearance_glb.get("stride"),
                        "min_clearance_m": clearance_glb.get("min_clearance_m"),
                    },
                }
            },
        },
        "meta": {"policy": state.policy.__dict__},
    }

    state.store.save_export(session_id, rev_id, scene_bundle)
    add_trace_event(trace, "plan_export_saved", {"rev_id": rev_id, "bundle_path": f"{base_exports}/scene_bundle.json"})

    return {"status": "ok", "session_id": session_id, "rev_id": rev_id, "scene_bundle": scene_bundle}
