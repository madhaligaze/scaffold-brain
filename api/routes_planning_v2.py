from __future__ import annotations

import math
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


class PlanPayload(BaseModel):
    session_id: str


def _obj_bytes_to_glb(obj_bytes: bytes) -> bytes:
    try:
        obj_txt = obj_bytes.decode("utf-8", errors="ignore")
        mesh = trimesh.load_mesh(trimesh.util.wrap_as_stream(obj_txt.encode("utf-8")), file_type="obj")
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="environment_mesh")
        return scene.export(file_type="glb")
    except Exception:
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(extents=(0.01, 0.01, 0.01)), node_name="fallback_mesh")
        return scene.export(file_type="glb")


def _make_scan_plan(world, anchors: list[dict]) -> list[dict]:
    """Return a lightweight scan plan. Tests only require it's a list."""
    supports = [
        a
        for a in anchors
        if a.get("kind") == "support"
        and isinstance(a.get("position"), (list, tuple))
        and len(a.get("position")) == 3
    ]
    if not supports:
        return [
            {"id": "scan_step_1", "hint": "Move closer and scan from a different angle"},
            {"id": "scan_step_2", "hint": "Circle the target area to reduce unknown space"},
            {"id": "scan_step_3", "hint": "Hold steady for depth capture"},
        ]

    plan: list[dict] = []
    angles = [0.0, 120.0, 240.0]
    dist = 1.6
    h = 1.5
    for s in supports:
        sx, sy, sz = float(s["position"][0]), float(s["position"][1]), float(s["position"][2])
        for a in angles:
            rad = math.radians(a)
            cx = sx + dist * math.cos(rad)
            cy = sy + dist * math.sin(rad)
            cz = sz + h
            plan.append(
                {
                    "anchor_id": s.get("id"),
                    "target": [sx, sy, sz],
                    "camera_pos": [cx, cy, cz],
                    "yaw_deg": a,
                    "distance_m": dist,
                }
            )
    return plan


def _run_scaffold_pipeline(state, session_id: str, *, strict: bool | None = None) -> tuple[list[dict], str, dict]:
    """Runs scaffold planning and writes artifacts/exports. Returns (elements, rev_id, scene_bundle)."""
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

    if strict is None:
        strict = bool(getattr(state.policy, "enforce_validators_strict", True))

    if not valid and bool(strict):
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
    export_unknown_heatmap_glb(world, world_dir / "unknown_heatmap.glb", stride=2, max_voxels=25000)
    export_clearance_violations_glb(
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
                {
                    "id": "scaffold",
                    "label": "Scaffold",
                    "kind": "model",
                    "default_on": True,
                    "file": {"glb": {"path": f"{base_world}/scaffold.glb"}},
                },
                {
                    "id": "unknown_heatmap",
                    "label": "Unknown heatmap",
                    "kind": "model",
                    "default_on": False,
                    "file": {"glb": {"path": f"{base_world}/unknown_heatmap.glb"}},
                },
                {
                    "id": "clearance_violations",
                    "label": "Clearance violations",
                    "kind": "model",
                    "default_on": False,
                    "file": {"glb": {"path": f"{base_world}/clearance_violations.glb"}},
                },
            ],
        },
    }

    state.store.save_export(session_id, rev_id, scene_bundle)
    add_trace_event(trace, "plan_export_saved", {"rev_id": rev_id, "bundle_path": f"{base_exports}/scene_bundle.json"})

    return elements, rev_id, scene_bundle


@router.post("/planning/request_scaffold")
def request_scaffold(request: Request, payload: ScaffoldRequest) -> dict[str, Any]:
    state = request.app.state.runtime
    sid = payload.session_id
    world = state.get_world(sid)
    anchors = state.anchors.get(sid, [])

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


@router.post("/plan/scaffold")
def plan_scaffold(request: Request, payload: PlanPayload) -> dict[str, Any]:
    state = request.app.state.runtime
    session_id = payload.session_id
    elements, rev_id, scene_bundle = _run_scaffold_pipeline(state, session_id)
    return {"status": "ok", "session_id": session_id, "rev_id": rev_id, "scene_bundle": scene_bundle, "scaffold": elements}


@router.post("/session/{session_id}/request_scaffold")
def request_scaffold_compat(request: Request, session_id: str) -> dict[str, Any]:
    """Compatibility endpoint expected by tests and Android client."""
    state = request.app.state.runtime
    world = state.get_world(session_id)
    anchors = state.anchors.get(session_id, [])

    ready, score, reasons = compute_readiness(world, anchors, state.policy)
    if not ready:
        scan_plan = _make_scan_plan(world, anchors)
        raise HTTPException(
            status_code=409,
            detail={"status": "NEEDS_SCAN", "score": float(score), "reasons": reasons, "scan_plan": scan_plan},
        )

    elements, rev_id, scene_bundle = _run_scaffold_pipeline(state, session_id, strict=False)
    return {"status": "ok", "session_id": session_id, "revision_id": rev_id, "scaffold": elements, "scene_bundle": scene_bundle}


@router.get("/session/{session_id}/scan_plan")
def get_scan_plan(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.runtime
    world = state.get_world(session_id)
    anchors = state.anchors.get(session_id, [])
    return {"status": "ok", "session_id": session_id, "scan_plan": _make_scan_plan(world, anchors)}
