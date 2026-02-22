from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError

from api.ingest import ingest_frame
from contracts.frame_packet import AnchorPoint, FramePacketMeta
from policy.unknown_space import apply_unknown_policy
from scanning.readiness import compute_readiness, compute_readiness_metrics
from validation.frame_validation import validate_and_normalize_frame_meta
from world.mesh_export import env_mesh_obj_bytes

router = APIRouter(tags=["session"])


class AnchorPayload(BaseModel):
    session_id: str
    anchors: list[AnchorPoint]


class LockPayload(BaseModel):
    session_id: str
    selected_variant: str | None = None
    measurements_json: str | None = None
    manual_measurements: list[dict] = []


@router.post("/session/create")
def create_session(request: Request) -> dict:
    state = request.app.state.runtime
    session_id = state.store.create_session()
    state.get_world(session_id)
    state.get_scene_graph(session_id)
    state.anchors[session_id] = []
    state.traces[session_id] = []
    state.last_timestamp[session_id] = float("-inf")
    return {"session_id": session_id}


@router.get("/policy/status")
def policy_status(request: Request) -> dict:
    state = request.app.state.runtime
    return state.policy_status()


@router.post("/session/frame")
async def post_frame(
    request: Request,
    meta: UploadFile = File(...),
    rgb: UploadFile = File(...),
    depth: UploadFile | None = File(default=None),
    pointcloud: UploadFile | None = File(default=None),
) -> dict:
    state = request.app.state.runtime
    try:
        meta_payload = FramePacketMeta.model_validate_json(await meta.read())
    except ValidationError as exc:
        # Tests expect a structured detail object, not a string.
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID_META", "errors": exc.errors()},
        ) from exc

    depth_bytes = await depth.read() if depth else None
    pointcloud_bytes = await pointcloud.read() if pointcloud else None

    if meta_payload.depth_meta and depth_bytes is None:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "INVALID_FRAMEPACKET",
                "errors": [{"field": "depth", "code": "MISSING", "msg": "depth file is required when depth_meta is provided"}],
            },
        )
    if meta_payload.pointcloud_meta and pointcloud_bytes is None:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "INVALID_FRAMEPACKET",
                "errors": [
                    {"field": "pointcloud", "code": "MISSING", "msg": "pointcloud file is required when pointcloud_meta is provided"}
                ],
            },
        )

    # If neither depth nor pointcloud is present (and no meta declares them), reject early.
    if (depth_bytes is None) and (pointcloud_bytes is None) and (meta_payload.depth_meta is None) and (meta_payload.pointcloud_meta is None):
        raise HTTPException(
            status_code=400,
            detail={"status": "NEEDS_GEOMETRY", "msg": "either depth or pointcloud is required"},
        )

    session_id = meta_payload.session_id
    rgb_bytes = await rgb.read()
    meta_dict = meta_payload.model_dump()

    last_ts = state.last_timestamp.get(session_id)
    validated_meta, errors = validate_and_normalize_frame_meta(
        meta_dict,
        rgb_bytes=rgb_bytes,
        last_timestamp=last_ts,
    )
    if errors:
        raise HTTPException(status_code=400, detail={"status": "INVALID_FRAMEPACKET", "errors": errors})

    # Update monotonic timestamp state (STAGE A)
    state.last_timestamp[session_id] = float(validated_meta["timestamp"])

    return ingest_frame(
        state,
        session_id,
        meta_payload.frame_id,
        meta_dict,
        rgb_bytes,
        depth_bytes,
        pointcloud_bytes,
        validated_meta=validated_meta,
    )


@router.post("/session/anchors")
def post_anchors(request: Request, payload: AnchorPayload) -> dict:
    state = request.app.state.runtime
    anchors = [a.model_dump() for a in payload.anchors]
    state.anchors[payload.session_id] = anchors
    state.store.save_anchors(payload.session_id, anchors)
    return {"status": "ok", "count": len(anchors)}


@router.post("/session/lock")
def lock_session(request: Request, payload: LockPayload) -> dict:
    state = request.app.state.runtime
    world = state.get_world(payload.session_id)
    overlays = world.compute_overlays(state.policy.__dict__)
    env_mesh_bytes = env_mesh_obj_bytes(world)

    rev_id = state.store.lock_revision(
        payload.session_id,
        world.serialize_state(),
        overlays,
        state.traces.get(payload.session_id, []),
        env_mesh_bytes=env_mesh_bytes,
    )
    state.last_rev[payload.session_id] = rev_id
    state.last_rev_meta[payload.session_id] = {
        "selected_variant": payload.selected_variant,
        "measurements_json": payload.measurements_json,
        "manual_measurements": payload.manual_measurements,
    }
    return {
        "session_id": payload.session_id,
        "rev_id": rev_id,
        "env_mesh_present": bool(env_mesh_bytes),
        "trace_ndjson": f"sessions/{payload.session_id}/world/{rev_id}/trace.ndjson",
        "tsdf_available": bool(world.metrics.get("tsdf_available", True)),
        "tsdf_reason": world.metrics.get("tsdf_reason"),
    }


@router.get("/session/{session_id}/readiness")
def session_readiness(request: Request, session_id: str) -> dict:
    state = request.app.state.runtime
    world = state.get_world(session_id)
    anchors = state.anchors.get(session_id, [])
    ready, score, reasons = compute_readiness(world, anchors, state.policy)
    metrics = compute_readiness_metrics(world, anchors, state.policy)
    return {
        "session_id": session_id,
        "ready": ready,
        "score": score,
        "reasons": reasons,
        "readiness_metrics": metrics,
    }


@router.get("/session/{session_id}/status")
def session_status(request: Request, session_id: str) -> dict:
    state = request.app.state.runtime
    world = state.get_world(session_id)
    anchors = state.anchors.get(session_id, [])
    ready, score, reasons = compute_readiness(world, anchors, state.policy)
    readiness_metrics = compute_readiness_metrics(world, anchors, state.policy)
    unknown = apply_unknown_policy(world, anchors, state.policy)
    sg = state.get_scene_graph(session_id)
    return {
        "session_id": session_id,
        "ready": ready,
        "score": score,
        "reasons": reasons,
        "readiness_metrics": readiness_metrics,
        "metrics": world.serialize_state(),
        "unknown_policy": unknown,
        "perception_unavailable": state.perception_unavailable,
        "scene_graph": sg.serialize(),
        "geometry_unavailable": not bool(world.metrics.get("tsdf_available", True)),
        "geometry_reason": world.metrics.get("tsdf_reason"),
        "policy_source": getattr(state, "policy_source", None)
        or getattr(getattr(state, "config", None), "policy_source", None),
    }
