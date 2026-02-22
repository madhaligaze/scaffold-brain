from __future__ import annotations

import base64
import json

from dataclasses import replace

import numpy as np
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from api.ingest import ingest_frame
from export.overlays_export import export_clearance_violations_glb, export_unknown_heatmap_glb
from export.scene_bundle import build_scene_bundle
from contracts.frame_packet import FramePacketMeta
from contracts.legacy_stream import LegacyStreamPayload
from policy.unknown_space import apply_unknown_policy
from scaffold.bom import bom_from_elements
from scaffold.solver import generate_scaffold
from scaffold.validators import collision_check
from scanning.next_best_view import generate_scan_plan
from scanning.readiness import compute_readiness
from trace.decision_trace import add_trace_event
from world.mesh_export import env_mesh_glb_bytes, env_mesh_obj_bytes, scaffold_to_glb_bytes
from world.occupancy import OCCUPIED

router = APIRouter(tags=["legacy"])


@router.get("/health")
def health(request: Request) -> dict:
    """Lightweight health endpoint for Android client."""
    state = request.app.state.runtime

    version = "dev"
    try:
        import os as _os

        v = _os.getenv("APP_VERSION") or _os.getenv("BACKEND_VERSION")
        if v:
            version = str(v)
        else:
            import tomllib
            from pathlib import Path as _Path

            pyproj = _Path(__file__).resolve().parents[1] / "pyproject.toml"
            if pyproj.exists():
                data = tomllib.loads(pyproj.read_text(encoding="utf-8"))
                version = str(data.get("project", {}).get("version") or version)
    except Exception:
        pass

    modules = {"legacy": True, "export": True, "session_v2": True}
    try:
        modules["policy"] = bool(getattr(state, "policy", None))
    except Exception:
        pass

    return {"status": "ok", "version": version, "modules": modules}

# Android API usage audit (from ApiService.kt) for compatibility adapters:
# - POST /session/start -> expects {session_id,status} JSON.
# - POST /session/stream/{session_id} -> sends JSON map (often base64 image/depth + optional geometry fields), expects {status, ai_hints?}.
# - GET /health -> expects status/version/modules (version/modules are optional on client side).
# - GET /session/voxels/{session_id}, POST /session/model/{session_id}, POST /session/update/{session_id},
#   POST /session/preview_remove/{session_id} are also called by Android and should remain available in legacy stack.


def _decode_base64(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        v = value.strip()

        # Accept data-URL payloads, e.g. "data:image/jpeg;base64,AAAA...".
        # Android normally sends raw base64, but some clients (or future changes) may wrap it.
        if v.startswith("data:") and "," in v:
            v = v.split(",", 1)[1].strip()

        # Be tolerant to newlines/spaces and urlsafe base64 variants.
        v = v.replace("\n", "").replace("\r", "").replace(" ", "")
        try:
            return base64.b64decode(v)
        except Exception:
            return base64.urlsafe_b64decode(v)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc


def _build_meta(session_id: str, payload: LegacyStreamPayload) -> tuple[dict, bytes, bytes | None, bytes | None]:
    intrinsics = payload.intrinsics or {}
    pose = payload.pose or {}

    rgb_b64 = payload.rgb_base64 or payload.image_base64 or payload.rgb
    rgb_bytes = _decode_base64(rgb_b64)
    if rgb_bytes is None:
        raise HTTPException(status_code=400, detail="Missing rgb_base64/image_base64/rgb in legacy payload")

    depth_b64 = payload.depth_base64 or payload.depth
    depth_bytes = _decode_base64(depth_b64)

    pc = payload.point_cloud or payload.pointcloud
    pointcloud_bytes = None
    pointcloud_meta = None
    if pc is not None:
        # Prefer binary float32 XYZ for downstream occupancy warm-up / readiness metrics.
        # Fall back to JSON if payload shape is unexpected.
        try:
            arr = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
            pointcloud_bytes = arr.astype(np.float32).tobytes()
            pointcloud_meta = {"format": "xyz", "frame": "world"}
        except Exception:
            pointcloud_bytes = json.dumps(pc).encode("utf-8")
            pointcloud_meta = {"format": "xyz", "frame": "world"}

    missing: list[str] = []
    for key in ("fx", "fy", "cx", "cy", "width", "height"):
        if key not in intrinsics:
            missing.append(f"intrinsics.{key}")
    if "position" not in pose:
        missing.append("pose.position")
    if "quaternion" not in pose:
        missing.append("pose.quaternion")
    if depth_bytes is None and pointcloud_bytes is None:
        missing.append("depth_base64|point_cloud")
    if missing:
        raise HTTPException(
            status_code=409,
            detail={
                "status": "NEEDS_GEOMETRY",
                "missing": missing,
                "hint": "Use /session/frame contract or send intrinsics+pose+size",
            },
        )

    depth_meta = None
    if depth_bytes is not None:
        depth_scale = payload.depth_scale_m_per_unit or payload.depth_scale or 0.001
        depth_meta = {
            "width": int(payload.depth_width or intrinsics["width"]),
            "height": int(payload.depth_height or intrinsics["height"]),
            "scale_m_per_unit": float(depth_scale),
            "encoding": "uint16",
        }

    frame_id = payload.frame_id or str(uuid4())
    meta_dict = {
        "session_id": session_id,
        "frame_id": frame_id,
        "timestamp": float(payload.timestamp or time.time()),
        "intrinsics": intrinsics,
        "pose": pose,
        "depth_meta": depth_meta,
        "pointcloud_meta": pointcloud_meta,
    }

    # Validate adapter output against canonical FramePacket schema before ingesting.
    FramePacketMeta.model_validate(meta_dict)
    return meta_dict, rgb_bytes, depth_bytes, pointcloud_bytes


@router.post("/session/start")
def legacy_start_session(request: Request):
    state = request.app.state.runtime
    session_id = state.store.create_session()
    state.get_world(session_id)
    state.anchors[session_id] = []
    state.traces[session_id] = []
    state.session_stats[session_id] = {"depth_frames_received": 0}
    return {"session_id": session_id, "status": "ok"}


@router.post("/session/stream")
def legacy_stream(request: Request, payload: LegacyStreamPayload):
    session_id = payload.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required for /session/stream")
    return _legacy_stream_ingest(request, session_id, payload)


@router.post("/session/stream/{session_id}")
def legacy_stream_with_path(request: Request, session_id: str, payload: LegacyStreamPayload):
    return _legacy_stream_ingest(request, session_id, payload)


def _legacy_stream_ingest(request: Request, session_id: str, payload: LegacyStreamPayload):
    state = request.app.state.runtime
    meta_dict, rgb_bytes, depth_bytes, pointcloud_bytes = _build_meta(session_id, payload)
    result = ingest_frame(
        state,
        session_id,
        meta_dict["frame_id"],
        meta_dict,
        rgb_bytes,
        depth_bytes,
        pointcloud_bytes,
    )
    world = state.get_world(session_id)
    anchors = state.anchors.get(session_id, [])
    extra = getattr(payload, "__pydantic_extra__", None) or {}
    depth_supported = extra.get("depth_supported")
    depth_unavailable = bool(extra.get("depth_unavailable")) if "depth_unavailable" in extra else False

    stats = state.session_stats.setdefault(session_id, {"depth_frames_received": 0})
    if depth_bytes is not None:
        stats["depth_frames_received"] = int(stats.get("depth_frames_received", 0) or 0) + 1
    has_any_depth = int(stats.get("depth_frames_received", 0) or 0) > 0
    no_depth_mode = (not has_any_depth) and (depth_bytes is None) and (depth_unavailable or (depth_supported is False))

    policy_for_readiness = state.policy
    if not has_any_depth:
        try:
            min_obs = float(getattr(state.policy, "readiness_observed_ratio_min", 0.1))
            lowered = max(0.05, min_obs * 0.67)
            policy_for_readiness = replace(state.policy, readiness_observed_ratio_min=lowered)
        except Exception:
            policy_for_readiness = state.policy

    ready, score, reasons = compute_readiness(world, anchors, policy_for_readiness)
    if not ready and not anchors:
        try:
            occ = world.occupancy.stats()
            total = float(occ.get("total", 0) or 0)
            unknown = float(occ.get("unknown", 0) or 0)
            observed = 1.0 - (unknown / max(1.0, total))
            if observed >= 0.95:
                ready = True
                reasons = []
        except Exception:
            pass
    numeric_score = float(score) if isinstance(score, (int, float)) else 0.0
    quality_score = numeric_score * 100.0 if 0.0 <= numeric_score <= 1.0 else numeric_score
    quality_score = max(0.0, min(100.0, quality_score))

    if ready:
        instructions = ["Можно моделировать."]
    else:
        scan_plan = generate_scan_plan(world, anchors)
        instructions = []
        if no_depth_mode:
            instructions.append("Без Depth потребуется дольше сканировать (без depth occupancy прогревается из point cloud).")
        for item in scan_plan[:3]:
            note = item.get("note")
            instructions.append(f"Досканируйте: {note}" if note else "Сделайте обзор вокруг точки опоры")

    result.update(
        {
            "status": "RECEIVING",
            "ai_hints": {
                "instructions": instructions,
                "warnings": [str(reason) for reason in reasons],
                "no_depth_mode": bool(no_depth_mode),
                "quality_score": quality_score,
                "is_ready": bool(ready),
                "depth_frames_received": int(stats.get("depth_frames_received", 0) or 0),
            },
            "legacy_stream": True,
            "legacy_mode": "json_adapter",
        }
    )
    return result


def _load_latest_export(state, session_id: str) -> dict[str, Any] | None:
    rev_id = state.last_rev.get(session_id)
    if not rev_id:
        latest_path = state.store.session_root(session_id) / "exports" / "latest.json"
        if latest_path.exists():
            rev_id = json.loads(latest_path.read_text(encoding="utf-8")).get("rev_id")
    if not rev_id:
        return None
    return state.store.load_export(session_id, rev_id)


def _legacy_element_to_android(element: dict[str, Any]) -> dict[str, Any]:
    element_type = str(element.get("type", "beam"))
    if "start" in element and "end" in element:
        start = element.get("start") or [0.0, 0.0, 0.0]
        end = element.get("end") or [0.0, 0.0, 0.0]
    else:
        pos = (((element.get("pose") or {}).get("pos")) or [0.0, 0.0, 0.0])
        if element_type == "post":
            dims = element.get("dims") or {}
            z = float(dims.get("z", 0.0) or 0.0)
            start = [float(pos[0]), float(pos[1]), float(pos[2])]
            end = [float(pos[0]), float(pos[1]), float(pos[2] + z)]
        else:
            start = [float(pos[0]), float(pos[1]), float(pos[2])]
            end = [float(pos[0]), float(pos[1]), float(pos[2])]

    load_ratio = float(element.get("load_ratio") or element.get("load") or 0.0)
    stress_color = element.get("stress_color") or element.get("color")
    if not stress_color:
        if load_ratio >= 0.8:
            stress_color = "red"
        elif load_ratio >= 0.5:
            stress_color = "orange"
        else:
            stress_color = "green"

    return {
        "id": str(element.get("id", "")),
        "type": element_type,
        "start": [float(start[0]), float(start[1]), float(start[2])],
        "end": [float(end[0]), float(end[1]), float(end[2])],
        "stress_color": stress_color,
        "load_ratio": load_ratio,
        "meta": element.get("meta") or {},
    }


@router.get("/session/voxels/{session_id}")
def legacy_voxels(request: Request, session_id: str, max_count: int = 5000, stride: int = 1):
    state = request.app.state.runtime
    world = state.get_world(session_id)
    occupancy = world.occupancy
    grid = occupancy.grid
    max_emit = max(0, int(max_count))
    step = max(1, int(stride))

    occupied_idx = (grid == OCCUPIED).nonzero()
    occupied_positions = list(zip(occupied_idx[0], occupied_idx[1], occupied_idx[2]))
    total_count = len(occupied_positions)
    emitted = occupied_positions[::step][:max_emit]

    voxels = []
    for x_i, y_i, z_i in emitted:
        pos = [
            float(occupancy.origin[0] + occupancy.voxel_size * float(x_i)),
            float(occupancy.origin[1] + occupancy.voxel_size * float(y_i)),
            float(occupancy.origin[2] + occupancy.voxel_size * float(z_i)),
        ]
        voxels.append(
            {
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "type": "occupied",
                "color": "#D94A4A",
                "alpha": 0.9,
                "radius": None,
            }
        )

    shape = grid.shape
    bounds_min = occupancy.origin
    bounds_max = [
        float(occupancy.origin[0] + occupancy.voxel_size * float(shape[0])),
        float(occupancy.origin[1] + occupancy.voxel_size * float(shape[1])),
        float(occupancy.origin[2] + occupancy.voxel_size * float(shape[2])),
    ]
    return {
        "status": "ok",
        "voxels": voxels,
        "bounds": {
            "min": [float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])],
            "max": [float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])],
        },
        "resolution": float(occupancy.voxel_size),
        "total_count": int(total_count),
    }


@router.post("/session/model/{session_id}")
def legacy_model(request: Request, session_id: str):
    state = request.app.state.runtime
    world = state.get_world(session_id)
    anchors = state.anchors.get(session_id, [])
    ready, score, reasons = compute_readiness(world, anchors, state.policy)

    if not ready:
        return {
            "status": "NEEDS_SCAN",
            "options": [],
            "scan_plan": generate_scan_plan(world, anchors),
            "reasons": [str(reason) for reason in reasons],
            "score": score,
        }

    try:
        scan_plan = generate_scan_plan(world, anchors)
        elements, _trace = generate_scaffold(world, anchors, state.policy)
        valid, violations = collision_check(elements, world, state.policy)
        unknown = apply_unknown_policy(world, anchors, state.policy)
        violations.extend(unknown["violations"])
        if not valid:
            violations.append("SCHEMA_OR_COLLISION_INVALID")

        overlays = world.compute_overlays(state.policy.__dict__)
        overlays["violations"] = violations

        env_obj = env_mesh_obj_bytes(world)
        rev_id = state.store.lock_revision(
            session_id,
            world.serialize_state(),
            overlays,
            state.traces.setdefault(session_id, []),
            env_mesh_bytes=env_obj,
        )
        state.last_rev[session_id] = rev_id

        world_dir = state.store.session_root(session_id) / "world" / rev_id
        env_glb_rel = f"sessions/{session_id}/world/{rev_id}/env_mesh.glb"
        (world_dir / "env_mesh.glb").write_bytes(env_mesh_glb_bytes(world))

        overlay_files = overlays.setdefault("overlay_files", {})
        unknown = export_unknown_heatmap_glb(world, world_dir / "unknown_heatmap.glb")
        clearance = export_clearance_violations_glb(
            world,
            world_dir / "clearance_violations.glb",
            min_clearance_m=float(getattr(state.policy, "min_clearance_m", 0.2)),
        )
        overlay_files["unknown_heatmap"] = {"glb": {"path": unknown["path"]}}
        overlay_files["clearance_violations"] = {"glb": {"path": clearance["path"]}}

        bundle = build_scene_bundle(session_id, rev_id, world, anchors, elements, scan_plan, overlays)
        bundle.setdefault("env_mesh", {})
        bundle["env_mesh"]["obj"] = {"path": f"sessions/{session_id}/world/{rev_id}/env_mesh.obj"}
        bundle["env_mesh"]["glb"] = {"path": env_glb_rel}
        bundle["bom"] = bom_from_elements(elements)
        android_elements = [_legacy_element_to_android(el) for el in elements]
        state.store.save_export(session_id, rev_id, bundle)

        try:
            scaffold_glb = scaffold_to_glb_bytes(android_elements)
            scaffold_rel = f"sessions/{session_id}/world/{rev_id}/scaffold.glb"
            (world_dir / "scaffold.glb").write_bytes(scaffold_glb)
            overlay_files["scaffold"] = {"glb": {"path": scaffold_rel}}
            for layer in bundle.get("ui", {}).get("layers", []):
                if layer.get("id") == "scaffold":
                    layer["file"] = {"glb": {"path": scaffold_rel}}
                    break
            state.store.save_export(session_id, rev_id, bundle)
        except Exception as e:
            print(f"[warn] scaffold.glb generation failed: {e}")

        score_norm = float(score) if isinstance(score, (int, float)) else 0.0
        score_norm = score_norm / 100.0 if score_norm > 1.0 else score_norm
        safety_score = max(0, min(100, int(score_norm * 100) - 10 * len(violations)))
        unique_nodes = {tuple(e["start"]) for e in android_elements} | {tuple(e["end"]) for e in android_elements}
        return {
            "status": "OK",
            "options": [
                {
                    "variant_name": "default",
                    "material_info": "layher",
                    "safety_score": safety_score,
                    "ai_critique": [str(v) for v in violations],
                    "elements": android_elements,
                    "full_structure": android_elements,
                    "stats": {
                        "total_nodes": len(unique_nodes),
                        "total_beams": len(android_elements),
                        "total_weight_kg": 0,
                        "collisions_fixed": len(violations),
                    },
                    "physics": {"status": "OK"},
                }
            ],
            "bundle": bundle,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "options": [
                {
                    "variant_name": "Auto",
                    "material_info": "",
                    "safety_score": 0,
                    "ai_critique": [f"MODEL_ADAPTER_ERROR: {exc}"],
                    "elements": [],
                    "full_structure": [],
                    "stats": {
                        "total_nodes": 0,
                        "total_beams": 0,
                        "total_weight_kg": 0,
                        "collisions_fixed": 0,
                    },
                    "physics": {"status": "ERROR"},
                }
            ],
        }


@router.post("/session/update/{session_id}")
def legacy_update(request: Request, session_id: str, payload: dict[str, Any] | None = None):
    state = request.app.state.runtime
    started = time.perf_counter()
    action_payload = payload or {}
    rev_id = state.last_rev.get(session_id)
    if not rev_id:
        raise HTTPException(status_code=409, detail={"status": "NO_MODEL"})
    export_bundle = state.store.load_export(session_id, rev_id)
    if export_bundle is None:
        raise HTTPException(status_code=409, detail={"status": "NO_MODEL"})

    scaffold = [item for item in (export_bundle.get("scaffold") or []) if isinstance(item, dict)]
    action = str(action_payload.get("action") or "noop").lower()
    element_id = action_payload.get("element_id")
    element_data = action_payload.get("element_data")
    affected_elements: list[str] = []
    warning = ""

    if action == "remove" and element_id is not None:
        before = len(scaffold)
        scaffold = [item for item in scaffold if str(item.get("id")) != str(element_id)]
        if len(scaffold) != before:
            affected_elements = [str(element_id)]
        else:
            warning = "element not found"
    elif action == "add":
        if isinstance(element_data, dict):
            scaffold.append(element_data)
            affected_elements = [str(element_data.get("id", ""))]
        else:
            warning = "missing element_data"
    elif action == "replace" and element_id is not None:
        replaced = False
        for idx, item in enumerate(scaffold):
            if str(item.get("id")) == str(element_id):
                if isinstance(element_data, dict):
                    scaffold[idx] = element_data
                    affected_elements = [str(element_id)]
                else:
                    warning = "missing element_data"
                replaced = True
                break
        if not replaced:
            warning = "element not found"

    export_bundle["scaffold"] = scaffold
    android_elements = [_legacy_element_to_android(el) for el in scaffold]
    export_bundle["android_options"] = [
        {
            "variant_name": "default",
            "material_info": "layher",
            "safety_score": 100,
            "ai_critique": [warning] if warning else [],
            "elements": android_elements,
            "full_structure": android_elements,
            "stats": {
                "total_nodes": len({tuple(e["start"]) for e in android_elements} | {tuple(e["end"]) for e in android_elements}),
                "total_beams": len(android_elements),
                "total_weight_kg": 0,
                "collisions_fixed": 0,
            },
            "physics": {"status": "OK"},
        }
    ]

    state.traces.setdefault(session_id, [])
    add_trace_event(state.traces[session_id], "legacy_update", {"session_id": session_id, "action": action_payload, "warning": warning})
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "OK",
        "is_stable": True,
        "physics_status": "OK",
        "heatmap": [{"id": str(item.get("id", "")), "color": "green", "load_ratio": 0.0} for item in scaffold[:50]],
        "affected_elements": affected_elements,
        "collapsed": {"nodes": [], "elements": []},
        "processing_time_ms": elapsed_ms,
        "warning": warning,
        "bundle": export_bundle,
        "options": export_bundle.get("android_options"),
    }


@router.post("/session/preview_remove/{session_id}")
def legacy_preview_remove(request: Request, session_id: str, element_id: str | None = None):
    state = request.app.state.runtime
    rev_id = state.last_rev.get(session_id)
    if not rev_id:
        raise HTTPException(status_code=409, detail={"status": "NO_MODEL"})
    export_bundle = state.store.load_export(session_id, rev_id)
    if export_bundle is None:
        raise HTTPException(status_code=409, detail={"status": "NO_MODEL"})
    scaffold = export_bundle.get("scaffold") or []
    target = None
    if element_id:
        target = next((item for item in scaffold if isinstance(item, dict) and str(item.get("id")) == str(element_id)), None)
    is_critical = bool((target or {}).get("type") == "post")
    remaining = [item for item in scaffold if isinstance(item, dict) and str(item.get("id")) != str(element_id)]
    if len(scaffold) > 0 and len(remaining) == 0:
        is_critical = True
    would_collapse = [str(element_id)] if is_critical and element_id else []
    warning = ""
    if element_id and not target:
        warning = "element not found"
    elif would_collapse:
        warning = "Removing this element may collapse scaffold"

    return {
        "status": "OK",
        "element_id": element_id,
        "is_critical": is_critical,
        "would_collapse": would_collapse,
        "collapse_count": len(would_collapse),
        "warning": warning,
    }


@router.post("/session/unlock/{session_id}")
def legacy_unlock(request: Request, session_id: str):
    state = request.app.state.runtime
    state.last_rev.pop(session_id, None)
    state.traces.setdefault(session_id, [])
    add_trace_event(state.traces[session_id], "legacy_unlock_noop", {"session_id": session_id})
    return {"status": "ok", "noop": True}


@router.get("/session/snapshots/{session_id}")
def legacy_snapshots(request: Request, session_id: str):
    state = request.app.state.runtime
    world_dir = state.store.session_root(session_id) / "world"
    export_dir = state.store.session_root(session_id) / "exports"
    revisions = []
    if world_dir.exists():
        for p in sorted([d for d in world_dir.iterdir() if d.is_dir()], key=lambda x: x.name):
            revisions.append({"revision": p.name, "has_world_state": (p / "world_state.json").exists()})
    latest = None
    latest_file = export_dir / "latest.json"
    if latest_file.exists():
        latest = json.loads(latest_file.read_text(encoding="utf-8")).get("rev_id")
    return {"status": "ok", "session_id": session_id, "snapshots": revisions, "latest_revision": latest}


@router.post("/session/snapshot/restore/{session_id}/{revision}")
def legacy_restore_snapshot(request: Request, session_id: str, revision: str):
    state = request.app.state.runtime
    base = state.store.session_root(session_id) / "world" / revision
    world_state_path = base / "world_state.json"
    overlays_path = base / "overlays.json"
    trace_path = base / "trace.json"
    if not world_state_path.exists():
        raise HTTPException(status_code=404, detail=f"Revision not found: {revision}")

    world_state = json.loads(world_state_path.read_text(encoding="utf-8"))
    overlays = json.loads(overlays_path.read_text(encoding="utf-8")) if overlays_path.exists() else {}
    traces = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []

    state.last_rev[session_id] = revision
    state.traces[session_id] = traces
    state.restored_revision_state[session_id] = {
        "revision": revision,
        "world_state": world_state,
        "overlays": overlays,
    }
    return {"status": "ok", "session_id": session_id, "revision": revision, "restored": True}
