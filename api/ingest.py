from __future__ import annotations

from typing import Any

from inference.extend_linear_objects import propose_anchor_linear_hypotheses
from perception.scene_graph import update_scene_graph_from_world
from trace.decision_trace import add_trace_event


def ingest_frame(
    runtime,
    session_id: str,
    frame_id: str,
    meta_dict: dict[str, Any],
    rgb_bytes: bytes,
    depth_bytes: bytes | None,
    pointcloud_bytes: bytes | None,
    *,
    validated_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    world = runtime.get_world(session_id)
    sg = runtime.get_scene_graph(session_id)

    runtime.anchors.setdefault(session_id, [])
    runtime.traces.setdefault(session_id, [])

    runtime.store.save_frame(
        session_id,
        frame_id,
        meta_dict,
        rgb_bytes,
        depth_bytes,
        pointcloud_bytes,
        validated_meta=validated_meta,
    )

    # Always prefer validated_meta for geometry integration.
    meta_for_geom = dict(validated_meta or meta_dict)
    try:
        meta_for_geom["_tracking_cfg"] = getattr(runtime.config, "tracking").model_dump()
    except Exception:
        pass
    world.update_from_frame(meta_for_geom, rgb_bytes, depth_bytes, pointcloud_bytes)
    add_trace_event(runtime.traces[session_id], "frame_ingested", {"frame_id": frame_id})

    anchors = runtime.anchors.get(session_id, [])
    fit = update_scene_graph_from_world(sg, world, anchors=anchors, every_n_frames=5)
    if fit:
        add_trace_event(runtime.traces[session_id], "scene_graph_updated", fit)

    hypotheses, needs_scan = propose_anchor_linear_hypotheses(world, anchors, runtime.policy)
    if hypotheses:
        sg.meta["linear_hypotheses"] = hypotheses
        stop_counts: dict[str, int] = {}
        for h in hypotheses:
            sr = ((h.get("proposal") or {}).get("stop_reason")) or "UNKNOWN"
            stop_counts[str(sr)] = int(stop_counts.get(str(sr), 0)) + 1
        add_trace_event(
            runtime.traces[session_id],
            "linear_hypotheses_updated",
            {"count": len(hypotheses), "stop_reasons": stop_counts},
        )
    if needs_scan:
        sg.meta["needs_scan"] = needs_scan
        add_trace_event(runtime.traces[session_id], "needs_scan_hints", {"count": len(needs_scan)})

    return {
        "status": "ok",
        "session_id": session_id,
        "frame_id": frame_id,
        "frames": world.metrics["frames"],
        "scene_graph_objects": len(sg.objects),
    }
