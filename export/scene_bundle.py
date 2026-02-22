from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from export.overlays_export import export_occupancy_npz, export_occupancy_slice_png

BUNDLE_VERSION = "1.1"


def _layers(overlays: dict[str, Any]) -> dict[str, Any]:
    files = overlays.get("overlay_files") or {}
    layers = [
        {"id": "environment_mesh", "label": "Environment mesh", "kind": "mesh", "default_on": True},
        {"id": "scaffold", "label": "Scaffold", "kind": "model", "default_on": True, "file": files.get("scaffold")},
        {"id": "unknown_heatmap", "label": "Unknown space", "kind": "overlay", "default_on": False, "file": files.get("unknown_heatmap")},
        {"id": "clearance_violations", "label": "Clearance violations", "kind": "overlay", "default_on": True, "file": files.get("clearance_violations")},
        {"id": "scan_hints", "label": "Scan hints", "kind": "overlay", "default_on": True, "file": files.get("scan_hints")},
        {"id": "trace", "label": "Decision trace", "kind": "debug", "default_on": False, "file": files.get("trace_ndjson")},
    ]
    for l in layers:
        if "file" in l and isinstance(l["file"], str):
            l["file"] = {"glb": {"path": l["file"]}}
        if "file" in l and not l["file"]:
            l.pop("file", None)
    return {"bundle_version": BUNDLE_VERSION, "layers": layers}


def build_scene_bundle(session_id: str, rev_id: str, world_model, anchors, scaffold, scan_plan, overlays, scene_graph=None) -> dict:
    objects = []
    meta = {}
    if scene_graph is not None:
        payload = scene_graph.serialize()
        objects = payload.get("objects", [])
        meta = payload.get("meta", {})

    world_dir = Path("sessions") / session_id / "world" / rev_id
    occ_npz = export_occupancy_npz(world_model, world_dir / "occupancy.npz")
    occ_png = export_occupancy_slice_png(world_model, world_dir / "occupancy_z.png", axis="z", frac=0.2)

    bundle = {
        "session_id": session_id,
        "revision_id": rev_id,
        "timestamp": time.time(),
        "env_mesh": {"obj": {"path": f"sessions/{session_id}/world/{rev_id}/env_mesh.obj"}, "glb": {"path": f"sessions/{session_id}/world/{rev_id}/env_mesh.glb"}},
        "trace": {"format": "ndjson", "path": f"sessions/{session_id}/world/{rev_id}/trace.ndjson", "json_path": f"sessions/{session_id}/world/{rev_id}/trace.json"},
        "objects": objects,
        "scene_meta": meta,
        "scaffold": scaffold,
        "anchors": anchors,
        "overlays": overlays,
        "scan_hints": scan_plan,
        "world": world_model.serialize_state(),
        "overlay_files": {
            "occupancy": {"npz": {"path": str(occ_npz["path"]).replace("\\", "/")}},
            "occupancy_slice": ({"png": {"path": str(occ_png["path"]).replace("\\", "/")}} if occ_png else {"png": None}),
        },
    }
    bundle.setdefault("bundle_version", BUNDLE_VERSION)
    bundle.setdefault("ui", {})
    bundle["ui"].update(_layers(overlays))
    if scene_graph is not None:
        try:
            bundle["scene_graph"] = scene_graph.serialize()
        except Exception:
            bundle["scene_graph"] = None
    return bundle
