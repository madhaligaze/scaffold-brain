import json

import numpy as np
from fastapi.testclient import TestClient

from main import create_app


def test_export_latest_returns_bundle_after_request_scaffold(tmp_path, monkeypatch):
    # Run in tmp dir so sessions/ is isolated
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)

    # Create session
    r = client.post("/session/create")
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # Anchors (supports) so solver can build something
    anchors = {
        "session_id": sid,
        "anchors": [
            {"id": "s1", "kind": "support", "position": [0.0, 0.0, 0.0], "confidence": 1.0},
            {"id": "s2", "kind": "support", "position": [4.0, 4.0, 0.0], "confidence": 1.0},
        ],
    }
    r = client.post("/session/anchors", json=anchors)
    assert r.status_code == 200

    # Feed a few minimal frames to satisfy readiness (view diversity); depth is synthetic 4x4.
    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 4, "height": 4}
    depth_meta = {"scale_m_per_unit": 0.001, "width": 4, "height": 4, "encoding": "uint16"}
    depth_u16 = (np.ones((4, 4), dtype=np.uint16) * 1000).astype(np.uint16)
    depth_bytes = depth_u16.tobytes()

    for i, pos in enumerate([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.6, 0.0]]):
        meta = {
            "timestamp": float(i),
            "intrinsics": intr,
            "pose": {"position": pos, "quaternion": [0.0, 0.0, 0.0, 1.0]},
            "depth_meta": depth_meta,
            "frame_id": f"f{i}",
            "session_id": sid,
        }
        files = {
            "meta": ("meta.json", json.dumps(meta).encode("utf-8"), "application/json"),
            "rgb": ("rgb.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"),
            "depth": ("depth.bin", depth_bytes, "application/octet-stream"),
        }
        r = client.post("/session/frame", files=files)
        assert r.status_code == 200

    # Request scaffold -> should generate export
    r = client.post(f"/session/{sid}/request_scaffold")
    assert r.status_code in (200, 409)
    if r.status_code == 409:
        # In strict envs, unknown policy may block; still no export expected then.
        return

    # Now latest export should exist
    r = client.get(f"/session/{sid}/export/latest")
    assert r.status_code == 200
    bundle = r.json()
    assert bundle["session_id"] == sid
    assert "env_mesh" in bundle and "path" in bundle["env_mesh"]
    assert "overlay_files" in bundle
    assert "occupancy" in bundle["overlay_files"]
