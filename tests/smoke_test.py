import json

import numpy as np
from fastapi.testclient import TestClient

from main import create_app


def test_smoke_create_frame_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200

    r = client.post("/session/create")
    assert r.status_code == 200
    sid = r.json()["session_id"]

    anchors = {
        "session_id": sid,
        "anchors": [
            {"id": "s1", "kind": "support", "position": [0.0, 0.0, 0.0], "confidence": 1.0},
            {"id": "s2", "kind": "support", "position": [4.0, 4.0, 0.0], "confidence": 1.0},
        ],
    }
    r = client.post("/session/anchors", json=anchors)
    assert r.status_code == 200

    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 4, "height": 4}
    depth_meta = {"scale_m_per_unit": 0.001, "width": 4, "height": 4, "encoding": "uint16"}
    depth_u16 = (np.ones((4, 4), dtype=np.uint16) * 1000).astype(np.uint16)

    meta = {
        "timestamp": 0.0,
        "intrinsics": intr,
        "pose": {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "depth_meta": depth_meta,
        "frame_id": "f0",
        "session_id": sid,
    }
    files = {
        "meta": ("meta.json", json.dumps(meta).encode("utf-8"), "application/json"),
        "rgb": ("rgb.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"),
        "depth": ("depth.bin", depth_u16.tobytes(), "application/octet-stream"),
    }
    r = client.post("/session/frame", files=files)
    assert r.status_code == 200

    r = client.get(f"/session/{sid}/status")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert "metrics" in body
    assert "unknown_policy" in body
