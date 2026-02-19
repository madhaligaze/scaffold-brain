import json

from fastapi.testclient import TestClient

from main import app


def test_request_scaffold_ready_returns_bundle():
    client = TestClient(app)
    session_id = client.post("/session/create").json()["session_id"]
    meta = {
        "session_id": session_id,
        "frame_id": "f1",
        "timestamp": 1.0,
        "intrinsics": {"fx": 100, "fy": 100, "cx": 1, "cy": 1, "width": 2, "height": 2},
        "pose": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
        "depth_meta": {"width": 2, "height": 2, "scale_m_per_unit": 0.001},
    }
    files = {
        "meta": ("meta.json", json.dumps(meta), "application/json"),
        "rgb": ("rgb.jpg", b"jpeg", "image/jpeg"),
        "depth": ("depth.u16", (1000).to_bytes(2, "little") * 4, "application/octet-stream"),
    }
    assert client.post("/session/frame", files=files).status_code == 200

    world = app.state.runtime.get_world(session_id)
    world.occupancy.grid.fill(1)

    res = client.post(f"/session/{session_id}/request_scaffold")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["scaffold"], list)
    assert "revision_id" in body
