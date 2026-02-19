import json

from fastapi.testclient import TestClient

from main import app


def test_end_to_end_needs_scan_then_scan_plan_available():
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
        "rgb": ("rgb.jpg", b"rgb", "image/jpeg"),
        "depth": ("depth.u16", (0).to_bytes(2, "little") * 4, "application/octet-stream"),
    }
    assert client.post("/session/frame", files=files).status_code == 200

    scaffold = client.post(f"/session/{session_id}/request_scaffold")
    assert scaffold.status_code == 409
    detail = scaffold.json()["detail"]
    assert detail["status"] == "NEEDS_SCAN"
    assert isinstance(detail["scan_plan"], list)

    plan = client.get(f"/session/{session_id}/scan_plan")
    assert plan.status_code == 200
    assert isinstance(plan.json()["scan_plan"], list)
