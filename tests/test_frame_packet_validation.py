import json

from fastapi.testclient import TestClient

from main import app


def _files(meta: dict, include_depth: bool = True):
    files = {
        "meta": ("meta.json", json.dumps(meta), "application/json"),
        "rgb": ("rgb.jpg", b"jpeg", "image/jpeg"),
    }
    if include_depth:
        files["depth"] = ("depth.u16", (1000).to_bytes(2, "little") * 4, "application/octet-stream")
    return files


def test_missing_pose_intrinsics_validation_error():
    client = TestClient(app)
    payload = {"session_id": "s1", "frame_id": "f1", "timestamp": 1.0, "depth_meta": {"width": 2, "height": 2, "scale_m_per_unit": 0.001}}
    res = client.post("/session/frame", files=_files(payload))
    assert res.status_code == 400


def test_missing_depth_and_pointcloud_error():
    client = TestClient(app)
    payload = {
        "session_id": "s1",
        "frame_id": "f1",
        "timestamp": 1.0,
        "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 2, "height": 2},
        "pose": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
    }
    res = client.post("/session/frame", files=_files(payload, include_depth=False))
    assert res.status_code == 400


def test_valid_meta_accepted():
    client = TestClient(app)
    payload = {
        "session_id": "s1",
        "frame_id": "f1",
        "timestamp": 1.0,
        "intrinsics": {"fx": 100, "fy": 100, "cx": 1, "cy": 1, "width": 2, "height": 2},
        "pose": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
        "depth_meta": {"width": 2, "height": 2, "scale_m_per_unit": 0.001},
    }
    res = client.post("/session/frame", files=_files(payload))
    assert res.status_code == 200
