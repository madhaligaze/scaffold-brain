import base64
import json

from fastapi.testclient import TestClient

from main import app


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_legacy_stream_proxies_to_ingest_and_increments_frames():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]
    payload = {
        "rgb_base64": _b64(b"rgb"),
        "depth_base64": _b64((1000).to_bytes(2, "little") * 4),
        "intrinsics": {"fx": 100, "fy": 100, "cx": 1, "cy": 1, "width": 2, "height": 2},
        "pose": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
        "timestamp": 1.0,
        "frame_id": "legacy-f1",
    }
    res = client.post(f"/session/stream/{session_id}", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["legacy_stream"] is True
    assert body["legacy_mode"] == "json_adapter"

    status = client.get(f"/session/{session_id}/status")
    assert status.status_code == 200
    assert status.json()["metrics"]["metrics"]["frames"] == 1


def test_legacy_stream_returns_409_when_geometry_missing():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]
    payload = {"rgb_base64": _b64(b"rgb")}
    res = client.post(f"/session/stream/{session_id}", json=payload)
    assert res.status_code == 409
    detail = res.json()["detail"]
    assert detail["status"] == "NEEDS_GEOMETRY"
    assert "missing" in detail


def test_legacy_unlock_and_snapshot_restore_flow():
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
        "depth": ("depth.u16", (1000).to_bytes(2, "little") * 4, "application/octet-stream"),
    }
    assert client.post("/session/frame", files=files).status_code == 200
    app.state.runtime.get_world(session_id).occupancy.grid.fill(1)
    req = client.post(f"/session/{session_id}/request_scaffold")
    assert req.status_code == 200
    revision = req.json()["revision_id"]

    snapshots = client.get(f"/session/snapshots/{session_id}")
    assert snapshots.status_code == 200
    s_body = snapshots.json()
    assert s_body["status"] == "ok"
    assert isinstance(s_body["snapshots"], list)

    unlock = client.post(f"/session/unlock/{session_id}")
    assert unlock.status_code == 200
    assert unlock.json()["noop"] is True

    restore = client.post(f"/session/snapshot/restore/{session_id}/{revision}")
    assert restore.status_code == 200
    assert restore.json()["restored"] is True


def test_legacy_restore_404_for_missing_revision():
    client = TestClient(app)
    session_id = client.post("/session/create").json()["session_id"]
    res = client.post(f"/session/snapshot/restore/{session_id}/missing")
    assert res.status_code == 404
