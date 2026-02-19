import base64

from fastapi.testclient import TestClient

from main import app


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _stream_payload(frame_id: str = "f1") -> dict:
    return {
        "rgb_base64": _b64(b"rgb"),
        "depth_base64": _b64((1000).to_bytes(2, "little") * 4),
        "intrinsics": {"fx": 100, "fy": 100, "cx": 1, "cy": 1, "width": 2, "height": 2},
        "pose": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
        "timestamp": 1.0,
        "frame_id": frame_id,
    }


def test_android_legacy_endpoints_are_present_and_json_safe():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]

    voxels_res = client.get(f"/session/voxels/{session_id}")
    assert voxels_res.status_code == 200
    voxels_body = voxels_res.json()
    for key in ["status", "voxels", "bounds", "resolution", "total_count"]:
        assert key in voxels_body

    model_res = client.post(f"/session/model/{session_id}")
    assert model_res.status_code == 200
    model_body = model_res.json()
    assert "status" in model_body
    assert "options" in model_body

    pre_update = client.post(f"/session/update/{session_id}", json={"action": "noop"})
    assert pre_update.status_code == 409
    assert pre_update.json()["detail"]["status"] == "NO_MODEL"

    pre_preview = client.post(f"/session/preview_remove/{session_id}", params={"element_id": "x"})
    assert pre_preview.status_code == 409
    assert pre_preview.json()["detail"]["status"] == "NO_MODEL"

    assert client.post(f"/session/stream/{session_id}", json=_stream_payload()).status_code == 200
    ready_world = app.state.runtime.get_world(session_id)
    ready_world.occupancy.grid.fill(1)
    model_ready_res = client.post(f"/session/model/{session_id}")
    assert model_ready_res.status_code == 200
    model_ready_body = model_ready_res.json()
    assert "status" in model_ready_body
    assert "options" in model_ready_body

    if model_ready_body.get("status") == "NEEDS_SCAN":
        post_update = client.post(f"/session/update/{session_id}", json={"action": "noop"})
        assert post_update.status_code in (200, 409)
        post_preview = client.post(f"/session/preview_remove/{session_id}", params={"element_id": "x"})
        assert post_preview.status_code in (200, 409)
    else:
        post_update = client.post(f"/session/update/{session_id}", json={"action": "noop"})
        assert post_update.status_code == 200
        update_body = post_update.json()
        for key in ["status", "is_stable", "physics_status", "heatmap", "affected_elements", "collapsed", "processing_time_ms"]:
            assert key in update_body

        post_preview = client.post(f"/session/preview_remove/{session_id}", params={"element_id": "x"})
        assert post_preview.status_code == 200
        preview_body = post_preview.json()
        for key in ["status", "element_id", "is_critical", "would_collapse", "collapse_count", "warning"]:
            assert key in preview_body
