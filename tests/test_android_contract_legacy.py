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


def test_android_legacy_contract_endpoints():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]

    pre_update = client.post(f"/session/update/{session_id}", json={"action": "noop"})
    assert pre_update.status_code == 409
    assert pre_update.json()["detail"]["status"] == "NO_MODEL"

    pre_preview = client.post(f"/session/preview_remove/{session_id}", params={"element_id": "missing"})
    assert pre_preview.status_code == 409
    assert pre_preview.json()["detail"]["status"] == "NO_MODEL"

    stream_res = client.post(f"/session/stream/{session_id}", json=_stream_payload())
    assert stream_res.status_code == 200
    stream_body = stream_res.json()
    assert isinstance(stream_body.get("status"), str)
    hints = stream_body.get("ai_hints")
    assert isinstance(hints, dict)
    assert isinstance(hints.get("instructions"), list)
    assert isinstance(hints.get("warnings"), list)
    assert isinstance(hints.get("quality_score"), (int, float))
    assert isinstance(hints.get("is_ready"), bool)

    model_res = client.post(f"/session/model/{session_id}")
    assert model_res.status_code == 200
    model_body = model_res.json()
    assert isinstance(model_body.get("status"), str)
    options = model_body.get("options")
    assert isinstance(options, list)
    if model_body.get("status") != "NEEDS_SCAN":
        assert len(options) >= 1
        option = options[0]
        for key in ["variant_name", "material_info", "safety_score", "ai_critique", "elements", "stats", "physics"]:
            assert key in option

    update_res = client.post(f"/session/update/{session_id}", json={"action": "noop"})
    if model_body.get("status") == "NEEDS_SCAN":
        assert update_res.status_code == 409
        assert update_res.json()["detail"]["status"] == "NO_MODEL"
    else:
        assert update_res.status_code == 200
    assert update_res.status_code in (200, 409)
    if update_res.status_code == 200:
        update_body = update_res.json()
        assert isinstance(update_body.get("status"), str)
        assert isinstance(update_body.get("is_stable"), bool)
        assert isinstance(update_body.get("physics_status"), str)
        assert isinstance(update_body.get("heatmap"), list)
        assert isinstance(update_body.get("affected_elements"), list)
        assert isinstance(update_body.get("collapsed"), dict)
        assert isinstance(update_body.get("processing_time_ms"), int)

    preview_res = client.post(f"/session/preview_remove/{session_id}", params={"element_id": "missing"})
    assert preview_res.status_code in (200, 409)
    if preview_res.status_code == 200:
        preview_body = preview_res.json()
        for key in ["status", "element_id", "is_critical", "would_collapse", "collapse_count", "warning"]:
            assert key in preview_body
    else:
        assert preview_res.json()["detail"]["status"] == "NO_MODEL"

    voxels_res = client.get(f"/session/voxels/{session_id}")
    assert voxels_res.status_code == 200
    voxels_body = voxels_res.json()
    for key in ["status", "voxels", "bounds", "resolution", "total_count"]:
        assert key in voxels_body
    assert isinstance(voxels_body["voxels"], list)


def test_legacy_stream_ai_hints_readiness_not_ready_and_ready():
    client = TestClient(app)

    not_ready_session = client.post("/session/start").json()["session_id"]
    not_ready_res = client.post(f"/session/stream/{not_ready_session}", json=_stream_payload("nr1"))
    assert not_ready_res.status_code == 200
    not_ready_hints = not_ready_res.json()["ai_hints"]
    assert not_ready_res.json()["status"] == "RECEIVING"
    assert not_ready_hints["is_ready"] is False

    ready_session = client.post("/session/start").json()["session_id"]
    app.state.runtime.get_world(ready_session).occupancy.grid.fill(1)
    ready_res = client.post(f"/session/stream/{ready_session}", json=_stream_payload("r1"))
    assert ready_res.status_code == 200
    ready_hints = ready_res.json()["ai_hints"]
    assert ready_res.json()["status"] == "RECEIVING"
    assert ready_hints["is_ready"] is True


def test_legacy_voxels_returns_required_shape_after_ingest():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]
    assert client.post(f"/session/stream/{session_id}", json=_stream_payload("vox")).status_code == 200

    res = client.get(f"/session/voxels/{session_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert isinstance(body["voxels"], list)
    assert isinstance(body["bounds"], dict)


def test_legacy_model_not_ready_returns_needs_scan():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]
    res = client.post(f"/session/model/{session_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "NEEDS_SCAN"
    assert isinstance(body["options"], list)


def test_legacy_model_ready_returns_elements_list():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]
    app.state.runtime.get_world(session_id).occupancy.grid.fill(1)
    res = client.post(f"/session/model/{session_id}")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["options"], list)
    if body["options"]:
        assert isinstance(body["options"][0]["elements"], list)


def test_legacy_update_returns_no_model_without_revision():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]
    res = client.post(f"/session/update/{session_id}", json={"action": "move", "element_id": "e1"})
    assert res.status_code == 409
    assert res.json()["detail"]["status"] == "NO_MODEL"


def test_legacy_preview_remove_returns_no_model_without_revision():
    client = TestClient(app)
    session_id = client.post("/session/start").json()["session_id"]
    res = client.post(f"/session/preview_remove/{session_id}", params={"element_id": "any"})
    assert res.status_code == 409
    assert res.json()["detail"]["status"] == "NO_MODEL"
