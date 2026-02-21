import base64
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import create_app


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _make_payload(i: int):
    ang = float(i) * (math.pi / 4.0)
    x = math.cos(ang)
    z = math.sin(ang)
    return {
        "frame_id": f"f{i}",
        "timestamp_ms": 1700000000000 + i * 33,
        "rgb_base64": _b64(b"FAKEJPEG" + bytes([i % 255])),
        "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        "pose": {"position": [x, 0.0, z], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "point_cloud": [[0.1, 0.0, 0.1], [0.2, 0.0, 0.1], [0.2, 0.1, 0.2], [0.0, 0.1, 0.2]],
    }


@pytest.mark.parametrize("n_frames", [8])
def test_release_smoke_e2e(tmp_path: Path, monkeypatch, n_frames: int):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    pol = app.state.runtime.policy
    setattr(pol, "readiness_observed_ratio_min", 0.01)
    setattr(pol, "min_viewpoints", 1)
    setattr(pol, "min_views_per_anchor", 1)

    client = TestClient(app)
    r = client.post("/session/start")
    assert r.status_code == 200
    sid = r.json()["session_id"]

    for i in range(n_frames):
        rr = client.post(f"/session/stream/{sid}", json=_make_payload(i))
        assert rr.status_code == 200, rr.text

    rr = client.post("/session/anchors", json={"session_id": sid, "anchors": [{"id": "a0", "kind": "support", "position": [0.0, 0.0, 0.0], "confidence": 1.0}]})
    assert rr.status_code == 200, rr.text

    rr = client.get(f"/session/{sid}/readiness")
    assert rr.status_code == 200, rr.text

    rr = client.post(f"/session/{sid}/request_scaffold")
    assert rr.status_code == 200, rr.text

    rr = client.get(f"/session/{sid}/export/latest")
    assert rr.status_code == 200, rr.text
    bundle = rr.json()
    layers = (((bundle.get("ui") or {}).get("layers")) or [])
    assert isinstance(layers, list)

    if layers:
        path = (((layers[0].get("file") or {}).get("glb") or {}).get("path")) or (layers[0].get("file") or {}).get("path")
        assert path
        rr = client.get(path)
        assert rr.status_code == 200, rr.text
        assert len(rr.content) > 0
