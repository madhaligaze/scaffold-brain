import json

import numpy as np
from fastapi.testclient import TestClient

from main import create_app


def _jpeg_min() -> bytes:
    # minimal JPEG SOI/EOI (not a valid photo, but enough for server-side non-fatal sanity)
    return b"\xff\xd8\xff\xd9"


def _depth_u16(w: int, h: int, v: int = 1000) -> bytes:
    arr = (np.ones((h, w), dtype=np.uint16) * int(v)).astype(np.uint16)
    return arr.tobytes()


def test_missing_intrinsics_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)

    sid = client.post("/session/create").json()["session_id"]

    meta = {
        "version": "1.0",
        "session_id": sid,
        "frame_id": "f0",
        "timestamp": 1.0,
        "pose": {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "depth_meta": {"scale_m_per_unit": 0.001, "width": 4, "height": 4, "encoding": "uint16"},
    }
    files = {
        "meta": ("meta.json", json.dumps(meta).encode("utf-8"), "application/json"),
        "rgb": ("rgb.jpg", _jpeg_min(), "image/jpeg"),
        "depth": ("depth.bin", _depth_u16(4, 4), "application/octet-stream"),
    }
    r = client.post("/session/frame", files=files)
    assert r.status_code == 400
    j = r.json()
    assert j["detail"]["status"] in ("INVALID_META", "INVALID_FRAMEPACKET")


def test_quaternion_zero_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)

    sid = client.post("/session/create").json()["session_id"]
    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 4, "height": 4}

    meta = {
        "version": "1.0",
        "session_id": sid,
        "frame_id": "f0",
        "timestamp": 1.0,
        "intrinsics": intr,
        "pose": {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 0.0]},
        "depth_meta": {"scale_m_per_unit": 0.001, "width": 4, "height": 4, "encoding": "uint16"},
    }
    files = {
        "meta": ("meta.json", json.dumps(meta).encode("utf-8"), "application/json"),
        "rgb": ("rgb.jpg", _jpeg_min(), "image/jpeg"),
        "depth": ("depth.bin", _depth_u16(4, 4), "application/octet-stream"),
    }
    r = client.post("/session/frame", files=files)
    assert r.status_code == 400
    j = r.json()
    assert j["detail"]["status"] == "INVALID_FRAMEPACKET"


def test_non_monotonic_timestamp_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)

    sid = client.post("/session/create").json()["session_id"]
    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 4, "height": 4}
    depth_meta = {"scale_m_per_unit": 0.001, "width": 4, "height": 4, "encoding": "uint16"}

    meta1 = {
        "version": "1.0",
        "session_id": sid,
        "frame_id": "f1",
        "timestamp": 10.0,
        "intrinsics": intr,
        "pose": {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "depth_meta": depth_meta,
    }
    r1 = client.post(
        "/session/frame",
        files={
            "meta": ("meta.json", json.dumps(meta1).encode("utf-8"), "application/json"),
            "rgb": ("rgb.jpg", _jpeg_min(), "image/jpeg"),
            "depth": ("depth.bin", _depth_u16(4, 4), "application/octet-stream"),
        },
    )
    assert r1.status_code == 200

    meta2 = dict(meta1)
    meta2["frame_id"] = "f2"
    meta2["timestamp"] = 9.0
    r2 = client.post(
        "/session/frame",
        files={
            "meta": ("meta.json", json.dumps(meta2).encode("utf-8"), "application/json"),
            "rgb": ("rgb.jpg", _jpeg_min(), "image/jpeg"),
            "depth": ("depth.bin", _depth_u16(4, 4), "application/octet-stream"),
        },
    )
    assert r2.status_code == 400
    assert r2.json()["detail"]["status"] == "INVALID_FRAMEPACKET"
