#!/usr/bin/env python3
"""Release smoke runner (no external deps)."""

import base64
import math
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import create_app  # noqa: E402


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def make_payload(i: int):
    ang = float(i) * (math.pi / 4.0)
    x = math.cos(ang)
    z = math.sin(ang)
    return {
        "frame_id": f"f{i}",
        "timestamp_ms": 1700000000000 + i * 33,
        "rgb_base64": b64(b"FAKEJPEG" + bytes([i % 255])),
        "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        "pose": {"position": [x, 0.0, z], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "point_cloud": [[0.1, 0.0, 0.1], [0.2, 0.0, 0.1], [0.2, 0.1, 0.2], [0.0, 0.1, 0.2]],
    }


def main() -> int:
    app = create_app()
    pol = app.state.runtime.policy
    setattr(pol, "readiness_observed_ratio_min", 0.01)
    setattr(pol, "min_viewpoints", 1)
    setattr(pol, "min_views_per_anchor", 1)
    client = TestClient(app)

    r = client.post("/session/start")
    if r.status_code != 200:
        return 2
    sid = r.json().get("session_id")

    for i in range(8):
        rr = client.post(f"/session/stream/{sid}", json=make_payload(i))
        if rr.status_code != 200:
            return 2

    rr = client.post("/session/anchors", json={"session_id": sid, "anchors": [{"id": "a0", "kind": "support", "position": [0.0, 0.0, 0.0], "confidence": 1.0}]})
    if rr.status_code != 200:
        return 2

    rr = client.get(f"/session/{sid}/readiness")
    if rr.status_code != 200:
        return 2

    rr = client.post(f"/session/{sid}/request_scaffold")
    if rr.status_code != 200:
        return 2

    rr = client.get(f"/session/{sid}/export/latest")
    if rr.status_code != 200:
        return 2

    print("OK: release smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
