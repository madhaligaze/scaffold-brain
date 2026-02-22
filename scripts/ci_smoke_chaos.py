from __future__ import annotations

import argparse
import base64
import math
import random
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import create_app


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _make_payload(i: int, frame_id: str) -> dict:
    ang = float(i) * (math.pi / 4.0)
    x = math.cos(ang)
    z = math.sin(ang)
    return {
        "frame_id": frame_id,
        "timestamp_ms": 1700000000000 + i * 33,
        "rgb_base64": _b64(b"FAKEJPEG" + bytes([i % 255])),
        "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        "pose": {"position": [x, 0.0, z], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "point_cloud": [[0.1, 0.0, 0.1], [0.2, 0.0, 0.1], [0.2, 0.1, 0.2], [0.0, 0.1, 0.2]],
    }


class ChaosClient:
    def __init__(self, client: TestClient, fail_rate: float, max_retries: int, jitter_ms: int):
        self.client = client
        self.fail_rate = float(fail_rate)
        self.max_retries = int(max_retries)
        self.jitter_ms = int(jitter_ms)
        self.simulated_failures = 0
        self.retries = 0
        self.requests = 0

    def request(self, method: str, url: str, json: dict | None = None):
        attempts = 0
        while True:
            attempts += 1
            if self.jitter_ms > 0:
                time.sleep(random.random() * (self.jitter_ms / 1000.0))
            if self.fail_rate > 0 and random.random() < self.fail_rate:
                self.simulated_failures += 1
                if attempts <= self.max_retries:
                    self.retries += 1
                    time.sleep(0.01 + random.random() * 0.05)
                    continue
                raise ConnectionError("simulated network failure")
            self.requests += 1
            return self.client.request(method, url, json=json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=220)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--fail-rate", type=float, default=0.05)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--jitter-ms", type=int, default=50)
    args = ap.parse_args()

    random.seed(0)

    app = create_app()
    pol = app.state.runtime.policy
    # Make readiness easier for CI synthetic frames.
    setattr(pol, "readiness_observed_ratio_min", 0.01)
    setattr(pol, "min_viewpoints", 1)
    setattr(pol, "min_views_per_anchor", 1)

    base = TestClient(app)
    client = ChaosClient(base, args.fail_rate, args.max_retries, args.jitter_ms)

    ok = 0
    t0 = time.time()

    for k in range(int(args.loops)):
        r = client.request("POST", "/session/start")
        if r.status_code != 200:
            raise SystemExit(f"start failed: {r.status_code} {r.text}")
        sid = r.json()["session_id"]

        for i in range(int(args.frames)):
            fid = f"l{k}-f{i}"
            rr = client.request("POST", f"/session/stream/{sid}", json=_make_payload(i, fid))
            if rr.status_code != 200:
                raise SystemExit(f"stream failed: {rr.status_code} {rr.text}")

        rr = client.request(
            "POST",
            "/session/anchors",
            json={
                "session_id": sid,
                "anchors": [{"id": "a0", "kind": "support", "position": [0.0, 0.0, 0.0], "confidence": 1.0}],
            },
        )
        if rr.status_code != 200:
            raise SystemExit(f"anchors failed: {rr.status_code} {rr.text}")

        rr = client.request("GET", f"/session/{sid}/readiness")
        if rr.status_code != 200:
            raise SystemExit(f"readiness failed: {rr.status_code} {rr.text}")

        rr = client.request("POST", f"/session/{sid}/request_scaffold")
        if rr.status_code != 200:
            raise SystemExit(f"compat request_scaffold failed: {rr.status_code} {rr.text}")

        rr = client.request("GET", f"/session/{sid}/export/latest")
        if rr.status_code != 200:
            raise SystemExit(f"export/latest failed: {rr.status_code} {rr.text}")

        ok += 1

    dt = time.time() - t0
    print(
        f"OK loops={ok} req={client.requests} simulated_failures={client.simulated_failures} retries={client.retries} time_s={dt:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
