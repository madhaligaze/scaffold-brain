from fastapi.testclient import TestClient

from main import create_app


def test_client_report_rate_limit_bursty():
    app = create_app()
    client = TestClient(app)

    payload = {
        "timestamp_ms": 1700000000000,
        "platform": "android",
        "session_id": "s-rate",
        "device": {"model": "pixel7", "manufacturer": "google", "sdk": 34},
        "errors": [
            {
                "timestamp_ms": 1700000000000,
                "tag": "NET",
                "message": "GET https://example.com/path?token=secret&x=1 failed",
                "stack": "stacktrace line 1\nline2",
            }
        ],
        "queued_actions": {"anchorsQueued": 1, "lockQueued": 0, "mismatchedBaseUrlItems": 0},
        "repro_pack": {
            "responses": [
                {
                    "endpoint": "/session/s-rate/readiness",
                    "timestamp_ms": 1700000000000,
                    "http_code": 200,
                    "body_snippet": "{\"ready\":false}",
                }
            ],
            "errors": [
                {
                    "endpoint": "/session/s-rate/stream",
                    "timestamp_ms": 1700000000000,
                    "http_code": 500,
                    "error_snippet": "POST https://example.com/x?key=secret failed",
                }
            ],
        },
    }

    assert client.post("/telemetry/client_report", json=payload).status_code == 200
    assert client.post("/telemetry/client_report", json=payload).status_code == 200
    assert client.post("/telemetry/client_report", json=payload).status_code == 200

    r4 = client.post("/telemetry/client_report", json=payload)
    assert r4.status_code == 429
    assert (r4.json().get("detail") or {}).get("status") == "RATE_LIMIT"
