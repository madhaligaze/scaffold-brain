"""Contract guard: the canonical `/v1` namespace and bare-path back-compat."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def test_v1_and_bare_paths_both_served():
    c = TestClient(app)
    # Legacy shim (used by Android) available under both bare and /v1.
    assert c.post("/session/start").status_code == 200
    assert c.post("/v1/session/start").status_code == 200
    # Health under both.
    assert c.get("/health").status_code == 200
    assert c.get("/v1/health").status_code == 200


def test_openapi_exposes_v1_namespace():
    c = TestClient(app)
    paths = c.get("/openapi.json").json()["paths"]
    v1_paths = [p for p in paths if p.startswith("/v1/")]
    assert v1_paths, "no canonical /v1/* paths registered"
    # A representative canonical endpoint is reachable under /v1.
    assert any(p == "/v1/session/start" for p in paths)
