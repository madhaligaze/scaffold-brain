from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from session.artifacts import ensure_dirs, prune_ndjson_tail

router = APIRouter(tags=["telemetry"])

_URL_RE = re.compile(r"(https?://[^\s]+)")


def _sanitize_url(text: str) -> str:
    # Strip query params to avoid leaking tokens/PII
    def repl(m):
        url = m.group(1)
        if "?" in url:
            url = url.split("?", 1)[0] + "?<redacted>"
        return url

    return _URL_RE.sub(repl, text)


def _sanitize_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Server-side sanitize as last line of defence.

    - Drop large/nested data
    - Truncate strings
    - Remove potentially sensitive keys
    """
    deny = (
        "rgb",
        "depth",
        "pose",
        "intrinsics",
        "point_cloud",
        "anchors",
        "position",
        "quaternion",
        "glb",
        "obj",
        "file",
        "files",
        "image",
        "bitmap",
        "base64",
        "url",
        "server",
        "base_url",
    )

    def clean_obj(o, depth=0):
        if depth > 3:
            return None
        if o is None:
            return None
        if isinstance(o, (int, float, bool)):
            return o
        if isinstance(o, str):
            s = _sanitize_url(o)
            return s[:2048]
        if isinstance(o, list):
            if len(o) > 50:
                return None
            if all(isinstance(x, (int, float)) for x in o):
                return None
            out = []
            for x in o:
                cx = clean_obj(x, depth + 1)
                if cx is not None:
                    out.append(cx)
            return out[:50]
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                ks = str(k).lower()
                if any(d in ks for d in deny):
                    continue
                cv = clean_obj(v, depth + 1)
                if cv is not None:
                    out[str(k)[:64]] = cv
            return out
        return None

    return clean_obj(payload, 0) or {}


class LogDeviceInfo(BaseModel):
    model: str | None = None
    manufacturer: str | None = None
    sdk: int | None = None


class LogPayload(BaseModel):
    event: str = Field(..., min_length=1, max_length=64)
    timestamp_ms: int = Field(..., ge=0)
    data: dict[str, Any] = Field(default_factory=dict)
    device: LogDeviceInfo | None = None


class CrashErrorItem(BaseModel):
    where: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=2048)
    timestamp_ms: int = Field(..., ge=0)
    stack: str | None = None
    fatal: bool = False


class CrashEnvelope(BaseModel):
    session_id: str | None = None
    timestamp_ms: int = Field(..., ge=0)
    app_version: str | None = None
    build: str | None = None
    platform: str | None = "android"
    device: LogDeviceInfo | None = None
    connection_status: str | None = None
    server_base_url: str | None = None
    last_export_rev: str | None = None
    loaded_export_rev: str | None = None
    last_revision_id: str | None = None
    client_stats: dict[str, Any] = Field(default_factory=dict)
    errors: list[CrashErrorItem] = Field(default_factory=list)


class ClientErrorItem(BaseModel):
    timestamp_ms: int = Field(..., ge=0)
    tag: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=512)
    stack: str | None = Field(default=None, max_length=8000)


class ClientReportEnvelope(BaseModel):
    session_id: str | None = None
    timestamp_ms: int = Field(..., ge=0)
    client_stats: dict[str, Any] = Field(default_factory=dict)
    last_export_rev: str | None = None
    queued_actions: dict[str, Any] = Field(default_factory=dict)
    last_errors: list[ClientErrorItem] = Field(default_factory=list)
    device: LogDeviceInfo | None = None


@router.post("/session/log/{session_id}")
def post_log(request: Request, session_id: str, payload: LogPayload) -> dict:
    """Append telemetry event into sessions/<sid>/telemetry/events.ndjson."""
    state = request.app.state.runtime
    root: Path = state.store.session_root(session_id)

    if not root.exists():
        raise HTTPException(status_code=404, detail={"status": "NO_SESSION", "session_id": session_id})

    telemetry_dir = ensure_dirs(root / "telemetry")
    path = telemetry_dir / "events.ndjson"

    rec = payload.model_dump()
    rec["session_id"] = session_id
    rec["ingested_at_ms"] = int(time.time() * 1000)

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"status": "WRITE_FAILED", "error": str(exc)}) from exc

    return {"status": "OK"}


@router.post("/session/report/{session_id}")
def post_session_report(request: Request, session_id: str, payload: CrashEnvelope) -> dict:
    """Backward-compatible crash report endpoint."""
    state = request.app.state.runtime

    if session_id == "global":
        root: Path = ensure_dirs(state.store.root / "_global")
    else:
        root = state.store.session_root(session_id)
        if not root.exists():
            raise HTTPException(status_code=404, detail={"status": "NO_SESSION", "session_id": session_id})

    telemetry_dir = ensure_dirs(root / "telemetry")
    path = telemetry_dir / "crash_reports.ndjson"

    rec = payload.model_dump()
    rec["session_id"] = session_id
    rec["ingested_at_ms"] = int(time.time() * 1000)

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"status": "WRITE_FAILED", "error": str(exc)}) from exc

    return {"status": "OK"}


@router.post("/telemetry/client_report")
def post_client_report(request: Request, payload: ClientReportEnvelope) -> dict:
    """Append a crash/diagnostics envelope into sessions/_global/telemetry/client_reports.ndjson.

    Release hardening:
      - Rate limit to avoid DDOS from bad networks.
      - Sanitize payload server-side (defence-in-depth).
    """
    state = request.app.state.runtime

    # Rate limit per session+device+client.
    sid = payload.session_id or "global"
    dev = (payload.device.model if payload.device else None) or "unknown_device"
    host = getattr(getattr(request, "client", None), "host", None) or "unknown_host"
    key = f"client_report:{sid}:{dev}:{host}"

    # Burst 3, then ~1 per 90s.
    allowed, retry_after = state.rate_limiter.allow(key, capacity=3.0, refill_per_sec=(1.0 / 90.0), cost=1.0)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"status": "RATE_LIMIT", "retry_after_s": float(retry_after)},
            headers={"Retry-After": str(int(max(1.0, retry_after)))},
        )

    root: Path = state.store.global_telemetry_dir()
    path = ensure_dirs(root) / "client_reports.ndjson"

    raw = payload.model_dump()
    rec = _sanitize_report(raw)
    rec["ingested_at_ms"] = int(time.time() * 1000)

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"status": "WRITE_FAILED", "error": str(exc)}) from exc

    retention = getattr(state.config, "retention", None)
    max_bytes = int(getattr(retention, "telemetry_max_file_bytes", 5_000_000)) if retention else 5_000_000
    max_lines = int(getattr(retention, "telemetry_max_lines", 20_000)) if retention else 20_000
    prune_ndjson_tail(path, max_bytes=max_bytes, max_lines=max_lines)

    return {"status": "OK"}
