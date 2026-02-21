from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from session.artifacts import ensure_dirs, prune_ndjson_tail

router = APIRouter(tags=["telemetry"])


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
    """Append a crash/diagnostics envelope into sessions/_global/telemetry/client_reports.ndjson."""
    state = request.app.state.runtime
    root: Path = state.store.global_telemetry_dir()
    path = ensure_dirs(root) / "client_reports.ndjson"

    rec = payload.model_dump()
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
