from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from session.artifacts import ensure_dirs

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
