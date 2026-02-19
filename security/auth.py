from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class ApiKey:
    key: str
    role: str


def _to_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    dump = getattr(cfg, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:
            return {}
    return {}


def _load_keys(config: Any) -> list[ApiKey]:
    cfg = _to_dict(config)
    sec = cfg.get("security") or {}
    keys = sec.get("api_keys") or []

    out: list[ApiKey] = []
    for item in keys:
        if isinstance(item, dict):
            k = str(item.get("key") or "")
            r = str(item.get("role") or "operator")
        else:
            k = str(getattr(item, "key", "") or "")
            r = str(getattr(item, "role", "operator") or "operator")

        if k:
            out.append(ApiKey(key=k, role=r))
    return out


def require_api_key(request: Request) -> str:
    state = request.app.state.runtime
    cfg = getattr(state, "config", None)
    keys = _load_keys(cfg)

    # Dev mode (no configured keys)
    if not keys:
        request.state.role = "dev"
        request.state.api_key_id = "dev"
        return "dev"

    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not key:
        raise HTTPException(
            status_code=401, detail={"status": "UNAUTHORIZED", "reason": "missing_api_key"}
        )

    for k in keys:
        if key == k.key:
            request.state.role = k.role
            request.state.api_key_id = k.key[:6] + "…" if len(k.key) > 6 else k.key
            return k.role

    raise HTTPException(
        status_code=403, detail={"status": "FORBIDDEN", "reason": "invalid_api_key"}
    )
