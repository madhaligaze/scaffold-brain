from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request


@dataclass
class _Bucket:
    reset_ts: float
    count: int = 0


@dataclass
class RateLimiter:
    window_seconds: int = 60
    max_requests: int = 240
    _buckets: dict[str, _Bucket] = field(default_factory=dict)

    def _key(self, request: Request) -> str:
        api_key_id = getattr(request.state, "api_key_id", None)
        if isinstance(api_key_id, str) and api_key_id:
            return "k:" + api_key_id

        ip = request.headers.get("x-forwarded-for")
        if ip:
            ip = ip.split(",")[0].strip()
        if not ip and request.client:
            ip = request.client.host
        return "ip:" + (ip or "unknown")

    def check(self, request: Request) -> None:
        now = float(time.time())
        key = self._key(request)

        bucket = self._buckets.get(key)
        if bucket is None or now >= bucket.reset_ts:
            bucket = _Bucket(reset_ts=now + float(self.window_seconds), count=0)
            self._buckets[key] = bucket

        bucket.count += 1
        if bucket.count > int(self.max_requests):
            retry_after = max(0, int(bucket.reset_ts - now))
            raise HTTPException(
                status_code=429,
                detail={
                    "status": "RATE_LIMITED",
                    "reason": "too_many_requests",
                    "window_seconds": int(self.window_seconds),
                    "max_requests": int(self.max_requests),
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )


def build_rate_limiter(config: Any) -> RateLimiter:
    cfg = {}
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        try:
            cfg = dict(dump())
        except Exception:
            cfg = {}

    if isinstance(config, dict):
        cfg = config

    rl = (cfg.get("rate_limit") or {}) if isinstance(cfg, dict) else {}
    return RateLimiter(
        window_seconds=int(rl.get("window_seconds") or 60),
        max_requests=int(rl.get("max_requests") or 240),
    )
