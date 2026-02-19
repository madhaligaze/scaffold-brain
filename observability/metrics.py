from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = None
    Histogram = None
    generate_latest = None

REQUESTS_TOTAL = None
REQUEST_LATENCY = None
RATE_LIMITED_TOTAL = None


def setup_metrics() -> None:
    global REQUESTS_TOTAL, REQUEST_LATENCY, RATE_LIMITED_TOTAL
    if Counter is None or Histogram is None:
        return

    if REQUESTS_TOTAL is None:
        REQUESTS_TOTAL = Counter(
            "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
        )

    if REQUEST_LATENCY is None:
        REQUEST_LATENCY = Histogram(
            "http_request_duration_seconds", "HTTP request latency", ["method", "path"]
        )

    if RATE_LIMITED_TOTAL is None:
        RATE_LIMITED_TOTAL = Counter(
            "http_rate_limited_total", "Total rate limited requests", ["path"]
        )


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = max(0.0, float(time.perf_counter() - t0))

    if REQUESTS_TOTAL is not None:
        REQUESTS_TOTAL.labels(request.method, request.url.path, str(response.status_code)).inc()

    if REQUEST_LATENCY is not None:
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(dt)

    return response


def metrics_response() -> Response:
    if generate_latest is None:
        return Response(content="prometheus_client_not_installed\n", media_type="text/plain")

    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
