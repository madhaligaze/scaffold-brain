from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes_export import router as export_router
from api.routes_legacy import router as legacy_router
from api.routes_log import router as log_router
from api.routes_planning_v2 import router as planning_router
from api.routes_session_v2 import router as session_router
from api.state import RuntimeState
from observability import metrics as metrics_mod
from observability.logging import setup_json_logging
from observability.metrics import metrics_middleware, metrics_response, setup_metrics
from observability.tracing import setup_tracing
from security.audit import write_audit_event
from security.auth import require_api_key
from security.rate_limit import build_rate_limiter


def init_runtime() -> RuntimeState:
    build = getattr(RuntimeState, "build", None)
    if callable(build):
        return RuntimeState.build()
    raise RuntimeError("RuntimeState.build is unavailable")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: start/stop background tasks (retention, etc.)."""
    task: asyncio.Task | None = None
    try:
        cfg = app.state.runtime.config
        retention = getattr(cfg, "retention", None)
        enabled = True if retention is None else bool(getattr(retention, "enabled", True))
        if enabled:
            max_age_days = int(getattr(retention, "max_age_days", 14))
            interval_min = int(getattr(retention, "cleanup_interval_minutes", 60))

            def _run_once():
                try:
                    return app.state.runtime.store.prune_sessions(max_age_days=max_age_days)
                except Exception:
                    return {"deleted": 0, "kept": 0}

            _run_once()

            async def _loop():
                while True:
                    await asyncio.sleep(max(60.0, float(interval_min) * 60.0))
                    _run_once()

            task = asyncio.create_task(_loop())
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except Exception:
                pass


def create_app() -> FastAPI:
    setup_json_logging(level="INFO")
    app = FastAPI(title="Backend-AI", version="5.1.0", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    app.state.runtime = init_runtime()
    obs_cfg = getattr(app.state.runtime.config, "observability", None)
    metrics_enabled = True if obs_cfg is None else bool(getattr(obs_cfg, "metrics_enabled", True))
    if metrics_enabled:
        setup_metrics()

    setup_tracing(app, app.state.runtime.config)
    app.state.rate_limiter = build_rate_limiter(app.state.runtime.config)

    @app.middleware("http")
    async def _metrics(request: Request, call_next):
        if not metrics_enabled:
            return await call_next(request)
        return await metrics_middleware(request, call_next)

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        cfg = app.state.runtime.config
        role = require_api_key(request)

        rl_cfg = getattr(cfg, "rate_limit", None)
        enabled = True if rl_cfg is None else bool(getattr(rl_cfg, "enabled", True))
        if enabled:
            try:
                app.state.rate_limiter.check(request)
            except Exception:
                if metrics_mod.RATE_LIMITED_TOTAL is not None:
                    metrics_mod.RATE_LIMITED_TOTAL.labels(request.url.path).inc()
                raise
        try:
            sec = getattr(cfg, "security", None)
            audit_enabled = True if sec is None else bool(getattr(sec, "audit_enabled", True))
            if audit_enabled:
                write_audit_event(
                    request.app.state.runtime.store.root,
                    {
                        "path": request.url.path,
                        "method": request.method,
                        "role": role,
                        "session": request.query_params.get("session_id"),
                    },
                )
        except Exception:
            pass

        return await call_next(request)

    app.include_router(session_router)
    app.include_router(planning_router)
    app.include_router(export_router)
    app.include_router(legacy_router)
    app.include_router(log_router)

    sessions_dir = Path(app.state.runtime.config.storage.sessions_root)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/sessions", StaticFiles(directory=str(sessions_dir)), name="sessions")

    @app.get("/metrics")
    def metrics():
        if not metrics_enabled:
            return {"status": "disabled"}
        return metrics_response()

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": app.version,
            "modules": {"legacy": True, "pipeline": True},
        }

    return app


app = create_app()
