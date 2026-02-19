from __future__ import annotations

from typing import Any


def setup_tracing(app, config: Any) -> None:
    # Opt-in only (enterprise deployments will set env vars or config flag).
    cfg = {}
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        try:
            cfg = dict(dump())
        except Exception:
            cfg = {}

    if isinstance(config, dict):
        cfg = config

    obs = (cfg.get("observability") or {}) if isinstance(cfg, dict) else {}
    enabled = bool(obs.get("otel_enabled", False))
    if not enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return

    service_name = str(obs.get("otel_service_name") or "backend-ai")
    endpoint = obs.get("otel_exporter_otlp_endpoint")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
