from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class StorageCfg(BaseModel):
    sessions_root: str = "sessions"


class WorldCfg(BaseModel):
    voxel_size_m: float = 0.20
    tsdf_trunc_m: float = 0.40
    min_clearance_m: float = 0.20


class TrackingCfg(BaseModel):
    icp_enabled: bool = True
    icp_apply_correction: bool = True
    icp_max_correspondence_m: float = 0.20
    icp_voxel_down_m: float = 0.06
    icp_min_fitness_apply: float = 0.35
    icp_max_rmse_apply: float = 0.06


class ExportOverlaysCfg(BaseModel):
    occupancy_npz: bool = True
    occupancy_slice_png: bool = True


class ExportCfg(BaseModel):
    env_mesh_format: str = "obj"
    overlays: ExportOverlaysCfg = Field(default_factory=ExportOverlaysCfg)


class PolicyCfg(BaseModel):
    policy_yaml_path: str | None = None


class SecurityApiKeyCfg(BaseModel):
    key: str
    role: str = "operator"


class SecurityCfg(BaseModel):
    # If empty => dev mode (no auth), but middleware still runs.
    api_keys: list[SecurityApiKeyCfg] = Field(default_factory=list)
    audit_enabled: bool = True


class RateLimitCfg(BaseModel):
    enabled: bool = True
    # Per API key if present, else per client IP.
    window_seconds: int = 60
    max_requests: int = 240


class ObservabilityCfg(BaseModel):
    json_logs: bool = True
    metrics_enabled: bool = True
    otel_enabled: bool = False
    otel_service_name: str = "backend-ai"
    # If empty => SDK will rely on OTEL_* env vars.
    otel_exporter_otlp_endpoint: str | None = None


class RetentionCfg(BaseModel):
    enabled: bool = True
    max_age_days: int = 14
    cleanup_interval_minutes: int = 60


class AppConfig(BaseModel):
    server: ServerCfg = Field(default_factory=ServerCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    world: WorldCfg = Field(default_factory=WorldCfg)
    tracking: TrackingCfg = Field(default_factory=TrackingCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    policy: PolicyCfg = Field(default_factory=PolicyCfg)
    security: SecurityCfg = Field(default_factory=SecurityCfg)
    rate_limit: RateLimitCfg = Field(default_factory=RateLimitCfg)
    observability: ObservabilityCfg = Field(default_factory=ObservabilityCfg)
    retention: RetentionCfg = Field(default_factory=RetentionCfg)


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config YAML must be a mapping, got {type(raw).__name__}")
    return AppConfig(**raw)


def find_default_config() -> Path | None:
    candidates = [
        Path("config") / "default.yaml",
        Path("Backend-AI") / "config" / "default.yaml",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None
