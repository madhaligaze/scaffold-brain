from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LegacyStreamPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str | None = None
    frame_id: str | None = None
    timestamp: float | None = None

    rgb_base64: str | None = None
    image_base64: str | None = None
    rgb: str | None = None

    depth_base64: str | None = None
    depth: str | None = None
    # New name used by Android client (meters per unit). Keep legacy depth_scale for compatibility.
    depth_scale_m_per_unit: float | None = Field(default=None, gt=0)
    depth_scale: float | None = Field(default=None, gt=0)
    depth_width: int | None = Field(default=None, gt=0)
    depth_height: int | None = Field(default=None, gt=0)

    point_cloud: list[list[float]] | None = None
    pointcloud: list[list[float]] | None = None

    intrinsics: dict | None = None
    pose: dict | None = None
