from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Intrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class Pose(BaseModel):
    position: list[float]
    quaternion: list[float]  # xyzw


class DepthMeta(BaseModel):
    scale_m_per_unit: float
    width: int
    height: int
    encoding: Literal["uint16"] = "uint16"


class PointCloudMeta(BaseModel):
    format: Literal["xyz"] = "xyz"
    frame: Literal["world", "camera"] = "world"


class FramePacketMeta(BaseModel):
    # Contract versioning (STAGE A)
    version: str = Field(default="1.0", description="FramePacket contract version")

    session_id: str
    frame_id: str
    timestamp: float

    intrinsics: Intrinsics
    pose: Pose

    depth_meta: Optional[DepthMeta] = None
    pointcloud_meta: Optional[PointCloudMeta] = None


class AnchorPoint(BaseModel):
    id: str
    kind: Literal["support", "boundary", "target"]
    position: list[float]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
