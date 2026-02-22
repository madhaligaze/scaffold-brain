from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _is_finite(x: float) -> bool:
    try:
        import math

        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


class Intrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @field_validator("fx", "fy", "cx", "cy")
    @classmethod
    def _finite(cls, v):
        if not _is_finite(v):
            raise ValueError("must be finite")
        return float(v)

    @field_validator("fx", "fy")
    @classmethod
    def _positive_focal(cls, v):
        v = float(v)
        if v <= 0.0:
            raise ValueError("must be > 0")
        return v

    @field_validator("width", "height")
    @classmethod
    def _positive_int(cls, v):
        if not isinstance(v, int):
            v = int(v)
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class Pose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: list[float]
    quaternion: list[float]  # xyzw

    @field_validator("position")
    @classmethod
    def _pos_len3_finite(cls, v):
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError("position must be length-3 list")
        out = []
        for x in v:
            if not _is_finite(x):
                raise ValueError("position must be finite")
            out.append(float(x))
        return out

    @field_validator("quaternion")
    @classmethod
    def _quat_len4_finite(cls, v):
        # IMPORTANT: do NOT reject zero-norm quat here - it must be rejected by validate_and_normalize_frame_meta
        # so that server returns INVALID_FRAMEPACKET (tests expect that).
        if not isinstance(v, (list, tuple)) or len(v) != 4:
            raise ValueError("quaternion must be length-4 list")
        out = []
        for x in v:
            if not _is_finite(x):
                raise ValueError("quaternion must be finite")
            out.append(float(x))
        return out


class DepthMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_m_per_unit: float
    width: int
    height: int
    encoding: Literal["uint16"] = "uint16"

    @field_validator("scale_m_per_unit")
    @classmethod
    def _scale_pos(cls, v):
        if not _is_finite(v):
            raise ValueError("scale_m_per_unit must be finite")
        v = float(v)
        if v <= 0.0:
            raise ValueError("scale_m_per_unit must be > 0")
        return v

    @field_validator("width", "height")
    @classmethod
    def _wh_pos(cls, v):
        if not isinstance(v, int):
            v = int(v)
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class PointCloudMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["xyz"] = "xyz"
    frame: Literal["world", "camera"] = "world"


class FramePacketMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Contract versioning (STAGE A)
    version: str = Field(default="1.0", description="FramePacket contract version")

    session_id: str
    frame_id: str
    timestamp: float

    intrinsics: Intrinsics
    pose: Pose

    depth_meta: Optional[DepthMeta] = None
    pointcloud_meta: Optional[PointCloudMeta] = None

    @field_validator("timestamp")
    @classmethod
    def _ts_nonneg_finite(cls, v):
        if not _is_finite(v):
            raise ValueError("timestamp must be finite")
        v = float(v)
        if v < 0.0:
            raise ValueError("timestamp must be >= 0")
        return v


class AnchorPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["support", "boundary", "target", "point"]
    position: list[float]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
