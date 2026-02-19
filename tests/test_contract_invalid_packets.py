from __future__ import annotations

import math

import pytest

from contracts.frame_packet import FramePacketMeta
from validation.frame_validation import validate_and_normalize_meta


def _base_meta() -> dict:
    return {
        "version": "1.0",
        "session_id": "s",
        "frame_id": "f1",
        "timestamp": 1000,
        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        "pose": {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        "depth_meta": {"width": 640, "height": 480, "scale_m_per_unit": 0.001},
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda m: m["intrinsics"].__setitem__("fx", 0.0),
        lambda m: m["intrinsics"].__setitem__("fy", -1.0),
        lambda m: m["intrinsics"].__setitem__("width", 0),
        lambda m: m["intrinsics"].__setitem__("height", -10),
        lambda m: m["pose"].__setitem__("position", [math.nan, 0.0, 0.0]),
        lambda m: m["pose"].__setitem__("quaternion", [math.inf, 0.0, 0.0, 1.0]),
        lambda m: m.__setitem__("timestamp", -1),
        lambda m: m["depth_meta"].__setitem__("scale_m_per_unit", 0.0),
        lambda m: m["depth_meta"].__setitem__("scale_m_per_unit", math.nan),
        lambda m: m["depth_meta"].__setitem__("width", 0),
        lambda m: m["depth_meta"].__setitem__("height", 0),
    ],
)
def test_invalid_packets_raise(mutator) -> None:
    meta = _base_meta()
    mutator(meta)
    with pytest.raises(Exception):
        FramePacketMeta.model_validate(meta)


def test_quaternion_is_normalized_by_validation_wrapper() -> None:
    meta = _base_meta()
    meta["pose"]["quaternion"] = [0.0, 0.0, 0.0, 10.0]
    out = validate_and_normalize_meta(FramePacketMeta.model_validate(meta))
    q = out["pose"]["quaternion"]
    norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) ** 0.5
    assert abs(norm - 1.0) < 1e-3
