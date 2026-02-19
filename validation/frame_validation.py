from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _as_float_list3(v: Any) -> list[float] | None:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    out: list[float] = []
    for x in v:
        if not _is_finite(x):
            return None
        out.append(float(x))
    return out


def _as_float_list4(v: Any) -> list[float] | None:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    out: list[float] = []
    for x in v:
        if not _is_finite(x):
            return None
        out.append(float(x))
    return out


def _normalize_quat_xyzw(q: list[float]) -> tuple[list[float] | None, str | None]:
    x, y, z, w = q
    n2 = x * x + y * y + z * z + w * w
    if not math.isfinite(n2) or n2 <= 1e-12:
        return None, "pose.quaternion has zero/invalid norm"
    n = math.sqrt(n2)
    return [x / n, y / n, z / n, w / n], None


def validate_and_normalize_frame_meta(
    meta: Dict[str, Any],
    *,
    rgb_bytes: bytes | None = None,
    last_timestamp: float | None = None,
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    """
    STAGE A: strict metrology validation + normalization.
    Returns (validated_meta, errors). On errors, validated_meta is None.
    """
    errs: List[Dict[str, Any]] = []

    if not isinstance(meta, dict):
        return None, [{"field": "meta", "code": "TYPE", "msg": "meta must be an object"}]

    out: Dict[str, Any] = {}

    # version
    ver = meta.get("version", "1.0")
    out["version"] = str(ver)

    # ids
    sid = meta.get("session_id")
    fid = meta.get("frame_id")
    if not isinstance(sid, str) or not sid.strip():
        errs.append({"field": "session_id", "code": "REQUIRED", "msg": "session_id is required"})
    else:
        out["session_id"] = sid.strip()
    if not isinstance(fid, str) or not fid.strip():
        errs.append({"field": "frame_id", "code": "REQUIRED", "msg": "frame_id is required"})
    else:
        out["frame_id"] = fid.strip()

    # timestamp
    ts = meta.get("timestamp")
    if not _is_finite(ts):
        errs.append({"field": "timestamp", "code": "INVALID", "msg": "timestamp must be finite number"})
    else:
        tsf = float(ts)
        if last_timestamp is not None and tsf < float(last_timestamp) - 1e-6:
            errs.append(
                {
                    "field": "timestamp",
                    "code": "NON_MONOTONIC",
                    "msg": f"timestamp must be monotonic (last={float(last_timestamp):.6f})",
                }
            )
        out["timestamp"] = tsf

    # intrinsics
    intr = meta.get("intrinsics")
    if not isinstance(intr, dict):
        errs.append({"field": "intrinsics", "code": "REQUIRED", "msg": "intrinsics is required"})
        intr = {}
    fx = intr.get("fx")
    fy = intr.get("fy")
    cx = intr.get("cx")
    cy = intr.get("cy")
    w = intr.get("width")
    h = intr.get("height")
    if not _is_finite(fx) or float(fx) <= 1e-9:
        errs.append({"field": "intrinsics.fx", "code": "INVALID", "msg": "fx must be > 0"})
    if not _is_finite(fy) or float(fy) <= 1e-9:
        errs.append({"field": "intrinsics.fy", "code": "INVALID", "msg": "fy must be > 0"})
    if not isinstance(w, int) or w <= 0 or w > 16384:
        errs.append({"field": "intrinsics.width", "code": "INVALID", "msg": "width must be int in (0..16384]"})
    if not isinstance(h, int) or h <= 0 or h > 16384:
        errs.append({"field": "intrinsics.height", "code": "INVALID", "msg": "height must be int in (0..16384]"})
    if _is_finite(cx) and isinstance(w, int) and w > 0:
        if float(cx) < -1.0 or float(cx) > float(w) + 1.0:
            errs.append({"field": "intrinsics.cx", "code": "INVALID", "msg": "cx out of bounds"})
    else:
        errs.append({"field": "intrinsics.cx", "code": "INVALID", "msg": "cx must be finite"})
    if _is_finite(cy) and isinstance(h, int) and h > 0:
        if float(cy) < -1.0 or float(cy) > float(h) + 1.0:
            errs.append({"field": "intrinsics.cy", "code": "INVALID", "msg": "cy out of bounds"})
    else:
        errs.append({"field": "intrinsics.cy", "code": "INVALID", "msg": "cy must be finite"})

    if len(errs) == 0:
        out["intrinsics"] = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(w),
            "height": int(h),
        }

    # pose
    pose = meta.get("pose")
    if not isinstance(pose, dict):
        errs.append({"field": "pose", "code": "REQUIRED", "msg": "pose is required"})
        pose = {}
    pos = _as_float_list3(pose.get("position"))
    if pos is None:
        errs.append({"field": "pose.position", "code": "INVALID", "msg": "position must be [x,y,z] finite"})
    quat = _as_float_list4(pose.get("quaternion"))
    if quat is None:
        errs.append({"field": "pose.quaternion", "code": "INVALID", "msg": "quaternion must be [x,y,z,w] finite"})
    else:
        qn, qerr = _normalize_quat_xyzw(quat)
        if qn is None:
            errs.append({"field": "pose.quaternion", "code": "INVALID", "msg": qerr or "invalid quaternion"})
        else:
            quat = qn

    if pos is not None and quat is not None and isinstance(quat, list):
        out["pose"] = {"position": pos, "quaternion": quat}

    # depth_meta
    depth_meta = meta.get("depth_meta")
    if depth_meta is not None:
        if not isinstance(depth_meta, dict):
            errs.append({"field": "depth_meta", "code": "TYPE", "msg": "depth_meta must be object"})
        else:
            dm_w = depth_meta.get("width")
            dm_h = depth_meta.get("height")
            dm_s = depth_meta.get("scale_m_per_unit")
            enc = depth_meta.get("encoding", "uint16")
            if enc != "uint16":
                errs.append({"field": "depth_meta.encoding", "code": "INVALID", "msg": "encoding must be uint16"})
            if not isinstance(dm_w, int) or not isinstance(dm_h, int) or dm_w <= 0 or dm_h <= 0:
                errs.append({"field": "depth_meta.size", "code": "INVALID", "msg": "depth width/height must be >0"})
            if not _is_finite(dm_s) or float(dm_s) <= 0.0 or float(dm_s) > 0.1:
                errs.append({"field": "depth_meta.scale_m_per_unit", "code": "INVALID", "msg": "scale must be (0..0.1]"})
            # if intrinsics validated, require size match
            if "intrinsics" in out and isinstance(dm_w, int) and isinstance(dm_h, int):
                if dm_w != int(out["intrinsics"]["width"]) or dm_h != int(out["intrinsics"]["height"]):
                    errs.append({"field": "depth_meta", "code": "MISMATCH", "msg": "depth size must match intrinsics"})
            if len([e for e in errs if str(e.get("field", "")).startswith("depth_meta")]) == 0:
                out["depth_meta"] = {
                    "width": int(dm_w),
                    "height": int(dm_h),
                    "scale_m_per_unit": float(dm_s),
                    "encoding": "uint16",
                }

    # pointcloud_meta (minimal)
    pc_meta = meta.get("pointcloud_meta")
    if pc_meta is not None:
        if not isinstance(pc_meta, dict):
            errs.append({"field": "pointcloud_meta", "code": "TYPE", "msg": "pointcloud_meta must be object"})
        else:
            fmt = pc_meta.get("format", "xyz")
            frm = pc_meta.get("frame", "world")
            if fmt not in ("xyz",):
                errs.append({"field": "pointcloud_meta.format", "code": "INVALID", "msg": "format must be xyz"})
            if frm not in ("world", "camera"):
                errs.append({"field": "pointcloud_meta.frame", "code": "INVALID", "msg": "frame must be world|camera"})
            if len([e for e in errs if str(e.get("field", "")).startswith("pointcloud_meta")]) == 0:
                out["pointcloud_meta"] = {"format": str(fmt), "frame": str(frm)}

    # optional quick RGB sanity (non-fatal)
    if rgb_bytes is not None and len(rgb_bytes) >= 2:
        if not (rgb_bytes[0] == 0xFF and rgb_bytes[1] == 0xD8):
            out["rgb_warning"] = "rgb does not look like JPEG (expected 0xFFD8)"

    if errs:
        return None, errs
    return out, []


from pydantic import BaseModel


def validate_and_normalize_meta(meta_payload: BaseModel) -> dict[str, Any]:
    """Stable wrapper for callers/tests."""
    fn = globals().get("validate_frame_meta") or globals().get("validate_and_normalize")
    if callable(fn):
        out = fn(meta_payload)
        if isinstance(out, BaseModel):
            return out.model_dump()
        if isinstance(out, dict):
            return out
    if isinstance(meta_payload, BaseModel):
        payload = meta_payload.model_dump()
    else:
        payload = dict(meta_payload)
    normalized, errs = validate_and_normalize_frame_meta(payload)
    if errs:
        raise ValueError(errs)
    return normalized or payload


def validate_meta_dict(meta_dict: dict[str, Any]) -> dict[str, Any]:
    fn = globals().get("validate_frame_meta_dict")
    if callable(fn):
        return fn(meta_dict)
    try:
        from contracts.frame_packet import FramePacketMeta

        meta = FramePacketMeta.model_validate(meta_dict)
        return validate_and_normalize_meta(meta)
    except Exception:
        return meta_dict
