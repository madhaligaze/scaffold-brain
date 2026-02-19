from __future__ import annotations

import math


def make_scan_hint(position: list[float], *, look_at: list[float] | None = None, note: str = "") -> dict:
    if look_at is None:
        look_at = [position[0], position[1], position[2] - 1.0]
    dx = float(look_at[0] - position[0])
    dy = float(look_at[1] - position[1])
    dz = float(look_at[2] - position[2])
    dist = float(math.sqrt(dx * dx + dy * dy + dz * dz))
    return {
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "look_at": [float(look_at[0]), float(look_at[1]), float(look_at[2])],
        "distance_m": dist,
        "note": note or "Scan this region",
        "kind": "scan_hint",
    }
