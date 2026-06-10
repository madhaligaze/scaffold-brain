"""The grid solver must step around scanned obstacles at generation time
(regression for the old ``del world_model`` that ignored the scan)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scaffold.solver import generate_scaffold
from world.occupancy import FREE, OCCUPIED


def _policy(**overrides):
    base = dict(
        scaffold_grid_step_m=2.0,
        scaffold_default_height_m=4.0,
        scaffold_min_bay_m=1.2,
        scaffold_max_bay_m=3.0,
        scaffold_deck_levels_m=[2.0, 4.0],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _square_anchors(size: float = 4.0):
    return [
        {"kind": "support", "position": [0.0, 0.0, 0.0]},
        {"kind": "support", "position": [size, 0.0, 0.0]},
        {"kind": "support", "position": [0.0, size, 0.0]},
        {"kind": "support", "position": [size, size, 0.0]},
    ]


class _FakeOcc:
    """Occupancy stub: a single axis-aligned occupied box, everything else FREE."""

    def __init__(self, box_min, box_max):
        self.box_min = np.asarray(box_min, dtype=float)
        self.box_max = np.asarray(box_max, dtype=float)

    def query(self, points):
        out = []
        for p in points:
            q = np.asarray(p, dtype=float)
            inside = bool(np.all(q >= self.box_min) and np.all(q <= self.box_max))
            out.append(OCCUPIED if inside else FREE)
        return out


class _FakeWorld:
    def __init__(self, occ):
        self.occupancy = occ


def _posts(elements):
    return [e for e in elements if e.get("type") == "post"]


def _post_xy(elements):
    return {(round(e["pose"]["position"][0], 3), round(e["pose"]["position"][1], 3)) for e in _posts(elements)}


def test_no_world_is_unchanged_full_grid():
    elements, meta = generate_scaffold(None, _square_anchors(), _policy())
    assert meta["world_aware"] is False
    assert meta["posts_blocked"] == 0
    # The centre grid column (1.4, 1.4) exists when nothing is scanned.
    assert (1.4, 1.4) in _post_xy(elements)


def test_occupied_column_is_skipped():
    # Occupy a tall box around the centre grid column at (1.4, 1.4).
    world = _FakeWorld(_FakeOcc(box_min=[1.0, 1.0, -1.0], box_max=[1.8, 1.8, 6.0]))
    elements, meta = generate_scaffold(world, _square_anchors(), _policy())

    assert meta["world_aware"] is True
    assert meta["posts_blocked"] >= 1
    # No post may be planted inside the scanned obstacle.
    assert (1.4, 1.4) not in _post_xy(elements)
    # The rest of the layout still gets built.
    assert len(_posts(elements)) >= 4


def test_clear_scan_blocks_nothing():
    # A world whose occupancy reports everything FREE must not prune anything.
    world = _FakeWorld(_FakeOcc(box_min=[100.0, 100.0, 100.0], box_max=[101.0, 101.0, 101.0]))
    elements, meta = generate_scaffold(world, _square_anchors(), _policy())
    assert meta["world_aware"] is True
    assert meta["posts_blocked"] == 0
    assert (1.4, 1.4) in _post_xy(elements)
