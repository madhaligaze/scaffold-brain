from policy.policy_config import PolicyConfig
from scaffold.repair import repair_elements
from scaffold.search import search_scaffolds
from world.occupancy import FREE, OCCUPIED, UNKNOWN


class _Occ:
    def __init__(self):
        self._occ = {(0.0, 0.0, 0.0): int(OCCUPIED)}

    def query(self, pts):
        out = []
        for p in pts:
            k = (float(p[0]), float(p[1]), float(p[2]))
            out.append(self._occ.get(k, int(FREE)))
        return out

    def stats(self, points=None):
        return {"unknown": 90, "free": 10, "occupied": 0, "total": 100}


class _World:
    occupancy = _Occ()

    def query_distance(self, points):
        return [1.0 for _ in points]

    def serialize_state(self):
        return {}

    def compute_overlays(self, policy):
        return {}


def test_search_returns_candidate():
    policy = PolicyConfig.from_config({"policy": {"planner_max_candidates": 4}})
    world = _World()
    anchors = [{"kind": "support", "position": [0.0, 0.0, 0.0], "id": "s0"}]
    best, ranked = search_scaffolds(world, anchors, policy, trace=[])
    assert ranked
    assert best.candidate_id != "none"


def test_repair_moves_off_occupied_or_unknown():
    policy = PolicyConfig.from_config(
        {"policy": {"planner_repair_rounds": 4, "planner_max_shift_m": 0.2, "unknown_mode": "forbid"}}
    )
    world = _World()
    elements = [
        {
            "id": "e0",
            "pose": {"pos": [0.0, 0.0, 0.0], "quat": [0, 0, 0, 1]},
            "type": "post",
            "dims": {"x": 0.1, "y": 0.1, "z": 1.0},
        }
    ]
    repaired, meta = repair_elements(elements, world, policy, trace=[])
    assert meta["moved_elements"] >= 0
    assert repaired[0]["pose"]["pos"] != [0.0, 0.0, 0.0]
