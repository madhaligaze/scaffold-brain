from policy.policy_config import PolicyConfig
from policy.unknown_space import apply_unknown_policy


class _Occ:
    def stats(self, points=None):
        del points
        # 80% unknown in sample
        return {"unknown": 80, "free": 10, "occupied": 10, "total": 100}

    def query(self, points):
        return [0 for _ in points]


class _World:
    occupancy = _Occ()


def test_unknown_ratio_computed():
    policy = PolicyConfig.from_config({"policy": {"unknown_mode": "forbid", "unknown_ratio_near_support_max": 0.6, "unknown_buffer_m": 0.0}})
    world = _World()
    anchors = [{"kind": "support", "position": [0.0, 0.0, 0.0]}]
    rep = apply_unknown_policy(world, anchors, policy)
    assert 0.79 < rep["unknown_ratio_near_support"] < 0.81
    assert rep["violations"], "should violate forbid mode when ratio is high"
