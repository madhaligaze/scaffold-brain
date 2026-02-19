from policy.policy_config import PolicyConfig
from policy.unknown_space import apply_unknown_policy, check_points_against_unknown
from world.occupancy import UNKNOWN
from world.world_model import WorldModel


def test_unknown_policy_forbid_flags_support_in_unknown():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    world.occupancy.grid[:] = UNKNOWN
    anchors = [{"id": "s1", "kind": "support", "position": [0.0, 0.0, 0.0]}]
    policy = PolicyConfig.from_config({"policy": {"unknown_mode": "forbid", "unknown_buffer_m": 0.5}})
    rep = apply_unknown_policy(world, anchors, policy)
    assert rep["mode"] == "forbid"
    assert rep["counts"]["violations"] >= 1


def test_unknown_policy_allow_has_no_violations():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    anchors = [{"id": "s1", "kind": "support", "position": [0.0, 0.0, 0.0]}]
    policy = PolicyConfig.from_config({"policy": {"unknown_mode": "allow", "unknown_buffer_m": 0.5}})
    rep = apply_unknown_policy(world, anchors, policy)
    assert rep["mode"] == "allow"
    assert rep["violations"] == []


def test_unknown_policy_buffer_flags_near_unknown():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    pts = [[1.0, 1.0, 1.0]]
    v = check_points_against_unknown(world, pts, mode="buffer", buffer_m=0.5)
    assert len(v) >= 1
