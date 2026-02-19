from policy.policy_config import PolicyConfig
from scaffold.solver import generate_scaffold
from scaffold.validators import validate_all
from world.world_model import WorldModel


def _policy() -> PolicyConfig:
    return PolicyConfig(
        unknown_mode="allow",
        unknown_buffer_m=0.5,
        min_clearance_m=0.0,
        readiness_observed_ratio_min=0.0,
        unknown_ratio_near_support_max=1.0,
        scaffold_grid_step_m=2.0,
        scaffold_default_height_m=4.0,
        scaffold_deck_levels_m=[2.0, 4.0],
    )


def test_scaffold_generates_and_validates_basic():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    anchors = [
        {"id": "s1", "kind": "support", "position": [0.0, 0.0, 0.0]},
        {"id": "s2", "kind": "support", "position": [4.0, 4.0, 0.0]},
    ]

    elems, tr = generate_scaffold(world, anchors, _policy())
    assert len(elems) > 0
    assert len(tr) > 0

    ok, violations = validate_all(elems, world, _policy())
    assert ok is True, violations
    assert violations == []


def test_scaffold_requires_posts_and_access():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    policy = _policy()
    elems = [{"type": "deck", "pose": {"position": [0, 0, 2], "quaternion": [0, 0, 0, 1]}, "dims": {}}]
    ok, violations = validate_all(elems, world, policy)
    assert ok is False
    assert any(v["type"].startswith("STABILITY_") for v in violations)
    assert any(v["type"].startswith("ACCESS_") for v in violations)
