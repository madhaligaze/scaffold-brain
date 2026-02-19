from inference.extend_linear_objects import propose_extension
from policy.policy_config import PolicyConfig
from world.world_model import WorldModel


def _default_policy() -> PolicyConfig:
    return PolicyConfig(
        unknown_mode="forbid",
        unknown_buffer_m=0.5,
        min_clearance_m=0.2,
        readiness_observed_ratio_min=0.1,
        unknown_ratio_near_support_max=0.6,
    )


def test_extension_unknown_needs_scan():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    policy = _default_policy()
    ext = propose_extension(
        p0=[0.0, 0.0, 0.0],
        p1=[0.5, 0.0, 0.0],
        world_model=world,
        policy=policy,
        step_m=0.2,
        max_extend_m=1.0,
    )
    assert ext["stop_reason"] == "UNKNOWN"
    assert ext["needs_scan"] is not None


def test_extension_out_of_scope():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    policy = _default_policy()
    ext = propose_extension(
        p0=[4.8, 0.0, 0.0],
        p1=[4.9, 0.0, 0.0],
        world_model=world,
        policy=policy,
        step_m=0.3,
        max_extend_m=5.0,
    )
    assert ext["stop_reason"] in {"OUT_OF_SCOPE", "UNKNOWN"}
