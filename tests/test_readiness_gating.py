import numpy as np

from policy.policy_config import PolicyConfig
from scanning.readiness import compute_readiness
from world.world_model import WorldModel


def _policy() -> PolicyConfig:
    return PolicyConfig(
        unknown_mode="forbid",
        unknown_buffer_m=0.5,
        min_clearance_m=0.2,
        readiness_observed_ratio_min=0.01,
        unknown_ratio_near_support_max=0.9,
    )


def test_readiness_blocks_without_coverage_and_views():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    anchors = [{"id": "s1", "kind": "support", "position": [0.0, 0.0, 0.0]}]
    ok, score, reasons = compute_readiness(world, anchors, _policy())
    assert ok is False
    assert any(r.startswith("LOW_OBSERVED_RATIO") for r in reasons)
    assert any(r.startswith("LOW_VIEW_DIVERSITY") for r in reasons)


def test_readiness_passes_after_some_frames_and_views():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    anchors = [{"id": "s1", "kind": "support", "position": [0.0, 0.0, 0.0]}]

    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 4, "height": 4}
    depth_meta = {"width": 4, "height": 4, "scale_m_per_unit": 0.001}
    depth_u16 = (np.ones((4, 4), dtype=np.uint16) * 1000).astype(np.uint16)
    depth_bytes = depth_u16.tobytes()

    # 3 distinct camera positions -> should satisfy view diversity
    poses = [
        {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        {"position": [0.6, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        {"position": [0.0, 0.6, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
    ]
    for i, pose in enumerate(poses):
        meta = {
            "session_id": "s",
            "frame_id": f"f{i}",
            "timestamp": float(i),
            "intrinsics": intr,
            "pose": pose,
            "depth_meta": depth_meta,
        }
        world.update_from_frame(meta, rgb=b"", depth_bytes=depth_bytes, pointcloud_bytes=None)

    ok, score, reasons = compute_readiness(world, anchors, _policy())
    assert ok is True, reasons
    assert score > 0.0
