import numpy as np

from world.world_model import WorldModel


def test_tsdf_esdf_smoke_identity_pose_plane_depth():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)

    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 4, "height": 4}
    pose = {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}
    depth_meta = {"width": 4, "height": 4, "scale_m_per_unit": 0.001}

    depth_u16 = (np.ones((4, 4), dtype=np.uint16) * 1000).astype(np.uint16)
    depth_bytes = depth_u16.tobytes()
    meta = {
        "session_id": "s",
        "frame_id": "f1",
        "timestamp": 1.0,
        "intrinsics": intr,
        "pose": pose,
        "depth_meta": depth_meta,
    }

    for i in range(3):
        meta["frame_id"] = f"f{i}"
        world.update_from_frame(meta, rgb=b"", depth_bytes=depth_bytes, pointcloud_bytes=None)

    stats = world.occupancy.stats()
    assert stats["observed_ratio"] > 0.0

    d = world.query_distance([[0.0, 0.0, 0.0]])[0]
    assert d >= 0.0

    obj = world.export_env_mesh_obj()
    assert isinstance(obj, (bytes, bytearray))
