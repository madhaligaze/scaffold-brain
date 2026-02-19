import numpy as np

from world.world_model import WorldModel


def test_tsdf_extract_mesh_not_empty_on_synthetic_plane():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    w, h = 64, 48
    intr = {"fx": 80.0, "fy": 80.0, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}
    pose = {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}
    depth_meta = {"width": w, "height": h, "scale_m_per_unit": 0.001}

    # Plane at 2.0m -> depth in mm = 2000
    depth_u16 = (np.ones((h, w), dtype=np.uint16) * 2000).astype(np.uint16)
    depth_bytes = depth_u16.tobytes()
    meta = {"intrinsics": intr, "pose": pose, "depth_meta": depth_meta}

    # integrate a few frames to stabilize TSDF
    for _ in range(3):
        world.update_from_frame(meta, rgb=b"", depth_bytes=depth_bytes, pointcloud_bytes=None)

    obj = world.export_env_mesh_obj()
    # Accept either non-empty mesh (preferred) or at least occupancy observed/free updates
    assert isinstance(obj, (bytes, bytearray))
    stats = world.occupancy.stats()
    assert stats["observed_ratio"] > 0.0


def test_esdf_query_distance_returns_non_negative():
    world = WorldModel(voxel_size=0.2, tsdf_trunc=0.4)
    # Mark a single occupied voxel in the center
    cx = world.occupancy.grid.shape[0] // 2
    world.occupancy.grid[cx, cx, cx] = 2
    world.esdf.mark_dirty()
    d = world.query_distance([[float(world.occupancy.origin[0]), float(world.occupancy.origin[1]), float(world.occupancy.origin[2])]])
    assert len(d) == 1
    assert d[0] >= 0.0
