from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ICPResult:
    ok: bool
    fitness: float | None = None
    rmse: float | None = None
    transform_src_to_tgt: np.ndarray | None = None
    reason: str | None = None


def run_icp_point_to_plane(
    *,
    source_pts_world: np.ndarray,
    target_pts_world: np.ndarray,
    max_correspondence_distance_m: float = 0.20,
    voxel_downsample_m: float = 0.06,
    max_points: int = 6000,
) -> ICPResult:
    """
    Runs a light ICP alignment: source -> target in WORLD frame.

    Returns transform that maps source points into target frame.
    """
    try:
        import open3d as o3d
    except Exception as exc:
        return ICPResult(ok=False, reason=f"open3d_unavailable:{exc}")

    if source_pts_world is None or target_pts_world is None:
        return ICPResult(ok=False, reason="missing_points")
    source_pts_world = np.asarray(source_pts_world, dtype=np.float64).reshape(-1, 3)
    target_pts_world = np.asarray(target_pts_world, dtype=np.float64).reshape(-1, 3)
    if source_pts_world.shape[0] < 200 or target_pts_world.shape[0] < 200:
        return ICPResult(ok=False, reason="too_few_points")

    def _subsample(pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] <= max_points:
            return pts
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        return pts[idx]

    src_pts = _subsample(source_pts_world)
    tgt_pts = _subsample(target_pts_world)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    if voxel_downsample_m > 1e-6:
        src = src.voxel_down_sample(float(voxel_downsample_m))
        tgt = tgt.voxel_down_sample(float(voxel_downsample_m))

    try:
        tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=30))
    except Exception:
        pass

    init = np.eye(4, dtype=np.float64)

    try:
        if len(tgt.normals) > 0:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            float(max_correspondence_distance_m),
            init,
            estimation,
        )
        T = np.asarray(reg.transformation, dtype=np.float64)
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        return ICPResult(ok=True, fitness=fitness, rmse=rmse, transform_src_to_tgt=T)
    except Exception as exc:
        return ICPResult(ok=False, reason=f"icp_failed:{exc}")
