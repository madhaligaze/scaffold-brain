"""TSDF integrator for optional mesh extraction from depth frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d

    _O3D_AVAILABLE = True
except Exception:
    _O3D_AVAILABLE = False


@dataclass
class TsdfConfig:
    voxel_length: float = 0.02
    sdf_trunc: float = 0.08
    max_depth: float = 8.0


def _pose7_to_matrix(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    rot = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return tf


class TSDFIntegrator:
    def __init__(self, config: Optional[TsdfConfig] = None):
        self.config = config or TsdfConfig()
        self.frames_integrated = 0
        self._volume = None
        if _O3D_AVAILABLE:
            self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=float(self.config.voxel_length),
                sdf_trunc=float(self.config.sdf_trunc),
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
            )

    @property
    def available(self) -> bool:
        return _O3D_AVAILABLE and self._volume is not None

    def integrate_depth(
        self,
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        pose_world_from_camera_7: Tuple[float, float, float, float, float, float, float],
        depth_trunc: Optional[float] = None,
    ) -> bool:
        if not self.available:
            return False
        if depth_m.ndim != 2:
            return False

        h, w = depth_m.shape
        if h < 2 or w < 2 or fx <= 0 or fy <= 0:
            return False

        depth_img = o3d.geometry.Image(depth_m.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8)),
            depth=depth_img,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc or self.config.max_depth),
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, float(fx), float(fy), float(cx), float(cy))
        tx, ty, tz, qx, qy, qz, qw = pose_world_from_camera_7
        extrinsic = _pose7_to_matrix(tx, ty, tz, qx, qy, qz, qw)

        self._volume.integrate(rgbd, intrinsic, extrinsic)
        self.frames_integrated += 1
        return True

    def extract_mesh(self):
        if not self.available:
            return None
        mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self):
        if not self.available:
            return None
        return self._volume.extract_point_cloud()
