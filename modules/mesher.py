"""
Point Cloud Processor — Обработка облаков точек с Open3D
========================================================
ЗАДАЧА: Превратить сырые точки от ARCore в чистую модель.

УЛУЧШЕНИЯ v4.0:
  - Добавлен downsample для больших облаков
  - Улучшена фильтрация шума
  - Добавлен Poisson reconstruction
  - Статистика качества облака
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except Exception:
    o3d = None  # type: ignore
    OPEN3D_AVAILABLE = False


class PointCloudProcessor:
    """Процессор облаков точек с Open3D."""

    def __init__(self):
        """Инициализация процессора."""
        self.last_pcd: Optional[Any] = None
        self.last_mesh: Optional[Any] = None

    def process_raw_points(
        self,
        raw_points: List[List[float]],
        filter_noise: bool = True,
        compute_normals: bool = True,
        downsample_voxel_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Основной метод обработки облака точек."""
        if not OPEN3D_AVAILABLE:
            return {
                "cleaned_points": raw_points,
                "normals": [],
                "statistics": {"error": "open3d not installed"},
            }

        start_time = time.time()

        if not raw_points:
            return {
                "cleaned_points": [],
                "normals": [],
                "statistics": {"error": "Empty point cloud"},
            }

        pcd = o3d.geometry.PointCloud()
        points_array = np.array(raw_points)

        if points_array.ndim != 2 or points_array.shape[1] < 3:
            return {
                "cleaned_points": [],
                "normals": [],
                "statistics": {"error": "Invalid point cloud shape"},
            }

        # Фильтр: берем только x, y, z (ARCore может присылать [x,y,z,confidence])
        if points_array.shape[1] > 3:
            points_array = points_array[:, :3]

        pcd.points = o3d.utility.Vector3dVector(points_array)

        original_count = len(pcd.points)

        # 2. Фильтрация шума (Statistical Outlier Removal)
        if filter_noise and original_count > 20:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            removed_count = original_count - len(pcd.points)
        else:
            removed_count = 0

        # 3. Downsampling (опционально, для больших облаков)
        downsampled = False
        if downsample_voxel_size and len(pcd.points) > 10000:
            pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
            downsampled = True

        # 4. Вычисление нормалей
        normals_list: List[List[float]] = []
        if compute_normals and len(pcd.points) > 3:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            # Ориентация нормалей (чтобы смотрели "наружу")
            pcd.orient_normals_consistent_tangent_plane(k=15)
            normals_list = np.asarray(pcd.normals).tolist()

        # 5. Сохраняем для потенциального Poisson
        self.last_pcd = pcd

        cleaned_points = np.asarray(pcd.points).tolist()

        elapsed = time.time() - start_time

        return {
            "cleaned_points": cleaned_points,
            "normals": normals_list,
            "statistics": {
                "original_points": original_count,
                "cleaned_points": len(cleaned_points),
                "removed_outliers": removed_count,
                "has_normals": len(normals_list) > 0,
                "downsampled": downsampled,
                "processing_time_ms": int(elapsed * 1000),
            },
        }

    def poisson_reconstruction(
        self,
        depth: int = 9,
        scale: float = 1.1,
        min_density: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        """Poisson Surface Reconstruction — создает красивую mesh поверхность."""
        if not OPEN3D_AVAILABLE or self.last_pcd is None or len(self.last_pcd.points) < 100:
            return None

        start_time = time.time()

        # Проверка нормалей
        if not self.last_pcd.has_normals():
            self.last_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            self.last_pcd.orient_normals_consistent_tangent_plane(k=15)

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.last_pcd,
            depth=depth,
            scale=scale,
        )

        # Удаляем вершины с низкой плотностью (артефакты)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Опционально: сглаживание
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)

        # Вычисляем нормали для освещения
        mesh.compute_vertex_normals()

        self.last_mesh = mesh

        elapsed = time.time() - start_time

        vertices = np.asarray(mesh.vertices).tolist()
        faces = np.asarray(mesh.triangles).tolist()
        normals = np.asarray(mesh.vertex_normals).tolist()

        return {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "statistics": {
                "vertex_count": len(vertices),
                "face_count": len(faces),
                "reconstruction_time_ms": int(elapsed * 1000),
            },
        }

    def export_mesh(self, filepath: str, format: str = "obj") -> bool:
        """Экспорт mesh в файл."""
        del format
        if not OPEN3D_AVAILABLE or self.last_mesh is None:
            return False

        try:
            o3d.io.write_triangle_mesh(filepath, self.last_mesh)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    def get_bounding_box(self) -> Optional[Dict[str, Any]]:
        """Получить bounding box облака точек."""
        if not OPEN3D_AVAILABLE or self.last_pcd is None:
            return None

        bbox = self.last_pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        return {
            "min": {"x": min_bound[0], "y": min_bound[1], "z": min_bound[2]},
            "max": {"x": max_bound[0], "y": max_bound[1], "z": max_bound[2]},
            "center": {
                "x": (min_bound[0] + max_bound[0]) / 2,
                "y": (min_bound[1] + max_bound[1]) / 2,
                "z": (min_bound[2] + max_bound[2]) / 2,
            },
            "size": {
                "width": max_bound[0] - min_bound[0],
                "height": max_bound[2] - min_bound[2],
                "depth": max_bound[1] - min_bound[1],
            },
        }


if __name__ == "__main__":
    print("🧪 TESTING POINT CLOUD PROCESSOR")
    print("=" * 70)

    test_points: List[List[float]] = []
    for x in np.linspace(0, 2, 20):
        for y in np.linspace(0, 2, 20):
            for z in np.linspace(0, 2, 20):
                test_points.append(
                    [
                        x + np.random.normal(0, 0.01),
                        y + np.random.normal(0, 0.01),
                        z + np.random.normal(0, 0.01),
                    ]
                )

    for _ in range(50):
        test_points.append(
            [
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
            ]
        )

    print(f"Test cloud: {len(test_points)} points (with noise)")

    processor = PointCloudProcessor()
    result = processor.process_raw_points(
        test_points,
        filter_noise=True,
        compute_normals=True,
        downsample_voxel_size=0.05,
    )

    print("\n📊 Processing Results:")
    print(f"  Original: {result['statistics'].get('original_points', 'N/A')}")
    print(f"  Cleaned: {result['statistics'].get('cleaned_points', 'N/A')}")
    print(f"  Removed outliers: {result['statistics'].get('removed_outliers', 'N/A')}")
    print(f"  Has normals: {result['statistics'].get('has_normals', False)}")
    print(f"  Processing time: {result['statistics'].get('processing_time_ms', 0)}ms")

    bbox = processor.get_bounding_box()
    if bbox:
        print("\n📦 Bounding Box:")
        print(
            f"  Size: {bbox['size']['width']:.2f} x {bbox['size']['depth']:.2f} x {bbox['size']['height']:.2f} m"
        )

    print("\n✓ Test passed!")
