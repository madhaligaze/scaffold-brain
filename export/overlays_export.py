from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from world.occupancy import FREE, UNKNOWN


def export_occupancy_npz(world_model, out_path: Path) -> dict[str, Any]:
    """
    MVP overlay export for Android:
      - occupancy grid as compressed npz (grid + origin + voxel_size)

    This is intentionally simple and stable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    occ = world_model.occupancy
    np.savez_compressed(
        str(out_path),
        grid=occ.grid.astype(np.uint8),
        origin=np.asarray(occ.origin, dtype=np.float32),
        voxel_size=np.asarray([float(occ.voxel_size)], dtype=np.float32),
    )
    return {"format": "npz", "path": str(out_path)}


def export_occupancy_slice_png(
    world_model,
    out_path: Path,
    *,
    axis: str = "z",
    frac: float = 0.2,
) -> dict[str, Any] | None:
    """
    Optional quick-look PNG slice. If Pillow is unavailable, return None.
    UNKNOWN=0 (dark), FREE=1 (mid), OCCUPIED=2 (bright).
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    occ = world_model.occupancy
    g = occ.grid.astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    axis = axis.lower().strip()
    frac = float(np.clip(frac, 0.0, 1.0))

    if axis == "x":
        i = int(frac * max(1, g.shape[0] - 1))
        sl = g[i, :, :]
    elif axis == "y":
        i = int(frac * max(1, g.shape[1] - 1))
        sl = g[:, i, :]
    else:
        i = int(frac * max(1, g.shape[2] - 1))
        sl = g[:, :, i]

    # Map 0/1/2 -> 0/128/255
    img = (sl.astype(np.float32) * 127.5).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L")
    pil.save(str(out_path))
    return {"format": "png", "path": str(out_path), "axis": axis, "frac": frac}



def _voxel_boxes_mesh(
    *,
    indices_ijk: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> trimesh.Trimesh:
    """
    Build a single mesh consisting of axis-aligned voxel cubes for each ijk.

    indices_ijk: (N,3) int array in occupancy grid coordinates.
    origin: world-space origin of grid.
    voxel_size: meters.
    """
    indices_ijk = np.asarray(indices_ijk, dtype=np.int32).reshape(-1, 3)
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    s = float(voxel_size)

    base = trimesh.creation.box(extents=(s, s, s))
    meshes: list[trimesh.Trimesh] = []
    for i, j, k in indices_ijk:
        cx, cy, cz = origin + (np.asarray([i, j, k], dtype=np.float32) + 0.5) * s
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [float(cx), float(cy), float(cz)]
        meshes.append(base.copy().apply_transform(transform))

    if not meshes:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )

    return trimesh.util.concatenate(meshes)



def export_unknown_heatmap_glb(
    world_model,
    out_path: Path,
    *,
    stride: int = 2,
    max_voxels: int = 25000,
) -> dict[str, Any]:
    """
    Export unknown-space overlay as GLB geometry (voxel cubes).

    Conservative: visualizes UNKNOWN voxels from occupancy.
    """
    occ = world_model.occupancy
    g = occ.grid
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s = max(1, int(stride))
    unknown = g == UNKNOWN
    if s > 1:
        unknown = unknown[::s, ::s, ::s]

    idx = np.argwhere(unknown)
    if idx.shape[0] > int(max_voxels):
        sel = np.random.choice(idx.shape[0], size=int(max_voxels), replace=False)
        idx = idx[sel]

    if s > 1:
        idx = idx * s

    mesh = _voxel_boxes_mesh(indices_ijk=idx, origin=occ.origin, voxel_size=float(occ.voxel_size))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="unknown_heatmap")
    glb = scene.export(file_type="glb")
    out_path.write_bytes(glb)

    return {"format": "glb", "path": str(out_path), "stride": s, "count": int(idx.shape[0])}



def export_clearance_violations_glb(
    world_model,
    out_path: Path,
    *,
    min_clearance_m: float,
    stride: int = 2,
    max_voxels: int = 25000,
) -> dict[str, Any]:
    """
    Export clearance violations as GLB geometry (voxel cubes).

    A voxel is a violation if it is FREE but ESDF distance < min_clearance_m.
    """
    occ = world_model.occupancy
    g = occ.grid
    out_path.parent.mkdir(parents=True, exist_ok=True)

    esdf = getattr(world_model, "esdf", None)
    dist = None
    try:
        if esdf is not None:
            esdf.build_from_occupancy(g, np.asarray(occ.origin, dtype=np.float32), float(occ.voxel_size))
            dist = getattr(esdf, "_dist_m", None)
    except Exception:
        dist = None

    if dist is None:
        scene = trimesh.Scene()
        out_path.write_bytes(scene.export(file_type="glb"))
        return {"format": "glb", "path": str(out_path), "count": 0, "reason": "esdf_unavailable"}

    dist = np.asarray(dist, dtype=np.float32)
    s = max(1, int(stride))

    free = g == FREE
    viol = free & (dist < float(min_clearance_m))
    if s > 1:
        viol = viol[::s, ::s, ::s]

    idx = np.argwhere(viol)
    if idx.shape[0] > int(max_voxels):
        sel = np.random.choice(idx.shape[0], size=int(max_voxels), replace=False)
        idx = idx[sel]

    if s > 1:
        idx = idx * s

    mesh = _voxel_boxes_mesh(indices_ijk=idx, origin=occ.origin, voxel_size=float(occ.voxel_size))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="clearance_violations")
    glb = scene.export(file_type="glb")
    out_path.write_bytes(glb)

    return {
        "format": "glb",
        "path": str(out_path),
        "stride": s,
        "count": int(idx.shape[0]),
        "min_clearance_m": float(min_clearance_m),
    }
