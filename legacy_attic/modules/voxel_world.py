"""
VoxelWorld — воксельная карта рабочего пространства.
====================================================

Этап 1:
- tri-state мир: UNKNOWN/FREE/OCCUPIED
- depth free-space carving: FREE по лучу, OCCUPIED на поверхности
- консервативные коллизии: UNKNOWN можно считать заблокированным
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


@dataclass
class Obstacle:
    id: str
    type: str
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]


class VoxelCollisionSolver:
    """Compatibility collision solver. Keeps structure unchanged if no precise data."""

    def __init__(self, clearance: float = 0.15):
        self.clearance = clearance

    def resolve_collisions(self, nodes: List[Dict], beams: List[Dict], obstacles: List[Obstacle]):
        return {
            "success": True,
            "nodes": nodes,
            "beams": beams,
            "moved_nodes": 0,
            "removed_beams": 0,
            "obstacles": len(obstacles),
        }


class VoxelWorld:
    """Sparse 3D grid for occupancy/free/unknown queries."""

    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1

    PIPE = 10
    WALL = 11
    FLOOR = 12

    def __init__(
        self,
        resolution: float = 0.05,
        bounds_min: Tuple[float, float, float] = (-4.0, -4.0, -1.0),
        bounds_max: Tuple[float, float, float] = (4.0, 4.0, 3.0),
    ):
        self.resolution = float(resolution)
        self.bounds_min = tuple(map(float, bounds_min))
        self.bounds_max = tuple(map(float, bounds_max))

        self.occupied: Set[Tuple[int, int, int]] = set()
        self.free: Set[Tuple[int, int, int]] = set()

        # Stage 3: safety overlays inflated around tentative/needs_scan objects for planning.
        self._safety_occupied: Set[Tuple[int, int, int]] = set()
        self._safety_types: Dict[Tuple[int, int, int], int] = {}

        self._types: Dict[Tuple[int, int, int], int] = {}
        self._grid: Dict[Tuple[int, int, int], int] = {}
        self._last_depth_stats: Dict[str, float] = {}

    def add_point_cloud(self, points: List[List[float]], vtype: int = OCCUPIED) -> int:
        """Fallback fill: point cloud points become OCCUPIED."""
        count = 0
        for p in points:
            if len(p) < 3:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if z < self.bounds_min[2] - 1.0 or z > self.bounds_max[2] + 30.0:
                continue

            coord = self._to_grid(x, y, z)
            if not self._in_bounds_idx(coord):
                continue

            if coord not in self.occupied:
                self.occupied.add(coord)
                self.free.discard(coord)
                self._types[coord] = vtype
                self._grid[coord] = vtype
                count += 1

        return count

    def mark_box(self, center: Dict, dims: Dict, vtype: int = OCCUPIED) -> int:
        """Fill an approximate box region as OCCUPIED (vectorized grid generation)."""
        cx, cy, cz = float(center["x"]), float(center["y"]), float(center["z"])
        hw = float(dims.get("width", 0.2)) / 2
        hd = float(dims.get("depth", 0.2)) / 2
        hh = float(dims.get("height", 0.5)) / 2
        r = self.resolution

        xs = np.arange(cx - hw, cx + hw + r, r, dtype=np.float64)
        ys = np.arange(cy - hd, cy + hd + r, r, dtype=np.float64)
        zs = np.arange(cz - hh, cz + hh + r, r, dtype=np.float64)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            return 0

        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        coords = np.stack(
            [
                np.floor(xx / r).astype(np.int64),
                np.floor(yy / r).astype(np.int64),
                np.floor(zz / r).astype(np.int64),
            ],
            axis=-1,
        ).reshape(-1, 3)

        if coords.size == 0:
            return 0

        mn = np.array(self._to_grid(*self.bounds_min), dtype=np.int64)
        mx = np.array(self._to_grid(*self.bounds_max), dtype=np.int64)
        in_bounds = np.all((coords >= mn) & (coords <= mx), axis=1)
        if not np.any(in_bounds):
            return 0

        coords = np.unique(coords[in_bounds], axis=0)

        count = 0
        for c in coords:
            coord = (int(c[0]), int(c[1]), int(c[2]))
            if coord in self.occupied:
                continue
            self.occupied.add(coord)
            self.free.discard(coord)
            self._types[coord] = vtype
            self._grid[coord] = vtype
            count += 1

        return count

    def ingest_yolo_detections(self, detections: List[Dict], fallback_depth: float = 2.0) -> None:
        """Coarse fill from 2D detections (without true depth)."""
        type_map = {
            "pipe_obstacle": self.PIPE,
            "wall": self.WALL,
            "floor_slab": self.FLOOR,
            "cable_tray": self.PIPE,
            "column": self.OCCUPIED,
        }
        for det in detections:
            pos = det.get("position", {})
            dims = det.get("dimensions", {"width": 0.3, "depth": 0.3, "height": 0.5})
            if not pos:
                continue
            if float(pos.get("z", 0.0) or 0.0) == 0.0:
                pos = {**pos, "z": float(fallback_depth)}
            vtype = type_map.get(det.get("type", ""), self.OCCUPIED)
            self.mark_box(pos, dims, vtype=vtype)

    def ingest_depth_map(
        self,
        depth_bytes: bytes,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx_px: float,
        cy_px: float,
        camera_pose: List[float],
        depth_scale: float = 1000.0,
        confidence_bytes: Optional[bytes] = None,
        confidence_threshold: int = 1,
        max_range: float = 8.0,
        pixel_step: int = 4,
    ) -> Dict[str, float]:
        """Integrate depth into FREE/OCCUPIED sets."""
        stats = {"samples": 0.0, "occupied_added": 0.0, "free_added": 0.0, "conflicts": 0.0}

        if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
            self._last_depth_stats = stats
            return stats
        if len(camera_pose) < 7:
            self._last_depth_stats = stats
            return stats
        if len(depth_bytes) < width * height * 2:
            self._last_depth_stats = stats
            return stats

        depth_u16 = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(height, width)
        depth_m = depth_u16.astype(np.float32) / float(depth_scale)

        conf = None
        if confidence_bytes is not None and len(confidence_bytes) >= width * height:
            conf = np.frombuffer(confidence_bytes, dtype=np.uint8).reshape(height, width)

        rot = self._quat_to_rotation(*camera_pose[3:7])
        cam_t = np.array(camera_pose[:3], dtype=np.float32)

        step = max(1, int(pixel_step))
        u_coords = np.arange(0, width, step, dtype=np.int32)
        v_coords = np.arange(0, height, step, dtype=np.int32)
        uu, vv = np.meshgrid(u_coords, v_coords)

        sampled_depth = depth_m[vv, uu]
        valid_mask = (sampled_depth > 0.0) & (sampled_depth <= float(max_range))

        if conf is not None:
            sampled_conf = conf[vv, uu]
            valid_mask &= sampled_conf >= int(confidence_threshold)

        if not np.any(valid_mask):
            self._last_depth_stats = stats
            return stats

        u_valid = uu[valid_mask].astype(np.float32)
        v_valid = vv[valid_mask].astype(np.float32)
        d_valid = sampled_depth[valid_mask].astype(np.float32)

        xc = (u_valid - float(cx_px)) * d_valid / float(fx)
        yc = (v_valid - float(cy_px)) * d_valid / float(fy)
        cam_points = np.stack((xc, yc, d_valid), axis=1)
        world_points = (cam_points @ rot.T) + cam_t

        in_bounds_mask = (
            (world_points[:, 0] >= float(self.bounds_min[0]))
            & (world_points[:, 0] <= float(self.bounds_max[0]))
            & (world_points[:, 1] >= float(self.bounds_min[1]))
            & (world_points[:, 1] <= float(self.bounds_max[1]))
            & (world_points[:, 2] >= float(self.bounds_min[2]))
            & (world_points[:, 2] <= float(self.bounds_max[2]))
        )

        valid_world_points = world_points[in_bounds_mask]
        stats["samples"] = float(valid_world_points.shape[0])

        for p_world in valid_world_points:
            free_added, conflicts = self._mark_free_ray(cam_t, p_world)
            stats["free_added"] += float(free_added)
            stats["conflicts"] += float(conflicts)

            occ_coord = self._to_grid(float(p_world[0]), float(p_world[1]), float(p_world[2]))
            if occ_coord in self.free:
                stats["conflicts"] += 1.0
                self.free.discard(occ_coord)

            if occ_coord not in self.occupied:
                self.occupied.add(occ_coord)
                self._types.setdefault(occ_coord, self.OCCUPIED)
                self._grid[occ_coord] = self._types.get(occ_coord, self.OCCUPIED)
                stats["occupied_added"] += 1.0

        self._last_depth_stats = stats
        return stats

    def get_last_depth_stats(self) -> Dict[str, float]:
        return dict(self._last_depth_stats or {})

    def get_coverage_metrics(self) -> Dict[str, float]:
        total = float(self._total_voxels_in_bounds())
        known = float(len(self.occupied) + len(self.free))
        unknown = max(0.0, total - known)
        coverage = (known / total) if total > 0 else 0.0
        holes = (unknown / total) if total > 0 else 1.0
        return {
            "bounds_total_voxels": total,
            "known_voxels": known,
            "unknown_voxels": unknown,
            "coverage": coverage,
            "holes": holes,
            # Backward-compatible alias.
            "unknown": holes,
        }

    def get_quality_metrics(self) -> Dict[str, float]:
        coverage = self.get_coverage_metrics()
        return {
            "coverage_pct": coverage["coverage"],
            "unknown_pct": coverage["holes"],
            "last_depth_stats": self.get_last_depth_stats(),
            **coverage,
        }

    def segment_unknown_ratio(
        self,
        start_world: Tuple[float, float, float],
        end_world: Tuple[float, float, float],
        step_m: Optional[float] = None,
        clearance: float = 0.0,
    ) -> float:
        """Estimate UNKNOWN ratio along a segment.

        UNKNOWN := cells that are neither in OCCUPIED nor in FREE.

        Useful for scoring next-best-view proposals and conservative gating.
        """
        p1 = np.array(self._as_tuple(start_world), dtype=float)
        p2 = np.array(self._as_tuple(end_world), dtype=float)
        dist = float(np.linalg.norm(p2 - p1))
        if dist <= 1e-9:
            return 1.0

        if step_m is None:
            step_m = max(float(self.resolution), 1e-3)

        steps = max(int(dist / float(step_m)), 1)
        extra = max(0, int(float(clearance) / float(self.resolution)))

        unknown = 0
        total = 0
        for i in range(steps + 1):
            t = i / float(steps)
            p = p1 + (p2 - p1) * t
            vx, vy, vz = self._to_grid(*p)
            for dx in range(-extra, extra + 1):
                for dz in range(-extra, extra + 1):
                    g = (vx + dx, vy, vz + dz)
                    total += 1
                    if (g not in self.occupied) and (g not in self.free):
                        unknown += 1

        if total <= 0:
            return 1.0
        return float(unknown) / float(total)

    def is_blocked(
        self,
        start: Union[Tuple[float, float, float], Dict[str, float]],
        end: Union[Tuple[float, float, float], Dict[str, float]],
        clearance: float = 0.05,
        unknown_is_blocked: bool = True,
    ) -> bool:
        """Raymarching collision check; UNKNOWN can be treated as blocked."""
        p1 = np.array(self._as_tuple(start), dtype=float)
        p2 = np.array(self._as_tuple(end), dtype=float)
        dist = float(np.linalg.norm(p2 - p1))

        if dist < self.resolution:
            return False

        steps = max(int(dist / (self.resolution / 2)), 2)
        extra = max(0, int(float(clearance) / self.resolution))

        for i in range(steps + 1):
            t = i / steps
            p = p1 + (p2 - p1) * t
            vx, vy, vz = self._to_grid(float(p[0]), float(p[1]), float(p[2]))
            for dx in range(-extra, extra + 1):
                for dz in range(-extra, extra + 1):
                    check = (vx + dx, vy, vz + dz)

                    if check in self.occupied or check in self._safety_occupied:
                        vtype = self._types.get(check, self.OCCUPIED)
                        if vtype != self.FLOOR:
                            return True
                        continue

                    if unknown_is_blocked and (check not in self.free):
                        return True

        return False


    # ── Stage 3: safety policy overlays ────────────────────────────────────

    def clear_safety_overlays(self) -> None:
        self._safety_occupied.clear()
        self._safety_types.clear()

    def apply_safety_overlays(
        self,
        world_objects: List[Dict[str, Any]],
        *,
        clearance_min: float = 0.15,
        clearance_tentative: float = 0.30,
        clearance_needs_scan: float = 0.50,
        max_points_per_object: int = 2000,
    ) -> Dict[str, Any]:
        """Inflate occupied space around objects for conservative planning.

        This does NOT assert new geometry - it is a planning-time overlay.
        """
        self.clear_safety_overlays()
        if not world_objects:
            return {"overlay_voxels": 0, "objects": 0}

        overlay_count = 0

        def _buffer_for(o: Dict[str, Any]) -> float:
            st = (o.get("status") or "TENTATIVE").upper()
            needs = bool(o.get("needs_scan", False))
            if needs:
                return float(clearance_needs_scan)
            if st == "CONFIRMED":
                return float(clearance_min)
            if st == "TENTATIVE":
                return float(clearance_tentative)
            return float(clearance_needs_scan)

        def _mark_sphere(center_xyz: np.ndarray, radius_m: float) -> None:
            nonlocal overlay_count
            r = float(max(self.resolution, 1e-6))
            extra = int(math.ceil(radius_m / r))
            vx, vy, vz = self._to_grid(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2]))
            for dx in range(-extra, extra + 1):
                for dy in range(-extra, extra + 1):
                    for dz in range(-extra, extra + 1):
                        coord = (vx + dx, vy + dy, vz + dz)
                        if not self._in_bounds_idx(coord):
                            continue
                        if coord in self._safety_occupied:
                            continue
                        self._safety_occupied.add(coord)
                        self._safety_types[coord] = self.OCCUPIED
                        overlay_count += 1

        for o in world_objects:
            pose = o.get("pose", {}) or {}
            dims = o.get("dimensions", {}) or {}
            buf = _buffer_for(o)

            g = (o.get("geometry_type") or "MESH_PROXY").upper()

            if g == "CYLINDER" and "axis" in pose and "radius" in dims:
                try:
                    axis = np.array(pose.get("axis", [0, 0, 1]), dtype=np.float64)
                    an = float(np.linalg.norm(axis))
                    if an < 1e-9:
                        axis = np.array([0, 0, 1], dtype=np.float64)
                        an = 1.0
                    axis = axis / an
                    origin = np.array(pose.get("position", [0, 0, 0]), dtype=np.float64)

                    seg = o.get("observable_segment") or {}
                    if seg:
                        t0 = float(seg.get("t0", -0.1))
                        t1 = float(seg.get("t1", 0.1))
                    else:
                        L = float(dims.get("length", 0.2))
                        t0, t1 = -L / 2.0, L / 2.0

                    for h in (o.get("extension_hypotheses") or []):
                        if h.get("stop_reason") in ("OCCUPIED", "OTHER_OBJECT", "MAX_LENGTH") and float(h.get("confidence", 0.0)) >= 0.6:
                            if h.get("end") == "neg_end":
                                t0 = min(t0, float(h.get("t_end", t0)))
                            elif h.get("end") == "pos_end":
                                t1 = max(t1, float(h.get("t_end", t1)))

                    radius = float(dims.get("radius", 0.05)) + buf
                    length = float(abs(t1 - t0))
                    n_samples = max(2, int(length / max(self.resolution, 0.02)))
                    n_samples = min(n_samples, max_points_per_object)

                    for k in range(n_samples + 1):
                        t = float(t0 + (t1 - t0) * (k / max(1, n_samples)))
                        p = origin + axis * t
                        _mark_sphere(p, radius)
                except Exception:
                    pass
                continue

            try:
                center = np.array(pose.get("position", [0, 0, 0]), dtype=np.float64)
                if "dx" in dims:
                    dx = float(dims.get("dx", 0.2)) + 2 * buf
                    dy = float(dims.get("dy", 0.2)) + 2 * buf
                    dz = float(dims.get("dz", 0.2)) + 2 * buf
                else:
                    r0 = float(dims.get("radius", 0.05)) + buf
                    L0 = float(dims.get("length", 0.2)) + 2 * buf
                    dx, dy, dz = 2 * r0, 2 * r0, L0

                r = float(self.resolution)
                xs = np.arange(center[0] - dx / 2.0, center[0] + dx / 2.0 + r, r)
                ys = np.arange(center[1] - dy / 2.0, center[1] + dy / 2.0 + r, r)
                zs = np.arange(center[2] - dz / 2.0, center[2] + dz / 2.0 + r, r)

                for x in xs:
                    for y in ys:
                        for z in zs:
                            coord = self._to_grid(float(x), float(y), float(z))
                            if not self._in_bounds_idx(coord):
                                continue
                            if coord in self._safety_occupied:
                                continue
                            self._safety_occupied.add(coord)
                            self._safety_types[coord] = self.OCCUPIED
                            overlay_count += 1
            except Exception:
                continue

        return {"overlay_voxels": int(overlay_count), "objects": int(len(world_objects))}

    def get_state(self, x: float, y: float, z: float) -> int:
        coord = self._to_grid(x, y, z)
        if coord in self.occupied or coord in self._safety_occupied:
            return self.OCCUPIED
        if coord in self.free:
            return self.FREE
        return self.UNKNOWN

    def get_type(self, x: float, y: float, z: float) -> int:
        """Backward-compatible alias."""
        state = self.get_state(x, y, z)
        if state == self.OCCUPIED:
            return self._types.get(self._to_grid(x, y, z), self.OCCUPIED)
        if state == self.FREE:
            return self.FREE
        return self.UNKNOWN

    def get_floor_z(self, x: float, y: float, search_below: float = 5.0) -> Optional[float]:
        vx, vy, vz = self._to_grid(x, y, 0.0)
        end_vi = vz - int(search_below / self.resolution)
        for vi in range(vz, end_vi, -1):
            if self._types.get((vx, vy, vi), self.FREE) == self.FLOOR:
                return vi * self.resolution
        return None

    def clear(self) -> None:
        self.occupied.clear()
        self.free.clear()
        self._safety_occupied.clear()
        self._safety_types.clear()
        self._types.clear()
        self._grid.clear()
        self._last_depth_stats = {}

    @property
    def total_voxels(self) -> int:
        return len(self.occupied)

    @property
    def total_known_voxels(self) -> int:
        return len(self.occupied) + len(self.free)

    def to_ar_mesh(self) -> Dict:
        voxels = []
        for (vx, vy, vz) in self.occupied:
            voxels.append(
                {
                    "x": vx * self.resolution,
                    "y": vy * self.resolution,
                    "z": vz * self.resolution,
                    "state": "occupied",
                    "type": self._types.get((vx, vy, vz), self.OCCUPIED),
                }
            )
        for (vx, vy, vz) in self.free:
            voxels.append(
                {
                    "x": vx * self.resolution,
                    "y": vy * self.resolution,
                    "z": vz * self.resolution,
                    "state": "free",
                    "type": self.FREE,
                }
            )
        return {
            "voxels": voxels,
            "resolution": self.resolution,
            "bounds": {"min": self.bounds_min, "max": self.bounds_max},
        }

    def _mark_free_ray(self, origin_w: np.ndarray, surface_w: np.ndarray) -> Tuple[int, int]:
        o = origin_w.astype(np.float32)
        s = surface_w.astype(np.float32)
        direction = s - o
        dist = float(np.linalg.norm(direction))
        if dist <= 1e-6:
            return 0, 0

        direction /= dist
        step = self.resolution / 2.0
        n = max(int(dist / step), 1)

        free_added = 0
        conflicts = 0
        for i in range(n):
            p = o + direction * (i * step)
            if not self._in_bounds_xyz(float(p[0]), float(p[1]), float(p[2])):
                continue
            coord = self._to_grid(float(p[0]), float(p[1]), float(p[2]))
            if coord in self.occupied:
                conflicts += 1
                continue
            if coord not in self.free:
                self.free.add(coord)
                free_added += 1
        return free_added, conflicts

    def raycast_distance(
        self,
        origin_world: Tuple[float, float, float],
        direction_world: Tuple[float, float, float],
        max_dist: float = 8.0,
        step: Optional[float] = None,
        unknown_is_blocked: bool = False,
        include_safety: bool = True,
        return_unknown_hit: bool = False,
    ) -> Optional[float]:
        """Approximate raycast against OCCUPIED voxels.

        Returns distance to first OCCUPIED voxel along the ray, or None if nothing hit.

        Compatibility:
        - if return_unknown_hit=False: returns Optional[float]
        - if return_unknown_hit=True: returns Optional[Tuple[float, bool]]
          where bool indicates whether hit happened on UNKNOWN.
        """
        ox, oy, oz = map(float, origin_world)
        dx, dy, dz = map(float, direction_world)
        nrm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if nrm <= 1e-9:
            return None

        dx, dy, dz = dx / nrm, dy / nrm, dz / nrm
        # A reasonable default: half-voxel step
        if step is None:
            step = max(0.5 * float(self.resolution), 1e-3)

        dist = 0.0
        while dist <= max_dist:
            x = ox + dx * dist
            y = oy + dy * dist
            z = oz + dz * dist
            if not self._in_bounds_xyz(x, y, z):
                dist += step
                continue
            g = self._to_grid(x, y, z)
            if include_safety and g in self._safety_occupied:
                return (dist, False) if return_unknown_hit else dist
            if g in self.occupied:
                return (dist, False) if return_unknown_hit else dist
            if unknown_is_blocked and g not in self.free:
                return (dist, True) if return_unknown_hit else dist
            dist += step
        return None

    def unknown_ratio_along_ray(
        self,
        origin_world: Tuple[float, float, float],
        direction_world: Tuple[float, float, float],
        max_dist: float = 8.0,
        step: Optional[float] = None,
    ) -> float:
        """Estimate UNKNOWN ratio along a ray segment."""
        ox, oy, oz = map(float, origin_world)
        dx, dy, dz = map(float, direction_world)
        nrm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if nrm <= 1e-9:
            return 1.0
        dx, dy, dz = dx / nrm, dy / nrm, dz / nrm
        if step is None:
            step = max(float(self.resolution), 1e-3)

        unknown = 0
        total = 0
        dist = 0.0
        while dist <= float(max_dist):
            x = ox + dx * dist
            y = oy + dy * dist
            z = oz + dz * dist
            if self._in_bounds_xyz(x, y, z):
                g = self._to_grid(x, y, z)
                total += 1
                if g not in self.occupied and g not in self.free:
                    unknown += 1
            dist += float(step)
        if total <= 0:
            return 1.0
        return float(unknown) / float(total)

    def unknown_fraction_in_box(
        self,
        center_world: Tuple[float, float, float],
        half_extents_m: Tuple[float, float, float],
        sample_step_vox: int = 2,
        step: Optional[float] = None,
    ) -> float:
        """Estimate UNKNOWN fraction in an axis-aligned box.

        UNKNOWN := grid cells not in OCCUPIED and not in FREE.
        Used for conservative gating (do not plan / do not confirm completions).
        """
        cx, cy, cz = center_world
        hx, hy, hz = half_extents_m

        if step is not None:
            sample_step_vox = max(1, int(round(float(step) / max(float(self.resolution), 1e-9))))

        # Convert to grid bounds
        gmin = self._to_grid(cx - hx, cy - hy, cz - hz)
        gmax = self._to_grid(cx + hx, cy + hy, cz + hz)

        # Ensure ordering
        ix0, iy0, iz0 = (min(gmin[0], gmax[0]), min(gmin[1], gmax[1]), min(gmin[2], gmax[2]))
        ix1, iy1, iz1 = (max(gmin[0], gmax[0]), max(gmin[1], gmax[1]), max(gmin[2], gmax[2]))

        total = 0
        unknown = 0
        step = max(1, int(sample_step_vox))
        for ix in range(ix0, ix1 + 1, step):
            for iy in range(iy0, iy1 + 1, step):
                for iz in range(iz0, iz1 + 1, step):
                    total += 1
                    g = (ix, iy, iz)
                    if g not in self.occupied and g not in self.free:
                        unknown += 1
        if total <= 0:
            return 1.0
        return float(unknown) / float(total)

    def _to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        r = self.resolution
        return (int(math.floor(x / r)), int(math.floor(y / r)), int(math.floor(z / r)))

    def _in_bounds_xyz(self, x: float, y: float, z: float) -> bool:
        return (
            self.bounds_min[0] <= x <= self.bounds_max[0]
            and self.bounds_min[1] <= y <= self.bounds_max[1]
            and self.bounds_min[2] <= z <= self.bounds_max[2]
        )

    def _in_bounds_idx(self, coord: Tuple[int, int, int]) -> bool:
        x, y, z = coord
        mn = self._to_grid(*self.bounds_min)
        mx = self._to_grid(*self.bounds_max)
        return (mn[0] <= x <= mx[0]) and (mn[1] <= y <= mx[1]) and (mn[2] <= z <= mx[2])

    def _total_voxels_in_bounds(self) -> int:
        mn = self._to_grid(*self.bounds_min)
        mx = self._to_grid(*self.bounds_max)
        return (mx[0] - mn[0] + 1) * (mx[1] - mn[1] + 1) * (mx[2] - mn[2] + 1)

    @staticmethod
    def _as_tuple(p: Union[Tuple[float, float, float], Dict[str, float]]) -> Tuple[float, float, float]:
        if isinstance(p, dict):
            return (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
        return (float(p[0]), float(p[1]), float(p[2]))

    @staticmethod
    def _quat_to_rotation(qx, qy, qz, qw) -> np.ndarray:
        return np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]
        )
