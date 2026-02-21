from __future__ import annotations

import math

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class OccupancyGrid:
    """
    STAGE B: occupancy + uncertainty as a probabilistic-ish model.
    - grid: UNKNOWN/FREE/OCCUPIED
    - weights: how many observations touched this voxel (uint16)
    - conflict_count: how often a voxel flipped between FREE/OCCUPIED
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.2,
        # Use a cubic grid by default to avoid z-mid indexing failures in downstream
        # ESDF/mesh logic and tests.
        dims: tuple[int, int, int] = (128, 128, 128),
        origin: tuple[float, float, float] = (-12.8, -12.8, -1.0),
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.origin = np.array(origin, dtype=np.float32)
        self.grid = np.zeros(dims, dtype=np.uint8)  # UNKNOWN
        self.weights = np.zeros(dims, dtype=np.uint16)
        self.conflict_count = 0

    def _pix_to_cam(self, u: int, v: int, z_m: float, intr: dict) -> np.ndarray:
        x = (float(u) - float(intr["cx"])) * z_m / float(intr["fx"])
        y = (float(v) - float(intr["cy"])) * z_m / float(intr["fy"])
        return np.array([x, y, z_m], dtype=np.float32)

    def _in_bounds(self, idx: np.ndarray) -> bool:
        shp = np.asarray(self.grid.shape, dtype=np.int32)
        return bool(np.all(idx >= 0) and np.all(idx < shp))

    def in_bounds(self, *args) -> bool:
        # Public helper expected by tests and validators.
        try:
            if len(args) == 1:
                a = np.asarray(args[0], dtype=np.int32).reshape(3)
                return self._in_bounds(a)
            if len(args) == 3:
                x, y, z = int(args[0]), int(args[1]), int(args[2])
                sx, sy, sz = self.grid.shape
                return 0 <= x < int(sx) and 0 <= y < int(sy) and 0 <= z < int(sz)
        except Exception:
            return False
        return False

    def _touch(self, idx: np.ndarray, new_state: int) -> bool:
        if not self._in_bounds(idx):
            return False
        x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
        prev = int(self.grid[x, y, z])
        if prev in (FREE, OCCUPIED) and new_state in (FREE, OCCUPIED) and prev != new_state:
            self.conflict_count += 1
        self.grid[x, y, z] = np.uint8(new_state)
        if self.weights[x, y, z] < np.uint16(65535):
            self.weights[x, y, z] = np.uint16(int(self.weights[x, y, z]) + 1)
        return True

    def integrate_depth(self, depth_u16: np.ndarray, intr: dict, pose: dict, depth_scale: float) -> None:
        # Mark FREE along ray, OCCUPIED at surface (cheap/fast, conservative)
        from world.transform import pose_to_matrix

        T = pose_to_matrix(pose)
        h, w = depth_u16.shape
        step = 2

        cam_w = (T @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))[:3]

        for v in range(0, h, step):
            for u in range(0, w, step):
                d = int(depth_u16[v, u])
                if d <= 0:
                    continue
                z = float(d) * float(depth_scale)
                if z <= 0.05 or z > 10.0:
                    continue

                p_cam = self._pix_to_cam(u, v, z, intr)
                p_w = (T @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float32))[:3]

                dir = p_w - cam_w
                dist = float(np.linalg.norm(dir))
                if dist <= 1e-6:
                    continue
                dir = dir / dist

                n_steps = int(min(96, max(1, dist / self.voxel_size)))
                # FREE along ray (avoid marking last cell as FREE)
                for i in range(0, max(1, n_steps - 1)):
                    t = (float(i) / float(n_steps)) * dist
                    q = cam_w + dir * t
                    idx = ((q - self.origin) / self.voxel_size).astype(np.int32)
                    self._touch(idx, FREE)

                # OCCUPIED at surface
                idx2 = ((p_w - self.origin) / self.voxel_size).astype(np.int32)
                self._touch(idx2, OCCUPIED)

    def occupied_points(
        self,
        *,
        box_min: list[float] | None = None,
        box_max: list[float] | None = None,
        max_points: int = 50_000,
    ) -> list[list[float]]:
        """Return centers of occupied voxels as point list (world coords)."""
        g = self.grid
        if box_min is not None and box_max is not None:
            bmin = np.asarray(box_min, dtype=np.float32).reshape(3)
            bmax = np.asarray(box_max, dtype=np.float32).reshape(3)
            lo = np.minimum(bmin, bmax)
            hi = np.maximum(bmin, bmax)
            i0 = ((lo - self.origin) / self.voxel_size).astype(np.int32)
            i1 = ((hi - self.origin) / self.voxel_size).astype(np.int32) + 1
            i0 = np.maximum(i0, 0)
            i1 = np.minimum(i1, np.asarray(g.shape, dtype=np.int32))
            if np.any(i1 <= i0):
                return []
            sub = g[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
            idx = np.argwhere(sub == OCCUPIED)
            if idx.size == 0:
                return []
            idx = idx + i0.reshape(1, 3)
        else:
            idx = np.argwhere(g == OCCUPIED)
            if idx.size == 0:
                return []

        if max_points > 0 and idx.shape[0] > int(max_points):
            step = int(max(1, idx.shape[0] // int(max_points)))
            idx = idx[::step]

        pts = self.origin.reshape(1, 3) + (idx.astype(np.float32) + 0.5) * float(self.voxel_size)
        return pts.tolist()

    def query(self, points: list[list[float]]) -> list[int]:
        pts = np.array(points, dtype=np.float32)
        idx = ((pts - self.origin) / self.voxel_size).astype(np.int32)
        shp = np.asarray(self.grid.shape, dtype=np.int32)
        out: list[int] = []
        for i in idx:
            if np.any(i < 0) or np.any(i >= shp):
                out.append(int(UNKNOWN))
            else:
                out.append(int(self.grid[int(i[0]), int(i[1]), int(i[2])]))
        return out

    def stats(self, points: list[list[float]] | None = None) -> dict:
        g = self.grid
        if points is None:
            u = int(np.sum(g == UNKNOWN))
            f = int(np.sum(g == FREE))
            o = int(np.sum(g == OCCUPIED))
            tot = int(g.size)
            observed = int(f + o)
            obs_ratio = float(observed) / float(tot) if tot > 0 else 0.0
            unk_ratio = float(u) / float(tot) if tot > 0 else 0.0
            return {
                "unknown": u,
                "free": f,
                "occupied": o,
                "total": tot,
                "observed_ratio": float(obs_ratio),
                "unknown_ratio": float(unk_ratio),
            }

        occ = {"unknown": 0, "free": 0, "occupied": 0, "total": 0}
        for p in points:
            c = np.array(p, dtype=np.float32)
            idx = ((c - self.origin) / self.voxel_size).astype(np.int32)
            r = 2
            i0 = np.maximum(idx - r, 0)
            i1 = np.minimum(idx + r + 1, np.asarray(self.grid.shape, dtype=np.int32))
            sub = g[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]]
            occ["unknown"] += int(np.sum(sub == UNKNOWN))
            occ["free"] += int(np.sum(sub == FREE))
            occ["occupied"] += int(np.sum(sub == OCCUPIED))
            occ["total"] += int(sub.size)
        tot = int(occ.get("total", 0) or 0)
        u = int(occ.get("unknown", 0) or 0)
        f = int(occ.get("free", 0) or 0)
        o = int(occ.get("occupied", 0) or 0)
        observed = int(f + o)
        occ["observed_ratio"] = float(observed) / float(tot) if tot > 0 else 0.0
        occ["unknown_ratio"] = float(u) / float(tot) if tot > 0 else 0.0
        return occ

    def stats_aabb(self, box_min: list[float], box_max: list[float]) -> dict:
        bmin = np.asarray(box_min, dtype=np.float32).reshape(3)
        bmax = np.asarray(box_max, dtype=np.float32).reshape(3)
        lo = np.minimum(bmin, bmax)
        hi = np.maximum(bmin, bmax)

        i0 = ((lo - self.origin) / self.voxel_size).astype(np.int32)
        i1 = ((hi - self.origin) / self.voxel_size).astype(np.int32) + 1
        i0 = np.maximum(i0, 0)
        i1 = np.minimum(i1, np.asarray(self.grid.shape, dtype=np.int32))
        if np.any(i1 <= i0):
            return {"unknown": 0, "free": 0, "occupied": 0, "total": 0}
        sub = self.grid[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]]
        return {
            "unknown": int(np.sum(sub == UNKNOWN)),
            "free": int(np.sum(sub == FREE)),
            "occupied": int(np.sum(sub == OCCUPIED)),
            "total": int(sub.size),
        }

    def weight_histogram(self, *, bins: tuple[int, ...] = (0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000)) -> dict:
        w = self.weights.astype(np.int32).ravel()
        edges = np.asarray(bins, dtype=np.int32)
        out: dict[str, int] = {}
        for i in range(len(edges) - 1):
            lo = int(edges[i])
            hi = int(edges[i + 1])
            out[f"[{lo},{hi})"] = int(np.sum((w >= lo) & (w < hi)))
        out[f"[{int(edges[-1])},inf)"] = int(np.sum(w >= int(edges[-1])))
        return out

    def integrate_pointcloud_rays(
        self,
        cam_pos_world: np.ndarray,
        points_world: np.ndarray,
        *,
        max_points: int = 6000,
        stride: int | None = None,
        max_range_m: float = 8.0,
    ) -> dict:
        """
        Lightweight occupancy integration from a sparse pointcloud (no depth image).

        For each point: mark FREE along the ray from camera to point, and OCCUPIED at the point.
        This is intentionally conservative and is primarily used to stabilize readiness metrics on
        pipelines that do not provide depth frames.
        """
        if points_world is None:
            return {"touched": 0, "used_points": 0}

        cam = np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return {"touched": 0, "used_points": 0}

        # Subsample deterministically to cap work.
        n = int(pts.shape[0])
        if stride is None:
            stride = max(1, n // int(max_points)) if n > max_points else 1
        pts = pts[:: int(stride)]
        if pts.shape[0] > int(max_points):
            pts = pts[: int(max_points)]

        voxel = float(self.voxel_size)
        step_m = max(0.06, voxel * 0.75)

        touched = 0
        for p in pts:
            v = p - cam
            d = float(np.linalg.norm(v))
            if d < 1e-6:
                continue
            if d > float(max_range_m):
                # Cap far points - we only need local observation.
                p = cam + v * (float(max_range_m) / d)
                v = p - cam
                d = float(max_range_m)

            # Sample along the ray. Avoid marking the endpoint as FREE.
            steps = int(max(1, math.floor(d / step_m)))
            for i in range(steps):
                t = float(i) / float(steps)
                q = cam + v * t
                idx = ((q - self.origin) / self.voxel_size).astype(np.int32)
                ok = self._touch(idx, FREE)
                if ok:
                    touched += 1

            # Mark the endpoint as OCCUPIED.
            idx2 = ((p - self.origin) / self.voxel_size).astype(np.int32)
            if self._touch(idx2, OCCUPIED):
                touched += 1

        return {"touched": int(touched), "used_points": int(pts.shape[0]), "stride": int(stride)}
