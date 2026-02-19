# modules/primitive_fitting.py
"""
Primitive fitting (Stage 2)
---------------------------

For important classes (pipes/beams), fit simple parametric primitives from 3D points:

- Cylinder (approx): PCA axis + robust radius from distances to axis + length range.
- Oriented box: PCA frame + extents in that frame.

This is intentionally lightweight and avoids hard dependencies beyond numpy.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns center, eigenvectors (3x3 columns), eigenvalues
    center = points.mean(axis=0)
    X = points - center
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    return center, V, w


def fit_oriented_box(points_world: np.ndarray) -> Optional[Dict[str, Any]]:
    if points_world.shape[0] < 30:
        return None
    c, V, w = _pca_frame(points_world)
    # Rotate into PCA frame
    X = (points_world - c) @ V
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    extents = (mx - mn)
    # pose: position is center in world, orientation is PCA basis (not quaternion here)
    return {
        "geometry_type": "BOX",
        "pose": {"position": c.tolist(), "orientation_matrix": V.tolist()},
        "dimensions": {"dx": float(extents[0]), "dy": float(extents[1]), "dz": float(extents[2])},
    }


def fit_cylinder(points_world: np.ndarray) -> Optional[Dict[str, Any]]:
    if points_world.shape[0] < 50:
        return None
    c, V, w = _pca_frame(points_world)
    axis = V[:, 0]
    X = points_world - c
    # project on axis
    t = X @ axis
    # distance to axis
    proj = np.outer(t, axis)
    perp = X - proj
    r = np.linalg.norm(perp, axis=1)
    # robust radius: median trimmed
    med = float(np.median(r))
    keep = np.abs(r - med) < max(0.02, 3.0 * float(np.std(r)) + 1e-6)
    r2 = r[keep]
    if r2.size < 30:
        return None
    radius = float(np.median(r2))
    t2 = t[keep]
    length = float(t2.max() - t2.min())

    # cylinder center along axis at mid of t-range
    t_mid = float((t2.max() + t2.min()) / 2.0)
    center = c + axis * t_mid

    return {
        "geometry_type": "CYLINDER",
        "pose": {"position": center.tolist(), "axis": axis.tolist()},
        "dimensions": {"radius": radius, "length": length},
    }


def classify_for_fitting(class_label: str) -> str:
    lbl = class_label.lower()
    if any(k in lbl for k in ["pipe", "tube"]):
        return "CYLINDER"
    if any(k in lbl for k in ["beam", "bar", "ledger", "plank"]):
        return "BOX"
    return "NONE"
