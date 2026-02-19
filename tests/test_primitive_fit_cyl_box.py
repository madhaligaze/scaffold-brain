from __future__ import annotations

import numpy as np

from perception.primitive_fit import fit_box_pca, fit_cylinder_pca


def test_fit_cylinder_pca_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    theta = rng.uniform(0, 2 * np.pi, size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    x = 0.2 * np.cos(theta) + rng.normal(0, 0.002, size=n)
    y = 0.2 * np.sin(theta) + rng.normal(0, 0.002, size=n)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    res = fit_cylinder_pca(pts)
    assert res.kind == "cylinder"
    assert res.params.get("ok") is True
    assert abs(float(res.params["radius_m"]) - 0.2) < 0.03


def test_fit_box_pca_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.uniform(-0.5, 0.5, size=n)
    y = rng.uniform(-0.2, 0.2, size=n)
    z = rng.uniform(-0.1, 0.1, size=n)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    res = fit_box_pca(pts)
    assert res.kind == "box"
    assert res.params.get("ok") is True
    ex = res.params["extents"]
    ex = sorted([abs(float(e)) for e in ex], reverse=True)
    assert abs(ex[0] - 0.5) < 0.08
    assert abs(ex[1] - 0.2) < 0.05
    assert abs(ex[2] - 0.1) < 0.04
