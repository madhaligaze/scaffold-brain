import numpy as np

from perception.primitive_fit import fit_plane_ransac


def test_fit_plane_ransac_synthetic_plane():
    # Plane: y = 0.5 -> normal [0,1,0], d = -0.5
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=2000).astype(np.float32)
    z = rng.uniform(-1.0, 1.0, size=2000).astype(np.float32)
    y = (np.ones_like(x) * 0.5).astype(np.float32)
    pts = np.stack([x, y, z], axis=1)
    # small noise
    pts += rng.normal(0.0, 0.002, size=pts.shape).astype(np.float32)

    fit = fit_plane_ransac(pts, n_iter=200, dist_thresh=0.01, min_inliers=800, rng_seed=0)
    assert fit is not None
    n = np.array(fit["normal"], dtype=np.float32)
    # normal should be close to +Y
    assert float(n[1]) > 0.9
    assert abs(float(fit["d"]) + 0.5) < 0.05
