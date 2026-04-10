"""
features.py — Per-point feature extraction for the 5-class ML classifier.

15 geometric features per point using a k-NN neighbourhood (cKDTree + PCA).
All features are float32.  Typical runtime: ~15 ms per scene on one CPU core.

Coordinate convention (inherited from generate_dataset.py):
  X right, Y up (height), Z toward cameras  [metres]
"""

import numpy as np
from scipy.spatial import cKDTree


FEATURE_NAMES: list[str] = [
    "y",                  # 0  absolute height
    "y_norm",             # 1  relative height in scene
    "z",                  # 2  depth (distance to cameras)
    "dist_xz",            # 3  radial horizontal distance to origin
    "local_density",      # 4  point count within radius 0.15 m
    "nbr_y_mean",         # 5  mean Y of k neighbours
    "nbr_y_std",          # 6  std  Y of k neighbours (local roughness)
    "height_range_local", # 7  max-min Y within neighbourhood
    "normal_y",           # 8  Y-component of estimated surface normal
    "curvature",          # 9  λ3 / (λ1+λ2+λ3)
    "planarity",          # 10 (λ2−λ3) / λ1
    "linearity",          # 11 (λ1−λ2) / λ1
    "sphericity",         # 12 λ3 / λ1
    "verticality",        # 13 |normal_y|
    "dist_centroid_xz",   # 14 XZ distance to scene centroid
]

_N_FEATURES = len(FEATURE_NAMES)   # 15


def extract_features(
    pts: np.ndarray,
    k: int = 20,
    radius: float = 0.15,
) -> np.ndarray:
    """
    Extract 15 per-point geometric features.

    Parameters
    ----------
    pts    : (N, 3) float array  [x, y, z]
    k      : number of nearest neighbours for local PCA / stats
    radius : search radius for local_density (metres)

    Returns
    -------
    feats : (N, 15) float32 array
    """
    pts = np.asarray(pts, dtype=np.float32)
    N = len(pts)

    feats = np.empty((N, _N_FEATURES), dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # ── Scene-level scalars ──────────────────────────────────────────────────
    y_min = float(y.min())
    y_range = float(y.max()) - y_min
    if y_range < 1e-6:
        y_range = 1.0

    centroid_x = float(x.mean())
    centroid_z = float(z.mean())

    feats[:, 0] = y
    feats[:, 1] = (y - y_min) / y_range
    feats[:, 2] = z
    feats[:, 3] = np.sqrt(x ** 2 + z ** 2)
    feats[:, 14] = np.sqrt((x - centroid_x) ** 2 + (z - centroid_z) ** 2)

    # ── KD-tree ──────────────────────────────────────────────────────────────
    tree = cKDTree(pts)

    # local_density: points within radius (excluding self, +1 to include self)
    counts = tree.query_ball_point(pts, r=radius, return_length=True)
    feats[:, 4] = np.asarray(counts, dtype=np.float32) - 1  # exclude self

    # k nearest neighbours (includes self at index 0)
    k_eff = min(k, N)
    _, idx = tree.query(pts, k=k_eff)   # (N, k_eff)

    # ── Per-point neighbourhood stats & PCA ─────────────────────────────────
    # Pre-fetch Y values for all neighbours in one shot
    nbr_y = y[idx]   # (N, k_eff)

    feats[:, 5] = nbr_y.mean(axis=1)
    feats[:, 6] = nbr_y.std(axis=1)
    feats[:, 7] = nbr_y.max(axis=1) - nbr_y.min(axis=1)

    # PCA over XYZ neighbourhood
    nbr_pts = pts[idx]   # (N, k_eff, 3)
    centroid_nbr = nbr_pts.mean(axis=1, keepdims=True)   # (N, 1, 3)
    centered = nbr_pts - centroid_nbr                     # (N, k_eff, 3)

    # Covariance  (N, 3, 3)
    # cov_i = (1/k) * centered_i.T @ centered_i
    cov = np.einsum("nki,nkj->nij", centered, centered) / k_eff  # (N, 3, 3)

    # Eigendecomposition: eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # (N,3), (N,3,3)

    # Reorder: λ1 ≥ λ2 ≥ λ3  (descending)
    lam1 = eigenvalues[:, 2]   # largest
    lam2 = eigenvalues[:, 1]
    lam3 = eigenvalues[:, 0]   # smallest

    # Normal = eigenvector of smallest eigenvalue  shape (N, 3)
    normal = eigenvectors[:, :, 0]    # column 0 → smallest eigenvalue

    # Ensure numerical stability
    lam_sum = lam1 + lam2 + lam3
    lam_sum = np.where(lam_sum < 1e-10, 1e-10, lam_sum)
    lam1_safe = np.where(lam1 < 1e-10, 1e-10, lam1)

    feats[:, 8] = normal[:, 1]                           # normal_y
    feats[:, 9] = lam3 / lam_sum                         # curvature
    feats[:, 10] = (lam2 - lam3) / lam1_safe             # planarity
    feats[:, 11] = (lam1 - lam2) / lam1_safe             # linearity
    feats[:, 12] = lam3 / lam1_safe                      # sphericity
    feats[:, 13] = np.abs(normal[:, 1])                  # verticality

    return feats
