"""
features.py — Per-point feature extraction for the 5-class ML classifier.

16 geometric features per point using a k-NN neighbourhood (cKDTree + PCA).
All features are float32.

Positional features dist_xz and dist_centroid_xz have been intentionally
removed: they caused the model to learn object positions rather than shapes,
producing ring-shaped vehicle predictions and systematic person over-prediction
on real data. Replaced by three geometry-only features: normal_y_std,
lam_ratio_12, and planarity_large.

Coordinate convention (inherited from generate_dataset.py):
  X right, Y up (height), Z toward cameras  [metres]
"""

import numpy as np
from scipy.spatial import cKDTree


FEATURE_NAMES: list[str] = [
    "y",                  # 0  absolute height (critical: floor@0, pallet@0.144)
    "y_norm",             # 1  relative height in scene
    "z",                  # 2  depth — proxy for sensor noise level
    "local_density",      # 3  point count within radius 0.15 m
    "nbr_y_mean",         # 4  mean Y of k neighbours
    "nbr_y_std",          # 5  std  Y of k neighbours (local roughness)
    "height_range_local", # 6  max-min Y within neighbourhood
    "normal_y",           # 7  Y-component of estimated surface normal
    "curvature",          # 8  λ3 / (λ1+λ2+λ3)
    "planarity",          # 9  (λ2−λ3) / λ1   [k=20]
    "linearity",          # 10 (λ1−λ2) / λ1   [k=20]
    "sphericity",         # 11 λ3 / λ1
    "verticality",        # 12 |normal_y|
    "normal_y_std",       # 13 std of normal_y across k neighbours — surface regularity
    "lam_ratio_12",       # 14 λ1 / λ2 — elongation: high for needles/arms, ~1 for planes
    "planarity_large",    # 15 (λ2−λ3) / λ1  at k=50 — macro-scale flatness
]

_N_FEATURES = len(FEATURE_NAMES)   # 16

# Default neighbourhood parameters — exposed as constants so callers can reference them
# without hard-coding the numbers (e.g. for documentation or validation).
K_NEIGHBORS: int = 20         # k nearest neighbours for local PCA / stats
K_LARGE: int = 50             # k for macro-scale PCA (planarity_large)
LOCAL_RADIUS_M: float = 0.15  # search radius for local_density [metres]


def _pca_features(pts: np.ndarray, idx: np.ndarray) -> tuple:
    """
    Vectorised PCA over k-NN neighbourhoods.

    Returns (lam1, lam2, lam3, normal_y) where each is shape (N,).
    lam1 ≥ lam2 ≥ lam3 (descending).
    """
    k_eff = idx.shape[1]
    nbr_pts = pts[idx]                                     # (N, k, 3)
    centroid_nbr = nbr_pts.mean(axis=1, keepdims=True)     # (N, 1, 3)
    centered = nbr_pts - centroid_nbr                      # (N, k, 3)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k_eff  # (N, 3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)        # ascending order
    lam1 = eigenvalues[:, 2]
    lam2 = eigenvalues[:, 1]
    lam3 = eigenvalues[:, 0]
    normal_y = eigenvectors[:, 1, 0]                       # Y-component of min-eigenvector
    return lam1, lam2, lam3, normal_y


def extract_features(
    pts: np.ndarray,
    k: int = K_NEIGHBORS,
    k_large: int = K_LARGE,
    radius: float = LOCAL_RADIUS_M,
) -> np.ndarray:
    """
    Extract 16 per-point geometric features.

    Parameters
    ----------
    pts     : (N, 3) float array  [x, y, z]
    k       : neighbours for local PCA / stats (default 20)
    k_large : neighbours for macro-scale planarity (default 50)
    radius  : search radius for local_density (metres)

    Returns
    -------
    feats : (N, 16) float32 array, columns match FEATURE_NAMES
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

    feats[:, 0] = y
    feats[:, 1] = (y - y_min) / y_range
    feats[:, 2] = z

    # ── KD-tree ──────────────────────────────────────────────────────────────
    tree = cKDTree(pts)

    # local_density: points within radius (excluding self)
    counts = tree.query_ball_point(pts, r=radius, return_length=True)
    feats[:, 3] = np.asarray(counts, dtype=np.float32) - 1

    # k nearest neighbours (includes self at index 0)
    k_eff = min(k, N)
    _, idx = tree.query(pts, k=k_eff)   # (N, k_eff)

    # ── Neighbourhood height stats ───────────────────────────────────────────
    nbr_y = y[idx]   # (N, k_eff)
    feats[:, 4] = nbr_y.mean(axis=1)
    feats[:, 5] = nbr_y.std(axis=1)
    feats[:, 6] = nbr_y.max(axis=1) - nbr_y.min(axis=1)

    # ── Local PCA (k=20) ────────────────────────────────────────────────────
    lam1, lam2, lam3, normal_y = _pca_features(pts, idx)

    lam_sum = lam1 + lam2 + lam3
    lam_sum = np.where(lam_sum < 1e-10, 1e-10, lam_sum)
    lam1_safe = np.where(lam1 < 1e-10, 1e-10, lam1)
    lam2_safe = np.where(lam2 < 1e-10, 1e-10, lam2)

    feats[:, 7]  = normal_y                            # normal_y
    feats[:, 8]  = lam3 / lam_sum                      # curvature
    feats[:, 9]  = (lam2 - lam3) / lam1_safe           # planarity
    feats[:, 10] = (lam1 - lam2) / lam1_safe           # linearity
    feats[:, 11] = lam3 / lam1_safe                    # sphericity
    feats[:, 12] = np.abs(normal_y)                    # verticality

    # normal_y_std: std of normal_y across neighbours — surface regularity
    # Low for flat surfaces (cargo face, floor), high for complex geometry (person)
    feats[:, 13] = np.abs(normal_y)[idx].std(axis=1)

    # lam_ratio_12: λ1/λ2 — ~1 for planar patches, large for elongated structures
    # Helps distinguish fork arms / cylindrical objects from flat box faces
    feats[:, 14] = lam1 / lam2_safe

    # ── Macro-scale PCA (k=50) ──────────────────────────────────────────────
    k_large_eff = min(k_large, N)
    _, idx_large = tree.query(pts, k=k_large_eff)
    lam1_l, lam2_l, lam3_l, _ = _pca_features(pts, idx_large)
    lam1_l_safe = np.where(lam1_l < 1e-10, 1e-10, lam1_l)
    feats[:, 15] = (lam2_l - lam3_l) / lam1_l_safe    # planarity_large

    return feats
