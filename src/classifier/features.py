"""
features.py — Per-point feature extraction for the 5-class ML classifier.

19 geometric features per point using a k-NN neighbourhood (cKDTree + PCA).
All features are float32.

Positional features dist_xz and dist_centroid_xz have been intentionally
removed: they caused the model to learn object positions rather than shapes,
producing ring-shaped vehicle predictions and systematic person over-prediction
on real data. Replaced by three geometry-only features: normal_y_std,
lam_ratio_12, and planarity_large.

Feature z (depth) has been removed: in synthetic data objects occupy fixed Z
ranges, so the model memorised Z-band → class, producing parallel stripe
artefacts on real captures. Sensor-noise proxy role is covered by
local_density, nbr_y_std, normal_y_std, and planarity.

Coordinate convention (inherited from generate_dataset.py):
  X right, Y up (height), Z toward cameras  [metres]
"""

import numpy as np
from scipy.spatial import cKDTree


FEATURE_NAMES: list[str] = [
    "y",                        # 0  absolute height (critical: floor@0, pallet@0.144)
    "y_norm",                   # 1  relative height in scene
    "local_density",            # 2  point count within radius 0.15 m
    "nbr_y_mean",               # 3  mean Y of k neighbours
    "nbr_y_std",                # 4  std  Y of k neighbours (local roughness)
    "height_range_local",       # 5  max-min Y within neighbourhood
    "normal_y",                 # 6  Y-component of estimated surface normal
    "curvature",                # 7  λ3 / (λ1+λ2+λ3)
    "planarity",                # 8  (λ2−λ3) / λ1   [k=20]
    "linearity",                # 9  (λ1−λ2) / λ1   [k=20]
    "sphericity",               # 10 λ3 / λ1
    "verticality",              # 11 |normal_y|
    "normal_y_std",             # 12 std of normal_y across k neighbours — surface regularity
    "lam_ratio_12",             # 13 λ1 / λ2 — elongation: high for needles/arms, ~1 for planes
    "planarity_large",          # 14 (λ2−λ3) / λ1  at k=50 — macro-scale flatness
    "height_above_local_floor", # 15 Y minus estimated floor Y in local 0.5 m XZ cell
    "linearity_large",          # 16 (λ1−λ2) / λ1  at k=50 — cylinder/needle at macro scale
    "sphericity_large",         # 17 λ3 / λ1       at k=50 — volumetric spread at macro scale
    "verticality_large",        # 18 |normal_y|    at k=50 — macro-scale vertical alignment
]

_N_FEATURES = len(FEATURE_NAMES)   # 19

# Default neighbourhood parameters — exposed as constants so callers can reference them
# without hard-coding the numbers (e.g. for documentation or validation).
K_NEIGHBORS: int = 20         # k nearest neighbours for local PCA / stats
K_LARGE: int = 50             # k for macro-scale PCA (planarity_large)
LOCAL_RADIUS_M: float = 0.15  # search radius for local_density [metres]
FLOOR_CELL_SIZE_M: float = 0.5  # XZ cell side for local floor estimation [metres]


def _height_above_local_floor(pts: np.ndarray) -> np.ndarray:
    """
    Estimate local floor height per XZ grid cell and return per-point height above it.

    Algorithm (vectorised, no per-point Python loops):
      1. Assign each point to a 0.5 m × 0.5 m XZ cell.
      2. Sort points by cell id; compute median Y per contiguous cell slice.
      3. Map cell medians back to each point; subtract from pts[:,1].

    Fallback: global median Y for degenerate single-cell or empty scenes.

    Returns float32 array of shape (N,).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Integer cell indices
    ix = np.floor(x.astype(np.float64) / FLOOR_CELL_SIZE_M).astype(np.int64)
    iz = np.floor(z.astype(np.float64) / FLOOR_CELL_SIZE_M).astype(np.int64)

    # Encode (ix, iz) as a single int64.  Safe for scenes ≤ ~1e9 cells wide.
    cell_code = ix * 1_000_003 + iz

    # Sort by cell so each cell occupies a contiguous slice — O(N log N)
    sort_order = np.argsort(cell_code, kind="stable")
    sorted_codes = cell_code[sort_order]
    sorted_y = y[sort_order]

    # Cell boundaries
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    cell_starts = np.concatenate([[0], boundaries])
    cell_ends = np.concatenate([boundaries, [len(y)]])

    global_floor_y = float(np.median(y))

    # Per-cell median — loop is O(n_cells), not O(N)
    floor_y_sorted = np.empty(len(y), dtype=np.float32)
    for s, e in zip(cell_starts, cell_ends):
        med = float(np.median(sorted_y[s:e])) if e > s else global_floor_y
        floor_y_sorted[s:e] = med

    # Unsort: restore original point order
    unsort_order = np.empty_like(sort_order)
    unsort_order[sort_order] = np.arange(len(y), dtype=sort_order.dtype)
    floor_y = floor_y_sorted[unsort_order]

    return (y - floor_y).astype(np.float32)


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
    Extract 19 per-point geometric features.

    Parameters
    ----------
    pts     : (N, 3) float array  [x, y, z]
    k       : neighbours for local PCA / stats (default 20)
    k_large : neighbours for macro-scale planarity (default 50)
    radius  : search radius for local_density (metres)

    Returns
    -------
    feats : (N, 19) float32 array, columns match FEATURE_NAMES
    """
    pts = np.asarray(pts, dtype=np.float32)
    N = len(pts)

    feats = np.empty((N, _N_FEATURES), dtype=np.float32)

    y = pts[:, 1]

    # ── Scene-level scalars ──────────────────────────────────────────────────
    y_min = float(y.min())
    y_range = float(y.max()) - y_min
    if y_range < 1e-6:
        y_range = 1.0

    feats[:, 0] = y
    feats[:, 1] = (y - y_min) / y_range

    # ── KD-tree ──────────────────────────────────────────────────────────────
    tree = cKDTree(pts)

    # local_density: points within radius (excluding self)
    counts = tree.query_ball_point(pts, r=radius, return_length=True)
    feats[:, 2] = np.asarray(counts, dtype=np.float32) - 1

    # k nearest neighbours (includes self at index 0)
    k_eff = min(k, N)
    _, idx = tree.query(pts, k=k_eff)   # (N, k_eff)

    # ── Neighbourhood height stats ───────────────────────────────────────────
    nbr_y = y[idx]   # (N, k_eff)
    feats[:, 3] = nbr_y.mean(axis=1)
    feats[:, 4] = nbr_y.std(axis=1)
    feats[:, 5] = nbr_y.max(axis=1) - nbr_y.min(axis=1)

    # ── Local PCA (k=20) ────────────────────────────────────────────────────
    lam1, lam2, lam3, normal_y = _pca_features(pts, idx)

    lam_sum = lam1 + lam2 + lam3
    lam_sum = np.where(lam_sum < 1e-10, 1e-10, lam_sum)
    lam1_safe = np.where(lam1 < 1e-10, 1e-10, lam1)
    lam2_safe = np.where(lam2 < 1e-10, 1e-10, lam2)

    feats[:, 6]  = normal_y                            # normal_y
    feats[:, 7]  = lam3 / lam_sum                      # curvature
    feats[:, 8]  = (lam2 - lam3) / lam1_safe           # planarity
    feats[:, 9]  = (lam1 - lam2) / lam1_safe           # linearity
    feats[:, 10] = lam3 / lam1_safe                    # sphericity
    feats[:, 11] = np.abs(normal_y)                    # verticality

    # normal_y_std: std of normal_y across neighbours — surface regularity
    # Low for flat surfaces (cargo face, floor), high for complex geometry (person)
    feats[:, 12] = np.abs(normal_y)[idx].std(axis=1)

    # lam_ratio_12: λ1/λ2 — ~1 for planar patches, large for elongated structures
    # Helps distinguish fork arms / cylindrical objects from flat box faces
    feats[:, 13] = lam1 / lam2_safe

    # ── Macro-scale PCA (k=50) ──────────────────────────────────────────────
    k_large_eff = min(k_large, N)
    _, idx_large = tree.query(pts, k=k_large_eff)
    lam1_l, lam2_l, lam3_l, normal_y_l = _pca_features(pts, idx_large)
    lam1_l_safe = np.where(lam1_l < 1e-10, 1e-10, lam1_l)
    feats[:, 14] = (lam2_l - lam3_l) / lam1_l_safe    # planarity_large

    # ── Local floor height (XZ grid, 0.5 m cells) ───────────────────────────
    feats[:, 15] = _height_above_local_floor(pts)      # height_above_local_floor

    # ── Extra macro-scale descriptors (reuse k=50 eigendecomposition) ───────
    feats[:, 16] = (lam1_l - lam2_l) / lam1_l_safe    # linearity_large
    feats[:, 17] = lam3_l / lam1_l_safe               # sphericity_large
    feats[:, 18] = np.abs(normal_y_l)                 # verticality_large

    return feats
