"""
Cluster-level feature extraction for the ML cargo classifier.

23 features per cluster, all invariant to translation.  The feature vector
is designed to discriminate {cargo, person, vehicle, floor_residual, other}
at the cluster level — complementary to the per-point classifier in
src/classifier/ which operates at a finer local scale.

Usage::

    from cargo_geometric.cluster_features import extract_cluster_features, \
        extract_features_batch, FEATURE_NAMES

    # Single cluster
    feat = extract_cluster_features(cluster_pts, floor_y, anchor)   # (23,)

    # Batch of clusters
    X = extract_features_batch(clusters, floor_y, anchor)           # (N, 23)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d

from cargo_geometric.anchor import AnchorInfo


# ── Feature names (canonical order, persisted in pickle) ────────────────────

FEATURE_NAMES: list[str] = [
    "n_points",                        # 0
    "log10_n_points",                  # 1
    "obb_long",                        # 2
    "obb_short",                       # 3
    "obb_height",                      # 4
    "aspect_long_short",               # 5
    "aspect_height_long",              # 6
    "base_area",                       # 7
    "volume_obb",                      # 8
    "density_xz",                      # 9
    "density_volume",                  # 10
    "height_above_floor_min",          # 11
    "height_above_floor_max",          # 12
    "top_slab_frac",                   # 13
    "bottom_slab_frac",                # 14
    "top_slab_planarity",              # 15
    "vertical_profile_entropy",        # 16
    "fill_ratio",                      # 17
    "pca_linearity",                   # 18
    "pca_planarity",                   # 19
    "pca_sphericity",                  # 20
    "anchor_inside_frac",              # 21
    "center_offset_to_anchor_center_norm",  # 22
]

assert len(FEATURE_NAMES) == 23, "FEATURE_NAMES must have exactly 23 entries"

_N_FEATURES = 23
_SLAB_FRAC = 0.15   # top/bottom slab = top/bottom 15% of height band
_N_VBINS = 10       # bins for vertical profile entropy


# ── Internal OBB helpers ─────────────────────────────────────────────────────

def _obb_dims(pts: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Return (long, short, height, extent, R) from an OBB fit.

    ``long`` and ``short`` are the two horizontal extents; ``height`` is the
    vertical extent aligned with world-Y.  Raises RuntimeError for degenerate
    clusters (forwarded to caller).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    extent = np.asarray(obb.extent, dtype=np.float64)
    R = np.asarray(obb.R, dtype=np.float64)

    # Identify vertical axis (most aligned with world Y)
    y_alignment = np.abs(R[1, :])
    v_idx = int(np.argmax(y_alignment))
    horiz = [i for i in range(3) if i != v_idx]
    a, b = float(extent[horiz[0]]), float(extent[horiz[1]])
    long_, short_ = (a, b) if a >= b else (b, a)
    height_ = float(extent[v_idx])
    return long_, short_, height_, extent, R


# ── Public API ───────────────────────────────────────────────────────────────

def extract_cluster_features(
    cluster_pts: np.ndarray,
    floor_y: float,
    anchor: AnchorInfo,
) -> np.ndarray:
    """Extract the 23-dim feature vector for one cluster.

    Parameters
    ----------
    cluster_pts : (M, 3) float array — 3D points of the cluster
    floor_y : float — mean Y of the detected floor plane
    anchor : AnchorInfo — XZ bounding box of the dominant blob

    Returns
    -------
    (23,) float64 array.  Degenerate clusters (≤3 pts or OBB error) return
    all-zeros (treated as ``other`` in downstream classification).
    """
    feat = np.zeros(_N_FEATURES, dtype=np.float64)

    if cluster_pts is None or len(cluster_pts) < 4:
        return feat  # degenerate — all zeros

    pts = np.asarray(cluster_pts, dtype=np.float64)
    n = len(pts)

    # ── OBB (may raise on degenerate shapes) ────────────────────────────────
    try:
        obb_long, obb_short, obb_height, extent, R = _obb_dims(pts)
    except (RuntimeError, ValueError):
        return feat  # degenerate — all zeros

    # ── Features 0–1: point count ────────────────────────────────────────────
    feat[0] = float(n)
    feat[1] = float(np.log10(max(n, 1)))

    # ── Features 2–4: OBB dimensions ────────────────────────────────────────
    feat[2] = obb_long
    feat[3] = obb_short
    feat[4] = obb_height

    # ── Features 5–6: aspect ratios ─────────────────────────────────────────
    feat[5] = obb_long / max(obb_short, 1e-6)          # long/short
    feat[6] = obb_height / max(obb_long, 1e-6)         # height/long

    # ── Features 7–8: base area, OBB volume ─────────────────────────────────
    feat[7] = obb_long * obb_short
    feat[8] = obb_long * obb_short * obb_height

    # ── Features 9–10: density ──────────────────────────────────────────────
    feat[9]  = n / max(feat[7], 1e-6)           # pts / base_area (XZ footprint)
    feat[10] = n / max(feat[8], 1e-6)           # pts / obb_volume

    # ── Features 11–12: height above floor ──────────────────────────────────
    y = pts[:, 1]
    feat[11] = float(y.min()) - floor_y         # lowest point relative to floor
    feat[12] = float(y.max()) - floor_y         # highest point relative to floor

    # ── Features 13–16: top/bottom slab ─────────────────────────────────────
    y_min, y_max = float(y.min()), float(y.max())
    h_range = y_max - y_min
    if h_range > 1e-6:
        slab_h = _SLAB_FRAC * h_range
        top_mask    = y >= (y_max - slab_h)
        bottom_mask = y <= (y_min + slab_h)
        feat[13] = float(top_mask.sum()) / n       # top_slab_frac
        feat[14] = float(bottom_mask.sum()) / n    # bottom_slab_frac

        # top_slab_planarity: std of Y within the top slab — low → flat lid (cargo);
        # high → rounded head (person)
        if top_mask.sum() >= 3:
            feat[15] = float(y[top_mask].std())
        else:
            feat[15] = 0.0
    else:
        feat[13] = 1.0   # flat slab
        feat[14] = 1.0
        feat[15] = 0.0

    # ── Feature 16: vertical profile entropy ─────────────────────────────────
    if h_range > 1e-6:
        hist, _ = np.histogram(y, bins=_N_VBINS, range=(y_min, y_max))
        prob = hist / max(hist.sum(), 1)
        prob = prob[prob > 0]
        feat[16] = float(-np.sum(prob * np.log(prob + 1e-12)))
    else:
        feat[16] = 0.0

    # ── Feature 17: fill_ratio (pts vs expected for a solid at this density) ─
    # We estimate the voxel-grid fill: occupied cells / bounding box volume
    # Approximation: (n_pts / density_volume) / obb_volume already captured via
    # density_volume, so we use a simpler proxy:
    # fill_ratio = density_volume * voxel_volume (≈ mean spacing^3)
    # Instead, use a bounding-box fill: project onto XZ, count unique 5cm cells.
    cell = 0.05
    xz_cells = set(
        zip(
            ((pts[:, 0] - pts[:, 0].min()) / cell).astype(int).tolist(),
            ((pts[:, 2] - pts[:, 2].min()) / cell).astype(int).tolist(),
        )
    )
    expected_cells = max((obb_long / cell) * (obb_short / cell), 1.0)
    feat[17] = len(xz_cells) / expected_cells

    # ── Features 18–20: PCA shape descriptors ───────────────────────────────
    centered = pts - pts.mean(axis=0)
    if n >= 3:
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]   # descending
        e1, e2, e3 = (float(v) for v in eigvals)
        denom = max(e1, 1e-12)
        feat[18] = (e1 - e2) / denom   # linearity
        feat[19] = (e2 - e3) / denom   # planarity
        feat[20] = e3 / denom           # sphericity
    # else remain 0

    # ── Feature 21: anchor_inside_frac ──────────────────────────────────────
    # Fraction of cluster points inside the un-expanded anchor bbox
    ax0, ax1 = anchor.x_min, anchor.x_max
    az0, az1 = anchor.z_min, anchor.z_max
    inside = (
        (pts[:, 0] >= ax0) & (pts[:, 0] <= ax1) &
        (pts[:, 2] >= az0) & (pts[:, 2] <= az1)
    )
    feat[21] = float(inside.sum()) / n

    # ── Feature 22: normalised offset from anchor centre ────────────────────
    cx, cz = float(pts[:, 0].mean()), float(pts[:, 2].mean())
    acx, acz = anchor.center_xz
    anchor_diag = max(
        np.sqrt((ax1 - ax0) ** 2 + (az1 - az0) ** 2), 1e-6
    )
    feat[22] = np.sqrt((cx - acx) ** 2 + (cz - acz) ** 2) / anchor_diag

    return feat


def extract_features_batch(
    clusters: Sequence[np.ndarray],
    floor_y: float,
    anchor: AnchorInfo,
) -> np.ndarray:
    """Extract features for a list of clusters.

    Parameters
    ----------
    clusters : sequence of (M_i, 3) arrays

    Returns
    -------
    (N, 23) float64 array, one row per cluster.
    """
    rows = [
        extract_cluster_features(c, floor_y, anchor)
        for c in clusters
    ]
    if not rows:
        return np.zeros((0, _N_FEATURES), dtype=np.float64)
    return np.vstack(rows)
