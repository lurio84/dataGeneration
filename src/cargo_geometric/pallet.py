"""
Stage 2 — pallet detection.

Look for horizontal cluster(s) sitting in a Y slice above the floor that
plausibly correspond to the top face of a pallet (EUR ~1.2 × 0.8 × 0.144 m).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d

from cargo_geometric.params import GeometricParams
from cargo_geometric.anchor import AnchorInfo, points_inside_anchor


@dataclass
class PalletInfo:
    center: np.ndarray             # (3,) world-space xyz of cluster centroid
    obb_extent: np.ndarray         # (3,) extent of fit OBB
    obb_R: np.ndarray              # (3, 3) rotation
    footprint_w: float             # larger horizontal dim
    footprint_d: float             # smaller horizontal dim
    aspect: float                  # long / short
    n_points: int
    point_indices: np.ndarray      # indices into the slice array
    cluster_id: int


@dataclass
class PalletDetectionResult:
    pallets: list[PalletInfo]
    slice_mask: np.ndarray         # bool mask over pts_no_floor for the Y slice
    cluster_labels: np.ndarray     # DBSCAN labels for the slice points (-1 = noise)


def _horizontal_dims(extent: np.ndarray, R: np.ndarray) -> tuple[float, float, int]:
    """Return (long, short, vertical_axis_idx) given OBB extent and rotation.

    Vertical axis is the OBB axis whose rotated direction is closest to
    world Y. Horizontal dims are the other two extents.
    """
    y_alignment = np.abs(R[1, :])           # |dot of each OBB axis with world Y|
    v_idx = int(np.argmax(y_alignment))
    horiz = [i for i in range(3) if i != v_idx]
    a, b = float(extent[horiz[0]]), float(extent[horiz[1]])
    long_, short_ = (a, b) if a >= b else (b, a)
    return long_, short_, v_idx


def detect_pallets(
    pts_no_floor: np.ndarray,
    floor_y: float,
    params: GeometricParams,
    anchor: AnchorInfo | None = None,
) -> PalletDetectionResult:
    """Detect pallet candidate(s) as horizontal clusters in a Y slice.

    If `anchor` is provided, the slice is restricted to its XZ footprint
    expanded by `params.anchor_xz_margin`. This is required for real BBB
    scenes where the global slice is contaminated by widely-spread floor
    noise across the ~5×4 m FUSION3D fusion volume.

    Returns candidates sorted by point count (descending). Empty list if
    nothing passes the geometric filters (caller falls back to Stage 3b).
    """
    y = pts_no_floor[:, 1]
    y_lo = floor_y + params.pallet_slice_y_min
    y_hi = floor_y + params.pallet_slice_y_max
    slice_mask = (y >= y_lo) & (y <= y_hi)
    if anchor is not None:
        slice_mask &= points_inside_anchor(pts_no_floor, anchor, params.anchor_xz_margin)
    slice_pts = pts_no_floor[slice_mask]

    if len(slice_pts) < params.pallet_dbscan_min_pts:
        return PalletDetectionResult(
            pallets=[],
            slice_mask=slice_mask,
            cluster_labels=np.full(len(slice_pts), -1, dtype=np.int32),
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(slice_pts.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=params.pallet_dbscan_eps,
            min_points=params.pallet_dbscan_min_pts,
            print_progress=False,
        ),
        dtype=np.int32,
    )

    candidates: list[PalletInfo] = []
    for cid in np.unique(labels):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < params.pallet_dbscan_min_pts:
            continue
        cluster = slice_pts[idx]

        try:
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(cluster.astype(np.float64))
            )
        except RuntimeError:
            continue

        extent = np.asarray(obb.extent, dtype=np.float64)
        R = np.asarray(obb.R, dtype=np.float64)
        long_, short_, _v = _horizontal_dims(extent, R)
        if short_ < 1e-6:
            continue
        aspect = long_ / short_

        if short_ < params.pallet_min_short:
            continue
        if long_ < params.pallet_min_long:
            continue
        if long_ > params.pallet_max_long:
            continue
        if aspect > params.pallet_max_aspect:
            continue

        density = len(cluster) / (long_ * short_ + 1e-9)
        if density < params.pallet_min_density_per_m2:
            continue

        candidates.append(
            PalletInfo(
                center=np.asarray(obb.center, dtype=np.float64),
                obb_extent=extent,
                obb_R=R,
                footprint_w=long_,
                footprint_d=short_,
                aspect=aspect,
                n_points=int(len(cluster)),
                point_indices=idx,
                cluster_id=int(cid),
            )
        )

    candidates.sort(key=lambda p: p.n_points, reverse=True)
    return PalletDetectionResult(
        pallets=candidates,
        slice_mask=slice_mask,
        cluster_labels=labels,
    )
