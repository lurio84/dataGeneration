"""
Stage 3 — cargo extraction.

Given the anchor (XZ bbox of the dominant standing blob) and the floor,
isolate the *cargo* points by:

  1. Restricting to the anchor's XZ footprint.
  2. Removing the floor band (already done upstream).
  3. 3D-DBSCAN of the remaining points.
  4. Keeping the largest cluster — the cargo bulto.

Persons or equipment that happen to fall inside the anchor's bbox margin
but are *spatially disconnected* from the bulto end up in smaller
clusters and are dropped here. The largest 3D cluster is robustly the
cargo: persons are short and thin, equipment fragments are sparse.

Sub-cluster info (`cargo_dbscan_min_pts` / `cargo_dbscan_eps`) is also
exposed so a future stage can split per-box if needed without changing
the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from cargo_geometric.params import GeometricParams
from cargo_geometric.anchor import AnchorInfo, points_inside_anchor


@dataclass
class CargoExtractionResult:
    cargo_pts: np.ndarray              # (M, 3) float32 — cargo points in world
    cargo_global_idx: np.ndarray       # (M,) int — indices into pts_no_floor
    n_total_in_anchor: int             # how many non-floor points were in anchor footprint
    n_clusters: int                    # how many 3D DBSCAN clusters found
    chosen_cluster_id: int             # cluster_dbscan id of the kept cluster
    chosen_cluster_size: int


def extract_cargo(
    pts_no_floor: np.ndarray,
    anchor: AnchorInfo,
    params: GeometricParams,
) -> CargoExtractionResult | None:
    """Return the cargo cluster inside the anchor footprint, or None on failure."""
    in_anchor = points_inside_anchor(pts_no_floor, anchor, params.anchor_xz_margin)
    sub_idx = np.where(in_anchor)[0]
    if len(sub_idx) < params.cargo_dbscan_min_pts:
        return None
    sub_pts = pts_no_floor[sub_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_pts.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=params.cargo_dbscan_eps,
            min_points=params.cargo_dbscan_min_pts,
            print_progress=False,
        ),
        dtype=np.int32,
    )

    if labels.max() < 0:
        return None

    sizes = np.bincount(labels[labels >= 0])
    best = int(np.argmax(sizes))
    keep = labels == best
    cargo_pts = sub_pts[keep]
    cargo_global_idx = sub_idx[keep]

    return CargoExtractionResult(
        cargo_pts=cargo_pts,
        cargo_global_idx=cargo_global_idx,
        n_total_in_anchor=int(len(sub_idx)),
        n_clusters=int(labels.max() + 1),
        chosen_cluster_id=best,
        chosen_cluster_size=int(sizes[best]),
    )
