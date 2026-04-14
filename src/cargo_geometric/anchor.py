"""
Cargo anchor (bulto) detection — Stage 2 helper.

Real BBB scenes are wide FUSION3D fusions (~5×4 m) where the cargo bulto
is a localised dense region. We find it by clustering the top-down (XZ)
projection of points above floor + anchor_y_min and picking the largest
cluster. Its XZ footprint is then used as a ROI to restrict pallet
detection (Stage 2) and cargo extraction (Stage 3) to a sane area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from cargo_geometric.params import GeometricParams


@dataclass
class AnchorInfo:
    n_points: int                  # points in the rank-0 cluster (above anchor_y_min)
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    height: float                  # max y - min y of the cluster
    center_xz: tuple[float, float]


def find_anchor(
    pts_no_floor: np.ndarray,
    floor_y: float,
    params: GeometricParams,
) -> AnchorInfo | None:
    """Cluster the top-down projection of points above floor + anchor_y_min,
    return the XZ bounding box of the largest cluster.
    """
    high_mask = pts_no_floor[:, 1] > floor_y + params.anchor_y_min
    high = pts_no_floor[high_mask]
    if len(high) < params.anchor_dbscan_min_pts:
        return None

    xz = high.copy()
    xz[:, 1] = 0.0       # collapse Y so DBSCAN clusters in XZ only
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xz.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=params.anchor_dbscan_eps,
            min_points=params.anchor_dbscan_min_pts,
            print_progress=False,
        ),
        dtype=np.int32,
    )
    if labels.max() < 0:
        return None

    sizes = np.bincount(labels[labels >= 0])
    best = int(np.argmax(sizes))
    cluster = high[labels == best]

    x0, x1 = float(cluster[:, 0].min()), float(cluster[:, 0].max())
    z0, z1 = float(cluster[:, 2].min()), float(cluster[:, 2].max())
    y0, y1 = float(cluster[:, 1].min()), float(cluster[:, 1].max())

    return AnchorInfo(
        n_points=int(len(cluster)),
        x_min=x0, x_max=x1, z_min=z0, z_max=z1,
        height=y1 - y0,
        center_xz=((x0 + x1) / 2, (z0 + z1) / 2),
    )


def points_inside_anchor(
    pts: np.ndarray,
    anchor: AnchorInfo,
    margin: float,
) -> np.ndarray:
    """Boolean mask: which points fall inside the anchor's XZ footprint
    expanded by `margin` on every side.
    """
    x = pts[:, 0]
    z = pts[:, 2]
    return (
        (x >= anchor.x_min - margin) & (x <= anchor.x_max + margin)
        & (z >= anchor.z_min - margin) & (z <= anchor.z_max + margin)
    )
