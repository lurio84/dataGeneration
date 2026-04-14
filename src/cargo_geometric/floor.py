"""
Stage 0+1 — preprocess (read + align + voxelize) and floor removal with
height validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from classifier.predict import read_ply_xyz, align_to_synthetic
from cargo_geometric.params import GeometricParams


@dataclass
class FloorResult:
    pts: np.ndarray            # (M, 3) points with floor removed
    floor_mask: np.ndarray     # (N,) bool — True where input point belongs to floor
    floor_y: float             # mean y of accepted plane inliers
    plane: tuple               # (a, b, c, d) of accepted plane
    attempts: int              # how many RANSAC attempts were tried


def preprocess(ply_path: Path, params: GeometricParams) -> np.ndarray:
    """Read PLY, align to Y-up floor=0, voxel down-sample.

    Returns (N, 3) float32 array.
    """
    pts = read_ply_xyz(Path(ply_path))
    pts, _info = align_to_synthetic(pts, mode="auto")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(params.voxel_size)
    return np.asarray(pcd.points, dtype=np.float32)


def remove_floor(pts: np.ndarray, params: GeometricParams) -> FloorResult:
    """RANSAC-fit the dominant plane, validate it is near y=0 and ~horizontal.

    Retries on subset (excluding rejected inliers) up to floor_max_attempts.
    Raises RuntimeError if no valid plane is found.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    n_total = len(pts)
    work_idx = np.arange(n_total)
    attempts = 0

    while attempts < params.floor_max_attempts and len(work_idx) > params.floor_ransac_n:
        attempts += 1
        sub = pcd.select_by_index(work_idx.tolist())
        plane_model, inliers = sub.segment_plane(
            distance_threshold=params.floor_ransac_dist,
            ransac_n=params.floor_ransac_n,
            num_iterations=params.floor_ransac_iters,
        )
        a, b, c, d = plane_model
        inlier_global = work_idx[np.asarray(inliers, dtype=np.int64)]
        mean_y = float(pts[inlier_global, 1].mean())

        normal = np.array([a, b, c], dtype=np.float64)
        normal /= np.linalg.norm(normal) + 1e-12
        vert = abs(float(normal[1]))

        if vert >= params.floor_normal_min_y and abs(mean_y) <= params.floor_max_abs_y:
            mask = np.zeros(n_total, dtype=bool)
            mask[inlier_global] = True
            # Second pass: also strip everything inside a Y band around the
            # accepted plane height. RANSAC distance_threshold (~25 mm) is
            # narrower than real floor noise (σ≈15 mm), so without this we
            # leave a thick mat of floor pts that confuses later stages.
            band = params.floor_band
            mask |= (pts[:, 1] >= mean_y - band) & (pts[:, 1] <= mean_y + band)
            return FloorResult(
                pts=pts[~mask],
                floor_mask=mask,
                floor_y=mean_y,
                plane=(float(a), float(b), float(c), float(d)),
                attempts=attempts,
            )

        work_idx = np.setdiff1d(work_idx, inlier_global, assume_unique=True)

    raise RuntimeError(
        f"remove_floor: no valid plane after {attempts} attempts "
        f"(thresholds |y|<{params.floor_max_abs_y}, |ny|>{params.floor_normal_min_y})"
    )
