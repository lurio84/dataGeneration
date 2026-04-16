"""
Stage 0+1 — preprocess (read + align + voxelize) and floor removal with
height validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from classifier.predict import read_ply_xyz, align_to_synthetic, _detect_floor_y
from cargo_geometric.params import GeometricParams


@dataclass
class FloorResult:
    pts: np.ndarray            # (M, 3) points with floor removed
    floor_mask: np.ndarray     # (N,) bool — True where input point belongs to floor
    floor_y: float             # mean y of accepted plane inliers
    plane: tuple               # (a, b, c, d) of accepted plane
    attempts: int              # how many RANSAC attempts were tried


def preprocess(ply_path: Path, params: GeometricParams,
               align_mode: str = "auto") -> np.ndarray:
    """Read PLY, align to Y-up floor=0, voxel down-sample.

    Parameters
    ----------
    align_mode : 'auto' | 'fusion3d' | 'none'
        'auto'     — detect floor axis automatically (default, existing behaviour).
        'fusion3d' — force Z→Y swap unconditionally (FUSION3D: Z=up, X=depth,
                     Y=horizontal).  Unlike passing 'fusion3d' to
                     align_to_synthetic(), this variant does NOT rely on
                     _detect_floor_axis(), so it is safe for preprocessed clouds
                     where the floor is absent and the histogram is unreliable.
        'none'     — no alignment (cloud already in synthetic convention).

    Returns (N, 3) float32 array.
    """
    pts = read_ply_xyz(Path(ply_path))

    if align_mode == "fusion3d":
        # FUSION3D convention: X=depth, Y=horizontal, Z=vertical (up).
        # Forced Z→Y swap: new_X=old_X, new_Y=old_Z, new_Z=old_Y.
        # _detect_floor_axis() is NOT called — avoids mis-detection on
        # preprocessed clouds where there is no floor peak in the histogram.
        pts = pts[:, [0, 2, 1]].copy()
        y_off = _detect_floor_y(pts)   # mode of lowest 30 % → Y=0 at floor level
        pts = pts.copy()
        pts[:, 1] -= y_off
    else:
        pts, _info = align_to_synthetic(pts, mode=align_mode)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(params.voxel_size)
    return np.asarray(pcd.points, dtype=np.float32)


def synthetic_floor(pts: np.ndarray, params: GeometricParams) -> FloorResult:
    """Create a FloorResult without RANSAC for clouds with floor pre-removed.

    Estimates floor_y as ``min(Y) - floor_band``, placing the virtual floor
    slightly below the lowest surviving point (typically the pallet bottom
    after FUSION3D alignment).  No points are removed from ``pts``.

    Returns a FloorResult whose ``floor_mask`` is all-False so that
    ``pts_no_floor = pts`` (all points pass through to downstream stages).
    """
    floor_y = float(pts[:, 1].min()) - params.floor_band
    return FloorResult(
        pts=pts,
        floor_mask=np.zeros(len(pts), dtype=bool),
        floor_y=floor_y,
        plane=(0.0, 1.0, 0.0, float(-floor_y)),
        attempts=0,
    )


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
