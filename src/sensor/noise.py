"""
sensor/noise.py  —  Sensor simulation, sampling and degradation.

Moved from generate_dataset.py (B9 refactor).
"""

import numpy as np
import open3d as o3d


def sample_labeled(
    mesh: o3d.geometry.TriangleMesh,
    label: int,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample mesh surface → (pts [N,3], labels [N,])."""
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(pcd.points, dtype=np.float32)
    lbs = np.full(len(pts), label, dtype=np.uint8)
    return pts, lbs


def sample_floor(cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random uniform points on Y=0 plane within floor_extent_x / floor_extent_z."""
    ext_x = cfg["floor_extent_x"]
    ext_z = cfg["floor_extent_z"]
    n = cfg["pts_floor"]
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-ext_x, ext_x, n).astype(np.float32)
    pts[:, 2] = rng.uniform(-ext_z, ext_z, n).astype(np.float32)
    lbs = np.zeros(n, dtype=np.uint8)
    return pts, lbs


def camera_arc_filter(
    pts: np.ndarray,
    labels: np.ndarray,
    cameras: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only points inside the FOV cone of at least one camera.
    Each camera looks toward the origin from its 'pos'.
    Simple angle-based filter — no raycasting occlusion (TODO v2).
    """
    visible = np.zeros(len(pts), dtype=bool)
    for cam in cameras:
        cam_pos = np.array(cam["pos"], dtype=np.float64)
        fov_half = np.deg2rad(cam["fov_deg"] / 2.0)
        view_dir = -cam_pos / np.linalg.norm(cam_pos)   # looks at origin

        to_pts = pts.astype(np.float64) - cam_pos       # (N, 3)
        norms = np.linalg.norm(to_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        cos_a = (to_pts / norms) @ view_dir              # (N,)
        visible |= cos_a > np.cos(fov_half)

    return pts[visible], labels[visible]


def apply_distance_density(
    pts: np.ndarray,
    labels: np.ndarray,
    cameras: list[dict],
    rng: np.random.Generator,
    ref_dist: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate stereo-camera density falloff: density ∝ 1/d².
    At ref_dist (1.5m): no extra dropout.
    At 3m: ~75% of remaining points dropped.
    Matches the real data pattern where distant floor/walls are sparse.
    """
    min_dist = np.full(len(pts), np.inf)
    for cam in cameras:
        cam_pos = np.array(cam["pos"], dtype=np.float64)
        d = np.linalg.norm(pts.astype(np.float64) - cam_pos, axis=1)
        min_dist = np.minimum(min_dist, d)

    # p_keep = (ref_dist / d)²  clamped to [0.15, 1.0]
    p_keep = np.clip((ref_dist / np.maximum(min_dist, ref_dist)) ** 2, 0.15, 1.0)
    keep = rng.random(len(pts)) < p_keep
    return pts[keep], labels[keep]


def degrade_labeled(
    pts: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate real-sensor imperfections while keeping label correspondence:
      1. Voxel downsampling   — limits spatial resolution
      2. Gaussian noise       — calibrated to FUSION3D σ≈10mm
      3. Point dropout        — simulates reflectance / occlusion losses
      4. Local outliers       — stray reflections, multi-path artefacts
    """
    if len(pts) == 0:
        return pts, labels

    # 1. Voxel downsampling (numpy, label-safe)
    voxel_idx = np.floor(pts / cfg["voxel_size"]).astype(np.int64)
    _, unique = np.unique(voxel_idx, axis=0, return_index=True)
    pts, labels = pts[unique], labels[unique]

    # 2. Gaussian noise
    pts = pts + rng.normal(0.0, cfg["noise_std"], pts.shape).astype(np.float32)

    # 3. Dropout
    keep = rng.random(len(pts)) > cfg["dropout_ratio"]
    pts, labels = pts[keep], labels[keep]
    if len(pts) == 0:
        return pts, labels

    # 4. Local outliers (label = 255 → "unlabeled / artefact")
    n_out = max(1, int(len(pts) * cfg["outlier_ratio"]))
    anchors = pts[rng.integers(0, len(pts), n_out)]
    offsets = rng.normal(0.0, cfg["local_outlier_std"], (n_out, 3)).astype(np.float32)
    out_pts = anchors + offsets
    out_lbs = np.full(n_out, 255, dtype=np.uint8)

    pts    = np.vstack([pts, out_pts])
    labels = np.concatenate([labels, out_lbs])
    return pts, labels
