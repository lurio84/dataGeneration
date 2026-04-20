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
    cameras: "list[dict] | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample mesh surface → (pts [N,3], labels [N,]).

    When cameras is provided, only camera-facing triangles are sampled (backface
    culling). Contact surfaces (box bottom, pallet underside, fork underside) are
    never sampled, so sensor noise cannot push their points through adjacent objects.
    """
    if cameras:
        mesh.compute_triangle_normals()
        oversample = max(int(n_points * 3), n_points + 2_000)
        pcd = mesh.sample_points_uniformly(number_of_points=oversample,
                                           use_triangle_normal=True)
        pts = np.asarray(pcd.points, dtype=np.float32)
        nrm = np.asarray(pcd.normals, dtype=np.float64)

        # Keep points whose triangle normal faces at least one camera.
        cam_positions = [np.array(c["pos"], dtype=np.float64) for c in cameras]
        visible = np.zeros(len(pts), dtype=bool)
        for cp in cam_positions:
            to_cam = cp - pts.astype(np.float64)        # (N,3), unnormalised
            visible |= np.einsum("ij,ij->i", nrm, to_cam) > 0

        pts = pts[visible]
        if len(pts) == 0:
            # Degenerate mesh or all normals zero — fall back to full sampling.
            pcd = mesh.sample_points_uniformly(number_of_points=n_points)
            pts = np.asarray(pcd.points, dtype=np.float32)
        elif len(pts) > n_points:
            pts = pts[:n_points]
    else:
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


def compute_axial_noise(
    pts: np.ndarray,
    cameras: list[dict],
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Per-point axial noise calibrated to FUSION3D BBB empirical data.

    Physics modelled:
      (a) Non-Gaussian mixture: 60% Gaussian core + 40% t-Student heavy tail.
      (b) Quadratic depth scaling: σ(Z) = σ_ref · (Z/Z_ref)².
      (c) Anisotropic: noise projected along camera ray, so tilted surfaces
          automatically receive more noise perpendicular to their face.

    Calibration targets (from real BBB captures):
      frontal 3.5m  → σ_core ≈ 10mm, σ_realistic ≈ 16mm
      42° tilt 4.3m → σ_axial ≈ 27mm
      42° tilt 4.6m → σ_axial ≈ 41mm

    Returns displacement array (N, 3) float32 to add to pts.
    """
    N = len(pts)
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)

    gaussian_frac  = cfg.get("noise_gaussian_frac", 0.60)
    core_ref       = cfg.get("noise_core_ref",      0.010)   # m at z_ref
    tail_ref       = cfg.get("noise_tail_ref",      0.025)   # m at z_ref
    tail_df        = cfg.get("noise_tail_df",       4)
    z_ref          = cfg.get("noise_z_ref",         3.0)     # m

    cam_positions = [np.array(c["pos"], dtype=np.float64) for c in cameras]
    pts64 = pts.astype(np.float64)

    # 1. Distance to nearest camera for each point → depth Z
    all_dists = np.stack(
        [np.linalg.norm(pts64 - cp, axis=1) for cp in cam_positions], axis=1
    )                                                          # (N, n_cams)
    nearest_idx = np.argmin(all_dists, axis=1)                # (N,)
    depth = all_dists[np.arange(N), nearest_idx]              # (N,)

    # 2. Per-point σ — quadratic depth scaling
    scale = (depth / z_ref) ** 2
    sigma_core = core_ref * scale                             # (N,)
    sigma_tail = tail_ref * scale                             # (N,)

    # 3. Unit ray direction: camera → point
    rays = np.empty((N, 3), dtype=np.float64)
    for i_cam, cp in enumerate(cam_positions):
        mask = nearest_idx == i_cam
        if not np.any(mask):
            continue
        diff = pts64[mask] - cp
        norm = np.linalg.norm(diff, axis=1, keepdims=True)
        rays[mask] = diff / np.maximum(norm, 1e-9)

    # 4. Noise magnitude from mixture distribution
    is_gauss = rng.random(N) < gaussian_frac
    scalar = np.empty(N, dtype=np.float64)

    n_g = int(is_gauss.sum())
    if n_g:
        scalar[is_gauss] = rng.standard_normal(n_g) * sigma_core[is_gauss]

    n_t = N - n_g
    if n_t:
        # t-distribution: std = sqrt(df/(df-2)) for df>2; normalise to σ=1 then scale
        t_std = np.sqrt(tail_df / (tail_df - 2)) if tail_df > 2 else 1.0
        scalar[~is_gauss] = (rng.standard_t(tail_df, n_t) / t_std) * sigma_tail[~is_gauss]

    # 5. Project scalar noise along ray → XYZ displacement
    return (scalar[:, None] * rays).astype(np.float32)


def degrade_labeled(
    pts: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate real-sensor imperfections while keeping label correspondence:
      1. Voxel downsampling   — limits spatial resolution
      2. Axial noise mixture  — non-Gaussian, depth-scaled, ray-aligned
      3. Point dropout        — simulates reflectance / occlusion losses
      4. Local outliers       — stray reflections, multi-path artefacts
    """
    if len(pts) == 0:
        return pts, labels

    # 1. Voxel downsampling (numpy, label-safe)
    voxel_idx = np.floor(pts / cfg["voxel_size"]).astype(np.int64)
    _, unique = np.unique(voxel_idx, axis=0, return_index=True)
    pts, labels = pts[unique], labels[unique]

    # 2. Axial noise: non-Gaussian mixture, quadratic depth, ray-aligned
    pts = (pts.astype(np.float64) + compute_axial_noise(pts, cfg["cameras"], cfg, rng)).astype(np.float32)

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
