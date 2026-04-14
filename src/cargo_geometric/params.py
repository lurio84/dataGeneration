"""Geometric pipeline parameters (single source of truth)."""

from dataclasses import dataclass, field


@dataclass
class GeometricParams:
    voxel_size: float = 0.035

    floor_ransac_dist: float = 0.025
    floor_ransac_n: int = 3
    floor_ransac_iters: int = 500
    floor_max_abs_y: float = 0.10
    floor_normal_min_y: float = 0.9
    floor_max_attempts: int = 3
    floor_band: float = 0.05

    # Anchor (bulto) detection: 2D XZ DBSCAN of points above floor+anchor_y_min.
    # Real BBB scenes are wide (~5x4m FUSION3D fusions) with much background
    # noise; finding the cargo blob first lets us restrict pallet search to
    # its XZ footprint instead of the whole scene.
    anchor_y_min: float = 0.20
    anchor_dbscan_eps: float = 0.10
    anchor_dbscan_min_pts: int = 30
    anchor_xz_margin: float = 0.30        # expand footprint outwards before pallet search

    pallet_slice_y_min: float = 0.05
    pallet_slice_y_max: float = 0.20
    pallet_dbscan_eps: float = 0.10
    pallet_dbscan_min_pts: int = 30
    pallet_min_short: float = 0.30
    pallet_min_long: float = 0.50
    pallet_max_long: float = 1.80
    pallet_max_aspect: float = 2.5
    pallet_min_density_per_m2: float = 200.0

    pallet_top_offset: float = 0.072
    cargo_above_pallet_eps: float = 0.02
    pallet_overhang: float = 0.10
    cargo_dbscan_eps: float = 0.08
    cargo_dbscan_min_pts: int = 30
    cargo_cluster_inside_frac: float = 0.30

    fallback_dbscan_eps: float = 0.10
    fallback_dbscan_min_pts: int = 100
    fallback_min_height: float = 0.20
    fallback_max_height: float = 2.50
    fallback_min_base: float = 0.40
    person_max_height: float = 1.90
    person_min_height: float = 1.40
    person_max_base: float = 0.50

    pallet_min_w: float = 1.20
    pallet_min_d: float = 0.80

    # Cluster-level ML classifier (Paso 5)
    cluster_classifier_path: str | None = None   # path to .pkl; None = legacy rank-0
    cluster_classifier_model: str = "lgbm"       # "lgbm" | "rf" (informational only)
    cluster_min_cargo_prob: float = 0.5          # minimum cargo probability threshold
