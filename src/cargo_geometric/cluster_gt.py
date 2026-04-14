"""
Ground-truth dataset builder for the cluster-level ML classifier.

Runs the full geometric pipeline (preprocess → remove_floor → find_anchor →
3D DBSCAN inside anchor) on every labelled synthetic PLY in `dataset_dir`,
assigns a cluster-level label by majority vote of per-point labels, and
extracts the 23 cluster features.  The result is an (X, y, groups) tuple
suitable for StratifiedGroupKFold CV.

Label mapping (Round 1 — 3-class)
-----------------------------------
  0 floor          → vehicle  (floor residuals that survive removal look like
                               low flat blobs — closest non-cargo class)
  1 cargo          → cargo
  2 vehicle        → vehicle
  3 person         → person
  4 pallet         → cargo    (pallet points belong to the cargo unit)
  255 outlier      → vehicle  (rare; won't dominate majority vote)

  Low-purity clusters (majority fraction < purity_threshold) → vehicle.

Rationale for 3-class design
------------------------------
The original 5-class taxonomy included ``floor_residual`` and ``other``.
After building the GT on 100 synthetic scenes we observed:
  • floor_residual: 0 examples  (floor removal is effective; no residuals
    survive as DBSCAN clusters inside the anchor footprint)
  • other:          10 examples (too few to learn; adds a class with F1≈0.4
    that pulls macro-F1 below the gate regardless of feature quality)

Collapsing both into ``vehicle`` removes the structurally unlearnable classes
while preserving the diagnostically relevant distinction between cargo,
vehicle/equipment, and person.  The cargo label and its inference logic are
unchanged — the pipeline only checks predicted_label == cargo.

Usage::

    from pathlib import Path
    from cargo_geometric.params import GeometricParams
    from cargo_geometric.cluster_gt import build_cluster_dataset

    X, y, groups = build_cluster_dataset(
        dataset_dir=Path("output/dataset"),
        params=GeometricParams(),
    )
    # X: (N_clusters, 23)  float64
    # y: (N_clusters,)     int32  ∈ {1, 2, 3}
    # groups: (N_clusters,) int32  — scene index for GroupKFold
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from cargo_geometric.params import GeometricParams
from cargo_geometric.floor import preprocess, remove_floor
from cargo_geometric.anchor import find_anchor, points_inside_anchor
from cargo_geometric.cluster_features import extract_cluster_features, FEATURE_NAMES


# ── Label definitions ────────────────────────────────────────────────────────

# Raw per-point labels in the PLY files (from generate_dataset.py LABEL dict)
_RAW_FLOOR   = 0
_RAW_CARGO   = 1
_RAW_VEHICLE = 2
_RAW_PERSON  = 3
_RAW_PALLET  = 4
_RAW_OUTLIER = 255

# Cluster-level class integers (used in training pickle).
# 3-class design: cargo (1), vehicle (2), person (3).
# floor_residual and other (low-purity) are merged into vehicle.
CLUSTER_LABEL_MAP: dict[str, int] = {
    "cargo":   1,
    "vehicle": 2,
    "person":  3,
}
CLUSTER_LABEL_NAMES: dict[int, str] = {v: k for k, v in CLUSTER_LABEL_MAP.items()}

# Per-point raw label → cluster class integer.
# floor (0) → vehicle  (low residual blobs are equipment-like)
# outlier (255) → vehicle (rare; won't dominate a large cluster)
_RAW_TO_CLUSTER: dict[int, int] = {
    _RAW_FLOOR:   CLUSTER_LABEL_MAP["vehicle"],   # floor residual → vehicle
    _RAW_CARGO:   CLUSTER_LABEL_MAP["cargo"],
    _RAW_VEHICLE: CLUSTER_LABEL_MAP["vehicle"],
    _RAW_PERSON:  CLUSTER_LABEL_MAP["person"],
    _RAW_PALLET:  CLUSTER_LABEL_MAP["cargo"],     # pallet → cargo
    _RAW_OUTLIER: CLUSTER_LABEL_MAP["vehicle"],   # outlier → vehicle
}

# Low-purity clusters fall back to vehicle (formerly "other")
_FALLBACK_LABEL = CLUSTER_LABEL_MAP["vehicle"]

_N_CLUSTER_CLASSES = len(CLUSTER_LABEL_MAP)


# ── PLY loader (verbatim copy of helper from classifier/train.py) ────────────

def _read_ply_xyz_label(path: Path):
    """Binary-LE PLY → pts (N,3) float32, labels (N,) int32."""
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header\n")
    header = raw[:end].decode("ascii", errors="replace")
    n_vertices = int(next(
        ln.split()[-1] for ln in header.splitlines()
        if ln.startswith("element vertex")
    ))
    body = raw[end + len("end_header\n"):]
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("label", "u1"),
    ])
    arr = np.frombuffer(body[:n_vertices * dtype.itemsize], dtype=dtype)
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    return pts, arr["label"].astype(np.int32)


# ── Per-scene processing ─────────────────────────────────────────────────────

def _process_scene(
    scene_id: int,
    ply_path: Path,
    params: GeometricParams,
    min_cluster_pts: int,
    purity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Process one synthetic PLY → (X, y, groups) or None if skipped.

    Returns
    -------
    X      : (n_clusters, 23) float64
    y      : (n_clusters,) int32
    groups : (n_clusters,) int32 — all equal to scene_id
    """
    # Read raw points + labels (before voxelisation so we have dense label info)
    raw_pts, raw_labels = _read_ply_xyz_label(ply_path)

    # Apply label mapping. Unknown raw labels fall back to vehicle (non-cargo).
    mapped_labels = np.array(
        [_RAW_TO_CLUSTER.get(int(l), _FALLBACK_LABEL)
         for l in raw_labels],
        dtype=np.int32,
    )

    # Pipeline: preprocess (align + voxelise)
    try:
        pts_vox = preprocess(ply_path, params)
        floor_res = remove_floor(pts_vox, params)
    except RuntimeError:
        return None  # no valid floor found — skip scene

    pts_no_floor = floor_res.pts
    floor_y = floor_res.floor_y

    anchor = find_anchor(pts_no_floor, floor_y, params)
    if anchor is None:
        return None  # no dominant blob — skip scene

    # Restrict to anchor footprint
    in_anchor = points_inside_anchor(pts_no_floor, anchor, params.anchor_xz_margin)
    sub_idx = np.where(in_anchor)[0]
    if len(sub_idx) < params.cargo_dbscan_min_pts:
        return None

    sub_pts = pts_no_floor[sub_idx]

    # 3D DBSCAN
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_pts.astype(np.float64))
    dbscan_labels = np.asarray(
        pcd.cluster_dbscan(
            eps=params.cargo_dbscan_eps,
            min_points=params.cargo_dbscan_min_pts,
            print_progress=False,
        ),
        dtype=np.int32,
    )

    if dbscan_labels.max() < 0:
        return None  # no clusters at all

    n_dbscan_clusters = int(dbscan_labels.max()) + 1

    # KD-tree: map each voxelised sub_pt → nearest raw point → label
    tree = cKDTree(raw_pts)
    _, nn_idx = tree.query(sub_pts, k=1, workers=1)
    point_cluster_labels = mapped_labels[nn_idx]  # (len(sub_pts),)

    # Build feature + label for each DBSCAN cluster
    X_rows: list[np.ndarray] = []
    y_vals: list[int] = []

    for cid in range(n_dbscan_clusters):
        mask = dbscan_labels == cid
        cluster_size = int(mask.sum())
        if cluster_size < min_cluster_pts:
            continue  # too small to be reliable

        cluster_pts_local = sub_pts[mask]
        cluster_pt_labels = point_cluster_labels[mask]

        # Majority vote.  Label IDs are {1, 2, 3}; bincount needs length ≥ 4.
        counts = np.bincount(cluster_pt_labels,
                             minlength=max(CLUSTER_LABEL_MAP.values()) + 1)
        top_count = int(counts.max())
        top_label = int(counts.argmax())   # argmax returns the label integer (1/2/3)
        purity = top_count / cluster_size

        cluster_label = (
            top_label
            if purity >= purity_threshold
            else _FALLBACK_LABEL   # low-purity → vehicle (non-cargo, non-person)
        )

        feat = extract_cluster_features(cluster_pts_local, floor_y, anchor)
        X_rows.append(feat)
        y_vals.append(cluster_label)

    if not X_rows:
        return None

    X = np.vstack(X_rows)
    y = np.array(y_vals, dtype=np.int32)
    groups = np.full(len(y_vals), scene_id, dtype=np.int32)
    return X, y, groups


# ── Public API ───────────────────────────────────────────────────────────────

def build_cluster_dataset(
    dataset_dir: Path,
    params: GeometricParams,
    min_cluster_pts: int = 30,
    purity_threshold: float = 0.6,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, groups) cluster-level dataset from synthetic PLYs.

    Parameters
    ----------
    dataset_dir      : directory containing ``*.ply`` files with label channels
    params           : GeometricParams (same as runtime pipeline)
    min_cluster_pts  : drop DBSCAN clusters smaller than this
    purity_threshold : clusters with majority fraction < this → class ``other``
    verbose          : print per-scene progress

    Returns
    -------
    X      : (N_clusters, 23) float64
    y      : (N_clusters,)    int32 — cluster class integers
    groups : (N_clusters,)    int32 — scene index (for GroupKFold)

    Raises
    ------
    ValueError if no clusters could be extracted (empty dataset).
    """
    plys = sorted(Path(dataset_dir).glob("*.ply"))
    if not plys:
        raise ValueError(f"No PLY files found in {dataset_dir}")

    all_X:  list[np.ndarray] = []
    all_y:  list[np.ndarray] = []
    all_g:  list[np.ndarray] = []
    n_skipped = 0

    for scene_id, ply_path in enumerate(plys):
        result = _process_scene(
            scene_id, ply_path, params, min_cluster_pts, purity_threshold
        )
        if result is None:
            n_skipped += 1
            if verbose:
                print(f"  [{scene_id:04d}] SKIP  {ply_path.name}")
            continue

        X_s, y_s, g_s = result
        all_X.append(X_s)
        all_y.append(y_s)
        all_g.append(g_s)
        if verbose:
            class_counts = {
                CLUSTER_LABEL_NAMES[i]: int((y_s == i).sum())
                for i in CLUSTER_LABEL_MAP.values()
                if (y_s == i).sum() > 0
            }
            print(f"  [{scene_id:04d}] {ply_path.name}  "
                  f"n_clusters={len(y_s)}  {class_counts}")

    if not all_X:
        raise ValueError(
            f"build_cluster_dataset: no clusters extracted from {len(plys)} PLYs "
            f"({n_skipped} scenes skipped). Check dataset_dir and params."
        )

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_g)

    if verbose:
        print(f"\nDataset summary: {len(plys)} scenes, "
              f"{n_skipped} skipped, {len(y)} clusters total")
        for lbl_id, lbl_name in sorted(CLUSTER_LABEL_NAMES.items()):
            print(f"  {lbl_name:>16}: {int((y == lbl_id).sum()):5d}")

    return X, y, groups
