#!/usr/bin/env python3
"""
Export colored PLYs for visual validation in CloudCompare.

For each of 5 representative scenarios (Esc03/06/08/11/17, Captura_01):
  EscXX_Cap01_segmented.ply  — full cloud, per-cluster colors
  EscXX_Cap01_anchor.ply     — anchor points only (yellow)
  EscXX_Cap01_info.txt       — textual summary

PLY files are written manually (no PCL/camera blocks).
Colors:
  cargo   → red   (220, 50, 50)
  vehicle → blue  (0, 120, 255)
  person  → green (39, 174, 96)
  pallet  → yellow (255, 210, 0)
  floor   → dark grey (60, 60, 60)
  noise / unclassified → light grey (160, 160, 160)
  anchor (anchor.ply) → yellow (255, 220, 50)

Run from: /home/lronquilloext/Documents/Logicarc/datageneration/
  python3 scripts/export_visual_plys.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cargo_geometric.params import GeometricParams                       # noqa: E402
from cargo_geometric.floor import preprocess, synthetic_floor            # noqa: E402
from cargo_geometric.anchor import find_anchor, points_inside_anchor     # noqa: E402
from cargo_geometric.cargo import extract_cargo                          # noqa: E402
from cargo_geometric.volume import height_field_volume, obb_volume       # noqa: E402
from cargo_geometric.cluster_classifier import ClusterClassifier         # noqa: E402

RESOURCES   = REPO_ROOT.parent / "Resources"
TIME_PROCESS = RESOURCES / "time_process"
CLUSTERING  = RESOURCES / "clustering"
OUT_DIR     = REPO_ROOT / "output" / "eval_real_20esc" / "visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_PATH = REPO_ROOT / "models" / "cluster_classifier_lgbm.pkl"

SCENARIOS = [
    ("Escenario_03", "Captura_01", "Esc03"),
    ("Escenario_06", "Captura_01", "Esc06"),
    ("Escenario_08", "Captura_01", "Esc08"),
    ("Escenario_11", "Captura_01", "Esc11"),
    ("Escenario_17", "Captura_01", "Esc17"),
]

# ── Colors ─────────────────────────────────────────────────────────────────────
COLOR_CARGO   = (220,  50,  50)
COLOR_VEHICLE = (  0, 120, 255)
COLOR_PERSON  = ( 39, 174,  96)
COLOR_PALLET  = (255, 210,   0)
COLOR_FLOOR   = ( 60,  60,  60)
COLOR_NOISE   = (160, 160, 160)
COLOR_ANCHOR  = (255, 220,  50)


# ── PLY writer (manual, CloudCompare-safe) ─────────────────────────────────────

def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    """Write ASCII PLY with x y z r g b (uint8). No camera/face blocks."""
    n = len(pts)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    header = "\n".join(lines) + "\n"
    rows = np.column_stack([
        pts.astype(np.float32),
        rgb.astype(np.uint8),
    ])
    with path.open("w") as f:
        f.write(header)
        for row in rows:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}"
                    f" {int(row[3])} {int(row[4])} {int(row[5])}\n")
    print(f"  Wrote {n:>7d} pts → {path.name}")


# ── Paula reference parser ─────────────────────────────────────────────────────

import re as _re
_VOL_RE = _re.compile(
    r"\[GetVolume\] Best estimation with (?:ConvexHull|MomentOfIntertia): ([\d.]+) m3"
)

def parse_paula(esc_name: str, ply_stem: str) -> tuple[float | None, int]:
    d = CLUSTERING / esc_name
    files = sorted(d.glob(f"{ply_stem}_C*_get_bounding_box.txt"))
    if not files:
        return None, 0
    total = 0.0
    for f in files:
        m = _VOL_RE.search(f.read_text(errors="replace"))
        if m:
            total += float(m.group(1))
    return total, len(files)


# ── Per-scenario processing ────────────────────────────────────────────────────

def process_scenario(esc_name: str, cap_name: str, tag: str,
                     classifier: ClusterClassifier | None) -> None:
    ply_path = TIME_PROCESS / esc_name / f"{cap_name}_tri_cloud.ply"
    print(f"\n{'='*60}")
    print(f"[{tag}] {ply_path.name}")

    params = GeometricParams()
    params.cargo_dbscan_eps      = params.preprocessed_dbscan_eps
    params.cargo_dbscan_min_pts  = params.preprocessed_dbscan_min_pts

    # ── Stage 0 ──────────────────────────────────────────────────────────────
    pts = preprocess(ply_path, params, align_mode="fusion3d")
    print(f"  N={len(pts)} pts after voxel")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    floor = synthetic_floor(pts, params)
    nonfloor_idx = np.where(~floor.floor_mask)[0]
    pts_no_floor = pts[nonfloor_idx]
    print(f"  floor={floor.floor_mask.sum()} pts  y={floor.floor_y:.3f}")

    # ── Stage 2a anchor ───────────────────────────────────────────────────────
    anchor = find_anchor(pts_no_floor, floor.floor_y, params)
    if anchor is None:
        print("  ANCHOR=None, skipping")
        return

    in_anchor_mask = points_inside_anchor(pts_no_floor, anchor, params.anchor_xz_margin)
    anchor_pts = pts_no_floor[in_anchor_mask]
    print(
        f"  anchor={anchor.n_points}pts  "
        f"XZ={anchor.x_max-anchor.x_min:.2f}x{anchor.z_max-anchor.z_min:.2f}m  "
        f"h={anchor.height:.2f}m  anchor_pts={len(anchor_pts)}"
    )

    # ── Stage 3 cargo ─────────────────────────────────────────────────────────
    cargo_res = extract_cargo(pts_no_floor, anchor, params, classifier=classifier)

    # ── Stage 4 volumes ───────────────────────────────────────────────────────
    vol_hf = 0.0
    vol_anchor_ch = 0.0
    anchor_dims = [0.0, 0.0, 0.0]
    if cargo_res is not None:
        v = height_field_volume(cargo_res.cargo_pts, floor.floor_y,
                                horizontal_fill=True)
        vol_hf = v["volume_m3"]
    if len(anchor_pts) >= 4:
        av = obb_volume(anchor_pts)
        vol_anchor_ch = av["convhull_volume_m3"]
        ext = anchor_pts.max(axis=0) - anchor_pts.min(axis=0)
        anchor_dims = sorted(ext.tolist(), reverse=True)

    vol_paula, n_clusters_paula = parse_paula(esc_name, f"{cap_name}_tri_cloud")
    print(
        f"  vol_hf={vol_hf:.3f}  anchor_ch={vol_anchor_ch:.3f}  "
        f"paula={vol_paula}  dims={anchor_dims[0]:.3f}x{anchor_dims[1]:.3f}x{anchor_dims[2]:.3f}"
    )

    # ── Build per-point color arrays ──────────────────────────────────────────
    n_total = len(pts)
    rgb = np.full((n_total, 3), COLOR_NOISE, dtype=np.uint8)

    # Floor points
    rgb[floor.floor_mask] = COLOR_FLOOR

    # DBSCAN on anchor — need cluster assignments for coloring
    sub_idx = np.where(in_anchor_mask)[0]   # indices into pts_no_floor
    sub_pts = pts_no_floor[sub_idx]

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

    # Map cluster IDs → colors using classifier predictions
    n_clusters = int(dbscan_labels.max()) + 1 if dbscan_labels.max() >= 0 else 0
    cluster_colors: dict[int, tuple] = {}

    if classifier is not None and cargo_res is not None and \
            cargo_res.cluster_labels_pred is not None:
        for cid, pred in enumerate(cargo_res.cluster_labels_pred):
            name = classifier.inv_label_map.get(int(pred), "")
            if name == "vehicle":
                cluster_colors[cid] = COLOR_VEHICLE
            elif name == "person":
                cluster_colors[cid] = COLOR_PERSON
            elif name == "pallet":
                cluster_colors[cid] = COLOR_PALLET
            else:
                cluster_colors[cid] = COLOR_CARGO  # cargo or unknown → red
    else:
        # No classifier: cargo cluster(s) red, rest grey
        if cargo_res is not None:
            if cargo_res.cargo_policy == "rank0":
                cluster_colors[cargo_res.chosen_cluster_id] = COLOR_CARGO
            else:
                for cid in cargo_res.cluster_ids_classified_as_cargo:
                    cluster_colors[cid] = COLOR_CARGO

    # Paint anchor sub-cloud by cluster
    for cid in range(n_clusters):
        mask = dbscan_labels == cid
        color = cluster_colors.get(cid, COLOR_NOISE)
        global_sub = sub_idx[mask]            # indices into pts_no_floor
        global_pts = nonfloor_idx[global_sub]  # indices into pts (with floor)
        rgb[global_pts] = color

    # DBSCAN noise (label=-1) in anchor → grey
    noise_mask = dbscan_labels == -1
    if noise_mask.any():
        g_noise = nonfloor_idx[sub_idx[noise_mask]]
        rgb[g_noise] = COLOR_NOISE

    # Non-anchor non-floor points stay NOISE color (already set)

    # ── Export 1: segmented full cloud ────────────────────────────────────────
    seg_path = OUT_DIR / f"{tag}_Cap01_segmented.ply"
    write_ply(seg_path, pts, rgb)

    # ── Export 2: anchor points only (yellow) ────────────────────────────────
    anchor_global_idx = nonfloor_idx[sub_idx]  # indices into pts
    anchor_rgb = np.full((len(anchor_pts), 3), COLOR_ANCHOR, dtype=np.uint8)
    anch_path = OUT_DIR / f"{tag}_Cap01_anchor.ply"
    write_ply(anch_path, pts[anchor_global_idx], anchor_rgb)

    # ── Export 3: info.txt ────────────────────────────────────────────────────
    cluster_summary_lines = []
    if cargo_res is not None and cargo_res.cluster_labels_pred is not None:
        sizes = [int((dbscan_labels == cid).sum()) for cid in range(n_clusters)]
        for cid, (pred, sz) in enumerate(zip(cargo_res.cluster_labels_pred, sizes)):
            name = classifier.inv_label_map.get(int(pred), str(pred)) if classifier else str(pred)
            proba_str = ""
            if cargo_res.cluster_proba is not None:
                p = cargo_res.cluster_proba[cid]
                proba_str = "  proba=[" + ", ".join(f"{x:.2f}" for x in p) + "]"
            kept = cid in cargo_res.cluster_ids_classified_as_cargo
            cluster_summary_lines.append(
                f"  cluster {cid}: {sz:>5} pts  label={name}  kept={kept}{proba_str}"
            )
    elif cargo_res is not None:
        sizes = [int((dbscan_labels == cid).sum()) for cid in range(n_clusters)]
        for cid, sz in enumerate(sizes):
            kept = (cargo_res.cargo_policy == "rank0"
                    and cid == cargo_res.chosen_cluster_id)
            cluster_summary_lines.append(
                f"  cluster {cid}: {sz:>5} pts  [no classifier]  kept={kept}"
            )

    ratio_str = (
        f"{vol_anchor_ch / vol_paula:.4f}" if vol_paula and vol_paula > 0 else "n/a"
    )

    info_lines = [
        f"Scenario: {esc_name}  Captura: {cap_name}",
        f"PLY: {ply_path.name}",
        "",
        "── Points ──────────────────────────────────────────────",
        f"  Total (voxelized):    {len(pts):>7d}",
        f"  Floor:                {int(floor.floor_mask.sum()):>7d}",
        f"  Non-floor:            {len(pts_no_floor):>7d}",
        f"  Anchor (with margin): {len(anchor_pts):>7d}",
        f"  Cargo (post-ML):      {len(cargo_res.cargo_pts) if cargo_res else 0:>7d}",
        "",
        "── Volumes ─────────────────────────────────────────────",
        f"  anchor_ch  (ConvexHull on full anchor): {vol_anchor_ch:.4f} m³",
        f"  vol_hf     (height-field on cargo):     {vol_hf:.4f} m³",
        f"  vol_paula  (reference, sum clusters):   {vol_paula if vol_paula else 'n/a'} m³  ({n_clusters_paula} cluster(s))",
        f"  ratio_anchor_vs_paula:                  {ratio_str}",
        "",
        "── Anchor dimensions ───────────────────────────────────",
        f"  L×W×H (sorted desc): {anchor_dims[0]:.3f} × {anchor_dims[1]:.3f} × {anchor_dims[2]:.3f} m",
        f"  XZ footprint: {anchor.x_max-anchor.x_min:.3f} m (X)  {anchor.z_max-anchor.z_min:.3f} m (Z)",
        f"  Height:       {anchor.height:.3f} m",
        "",
        f"── Clusters (DBSCAN, eps={params.cargo_dbscan_eps:.3f}, min_pts={params.cargo_dbscan_min_pts}) ───",
        f"  Total clusters: {n_clusters}  (+ noise: {int((dbscan_labels==-1).sum())} pts)",
        f"  cargo_source: {cargo_res.cargo_source if cargo_res else 'N/A'}",
        f"  cargo_policy: {cargo_res.cargo_policy if cargo_res else 'N/A'}",
    ] + cluster_summary_lines

    info_path = OUT_DIR / f"{tag}_Cap01_info.txt"
    info_path.write_text("\n".join(info_lines) + "\n")
    print(f"  Info → {info_path.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # Load classifier
    classifier = None
    if CLASSIFIER_PATH.exists():
        try:
            classifier = ClusterClassifier.load(CLASSIFIER_PATH)
            print(f"Classifier loaded: {CLASSIFIER_PATH.name}")
            print(f"  label_map: {classifier.label_map}")
        except Exception as e:
            print(f"WARNING: Could not load classifier ({e}), running without")
    else:
        print(f"Classifier not found at {CLASSIFIER_PATH}, running without")

    for esc_name, cap_name, tag in SCENARIOS:
        try:
            process_scenario(esc_name, cap_name, tag, classifier)
        except Exception as e:
            import traceback
            print(f"  ERROR in {tag}: {e}")
            traceback.print_exc()

    print(f"\nDone. Visual PLYs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
