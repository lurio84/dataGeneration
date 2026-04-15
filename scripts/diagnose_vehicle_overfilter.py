#!/usr/bin/env python3
"""
Diagnose vehicle over-filtering in BBB real scenes.

For each BBB scene in output/bbb_vox035/, re-runs the geometric pipeline
up to the cluster classifier (preprocess → floor → anchor → DBSCAN →
predict_proba) and reports per-cluster details for every cluster predicted
as *vehicle*.  The goal is to distinguish:

  - Confident vehicle predictions (max_proba > 0.75): model is sure → the
    classifier is genuinely wrong about real cargo; needs retraining.
  - Uncertain vehicle predictions (max_proba ≤ 0.75): model is unsure →
    a confidence-thresholded exclusion (Case A fix) will recover these.

Usage (run from datageneration/ root):
    python3 scripts/diagnose_vehicle_overfilter.py

Outputs:
    output/diag_vehicle/bbb_N_vehicle_clusters.txt  — per-scene report
    (also printed to stdout)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cargo_geometric.params import GeometricParams                        # noqa: E402
from cargo_geometric.floor import preprocess, remove_floor                # noqa: E402
from cargo_geometric.anchor import find_anchor, points_inside_anchor      # noqa: E402
from cargo_geometric.cluster_features import (                            # noqa: E402
    extract_features_batch,
    _obb_dims,
)
from cargo_geometric.cluster_classifier import ClusterClassifier          # noqa: E402

import open3d as o3d                                                       # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
BBB_DIR    = ROOT / "output" / "bbb_vox035"
MODEL_PATH = ROOT / "models" / "cluster_classifier_lgbm.pkl"
OUT_DIR    = ROOT / "output" / "diag_vehicle"

# Confidence threshold used in the proposed fix.
CONFIDENCE_THRESHOLD = 0.75


def _dbscan_clusters(
    sub_pts: np.ndarray,
    eps: float,
    min_pts: int,
) -> tuple[np.ndarray, int]:
    """Run 3-D DBSCAN; return (labels, n_clusters).  Noise = −1."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_pts.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False),
        dtype=np.int32,
    )
    n = int(labels.max()) + 1 if labels.max() >= 0 else 0
    return labels, n


def _report_scene(
    ply_path: Path,
    classifier: ClusterClassifier,
    params: GeometricParams,
) -> str:
    """Run pipeline on one BBB scene and return a formatted report string."""

    lines: list[str] = []
    scene_name = ply_path.stem

    lines.append("=" * 72)
    lines.append(f"Scene: {scene_name}")
    lines.append("=" * 72)

    # ── Stage 0: preprocess ───────────────────────────────────────────────────
    pts = preprocess(ply_path, params)

    # ── Stage 1: floor removal ────────────────────────────────────────────────
    floor = remove_floor(pts, params)
    nonfloor_idx = np.where(~floor.floor_mask)[0]
    pts_no_floor = pts[nonfloor_idx]
    floor_y = floor.floor_y

    # ── Stage 2a: anchor ─────────────────────────────────────────────────────
    anchor = find_anchor(pts_no_floor, floor_y, params)
    if anchor is None:
        lines.append("  [SKIP] No anchor found.\n")
        return "\n".join(lines)

    # ── Restrict to anchor footprint (same as extract_cargo) ─────────────────
    in_anchor = points_inside_anchor(pts_no_floor, anchor, params.anchor_xz_margin)
    sub_idx   = np.where(in_anchor)[0]
    if len(sub_idx) < params.cargo_dbscan_min_pts:
        lines.append("  [SKIP] Too few points in anchor.\n")
        return "\n".join(lines)
    sub_pts = pts_no_floor[sub_idx]

    # ── 3D DBSCAN (same eps / min_pts as cargo extraction) ───────────────────
    dbscan_labels, n_clusters = _dbscan_clusters(
        sub_pts, params.cargo_dbscan_eps, params.cargo_dbscan_min_pts
    )
    if n_clusters == 0:
        lines.append("  [SKIP] DBSCAN found no clusters.\n")
        return "\n".join(lines)

    clusters_pts = [sub_pts[dbscan_labels == cid] for cid in range(n_clusters)]
    sizes        = np.bincount(dbscan_labels[dbscan_labels >= 0])

    # ── Feature extraction + predict_proba ───────────────────────────────────
    X = extract_features_batch(clusters_pts, floor_y=float(pts_no_floor[:, 1].min()),
                               anchor=anchor)
    preds, proba = classifier.predict_with_proba(X)   # (n, n_classes)

    # Map class index → column index (model's predict_proba column order)
    classes = list(classifier.model.classes_)           # list of int labels
    inv_lbl = classifier.inv_label_map                  # int → class name

    vehicle_lbl = classifier.label_map.get("vehicle")

    lines.append(
        f"  n_clusters={n_clusters}  n_points_in_anchor={len(sub_pts)}"
        f"  floor_y={floor_y:.4f}"
    )
    lines.append("")

    # ── Header ───────────────────────────────────────────────────────────────
    col_names = [inv_lbl.get(c, str(c)) for c in classes]
    proba_hdr = "  ".join(f"P({n})" for n in col_names)
    lines.append(
        f"{'cid':>4} {'n_pts':>6} {'pred':>8}  {proba_hdr}"
        f"  {'obb_long':>8} {'obb_short':>9} {'obb_h':>6}"
        f"  {'asp_ls':>6} {'asp_hl':>6}"
        f"  {'ctr_x':>7} {'ctr_z':>7} {'y_min_fl':>9}"
    )
    lines.append("-" * 115)

    # ── Per-vehicle-cluster rows ──────────────────────────────────────────────
    n_vehicle_total   = 0
    n_vehicle_dubious = 0   # max_proba < CONFIDENCE_THRESHOLD

    for cid in range(n_clusters):
        pred = int(preds[cid])
        if pred != vehicle_lbl:
            continue

        n_vehicle_total += 1
        p_row    = proba[cid]                          # (n_classes,) float64
        max_p    = float(p_row.max())
        if max_p < CONFIDENCE_THRESHOLD:
            n_vehicle_dubious += 1

        p_str = "  ".join(f"{p:7.4f}" for p in p_row)

        cpts = clusters_pts[cid]
        try:
            obb_long, obb_short, obb_h, _, _ = _obb_dims(cpts)
        except Exception:
            obb_long = obb_short = obb_h = float("nan")

        asp_ls = obb_long / max(obb_short, 1e-6)
        asp_hl = obb_h    / max(obb_long,  1e-6)

        ctr_x = float(cpts[:, 0].mean())
        ctr_z = float(cpts[:, 2].mean())
        y_min_above_floor = float(cpts[:, 1].min()) - floor_y

        pred_name = inv_lbl.get(pred, str(pred))
        marker = " *" if max_p < CONFIDENCE_THRESHOLD else "  "
        lines.append(
            f"{cid:>4} {sizes[cid]:>6} {pred_name:>8}  {p_str}"
            f"  {obb_long:>8.3f} {obb_short:>9.3f} {obb_h:>6.3f}"
            f"  {asp_ls:>6.2f} {asp_hl:>6.2f}"
            f"  {ctr_x:>7.3f} {ctr_z:>7.3f} {y_min_above_floor:>9.3f}"
            f"{marker}"
        )

    if n_vehicle_total == 0:
        lines.append("  (no vehicle-predicted clusters)")

    lines.append("-" * 115)
    lines.append(
        f"  vehicle clusters total : {n_vehicle_total}"
    )
    lines.append(
        f"  dubious (max_p < {CONFIDENCE_THRESHOLD:.2f}) : {n_vehicle_dubious}"
        f"  ({100*n_vehicle_dubious/max(n_vehicle_total,1):.0f}%)"
    )
    lines.append(f"  confident (max_p >= {CONFIDENCE_THRESHOLD:.2f}): "
                 f"{n_vehicle_total - n_vehicle_dubious}")
    lines.append("")
    lines.append("  * = cluster would be RETAINED with confidence-thresholded fix")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params = GeometricParams()

    if not MODEL_PATH.exists():
        print(f"ERROR: classifier not found at {MODEL_PATH}", file=sys.stderr)
        return 1

    classifier = ClusterClassifier.load(MODEL_PATH)

    ply_files = sorted(BBB_DIR.glob("bbb_*_vox035.ply"))
    if not ply_files:
        print(f"ERROR: no PLY files found in {BBB_DIR}", file=sys.stderr)
        return 1

    global_total   = 0
    global_dubious = 0

    for ply_path in ply_files:
        report = _report_scene(ply_path, classifier, params)
        print(report)

        out_txt = OUT_DIR / f"{ply_path.stem}_vehicle_clusters.txt"
        out_txt.write_text(report)
        print(f"  → saved: {out_txt.relative_to(ROOT)}")

        # Extract summary counts from report for the global tally
        for line in report.splitlines():
            if line.strip().startswith("vehicle clusters total"):
                try:
                    global_total += int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.strip().startswith("dubious"):
                try:
                    global_dubious += int(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

    print("=" * 72)
    print("GLOBAL SUMMARY (all 6 BBB scenes)")
    print("=" * 72)
    print(f"  vehicle clusters total : {global_total}")
    print(f"  dubious (max_p < {CONFIDENCE_THRESHOLD:.2f}) : {global_dubious}"
          f"  ({100*global_dubious/max(global_total,1):.0f}%)")
    print(f"  confident              : {global_total - global_dubious}")
    if global_dubious > global_total * 0.3:
        print("\n  → RECOMMENDATION: Case A (confidence-threshold fix) is likely effective.")
        print("    Most vehicle-cluster exclusions are uncertain; lowering effective")
        print("    exclusion by requiring max_proba >= 0.75 should recover real cargo.")
    else:
        print("\n  → WARNING: Most vehicle predictions are confident (> 0.75).")
        print("    Case A fix will have limited effect. Report examples to user")
        print("    and wait for instructions (likely requires dataset retraining).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
