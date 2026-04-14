#!/usr/bin/env python3
"""
Geometric pipeline runner — Stage 0 (preprocess) → Stage 1 (floor removal)
→ Stage 2 (anchor + pallet detection) → Stage 3 (cargo extraction).

Usage:
    python3 scripts/run_geometric.py <input.ply> <output_dir> [--cluster-classifier PATH]

Arguments:
    input.ply        Input point cloud (PLY).
    output_dir       Directory for debug PLY and JSON metadata.
    --cluster-classifier PATH
                     Path to cluster classifier .pkl (produced by
                     train_cluster_classifier.py).  If omitted, falls back to
                     rank-0 heuristic (original behaviour).

Outputs (in <output_dir>):
    <stem>_stage123.ply     debug PLY (2=floor, 1=rest, 3=cargo)
    <stem>_stage123.json    full metadata including cluster_classifier section

Run from datageneration/ root.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cargo_geometric.params import GeometricParams                      # noqa: E402
from cargo_geometric.floor import preprocess, remove_floor              # noqa: E402
from cargo_geometric.anchor import find_anchor                          # noqa: E402
from cargo_geometric.pallet import detect_pallets                       # noqa: E402
from cargo_geometric.cargo import extract_cargo                         # noqa: E402
from ply_io.ply import save_ply                                         # noqa: E402


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Geometric cargo detection pipeline"
    )
    p.add_argument("input_ply",  type=Path, help="Input PLY file")
    p.add_argument("output_dir", type=Path, help="Output directory")
    p.add_argument(
        "--cluster-classifier", dest="cluster_classifier", type=Path,
        default=None, metavar="PATH",
        help="Path to cluster classifier .pkl (optional; enables ML cargo detection)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    in_path  = args.input_ply
    out_dir  = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    params = GeometricParams()

    # ── Load classifier (optional) ────────────────────────────────────────────
    classifier = None
    classifier_path_str = None
    if args.cluster_classifier is not None:
        classifier_path_str = str(args.cluster_classifier)
        try:
            from cargo_geometric.cluster_classifier import ClusterClassifier
            classifier = ClusterClassifier.load(args.cluster_classifier)
        except FileNotFoundError:
            warnings.warn(
                f"Cluster classifier not found: {args.cluster_classifier} — "
                f"falling back to rank-0 heuristic",
                stacklevel=2,
            )
            classifier = None
        except Exception as exc:
            warnings.warn(
                f"Failed to load cluster classifier ({exc}) — "
                f"falling back to rank-0 heuristic",
                stacklevel=2,
            )
            classifier = None

    # ── Stage 0: preprocess ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    pts = preprocess(in_path, params)
    t_pre = time.perf_counter() - t0

    # ── Stage 1: floor removal ────────────────────────────────────────────────
    t1 = time.perf_counter()
    floor = remove_floor(pts, params)
    t_floor = time.perf_counter() - t1

    nonfloor_idx = np.where(~floor.floor_mask)[0]
    pts_no_floor = pts[nonfloor_idx]

    # ── Stage 2a: anchor detection ────────────────────────────────────────────
    t2a = time.perf_counter()
    anchor = find_anchor(pts_no_floor, floor.floor_y, params)
    t_anchor = time.perf_counter() - t2a

    # ── Stage 2b: pallet detection ────────────────────────────────────────────
    t2b = time.perf_counter()
    pallet_res = detect_pallets(pts_no_floor, floor.floor_y, params, anchor=anchor)
    t_pallet = time.perf_counter() - t2b

    # ── Stage 3: cargo extraction ─────────────────────────────────────────────
    t3 = time.perf_counter()
    cargo_res = (
        extract_cargo(pts_no_floor, anchor, params, classifier=classifier)
        if anchor is not None else None
    )
    t_cargo = time.perf_counter() - t3

    # ── Debug PLY ─────────────────────────────────────────────────────────────
    # Labels: 2=floor (blue), 1=rest (red), 3=cargo (green)
    labels = np.full(len(pts), 1, dtype=np.uint8)
    labels[floor.floor_mask] = 2
    if cargo_res is not None:
        global_cargo = nonfloor_idx[cargo_res.cargo_global_idx]
        labels[global_cargo] = 3

    stem    = in_path.stem
    out_ply  = out_dir / f"{stem}_stage123.ply"
    out_json = out_dir / f"{stem}_stage123.json"

    save_ply(out_ply, pts, labels)

    # ── JSON metadata ─────────────────────────────────────────────────────────
    cluster_classifier_meta: dict = {
        "enabled": classifier is not None,
        "path":    classifier_path_str,
    }
    if cargo_res is not None and cargo_res.cluster_labels_pred is not None:
        cluster_classifier_meta.update({
            "n_clusters_seen":          cargo_res.n_clusters,
            "cluster_labels_pred":      cargo_res.cluster_labels_pred.tolist(),
            "cluster_labels_str": [
                classifier.inv_label_map.get(int(p), str(p))
                for p in cargo_res.cluster_labels_pred
            ] if classifier is not None else None,
            "cluster_ids_classified_as_cargo":
                cargo_res.cluster_ids_classified_as_cargo,
            "n_clusters_cargo": cargo_res.n_clusters_cargo,
        })

    meta = {
        "input": str(in_path),
        "voxel_size": params.voxel_size,
        "n_points_after_voxel": int(len(pts)),
        "floor": {
            "y_mean": floor.floor_y,
            "n_points": int(floor.floor_mask.sum()),
            "ransac_attempts": floor.attempts,
            "plane_abcd": list(floor.plane),
        },
        "anchor": (
            None if anchor is None else {
                "n_points": anchor.n_points,
                "x_range": [anchor.x_min, anchor.x_max],
                "z_range": [anchor.z_min, anchor.z_max],
                "height": round(anchor.height, 3),
                "center_xz": list(anchor.center_xz),
            }
        ),
        "pallet": {
            "n_candidates": len(pallet_res.pallets),
            "candidates": [
                {
                    "rank": i,
                    "cluster_id": int(p.cluster_id),
                    "n_points": p.n_points,
                    "center": p.center.tolist(),
                    "footprint_w": round(p.footprint_w, 4),
                    "footprint_d": round(p.footprint_d, 4),
                    "aspect": round(p.aspect, 3),
                }
                for i, p in enumerate(pallet_res.pallets)
            ],
        },
        "cargo": (
            None if cargo_res is None else {
                "n_points": int(len(cargo_res.cargo_pts)),
                "n_in_anchor": cargo_res.n_total_in_anchor,
                "n_clusters_in_anchor": cargo_res.n_clusters,
                "chosen_cluster_id": cargo_res.chosen_cluster_id,
                "chosen_cluster_size": cargo_res.chosen_cluster_size,
                "cargo_source": cargo_res.cargo_source,
                "n_clusters_cargo": cargo_res.n_clusters_cargo,
            }
        ),
        "cluster_classifier": cluster_classifier_meta,
        "timing_s": {
            "preprocess": round(t_pre, 3),
            "floor_removal": round(t_floor, 3),
            "anchor": round(t_anchor, 3),
            "pallet_detect": round(t_pallet, 3),
            "cargo_extract": round(t_cargo, 3),
        },
    }
    out_json.write_text(json.dumps(meta, indent=2))

    # ── Console summary ───────────────────────────────────────────────────────
    anchor_str = (
        f"anchor={anchor.n_points}pts "
        f"{anchor.x_max-anchor.x_min:.2f}x{anchor.z_max-anchor.z_min:.2f}m"
        if anchor is not None else "anchor=NONE"
    )
    if cargo_res is not None:
        src_tag = f"[{cargo_res.cargo_source}]"
        cargo_str = (
            f"cargo={int(len(cargo_res.cargo_pts))}pts "
            f"({cargo_res.n_clusters} clusters, {cargo_res.n_clusters_cargo} cargo) "
            f"{src_tag}"
        )
    else:
        cargo_str = "cargo=NONE"

    print(
        f"[{stem}] N={len(pts):>7d} floor={int(floor.floor_mask.sum()):>6d} "
        f"y={floor.floor_y:+.3f} {anchor_str} {cargo_str}  "
        f"t={t_pre+t_floor+t_anchor+t_pallet+t_cargo:.2f}s → {out_ply.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
