#!/usr/bin/env python3
"""
Geometric pipeline runner — Stage 0 (preprocess) → Stage 1 (floor removal)
→ Stage 2 (pallet detection).

Usage:
    python3 scripts/run_geometric.py <input.ply> <output_dir>

Outputs (in <output_dir>):
    <stem>_stage12.ply     debug PLY (2=floor blue, 1=rest red, 4=pallet yellow)
    <stem>_stage12.json    metadata: floor + list of pallet candidates

Run from datageneration/ root.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cargo_geometric.params import GeometricParams                 # noqa: E402
from cargo_geometric.floor import preprocess, remove_floor         # noqa: E402
from cargo_geometric.anchor import find_anchor, points_inside_anchor  # noqa: E402
from cargo_geometric.pallet import detect_pallets                  # noqa: E402
from cargo_geometric.cargo import extract_cargo                    # noqa: E402
from ply_io.ply import save_ply                                    # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: run_geometric.py <input.ply> <output_dir>", file=sys.stderr)
        return 2

    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    params = GeometricParams()

    t0 = time.perf_counter()
    pts = preprocess(in_path, params)
    t_pre = time.perf_counter() - t0

    t1 = time.perf_counter()
    floor = remove_floor(pts, params)
    t_floor = time.perf_counter() - t1

    # Stage 2 operates on the non-floor subset
    nonfloor_idx = np.where(~floor.floor_mask)[0]
    pts_no_floor = pts[nonfloor_idx]

    t2a = time.perf_counter()
    anchor = find_anchor(pts_no_floor, floor.floor_y, params)
    t_anchor = time.perf_counter() - t2a

    t2b = time.perf_counter()
    pallet_res = detect_pallets(pts_no_floor, floor.floor_y, params, anchor=anchor)
    t_pallet = time.perf_counter() - t2b

    t3 = time.perf_counter()
    cargo_res = extract_cargo(pts_no_floor, anchor, params) if anchor is not None else None
    t_cargo = time.perf_counter() - t3

    # Debug viz labels:
    #   2=floor (blue), 1=rest non-cargo (red, includes person + outside-anchor),
    #   3=cargo (green) — only the kept 3D-DBSCAN cluster
    labels = np.full(len(pts), 1, dtype=np.uint8)
    labels[floor.floor_mask] = 2
    if cargo_res is not None:
        global_cargo = nonfloor_idx[cargo_res.cargo_global_idx]
        labels[global_cargo] = 3

    stem = in_path.stem
    out_ply = out_dir / f"{stem}_stage123.ply"
    out_json = out_dir / f"{stem}_stage123.json"

    save_ply(out_ply, pts, labels)

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
            }
        ),
        "timing_s": {
            "preprocess": round(t_pre, 3),
            "floor_removal": round(t_floor, 3),
            "anchor": round(t_anchor, 3),
            "pallet_detect": round(t_pallet, 3),
            "cargo_extract": round(t_cargo, 3),
        },
    }
    out_json.write_text(json.dumps(meta, indent=2))

    pal_summary = (
        f"pallets={len(pallet_res.pallets)}"
        + (
            "  best={:.2f}x{:.2f}m N={}".format(
                pallet_res.pallets[0].footprint_w,
                pallet_res.pallets[0].footprint_d,
                pallet_res.pallets[0].n_points,
            )
            if pallet_res.pallets
            else "  (none)"
        )
    )
    anchor_str = (
        f"anchor={anchor.n_points}pts {anchor.x_max-anchor.x_min:.2f}x{anchor.z_max-anchor.z_min:.2f}"
        if anchor is not None else "anchor=NONE"
    )
    cargo_str = (
        f"cargo={int(len(cargo_res.cargo_pts))}pts ({cargo_res.n_clusters} clusters)"
        if cargo_res is not None else "cargo=NONE"
    )
    print(
        f"[{stem}] N={len(pts):>7d} floor={int(floor.floor_mask.sum()):>6d} "
        f"y={floor.floor_y:+.3f} {anchor_str} {cargo_str}  "
        f"t={t_pre+t_floor+t_anchor+t_pallet+t_cargo:.2f}s → {out_ply.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
