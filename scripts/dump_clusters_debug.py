#!/usr/bin/env python3
"""
Dump pre-scoring DBSCAN clusters for the 5 problematic scenarios
(Esc07, 08, 09, 10, 13, Cap01) so we can inspect visually whether the
cargo is a separate cluster that loses the scoring, or whether it is
fused with the jack/pallet into a single cluster.

Writes one PLY per scenario into results/debug_clusters/, with:
  - floor          : dark grey
  - outside anchor : light grey
  - clusters inside anchor, sorted by size, colored:
      rank0 red, rank1 green, rank2 blue, rank3 yellow,
      rank4 magenta, rank5 cyan, rest orange
  - noise (DBSCAN label -1) inside anchor : dim purple

Also prints a table per scenario with per-cluster stats:
  id | n_pts | h_max | xz_extent | h/xz | dist | score_nh_d
so we can decide which scoring fix (if any) generalizes.

Run from datageneration/ root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cargo_geometric.params import GeometricParams
from cargo_geometric.floor import preprocess, remove_floor
from cargo_geometric.anchor import find_anchor, points_inside_anchor


SCENARIOS = [7, 8, 9, 10, 13]

PALETTE = np.array([
    (220, 50, 50),    # rank0 red
    (60, 200, 80),    # rank1 green
    (60, 120, 255),   # rank2 blue
    (240, 220, 60),   # rank3 yellow
    (220, 80, 220),   # rank4 magenta
    (60, 220, 220),   # rank5 cyan
    (250, 150, 40),   # rest  orange
], dtype=np.uint8)

FLOOR_RGB   = np.array((55, 55, 55),   dtype=np.uint8)
OUTSIDE_RGB = np.array((110, 110, 110), dtype=np.uint8)
NOISE_RGB   = np.array((90, 40, 110),  dtype=np.uint8)


def _write_ply_rgb(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    n = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    data = np.empty(n, dtype=dtype)
    data["x"] = pts[:, 0].astype(np.float32)
    data["y"] = pts[:, 1].astype(np.float32)
    data["z"] = pts[:, 2].astype(np.float32)
    data["r"] = rgb[:, 0]
    data["g"] = rgb[:, 1]
    data["b"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def process_scenario(esc: int, out_dir: Path, params: GeometricParams) -> None:
    in_ply = (
        ROOT / "predictions" / "pipeline_demo"
        / f"Escenario_{esc:02d}" / "Captura_01_person_filtered.ply"
    )
    if not in_ply.exists():
        print(f"[Esc{esc:02d}] SKIP — missing {in_ply}")
        return

    pts = preprocess(in_ply, params)
    floor = remove_floor(pts, params)
    nonfloor_idx = np.where(~floor.floor_mask)[0]
    pts_nf = pts[nonfloor_idx]

    anchor = find_anchor(pts_nf, floor.floor_y, params)
    if anchor is None:
        print(f"[Esc{esc:02d}] anchor=NONE — nothing to cluster")
        return

    in_anch_mask = points_inside_anchor(pts_nf, anchor, params.anchor_xz_margin)
    sub_idx = np.where(in_anch_mask)[0]
    sub_pts = pts_nf[sub_idx]

    if len(sub_pts) < params.cargo_dbscan_min_pts:
        print(f"[Esc{esc:02d}] too few points in anchor ({len(sub_pts)})")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_pts.astype(np.float64))
    lbl = np.asarray(
        pcd.cluster_dbscan(
            eps=params.cargo_dbscan_eps,
            min_points=params.cargo_dbscan_min_pts,
            print_progress=False,
        ),
        dtype=np.int32,
    )

    n_clusters = int(lbl.max()) + 1 if lbl.max() >= 0 else 0
    cx, cz = float(anchor.center_xz[0]), float(anchor.center_xz[1])

    # Per-cluster stats
    stats = []
    for cid in range(n_clusters):
        m = lbl == cid
        c = sub_pts[m]
        if len(c) < params.cargo_dbscan_min_pts:
            continue
        h_max = float(c[:, 1].max() - floor.floor_y)
        xz_extent = float(
            np.hypot(c[:, 0].max() - c[:, 0].min(),
                     c[:, 2].max() - c[:, 2].min())
        )
        cent = c[:, [0, 2]].mean(axis=0)
        dist = float(np.hypot(cent[0] - cx, cent[1] - cz))
        score = len(c) * h_max / (0.3 + dist)
        ratio = h_max / xz_extent if xz_extent > 1e-6 else float("inf")
        stats.append({
            "cid": cid, "n": int(m.sum()), "h": h_max,
            "xz": xz_extent, "h_xz": ratio, "dist": dist, "score": score,
        })

    # Rank by size (so colors are stable with size, independent of DBSCAN cid)
    stats_by_size = sorted(stats, key=lambda s: -s["n"])
    cid_to_rank = {s["cid"]: r for r, s in enumerate(stats_by_size)}
    chosen_cid = max(stats, key=lambda s: s["score"])["cid"] if stats else -1

    # Keep only top-6 clusters (by size). Discard the rest entirely.
    TOP_K = 6
    keep_cids = {s["cid"] for s in stats_by_size[:TOP_K]}

    keep_pts_list = []
    keep_rgb_list = []

    # Background: full cloud (floor + non-floor) dimmed, decimated x5 so the
    # scene is visible but doesn't overwhelm the colored clusters.
    bg_all = pts
    step = 5
    bg = bg_all[::step]
    keep_pts_list.append(bg)
    keep_rgb_list.append(
        np.tile(np.array((45, 45, 45), dtype=np.uint8), (len(bg), 1))
    )

    for cid in keep_cids:
        m_local = lbl == cid
        color = PALETTE[min(cid_to_rank[cid], len(PALETTE) - 1)]
        p = sub_pts[m_local]
        keep_pts_list.append(p)
        keep_rgb_list.append(np.tile(color, (len(p), 1)).astype(np.uint8))

    # Anchor markers: center cross + 4 bbox corners, in white.
    # Materialize them as dense point clouds so they're visible in CC.
    y_lo = float(floor.floor_y) + 0.02
    y_hi = float(pts_nf[:, 1].max())
    marker_pts = []
    # Vertical white line at anchor center
    for y in np.linspace(y_lo, y_hi, 80):
        marker_pts.append([cx, y, cz])
    # 4 corner verticals (anchor XZ bbox)
    for xc, zc in [
        (anchor.x_min, anchor.z_min),
        (anchor.x_min, anchor.z_max),
        (anchor.x_max, anchor.z_min),
        (anchor.x_max, anchor.z_max),
    ]:
        for y in np.linspace(y_lo, y_hi, 40):
            marker_pts.append([xc, y, zc])
    marker_pts = np.asarray(marker_pts, dtype=np.float32)
    marker_rgb = np.tile(np.array((255, 255, 255), dtype=np.uint8),
                         (len(marker_pts), 1))

    keep_pts_list.append(marker_pts)
    keep_rgb_list.append(marker_rgb)

    all_pts = np.concatenate(keep_pts_list, axis=0)
    all_rgb = np.concatenate(keep_rgb_list, axis=0)

    out_ply = out_dir / f"Esc{esc:02d}_Cap01_clusters.ply"
    _write_ply_rgb(out_ply, all_pts, all_rgb)

    # Print table
    print(f"\n[Esc{esc:02d}] anchor center=({cx:+.2f},{cz:+.2f}) "
          f"floor_y={floor.floor_y:+.3f} clusters={n_clusters}")
    print(f"  {'rank':>4} {'cid':>3} {'col':>7} {'n':>6} "
          f"{'h':>5} {'xz':>5} {'h/xz':>5} {'dist':>5} {'score':>9}  winner")
    color_names = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange"]
    for s in stats_by_size:
        rank = cid_to_rank[s["cid"]]
        cname = color_names[min(rank, len(color_names) - 1)]
        win = "  <== score" if s["cid"] == chosen_cid else ""
        print(f"  {rank:>4} {s['cid']:>3} {cname:>7} {s['n']:>6} "
              f"{s['h']:>5.2f} {s['xz']:>5.2f} {s['h_xz']:>5.2f} "
              f"{s['dist']:>5.2f} {s['score']:>9.1f}{win}")

    print(f"  -> {out_ply.relative_to(ROOT)}")


def main() -> int:
    out_dir = ROOT / "results" / "debug_clusters"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = GeometricParams()
    for esc in SCENARIOS:
        process_scenario(esc, out_dir, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
