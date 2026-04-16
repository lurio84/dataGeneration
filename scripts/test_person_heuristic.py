#!/usr/bin/env python3
"""
test_person_heuristic.py — Detect standing persons in fused person+cargo clusters
using a geometry-only (no ML) heuristic.

Algorithm:
  1. Voxelize the aligned (Y-up, floor≈Y=0) cloud to 0.035 m.
  2. Separate floor (Y < FLOOR_Y_THRESH) from non-floor.
  3. Project non-floor on XZ plane, divide into CELL_SIZE_M × CELL_SIZE_M cells.
  4. Per cell: compute height_max (max Y) and height_range (Y_max − Y_min).
  5. "Tall column" cells: height_max > MIN_HEIGHT_M AND height_range > MIN_HEIGHT_RANGE_M.
  6. Find 4-connected components of tall-column cells on the 2D grid.
  7. For each component:
     - footprint_m2 = n_cells × CELL_SIZE_M²
     - on_edge = any cell is within EDGE_MARGIN_CELLS of the non-floor bounding box edge
  8. Person candidates: footprint_m2 < MAX_FOOTPRINT_M2 AND on_edge.
  9. Points in person-candidate cells → boolean mask True = person.

Run from: /home/lronquilloext/Documents/Logicarc/datageneration/
  python3 scripts/test_person_heuristic.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.predict import read_ply_xyz, align_to_synthetic  # noqa: E402

RESOURCES = REPO_ROOT.parent / "Resources"
TIME_PROCESS = RESOURCES / "time_process"
OUT_DIR = REPO_ROOT / "output" / "eval_real_20esc" / "heuristic"

# ── Tunable parameters ────────────────────────────────────────────────────────
VOXEL_SIZE_M        = 0.035   # voxelization resolution (matches pipeline)
CELL_SIZE_M         = 0.15    # XZ grid cell size
MIN_HEIGHT_M        = 1.0     # height_max threshold for "tall column"
MIN_HEIGHT_RANGE_M  = 0.8     # height_range threshold for "tall column"
MAX_FOOTPRINT_M2    = 0.5     # max XZ footprint to be a person candidate
EDGE_MARGIN_CELLS   = 2       # cells from the grid edge counted as "edge"
FLOOR_Y_THRESH      = 0.05    # Y below which points are floor (post-alignment)

# ── Scenarios to evaluate ─────────────────────────────────────────────────────
SCENARIOS = [(f"Escenario_{i:02d}", "Captura_01") for i in range(1, 21)]

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_CARGO  = (220,  50,  50)   # red
COLOR_PERSON = ( 39, 174,  96)   # green
COLOR_FLOOR  = (100, 100, 100)   # grey


# ─────────────────────────────────────────────────────────────────────────────
# Voxelization
# ─────────────────────────────────────────────────────────────────────────────

def voxelize(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Downsample by assigning each point to a voxel key (floor division) and
    computing the centroid of each occupied voxel.

    Returns (M, 3) float32 with M <= N.
    """
    keys = np.floor(pts / voxel_size).astype(np.int32)
    # Pack 3 ints into a single int64 key for unique() — works for scenes < 2^20 voxels per axis.
    k = (keys[:, 0].astype(np.int64) * 1_000_003
         + keys[:, 1].astype(np.int64) * 1_009
         + keys[:, 2].astype(np.int64))
    order = np.argsort(k, kind="stable")
    k_sorted = k[order]
    pts_sorted = pts[order]

    _, first_idx = np.unique(k_sorted, return_index=True)
    split = np.split(pts_sorted, first_idx[1:])
    centroids = np.array([seg.mean(axis=0) for seg in split], dtype=np.float32)
    return centroids


# ─────────────────────────────────────────────────────────────────────────────
# Core heuristic
# ─────────────────────────────────────────────────────────────────────────────

def detect_person_mask(pts_aligned: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Given an aligned point cloud (N, 3) float32 with Y-up, floor≈Y=0,
    return:
      mask  : (N,) bool  — True = detected as person
      info  : dict with diagnostics

    Parameters are the module-level constants above.
    """
    N = len(pts_aligned)

    # ── 1. Separate floor from non-floor ──────────────────────────────────────
    floor_mask = pts_aligned[:, 1] < FLOOR_Y_THRESH
    nonfloor = pts_aligned[~floor_mask]

    if len(nonfloor) == 0:
        return np.zeros(N, dtype=bool), {"reason": "no non-floor points"}

    # ── 2. Build XZ grid ──────────────────────────────────────────────────────
    xs = nonfloor[:, 0]
    zs = nonfloor[:, 2]
    ys = nonfloor[:, 1]

    x_min, z_min = xs.min(), zs.min()
    # Grid indices for each non-floor point
    ci = np.floor((xs - x_min) / CELL_SIZE_M).astype(np.int32)
    cj = np.floor((zs - z_min) / CELL_SIZE_M).astype(np.int32)

    n_ci = int(ci.max()) + 1
    n_cj = int(cj.max()) + 1

    # ── 3. Per-cell stats ─────────────────────────────────────────────────────
    cell_y_max   = np.full((n_ci, n_cj), -np.inf, dtype=np.float32)
    cell_y_min   = np.full((n_ci, n_cj),  np.inf, dtype=np.float32)
    cell_n_pts   = np.zeros((n_ci, n_cj), dtype=np.int32)

    np.maximum.at(cell_y_max, (ci, cj), ys)
    np.minimum.at(cell_y_min, (ci, cj), ys)
    np.add.at(cell_n_pts, (ci, cj), 1)

    cell_y_range = cell_y_max - cell_y_min
    # Cells with no points → range = -inf - inf → set to 0
    no_pts = cell_n_pts == 0
    cell_y_max[no_pts]   = 0.0
    cell_y_min[no_pts]   = 0.0
    cell_y_range[no_pts] = 0.0

    # ── 4. Tall-column cells ──────────────────────────────────────────────────
    tall_col = (
        (cell_y_max   >= MIN_HEIGHT_M)
        & (cell_y_range >= MIN_HEIGHT_RANGE_M)
        & (~no_pts)
    )

    n_tall_cells = int(tall_col.sum())
    if n_tall_cells == 0:
        return np.zeros(N, dtype=bool), {
            "reason": (
                f"no cells with height_max>={MIN_HEIGHT_M}m "
                f"AND height_range>={MIN_HEIGHT_RANGE_M}m"
            ),
            "n_tall_cells": 0,
        }

    # ── 5. Connected components (4-connectivity) on tall-column grid ──────────
    visited = np.zeros((n_ci, n_cj), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    tall_idx = list(zip(*np.where(tall_col)))

    for start in tall_idx:
        si, sj = start
        if visited[si, sj]:
            continue
        # BFS
        comp: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque()
        q.append((si, sj))
        visited[si, sj] = True
        while q:
            i, j = q.popleft()
            comp.append((i, j))
            for di, dj in dirs:
                ni2, nj2 = i + di, j + dj
                if 0 <= ni2 < n_ci and 0 <= nj2 < n_cj:
                    if tall_col[ni2, nj2] and not visited[ni2, nj2]:
                        visited[ni2, nj2] = True
                        q.append((ni2, nj2))
        components.append(comp)

    # ── 6. Filter components: footprint + edge ────────────────────────────────
    person_cells: set[tuple[int, int]] = set()
    comp_info: list[dict] = []

    for comp in components:
        n_cells = len(comp)
        footprint_m2 = n_cells * CELL_SIZE_M * CELL_SIZE_M

        comp_is = [c[0] for c in comp]
        comp_js = [c[1] for c in comp]
        ci_min_comp, ci_max_comp = min(comp_is), max(comp_is)
        cj_min_comp, cj_max_comp = min(comp_js), max(comp_js)

        on_edge = (
            ci_min_comp <= EDGE_MARGIN_CELLS - 1
            or ci_max_comp >= n_ci - EDGE_MARGIN_CELLS
            or cj_min_comp <= EDGE_MARGIN_CELLS - 1
            or cj_max_comp >= n_cj - EDGE_MARGIN_CELLS
        )

        is_person = footprint_m2 < MAX_FOOTPRINT_M2 and on_edge
        comp_info.append({
            "n_cells": n_cells,
            "footprint_m2": footprint_m2,
            "on_edge": on_edge,
            "is_person": is_person,
            "y_max": float(max(cell_y_max[c] for c in comp)),
            "y_range": float(max(cell_y_range[c] for c in comp)),
            "ci_min": ci_min_comp, "ci_max": ci_max_comp,
            "cj_min": cj_min_comp, "cj_max": cj_max_comp,
        })

        if is_person:
            for c in comp:
                person_cells.add(c)

    # ── 7. Build per-point person mask ────────────────────────────────────────
    person_mask_nonfloor = np.zeros(len(nonfloor), dtype=bool)
    if person_cells:
        for idx, (pci, pcj) in enumerate(zip(ci, cj)):
            if (pci, pcj) in person_cells:
                person_mask_nonfloor[idx] = True

    # Map back to full point array
    full_person_mask = np.zeros(N, dtype=bool)
    nonfloor_indices = np.where(~floor_mask)[0]
    full_person_mask[nonfloor_indices[person_mask_nonfloor]] = True

    info = {
        "n_tall_cells": n_tall_cells,
        "n_components": len(components),
        "components": comp_info,
        "n_person_components": sum(1 for c in comp_info if c["is_person"]),
        "grid_shape": (n_ci, n_cj),
        "x_min": float(x_min),
        "z_min": float(z_min),
    }
    return full_person_mask, info


# ─────────────────────────────────────────────────────────────────────────────
# PLY writer (ASCII, CloudCompare-safe)
# ─────────────────────────────────────────────────────────────────────────────

def write_ply_ascii(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    """Write ASCII PLY x y z r g b. No camera block."""
    n = len(pts)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    pts_f = pts.astype(np.float32)
    rgb_u = rgb.astype(np.uint8)
    with path.open("w") as f:
        f.write(header)
        for i in range(n):
            f.write(
                f"{pts_f[i,0]:.6f} {pts_f[i,1]:.6f} {pts_f[i,2]:.6f}"
                f" {rgb_u[i,0]} {rgb_u[i,1]} {rgb_u[i,2]}\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_scenario(esc_name: str, cap_name: str) -> None:
    ply_path = TIME_PROCESS / esc_name / f"{cap_name}_tri_cloud.ply"
    tag = f"{esc_name}_{cap_name}"
    sep = "=" * 65

    print(f"\n{sep}")
    print(f"[{esc_name} / {cap_name}]  {ply_path.name}")
    print(sep)

    if not ply_path.exists():
        print(f"  ERROR: file not found: {ply_path}")
        return

    # ── Load & align ──────────────────────────────────────────────────────────
    raw = read_ply_xyz(ply_path)
    print(f"  Raw points loaded:  {len(raw):>8,}")

    pts_aligned, align_info = align_to_synthetic(raw, mode="auto")
    print(
        f"  Alignment:          floor_axis={align_info['floor_axis']}  "
        f"y_offset={align_info['y_offset']:+.4f}  "
        f"swapped={align_info['swapped']}"
    )

    # ── Voxelize ──────────────────────────────────────────────────────────────
    pts = voxelize(pts_aligned, VOXEL_SIZE_M)
    print(f"  After voxel {VOXEL_SIZE_M}m:    {len(pts):>8,}")

    # ── Run heuristic ─────────────────────────────────────────────────────────
    person_mask, info = detect_person_mask(pts)

    floor_mask = pts[:, 1] < FLOOR_Y_THRESH
    n_floor    = int(floor_mask.sum())
    n_total    = len(pts)
    n_person   = int(person_mask.sum())
    n_cargo    = n_total - n_floor - n_person

    print(f"\n  Point counts:")
    print(f"    Total (voxelized):  {n_total:>8,}")
    print(f"    Floor (Y<{FLOOR_Y_THRESH}m):    {n_floor:>8,}")
    print(f"    Person detected:    {n_person:>8,}")
    print(f"    Cargo conserved:    {n_cargo:>8,}")

    print(f"\n  Heuristic details:")
    print(f"    Grid shape (Xi×Zj):  {info.get('grid_shape', 'n/a')}")
    print(f"    Tall-column cells:   {info.get('n_tall_cells', 0)}")
    print(f"    Components found:    {info.get('n_components', 0)}")
    print(f"    Person components:   {info.get('n_person_components', 0)}")

    if "reason" in info:
        print(f"    No detection:  {info['reason']}")

    for idx, ci in enumerate(info.get("components", [])):
        status = "PERSON" if ci["is_person"] else "cargo/other"
        print(
            f"    Component {idx}: {ci['n_cells']} cells  "
            f"footprint={ci['footprint_m2']:.3f}m²  "
            f"on_edge={ci['on_edge']}  "
            f"y_max={ci['y_max']:.2f}m  y_range={ci['y_range']:.2f}m  "
            f"→ {status}"
        )
        if ci["is_person"]:
            # Real-world XZ extent
            x0 = info["x_min"] + ci["ci_min"] * CELL_SIZE_M
            x1 = info["x_min"] + (ci["ci_max"] + 1) * CELL_SIZE_M
            z0 = info["z_min"] + ci["cj_min"] * CELL_SIZE_M
            z1 = info["z_min"] + (ci["cj_max"] + 1) * CELL_SIZE_M
            print(f"      XZ extent:  X=[{x0:.2f},{x1:.2f}]m  Z=[{z0:.2f},{z1:.2f}]m")
            print(
                f"      Footprint:  {(x1-x0):.2f}m × {(z1-z0):.2f}m = {(x1-x0)*(z1-z0):.3f}m²"
            )

    # ── Export colored PLY ────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((n_total, 3), dtype=np.uint8)
    rgb[floor_mask]                        = COLOR_FLOOR
    rgb[~floor_mask & ~person_mask]        = COLOR_CARGO
    rgb[person_mask]                       = COLOR_PERSON

    out_ply = OUT_DIR / f"{tag}_heuristic.ply"
    write_ply_ascii(out_ply, pts, rgb)
    print(f"\n  Exported → {out_ply.relative_to(REPO_ROOT)}")
    print(f"    red=cargo({n_cargo:,})  green=person({n_person:,})  grey=floor({n_floor:,})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Person heuristic test")
    print(f"Parameters:")
    print(f"  VOXEL_SIZE_M       = {VOXEL_SIZE_M}")
    print(f"  CELL_SIZE_M        = {CELL_SIZE_M}")
    print(f"  MIN_HEIGHT_M       = {MIN_HEIGHT_M}")
    print(f"  MIN_HEIGHT_RANGE_M = {MIN_HEIGHT_RANGE_M}")
    print(f"  MAX_FOOTPRINT_M2   = {MAX_FOOTPRINT_M2}")
    print(f"  EDGE_MARGIN_CELLS  = {EDGE_MARGIN_CELLS}")
    print(f"  FLOOR_Y_THRESH     = {FLOOR_Y_THRESH}")

    for esc_name, cap_name in SCENARIOS:
        try:
            process_scenario(esc_name, cap_name)
        except Exception as exc:
            import traceback
            print(f"\n  ERROR in {esc_name}/{cap_name}: {exc}")
            traceback.print_exc()

    print(f"\nDone. PLYs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
