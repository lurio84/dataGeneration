"""
cargo_geometric/volume.py — 2.5D height-field volume estimation.

Computes cargo volume by discretising the XZ plane into a regular grid,
taking the maximum point height in each cell, and summing cell volumes.

This approach is shape-agnostic: it handles boxes, bags, cylinders, and
irregular loads equally well because it only relies on the top-surface
profile rather than any geometric model.

The height field integrates correctly for:
  * Axis-aligned boxes: footprint × height
  * Irregular bultos (bags, big-bags): area under the top surface
  * Semi-ellipsoidal shapes: approximates 2/3 · π · a · b · c
"""

from __future__ import annotations

import numpy as np


def height_field_volume(
    cargo_pts: np.ndarray,
    floor_y: float,
    cell_size: float = 0.04,
    min_height: float = 0.10,
) -> dict:
    """2.5D height-field volume of a point set above a known floor.

    For each XZ cell of side ``cell_size``, the local height is
    ``max(Y_in_cell) - floor_y``.  Cells whose local height is below
    ``min_height`` are treated as empty (floor residuals, noise, holes).

    Parameters
    ----------
    cargo_pts : (N, 3) float array
        Cargo points in world coordinates (X, Y, Z).
    floor_y : float
        World-Y coordinate of the floor plane.
    cell_size : float
        Grid cell side length in metres.  Default 0.04 m ≈ voxel_size × 1.2.
    min_height : float
        Minimum valid cell height in metres.  Cells below this threshold are
        excluded to suppress floor residuals and sensor noise.

    Returns
    -------
    dict with keys:
        ``volume_m3``    : float — Σ cell_area × cell_height over occupied cells
        ``footprint_m2`` : float — total area of occupied cells
        ``mean_height``  : float — mean height of occupied cells
        ``max_height``   : float — tallest occupied cell
        ``n_cells``      : int   — number of occupied cells
        ``height_field`` : np.ndarray (H, W) float64 — per-cell height (0 = empty)
        ``cell_size``    : float — echoed for traceability
    """
    _empty = {
        "volume_m3":    0.0,
        "footprint_m2": 0.0,
        "mean_height":  0.0,
        "max_height":   0.0,
        "n_cells":      0,
        "height_field": np.zeros((0, 0), dtype=np.float64),
        "cell_size":    cell_size,
    }

    if len(cargo_pts) == 0:
        return _empty

    x = cargo_pts[:, 0].astype(np.float64)
    y = cargo_pts[:, 1].astype(np.float64)
    z = cargo_pts[:, 2].astype(np.float64)

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    # Number of cells needed to cover [x_min, x_max] and [z_min, z_max].
    # ceil(range / cell_size) is exact when range is a multiple of cell_size,
    # and rounds up otherwise; max(1, ...) handles a single-column/row case.
    n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    n_rows = max(1, int(np.ceil((z_max - z_min) / cell_size)))

    # Grid indices: clipped to [0, n-1] so boundary points at exactly x_max / z_max
    # land in the last cell rather than triggering an out-of-bounds error.
    i_idx = np.clip(((x - x_min) / cell_size).astype(np.int64), 0, n_cols - 1)
    j_idx = np.clip(((z - z_min) / cell_size).astype(np.int64), 0, n_rows - 1)

    # Reduce: maximum Y per (row, col) cell.
    # Unvisited cells initialise to -inf so they never exceed min_height.
    hf = np.full((n_rows, n_cols), -np.inf, dtype=np.float64)
    np.maximum.at(hf, (j_idx, i_idx), y)

    # Convert absolute Y to height above floor.
    hf -= floor_y

    # Mask out cells below min_height (includes unvisited cells at -inf).
    valid = hf >= min_height
    hf[~valid] = 0.0  # zero out for clean height_field output

    n_cells = int(valid.sum())
    if n_cells == 0:
        return {**_empty, "height_field": hf}

    cell_area   = cell_size ** 2
    heights     = hf[valid]
    volume_m3   = float((heights * cell_area).sum())
    footprint   = float(n_cells * cell_area)
    mean_height = float(heights.mean())
    max_height  = float(heights.max())

    return {
        "volume_m3":    volume_m3,
        "footprint_m2": footprint,
        "mean_height":  mean_height,
        "max_height":   max_height,
        "n_cells":      n_cells,
        "height_field": hf,
        "cell_size":    cell_size,
    }
