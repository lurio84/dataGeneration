"""
cargo_geometric/volume.py — 2.5D height-field volume estimation.

Computes cargo volume by discretising the XZ plane into a regular grid,
taking the maximum point height in each cell, and summing cell volumes.

This approach is shape-agnostic: it handles boxes, bags, cylinders, and
irregular loads equally well because it only relies on the top-surface
profile rather than any geometric model.

Effective-floor offset
----------------------
The method integrates from ``floor_y + pallet_offset``, not from bare
``floor_y``.  By default ``pallet_offset = 0.144 m`` (EUR pallet height),
so the measurement represents the *cargo net volume above the pallet* rather
than the total column from the warehouse floor.

Cells whose max_Y is below ``floor_y + pallet_offset + min_cargo_h`` are
treated as empty (pallet surface, floor residuals, noise).  Valid cells
contribute ``(max_Y - floor_y - pallet_offset) × cell_area``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as nd_label


def height_field_volume(
    cargo_pts: np.ndarray,
    floor_y: float,
    cell_size: float = 0.04,
    pallet_offset: float = 0.144,
    min_cargo_h: float = 0.05,
    tall_threshold: float = 0.20,
) -> dict:
    """2.5D height-field volume of a point set above the effective floor.

    The *effective floor* is at ``floor_y + pallet_offset``.  Each XZ cell
    of side ``cell_size`` contributes ``max(0, max_Y_in_cell - effective_floor)``
    as its local height.  Cells whose net height is below ``min_cargo_h`` are
    treated as empty (pallet surface, noise, holes).

    Connected-component halo filter
    --------------------------------
    After thresholding at ``min_cargo_h``, the EUR pallet footprint (1.2 × 0.8 m)
    leaves a halo of cells just outside the cargo footprint that clear
    ``min_cargo_h`` due to sensor noise (σ ≈ 0.035 m) pushing some pallet-surface
    points above the threshold.  To suppress this halo without discarding genuine
    flat cargo:

    1. Build anchor mask M  = cells with net_height >= min_cargo_h.
    2. Build tall mask M_tall = cells with net_height >= tall_threshold (default 0.20 m).
    3. **If M_tall is empty** (all cargo is flat, none above tall_threshold):
       skip the filter — keep all M cells (permissive fallback for flat loads).
    4. **Otherwise**: compute 4-connected components of M; keep only components
       that contain at least one M_tall cell.  Isolated low-height halo components
       anchored only to the pallet surface are discarded.

    Parameters
    ----------
    cargo_pts : (N, 3) float array
        Cargo points in world coordinates (X, Y, Z).
    floor_y : float
        World-Y coordinate of the floor plane (from RANSAC floor removal).
    cell_size : float
        Grid cell side length in metres.  Default 0.04 m ≈ voxel_size × 1.2.
    pallet_offset : float
        Height of the pallet platform above ``floor_y`` in metres.
        Default 0.144 m = EUR pallet height.  Set to 0.0 when no pallet is
        present or when measuring total column height from the raw floor.
    min_cargo_h : float
        Minimum net cargo height in metres (above ``floor_y + pallet_offset``).
        Cells below this threshold are discarded to suppress pallet-surface
        points, floor residuals, and sensor noise.  Default 0.05 m.
    tall_threshold : float
        Minimum net height for a cell to count as a "tall anchor" for the
        connected-component halo filter.  Default 0.20 m.  When no cell
        reaches this height the filter is bypassed (flat-cargo fallback).

    Returns
    -------
    dict with keys:
        ``volume_m3``       : float — Σ cell_area × net_height over occupied cells
        ``footprint_m2``    : float — total area of occupied cells
        ``mean_height``     : float — mean net height of occupied cells
        ``max_height``      : float — tallest occupied cell (net height)
        ``n_cells``         : int   — number of occupied cells
        ``height_field``    : np.ndarray (H, W) float64 — net height per cell (0 = empty)
        ``cell_size``       : float — echoed for traceability
        ``pallet_offset``   : float — echoed for traceability
        ``tall_threshold``  : float — echoed for traceability
        ``halo_filter_used``: bool  — True when connected-component filter was applied
    """
    _empty = {
        "volume_m3":        0.0,
        "footprint_m2":     0.0,
        "mean_height":      0.0,
        "max_height":       0.0,
        "n_cells":          0,
        "height_field":     np.zeros((0, 0), dtype=np.float64),
        "cell_size":        cell_size,
        "pallet_offset":    pallet_offset,
        "tall_threshold":   tall_threshold,
        "halo_filter_used": False,
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
    # Unvisited cells initialise to -inf so they fall below min_cargo_h.
    hf = np.full((n_rows, n_cols), -np.inf, dtype=np.float64)
    np.maximum.at(hf, (j_idx, i_idx), y)

    # Convert absolute Y to net height above effective floor (floor + pallet).
    effective_floor = floor_y + pallet_offset
    hf -= effective_floor

    # ── Candidate mask: cells above min_cargo_h ───────────────────────────────
    valid = hf >= min_cargo_h

    # ── Connected-component halo filter ──────────────────────────────────────
    # Tall anchor mask: cells clearly above the pallet surface.
    tall = hf >= tall_threshold
    halo_filter_used = False

    if tall.any():
        # Label 4-connected components of the candidate mask.
        struct = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=bool)   # 4-connectivity
        labeled, n_components = nd_label(valid, structure=struct)

        # Keep only components that contain at least one tall cell.
        tall_component_ids = set(int(v) for v in np.unique(labeled[tall]) if v != 0)
        if len(tall_component_ids) < n_components:
            # Some components are halo-only — filter them out.
            keep_mask = np.isin(labeled, list(tall_component_ids))
            valid = valid & keep_mask
            halo_filter_used = True
    # else: flat-cargo fallback — no filter applied; keep all valid cells.

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
        "volume_m3":        volume_m3,
        "footprint_m2":     footprint,
        "mean_height":      mean_height,
        "max_height":       max_height,
        "n_cells":          n_cells,
        "height_field":     hf,
        "cell_size":        cell_size,
        "pallet_offset":    pallet_offset,
        "tall_threshold":   tall_threshold,
        "halo_filter_used": halo_filter_used,
    }
