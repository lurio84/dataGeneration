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
import open3d as o3d
from scipy.ndimage import label as nd_label, distance_transform_edt


def _fill_horizontal_sensor(hf: np.ndarray, tall_threshold: float) -> np.ndarray:
    """Fill occluded interior cells for horizontal-sensor clouds (e.g. FUSION3D).

    For horizontal-viewing sensors the top face of the cargo is not directly
    visible; the height field only has reliable coverage on the near face and
    the two side faces.  Two complementary passes correct this:

    Pass 1 — column-max elevation:
        For each X column (depth slice), the maximum height seen at any Z row
        in that column is the best available estimate of the cargo height at
        that depth.  All cells in the column — whether truly empty (−inf) or
        present-but-low (sub-tall artefacts from partial face visibility) — are
        elevated to that column maximum within the Z span defined by the
        outermost tall rows.

    Pass 2 — nearest-tall propagation for remaining empties:
        After Pass 1, truly empty cells that fall outside every column's tall
        span (e.g. the back face) are filled via distance_transform_edt from
        the nearest tall cell.

    For top-down (synthetic) sensors every cell in the cargo footprint already
    has a correct height, so this function is effectively a no-op: Pass 1 only
    elevates cells that are already at the column max (no change), and Pass 2
    finds no remaining empty cells within any span.

    Parameters
    ----------
    hf : (n_rows, n_cols) float64
        Net height array after subtracting effective floor.  Cells with no
        observed point have value ``−inf``.
    tall_threshold : float
        Minimum net height to qualify as a "tall anchor" cell (same value used
        by the halo filter in height_field_volume).

    Returns
    -------
    hf_out : (n_rows, n_cols) float64
        Copy of ``hf`` with corrected heights.  Tall-observed cells are never
        modified.
    """
    tall_obs = np.isfinite(hf) & (hf >= tall_threshold)
    if not tall_obs.any():
        return hf

    hf_out = hf.copy()

    # ── Pass 1: column-max elevation ─────────────────────────────────────────
    # For each X column, find the Z span of tall cells and the column max height.
    # Elevate all below-max cells within that span to the column max.
    span_mask = np.zeros(hf.shape, dtype=bool)   # cells within some column's tall span
    for col in range(hf.shape[1]):
        col_tall_rows = np.where(tall_obs[:, col])[0]
        if len(col_tall_rows) == 0:
            continue
        col_max_h = float(hf[col_tall_rows, col].max())
        r0, r1 = int(col_tall_rows[0]), int(col_tall_rows[-1])
        span_mask[r0:r1 + 1, col] = True
        for row in range(r0, r1 + 1):
            if hf_out[row, col] < col_max_h:   # includes −inf and sub-max observed
                hf_out[row, col] = col_max_h

    # ── Pass 2: nearest-tall propagation for remaining empty cells ────────────
    # After Pass 1 some cells within span_mask may still be −inf
    # (e.g. columns that gained no tall neighbour from Pass 1 due to span gaps).
    # Fill them from the nearest tall cell in hf_out.
    tall_now = np.isfinite(hf_out) & (hf_out >= tall_threshold)
    remaining_empty = span_mask & ~np.isfinite(hf_out)
    if remaining_empty.any() and tall_now.any():
        _, nn = distance_transform_edt(~tall_now, return_indices=True)
        ri, ci = np.where(remaining_empty)
        hf_out[ri, ci] = hf_out[nn[0][ri, ci], nn[1][ri, ci]]

    return hf_out


def height_field_volume(
    cargo_pts: np.ndarray,
    floor_y: float,
    cell_size: float = 0.04,
    pallet_offset: float = 0.144,
    min_cargo_h: float = 0.05,
    tall_threshold: float = 0.20,
    horizontal_fill: bool = False,
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

    Horizontal-sensor fill (``horizontal_fill=True``)
    --------------------------------------------------
    When set, applies ``_fill_horizontal_sensor`` before the candidate mask step.
    This corrects for horizontal-viewing sensors (e.g. FUSION3D) where the top
    face of the cargo is occluded: interior cells are filled using per-column
    max-height propagation so that each depth slice carries the highest reliably
    observed height for that slice.  Leave as ``False`` (default) for top-down
    synthetic sensors.

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
    horizontal_fill : bool
        When True, apply horizontal-sensor occlusion fill before computing
        the candidate mask.  Use for FUSION3D/horizontal-viewing sensors.
        Default False (top-down / synthetic data).

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

    # ── Horizontal-sensor occlusion fill (opt-in) ────────────────────────────
    # For horizontal-viewing sensors (e.g. FUSION3D) the top face of cargo is
    # occluded; only vertical faces have coverage.  _fill_horizontal_sensor
    # propagates tall-neighbour heights into interior cells using per-column
    # max-height elevation.  Disabled by default for top-down/synthetic data.
    if horizontal_fill:
        hf = _fill_horizontal_sensor(hf, tall_threshold)

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


def obb_volume(cargo_pts: np.ndarray) -> dict:
    """Compute Oriented Bounding Box (OBB) and ConvexHull volumes for cargo points.

    Both methods are standard in logistics cubicaje:

    * **OBB** (Oriented Bounding Box): the tightest box aligned to the principal
      axes of the point cloud.  Suitable for regular, box-shaped cargo where
      the stack has a dominant orientation.  Computed with Open3D's
      ``OrientedBoundingBox.create_from_points()``.

    * **ConvexHull**: smallest convex polyhedron enclosing all points.
      A strict upper bound — includes any concave voids in the cargo.
      Computed with Open3D's ``compute_convex_hull()``.

    Dimensions are returned **sorted descending** (longest → shortest) so they
    are directly comparable with Paula's ``Global size (W, H, D)`` report.

    Parameters
    ----------
    cargo_pts : (N, 3) float array
        Cargo points in world coordinates (X, Y, Z).  Requires N ≥ 4 for
        ConvexHull; N ≥ 1 for OBB (falls back gracefully otherwise).

    Returns
    -------
    dict with keys:
        ``obb_volume_m3``    : float — OBB volume in cubic metres
        ``obb_dims``         : list[float] — OBB edge lengths [L, W, H] sorted
                               descending (metres); all three are the extents of
                               the oriented box along its principal axes.
        ``obb_object``       : open3d.geometry.OrientedBoundingBox | None
        ``convhull_volume_m3``: float — ConvexHull volume in cubic metres
                               (0.0 when fewer than 4 non-coplanar points)
    """
    _empty = {
        "obb_volume_m3":     0.0,
        "obb_dims":          [0.0, 0.0, 0.0],
        "obb_object":        None,
        "convhull_volume_m3": 0.0,
    }

    if len(cargo_pts) < 4:
        return _empty

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cargo_pts.astype(np.float64))

    # ── OBB ──────────────────────────────────────────────────────────────────
    try:
        obb = pcd.get_oriented_bounding_box()
        dims_raw = np.sort(np.abs(obb.extent))[::-1].tolist()   # descending
        obb_vol  = float(np.prod(obb.extent))
    except Exception:
        return _empty

    # ── ConvexHull ────────────────────────────────────────────────────────────
    convhull_vol = 0.0
    try:
        hull_mesh, _ = pcd.compute_convex_hull()
        hull_mesh.orient_triangles()
        convhull_vol = float(hull_mesh.get_volume())
    except Exception:
        pass   # non-manifold / degenerate cloud — keep 0.0

    return {
        "obb_volume_m3":      obb_vol,
        "obb_dims":           dims_raw,
        "obb_object":         obb,
        "convhull_volume_m3": convhull_vol,
    }
