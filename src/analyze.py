"""
analyze.py  —  Synthetic vs Real point cloud comparison
Compares generated dataset against FUSION3D merged captures.

Run from src/  with paths as constants below.
Saves figures to ../output/analysis/
"""

import os
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────────
SYNTH_DIR   = Path("../output/dataset")
REAL_DIR    = Path("../../Resources/Capturas_BBB/2026_03_23")
OUT_DIR     = Path("../output/analysis")
N_SYNTH_MAX = 50    # synthetic scenes to load (balance speed vs stats)

LABEL_NAMES = {0: "floor", 1: "cargo", 2: "vehicle", 3: "person", 4: "pallet", 255: "outlier"}
LABEL_COLORS = {0: "#8B7355", 1: "#E67E22", 2: "#2980B9", 3: "#27AE60", 4: "#F39C12", 255: "#95A5A6"}

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_synth_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (pts [N,3], labels [N,]) from synthetic PLY.
    Handles both old format (x y z label) and new format (x y z red green blue label).
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line.decode("ascii", errors="ignore").strip())
            if header_lines[-1] == "end_header":
                break
        raw = f.read()
    has_rgb = any("red" in l for l in header_lines)
    if has_rgb:
        dtype = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),
                          ("red","u1"),("green","u1"),("blue","u1"),("label","u1")])
    else:
        dtype = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("label","u1")])
    data = np.frombuffer(raw, dtype=dtype)
    pts = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    lbs = data["label"]
    return pts, lbs


def load_real_ply(path: Path) -> np.ndarray:
    """Return pts [N,3] from real FUSION3D PLY (RGB, no label)."""
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)


def detect_floor_axis(pts: np.ndarray) -> int:
    """
    Detect which axis (0=X, 1=Y, 2=Z) is the vertical/height axis.
    The floor axis has the highest point count near one value (histogram peak).
    Returns the axis index.
    In real FUSION3D: floor is at Z≈0 (axis 2).
    In synthetic: floor is at Y=0 (axis 1).
    """
    best_axis, best_peak = 0, 0
    for ax in range(3):
        vals = pts[:, ax]
        hist, _ = np.histogram(vals, bins=100)
        if hist.max() > best_peak:
            best_peak = hist.max()
            best_axis = ax
    return best_axis


def align_real_to_synthetic(pts: np.ndarray) -> np.ndarray:
    """
    Reorder real FUSION3D axes so that the floor axis becomes Y (synthetic convention).
    Real FUSION3D: floor axis = Z (index 2) → swap Y and Z.
    Returns pts with axes (X, height, depth) = (X, Y, Z) in synthetic convention.
    """
    floor_ax = detect_floor_axis(pts)
    if floor_ax == 1:
        return pts  # already Y-up
    # Reorder: make floor_ax → Y (index 1)
    # For real FUSION3D (floor_ax=2): new order = [0, 2, 1] (X, Z, Y)
    other_axes = [i for i in range(3) if i != floor_ax]
    return pts[:, [other_axes[0], floor_ax, other_axes[1]]].copy()


def detect_floor_height(pts: np.ndarray, axis: int = 1) -> float:
    """
    Estimate floor coordinate along `axis` using the mode of the histogram
    in the lowest 30% of the point cloud.  Returns the floor value.
    """
    vals = pts[:, axis]
    low = np.percentile(vals, 30)
    hist, edges = np.histogram(vals[vals < low], bins=200)
    return float(edges[np.argmax(hist)])


def floor_relative_height(pts: np.ndarray) -> np.ndarray:
    """
    Detect floor plane with RANSAC and return per-point height above floor.
    Works regardless of which world axis is 'up'.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.025, ransac_n=3, num_iterations=500
    )
    a, b, c, d = plane_model   # ax+by+cz+d=0,  normal=(a,b,c)
    normal = np.array([a, b, c])
    # height = signed distance, positive = above floor
    height = (pts @ normal + d) / np.linalg.norm(normal)
    # ensure "above floor" is positive
    if height.mean() < 0:
        height = -height
    return height.astype(np.float32)


# ── Statistics helpers ─────────────────────────────────────────────────────────

def local_roughness_grid(
    pts: np.ndarray,
    height: np.ndarray,
    floor_band_m: float = 0.06,
    cell_m: float = 0.05,
) -> np.ndarray:
    """
    Compute per-cell std-dev of height within a 2D (XZ) grid.
    Only uses points within floor_band_m of the floor (height ≈ 0)
    to avoid mixing floor and cargo in the same cell.
    Returns array of std values for cells with ≥8 points.
    """
    floor_mask = np.abs(height) < floor_band_m
    pts_f, h_f = pts[floor_mask], height[floor_mask]
    if len(pts_f) < 50:
        return np.array([], dtype=np.float32)

    xi = np.floor(pts_f[:, 0] / cell_m).astype(np.int32)
    zi = np.floor(pts_f[:, 2] / cell_m).astype(np.int32)
    keys = xi.astype(np.int64) * 1_000_000 + zi.astype(np.int64)
    _, inv = np.unique(keys, return_inverse=True)
    stds = []
    for k_idx in range(int(inv.max()) + 1):
        mask = inv == k_idx
        if mask.sum() >= 8:
            stds.append(h_f[mask].std())
    return np.array(stds, dtype=np.float32)


def nn_distances(pts: np.ndarray, sample: int = 5000) -> np.ndarray:
    """Mean nearest-neighbour distance sampled from the cloud (O(N log N))."""
    idx = np.random.default_rng(0).choice(len(pts), min(sample, len(pts)), replace=False)
    sub = pts[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    return dists.astype(np.float32)


def crop_to_roi(
    pts: np.ndarray,
    x_range: tuple[float, float] = (-2.5, 2.5),
    y_range: tuple[float, float] = (-0.15, 2.5),
    z_range: tuple[float, float] = (-2.0, 2.0),
) -> np.ndarray:
    """
    Restrict real FUSION3D cloud to the same spatial region as the synthetic scenes.
    Excludes walls, ceiling, far background, and sub-floor noise.
    Coordinate convention after align_real_to_synthetic: Y = height.
    Bounds chosen from the observed real-data envelope:
      X ≈ ±2.8m, Z ≈ ±2.0m, height 0–2.5m.
    Using ±2.5 / ±2.0 to stay within reliable stereo range.
    """
    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
    )
    return pts[mask]


def _roughness_band(
    pts: np.ndarray,
    height: np.ndarray,
    y_lo: float = 0.20,
    y_hi: float = 1.60,
    cell_m: float = 0.05,
) -> np.ndarray:
    """
    Compute per-cell std-dev of height for points within a specific height band
    (y_lo ≤ height < y_hi).  Used to compare roughness on cargo surfaces
    independently from the floor band.
    Returns array of std values for cells with ≥8 points.
    """
    mask = (height >= y_lo) & (height < y_hi)
    pts_b, h_b = pts[mask], height[mask]
    if len(pts_b) < 50:
        return np.array([], dtype=np.float32)
    xi = np.floor(pts_b[:, 0] / cell_m).astype(np.int32)
    zi = np.floor(pts_b[:, 2] / cell_m).astype(np.int32)
    keys = xi.astype(np.int64) * 1_000_000 + zi.astype(np.int64)
    _, inv = np.unique(keys, return_inverse=True)
    stds = []
    for k_idx in range(int(inv.max()) + 1):
        mask2 = inv == k_idx
        if mask2.sum() >= 8:
            stds.append(h_b[mask2].std())
    return np.array(stds, dtype=np.float32)


# ── Main analysis ──────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load synthetic ──
    synth_plys = sorted(SYNTH_DIR.glob("*.ply"))[:N_SYNTH_MAX]
    if not synth_plys:
        raise RuntimeError(f"No synthetic PLYs found in {SYNTH_DIR}. Run generate_dataset.py first.")
    print(f"Loading {len(synth_plys)} synthetic scenes …")

    synth_pts_list, synth_lbs_list = [], []
    for p in synth_plys:
        pts, lbs = load_synth_ply(p)
        synth_pts_list.append(pts)
        synth_lbs_list.append(lbs)

    # ── Load real ──
    real_plys = sorted(REAL_DIR.glob("*/FUSION3D/*.ply"))
    if not real_plys:
        raise RuntimeError(f"No real PLYs found in {REAL_DIR}.")
    print(f"Loading {len(real_plys)} real scenes …")
    # Align real data to synthetic convention: height → Y axis
    real_pts_list = [align_real_to_synthetic(load_real_ply(p)) for p in real_plys]
    real_floor_ax = detect_floor_axis(load_real_ply(real_plys[0]))
    print(f"  Real FUSION3D floor axis: {'XYZ'[real_floor_ax]} → remapped to Y")

    # ── Crop real clouds to ROI matching synthetic extent (fair comparison) ──
    real_counts_full = [len(p) for p in real_pts_list]               # full count before crop
    real_pts_full    = real_pts_list                                  # keep full for density maps
    real_pts_list    = [crop_to_roi(p) for p in real_pts_list]
    real_counts_roi  = [len(p) for p in real_pts_list]
    print(f"  After ROI crop: {np.mean(real_counts_roi):.0f} ± {np.std(real_counts_roi):.0f} pts/scene"
          f"  (was {np.mean(real_counts_full):.0f} full)")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1: Scene overview — point count + extent distributions
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 1: point counts and extents …")

    synth_counts = [len(p) for p in synth_pts_list]
    real_counts  = real_counts_roi   # use ROI counts throughout for fair comparison

    def bbox_span(pts_list):
        return np.array([(p.max(0) - p.min(0)) for p in pts_list])   # (N,3)

    synth_spans = bbox_span(synth_pts_list)
    real_spans  = bbox_span(real_pts_list)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Scene statistics: Synthetic vs Real FUSION3D", fontweight="bold")

    axes[0].hist(synth_counts, bins=20, alpha=0.7, color="#E67E22", label="Synthetic")
    axes[0].axvline(np.mean(real_counts), color="#2980B9", lw=2, label=f"Real mean ({int(np.mean(real_counts)):,})")
    axes[0].set_xlabel("Point count"); axes[0].set_ylabel("Frequency"); axes[0].legend(); axes[0].set_title("Points per scene")

    for ax_i, (axis_label, sym_i) in enumerate(zip(["X","Y","Z"], [0,1,2]), start=1):
        axes[ax_i].hist(synth_spans[:,sym_i], bins=10, alpha=0.7, color="#E67E22", label="Synthetic")
        axes[ax_i].axvline(real_spans[:,sym_i].mean(), color="#2980B9", lw=2,
                           label=f"Real mean ({real_spans[:,sym_i].mean():.2f}m)")
        axes[ax_i].set_xlabel(f"{axis_label} span (m)"); axes[ax_i].set_title(f"Bounding box {axis_label} span")
        axes[ax_i].legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_scene_stats.png", dpi=150)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2: Height distribution (floor-relative)
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 2: height profiles …")

    # Synthetic: Y is height, floor at Y=0
    synth_heights = np.concatenate([pts[:, 1] for pts in synth_pts_list])
    synth_heights = synth_heights[synth_heights < 2.5]   # clip ceiling artefacts

    # Real: after alignment Y is height — RANSAC refines floor offset
    print("  RANSAC floor detection on real data …")
    real_h_all = []
    for rpts in real_pts_list[:4]:  # first 4 to save time
        h = floor_relative_height(rpts)
        real_h_all.append(h)
    real_heights = np.concatenate(real_h_all)
    real_heights = real_heights[(real_heights > -0.3) & (real_heights < 3.0)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Height distribution (floor-relative)", fontweight="bold")

    bins = np.linspace(-0.3, 2.5, 80)
    axes[0].hist(synth_heights, bins=bins, color="#E67E22", alpha=0.8, density=True)
    axes[0].set_xlabel("Height above floor (m)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Synthetic"); axes[0].axvline(0, color="k", lw=1, ls="--")

    axes[1].hist(real_heights, bins=bins, color="#2980B9", alpha=0.8, density=True)
    axes[1].set_xlabel("Height above floor (m)")
    axes[1].set_title("Real FUSION3D"); axes[1].axvline(0, color="k", lw=1, ls="--")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_height_dist.png", dpi=150)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3: Top-down (XZ) density heatmaps
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 3: XZ density heatmaps …")

    def density_map(pts_list, xrange, zrange, bins=60):
        all_pts = np.concatenate(pts_list)
        x, z = all_pts[:, 0], all_pts[:, 2]
        H, xe, ze = np.histogram2d(x, z, bins=bins,
                                    range=[xrange, zrange], density=True)
        return H.T, xe, ze

    xr = (-3.5, 3.5); zr = (-3.5, 3.5)
    HS, xeS, zeS = density_map(synth_pts_list, xr, zr)

    # Real: use full cloud (pre-ROI) so the density map shows the complete scene
    HR, xeR, zeR = density_map(real_pts_full, (-3.5, 3.5), (-3.5, 3.5))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Top-down (XZ) point density", fontweight="bold")
    for ax, H, xe, ze, title in zip(axes, [HS, HR], [xeS, xeR], [zeS, zeR],
                                     ["Synthetic", "Real FUSION3D"]):
        im = ax.pcolormesh(xe, ze, H, cmap="hot_r")
        plt.colorbar(im, ax=ax, label="density")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
        ax.set_title(title); ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_density_xz.png", dpi=150)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 4: Local roughness (noise σ)
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 4: local roughness / noise σ …")

    print("  Computing roughness for synthetic (sample 5 scenes) …")
    synth_rough_list = []
    synth_rough_cargo_list = []
    for pts in synth_pts_list[:5]:
        h = pts[:, 1]   # Y is height in synthetic (floor at Y=0)
        r_floor = local_roughness_grid(pts, h, floor_band_m=0.06)
        r_cargo = local_roughness_grid(pts, h, floor_band_m=0.06,
                                       cell_m=0.05) if False else \
                  _roughness_band(pts, h, y_lo=0.20, y_hi=1.60)
        if len(r_floor): synth_rough_list.append(r_floor)
        if len(r_cargo): synth_rough_cargo_list.append(r_cargo)
    synth_rough       = np.concatenate(synth_rough_list)       if synth_rough_list       else np.array([0.0])
    synth_rough_cargo = np.concatenate(synth_rough_cargo_list) if synth_rough_cargo_list else np.array([0.0])

    print("  Computing roughness for real (sample 4 scenes, ROI-cropped) …")
    real_rough_list = []
    real_rough_cargo_list = []
    for pts in real_pts_list[:4]:
        h = floor_relative_height(pts)
        r_floor = local_roughness_grid(pts, h, floor_band_m=0.06)
        r_cargo = _roughness_band(pts, h, y_lo=0.20, y_hi=1.60)
        if len(r_floor): real_rough_list.append(r_floor)
        if len(r_cargo): real_rough_cargo_list.append(r_cargo)
    real_rough       = np.concatenate(real_rough_list)       if real_rough_list       else np.array([0.015])
    real_rough_cargo = np.concatenate(real_rough_cargo_list) if real_rough_cargo_list else np.array([0.015])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Local surface roughness σ  (5cm grid cells, proxy for sensor noise)", fontweight="bold")
    bins_r = np.linspace(0, 0.08, 60)
    for ax, data, color, title in zip(axes,
                                       [synth_rough, real_rough],
                                       ["#E67E22", "#2980B9"],
                                       ["Synthetic", "Real FUSION3D"]):
        ax.hist(data * 1000, bins=bins_r * 1000, color=color, alpha=0.8, density=True)
        ax.axvline(np.median(data) * 1000, color="k", lw=2,
                   label=f"median {np.median(data)*1000:.1f} mm")
        ax.set_xlabel("σ (mm)"); ax.set_ylabel("Density")
        ax.set_title(title); ax.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_roughness.png", dpi=150)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 5: Nearest-neighbour distance (point spacing)
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 5: nearest-neighbour distances …")

    synth_nn = np.concatenate([nn_distances(p) for p in synth_pts_list[:5]])
    real_nn  = np.concatenate([nn_distances(p) for p in real_pts_list[:4]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Nearest-neighbour point spacing (random 5k sample/scene)", fontweight="bold")
    bins_nn = np.linspace(0, 0.12, 60)
    for ax, data, color, title in zip(axes,
                                       [synth_nn, real_nn],
                                       ["#E67E22", "#2980B9"],
                                       ["Synthetic", "Real FUSION3D"]):
        ax.hist(data * 1000, bins=bins_nn * 1000, color=color, alpha=0.8, density=True)
        ax.axvline(np.median(data) * 1000, color="k", lw=2,
                   label=f"median {np.median(data)*1000:.1f} mm")
        ax.set_xlabel("NN distance (mm)"); ax.set_ylabel("Density")
        ax.set_title(title); ax.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_nn_spacing.png", dpi=150)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 6: Synthetic label distribution (pie chart)
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 6: label distribution …")

    all_lbs = np.concatenate(synth_lbs_list)
    unique_lbs, counts = np.unique(all_lbs, return_counts=True)
    labels_str = [LABEL_NAMES.get(int(l), str(l)) for l in unique_lbs]
    colors = [LABEL_COLORS.get(int(l), "#AAAAAA") for l in unique_lbs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Synthetic label distribution  ({len(synth_plys)} scenes)", fontweight="bold")

    axes[0].pie(counts, labels=labels_str, colors=colors, autopct="%1.1f%%", startangle=140)
    axes[0].set_title("All points (incl. floor)")

    # Without floor for clearer cargo view
    no_floor_mask = all_lbs != 0
    ul2, c2 = np.unique(all_lbs[no_floor_mask], return_counts=True)
    l2 = [LABEL_NAMES.get(int(l), str(l)) for l in ul2]
    c2_colors = [LABEL_COLORS.get(int(l), "#AAAAAA") for l in ul2]
    axes[1].pie(c2, labels=l2, colors=c2_colors, autopct="%1.1f%%", startangle=140)
    axes[1].set_title("Without floor")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_label_dist.png", dpi=150)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 7: Overlay comparison — NN spacing + roughness on same axes
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 7: overlay comparison (synthetic vs real) …")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Synthetic vs Real FUSION3D — direct overlay", fontweight="bold")

    # NN spacing overlay
    bins_nn = np.linspace(0, 0.14, 70)
    axes[0].hist(synth_nn * 1000, bins=bins_nn * 1000, color="#E67E22", alpha=0.6,
                 density=True, label=f"Synthetic  median={np.median(synth_nn)*1000:.1f}mm")
    axes[0].hist(real_nn * 1000,  bins=bins_nn * 1000, color="#2980B9", alpha=0.6,
                 density=True, label=f"Real       median={np.median(real_nn)*1000:.1f}mm")
    axes[0].axvline(np.median(synth_nn) * 1000, color="#E67E22", lw=2, ls="--")
    axes[0].axvline(np.median(real_nn)  * 1000, color="#2980B9", lw=2, ls="--")
    axes[0].set_xlabel("NN distance (mm)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Nearest-neighbour spacing"); axes[0].legend()

    # Roughness overlay
    bins_r = np.linspace(0, 0.08, 60)
    axes[1].hist(synth_rough * 1000, bins=bins_r * 1000, color="#E67E22", alpha=0.6,
                 density=True, label=f"Synthetic  median={np.median(synth_rough)*1000:.1f}mm")
    axes[1].hist(real_rough  * 1000, bins=bins_r * 1000, color="#2980B9", alpha=0.6,
                 density=True, label=f"Real       median={np.median(real_rough)*1000:.1f}mm")
    axes[1].axvline(np.median(synth_rough) * 1000, color="#E67E22", lw=2, ls="--")
    axes[1].axvline(np.median(real_rough)  * 1000, color="#2980B9", lw=2, ls="--")
    axes[1].set_xlabel("σ (mm)"); axes[1].set_ylabel("Density")
    axes[1].set_title("Local roughness σ (5cm grid)"); axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_overlay.png", dpi=150)
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────────

    # Footprint XZ extents
    synth_all_pts = np.concatenate(synth_pts_list)
    real_all_pts  = np.concatenate(real_pts_list)

    def xz_extent(pts):
        return (pts[:, 0].min(), pts[:, 0].max(),
                pts[:, 2].min(), pts[:, 2].max())

    sxmin, sxmax, szmin, szmax = xz_extent(synth_all_pts)
    rxmin, rxmax, rzmin, rzmax = xz_extent(real_all_pts)

    # Label percentages
    total_pts = all_lbs.size
    label_pct = {
        LABEL_NAMES.get(int(l), str(l)): 100.0 * c / total_pts
        for l, c in zip(unique_lbs, counts)
    }

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                     ANALYSIS SUMMARY                            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ 1. POINT COUNT                                                   ║")
    print(f"║    Synthetic  {np.mean(synth_counts):>8.0f} ± {np.std(synth_counts):<6.0f} pts/scene ({len(synth_plys)} scenes)   ║")
    print(f"║    Real full  {np.mean(real_counts_full):>8.0f} ± {np.std(real_counts_full):<6.0f} pts/scene ({len(real_plys)} scenes)    ║")
    print(f"║    Real ROI   {np.mean(real_counts_roi):>8.0f} ± {np.std(real_counts_roi):<6.0f} pts/scene (cropped)           ║")
    ratio = np.mean(real_counts_full) / np.mean(synth_counts)
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ 2. NN SPACING (point density)                                    ║")
    print(f"║    Synthetic  median={np.median(synth_nn)*1000:>5.1f}mm  mean={np.mean(synth_nn)*1000:>5.1f}mm            ║")
    print(f"║    Real       median={np.median(real_nn)*1000:>5.1f}mm  mean={np.mean(real_nn)*1000:>5.1f}mm            ║")
    nn_ratio = np.median(real_nn) / np.median(synth_nn)
    voxel_cur_local = 0.019  # keep in sync with CFG
    if nn_ratio > 1.0:
        voxel_recommended = voxel_cur_local * nn_ratio
        print(f"║    Synth {1/nn_ratio:.2f}x denser than real → could ↑ voxel_size          ║")
    else:
        voxel_recommended = voxel_cur_local
        print(f"║    Synth sparser than real → voxel_size OK or ↓ slightly         ║")
    print(f"║    Suggested voxel_size ≈ {voxel_recommended*1000:.0f}mm  (currently {voxel_cur_local*1000:.0f}mm)        ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ 3. LOCAL ROUGHNESS σ  (ROI-cropped real, 5cm grid cells)         ║")
    print(f"║    Floor band (|h|<6cm)                                          ║")
    print(f"║      Synthetic  median={np.median(synth_rough)*1000:>5.1f}mm                           ║")
    print(f"║      Real       median={np.median(real_rough)*1000:>5.1f}mm  (target ≈ 29mm)          ║")
    rough_gap = np.median(real_rough)*1000 - np.median(synth_rough)*1000
    print(f"║      Gap: {rough_gap:+.1f}mm                                             ║")
    s_cargo_med = np.median(synth_rough_cargo)*1000 if len(synth_rough_cargo) else float("nan")
    r_cargo_med = np.median(real_rough_cargo)*1000  if len(real_rough_cargo)  else float("nan")
    print(f"║    Cargo band (20cm–160cm)                                       ║")
    print(f"║      Synthetic  median={s_cargo_med:>5.1f}mm                           ║")
    print(f"║      Real       median={r_cargo_med:>5.1f}mm                           ║")
    cargo_gap = r_cargo_med - s_cargo_med
    print(f"║      Gap: {cargo_gap:+.1f}mm  (surfaces include box tops/sides)         ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ 4. FOOTPRINT XZ  (real = ROI-cropped, synth = full scene)       ║")
    print(f"║    Synthetic  X:[{sxmin:+.2f}, {sxmax:+.2f}]m  Z:[{szmin:+.2f}, {szmax:+.2f}]m      ║")
    print(f"║    Real (ROI) X:[{rxmin:+.2f}, {rxmax:+.2f}]m  Z:[{rzmin:+.2f}, {rzmax:+.2f}]m      ║")
    print(f"║    Synth X-span={sxmax-sxmin:.2f}m  Real X-span={rxmax-rxmin:.2f}m                   ║")
    print(f"║    Synth Z-span={szmax-szmin:.2f}m  Real Z-span={rzmax-rzmin:.2f}m                   ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ 5. LABEL DISTRIBUTION (synthetic)                                ║")
    for name, pct in sorted(label_pct.items(), key=lambda x: -x[1]):
        print(f"║    {name:<10s}  {pct:>5.1f}%                                          ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    noise_std_cur = 0.030  # keep in sync with generate_dataset.py CFG
    voxel_cur     = 0.019
    print(f"║ PARAMETER STATUS  (noise={noise_std_cur*1000:.0f}mm  voxel={voxel_cur*1000:.0f}mm)              ║")
    if nn_ratio > 1.5:
        print(f"║  ⚠ NN spacing: real {nn_ratio:.1f}x sparser → increase voxel_size        ║")
    else:
        print(f"║  ✓ NN spacing: within 1.5x  ({np.median(synth_nn)*1000:.1f}mm vs {np.median(real_nn)*1000:.1f}mm)        ║")
    if abs(rough_gap) < 5:
        print(f"║  ✓ Floor roughness gap < 5mm  ({rough_gap:+.1f}mm)                    ║")
    elif rough_gap > 0:
        print(f"║  △ Floor roughness: synth {rough_gap:.1f}mm short → ↑ noise_std        ║")
    else:
        print(f"║  △ Floor roughness: synth {-rough_gap:.1f}mm over → ↓ noise_std         ║")
    if abs(cargo_gap) < 5:
        print(f"║  ✓ Cargo roughness gap < 5mm  ({cargo_gap:+.1f}mm)                    ║")
    elif cargo_gap > 0:
        print(f"║  △ Cargo roughness: synth {cargo_gap:.1f}mm short → ↑ noise_std        ║")
    else:
        print(f"║  △ Cargo roughness: synth {-cargo_gap:.1f}mm over → ↓ noise_std         ║")
    ratio_roi = np.mean(real_counts_roi) / np.mean(synth_counts)
    print(f"║  ℹ Point count ratio (ROI): {ratio_roi:.1f}x  "
          f"({np.mean(synth_counts):.0f} vs {np.mean(real_counts_roi):.0f} pts)  ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print(f"\nSaved figures → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
