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
    """Return (pts [N,3], labels [N,]) from synthetic PLY (binary, has label property)."""
    dtype = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("label","u1")])
    with open(path, "rb") as f:
        # skip header
        while True:
            line = f.readline()
            if line.strip() == b"end_header":
                break
        data = np.frombuffer(f.read(), dtype=dtype)
    pts = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    lbs = data["label"]
    return pts, lbs


def load_real_ply(path: Path) -> np.ndarray:
    """Return pts [N,3] from real FUSION3D PLY (RGB, no label)."""
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)


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
    real_pts_list = [load_real_ply(p) for p in real_plys]

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1: Scene overview — point count + extent distributions
    # ─────────────────────────────────────────────────────────────────────────
    print("Figure 1: point counts and extents …")

    synth_counts = [len(p) for p in synth_pts_list]
    real_counts  = [len(p) for p in real_pts_list]

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

    # Real: RANSAC floor detection on first real scene (reference)
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

    # Real: normalise coordinates to match synthetic scale
    # Real X and Z span ~5-6m; keep as-is, same range
    real_pts_xz = [p[:, [0, 2]] for p in real_pts_list]
    HR, xeR, zeR = density_map(real_pts_list, (-3.5, 3.5), (-3.5, 3.5))

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
    for pts in synth_pts_list[:5]:
        h = pts[:, 1]   # Y is height in synthetic (floor at Y=0)
        r = local_roughness_grid(pts, h)
        if len(r):
            synth_rough_list.append(r)
    synth_rough = np.concatenate(synth_rough_list) if synth_rough_list else np.array([0.0])

    print("  Computing roughness for real (sample 4 scenes) …")
    real_rough_list = []
    for pts in real_pts_list[:4]:
        h = floor_relative_height(pts)
        r = local_roughness_grid(pts, h)
        if len(r):
            real_rough_list.append(r)
    real_rough = np.concatenate(real_rough_list) if real_rough_list else np.array([0.015])

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

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n─── Summary ─────────────────────────────────────────────────────────")
    print(f"Synthetic  │ scenes={len(synth_plys)}  pts/scene: {np.mean(synth_counts):.0f} ± {np.std(synth_counts):.0f}")
    print(f"Real       │ scenes={len(real_plys)}   pts/scene: {np.mean(real_counts):.0f} ± {np.std(real_counts):.0f}")
    print(f"\nRoughness σ  Synthetic: {np.median(synth_rough)*1000:.1f}mm  │  Real: {np.median(real_rough)*1000:.1f}mm")
    print(f"NN spacing   Synthetic: {np.median(synth_nn)*1000:.1f}mm     │  Real: {np.median(real_nn)*1000:.1f}mm")
    print(f"\nSaved figures → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
