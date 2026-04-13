"""
view_png.py  —  Generate 3-view PNG (top / front / side) of synthetic scenes.

Usage (from src/):
    python3 view_png.py 5           # scene 00005, save PNG + show
    python3 view_png.py 5 --save    # save to output/previews/ only (no display)
    python3 view_png.py 5 --compare # side-by-side with real FUSION3D scene 1

Open synthetic PLYs in CloudCompare:
    File → Open → select output/dataset/00005.ply
    The 'label' scalar field appears in the DB tree.
    Select cloud → Edit → Colors → Height Ramp (or scalar field ramp).
"""

import os, sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import open3d as o3d

# Use non-interactive backend when --save, interactive otherwise
args_raw = sys.argv[1:]
SAVE_ONLY = "--save" in args_raw
COMPARE   = "--compare" in args_raw
if SAVE_ONLY:
    matplotlib.use("Agg")

SYNTH_DIR   = Path("../output/dataset")
REAL_DIR    = Path("../../Resources/Capturas_BBB/2026_03_23")
PREVIEW_DIR = Path("../output/previews")

LABEL_COLORS = {
    0:   "#8B7355",   # floor   — marrón
    1:   "#E67E22",   # cargo   — naranja
    2:   "#2980B9",   # vehicle — azul
    3:   "#27AE60",   # person  — verde
    4:   "#F39C12",   # pallet  — amarillo
    255: "#95A5A6",   # outlier — gris
}
LABEL_NAMES = {0:"floor", 1:"cargo", 2:"vehicle", 3:"person", 4:"pallet", 255:"outlier"}

MAX_PTS = 30_000   # max points to render (subsample for speed)


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_synth(path: Path):
    # Support both old (x y z label) and new (x y z r g b label) formats
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
    lbs = data["label"].astype(np.int32)
    return pts, lbs


def load_real_aligned(path: Path):
    """Load real PLY and align floor to Y=0 via RANSAC."""
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)

    # RANSAC floor
    plane, _ = pcd.segment_plane(distance_threshold=0.025, ransac_n=3, num_iterations=500)
    a, b, c, d = plane
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Height above floor (ensure positive = above)
    height = (pts.astype(np.float64) @ normal + d).astype(np.float32)
    if height.mean() < 0:
        normal = -normal; height = -height

    # Rodrigues rotation: floor normal → Y axis
    y_axis = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, y_axis)
    s = float(np.linalg.norm(v))
    c_val = float(np.dot(normal, y_axis))
    if s > 1e-6:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c_val) / (s * s)
    else:
        R = np.eye(3) * (1 if c_val > 0 else -1)

    pts_rot = (pts.astype(np.float64) @ R.T).astype(np.float32)
    pts_rot[:, 1] -= np.percentile(pts_rot[:, 1], 2)   # floor at Y≈0
    pts_rot[:, 0] -= pts_rot[:, 0].mean()               # centre XZ
    pts_rot[:, 2] -= pts_rot[:, 2].mean()

    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    return pts_rot, colors


def subsample(pts, lbs_or_colors, n=MAX_PTS):
    if len(pts) <= n:
        return pts, lbs_or_colors
    idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
    return pts[idx], lbs_or_colors[idx]


# ── 3-view plot ────────────────────────────────────────────────────────────────

def plot_3views(ax_top, ax_front, ax_side, pts, colors_rgba, title=""):
    """
    Fill 3 axes with scatter plots.
    colors_rgba: (N,4) RGBA float array.
    """
    s = 0.5   # point size
    ax_top.scatter(pts[:,0],  pts[:,2],  c=colors_rgba, s=s, linewidths=0)
    ax_front.scatter(pts[:,0], pts[:,1], c=colors_rgba, s=s, linewidths=0)
    ax_side.scatter(pts[:,2],  pts[:,1], c=colors_rgba, s=s, linewidths=0)

    ax_top.set_xlabel("X (m)");  ax_top.set_ylabel("Z (m)")
    ax_top.set_title(f"{title}\nTop (XZ)")
    ax_front.set_xlabel("X (m)"); ax_front.set_ylabel("Y (m)")
    ax_front.set_title("Front (XY)")
    ax_side.set_xlabel("Z (m)");  ax_side.set_ylabel("Y (m)")
    ax_side.set_title("Side (ZY)")

    for ax in [ax_top, ax_front, ax_side]:
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.5)


def label_colors_rgba(lbs, alpha=0.7):
    import matplotlib.colors as mc
    out = np.zeros((len(lbs), 4), dtype=np.float32)
    for lv, hex_color in LABEL_COLORS.items():
        mask = lbs == lv
        rgba = mc.to_rgba(hex_color, alpha=alpha)
        out[mask] = rgba
    # Unrecognised labels → magenta
    known = np.isin(lbs, list(LABEL_COLORS.keys()))
    out[~known] = [1, 0, 1, alpha]
    return out


def real_colors_rgba(colors_uint8, alpha=0.7):
    rgba = np.ones((len(colors_uint8), 4), dtype=np.float32)
    rgba[:, :3] = colors_uint8.astype(np.float32) / 255.0
    rgba[:, 3] = alpha
    return rgba


def legend_patches():
    import matplotlib.colors as mc
    return [
        mpatches.Patch(color=mc.to_rgba(c), label=LABEL_NAMES[lv])
        for lv, c in LABEL_COLORS.items()
        if lv != 255
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    raw = [a for a in args_raw if not a.startswith("--")]
    idx = int(raw[0]) if raw else 0

    ply_s = SYNTH_DIR / f"{idx:05d}.ply"
    if not ply_s.exists():
        print(f"Not found: {ply_s}"); sys.exit(1)

    pts_s, lbs_s = load_synth(ply_s)
    pts_s, lbs_s = subsample(pts_s, lbs_s)
    rgba_s = label_colors_rgba(lbs_s)

    if COMPARE:
        real_plys = sorted(REAL_DIR.glob("*/FUSION3D/*.ply"))
        if not real_plys:
            print(f"No real PLYs in {REAL_DIR}"); sys.exit(1)
        ply_r = real_plys[idx % len(real_plys)]
        pts_r, c_r = load_real_aligned(ply_r)
        if c_r is None:
            c_r = np.full((len(pts_r), 3), 180, dtype=np.uint8)
        pts_r, c_r = subsample(pts_r, c_r)
        rgba_r = real_colors_rgba(c_r)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"Synthetic #{idx:05d}  (top)   vs   Real FUSION3D scene {ply_r.parent.parent.name}  (bottom)",
            fontsize=12, fontweight="bold"
        )
        plot_3views(*axes[0], pts_s, rgba_s, title="Synthetic")
        plot_3views(*axes[1], pts_r, rgba_r, title="Real FUSION3D")
        fig.legend(handles=legend_patches(), loc="lower center", ncol=5,
                   title="Synthetic labels", fontsize=9)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Synthetic scene {idx:05d}", fontsize=12, fontweight="bold")
        plot_3views(*axes, pts_s, rgba_s)
        fig.legend(handles=legend_patches(), loc="lower center", ncol=6,
                   title="Labels", fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_compare" if COMPARE else ""
    out_path = PREVIEW_DIR / f"{idx:05d}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    if not SAVE_ONLY:
        plt.show()


if __name__ == "__main__":
    main()
