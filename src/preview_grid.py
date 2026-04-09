"""
preview_grid.py  —  One PNG per PLY scene (3 views: top / front / side).

Usage (from src/):
    python3 preview_grid.py          # all scenes in dataset
    python3 preview_grid.py --n 20   # first 20 scenes
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc

SYNTH_DIR   = Path("../output/dataset")
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

COLS = 2          # scenes per row (in batch mode; 5 in full-grid mode)
MAX_PTS = 10_000  # subsample per scene for speed


def load_synth(path: Path):
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


def subsample(pts, lbs, n=MAX_PTS, seed=0):
    if len(pts) <= n:
        return pts, lbs
    idx = np.random.default_rng(seed).choice(len(pts), n, replace=False)
    return pts[idx], lbs[idx]


def label_rgba(lbs, alpha=0.8):
    out = np.zeros((len(lbs), 4), dtype=np.float32)
    for lv, hex_c in LABEL_COLORS.items():
        mask = lbs == lv
        out[mask] = mc.to_rgba(hex_c, alpha=alpha)
    unknown = ~np.isin(lbs, list(LABEL_COLORS.keys()))
    out[unknown] = [1, 0, 1, alpha]
    return out


def scene_title(path: Path, lbs: np.ndarray) -> str:
    present = []
    lv_map = {1:"cargo", 2:"veh", 3:"pers", 4:"palet"}
    for lv, name in lv_map.items():
        if np.any(lbs == lv):
            present.append(name)
    return f"{path.stem}  [{', '.join(present)}]"


def draw_scene(ax, pts, lbs, view="top", point_size=2.0):
    """Draw one thumbnail on ax. Floor rendered faint; objects at full opacity."""
    floor_mask = lbs == 0
    obj_mask   = ~floor_mask

    def coords(p, v):
        if v == "top":   return p[:,0], p[:,2], "X", "Z"
        if v == "front": return p[:,0], p[:,1], "X", "Y"
        return p[:,2], p[:,1], "Z", "Y"

    xc, yc, xl, yl = coords(pts, view)

    # Floor: small, low alpha — present but not dominant
    if floor_mask.any():
        ax.scatter(xc[floor_mask], yc[floor_mask],
                   c=[LABEL_COLORS[0]], s=point_size * 0.4, linewidths=0, alpha=0.25)

    # Objects: full size and opacity
    if obj_mask.any():
        rgba_obj = label_rgba(lbs[obj_mask], alpha=0.9)
        ax.scatter(xc[obj_mask], yc[obj_mask],
                   c=rgba_obj, s=point_size, linewidths=0)

    ax.set_xlabel(xl, fontsize=5, labelpad=1)
    ax.set_ylabel(yl, fontsize=5, labelpad=1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=4, pad=1)
    ax.grid(True, lw=0.2, alpha=0.4)


def legend_patches():
    return [
        mpatches.Patch(color=mc.to_rgba(c), label=f"{LABEL_NAMES[lv]}")
        for lv, c in LABEL_COLORS.items()
    ]


def render_scene(ply: Path, out_path: Path) -> None:
    """Render one PLY as a 3-view PNG (top / front / side)."""
    pts, lbs = load_synth(ply)
    pts, lbs = subsample(pts, lbs)
    title = scene_title(ply, lbs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, view in zip(axes, ["top", "front", "side"]):
        draw_scene(ax, pts, lbs, view=view, point_size=2.0)
        ax.set_title(view, fontsize=9)

    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.legend(handles=legend_patches(), loc="lower center", ncol=7,
               fontsize=8, title="Labels", title_fontsize=9,
               framealpha=0.9, markerscale=2)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = sys.argv[1:]
    n_scenes = None   # None = all

    for i, a in enumerate(args):
        if a == "--n" and i+1 < len(args):
            n_scenes = int(args[i+1])

    all_plys = sorted(SYNTH_DIR.glob("*.ply"))
    if not all_plys:
        print(f"No PLYs in {SYNTH_DIR}"); sys.exit(1)

    plys = all_plys[:n_scenes] if n_scenes else all_plys
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old batch/grid PNGs
    for old in PREVIEW_DIR.glob("*.png"):
        old.unlink()

    print(f"Generating {len(plys)} PNGs → {PREVIEW_DIR.resolve()}")
    for i, ply in enumerate(plys, 1):
        out = PREVIEW_DIR / f"{ply.stem}.png"
        render_scene(ply, out)
        if i % 10 == 0 or i == len(plys):
            print(f"  [{i}/{len(plys)}]")


if __name__ == "__main__":
    main()
