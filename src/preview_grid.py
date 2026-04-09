"""
preview_grid.py  —  Visual overview of the synthetic dataset.

Usage (from src/):
    python3 preview_grid.py                      # 50 scenes, one big grid
    python3 preview_grid.py --views 3            # 3 views per scene
    python3 preview_grid.py --batch 10           # multiple PNGs, 10 scenes each
    python3 preview_grid.py --batch 10 --views 3
    python3 preview_grid.py --n 20 --views 3
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


def draw_scene(ax, pts, lbs, view="top", point_size=1.0):
    """Draw one thumbnail on ax."""
    rgba = label_rgba(lbs)
    if view == "top":      # XZ floor plan
        ax.scatter(pts[:,0], pts[:,2], c=rgba, s=point_size, linewidths=0)
        ax.set_xlabel("X", fontsize=5, labelpad=1)
        ax.set_ylabel("Z", fontsize=5, labelpad=1)
    elif view == "front":  # XY elevation
        ax.scatter(pts[:,0], pts[:,1], c=rgba, s=point_size, linewidths=0)
        ax.set_xlabel("X", fontsize=5, labelpad=1)
        ax.set_ylabel("Y", fontsize=5, labelpad=1)
    elif view == "side":   # ZY
        ax.scatter(pts[:,2], pts[:,1], c=rgba, s=point_size, linewidths=0)
        ax.set_xlabel("Z", fontsize=5, labelpad=1)
        ax.set_ylabel("Y", fontsize=5, labelpad=1)

    ax.set_aspect("equal")
    ax.tick_params(labelsize=4, pad=1)
    ax.grid(True, lw=0.2, alpha=0.4)


def legend_patches():
    return [
        mpatches.Patch(color=mc.to_rgba(c), label=f"{LABEL_NAMES[lv]}")
        for lv, c in LABEL_COLORS.items()
    ]


def render_batch(plys: list, views: list, batch_idx: int, total_batches: int,
                 cols: int = COLS) -> Path:
    """Render one batch of scenes → returns saved path."""
    n_views = len(views)
    n_cols = cols * n_views
    n_rows = int(np.ceil(len(plys) / cols))

    cell_w = 4.0
    cell_h = 3.8
    fig_w = n_cols * cell_w
    fig_h = n_rows * cell_h + 0.7

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for scene_i, ply in enumerate(plys):
        pts, lbs = load_synth(ply)
        pts, lbs = subsample(pts, lbs, seed=scene_i)
        title = scene_title(ply, lbs)

        row = scene_i // cols
        base_col = (scene_i % cols) * n_views

        for v_i, view in enumerate(views):
            ax = axes[row][base_col + v_i]
            draw_scene(ax, pts, lbs, view=view, point_size=2.0)
            ax.set_title(f"{title} · {view}" if v_i == 0 else view, fontsize=7, pad=2)

    for ax in axes.flat[len(plys) * n_views:]:
        ax.set_visible(False)

    fig.legend(handles=legend_patches(), loc="lower center", ncol=7,
               fontsize=9, title="Labels", title_fontsize=10,
               framealpha=0.9, markerscale=2)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEW_DIR / f"batch_{batch_idx:02d}of{total_batches:02d}.png"
    plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.5, w_pad=0.4)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = sys.argv[1:]
    n_scenes  = 50
    n_views   = 3
    batch_sz  = None   # None = single big grid

    for i, a in enumerate(args):
        if a == "--n"     and i+1 < len(args): n_scenes = int(args[i+1])
        if a == "--views" and i+1 < len(args): n_views  = int(args[i+1])
        if a == "--batch" and i+1 < len(args): batch_sz = int(args[i+1])

    all_plys = sorted(SYNTH_DIR.glob("*.ply"))
    if not all_plys:
        print(f"No PLYs in {SYNTH_DIR}"); sys.exit(1)

    rng = np.random.default_rng(7)
    chosen = sorted(rng.choice(len(all_plys), min(n_scenes, len(all_plys)), replace=False))
    plys = [all_plys[i] for i in chosen]
    views = ["top", "front", "side"][:n_views]

    if batch_sz is None:
        # ── single big grid (legacy mode) ──
        cols = 5
        total_batches = 1
        out = render_batch(plys, views, 1, 1, cols=cols)
        out.rename(PREVIEW_DIR / f"dataset_grid_n{len(plys)}_v{n_views}.png")
        print(f"Saved → {(PREVIEW_DIR / f'dataset_grid_n{len(plys)}_v{n_views}.png').resolve()}")
    else:
        # ── multiple small images ──
        batches = [plys[i:i+batch_sz] for i in range(0, len(plys), batch_sz)]
        total = len(batches)
        print(f"{len(plys)} scenes → {total} images of ≤{batch_sz} scenes each  ({n_views} views/scene)")
        for b_i, batch in enumerate(batches, 1):
            out = render_batch(batch, views, b_i, total)
            print(f"  [{b_i}/{total}] {out.name}  ({len(batch)} scenes)")


if __name__ == "__main__":
    main()
