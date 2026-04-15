"""Render top-down + isometric PNGs of geo_verify_v2 PLYs using the label
channel (0=floor, 1=rest/non-cargo, 3=cargo). Uses matplotlib (offscreen)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_ply(path: Path):
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header\n")
    header = raw[:end].decode()
    n = int(next(
        ln.split()[-1] for ln in header.splitlines()
        if ln.startswith("element vertex")
    ))
    body = raw[end + len("end_header\n"):]
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("label", "u1"),
    ])
    a = np.frombuffer(body[:n * dt.itemsize], dtype=dt)
    pts = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
    return pts, a["label"].astype(np.int32)


COLORS = {
    0: (0.35, 0.35, 0.35),   # floor  — gris
    1: (0.25, 0.70, 0.35),   # rest/non-cargo — verde (person/vehicle/other)
    2: (0.35, 0.35, 0.35),   # legacy floor
    3: (0.90, 0.15, 0.15),   # cargo — rojo
}


def render_scene(ply_path: Path, out_png: Path):
    pts, lbl = read_ply(ply_path)
    rgb = np.array([COLORS.get(int(l), (0.6, 0.6, 0.6)) for l in lbl])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Top-down (XZ) ────────────────────────────────────────────────────────
    order = np.argsort(pts[:, 1])
    axes[0].scatter(pts[order, 0], pts[order, 2], c=rgb[order], s=0.6, marker=".")
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Z (m)")
    axes[0].set_title("Top-down (XZ)")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)

    # ── Front (XY) ───────────────────────────────────────────────────────────
    order = np.argsort(pts[:, 2])
    axes[1].scatter(pts[order, 0], pts[order, 1], c=rgb[order], s=0.6, marker=".")
    axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")
    axes[1].set_title("Front (XY)")
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3)

    # ── Side (ZY) ────────────────────────────────────────────────────────────
    order = np.argsort(pts[:, 0])
    axes[2].scatter(pts[order, 2], pts[order, 1], c=rgb[order], s=0.6, marker=".")
    axes[2].set_xlabel("Z (m)"); axes[2].set_ylabel("Y (m)")
    axes[2].set_title("Side (ZY)")
    axes[2].set_aspect("equal"); axes[2].grid(True, alpha=0.3)

    n_cargo = int((lbl == 3).sum())
    n_rest = int((lbl == 1).sum())
    n_floor = int(((lbl == 0) | (lbl == 2)).sum())
    fig.suptitle(
        f"{ply_path.stem}   "
        f"cargo(red)={n_cargo}   rest(green)={n_rest}   floor(grey)={n_floor}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    in_dir = root / "output" / "geo_verify_v2"
    out_dir = root / "output" / "geo_verify_v2_png"
    out_dir.mkdir(parents=True, exist_ok=True)

    plys = sorted(in_dir.glob("bbb_*_stage123.ply"))
    for p in plys:
        out = out_dir / (p.stem + ".png")
        render_scene(p, out)
        print(f"  wrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()
