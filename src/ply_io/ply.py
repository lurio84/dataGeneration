"""
ply_io/ply.py  —  PLY I/O, label map and colour helpers.

Moved from generate_dataset.py (B9 refactor).
"""

import numpy as np

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL: dict[str, int] = {"floor": 0, "cargo": 1, "vehicle": 2, "person": 3, "pallet": 4}

# ── Label → RGB colour map ────────────────────────────────────────────────────
#  Colours are baked into the PLY so CloudCompare shows them immediately.
#  The numeric `label` field is also kept for programmatic use.

LABEL_RGB: dict[int, tuple[int, int, int]] = {
    0:   ( 60,  60,  60),   # floor   — gris oscuro, retrocede sobre fondo oscuro CC
    1:   (220,  50,  50),   # cargo   — rojo
    2:   (  0, 120, 255),   # vehicle — azul vivo
    3:   ( 39, 174,  96),   # person  — verde
    4:   (255, 210,   0),   # pallet  — amarillo
    255: ( 60,  60,  60),   # outlier — mismo gris oscuro que suelo
}
_DEFAULT_RGB = (255, 0, 255)   # magenta for unknown labels


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Map label array → uint8 RGB array (N, 3)."""
    rgb = np.full((len(labels), 3), _DEFAULT_RGB, dtype=np.uint8)
    for lv, color in LABEL_RGB.items():
        rgb[labels == lv] = color
    return rgb


# ── PLY export (binary, no PCL camera block) ───────────────────────────────────

def save_ply(path, pts: np.ndarray, labels: np.ndarray) -> None:
    """
    Write binary-little-endian PLY with: x y z red green blue label.
    - RGB encodes the label → CloudCompare shows colours immediately on open.
    - `label` scalar field is kept for programmatic use.
    Hand-written header → no PCL camera block.
    """
    n = len(pts)
    rgb = labels_to_rgb(labels)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar label\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "u1"),
    ])
    data = np.empty(n, dtype=dtype)
    data["x"] = pts[:, 0].astype(np.float32)
    data["y"] = pts[:, 1].astype(np.float32)
    data["z"] = pts[:, 2].astype(np.float32)
    data["red"]   = rgb[:, 0]
    data["green"] = rgb[:, 1]
    data["blue"]  = rgb[:, 2]
    data["label"] = labels.astype(np.uint8)

    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())
