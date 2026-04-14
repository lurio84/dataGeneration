"""
analyze_predictions.py — Distribución de labels por franja de altura.

Herramienta diagnóstica para Fase 0-D1 del plan deep-dancing-flute: aisla
la contribución de shift/catch-all vs mesh §12 mirando la fracción de
vehicle (y otras clases) sobre y<0.1m (suelo) vs y>=0.1m (no-suelo).

Uso (desde raíz de datageneration/):
    python3 scripts/analyze_predictions.py output/bench_v1_vox035_nocw/scene5.ply
    python3 scripts/analyze_predictions.py output/bench_v1_vox035_nocw/scene*.ply
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from classifier.predict import _parse_header  # noqa: E402


LABEL_NAMES = {0: "floor", 1: "cargo", 2: "vehicle", 3: "person", 4: "pallet", 255: "outlier"}
Y_FLOOR_CUTOFF = 0.1  # m


def read_ply_xyzlabel(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        raw = f.read()
    n, fields, fmt, start = _parse_header(raw)
    if fmt != "binary_little_endian":
        raise SystemExit(f"{path}: solo binary_little_endian soportado (got {fmt})")
    dtype = np.dtype([(name, "<" + dt) for name, dt in fields])
    arr = np.frombuffer(raw[start : start + n * dtype.itemsize], dtype=dtype)
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    if "label" not in arr.dtype.names:
        raise SystemExit(f"{path}: no tiene campo 'label'")
    return pts, arr["label"].astype(np.uint8)


def analyze_one(path: Path) -> None:
    pts, labels = read_ply_xyzlabel(path)
    total = len(pts)
    mask_low = pts[:, 1] < Y_FLOOR_CUTOFF
    n_low = int(mask_low.sum())
    n_hi = total - n_low

    print(f"\n=== {path.name}  N={total}  (y<{Y_FLOOR_CUTOFF}: {n_low}, y>={Y_FLOOR_CUTOFF}: {n_hi}) ===")
    print(f"{'class':8s} {'all%':>8s} {'low%':>8s} {'high%':>8s}")
    for lv, name in LABEL_NAMES.items():
        m = labels == lv
        if not m.any():
            continue
        all_pct = m.sum() / total * 100
        low_pct = (m & mask_low).sum() / max(n_low, 1) * 100
        hi_pct = (m & ~mask_low).sum() / max(n_hi, 1) * 100
        print(f"{name:8s} {all_pct:7.2f}% {low_pct:7.2f}% {hi_pct:7.2f}%")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    for p in paths:
        if not p.exists():
            print(f"SKIP {p} (no existe)", file=sys.stderr)
            continue
        analyze_one(p)


if __name__ == "__main__":
    main()
