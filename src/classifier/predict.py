"""
predict.py — Run inference on any PLY file and produce a labelled PLY.

Usage (from src/):
    python3 classifier/predict.py input.ply output_labeled.ply [--model rf|lgbm]

Input PLY: ASCII or binary-LE, must have at least x y z fields.
           Optional fields (red, green, blue, label) are accepted but ignored.
Output PLY: binary-LE with fields x y z red green blue label,
            colours taken from LABEL_RGB (same as generate_dataset.py).
"""

import argparse
import struct
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from classifier.features import extract_features
from generate_dataset import LABEL_RGB


# ── PLY reader (ASCII + binary-LE, generic field discovery) ──────────────────

def _parse_header(raw: bytes):
    """
    Return (n_vertices, fields, format_str, header_end_offset).
    fields: list of (name, numpy_dtype_char)  in declaration order.
    format_str: 'ascii' | 'binary_little_endian' | 'binary_big_endian'
    """
    end_tag = b"end_header\n"
    end_idx = raw.find(end_tag)
    header = raw[:end_idx].decode("ascii", errors="replace")
    body_start = end_idx + len(end_tag)

    format_str = "binary_little_endian"
    n_vertices = 0
    fields = []

    _DTYPE_MAP = {
        "float": "f4", "float32": "f4",
        "double": "f8", "float64": "f8",
        "int": "i4", "int32": "i4",
        "uint": "u4", "uint32": "u4",
        "short": "i2", "int16": "i2",
        "ushort": "u2", "uint16": "u2",
        "char": "i1", "int8": "i1",
        "uchar": "u1", "uint8": "u1",
    }

    in_vertex = False
    for line in header.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format":
            format_str = tokens[1]
        elif tokens[0] == "element":
            in_vertex = tokens[1] == "vertex"
            if in_vertex:
                n_vertices = int(tokens[2])
        elif tokens[0] == "property" and in_vertex:
            dtype_char = _DTYPE_MAP.get(tokens[1], "f4")
            fields.append((tokens[2], dtype_char))

    return n_vertices, fields, format_str, body_start


def read_ply_xyz(path: Path) -> np.ndarray:
    """
    Read a PLY file (ASCII or binary-LE) and return xyz as (N, 3) float32.
    Other fields are discarded.
    """
    with open(path, "rb") as f:
        raw = f.read()

    n_vertices, fields, fmt, body_start = _parse_header(raw)
    field_names = [f[0] for f in fields]

    if fmt == "ascii":
        # Read numeric lines after header
        text = raw[body_start:].decode("ascii", errors="replace")
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
            if len(rows) == n_vertices:
                break
        arr = np.array(rows, dtype=np.float32)
        # Map by column position using field_names order
        xi = field_names.index("x")
        yi = field_names.index("y")
        zi = field_names.index("z")
        return arr[:, [xi, yi, zi]]

    else:  # binary_little_endian (or big, treated as LE)
        dtype = np.dtype([(name, "<" + dt) for name, dt in fields])
        arr = np.frombuffer(
            raw[body_start: body_start + n_vertices * dtype.itemsize],
            dtype=dtype,
        )
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)


# ── PLY writer (binary-LE: x y z red green blue label) ───────────────────────

def write_labeled_ply(path: Path, pts: np.ndarray, labels: np.ndarray) -> None:
    """
    Write binary-LE PLY with fields x(f) y(f) z(f) r(B) g(B) b(B) label(B).
    Colours are looked up from LABEL_RGB.
    """
    N = len(pts)
    rgb = np.array(
        [LABEL_RGB.get(int(lbl), (255, 0, 255)) for lbl in labels],
        dtype=np.uint8,
    )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar label\n"
        "end_header\n"
    ).encode("ascii")

    pts_f = pts.astype(np.float32)
    record_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("label", "u1"),
    ])
    records = np.empty(N, dtype=record_dtype)
    records["x"] = pts_f[:, 0]
    records["y"] = pts_f[:, 1]
    records["z"] = pts_f[:, 2]
    records["r"] = rgb[:, 0]
    records["g"] = rgb[:, 1]
    records["b"] = rgb[:, 2]
    records["label"] = labels.astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(records.tobytes())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-point 5-class prediction")
    parser.add_argument("input",  help="Input PLY file (ASCII or binary)")
    parser.add_argument("output", help="Output labelled PLY file (binary-LE)")
    parser.add_argument("--model", choices=["rf", "lgbm"], default="rf",
                        help="Which model to use (default: rf)")
    parser.add_argument("--models-dir", default="../models",
                        help="Directory with saved .pkl files")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    model_path = models_dir / f"classifier_{args.model}.pkl"
    if not model_path.exists():
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        print("Run classifier/train.py first.", file=sys.stderr)
        sys.exit(1)

    payload = joblib.load(model_path)
    model = payload["model"]
    scaler = payload["scaler"]

    pts = read_ply_xyz(Path(args.input))
    print(f"Loaded {len(pts):,} points from {args.input}")

    feats = extract_features(pts)
    feats_scaled = scaler.transform(feats)
    labels = model.predict(feats_scaled).astype(np.uint8)

    unique, counts = np.unique(labels, return_counts=True)
    label_map = payload.get("label_map", {})
    inv_map = {v: k for k, v in label_map.items()}
    print("Predicted label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u} ({inv_map.get(int(u), '?')}): {c:,}")

    write_labeled_ply(Path(args.output), pts, labels)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
