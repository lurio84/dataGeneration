#!/usr/bin/env python3
"""
train_person_raw.py — Per-point person/cargo classifier on raw per-camera PLY clouds.

Phase 1: Extract pseudo-labeled features from BBB captures (Captura_01 per escenario).
Phase 2: Train RF + LGBM with leave-one-escenario-out cross-validation.

Usage (from datageneration/ root):
    # Train (phases 1+2):
    python scripts/train_person_raw.py \\
        --capturas-dir Resources/Capturas_BBB/2026_03_27/Escenarios_CATEC_25032026/BBB \\
        --output-dir models

    # Predict on new cloud:
    python scripts/train_person_raw.py \\
        --predict path/to/PLY_Izq.ply \\
        --model models/person_raw_classifier.pkl \\
        --output path/to/output_labeled.ply
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut

try:
    import lightgbm as lgb
    _LGBM_OK = True
except ImportError:
    _LGBM_OK = False

# ── Camera / geometry constants ───────────────────────────────────────────────
FX = FY = 1680.0
CX, CY = 1045.0, 758.0

THETA = np.radians(47.5)
_SIN_T = np.sin(THETA)
_COS_T = np.cos(THETA)
CAM_H_OFFSET = 3.80  # metres — Izq/Der cameras per config_base.ini

# ── Feature extraction constants ──────────────────────────────────────────────
K_SMALL = 20
K_LARGE = 50
EPS = 1e-10

FEATURE_NAMES = [
    "curvature",
    "planarity",
    "linearity",
    "sphericity",
    "lam_ratio12",
    "planarity_L",
    "linearity_L",
    "sphericity_L",
    "normal_y_std",
    "height",
]

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_UNLABELED = 0
LABEL_PERSON = 1
LABEL_CARGO = 2

SAMPLE_PER_CLASS = 10_000

# ── YOLO singleton ────────────────────────────────────────────────────────────
_yolo_model = None


def _find_yolo_weights() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "yolov8n.pt"
        if candidate.exists():
            return str(candidate)
    return "yolov8n.pt"  # let ultralytics handle download


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(_find_yolo_weights())
    return _yolo_model


# ── PLY I/O ───────────────────────────────────────────────────────────────────

def _parse_ply_header(f) -> tuple[int, bool]:
    """Return (n_vertices, is_binary_le) from open binary file."""
    n = 0
    binary_le = False
    for raw in f:
        line = raw.decode("ascii", errors="ignore").strip()
        if line.startswith("format binary_little_endian"):
            binary_le = True
        elif line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line == "end_header":
            break
    return n, binary_le


def read_ply_binary_xyzrgb(path: Path) -> np.ndarray:
    """Read binary-LE PLY (15 bytes/vertex: 3×float32 xyz + 3×uint8 rgb).

    Returns xyz [N, 3] float32 in camera frame.
    """
    with open(path, "rb") as f:
        n, binary_le = _parse_ply_header(f)
        if not binary_le or n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        raw = f.read(n * 15)
    actual = len(raw) // 15
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    arr = np.frombuffer(raw[: actual * 15], dtype=dt)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1)


def _write_ply_ascii(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write ASCII PLY compatible with CloudCompare."""
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# ── Coordinate transform ──────────────────────────────────────────────────────

def cam_to_world(xyz_cam: np.ndarray) -> np.ndarray:
    """Camera → world frame.

    Returns pts [N, 3]: [:, 0]=world_x, [:, 1]=height (world_z), [:, 2]=world_y.
    """
    cx = xyz_cam[:, 0]
    cy = xyz_cam[:, 1]
    cz = xyz_cam[:, 2]
    height  = CAM_H_OFFSET - cz * _SIN_T - cy * _COS_T
    world_x = cz * _COS_T + cy * _SIN_T
    world_y = -cx
    return np.stack([world_x, height, world_y], axis=1).astype(np.float32)


# ── YOLO pseudo-labelling ─────────────────────────────────────────────────────

def _detect_person_bbox(png_path: Path, conf: float = 0.5) -> Optional[list]:
    """Return [x1, y1, x2, y2] of highest-confidence person detection or None."""
    import cv2
    img = cv2.imread(str(png_path))
    if img is None:
        return None
    model = _get_yolo()
    results = model(img, conf=conf, imgsz=640, verbose=False)[0]
    best_conf, best_box = 0.0, None
    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue
        c = float(box.conf[0])
        if c > best_conf:
            best_conf = c
            best_box = box.xyxy[0].tolist()
    return best_box


def pseudo_label(xyz_cam: np.ndarray, world_pts: np.ndarray,
                 png_path: Path, conf: float = 0.5) -> np.ndarray:
    """Assign LABEL_PERSON / LABEL_CARGO / LABEL_UNLABELED per point."""
    labels = np.zeros(len(xyz_cam), dtype=np.int8)
    height = world_pts[:, 1]

    valid_z = xyz_cam[:, 2] > 0
    denom = np.where(valid_z, xyz_cam[:, 2], 1.0)
    u = np.where(valid_z, FX * xyz_cam[:, 0] / denom + CX, -1.0)
    v = np.where(valid_z, FY * xyz_cam[:, 1] / denom + CY, -1.0)

    bbox = _detect_person_bbox(png_path, conf)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        in_bbox = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        labels[in_bbox & (height > 0.3)] = LABEL_PERSON

    not_person = labels != LABEL_PERSON
    abs_wy = np.abs(world_pts[:, 2])
    labels[not_person & (height > 0.15) & (abs_wy < 1.5)] = LABEL_CARGO

    return labels


# ── Feature extraction ────────────────────────────────────────────────────────

def _batch_pca(pts: np.ndarray,
               indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched PCA on k-neighbourhoods.

    pts: [N, 3], indices: [N, k] into pts.
    Returns eigenvalues [N, 3] ascending, eigenvectors [N, 3, 3].
    """
    nbrs = pts[indices]                                        # [N, k, 3]
    centroid = nbrs.mean(axis=1, keepdims=True)
    centered = nbrs - centroid
    k = indices.shape[1]
    cov = np.einsum("nka,nkb->nab", centered, centered) / max(k - 1, 1)
    return np.linalg.eigh(cov)  # ascending eigenvalues


def extract_features(pts: np.ndarray) -> np.ndarray:
    """Extract 10 per-point features.

    pts [N, 3]: [:, 0]=world_x, [:, 1]=height, [:, 2]=world_y.
    Returns float32 [N, 10].
    """
    tree = KDTree(pts)
    # Single query at K_LARGE+1 (includes self at position 0); slice for K_SMALL.
    _, idx_all = tree.query(pts, k=K_LARGE + 1, workers=-1)
    idx20 = idx_all[:, 1:K_SMALL + 1]   # [N, K_SMALL]
    idx50 = idx_all[:, 1:]              # [N, K_LARGE]

    vals20, vecs20 = _batch_pca(pts, idx20)   # ascending: [:,0] = λ3 (smallest)
    vals50, _      = _batch_pca(pts, idx50)

    lam1_20, lam2_20, lam3_20 = vals20[:, 2], vals20[:, 1], vals20[:, 0]
    lam1_50, lam2_50, lam3_50 = vals50[:, 2], vals50[:, 1], vals50[:, 0]

    curvature   = lam3_20 / (lam1_20 + lam2_20 + lam3_20 + EPS)
    planarity   = (lam2_20 - lam3_20) / (lam1_20 + EPS)
    linearity   = (lam1_20 - lam2_20) / (lam1_20 + EPS)
    sphericity  = lam3_20 / (lam1_20 + EPS)
    lam_ratio12 = lam1_20 / (lam2_20 + EPS)

    planarity_L  = (lam2_50 - lam3_50) / (lam1_50 + EPS)
    linearity_L  = (lam1_50 - lam2_50) / (lam1_50 + EPS)
    sphericity_L = lam3_50 / (lam1_50 + EPS)

    # normal = min-eigenvalue eigenvector; height component = axis index 1
    normal_y     = np.abs(vecs20[:, 1, 0])                    # [N]
    normal_y_std = np.std(normal_y[idx20], axis=1)            # [N]

    height = pts[:, 1]

    return np.column_stack([
        curvature, planarity, linearity, sphericity, lam_ratio12,
        planarity_L, linearity_L, sphericity_L,
        normal_y_std, height,
    ]).astype(np.float32)


# ── Dataset building ──────────────────────────────────────────────────────────

def _process_cloud(ply_path: Path, png_path: Path, escenario_id: int,
                   rng: np.random.Generator) -> Optional[tuple]:
    """Pseudo-label → sample → extract features for one cloud.

    Returns (X [M, 10], y [M], groups [M]) or None if too few points.
    """
    xyz_cam = read_ply_binary_xyzrgb(ply_path)
    if len(xyz_cam) < 1000:
        return None

    world_pts = cam_to_world(xyz_cam)
    labels = pseudo_label(xyz_cam, world_pts, png_path)

    person_idx = np.where(labels == LABEL_PERSON)[0]
    cargo_idx  = np.where(labels == LABEL_CARGO)[0]

    if len(person_idx) < 100 or len(cargo_idx) < 100:
        return None

    if len(person_idx) > SAMPLE_PER_CLASS:
        person_idx = rng.choice(person_idx, SAMPLE_PER_CLASS, replace=False)
    if len(cargo_idx) > SAMPLE_PER_CLASS:
        cargo_idx = rng.choice(cargo_idx, SAMPLE_PER_CLASS, replace=False)

    subset_idx = np.concatenate([person_idx, cargo_idx])
    y_sub      = labels[subset_idx]

    # Extract features on the full cloud so neighbourhoods reflect real density,
    # then sample by index. Must match predict_cloud() which runs on full cloud.
    X_full = extract_features(world_pts)
    X = X_full[subset_idx]
    groups = np.full(len(y_sub), escenario_id, dtype=np.int32)
    return X, y_sub, groups


def build_dataset(capturas_dir: Path,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate escenarios 01-19, Captura_01 each. Returns (X, y, groups)."""
    Xs, ys, gs = [], [], []
    for esc_id in range(1, 20):
        esc     = f"Escenario_{esc_id:02d}"
        ply     = capturas_dir / esc / "Captura_01" / "PLY_Izq.ply"
        png     = capturas_dir / esc / "Captura_01" / "PNG_Izq.png"
        if not ply.exists():
            print(f"  [SKIP] {esc}: PLY not found")
            continue
        if not png.exists():
            print(f"  [SKIP] {esc}: PNG not found")
            continue

        t0 = time.time()
        result = _process_cloud(ply, png, esc_id, rng)
        dt = time.time() - t0

        if result is None:
            print(f"  [SKIP] {esc}: insufficient labeled points")
            continue

        X, y, groups = result
        Xs.append(X)
        ys.append(y)
        gs.append(groups)
        print(f"  {esc}: {len(y):>6} pts  "
              f"person={np.sum(y == LABEL_PERSON):>5}  "
              f"cargo={np.sum(y == LABEL_CARGO):>5}  "
              f"{dt:.1f}s")

    if not Xs:
        raise RuntimeError("No valid escenarios found — check --capturas-dir path.")

    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


# ── Cross-validation ──────────────────────────────────────────────────────────

def _run_loocv(model_fn, X: np.ndarray, y: np.ndarray,
               groups: np.ndarray, label: str) -> dict:
    """Leave-one-escenario-out CV. Returns metrics dict."""
    logo     = LeaveOneGroupOut()
    unique_g = np.unique(groups)
    fold_f1s = []

    for train_idx, test_idx in logo.split(X, y, groups):
        m = model_fn()
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        fold_f1s.append(
            f1_score(y[test_idx], pred, average="macro", zero_division=0)
        )

    mean_f1 = float(np.mean(fold_f1s))
    std_f1  = float(np.std(fold_f1s))
    print(f"  {label:5s}  macro-F1 = {mean_f1:.3f} ± {std_f1:.3f}")

    per_esc = {f"E{g:02d}": float(fold_f1s[i]) for i, g in enumerate(unique_g)}
    return {"mean_f1": mean_f1, "std_f1": std_f1, "per_escenario": per_esc}


def train_models(X: np.ndarray, y: np.ndarray,
                 groups: np.ndarray, seed: int = 42) -> tuple:
    """Run LOOCV for RF + LGBM, fit winner on full data. Returns (model, metrics)."""
    def rf_fn():
        return RandomForestClassifier(
            n_estimators=100, max_depth=12,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )

    def lgbm_fn():
        return lgb.LGBMClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=seed, verbose=-1,
        )

    print("\n── Cross-validation ─────────────────────────────────────────────")
    rf_cv   = _run_loocv(rf_fn, X, y, groups, "RF")
    lgbm_cv = _run_loocv(lgbm_fn, X, y, groups, "LGBM") if _LGBM_OK else None

    print("\n── Final training (full dataset) ────────────────────────────────")
    if lgbm_cv is not None and lgbm_cv["mean_f1"] >= rf_cv["mean_f1"]:
        best_label, best_fn = "LGBM", lgbm_fn
    else:
        best_label, best_fn = "RF", rf_fn

    best_model = best_fn()
    best_model.fit(X, y)
    print(f"  Best model: {best_label}")

    pred_train = best_model.predict(X)
    print(classification_report(
        y, pred_train,
        target_names=["person", "cargo"],
        labels=[LABEL_PERSON, LABEL_CARGO],
        zero_division=0,
    ))

    imp = getattr(best_model, "feature_importances_", None)
    metrics = {
        "best_model": best_label,
        "rf_cv": rf_cv,
        "lgbm_cv": lgbm_cv,
        "feature_importance": (
            {FEATURE_NAMES[i]: float(imp[i]) for i in range(len(FEATURE_NAMES))}
            if imp is not None else {}
        ),
    }
    return best_model, metrics


# ── Predict ───────────────────────────────────────────────────────────────────

_LABEL_COLORS = {
    LABEL_PERSON:    (39, 174, 96),
    LABEL_CARGO:     (220, 50, 50),
    LABEL_UNLABELED: (60, 60, 60),
}


def predict_cloud(ply_path: Path, model_path: Path, out_path: Path) -> None:
    """Predict per-point labels on a new raw cloud and write labelled PLY."""
    pkg   = joblib.load(model_path)
    model = pkg["model"]

    xyz_cam   = read_ply_binary_xyzrgb(ply_path)
    if len(xyz_cam) == 0:
        print("ERROR: empty cloud")
        return

    world_pts = cam_to_world(xyz_cam)
    X         = extract_features(world_pts)
    pred      = model.predict(X)

    rgb = np.array(
        [_LABEL_COLORS.get(int(p), (60, 60, 60)) for p in pred],
        dtype=np.uint8,
    )
    _write_ply_ascii(out_path, world_pts, rgb)

    print(f"Wrote {len(pred)} points → {out_path}")
    print(f"  person={np.sum(pred == LABEL_PERSON)}  "
          f"cargo={np.sum(pred == LABEL_CARGO)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Per-point person/cargo classifier on raw BBB PLY clouds."
    )
    # Training mode
    p.add_argument("--capturas-dir", type=Path,
                   help="BBB captures root (contains Escenario_XX/)")
    p.add_argument("--output-dir", type=Path,
                   help="Directory for model, dataset, and metrics outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-extract", action="store_true",
                   help="Re-extract features even if dataset NPZ already exists")
    # Predict mode
    p.add_argument("--predict", type=Path, metavar="PLY",
                   help="Raw PLY cloud to classify")
    p.add_argument("--model", type=Path,
                   help="Trained classifier .pkl (required with --predict)")
    p.add_argument("--output", type=Path,
                   help="Output labelled PLY path (required with --predict)")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.predict is not None:
        if args.model is None or args.output is None:
            print("ERROR: --predict requires --model and --output")
            sys.exit(1)
        predict_cloud(args.predict, args.model, args.output)
        return

    # ── Training mode ─────────────────────────────────────────────────────────
    if args.capturas_dir is None or args.output_dir is None:
        print("ERROR: training mode requires --capturas-dir and --output-dir")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dataset_path = args.output_dir / "person_raw_dataset.npz"

    # Phase 1 — feature extraction
    if not args.force_extract and dataset_path.exists():
        print(f"Loading existing dataset from {dataset_path}")
        data   = np.load(dataset_path)
        X, y, groups = data["X"], data["y"], data["groups"]
    else:
        print("── Phase 1: feature extraction ──────────────────────────────────")
        t0 = time.time()
        X, y, groups = build_dataset(args.capturas_dir, rng)
        np.savez(dataset_path, X=X, y=y, groups=groups)
        print(f"\nDataset → {dataset_path}  ({len(y)} pts, {time.time() - t0:.0f}s)")

    print(f"\nDataset: {len(y)} pts — "
          f"person={np.sum(y == LABEL_PERSON)}  cargo={np.sum(y == LABEL_CARGO)}")

    # Phase 2 — training
    print("\n── Phase 2: training ────────────────────────────────────────────")
    t0 = time.time()
    best_model, metrics = train_models(X, y, groups, seed=args.seed)

    model_path = args.output_dir / "person_raw_classifier.pkl"
    joblib.dump(
        {
            "model": best_model,
            "feature_names": FEATURE_NAMES,
            "label_map": {LABEL_PERSON: "person", LABEL_CARGO: "cargo"},
            "version": 1,
        },
        model_path,
    )
    print(f"\nModel   → {model_path}")

    metrics_path = args.output_dir / "person_raw_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics → {metrics_path}")
    print(f"\nTotal training time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
