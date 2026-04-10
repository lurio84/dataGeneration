"""
test_classifier.py — Pytest tests for the 5-class per-point ML classifier.

All tests are self-contained and generate minimal synthetic data in
tempfile.TemporaryDirectory — no dependency on output/dataset/.

Run from src/:
    python3 -m pytest test_classifier.py -v
"""

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure imports resolve from src/
sys.path.insert(0, str(Path(__file__).parent))

from classifier.features import extract_features, FEATURE_NAMES, _N_FEATURES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_random_pts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(np.float32)


def _write_binary_ply(path: Path, pts: np.ndarray, labels: np.ndarray) -> None:
    """Write minimal binary-LE PLY (x y z r g b label)."""
    N = len(pts)
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

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("label", "u1"),
    ])
    records = np.empty(N, dtype=dtype)
    records["x"] = pts[:, 0]
    records["y"] = pts[:, 1]
    records["z"] = pts[:, 2]
    records["r"] = 128
    records["g"] = 128
    records["b"] = 128
    records["label"] = labels.astype(np.uint8)

    with open(path, "wb") as f:
        f.write(header)
        f.write(records.tobytes())


def _make_tiny_dataset(tmp_dir: Path, n_scenes=6, n_pts_per_scene=150, seed=42):
    """
    Generate minimal PLY files covering all 5 classes.
    Labels are assigned in round-robin: 0,1,2,3,4,0,1,...
    Label 255 sprinkled in at 5% to test exclusion.
    """
    rng = np.random.default_rng(seed)
    for scene_id in range(n_scenes):
        pts = rng.standard_normal((n_pts_per_scene, 3)).astype(np.float32)
        labels = np.arange(n_pts_per_scene, dtype=np.uint8) % 5
        # Add a few 255 outliers
        outlier_idx = rng.integers(0, n_pts_per_scene, size=max(1, n_pts_per_scene // 20))
        labels[outlier_idx] = 255
        _write_binary_ply(tmp_dir / f"{scene_id:05d}.ply", pts, labels)


def _train_tiny_model(tmp_dir: Path, model_dir: Path):
    """Run train.py with a tiny dataset; return path to saved RF pkl."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-u", "classifier/train.py",
            "--data", str(tmp_dir),
            "--out", str(model_dir),
            "--seed", "0",
            "--n-jobs", "1",        # avoid loky worker spawn overhead in tests
            "--cv-estimators", "10",  # minimal trees — just enough for functional check
            "--cv-subsample", "1.0",  # no subsampling on tiny dataset (already tiny)
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed:\n{result.stderr}\n{result.stdout}")
    return model_dir / "classifier_rf.pkl"


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_features_shape(self):
        """extract_features returns (N, 15) for N arbitrary points."""
        pts = _make_random_pts(n=150)
        feats = extract_features(pts)
        assert feats.shape == (150, _N_FEATURES), (
            f"Expected (150, {_N_FEATURES}), got {feats.shape}"
        )

    def test_features_dtype(self):
        """Output is float32."""
        pts = _make_random_pts(n=50)
        feats = extract_features(pts)
        assert feats.dtype == np.float32

    def test_features_no_nan(self):
        """No NaN or Inf in feature matrix for random data."""
        pts = _make_random_pts(n=300)
        feats = extract_features(pts)
        assert not np.any(np.isnan(feats)), "NaN found in features"
        assert not np.any(np.isinf(feats)), "Inf found in features"

    def test_features_floor_low_y(self):
        """Points at Y≈0 have feature 'y' < 0.05."""
        N = 100
        pts = np.zeros((N, 3), dtype=np.float32)
        pts[:, 0] = np.linspace(-1, 1, N)   # X spread
        pts[:, 2] = np.linspace(-0.5, 0.5, N)  # Z spread
        # Y = 0 (floor) with tiny jitter
        pts[:, 1] = np.random.default_rng(0).uniform(0, 0.01, N).astype(np.float32)

        feats = extract_features(pts)
        y_feat = feats[:, 0]   # feature index 0 = y
        assert np.all(y_feat < 0.05), (
            f"Some floor points have y_feature ≥ 0.05: max={y_feat.max():.4f}"
        )


class TestTrainAndPredict:
    def test_train_saves_model(self, tmp_path):
        """train.py creates models/classifier_rf.pkl."""
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        _make_tiny_dataset(data_dir)
        pkl_path = _train_tiny_model(data_dir, model_dir)

        assert pkl_path.exists(), f"Expected {pkl_path} to exist after training"

    def test_model_has_scaler(self, tmp_path):
        """Saved pickle contains keys 'model', 'scaler', 'feature_names'."""
        import joblib

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        _make_tiny_dataset(data_dir)
        pkl_path = _train_tiny_model(data_dir, model_dir)

        payload = joblib.load(pkl_path)
        for key in ("model", "scaler", "feature_names"):
            assert key in payload, f"Key '{key}' missing from model pickle"
        assert payload["feature_names"] == FEATURE_NAMES

    def test_predict_output_ply(self, tmp_path):
        """predict.py produces a PLY file at the requested output path."""
        import subprocess

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        _make_tiny_dataset(data_dir)
        _train_tiny_model(data_dir, model_dir)

        # Write a small input PLY (binary, no label field needed)
        input_ply = tmp_path / "input.ply"
        pts = _make_random_pts(n=50)
        _write_binary_ply(input_ply, pts, np.zeros(50, dtype=np.uint8))

        output_ply = tmp_path / "output_labeled.ply"
        result = subprocess.run(
            [
                sys.executable, "classifier/predict.py",
                str(input_ply), str(output_ply),
                "--model", "rf",
                "--models-dir", str(model_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0, f"predict.py failed:\n{result.stderr}\n{result.stdout}"
        assert output_ply.exists(), "Output PLY was not created"

    def test_predict_labels_valid(self, tmp_path):
        """All labels in output PLY ∈ {0, 1, 2, 3, 4, 255}."""
        import subprocess

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        _make_tiny_dataset(data_dir)
        _train_tiny_model(data_dir, model_dir)

        input_ply = tmp_path / "input.ply"
        pts = _make_random_pts(n=80)
        _write_binary_ply(input_ply, pts, np.zeros(80, dtype=np.uint8))

        output_ply = tmp_path / "output_labeled.ply"
        subprocess.run(
            [
                sys.executable, "classifier/predict.py",
                str(input_ply), str(output_ply),
                "--model", "rf",
                "--models-dir", str(model_dir),
            ],
            capture_output=True,
            cwd=str(Path(__file__).parent),
            check=True,
        )

        # Parse output PLY label field
        with open(output_ply, "rb") as f:
            raw = f.read()
        end_idx = raw.find(b"end_header\n")
        body = raw[end_idx + len(b"end_header\n"):]
        # x(f4) y(f4) z(f4) r(u1) g(u1) b(u1) label(u1) = 16 bytes
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ("label", "u1"),
        ])
        arr = np.frombuffer(body[: 80 * dtype.itemsize], dtype=dtype)
        valid_labels = {0, 1, 2, 3, 4, 255}
        unique_labels = set(arr["label"].tolist())
        assert unique_labels.issubset(valid_labels), (
            f"Invalid labels in output PLY: {unique_labels - valid_labels}"
        )
