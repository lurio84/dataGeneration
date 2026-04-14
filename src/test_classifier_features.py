"""
test_classifier_features.py — Unit tests for height_above_local_floor.

Tests verify correctness of the local-floor estimation in isolation, without
running the full training pipeline.

Run from datageneration/:
    python3 -m pytest src/test_classifier_features.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from classifier.features import (
    FEATURE_NAMES,
    FLOOR_CELL_SIZE_M,
    _N_FEATURES,
    _height_above_local_floor,
    extract_features,
)

# Index of the feature under test
_FEAT_IDX = FEATURE_NAMES.index("height_above_local_floor")


class TestHeightAboveLocalFloor:
    def test_flat_floor_and_cube(self):
        """
        Dense floor at Y=0 + a cube cluster at Y=1.
        Floor points → height_above_local_floor ≈ 0.
        Cube points  → height_above_local_floor ≈ 1.

        Floor uses step=0.1 m → 25 floor pts per 0.5 m cell.
        Cube has 20 pts spread over ~4 cells (~5/cell).
        Floor (25) dominates cube (5) so per-cell median stays near 0.
        """
        rng = np.random.default_rng(0)

        # Floor: 100×100 grid (step=0.1 m), 10000 pts, Y≈0 + tiny jitter
        # → ~25 floor pts per 0.5 m cell
        xs = np.arange(0.0, 10.0, 0.1, dtype=np.float32)
        zs = np.arange(0.0, 10.0, 0.1, dtype=np.float32)
        gx, gz = np.meshgrid(xs, zs)
        floor_x = gx.ravel()
        floor_z = gz.ravel()
        floor_y = rng.uniform(0.0, 0.01, len(floor_x)).astype(np.float32)
        floor_pts = np.stack([floor_x, floor_y, floor_z], axis=1)

        # Cube: 20 points clustered near (1.25, 1.0, 1.25)
        # Same XZ cell as dense floor → floor still dominates median
        cube_pts = rng.uniform(
            [1.1, 0.95, 1.1], [1.4, 1.05, 1.4], size=(20, 3)
        ).astype(np.float32)

        pts = np.concatenate([floor_pts, cube_pts], axis=0)
        n_floor = len(floor_pts)

        h = _height_above_local_floor(pts)

        assert h[n_floor:].mean() == pytest.approx(1.0, abs=0.15), (
            f"Cube height_above_local_floor expected ≈1.0, got {h[n_floor:].mean():.3f}"
        )
        assert np.abs(h[:n_floor]).mean() < 0.05, (
            f"Floor height_above_local_floor expected ≈0.0, got mean={np.abs(h[:n_floor]).mean():.3f}"
        )

    def test_tilted_floor_compensates_gradient(self):
        """
        Floor with linear Y-gradient (Y = 0.1 * X).
        Cargo points sit 1 m above the local floor at two different X positions.
        height_above_local_floor should be ≈ 1.0 at both positions,
        NOT ≈ (0.1*X + 1.0), which would be the result of using global Y.
        """
        rng = np.random.default_rng(1)

        # Tilted floor: X in [0, 4], Z in [0, 1], Y = 0.1 * X + small jitter
        xs = np.arange(0.0, 4.05, 0.1, dtype=np.float32)
        zs = np.arange(0.0, 1.05, 0.1, dtype=np.float32)
        gx, gz = np.meshgrid(xs, zs)
        fx = gx.ravel()
        fz = gz.ravel()
        fy = (0.1 * fx + rng.uniform(0, 0.005, len(fx)).astype(np.float32))
        floor_pts = np.stack([fx, fy, fz], axis=1)

        # Cargo at two X positions, each 1 m above local floor
        # X≈0.25 → floor Y≈0.025; cargo Y≈1.025
        # X≈3.25 → floor Y≈0.325; cargo Y≈1.325
        cargo_pts = np.array([
            [0.25, 1.025, 0.5],
            [3.25, 1.325, 0.5],
        ], dtype=np.float32)

        pts = np.concatenate([floor_pts, cargo_pts], axis=0)
        n_floor = len(floor_pts)

        h = _height_above_local_floor(pts)

        h_cargo = h[n_floor:]
        assert h_cargo[0] == pytest.approx(1.0, abs=0.15), (
            f"Cargo at X=0.25: expected height≈1.0, got {h_cargo[0]:.3f}"
        )
        assert h_cargo[1] == pytest.approx(1.0, abs=0.15), (
            f"Cargo at X=3.25: expected height≈1.0, got {h_cargo[1]:.3f}"
        )
        # Global-Y baseline would give 1.025 and 1.325 — ensure we're below that
        global_y_median = float(np.median(pts[:, 1]))
        h_global_cargo0 = float(cargo_pts[0, 1]) - global_y_median
        assert abs(h_cargo[0] - 1.0) < abs(h_global_cargo0 - 1.0) + 0.01, (
            "Local floor estimate should be closer to 1.0 than global median"
        )

    def test_extract_features_shape_and_dtype(self):
        """
        extract_features returns (N, 16) float32.
        Also checks no NaN/Inf are introduced by height_above_local_floor.
        """
        rng = np.random.default_rng(2)
        pts = rng.standard_normal((300, 3)).astype(np.float32)
        feats = extract_features(pts)

        assert feats.shape == (300, _N_FEATURES), (
            f"Expected shape (300, {_N_FEATURES}), got {feats.shape}"
        )
        assert feats.dtype == np.float32, f"Expected float32, got {feats.dtype}"
        assert not np.any(np.isnan(feats[:, _FEAT_IDX])), (
            "NaN in height_above_local_floor"
        )
        assert not np.any(np.isinf(feats[:, _FEAT_IDX])), (
            "Inf in height_above_local_floor"
        )
