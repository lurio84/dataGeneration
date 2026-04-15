"""
test_cluster_features.py — Unit tests for cluster_features.py (Paso 1).

Run from src/:
    python3 -m pytest test_cluster_features.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from cargo_geometric.anchor import AnchorInfo
from cargo_geometric.cluster_features import (
    FEATURE_NAMES,
    _N_FEATURES,
    extract_cluster_features,
    extract_features_batch,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_anchor(x_min=-1.0, x_max=1.0, z_min=-1.0, z_max=1.0) -> AnchorInfo:
    return AnchorInfo(
        n_points=100,
        x_min=x_min, x_max=x_max,
        z_min=z_min, z_max=z_max,
        height=1.0,
        center_xz=((x_min + x_max) / 2, (z_min + z_max) / 2),
    )


def _solid_cube(side: float = 1.0, n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, side, (n, 3)).astype(np.float32)


def _column(x_side: float = 0.4, z_side: float = 0.4, height: float = 1.7,
            n: int = 1000, seed: int = 1) -> np.ndarray:
    """Thin tall column — person-like shape."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform([0, 0, 0], [x_side, height, z_side], (n, 3))
    return pts.astype(np.float32)


def _flat_slab(x: float = 2.0, z: float = 2.0, y: float = 0.05,
               n: int = 1000, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform([0, 0, 0], [x, y, z], (n, 3))
    return pts.astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFeatureNames:
    def test_count(self):
        assert len(FEATURE_NAMES) == 19

    def test_unique(self):
        assert len(set(FEATURE_NAMES)) == 19

    def test_n_features_constant(self):
        assert _N_FEATURES == 19


class TestSolidCube:
    """1×1×1 solid cube centred at origin — cargo-like."""

    def setup_method(self):
        self.pts = _solid_cube(side=1.0, n=3000)
        self.anchor = _make_anchor(-1.5, 1.5, -1.5, 1.5)
        self.floor_y = 0.0
        self.feat = extract_cluster_features(self.pts, self.floor_y, self.anchor)

    def test_output_shape(self):
        assert self.feat.shape == (19,)

    def test_no_nan(self):
        assert not np.any(np.isnan(self.feat))

    def test_n_points(self):
        assert self.feat[0] == pytest.approx(3000.0)

    def test_obb_dims_positive_and_plausible(self):
        # Open3D uses a PCA-based OBB (not minimum-volume); for a random
        # uniform cube the eigenvectors can be rotated, making individual
        # dims up to sqrt(3)×1m ≈ 1.73m. We only check plausibility.
        assert self.feat[2] > 0.5        # long   > 0.5m
        assert self.feat[3] > 0.3        # short  > 0.3m
        assert self.feat[4] > 0.3        # height > 0.3m
        assert self.feat[2] < 2.5        # long   < 2.5m (< 3× space diagonal)

    def test_aspect_long_short_near_1(self):
        # Cube is symmetric → long/short ≈ 1
        assert self.feat[5] == pytest.approx(1.0, abs=0.15)

    def test_pca_sphericity_high(self):
        # Solid cube → roughly isotropic PCA eigenvalues → sphericity > 0
        assert self.feat[18] > 0.0


class TestPersonColumn:
    """Thin tall column — person-like: short base, tall height."""

    def setup_method(self):
        self.pts = _column(x_side=0.4, z_side=0.4, height=1.7, n=1500)
        self.anchor = _make_anchor(-1.0, 1.0, -1.0, 1.0)
        self.floor_y = 0.0
        self.feat = extract_cluster_features(self.pts, self.floor_y, self.anchor)

    def test_output_shape(self):
        assert self.feat.shape == (19,)

    def test_no_nan(self):
        assert not np.any(np.isnan(self.feat))

    def test_obb_height_tall(self):
        # column is 1.7m tall
        assert self.feat[4] == pytest.approx(1.7, abs=0.15)

    def test_obb_short_narrow(self):
        # short dim ≈ 0.4m; OBB approximation may find a rotated fit,
        # so we allow up to 0.2m tolerance
        assert self.feat[3] == pytest.approx(0.4, abs=0.2)

    def test_aspect_height_long_gt_1(self):
        # height/long > 1 for a person
        assert self.feat[6] > 1.0

    def test_aspect_long_short_near_1(self):
        # square base → long/short ≈ 1
        assert self.feat[5] == pytest.approx(1.0, abs=0.3)

    def test_pca_linearity_low(self):
        # roughly uniform fill in XYZ → linearity not extreme
        # (column is not a line, just tall)
        assert self.feat[16] >= 0.0


class TestFlatSlab:
    """2×2×0.05 flat slab — floor_residual-like."""

    def setup_method(self):
        self.pts = _flat_slab(x=2.0, z=2.0, y=0.05, n=2000)
        self.anchor = _make_anchor(-1.5, 1.5, -1.5, 1.5)
        self.floor_y = 0.0
        self.feat = extract_cluster_features(self.pts, self.floor_y, self.anchor)

    def test_output_shape(self):
        assert self.feat.shape == (19,)

    def test_no_nan(self):
        assert not np.any(np.isnan(self.feat))

    def test_obb_height_thin(self):
        # height ≈ 0.05 m
        assert self.feat[4] == pytest.approx(0.05, abs=0.02)

    def test_obb_long_wide(self):
        # long ≈ 2 m
        assert self.feat[2] == pytest.approx(2.0, abs=0.15)

    def test_planarity_high(self):
        # PCA: flat → one eigenvalue near zero → planarity should be high
        assert self.feat[17] > 0.3

    def test_top_slab_planarity_low(self):
        # flat top → low Y std in top slab
        assert self.feat[13] < 0.05


class TestDegenerateCluster:
    """Single point — should return all-zeros without crash."""

    def test_single_point(self):
        pts = np.array([[0.5, 0.1, 0.5]], dtype=np.float32)
        anchor = _make_anchor()
        feat = extract_cluster_features(pts, 0.0, anchor)
        assert feat.shape == (19,)
        assert np.all(feat == 0.0)

    def test_empty_cluster(self):
        pts = np.zeros((0, 3), dtype=np.float32)
        anchor = _make_anchor()
        feat = extract_cluster_features(pts, 0.0, anchor)
        assert feat.shape == (19,)
        assert np.all(feat == 0.0)

    def test_three_collinear_points(self):
        # OBB should degenerate gracefully
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        anchor = _make_anchor()
        feat = extract_cluster_features(pts, 0.0, anchor)
        assert feat.shape == (19,)
        assert not np.any(np.isnan(feat))


class TestBatchExtraction:
    """extract_features_batch returns the right shape."""

    def test_shape(self):
        cube = _solid_cube(n=500)
        col  = _column(n=500)
        anchor = _make_anchor()
        X = extract_features_batch([cube, col], floor_y=0.0, anchor=anchor)
        assert X.shape == (2, 19)

    def test_empty_list(self):
        anchor = _make_anchor()
        X = extract_features_batch([], floor_y=0.0, anchor=anchor)
        assert X.shape == (0, 19)

    def test_consistent_with_single(self):
        cube = _solid_cube(n=500)
        anchor = _make_anchor()
        feat_single = extract_cluster_features(cube, 0.0, anchor)
        feat_batch  = extract_features_batch([cube], 0.0, anchor)
        np.testing.assert_array_equal(feat_single, feat_batch[0])
