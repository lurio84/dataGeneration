"""
test_cluster_gt.py — Tests for cluster_gt.py (Paso 2).

Run from src/:
    python3 -m pytest test_cluster_gt.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from cargo_geometric.params import GeometricParams
from cargo_geometric.cluster_gt import (
    CLUSTER_LABEL_MAP,
    CLUSTER_LABEL_NAMES,
    build_cluster_dataset,
    _RAW_TO_CLUSTER,
)


DATASET_DIR = Path(__file__).parent.parent / "output" / "dataset"
N_FEATURES = 19


# ── Label map sanity ─────────────────────────────────────────────────────────

class TestLabelMap:
    def test_keys(self):
        # 3-class design: floor_residual and other merged into vehicle
        assert set(CLUSTER_LABEL_MAP.keys()) == {"cargo", "vehicle", "person"}

    def test_pallet_maps_to_cargo(self):
        # raw label 4 (pallet) → cluster cargo (1)
        assert _RAW_TO_CLUSTER[4] == CLUSTER_LABEL_MAP["cargo"]

    def test_cargo_maps_to_cargo(self):
        assert _RAW_TO_CLUSTER[1] == CLUSTER_LABEL_MAP["cargo"]

    def test_person_maps_to_person(self):
        assert _RAW_TO_CLUSTER[3] == CLUSTER_LABEL_MAP["person"]

    def test_floor_maps_to_vehicle(self):
        # floor residuals (raw 0) merged into vehicle class
        assert _RAW_TO_CLUSTER[0] == CLUSTER_LABEL_MAP["vehicle"]

    def test_inverse_map_complete(self):
        for k, v in CLUSTER_LABEL_MAP.items():
            assert CLUSTER_LABEL_NAMES[v] == k


# ── Dataset build (3 synthetic PLYs) ─────────────────────────────────────────

@pytest.mark.skipif(
    not DATASET_DIR.exists() or len(list(DATASET_DIR.glob("*.ply"))) < 3,
    reason="Synthetic dataset not available"
)
class TestBuildClusterDataset:
    """Integration test: build GT from a few real synthetic PLYs."""

    @pytest.fixture(scope="class")
    def dataset(self):
        # Only use the first 5 scenes to keep the test fast
        params = GeometricParams()
        plys = sorted(DATASET_DIR.glob("*.ply"))[:5]
        # Temporarily symlink or just call with subset dir — instead,
        # we call _process_scene directly on specific files.
        # Since build_cluster_dataset takes a directory, we use it as-is
        # but verify the output makes sense for 5 scenes' worth.
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for p in plys:
                shutil.copy(p, tmp / p.name)
            X, y, groups = build_cluster_dataset(
                dataset_dir=tmp,
                params=params,
                min_cluster_pts=30,
                purity_threshold=0.6,
                verbose=False,
            )
        return X, y, groups

    def test_X_shape(self, dataset):
        X, y, groups = dataset
        assert X.ndim == 2
        assert X.shape[1] == N_FEATURES

    def test_y_shape(self, dataset):
        X, y, groups = dataset
        assert y.shape == (X.shape[0],)

    def test_groups_shape(self, dataset):
        X, y, groups = dataset
        assert groups.shape == (X.shape[0],)

    def test_at_least_one_cluster(self, dataset):
        X, y, _ = dataset
        assert X.shape[0] >= 1

    def test_no_nan_in_X(self, dataset):
        X, _, _ = dataset
        assert not np.any(np.isnan(X))

    def test_y_values_valid(self, dataset):
        _, y, _ = dataset
        valid_labels = set(CLUSTER_LABEL_MAP.values())   # {1, 2, 3}
        assert set(y.tolist()).issubset(valid_labels)

    def test_groups_bounded(self, dataset):
        _, y, groups = dataset
        # groups should be scene indices 0..4
        assert int(groups.min()) >= 0
        assert int(groups.max()) < 5

    def test_cargo_class_present(self, dataset):
        _, y, _ = dataset
        # Synthetic scenes always have cargo — must appear in GT
        cargo_label = CLUSTER_LABEL_MAP["cargo"]
        assert (y == cargo_label).sum() >= 1, (
            "No cargo clusters found in GT — check pallet→cargo mapping"
        )

    def test_class_distribution_printed(self, dataset, capsys):
        # Just a smoke-test that verbose=True doesn't crash
        params = GeometricParams()
        plys = sorted(DATASET_DIR.glob("*.ply"))[:2]
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for p in plys:
                shutil.copy(p, tmp / p.name)
            build_cluster_dataset(
                dataset_dir=tmp,
                params=params,
                min_cluster_pts=30,
                purity_threshold=0.6,
                verbose=True,
            )


# ── Error cases ───────────────────────────────────────────────────────────────

class TestBuildClusterDatasetErrors:
    def test_empty_dir_raises(self, tmp_path):
        params = GeometricParams()
        with pytest.raises(ValueError, match="No PLY files"):
            build_cluster_dataset(tmp_path, params)

    def test_nonexistent_dir_raises(self):
        params = GeometricParams()
        with pytest.raises((ValueError, FileNotFoundError)):
            build_cluster_dataset(Path("/nonexistent/path"), params)
