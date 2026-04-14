"""
test_cluster_classifier.py — Tests for ClusterClassifier (Paso 4).

Run from src/:
    python3 -m pytest test_cluster_classifier.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from cargo_geometric.cluster_classifier import ClusterClassifier
from cargo_geometric.cluster_features import FEATURE_NAMES

MODELS_DIR = Path(__file__).parent.parent / "models"
LGBM_PICKLE = MODELS_DIR / "cluster_classifier_lgbm.pkl"
RF_PICKLE   = MODELS_DIR / "cluster_classifier_rf.pkl"


# ── Load error handling ───────────────────────────────────────────────────────

class TestLoadErrors:
    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ClusterClassifier.load(tmp_path / "nonexistent.pkl")

    def test_wrong_version_raises_valueerror(self, tmp_path):
        import joblib
        bad_pickle = tmp_path / "bad.pkl"
        joblib.dump({"version": 99, "model": None, "scaler": None,
                     "feature_names": [], "label_map": {}}, bad_pickle)
        with pytest.raises(ValueError, match="unsupported pickle version"):
            ClusterClassifier.load(bad_pickle)

    def test_missing_keys_raises_valueerror(self, tmp_path):
        import joblib
        bad_pickle = tmp_path / "bad.pkl"
        joblib.dump({"version": 1, "model": None}, bad_pickle)
        with pytest.raises(ValueError, match="missing keys"):
            ClusterClassifier.load(bad_pickle)


# ── Round-trip predict tests (require trained pickles) ────────────────────────

@pytest.mark.skipif(
    not LGBM_PICKLE.exists(),
    reason="LGBM pickle not found — run train_cluster_classifier.py first"
)
class TestLGBMPredict:
    @pytest.fixture(scope="class")
    def cc(self):
        return ClusterClassifier.load(LGBM_PICKLE)

    def test_load_succeeds(self, cc):
        assert cc.model is not None
        assert cc.scaler is not None

    def test_feature_names_match(self, cc):
        assert cc.feature_names == FEATURE_NAMES

    def test_label_map_has_cargo(self, cc):
        assert "cargo" in cc.label_map

    def test_inv_label_map_populated(self, cc):
        assert len(cc.inv_label_map) == len(cc.label_map)

    def test_predict_output_shape(self, cc):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, len(FEATURE_NAMES)))
        preds = cc.predict(X)
        assert preds.shape == (5,)
        assert preds.dtype == np.int32

    def test_predict_labels_in_map(self, cc):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, len(FEATURE_NAMES)))
        preds = cc.predict(X)
        valid = set(cc.label_map.values())
        assert set(preds.tolist()).issubset(valid)

    def test_predict_with_proba_shape(self, cc):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((4, len(FEATURE_NAMES)))
        preds, proba = cc.predict_with_proba(X)
        assert preds.shape == (4,)
        assert proba.shape == (4, len(cc.label_map))

    def test_proba_sums_to_1(self, cc):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((10, len(FEATURE_NAMES)))
        _, proba = cc.predict_with_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_empty_input(self, cc):
        X = np.zeros((0, len(FEATURE_NAMES)))
        preds = cc.predict(X)
        assert preds.shape == (0,)

    def test_predict_with_proba_empty_input(self, cc):
        X = np.zeros((0, len(FEATURE_NAMES)))
        preds, proba = cc.predict_with_proba(X)
        assert preds.shape == (0,)
        assert proba.shape[0] == 0


@pytest.mark.skipif(
    not RF_PICKLE.exists(),
    reason="RF pickle not found — run train_cluster_classifier.py first"
)
class TestRFPredict:
    @pytest.fixture(scope="class")
    def cc(self):
        return ClusterClassifier.load(RF_PICKLE)

    def test_load_succeeds(self, cc):
        assert cc.model is not None

    def test_predict_output_valid(self, cc):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((8, len(FEATURE_NAMES)))
        preds = cc.predict(X)
        assert preds.shape == (8,)
        assert set(preds.tolist()).issubset(set(cc.label_map.values()))
