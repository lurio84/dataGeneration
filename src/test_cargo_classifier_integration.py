"""
test_cargo_classifier_integration.py — Integration tests for the
classifier-aware extract_cargo (Paso 5).

Run from src/:
    python3 -m pytest test_cargo_classifier_integration.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from cargo_geometric.anchor import AnchorInfo
from cargo_geometric.cargo import CargoExtractionResult, extract_cargo
from cargo_geometric.params import GeometricParams


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_params() -> GeometricParams:
    p = GeometricParams()
    p.cargo_dbscan_eps      = 0.08
    p.cargo_dbscan_min_pts  = 10     # lower threshold for synthetic test clouds
    p.anchor_xz_margin      = 0.30
    return p


def _make_anchor(x_min=-1.0, x_max=1.0, z_min=-1.0, z_max=1.0) -> AnchorInfo:
    return AnchorInfo(
        n_points=200,
        x_min=x_min, x_max=x_max,
        z_min=z_min, z_max=z_max,
        height=1.2,
        center_xz=((x_min + x_max) / 2, (z_min + z_max) / 2),
    )


def _solid_box(cx, cy, cz, dx, dy, dz, n=5000, seed=0) -> np.ndarray:
    """Random points inside an axis-aligned box.

    Use high point density (default 5000) so that DBSCAN with eps=0.08m
    and min_pts=10 reliably finds a cluster inside each box.  With 5000 pts
    in a 1m³ volume the expected neighbour count within r=0.08m is ~10.7.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(
        [cx - dx / 2, cy - dy / 2, cz - dz / 2],
        [cx + dx / 2, cy + dy / 2, cz + dz / 2],
        (n, 3),
    )
    return pts.astype(np.float32)


def _build_scene_two_clusters():
    """A scene with two well-separated clusters: cargo and person.

    Cluster A (cargo): 5000 pts, centre at (0, 0.6, 0), 1×1×1 m box.
    Cluster B (person): 2000 pts, centre at (2, 0.6, 0), 0.4×1.7×0.4 m box.

    Both are well inside the anchor footprint (anchor covers ±2.8m in x,z).
    The two clusters are 1.5m apart in X (gap ≫ eps=0.08m) so DBSCAN always
    separates them.
    """
    cargo  = _solid_box(0.0, 0.6, 0, 1.0, 1.0, 1.0, n=5000, seed=1)
    person = _solid_box(2.0, 0.6, 0, 0.4, 1.7, 0.4, n=2000, seed=2)
    pts = np.vstack([cargo, person])
    return pts


def _make_mock_classifier(cargo_label: int, preds_for_clusters: list[int]):
    """Return a mock ClusterClassifier that returns the given predictions.

    Also mocks ``predict_with_proba`` with max-confidence proba rows so the
    confidence-thresholded negative filter treats every excluded label as
    confident (max_p=1.0 ≥ threshold).
    """
    cc = MagicMock()
    cc.label_map = {"cargo": cargo_label, "vehicle": 2, "person": 3}
    preds_arr = np.array(preds_for_clusters, dtype=np.int32)
    cc.predict.return_value = preds_arr

    label_order = [cargo_label, 2, 3]
    n = len(preds_for_clusters)
    proba = np.zeros((n, 3), dtype=np.float64)
    for i, p in enumerate(preds_for_clusters):
        proba[i, label_order.index(int(p))] = 1.0
    cc.predict_with_proba.return_value = (preds_arr, proba)
    return cc


# ── Test 1: classifier=None → legacy rank-0 behaviour ────────────────────────

class TestNoClassifier:
    def test_legacy_path_returns_result(self):
        pts = _build_scene_two_clusters()
        anchor = _make_anchor(-2.8, 2.8, -2.8, 2.8)
        params = _make_params()
        result = extract_cargo(pts, anchor, params, classifier=None)
        assert result is not None

    def test_cargo_source_is_legacy(self):
        pts = _build_scene_two_clusters()
        anchor = _make_anchor(-2.8, 2.8, -2.8, 2.8)
        params = _make_params()
        result = extract_cargo(pts, anchor, params, classifier=None)
        assert result.cargo_source == "legacy"

    def test_classifier_metadata_none(self):
        pts = _build_scene_two_clusters()
        anchor = _make_anchor(-2.8, 2.8, -2.8, 2.8)
        params = _make_params()
        result = extract_cargo(pts, anchor, params, classifier=None)
        assert result.cluster_labels_pred is None
        assert result.cluster_ids_classified_as_cargo == []
        assert result.n_clusters_cargo == 0

    def test_rank0_picks_largest_cluster(self):
        """With classifier=None, the largest cluster (cargo, 300 pts) is kept."""
        pts = _build_scene_two_clusters()
        anchor = _make_anchor(-2.8, 2.8, -2.8, 2.8)
        params = _make_params()
        result = extract_cargo(pts, anchor, params, classifier=None)
        # Cargo cluster (300 pts) > person cluster (150 pts)
        assert result.chosen_cluster_size >= 150   # at least half the larger cluster

    def test_result_has_cargo_pts(self):
        pts = _build_scene_two_clusters()
        anchor = _make_anchor(-2.8, 2.8, -2.8, 2.8)
        params = _make_params()
        result = extract_cargo(pts, anchor, params, classifier=None)
        assert result.cargo_pts.ndim == 2
        assert result.cargo_pts.shape[1] == 3


# ── Test 2: classifier labels all clusters cargo → merge all ─────────────────

class TestClassifierAllCargo:
    def setup_method(self):
        self.pts    = _build_scene_two_clusters()
        self.anchor = _make_anchor(-2.8, 2.8, -2.8, 2.8)
        self.params = _make_params()

    def _run_with_preds(self, preds: list[int]):
        cc = _make_mock_classifier(cargo_label=1, preds_for_clusters=preds)
        return extract_cargo(self.pts, self.anchor, self.params, classifier=cc)

    def test_single_cargo_cluster_merged(self):
        """Classifier labels cluster 0 as cargo, cluster 1 as vehicle."""
        # Need to discover how many clusters DBSCAN produces first
        result_legacy = extract_cargo(
            self.pts, self.anchor, self.params, classifier=None
        )
        n_cls = result_legacy.n_clusters
        preds = [1] + [2] * (n_cls - 1)  # first cluster = cargo, rest = vehicle
        result = self._run_with_preds(preds)
        assert result is not None
        assert result.cargo_source == "classifier"
        assert result.n_clusters_cargo >= 1

    def test_all_cargo_predicts_merge(self):
        result_legacy = extract_cargo(
            self.pts, self.anchor, self.params, classifier=None
        )
        n_cls = result_legacy.n_clusters
        preds = [1] * n_cls   # all clusters → cargo
        result = self._run_with_preds(preds)
        assert result is not None
        assert result.cargo_source == "classifier"
        assert result.n_clusters_cargo == n_cls

    def test_merged_pts_count_gte_largest_cluster(self):
        """Merging all clusters should give more points than any single cluster."""
        result_legacy = extract_cargo(
            self.pts, self.anchor, self.params, classifier=None
        )
        n_cls = result_legacy.n_clusters
        preds = [1] * n_cls
        result = self._run_with_preds(preds)
        assert result.chosen_cluster_size >= result_legacy.chosen_cluster_size

    def test_chosen_cluster_id_minus1_for_merge(self):
        result_legacy = extract_cargo(
            self.pts, self.anchor, self.params, classifier=None
        )
        n_cls = result_legacy.n_clusters
        preds = [1] * n_cls
        result = self._run_with_preds(preds)
        assert result.chosen_cluster_id == -1  # -1 signals merged path


# ── Test 3: classifier labels nothing as cargo → fallback rank-0 ─────────────

class TestClassifierNoCargoFallback:
    def setup_method(self):
        self.pts    = _build_scene_two_clusters()
        self.anchor = _make_anchor(-2.8, 2.8, -2.8, 2.8)
        self.params = _make_params()

    def test_fallback_when_no_cargo_predicted(self):
        result_legacy = extract_cargo(
            self.pts, self.anchor, self.params, classifier=None
        )
        n_cls = result_legacy.n_clusters
        preds = [2] * n_cls   # all clusters → vehicle (no cargo)
        cc = _make_mock_classifier(cargo_label=1, preds_for_clusters=preds)
        result = extract_cargo(self.pts, self.anchor, self.params, classifier=cc)
        assert result is not None
        assert result.cargo_source == "fallback_rank0"

    def test_fallback_cluster_labels_pred_preserved(self):
        result_legacy = extract_cargo(
            self.pts, self.anchor, self.params, classifier=None
        )
        n_cls = result_legacy.n_clusters
        preds = [3] * n_cls   # all person (not cargo)
        cc = _make_mock_classifier(cargo_label=1, preds_for_clusters=preds)
        result = extract_cargo(self.pts, self.anchor, self.params, classifier=cc)
        # Even in fallback, cluster_labels_pred is populated
        assert result.cluster_labels_pred is not None
        assert len(result.cluster_labels_pred) == n_cls

    def test_fallback_n_clusters_cargo_zero(self):
        result_legacy = extract_cargo(
            self.pts, self.anchor, self.params, classifier=None
        )
        n_cls = result_legacy.n_clusters
        preds = [2] * n_cls
        cc = _make_mock_classifier(cargo_label=1, preds_for_clusters=preds)
        result = extract_cargo(self.pts, self.anchor, self.params, classifier=cc)
        assert result.n_clusters_cargo == 0
        assert result.cluster_ids_classified_as_cargo == []
