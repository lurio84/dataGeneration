"""
Stage 3 — cargo extraction.

Given the anchor (XZ bbox of the dominant standing blob) and the floor,
isolate the *cargo* points by:

  1. Restricting to the anchor's XZ footprint.
  2. Removing the floor band (already done upstream).
  3. 3D-DBSCAN of the remaining points.
  4a. (Legacy / no classifier) Keeping the largest cluster — the cargo bulto.
  4b. (With classifier) Negative-filter policy: all clusters whose predicted
      label is NOT ``person`` nor ``vehicle`` are retained as cargo.

When ``classifier`` is None, the behaviour is identical to the original
rank-0 heuristic (backwards-compatible with all existing tests).

When ``classifier`` is provided:
  - Each cluster's features are extracted and classified.
  - **Negative-filter policy**: clusters labelled ``person`` or ``vehicle``
    are *excluded*; the remainder (unlabelled or labelled ``cargo``) are
    merged into the cargo mask.  This is permissive by design — unknown
    classes default to cargo rather than being silently dropped.
  - If all clusters are excluded (scene is all persons/vehicles), falls back
    to rank-0 with ``cargo_source="fallback_rank0"``.
  - ``CargoExtractionResult`` gains extra metadata fields for diagnostics.

Sub-cluster info (``cargo_dbscan_min_pts`` / ``cargo_dbscan_eps``) is also
exposed so a future stage can split per-box if needed without changing the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

from cargo_geometric.params import GeometricParams
from cargo_geometric.anchor import AnchorInfo, points_inside_anchor
from cargo_geometric.cluster_features import extract_features_batch

if TYPE_CHECKING:
    from cargo_geometric.cluster_classifier import ClusterClassifier


@dataclass
class CargoExtractionResult:
    cargo_pts: np.ndarray              # (M, 3) float32 — cargo points in world
    cargo_global_idx: np.ndarray       # (M,) int — indices into pts_no_floor
    n_total_in_anchor: int             # how many non-floor points were in anchor footprint
    n_clusters: int                    # how many 3D DBSCAN clusters found
    chosen_cluster_id: int             # cluster_dbscan id of the kept cluster
                                       #   (rank-0 path) or -1 (merge path)
    chosen_cluster_size: int           # size of the kept cluster (rank-0) or
                                       #   total merged cargo points (ML path)

    # ── ML classifier metadata (None when classifier was not used) ──────────
    cargo_source: str = "legacy"
    # "legacy"         — classifier=None (original rank-0 behaviour)
    # "classifier"     — ML classifier selected cargo cluster(s)
    # "fallback_rank0" — ML classifier found no cargo; fell back to rank-0

    cargo_policy: str = "rank0"
    # "rank0"           — no classifier; keep largest cluster
    # "negative_filter" — ML classifier used; cargo = anchor \ (person ∪ vehicle)

    cluster_labels_pred: np.ndarray | None = None
    # (n_clusters,) int32 — predicted class for each DBSCAN cluster;
    # None when classifier was not used.

    cluster_ids_classified_as_cargo: list[int] = field(default_factory=list)
    # DBSCAN cluster IDs retained as cargo (negative-filter policy: all IDs
    # whose predicted label is NOT person nor vehicle, or whose vehicle/person
    # probability is below the confidence threshold).

    n_clusters_cargo: int = 0
    # Number of clusters retained as cargo (0 = fallback).

    cluster_proba: np.ndarray | None = None
    # (n_clusters, n_classes) float64 — class probabilities from the classifier.
    # None when classifier was not used.  Column order matches the model's
    # ``classes_`` attribute (see ClusterClassifier.model.classes_).


def extract_cargo(
    pts_no_floor: np.ndarray,
    anchor: AnchorInfo,
    params: GeometricParams,
    classifier: "ClusterClassifier | None" = None,
) -> CargoExtractionResult | None:
    """Return the cargo cluster(s) inside the anchor footprint, or None.

    Parameters
    ----------
    pts_no_floor : (N, 3) float32 — all points with floor removed
    anchor       : XZ bounding box of the dominant blob
    params       : pipeline parameters
    classifier   : optional ClusterClassifier; if None → rank-0 legacy path
    """
    # ── Restrict to anchor footprint ─────────────────────────────────────────
    in_anchor = points_inside_anchor(pts_no_floor, anchor, params.anchor_xz_margin)
    sub_idx = np.where(in_anchor)[0]
    if len(sub_idx) < params.cargo_dbscan_min_pts:
        return None
    sub_pts = pts_no_floor[sub_idx]

    # ── 3D DBSCAN ────────────────────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_pts.astype(np.float64))
    dbscan_labels = np.asarray(
        pcd.cluster_dbscan(
            eps=params.cargo_dbscan_eps,
            min_points=params.cargo_dbscan_min_pts,
            print_progress=False,
        ),
        dtype=np.int32,
    )

    if dbscan_labels.max() < 0:
        return None

    n_clusters = int(dbscan_labels.max()) + 1
    sizes = np.bincount(dbscan_labels[dbscan_labels >= 0])

    # ── Gather clusters ───────────────────────────────────────────────────────
    clusters_pts = [sub_pts[dbscan_labels == cid] for cid in range(n_clusters)]

    # ── Rama A: legacy / no classifier ───────────────────────────────────────
    if classifier is None:
        best = int(np.argmax(sizes))
        keep = dbscan_labels == best
        return CargoExtractionResult(
            cargo_pts=sub_pts[keep],
            cargo_global_idx=sub_idx[keep],
            n_total_in_anchor=int(len(sub_idx)),
            n_clusters=n_clusters,
            chosen_cluster_id=best,
            chosen_cluster_size=int(sizes[best]),
            cargo_source="legacy",
            cargo_policy="rank0",
            cluster_labels_pred=None,
            cluster_ids_classified_as_cargo=[],
            n_clusters_cargo=0,
        )

    # ── Rama B: ML classifier — negative-filter policy ────────────────────────
    X = extract_features_batch(clusters_pts, floor_y=_infer_floor_y(pts_no_floor),
                               anchor=anchor)
    preds, proba = classifier.predict_with_proba(X)  # (n,) int32, (n, n_classes)

    cargo_ids = _apply_negative_filter(
        preds,
        classifier.label_map,
        cluster_proba=proba,
        confidence_threshold=params.negative_filter_confidence,
    )

    if len(cargo_ids) == 0:
        # All clusters excluded (all labeled person/vehicle with high confidence)
        # — fall back to rank-0
        best = int(np.argmax(sizes))
        keep = dbscan_labels == best
        return CargoExtractionResult(
            cargo_pts=sub_pts[keep],
            cargo_global_idx=sub_idx[keep],
            n_total_in_anchor=int(len(sub_idx)),
            n_clusters=n_clusters,
            chosen_cluster_id=best,
            chosen_cluster_size=int(sizes[best]),
            cargo_source="fallback_rank0",
            cargo_policy="negative_filter",
            cluster_labels_pred=preds,
            cluster_ids_classified_as_cargo=[],
            n_clusters_cargo=0,
            cluster_proba=proba,
        )

    # Merge all retained clusters
    cargo_mask = np.zeros(len(sub_pts), dtype=bool)
    for cid in cargo_ids:
        cargo_mask |= dbscan_labels == cid

    cargo_pts       = sub_pts[cargo_mask]
    cargo_global    = sub_idx[cargo_mask]
    total_cargo_pts = int(cargo_mask.sum())

    return CargoExtractionResult(
        cargo_pts=cargo_pts,
        cargo_global_idx=cargo_global,
        n_total_in_anchor=int(len(sub_idx)),
        n_clusters=n_clusters,
        chosen_cluster_id=-1,                 # -1 signals merged multi-cluster result
        chosen_cluster_size=total_cargo_pts,
        cargo_source="classifier",
        cargo_policy="negative_filter",
        cluster_labels_pred=preds,
        cluster_ids_classified_as_cargo=cargo_ids,
        n_clusters_cargo=len(cargo_ids),
        cluster_proba=proba,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_negative_filter(
    cluster_labels_pred: np.ndarray,
    label_map: dict[str, int],
    cluster_proba: np.ndarray | None = None,
    confidence_threshold: float = 0.75,
) -> list[int]:
    """Return cluster IDs whose predicted label is not ``person`` or ``vehicle``.

    Implements the confidence-thresholded negative-filter policy:

    * A cluster is *excluded* (treated as non-cargo) only when:
      (a) its predicted label is ``person`` or ``vehicle``, AND
      (b) the classifier's maximum class probability for that cluster is
          >= ``confidence_threshold``.

    * If ``cluster_proba`` is None the threshold is not applied and the
      original hard-label behaviour is used (backwards-compatible with the
      legacy rank-0 path and with tests that do not supply probabilities).

    * Unlabelled classes (keys absent from ``label_map``) are treated
      permissively — they are *not* excluded regardless of probability.

    Parameters
    ----------
    cluster_labels_pred : (N,) int array — predicted label per cluster
    label_map           : mapping from class name to integer label
                          (e.g. ``{"cargo": 1, "vehicle": 2, "person": 3}``)
    cluster_proba       : (N, n_classes) float64 array of class probabilities,
                          or None to use hard-label exclusion.
    confidence_threshold : float — minimum max-probability for a
                          person/vehicle prediction to trigger exclusion.
                          Default 0.75.

    Returns
    -------
    list of cluster IDs (0-indexed) to retain as cargo.
    """
    excluded_labels: set[int] = set()
    for key in ("person", "vehicle"):
        lbl = label_map.get(key)
        if lbl is not None:
            excluded_labels.add(int(lbl))

    retained: list[int] = []
    for cid, pred in enumerate(cluster_labels_pred):
        if int(pred) not in excluded_labels:
            retained.append(cid)
            continue
        # Predicted as person or vehicle — only exclude if confident enough.
        if cluster_proba is None:
            # No probabilities available: hard exclusion (legacy behaviour).
            continue
        max_p = float(cluster_proba[cid].max())
        if max_p < confidence_threshold:
            # Uncertain prediction: keep cluster as cargo (permissive).
            retained.append(cid)
        # else: confident exclusion — do not retain.
    return retained


def _infer_floor_y(pts_no_floor: np.ndarray) -> float:
    """Approximate floor_y as the minimum Y in pts_no_floor.

    The exact floor_y is slightly below the minimum surviving point
    (because the floor_band removed the floor region).  For feature
    extraction this approximation is acceptable since ``height_above_floor``
    features use floor_y as a reference only.

    In practice the caller should pass floor_y from FloorResult when possible.
    The signature of extract_cargo does not carry floor_y to avoid API churn.
    The features that depend on floor_y are ``height_above_floor_min/max``,
    which will be slightly over-estimated by ≤ floor_band (~0.05 m).
    """
    if len(pts_no_floor) == 0:
        return 0.0
    return float(pts_no_floor[:, 1].min())
