"""
Stage 3 — cargo extraction.

Given the anchor (XZ bbox of the dominant standing blob) and the floor,
isolate the *cargo* points by:

  1. Restricting to the anchor's XZ footprint.
  2. Removing the floor band (already done upstream).
  3. 3D-DBSCAN of the remaining points.
  4a. (Legacy / no classifier) Keeping the largest cluster — the cargo bulto.
  4b. (With classifier) Running the cluster-level ML classifier; merging all
      clusters predicted as ``cargo`` into a single result.

When ``classifier`` is None, the behaviour is identical to the original
rank-0 heuristic (backwards-compatible with all existing tests).

When ``classifier`` is provided:
  - Each cluster's features are extracted and classified.
  - Clusters labelled ``cargo`` are merged into a single result.
  - If no cluster is labelled ``cargo``, falls back to rank-0 with
    ``cargo_source="fallback_rank0"``.
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

    cluster_labels_pred: np.ndarray | None = None
    # (n_clusters,) int32 — predicted class for each DBSCAN cluster;
    # None when classifier was not used.

    cluster_ids_classified_as_cargo: list[int] = field(default_factory=list)
    # DBSCAN cluster IDs that the classifier labelled as cargo.

    n_clusters_cargo: int = 0
    # Number of clusters the classifier labelled as cargo (0 = fallback).


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
            cluster_labels_pred=None,
            cluster_ids_classified_as_cargo=[],
            n_clusters_cargo=0,
        )

    # ── Rama B: ML classifier ─────────────────────────────────────────────────
    X = extract_features_batch(clusters_pts, floor_y=_infer_floor_y(pts_no_floor),
                               anchor=anchor)
    preds = classifier.predict(X)   # (n_clusters,) int32

    cargo_label = classifier.label_map.get("cargo")
    if cargo_label is None:
        # Label map doesn't have "cargo" key — fallback
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
            cluster_labels_pred=preds,
            cluster_ids_classified_as_cargo=[],
            n_clusters_cargo=0,
        )

    cargo_ids = [cid for cid, p in enumerate(preds) if p == cargo_label]

    if len(cargo_ids) == 0:
        # Classifier found no cargo — fall back to rank-0
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
            cluster_labels_pred=preds,
            cluster_ids_classified_as_cargo=[],
            n_clusters_cargo=0,
        )

    # Merge all cargo clusters into one result
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
        cluster_labels_pred=preds,
        cluster_ids_classified_as_cargo=cargo_ids,
        n_clusters_cargo=len(cargo_ids),
    )


# ── Internal helper ───────────────────────────────────────────────────────────

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
