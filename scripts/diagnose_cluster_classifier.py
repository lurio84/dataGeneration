"""Diagnose cluster classifier: is it really domain gap, or prior/GT/features?

Runs four checks:
  1. Class prior in synthetic training set + P(cargo) histogram on synthetic.
  2. Cluster GT purity stats (Paso 0 from plan).
  3. BBB vs synthetic feature distribution shift (per-feature).
  4. BBB cluster extraction + P(cargo) histogram.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import open3d as o3d

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from cargo_geometric.anchor import find_anchor, points_inside_anchor
from cargo_geometric.cluster_features import (
    FEATURE_NAMES,
    extract_cluster_features,
)
from cargo_geometric.cluster_gt import (
    CLUSTER_LABEL_MAP,
    CLUSTER_LABEL_NAMES,
    _read_ply_xyz_label,
    _RAW_TO_CLUSTER,
    _FALLBACK_LABEL,
)
from cargo_geometric.floor import preprocess, remove_floor
from cargo_geometric.params import GeometricParams
from scipy.spatial import cKDTree


def banner(t):
    print("\n" + "=" * 72)
    print(f"  {t}")
    print("=" * 72)


def extract_clusters_with_purity(
    ply_path: Path, params: GeometricParams, min_cluster_pts: int = 30
):
    """Return list of dicts: pts, label_counts, total_pts, features."""
    try:
        raw_pts, raw_labels = _read_ply_xyz_label(ply_path)
        has_labels = int(raw_labels.max()) > 0
    except ValueError:
        raw_pts, raw_labels, has_labels = None, None, False
    if has_labels:
        mapped = np.array(
            [_RAW_TO_CLUSTER.get(int(l), _FALLBACK_LABEL) for l in raw_labels],
            dtype=np.int32,
        )
    else:
        mapped = None

    try:
        pts_vox = preprocess(ply_path, params)
        fr = remove_floor(pts_vox, params)
    except RuntimeError:
        return None, None
    anchor = find_anchor(fr.pts, fr.floor_y, params)
    if anchor is None:
        return None, None
    in_anc = points_inside_anchor(fr.pts, anchor, params.anchor_xz_margin)
    sub = fr.pts[in_anc]
    if len(sub) < params.cargo_dbscan_min_pts:
        return None, None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
    labs = np.asarray(
        pcd.cluster_dbscan(
            eps=params.cargo_dbscan_eps,
            min_points=params.cargo_dbscan_min_pts,
            print_progress=False,
        ),
        dtype=np.int32,
    )
    if labs.max() < 0:
        return None, None

    pt_labels = None
    if mapped is not None and raw_pts is not None:
        tree = cKDTree(raw_pts)
        _, nn = tree.query(sub, k=1, workers=1)
        pt_labels = mapped[nn]

    clusters = []
    for cid in range(int(labs.max()) + 1):
        m = labs == cid
        n = int(m.sum())
        if n < min_cluster_pts:
            continue
        cpts = sub[m]
        feat = extract_cluster_features(cpts, fr.floor_y, anchor)
        info = {"n": n, "feat": feat}
        if pt_labels is not None:
            cl = pt_labels[m]
            counts = np.bincount(
                cl, minlength=max(CLUSTER_LABEL_MAP.values()) + 1
            )
            info["counts"] = counts
            info["top_label"] = int(counts.argmax())
            info["purity"] = float(counts.max()) / n
        clusters.append(info)
    return clusters, anchor


def main():
    params = GeometricParams()

    # ── 1. Load model ────────────────────────────────────────────────────────
    pkl = _ROOT / "models" / "cluster_classifier_lgbm.pkl"
    if not pkl.exists():
        pkl = _ROOT / "models" / "cluster_classifier_rf.pkl"
    payload = joblib.load(pkl)
    model = payload["model"]
    scaler = payload["scaler"]
    label_map = payload["label_map"]
    inv_lbl = {v: k for k, v in label_map.items()}
    cargo_id = label_map["cargo"]
    cargo_col = list(model.classes_).index(cargo_id)
    print(f"Loaded model: {pkl.name}")
    print(f"  classes = {list(model.classes_)}  → cargo col {cargo_col}")

    # ── 2. Scan synthetic dataset ────────────────────────────────────────────
    banner("1. SYNTHETIC: class prior, cluster GT purity, P(cargo) on val")
    synth_dir = _ROOT / "output" / "dataset"
    plys = sorted(synth_dir.glob("*.ply"))
    print(f"Scanning {len(plys)} synthetic PLYs ...")

    all_feats = []
    all_gt_labels = []
    all_scene_ids = []
    all_purities = []
    all_majority_labels = []

    for sid, p in enumerate(plys):
        clusters, _ = extract_clusters_with_purity(p, params)
        if clusters is None:
            continue
        for c in clusters:
            all_feats.append(c["feat"])
            all_scene_ids.append(sid)
            purity = c.get("purity", 0.0)
            all_purities.append(purity)
            top = c["top_label"]
            all_majority_labels.append(top)
            gt = top if purity >= 0.6 else _FALLBACK_LABEL
            all_gt_labels.append(gt)

    X = np.vstack(all_feats) if all_feats else np.zeros((0, 23))
    y = np.array(all_gt_labels, dtype=np.int32)
    groups = np.array(all_scene_ids, dtype=np.int32)
    purities = np.array(all_purities, dtype=np.float64)
    majorities = np.array(all_majority_labels, dtype=np.int32)
    print(f"  extracted {len(y)} clusters across {len(np.unique(groups))} scenes")

    # Class prior
    print("\nClass prior (after GT rules):")
    for lid, lname in sorted(inv_lbl.items()):
        n = int((y == lid).sum())
        print(f"  {lname:>8}: {n:5d}  ({n/len(y)*100:5.1f}%)")

    # Purity stats
    print("\nPurity stats (majority fraction per cluster):")
    print(f"  mean={purities.mean():.3f}  median={np.median(purities):.3f}")
    print(f"  p25={np.percentile(purities,25):.3f}  "
          f"p10={np.percentile(purities,10):.3f}")
    print(f"  purity<0.6: {int((purities<0.6).sum())} "
          f"({(purities<0.6).mean()*100:.1f}%)  → relabeled vehicle")
    print(f"  purity<0.8: {int((purities<0.8).sum())} "
          f"({(purities<0.8).mean()*100:.1f}%)")

    # For clusters whose *raw majority* is cargo, how pure are they really?
    cargo_mask = majorities == cargo_id
    if cargo_mask.any():
        pc = purities[cargo_mask]
        print(f"\nPurity of clusters whose majority is cargo (n={cargo_mask.sum()}):")
        print(f"  mean={pc.mean():.3f}  median={np.median(pc):.3f}  "
              f"p10={np.percentile(pc,10):.3f}")

    # P(cargo) histogram on the synthetic training set itself (held-out approx)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)
    pc_all = proba[:, cargo_col]
    print("\nP(cargo) distribution on ALL synthetic clusters:")
    edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.001]
    counts, _ = np.histogram(pc_all, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        bar = "█" * int(40 * c / max(counts.max(), 1))
        print(f"  [{lo:.2f}, {hi:.2f}): {c:5d}  {bar}")

    # Split by TRUE label
    print("\nP(cargo) by true GT label (mean, p10, p90):")
    for lid, lname in sorted(inv_lbl.items()):
        msk = y == lid
        if msk.sum() == 0:
            continue
        pp = pc_all[msk]
        print(f"  {lname:>8} (n={msk.sum():4d}): "
              f"mean={pp.mean():.3f}  "
              f"p10={np.percentile(pp,10):.3f}  "
              f"p90={np.percentile(pp,90):.3f}")

    # ── 3. BBB real ──────────────────────────────────────────────────────────
    banner("2. BBB REAL: features, P(cargo), feature shift")
    bbb_dir = _ROOT / "output" / "bbb_vox035"
    bbb_plys = sorted(bbb_dir.glob("*.ply"))
    bbb_feats_all = []
    bbb_proba_all = []
    for p in bbb_plys:
        clusters, _ = extract_clusters_with_purity(p, params)
        if clusters is None:
            print(f"  {p.name}: skipped")
            continue
        Xb = np.vstack([c["feat"] for c in clusters])
        Xbs = scaler.transform(Xb)
        pb = model.predict_proba(Xbs)[:, cargo_col]
        bbb_feats_all.append(Xb)
        bbb_proba_all.append(pb)
        print(f"  {p.name}: {len(clusters):3d} clusters, "
              f"P(cargo)>0.5: {int((pb>0.5).sum())}, "
              f">0.9: {int((pb>0.9).sum())}, "
              f">0.99: {int((pb>0.99).sum())}")

    Xbbb = np.vstack(bbb_feats_all) if bbb_feats_all else np.zeros((0, 23))
    pbbb = np.concatenate(bbb_proba_all) if bbb_proba_all else np.zeros(0)

    print(f"\nTotal BBB clusters: {len(pbbb)}")
    print("P(cargo) distribution on BBB:")
    counts, _ = np.histogram(pbbb, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        bar = "█" * int(40 * c / max(counts.max(), 1))
        print(f"  [{lo:.2f}, {hi:.2f}): {c:5d}  {bar}")

    # ── 4. Feature distribution shift ───────────────────────────────────────
    banner("3. FEATURE SHIFT: synthetic vs BBB (per feature)")
    print(f"{'feature':>36}  {'synth med':>10}  {'bbb med':>10}  "
          f"{'ratio':>8}  {'z (bbb vs synth)':>18}")
    for i, name in enumerate(FEATURE_NAMES):
        s = X[:, i]
        b = Xbbb[:, i] if len(Xbbb) else np.array([0.0])
        sm = np.median(s)
        bm = np.median(b)
        ratio = bm / sm if abs(sm) > 1e-9 else float("inf")
        sd = s.std() + 1e-9
        z = (bm - s.mean()) / sd
        flag = "  ⚠" if abs(z) > 2.0 else ""
        print(f"{name:>36}  {sm:10.3f}  {bm:10.3f}  "
              f"{ratio:8.3f}  {z:18.2f}{flag}")


if __name__ == "__main__":
    main()
