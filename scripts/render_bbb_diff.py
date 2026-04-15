"""Differentiated render: reconstruct per-cluster classification over the BBB
pipeline output and visualise cargo / confident-vehicle / confident-person /
rescued / floor / non-cluster with distinct colours.

This bypasses the label-3 cargo mask saved by run_geometric.py and instead
re-runs clustering+classification to colour every point by its cluster role.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d
import joblib

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from cargo_geometric.anchor import find_anchor, points_inside_anchor
from cargo_geometric.floor import preprocess, remove_floor
from cargo_geometric.params import GeometricParams
from cargo_geometric.cluster_features import extract_features_batch


COLORS = {
    "floor":         (0.60, 0.60, 0.60),   # gris claro
    "non_cluster":   (0.25, 0.25, 0.25),   # gris oscuro
    "cargo":         (0.90, 0.15, 0.15),   # rojo
    "vehicle":       (0.15, 0.35, 0.90),   # azul (excluido confiado)
    "person":        (0.15, 0.75, 0.35),   # verde (excluido confiado)
    "rescued":       (0.95, 0.60, 0.10),   # naranja (rescatado por umbral)
}

LBL_CARGO, LBL_VEHICLE, LBL_PERSON = 1, 2, 3
CONF_THRESHOLD = 0.75


def render_one(ply_path: Path, out_png: Path, model, scaler, cargo_col,
               vehicle_col, person_col):
    params = GeometricParams()

    pts_vox = preprocess(ply_path, params)
    fr = remove_floor(pts_vox, params)
    pts_nf = fr.pts
    floor_y = fr.floor_y

    # Floor points (removed ones) are not in fr.pts. For visual context, read
    # raw voxelised cloud and mark points below floor_band as floor.
    all_pts = pts_vox  # preprocess already voxelises
    floor_mask_all = all_pts[:, 1] < (floor_y + 0.03)

    # Non-floor subset (matches fr.pts approximately)
    # Build role array for all_pts, default to "non_cluster"
    roles = np.full(len(all_pts), -1, dtype=np.int8)  # -1 → non_cluster
    roles[floor_mask_all] = 0  # 0 → floor

    anc = find_anchor(pts_nf, floor_y, params)
    if anc is not None:
        in_a = points_inside_anchor(pts_nf, anc, params.anchor_xz_margin)
        sub = pts_nf[in_a]

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

        if labs.max() >= 0:
            n_cl = int(labs.max()) + 1
            cluster_pts_list = [sub[labs == i] for i in range(n_cl)]
            X = extract_features_batch(cluster_pts_list, floor_y, anc)
            Xs = scaler.transform(X)
            proba = model.predict_proba(Xs)
            preds = model.predict(Xs)

            # Map each sub point back to its position in all_pts via KDTree
            from scipy.spatial import cKDTree
            tree = cKDTree(all_pts)
            _, nn = tree.query(sub, k=1, workers=1)

            for cid in range(n_cl):
                mask_in_sub = labs == cid
                if not mask_in_sub.any():
                    continue
                max_p = float(proba[cid].max())
                pred_label = int(preds[cid])

                if pred_label == LBL_CARGO:
                    role = 1  # cargo
                elif pred_label == LBL_VEHICLE:
                    role = 2 if max_p >= CONF_THRESHOLD else 4  # vehicle / rescued
                elif pred_label == LBL_PERSON:
                    role = 3 if max_p >= CONF_THRESHOLD else 4  # person / rescued
                else:
                    role = 1

                global_idx = nn[mask_in_sub]
                roles[global_idx] = role

    rgb = np.zeros((len(all_pts), 3), dtype=np.float32)
    role_to_color = {
        -1: COLORS["non_cluster"],
         0: COLORS["floor"],
         1: COLORS["cargo"],
         2: COLORS["vehicle"],
         3: COLORS["person"],
         4: COLORS["rescued"],
    }
    counts = {}
    for r, c in role_to_color.items():
        m = roles == r
        rgb[m] = c
        counts[r] = int(m.sum())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    order = np.argsort(all_pts[:, 1])
    axes[0].scatter(all_pts[order, 0], all_pts[order, 2], c=rgb[order], s=0.6, marker=".")
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Z (m)")
    axes[0].set_title("Top-down (XZ)"); axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    order = np.argsort(all_pts[:, 2])
    axes[1].scatter(all_pts[order, 0], all_pts[order, 1], c=rgb[order], s=0.6, marker=".")
    axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")
    axes[1].set_title("Front (XY)"); axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    order = np.argsort(all_pts[:, 0])
    axes[2].scatter(all_pts[order, 2], all_pts[order, 1], c=rgb[order], s=0.6, marker=".")
    axes[2].set_xlabel("Z (m)"); axes[2].set_ylabel("Y (m)")
    axes[2].set_title("Side (ZY)"); axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker='o', lw=0, markerfacecolor=c, markeredgecolor='none',
                   markersize=9, label=f"{k} ({counts[r]})")
        for k, r, c in [
            ("cargo (red)", 1, COLORS["cargo"]),
            ("vehicle excl (blue)", 2, COLORS["vehicle"]),
            ("person excl (green)", 3, COLORS["person"]),
            ("rescued (orange)", 4, COLORS["rescued"]),
            ("floor", 0, COLORS["floor"]),
            ("non-cluster", -1, COLORS["non_cluster"]),
        ]
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85)

    fig.suptitle(f"{ply_path.stem}   (CONF_THRESHOLD={CONF_THRESHOLD})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    pkl = _ROOT / "models" / "cluster_classifier_lgbm.pkl"
    payload = joblib.load(pkl)
    model, scaler = payload["model"], payload["scaler"]
    classes = list(model.classes_)
    cargo_col = classes.index(LBL_CARGO)
    vehicle_col = classes.index(LBL_VEHICLE)
    person_col = classes.index(LBL_PERSON)

    in_dir = _ROOT / "output" / "bbb_vox035"
    out_dir = _ROOT / "output" / "geo_verify_v3_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(in_dir.glob("bbb_*_vox035.ply")):
        out = out_dir / (p.stem + "_diff.png")
        render_one(p, out, model, scaler, cargo_col, vehicle_col, person_col)
        print(f"  wrote {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
