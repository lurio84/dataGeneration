"""
validate_synthetic.py — Visual validation of the per-point classifier on synthetic scenes.

Loads a trained model, predicts on never-seen validation PLYs, and renders
side-by-side PNGs (GT / Prediction / Diff) plus a metrics summary.

Usage (from src/):
    python3 ../scripts/validate_synthetic.py \
        --model ../models/classifier_lgbm.pkl \
        --data ../output/validation_A ../output/validation_B \
        --out ../output/validation_visual
"""

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from classifier.features import extract_features, FEATURE_NAMES
from classifier.train import _read_ply_xyz_label
from generate_dataset import LABEL
from utils.preview_grid import (
    draw_scene, label_rgba, legend_patches, subsample, MAX_PTS,
    LABEL_COLORS, LABEL_NAMES,
)

CLASS_NAMES = list(LABEL.keys())
CLASS_IDS = list(LABEL.values())


def _render_comparison(pts, y_true, y_pred, scene_name, out_dir):
    """Render a 3-row × 3-col PNG: GT / Pred / Diff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mc

    # Subsample for rendering
    if len(pts) > MAX_PTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_PTS, replace=False)
        pts = pts[idx]
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    views = ["top", "front", "side"]

    # Row 0: Ground Truth
    for col, view in enumerate(views):
        draw_scene(axes[0, col], pts, y_true, view=view, point_size=2.0)
        axes[0, col].set_title(f"GT — {view}", fontsize=9)

    # Row 1: Prediction
    for col, view in enumerate(views):
        draw_scene(axes[1, col], pts, y_pred, view=view, point_size=2.0)
        axes[1, col].set_title(f"PRED — {view}", fontsize=9)

    # Row 2: Diff (correct=grey, wrong=bright by true label)
    correct = y_true == y_pred
    diff_labels = np.where(correct, -1, y_true)  # -1 = correct

    for col, view in enumerate(views):
        ax = axes[2, col]

        def coords(p, v):
            if v == "top":   return p[:, 0], p[:, 2]
            if v == "front": return p[:, 0], p[:, 1]
            return p[:, 2], p[:, 1]

        xc, yc = coords(pts, view)

        # Correct points: faint grey
        if correct.any():
            ax.scatter(xc[correct], yc[correct],
                       c="#C0C0C0", s=0.5, linewidths=0, alpha=0.2)

        # Wrong points: bright, colored by TRUE class
        wrong = ~correct
        if wrong.any():
            rgba = label_rgba(y_true[wrong], alpha=1.0)
            ax.scatter(xc[wrong], yc[wrong], c=rgba, s=6.0, linewidths=0)

        ax.set_aspect("equal")
        ax.tick_params(labelsize=4, pad=1)
        ax.grid(True, lw=0.2, alpha=0.4)
        ax.set_title(f"DIFF — {view}", fontsize=9)

    # Accuracy in title
    acc = correct.mean() * 100
    n_wrong = (~correct).sum()
    fig.suptitle(f"{scene_name}   acc={acc:.1f}%  errors={n_wrong}",
                 fontsize=11, fontweight="bold")

    fig.legend(
        handles=legend_patches(), loc="lower center", ncol=7,
        fontsize=8, title="Labels", title_fontsize=9,
        framealpha=0.9, markerscale=2,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_dir / f"{scene_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def validate(model_path, data_dirs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(model_path)
    model = payload["model"]
    scaler = payload["scaler"]
    saved_features = payload.get("feature_names", [])

    if len(saved_features) != len(FEATURE_NAMES):
        print(f"WARNING: model has {len(saved_features)} features, "
              f"code has {len(FEATURE_NAMES)}. Results may be wrong.",
              file=sys.stderr)

    print(f"Model: {model_path}")
    print(f"Features: {len(saved_features)}")
    print(f"Output: {out_dir}\n")

    all_true, all_pred = [], []
    rows = []

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        plys = sorted(data_dir.glob("*.ply"))
        if not plys:
            print(f"  No PLYs in {data_dir}, skipping.")
            continue

        batch_name = data_dir.name
        print(f"── Batch: {batch_name} ({len(plys)} scenes) ──")

        for ply in plys:
            pts, labels = _read_ply_xyz_label(ply)
            mask = labels != 255
            pts, labels = pts[mask], labels[mask]

            feats = extract_features(pts)
            feats_scaled = scaler.transform(feats).astype(np.float32)
            preds = model.predict(feats_scaled).astype(np.int32)
            labels_i32 = labels.astype(np.int32)

            acc = (preds == labels_i32).mean()
            f1 = f1_score(labels_i32, preds, average="macro", zero_division=0)

            # Per-class F1
            report = classification_report(
                labels_i32, preds,
                labels=CLASS_IDS, target_names=CLASS_NAMES,
                zero_division=0, output_dict=True,
            )
            per_class_f1 = {c: report[c]["f1-score"] for c in CLASS_NAMES if c in report}

            scene_name = f"{batch_name}_{ply.stem}"
            print(f"  {scene_name}: acc={acc:.3f}  F1-macro={f1:.3f}  "
                  + "  ".join(f"{c}={v:.2f}" for c, v in per_class_f1.items() if v > 0))

            _render_comparison(pts, labels_i32, preds, scene_name, out_dir)

            all_true.append(labels_i32)
            all_pred.append(preds)

            row = {"scene": scene_name, "n_points": len(pts),
                   "accuracy": f"{acc:.4f}", "f1_macro": f"{f1:.4f}"}
            for c in CLASS_NAMES:
                row[f"f1_{c}"] = f"{per_class_f1.get(c, 0):.4f}"
            rows.append(row)

    # Global metrics
    if all_true:
        y_all = np.concatenate(all_true)
        p_all = np.concatenate(all_pred)
        print(f"\n── Global ({len(y_all):,} points) ──")
        print(classification_report(
            y_all, p_all, labels=CLASS_IDS,
            target_names=CLASS_NAMES, zero_division=0,
        ))

        # Save CSV
        csv_path = out_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(f1_row for f1_row in rows)
        print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate per-point classifier on synthetic")
    parser.add_argument("--model", required=True, help="Path to classifier .pkl")
    parser.add_argument("--data", nargs="+", required=True,
                        help="Directories with validation PLYs")
    parser.add_argument("--out", default="../output/validation_visual")
    args = parser.parse_args()
    validate(args.model, args.data, args.out)


if __name__ == "__main__":
    main()
