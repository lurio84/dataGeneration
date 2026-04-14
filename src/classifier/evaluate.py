"""
evaluate.py — Evaluate RF and/or LightGBM classifiers on the labelled dataset.

Usage (from src/):
    python3 classifier/evaluate.py [--model rf|lgbm|both] [--data ../output/dataset]
    python3 classifier/evaluate.py --model lgbm --visualize-predictions

Outputs (saved to ../output/classifier_eval/):
  - confusion_matrix_{model}.png  — 5×5 normalised confusion matrix
  - feature_importance_rf.png     — RF feature importance bar chart
  - cv_comparison.png             — RF vs LightGBM F1 comparison table/bar
  - heldout_preds_{model}/        — per-scene 3-view PNGs of held-out fold 0
                                    predictions (with --visualize-predictions)
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from classifier.features import FEATURE_NAMES
from classifier.train import load_dataset
from generate_dataset import LABEL

CLASS_NAMES = list(LABEL.keys())  # ['floor','cargo','vehicle','person','pallet']
CLASS_IDS = list(LABEL.values())


def _plot_confusion_matrix(cm_norm: np.ndarray, title: str, out_path: "Path") -> None:
    """Save a normalised 5×5 confusion matrix as a PNG heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_feature_importance(rf_model, out_path: "Path") -> None:
    """Save a horizontal bar chart of RF feature importances (mean decrease in impurity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in idx]
    sorted_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_names[::-1], sorted_vals[::-1])
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("RF Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_comparison(results: dict, out_path):
    """Bar chart comparing F1-macro of RF vs LGBM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results.keys())
    f1s = [results[m]["f1_macro"] for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, f1s, color=["steelblue", "orange"][:len(models)])
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.4f}", ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-macro (CV mean)")
    ax.set_title("RF vs LightGBM — scene-level CV F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _render_from_memory(pts: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    """Render a point cloud + labels as a 3-view PNG (top/front/side).

    Reuses draw_scene + legend_patches from preview_grid so the layout matches
    bench_v1_vox035_nocw_png/ 1:1 for direct visual comparison synth vs real.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.preview_grid import draw_scene, legend_patches, MAX_PTS

    if len(pts) > MAX_PTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_PTS, replace=False)
        pts = pts[idx]
        labels = labels[idx]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, view in zip(axes, ["top", "front", "side"]):
        draw_scene(ax, pts, labels.astype(np.int32), view=view, point_size=2.0)
        ax.set_title(view, fontsize=9)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.legend(
        handles=legend_patches(), loc="lower center", ncol=7,
        fontsize=8, title="Labels", title_fontsize=9,
        framealpha=0.9, markerscale=2,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def visualize_predictions(
    model_name: str,
    data_dir: Path,
    models_dir: Path,
    out_dir: Path,
    n_scenes: int = 6,
    cache_path: "Path | None" = None,
) -> None:
    """Render held-out fold-0 predictions + ground truth for N scenes.

    Cierra el hueco diagnóstico de Fase 0-D3: ver visualmente cómo predice el
    modelo sobre sintético held-out, con el mismo layout que las 6 BBB reales,
    para distinguir "fallo del modelo" de "fallo del dominio".
    """
    model_path = models_dir / f"classifier_{model_name}.pkl"
    if not model_path.exists():
        print(f"  Model not found: {model_path}. Skipping visualize.", file=sys.stderr)
        return

    payload = joblib.load(model_path)
    model = payload["model"]
    scaler = payload["scaler"]

    X, y, groups, pts_xyz = load_dataset(data_dir, cache_path=cache_path)
    X_scaled = scaler.transform(X).astype(np.float32)

    sgkf = StratifiedGroupKFold(n_splits=5)
    tr, val = next(sgkf.split(X_scaled, y, groups=groups))

    # Refit on the training split so predictions are honestly held-out.
    m_cv = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
    m_cv.fit(X_scaled[tr], y[tr])
    preds = m_cv.predict(X_scaled[val])

    val_groups = groups[val]
    val_pts = pts_xyz[val]
    val_y = y[val]

    unique_scenes = np.unique(val_groups)[:n_scenes]
    png_dir = out_dir / f"heldout_preds_{model_name}"
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Rendering {len(unique_scenes)} held-out scenes → {png_dir} ──")
    for sid in unique_scenes:
        mask = val_groups == sid
        scene_pts = val_pts[mask]
        _render_from_memory(
            scene_pts, preds[mask],
            f"scene{int(sid)} — PRED ({model_name})",
            png_dir / f"scene{int(sid)}_pred.png",
        )
        _render_from_memory(
            scene_pts, val_y[mask],
            f"scene{int(sid)} — TRUTH",
            png_dir / f"scene{int(sid)}_truth.png",
        )
    print(f"  Saved {len(unique_scenes) * 2} PNGs to {png_dir}")


def evaluate_model(
    model_name: str,
    data_dir: Path,
    models_dir: Path,
    out_dir: Path,
    cache_path: "Path | None" = None,
) -> "dict | None":
    """
    Run scene-level 5-fold CV on a saved classifier and save evaluation plots.

    Parameters
    ----------
    model_name : 'rf' or 'lgbm'
    data_dir   : directory with labelled PLY files (the training dataset)
    models_dir : directory containing classifier_{model_name}.pkl
    out_dir    : where to save confusion matrix and feature importance PNGs

    Returns
    -------
    dict with 'f1_macro' key, or None if the model file is missing.
    """
    model_path = models_dir / f"classifier_{model_name}.pkl"
    if not model_path.exists():
        print(f"  Model not found: {model_path}. Skipping.", file=sys.stderr)
        return None

    payload = joblib.load(model_path)
    model = payload["model"]
    scaler = payload["scaler"]

    X, y, groups, _pts_xyz = load_dataset(data_dir, cache_path=cache_path)
    X_scaled = scaler.transform(X)

    # CV predictions (scene-level) for confusion matrix
    sgkf = StratifiedGroupKFold(n_splits=5)
    all_true, all_pred = [], []
    f1_macros = []
    for tr, val in sgkf.split(X_scaled, y, groups=groups):
        m_cv = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
        m_cv.fit(X_scaled[tr], y[tr])
        preds = m_cv.predict(X_scaled[val])
        all_true.extend(y[val])
        all_pred.extend(preds)
        f1_macros.append(f1_score(y[val], preds, average="macro", zero_division=0))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    f1_mac = np.mean(f1_macros)

    print(f"\n── {model_name.upper()} — CV Results ──")
    print(classification_report(
        all_true, all_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print(f"  Mean CV F1-macro: {f1_mac:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=CLASS_IDS)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(float) / row_sums

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(
        cm_norm,
        f"{model_name.upper()} — Normalised Confusion Matrix (CV)",
        out_dir / f"confusion_matrix_{model_name}.png",
    )

    if model_name == "rf":
        _plot_feature_importance(
            model,
            out_dir / "feature_importance_rf.png",
        )

    return {"f1_macro": f1_mac}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 5-class classifier")
    parser.add_argument("--model", choices=["rf", "lgbm", "both"], default="both")
    parser.add_argument("--data", "--dataset-dir", default="../output/dataset",
                        dest="data", help="Directory with labelled PLY files (default: %(default)s)")
    parser.add_argument("--output-dir", default="../output/classifier_eval",
                        help="Where to save evaluation plots (default: %(default)s)")
    parser.add_argument("--models-dir", default="../models")
    parser.add_argument("--visualize-predictions", action="store_true",
                        help="Save per-scene 3-view PNGs of held-out fold-0 predictions")
    parser.add_argument("--n-visualize", type=int, default=6,
                        help="How many held-out scenes to render (default 6)")
    parser.add_argument("--features-cache", default=None, metavar="PATH",
                        help="Path to a .npz feature cache (see train.py --features-cache)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.output_dir)

    cache_path = Path(args.features_cache) if args.features_cache else None
    models_to_eval = ["rf", "lgbm"] if args.model == "both" else [args.model]
    results = {}
    for m in models_to_eval:
        if args.visualize_predictions:
            visualize_predictions(
                m, data_dir, models_dir, out_dir,
                n_scenes=args.n_visualize, cache_path=cache_path,
            )
        else:
            res = evaluate_model(m, data_dir, models_dir, out_dir, cache_path=cache_path)
            if res:
                results[m] = res

    if len(results) == 2:
        _plot_comparison(results, out_dir / "cv_comparison.png")

    print("\n── Summary ──")
    for m, r in results.items():
        print(f"  {m.upper():6s} F1-macro: {r['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
