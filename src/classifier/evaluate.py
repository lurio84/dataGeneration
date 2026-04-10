"""
evaluate.py — Evaluate RF and/or LightGBM classifiers on the labelled dataset.

Usage (from src/):
    python3 classifier/evaluate.py [--model rf|lgbm|both] [--data ../output/dataset]

Outputs (saved to ../output/classifier_eval/):
  - confusion_matrix_{model}.png  — 5×5 normalised confusion matrix
  - feature_importance_rf.png     — RF feature importance bar chart
  - cv_comparison.png             — RF vs LightGBM F1 comparison table/bar
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


def _plot_confusion_matrix(cm_norm, title, out_path):
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


def _plot_feature_importance(rf_model, out_path):
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


def evaluate_model(model_name: str, data_dir: Path, models_dir: Path, out_dir: Path):
    model_path = models_dir / f"classifier_{model_name}.pkl"
    if not model_path.exists():
        print(f"  Model not found: {model_path}. Skipping.", file=sys.stderr)
        return None

    payload = joblib.load(model_path)
    model = payload["model"]
    scaler = payload["scaler"]

    X, y, groups = load_dataset(data_dir)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-class classifier")
    parser.add_argument("--model", choices=["rf", "lgbm", "both"], default="both")
    parser.add_argument("--data", default="../output/dataset")
    parser.add_argument("--models-dir", default="../models")
    args = parser.parse_args()

    data_dir = Path(args.data)
    models_dir = Path(args.models_dir)
    out_dir = Path("../output/classifier_eval")

    models_to_eval = ["rf", "lgbm"] if args.model == "both" else [args.model]
    results = {}
    for m in models_to_eval:
        res = evaluate_model(m, data_dir, models_dir, out_dir)
        if res:
            results[m] = res

    if len(results) == 2:
        _plot_comparison(results, out_dir / "cv_comparison.png")

    print("\n── Summary ──")
    for m, r in results.items():
        print(f"  {m.upper():6s} F1-macro: {r['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
