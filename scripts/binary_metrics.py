"""
binary_metrics.py — CV del clasificador 5-clases reportado en métrica binaria cargo vs rest.

Herramienta diagnóstica para Fase 0-D2 del plan deep-dancing-flute: si el
target del producto es "máscara limpia de cargo", saber el IoU(cargo) honesto
dice si el modelo actual ya es suficiente (ruta 1A, cero retrain) o si hace
falta entrenar con objetivo binario/3-clases (ruta 1B/1C).

Reutiliza load_dataset + StratifiedGroupKFold scene-level del train.py.
LGBM only (misma factory que bench_training.py).

Uso (desde datageneration/):
    python3 scripts/binary_metrics.py \\
        --data output/dataset \\
        --cv-subsample 0.3 \\
        --cv-estimators 100 \\
        --n-jobs -1 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    jaccard_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from classifier.train import load_dataset, subsample_for_cv  # noqa: E402
from generate_dataset import LABEL  # noqa: E402


CARGO_ID = LABEL["cargo"]  # 1


def make_lgbm(cv_estimators: int, seed: int, n_jobs: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        num_leaves=63,
        n_estimators=cv_estimators,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=seed,
        verbose=-1,
    )


def run_cv_binary(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_subsample: float,
    cv_estimators: int,
    seed: int,
    n_jobs: int,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    if cv_subsample < 1.0:
        X_cv, y_cv, g_cv, _ = subsample_for_cv(
            X_scaled, y, groups, cv_subsample, seed, weights=None
        )
    else:
        X_cv, y_cv, g_cv = X_scaled, y, groups

    print(f"  CV points: {len(X_cv):,} / {len(X_scaled):,}", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5)
    all_y_true, all_y_pred = [], []
    for fold, (tr, val) in enumerate(sgkf.split(X_cv, y_cv, groups=g_cv)):
        t0 = time.time()
        m = make_lgbm(cv_estimators, seed, n_jobs)
        m.fit(X_cv[tr], y_cv[tr])
        preds = m.predict(X_cv[val])
        all_y_true.append(y_cv[val])
        all_y_pred.append(preds)
        f1_m = f1_score(y_cv[val], preds, average="macro", zero_division=0)
        print(
            f"  fold {fold+1}: {time.time()-t0:.1f}s  "
            f"n_val={len(val):,}  f1_macro={f1_m:.4f}",
            flush=True,
        )

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    return y_true, y_pred


def report_multiclass(y_true, y_pred):
    print("\n=== 5-clases (baseline) ===")
    print(classification_report(
        y_true, y_pred,
        labels=list(LABEL.values()),
        target_names=list(LABEL.keys()),
        zero_division=0,
        digits=4,
    ))


def report_binary(y_true, y_pred):
    y_true_bin = (y_true == CARGO_ID).astype(np.int8)
    y_pred_bin = (y_pred == CARGO_ID).astype(np.int8)

    print("\n=== Binario cargo vs rest ===")
    print(classification_report(
        y_true_bin, y_pred_bin,
        labels=[0, 1],
        target_names=["not_cargo", "cargo"],
        zero_division=0,
        digits=4,
    ))

    iou_cargo = jaccard_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    iou_rest = jaccard_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, pos_label=1, average="binary", zero_division=0
    )
    print(f"\n  IoU(cargo)     = {iou_cargo:.4f}")
    print(f"  IoU(not_cargo) = {iou_rest:.4f}")
    print(f"  precision      = {prec:.4f}")
    print(f"  recall         = {rec:.4f}")
    print(f"  F1             = {f1:.4f}")

    support_cargo = int(y_true_bin.sum())
    print(f"\n  support cargo  = {support_cargo:,} / {len(y_true_bin):,} "
          f"({support_cargo/len(y_true_bin)*100:.2f}%)")

    print("\n--- Gate de decisión (plan deep-dancing-flute §Fase 1) ---")
    if iou_cargo >= 0.90:
        print(f"  IoU(cargo) = {iou_cargo:.4f} ≥ 0.90  → RUTA 1A (collapse post-hoc)")
    elif iou_cargo >= 0.85:
        print(f"  IoU(cargo) = {iou_cargo:.4f} ∈ [0.85, 0.90)  → RUTA 1C (4-clases) o 1A con threshold")
    else:
        print(f"  IoU(cargo) = {iou_cargo:.4f} < 0.85  → RUTA 1B (retrain 3-clases)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="output/dataset")
    p.add_argument("--cv-subsample", type=float, default=0.3)
    p.add_argument("--cv-estimators", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features-cache", default=None, metavar="PATH",
                   help="Path to a .npz feature cache (see train.py --features-cache)")
    args = p.parse_args()

    X, y, groups, _ = load_dataset(
        Path(args.data),
        n_jobs=args.n_jobs,
        cache_path=Path(args.features_cache) if args.features_cache else None,
    )
    print(f"Dataset: {len(X):,} pts, {X.shape[1]} features", flush=True)

    y_true, y_pred = run_cv_binary(
        X, y, groups,
        args.cv_subsample, args.cv_estimators, args.seed, args.n_jobs,
    )
    report_multiclass(y_true, y_pred)
    report_binary(y_true, y_pred)


if __name__ == "__main__":
    main()
