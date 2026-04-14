"""
train_cluster_classifier.py — Train RF and LightGBM cluster-level classifiers.

Runs the geometric pipeline on every synthetic PLY in `--data` to produce
cluster-level training examples (23 features per cluster), then trains both
RandomForest and LightGBM with scene-level cross-validation.

Usage (from the datageneration/ repo root):
    python3 scripts/train_cluster_classifier.py \\
        --data output/dataset \\
        --out  models \\
        --seed 42 \\
        --n-estimators 300 \\
        --purity-threshold 0.6 \\
        --min-cluster-pts 30

Outputs:
    models/cluster_classifier_lgbm.pkl
    models/cluster_classifier_rf.pkl

Gates (checked automatically):
    CV macro-F1 ≥ 0.85
    F1 cargo    ≥ 0.85
    recall cargo ≥ 0.85

Differences from the per-point classifier's train.py:
  - No --cv-subsample (dataset is ~300-800 clusters; CV is instantaneous)
  - No --floor-subsample (not applicable at cluster level)
  - No --class-weight-mult (DECISIONS §15 hallazgo 3: empeora F1)
  - class_weight="balanced" en RF/LGBM
  - Pickle key "label_map" uses CLUSTER_LABEL_MAP
  - Pickle key "version": 1 for future load-time validation
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# ── path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from cargo_geometric.params import GeometricParams
from cargo_geometric.cluster_features import FEATURE_NAMES
from cargo_geometric.cluster_gt import (
    CLUSTER_LABEL_MAP,
    CLUSTER_LABEL_NAMES,
    build_cluster_dataset,
)


# ── CV ────────────────────────────────────────────────────────────────────────

def run_cv(
    model_fn,
    X_scaled: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Scene-level StratifiedGroupKFold CV.

    Returns dict with f1_macro, f1_cargo, recall_cargo, cv_time_s,
    per-class F1 for all 5 classes.
    """
    t0 = time.time()

    label_ids   = sorted(CLUSTER_LABEL_MAP.values())
    label_names = [CLUSTER_LABEL_NAMES[i] for i in label_ids]
    cargo_id    = CLUSTER_LABEL_MAP["cargo"]

    # StratifiedGroupKFold needs at least n_splits unique groups per class.
    # With ~100 scenes and 5 classes, this is normally fine.
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    f1_macros: list[float] = []
    fold_reports: list[dict] = []
    fold_recall_cargo: list[float] = []

    print(f"\n── CV ({n_splits} folds, scene-level) — {model_name} ──", flush=True)
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_scaled, y, groups=groups)):
        t_fold = time.time()
        m = model_fn()
        m.fit(X_scaled[tr_idx], y[tr_idx])
        preds = m.predict(X_scaled[val_idx])
        f1_mac = float(f1_score(y[val_idx], preds, average="macro", zero_division=0))
        rec_cargo = float(recall_score(
            y[val_idx], preds, labels=[cargo_id], average="macro", zero_division=0
        ))
        f1_macros.append(f1_mac)
        fold_recall_cargo.append(rec_cargo)
        rpt = classification_report(
            y[val_idx], preds,
            labels=label_ids, target_names=label_names,
            zero_division=0, output_dict=True,
        )
        fold_reports.append(rpt)
        print(
            f"  fold {fold+1}: F1-macro={f1_mac:.4f}  "
            f"F1-cargo={rpt['cargo']['f1-score']:.4f}  "
            f"recall-cargo={rec_cargo:.4f}  "
            f"({time.time()-t_fold:.1f}s)",
            flush=True,
        )

    mean_f1         = float(np.mean(f1_macros))
    mean_rec_cargo  = float(np.mean(fold_recall_cargo))
    cv_time         = time.time() - t0

    print(f"\n  mean F1-macro      = {mean_f1:.4f}", flush=True)
    print(f"  mean recall-cargo  = {mean_rec_cargo:.4f}", flush=True)
    print("  F1 per class (mean across folds):", flush=True)

    per_class: dict[str, float] = {}
    for cls_name in label_names:
        vals = [r[cls_name]["f1-score"] for r in fold_reports if cls_name in r]
        f1_cls = float(np.mean(vals)) if vals else 0.0
        per_class[cls_name] = f1_cls
        print(f"    {cls_name:>16}: {f1_cls:.4f}", flush=True)

    return {
        "f1_macro":    mean_f1,
        "f1_cargo":    per_class.get("cargo", 0.0),
        "recall_cargo": mean_rec_cargo,
        "per_class":   per_class,
        "cv_time_s":   cv_time,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train cluster-level RF + LGBM classifiers"
    )
    parser.add_argument("--data",  default="output/dataset",
                        help="Directory with labelled synthetic PLYs")
    parser.add_argument("--out",   default="models",
                        help="Output directory for .pkl files")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="RF trees (CV and final retrain)")
    parser.add_argument("--purity-threshold", type=float, default=0.6,
                        help="Min per-cluster majority purity (default 0.6)")
    parser.add_argument("--min-cluster-pts", type=int, default=30,
                        help="Min DBSCAN cluster size to include (default 30)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-cv-folds", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build cluster-level dataset ───────────────────────────────────────────
    print(f"Building cluster GT from {data_dir} ...", flush=True)
    t0 = time.time()
    params = GeometricParams()
    X, y, groups = build_cluster_dataset(
        dataset_dir=data_dir,
        params=params,
        min_cluster_pts=args.min_cluster_pts,
        purity_threshold=args.purity_threshold,
        verbose=False,
    )
    print(f"  {len(y)} clusters from "
          f"{len(np.unique(groups))} scenes in {time.time()-t0:.1f}s", flush=True)

    # Class distribution
    label_ids   = sorted(CLUSTER_LABEL_MAP.values())
    label_names = [CLUSTER_LABEL_NAMES[i] for i in label_ids]
    print("  Class distribution:", flush=True)
    for lid, lname in zip(label_ids, label_names):
        n = int((y == lid).sum())
        print(f"    {lname:>16}: {n:5d}  ({n/len(y)*100:.1f}%)", flush=True)

    # Warn if any class has < 5 examples
    for lid, lname in zip(label_ids, label_names):
        n = int((y == lid).sum())
        if 0 < n < 5:
            print(
                f"  WARNING: class {lname!r} has only {n} examples — "
                f"consider collapsing into 'other'", flush=True
            )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ── Model factories ───────────────────────────────────────────────────────
    def make_rf():
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=20,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

    def make_lgbm():
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            num_leaves=63,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbose=-1,
        )

    # ── CV (RF then LGBM) ─────────────────────────────────────────────────────
    cv_rf   = run_cv(make_rf,   X_scaled, y, groups, "RandomForest",
                     n_splits=args.n_cv_folds, seed=args.seed)
    cv_lgbm = run_cv(make_lgbm, X_scaled, y, groups, "LightGBM",
                     n_splits=args.n_cv_folds, seed=args.seed)

    print("\n" + "="*60, flush=True)
    print("── CV Summary ──", flush=True)
    print(f"  RF    F1-macro={cv_rf['f1_macro']:.4f}  "
          f"F1-cargo={cv_rf['f1_cargo']:.4f}  "
          f"recall-cargo={cv_rf['recall_cargo']:.4f}", flush=True)
    print(f"  LGBM  F1-macro={cv_lgbm['f1_macro']:.4f}  "
          f"F1-cargo={cv_lgbm['f1_cargo']:.4f}  "
          f"recall-cargo={cv_lgbm['recall_cargo']:.4f}", flush=True)

    # ── Gate check ────────────────────────────────────────────────────────────
    best = cv_lgbm if cv_lgbm["f1_macro"] >= cv_rf["f1_macro"] else cv_rf
    gate_pass = (
        best["f1_macro"]    >= 0.85 and
        best["f1_cargo"]    >= 0.85 and
        best["recall_cargo"] >= 0.85
    )
    if gate_pass:
        print("\n✓ GATE PASSED — proceed to Paso 4", flush=True)
    else:
        print("\n✗ GATE NOT MET:", flush=True)
        if best["f1_macro"]    < 0.85:
            print(f"  macro-F1={best['f1_macro']:.4f} < 0.85", flush=True)
        if best["f1_cargo"]    < 0.85:
            print(f"  F1-cargo={best['f1_cargo']:.4f} < 0.85", flush=True)
        if best["recall_cargo"] < 0.85:
            print(f"  recall-cargo={best['recall_cargo']:.4f} < 0.85", flush=True)
        print("  → Iterate features or consider abort criterion (see plan §R5)", flush=True)

    # ── Retrain on full dataset ───────────────────────────────────────────────
    print("\nRetraining on full dataset ...", flush=True)

    rf_final   = make_rf()
    lgbm_final = make_lgbm()

    t = time.time()
    rf_final.fit(X_scaled, y)
    print(f"  RF   ({args.n_estimators} trees) in {time.time()-t:.1f}s", flush=True)

    t = time.time()
    lgbm_final.fit(X_scaled, y)
    print(f"  LGBM in {time.time()-t:.1f}s", flush=True)

    # ── Confusion matrix (full dataset, training-set check, not a test) ───────
    print("\n── Confusion matrix (RF on full training data) ──", flush=True)
    cm = confusion_matrix(y, rf_final.predict(X_scaled), labels=label_ids)
    header = "                  " + "  ".join(f"{n:>16}" for n in label_names)
    print(header, flush=True)
    for i, row_name in enumerate(label_names):
        row_str = "  ".join(f"{cm[i,j]:16d}" for j in range(len(label_ids)))
        print(f"{row_name:>16}  {row_str}", flush=True)

    # ── Save pickles ──────────────────────────────────────────────────────────
    pickle_base = {
        "version":       1,
        "scaler":        scaler,
        "feature_names": FEATURE_NAMES,
        "label_map":     CLUSTER_LABEL_MAP,
    }

    for name, model in [("rf", rf_final), ("lgbm", lgbm_final)]:
        path = out_dir / f"cluster_classifier_{name}.pkl"
        payload = {**pickle_base, "model": model}
        joblib.dump(payload, path)
        size_kb = path.stat().st_size / 1024
        print(f"Saved: {path}  ({size_kb:.0f} KB)", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
