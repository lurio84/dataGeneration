"""
train.py — Train RF and LightGBM 5-class per-point classifiers.

Usage (from src/):
    python3 -u classifier/train.py [--data ../output/dataset] [--out ../models]
                                   [--seed 42]
                                   [--cv-estimators 100]   # RF trees during CV
                                   [--cv-subsample 0.3]    # fraction of pts for CV folds

CV optimisations (no accuracy loss on large datasets):
  --cv-estimators 100  : RF uses 100 trees for CV; final retrain always uses --n-estimators.
                         With ≥1M points, RF F1 converges well before 100 trees.
  --cv-subsample  0.3  : Subsample 30% of points (stratified per scene) for CV folds only.
                         Final retrain always uses the full dataset.

CV is scene-level (StratifiedGroupKFold, group=scene_id) to avoid data leakage.
Label 255 (outlier) is always excluded from training and evaluation.
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from classifier.features import extract_features, FEATURE_NAMES
from generate_dataset import LABEL, LABEL_RGB  # noqa: F401

LABEL_NAMES = {v: k for k, v in LABEL.items()}


# ── PLY loader ────────────────────────────────────────────────────────────────

def _read_ply_xyz_label(path: Path):
    """Binary-LE PLY → pts (N,3) float32, labels (N,) uint8."""
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header\n")
    header = raw[:end].decode("ascii", errors="replace")
    n_vertices = int(next(
        l.split()[-1] for l in header.splitlines()
        if l.startswith("element vertex")
    ))
    body = raw[end + len("end_header\n"):]
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("label", "u1"),
    ])
    arr = np.frombuffer(body[:n_vertices * dtype.itemsize], dtype=dtype)
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
    return pts, arr["label"]


# ── Dataset loading (parallel across scenes) ─────────────────────────────────

def _process_scene(args):
    """Load one PLY, extract features, exclude label=255.  joblib worker."""
    scene_id, ply_path = args
    pts, labels = _read_ply_xyz_label(ply_path)
    mask = labels != 255
    pts, labels = pts[mask], labels[mask]
    if len(pts) == 0:
        return None
    feats = extract_features(pts)
    groups = np.full(len(pts), scene_id, dtype=np.int32)
    return feats, labels.astype(np.int32), groups


def load_dataset(data_dir: Path, n_jobs: int = -1):
    """
    Load all PLYs in parallel, extract features, exclude label=255.

    Returns X (N,15) float32, y (N,) int32, groups (N,) int32.
    """
    ply_files = sorted(data_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {data_dir}")

    print(f"Loading {len(ply_files)} PLY files from {data_dir} (n_jobs={n_jobs}) ...",
          flush=True)
    t0 = time.time()

    results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(_process_scene)((scene_id, ply))
        for scene_id, ply in enumerate(ply_files)
    )

    all_X, all_y, all_groups = [], [], []
    for res in results:
        if res is None:
            continue
        feats, labels, groups = res
        all_X.append(feats)
        all_y.append(labels)
        all_groups.append(groups)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    elapsed = time.time() - t0
    print(f"  {len(X):,} points from {len(ply_files)} scenes in {elapsed:.1f}s", flush=True)
    classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(classes, counts):
        print(f"    label {c} ({LABEL_NAMES.get(c, '?')}): {cnt:,} pts "
              f"({cnt/len(y)*100:.1f}%)", flush=True)

    return X, y, groups


# ── CV subsampling (stratified per scene) ────────────────────────────────────

def subsample_for_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     frac: float, seed: int) -> tuple:
    """
    Return a stratified subsample of frac*100% of the data for CV.
    Subsampling is done per-scene so every scene keeps its group identity.
    The full dataset is NOT touched — used only inside CV folds.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for scene_id in np.unique(groups):
        idx = np.where(groups == scene_id)[0]
        y_scene = y[idx]
        chosen = []
        for cls in np.unique(y_scene):
            cls_idx = idx[y_scene == cls]
            n_keep = max(1, int(len(cls_idx) * frac))
            chosen.append(rng.choice(cls_idx, size=n_keep, replace=False))
        keep.append(np.concatenate(chosen))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], groups[keep]


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(model_fn, X_scaled, y, groups, model_name: str,
           n_splits: int = 5, subsample: float = 1.0, seed: int = 42):
    """
    Scene-level StratifiedGroupKFold CV.

    If subsample < 1.0, each fold trains on a stratified subsample of the
    training split (saves time; final retrain always uses the full dataset).
    Returns mean F1-macro across folds.
    """
    if subsample < 1.0:
        X_cv, y_cv, g_cv = subsample_for_cv(X_scaled, y, groups, subsample, seed)
        print(f"  CV subsample {subsample:.0%}: {len(X_cv):,} / {len(X_scaled):,} pts",
              flush=True)
    else:
        X_cv, y_cv, g_cv = X_scaled, y, groups

    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    f1_macros, reports = [], []

    print(f"\n── CV ({n_splits} folds, scene-level) — {model_name} ──", flush=True)
    for fold, (tr, val) in enumerate(sgkf.split(X_cv, y_cv, groups=g_cv)):
        t0 = time.time()
        m = model_fn()
        m.fit(X_cv[tr], y_cv[tr])
        preds = m.predict(X_cv[val])
        f1_mac = f1_score(y_cv[val], preds, average="macro", zero_division=0)
        f1_macros.append(f1_mac)
        rpt = classification_report(
            y_cv[val], preds,
            labels=list(LABEL.values()),
            target_names=list(LABEL.keys()),
            zero_division=0,
            output_dict=True,
        )
        reports.append(rpt)
        print(f"  fold {fold+1}: F1-macro = {f1_mac:.4f}  ({time.time()-t0:.1f}s)",
              flush=True)

    mean_f1 = float(np.mean(f1_macros))
    print(f"  mean F1-macro = {mean_f1:.4f}", flush=True)
    print("  F1 per class (mean across folds):", flush=True)
    for cls_name in LABEL.keys():
        vals = [r[cls_name]["f1-score"] for r in reports if cls_name in r]
        if vals:
            print(f"    {cls_name:10s}: {np.mean(vals):.4f}", flush=True)

    return mean_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 5-class per-point classifier")
    parser.add_argument("--data", default="../output/dataset")
    parser.add_argument("--out",  default="../models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="RF trees in final retrain (default 300)")
    parser.add_argument("--cv-estimators", type=int, default=100,
                        help="RF trees during CV folds (default 100, ≪ final for speed)")
    parser.add_argument("--cv-subsample", type=float, default=0.3,
                        help="Fraction of pts used in CV folds (default 0.3). "
                             "Final retrain always uses full data.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallelism for RF/LGBM and feature extraction "
                             "(default -1 = all cores). Use 1 for tests.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CV config: rf_cv_trees={args.cv_estimators}, "
          f"cv_subsample={args.cv_subsample:.0%}, "
          f"final_rf_trees={args.n_estimators}", flush=True)

    # ── Load + scale ─────────────────────────────────────────────────────────
    X, y, groups = load_dataset(data_dir, n_jobs=args.n_jobs)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ── Model factories ──────────────────────────────────────────────────────
    def make_rf_cv():
        return RandomForestClassifier(
            n_estimators=args.cv_estimators,
            max_depth=20,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

    def make_rf_final():
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

    # ── Cross-validation ─────────────────────────────────────────────────────
    f1_rf = run_cv(make_rf_cv, X_scaled, y, groups, "RandomForest",
                   subsample=args.cv_subsample, seed=args.seed)
    f1_lgbm = run_cv(make_lgbm, X_scaled, y, groups, "LightGBM",
                     subsample=args.cv_subsample, seed=args.seed)

    print(f"\n── CV Summary ──", flush=True)
    print(f"  RF    F1-macro: {f1_rf:.4f}", flush=True)
    print(f"  LGBM  F1-macro: {f1_lgbm:.4f}", flush=True)

    # ── Retrain on FULL dataset ───────────────────────────────────────────────
    print("\nRetraining on full dataset ...", flush=True)

    rf = make_rf_final()
    t0 = time.time()
    rf.fit(X_scaled, y)
    print(f"  RF ({args.n_estimators} trees) trained in {time.time()-t0:.1f}s", flush=True)

    lgbm = make_lgbm()
    t0 = time.time()
    lgbm.fit(X_scaled, y)
    print(f"  LGBM trained in {time.time()-t0:.1f}s", flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    for name, model in [("rf", rf), ("lgbm", lgbm)]:
        path = out_dir / f"classifier_{name}.pkl"
        joblib.dump({
            "model":         model,
            "scaler":        scaler,
            "feature_names": FEATURE_NAMES,
            "label_map":     LABEL,
        }, path)
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
