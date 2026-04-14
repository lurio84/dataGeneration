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

Floor subsampling:
  --floor-subsample contextual:1.0,bulk:0.1
                       : Keep 100% of floor pts near non-floor (< 0.8 m) and 10% of the
                         rest.  Reduces floor dominance without losing boundary context.
                         Use 'none' to disable (default, keeps all floor pts).

Class weight multipliers:
  --class-weight-mult person:2,pallet:2
                       : Multiply the balanced sample weights of the listed classes by the
                         given factors.  Applied on top of class_weight='balanced'.

RF memory / size control:
  --max-samples-rf 2000000
                       : max_samples parameter for RandomForest in the final retrain.
                         Keeps pkl size manageable on large datasets.
                         Does NOT apply during CV (dataset already subsampled by
                         --cv-subsample).

LGBM-only CV:
  --lgbm-only-cv       : Skip RF during CV folds.  RF is still trained in the final
                         retrain as a sanity check.  Saves ~5-10x CV time with
                         negligible F1 difference vs RF CV.

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
from sklearn.utils.class_weight import compute_sample_weight

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
    return feats, labels.astype(np.int32), groups, pts


def load_dataset(data_dir: Path, n_jobs: int = -1):
    """
    Load all PLYs in parallel, extract features, exclude label=255.

    Returns X (N,F) float32, y (N,) int32, groups (N,) int32,
            pts_xyz (N,3) float32 — raw XYZ coordinates (needed for floor subsampling).
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

    all_X, all_y, all_groups, all_pts = [], [], [], []
    for res in results:
        if res is None:
            continue
        feats, labels, groups, pts = res
        all_X.append(feats)
        all_y.append(labels)
        all_groups.append(groups)
        all_pts.append(pts)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    pts_xyz = np.concatenate(all_pts, axis=0)

    elapsed = time.time() - t0
    print(f"  {len(X):,} points from {len(ply_files)} scenes in {elapsed:.1f}s", flush=True)
    classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(classes, counts):
        print(f"    label {c} ({LABEL_NAMES.get(c, '?')}): {cnt:,} pts "
              f"({cnt/len(y)*100:.1f}%)", flush=True)

    return X, y, groups, pts_xyz


# ── Floor subsampling ─────────────────────────────────────────────────────────

def _parse_floor_subsample(spec_str: str):
    """
    Parse '--floor-subsample contextual:1.0,bulk:0.1' → dict or None.
    'none' or None → no-op.
    """
    if spec_str is None or spec_str.strip().lower() == "none":
        return None
    result = {}
    for part in spec_str.split(","):
        k, v = part.split(":")
        result[k.strip()] = float(v.strip())
    return result


def apply_floor_subsample(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pts_xyz: np.ndarray,
    spec: dict,
    seed: int = 42,
    contextual_radius: float = 0.8,
) -> tuple:
    """
    Subsample floor points (label=0) based on proximity to non-floor points.

    spec keys:
      'contextual' : keep-fraction for floor pts within contextual_radius of non-floor
      'bulk'       : keep-fraction for the remaining floor pts

    Returns filtered (X, y, groups, pts_xyz).
    """
    from scipy.spatial import cKDTree

    floor_mask = y == 0
    nonfloor_mask = ~floor_mask
    n_floor = int(floor_mask.sum())

    if n_floor == 0:
        return X, y, groups, pts_xyz

    floor_idx = np.where(floor_mask)[0]
    nonfloor_pts = pts_xyz[nonfloor_mask]
    floor_pts = pts_xyz[floor_mask]

    contextual_frac = spec.get("contextual", 1.0)
    bulk_frac = spec.get("bulk", 0.1)
    rng = np.random.default_rng(seed)

    if len(nonfloor_pts) == 0:
        # Edge case: no non-floor pts → treat all as bulk
        n_keep = max(1, int(n_floor * bulk_frac))
        keep_floor = rng.choice(floor_idx, size=n_keep, replace=False)
        final_keep = np.sort(keep_floor)
        print(f"  --floor-subsample: floor {n_floor:,} → {len(final_keep):,} pts "
              f"(all bulk, no non-floor pts) → {len(final_keep):,} total pts",
              flush=True)
        return X[final_keep], y[final_keep], groups[final_keep], pts_xyz[final_keep]

    tree = cKDTree(nonfloor_pts)
    dists, _ = tree.query(floor_pts, k=1)

    is_contextual = dists < contextual_radius
    contextual_local = np.where(is_contextual)[0]
    bulk_local = np.where(~is_contextual)[0]

    contextual_global = floor_idx[contextual_local]
    bulk_global = floor_idx[bulk_local]

    # Contextual: keep fraction (default 100%)
    if contextual_frac < 1.0 and len(contextual_global) > 0:
        n_keep = max(1, int(len(contextual_global) * contextual_frac))
        contextual_keep = rng.choice(contextual_global, size=n_keep, replace=False)
    else:
        contextual_keep = contextual_global

    # Bulk: keep fraction (default 10%)
    if bulk_frac < 1.0 and len(bulk_global) > 0:
        n_keep = max(1, int(len(bulk_global) * bulk_frac))
        bulk_keep = rng.choice(bulk_global, size=n_keep, replace=False)
    elif len(bulk_global) > 0:
        bulk_keep = bulk_global
    else:
        bulk_keep = np.array([], dtype=np.int64)

    nonfloor_global = np.where(nonfloor_mask)[0]
    final_keep = np.sort(np.concatenate([nonfloor_global, contextual_keep, bulk_keep]))

    n_kept = len(contextual_keep) + len(bulk_keep)
    print(f"  --floor-subsample: floor {n_floor:,} → {n_kept:,} pts "
          f"(contextual {len(contextual_keep):,} + bulk {len(bulk_keep):,}) "
          f"→ {len(final_keep):,} total pts", flush=True)

    return X[final_keep], y[final_keep], groups[final_keep], pts_xyz[final_keep]


# ── Class weight multipliers ──────────────────────────────────────────────────

def _parse_class_weight_mult(spec_str: str) -> dict:
    """Parse 'person:2,pallet:2' → {'person': 2.0, 'pallet': 2.0}."""
    if not spec_str:
        return {}
    result = {}
    for part in spec_str.split(","):
        k, v = part.split(":")
        result[k.strip()] = float(v.strip())
    return result


def compute_class_sample_weights(y: np.ndarray, mult_dict: dict) -> np.ndarray:
    """
    Compute per-point sample weights = balanced_weight * class_multiplier.

    mult_dict: {'person': 2.0, 'pallet': 2.0} — class names from LABEL.
    Returns float32 array of shape (N,).
    """
    sw = compute_sample_weight("balanced", y)
    for cls_name, mult in mult_dict.items():
        cls_id = LABEL.get(cls_name)
        if cls_id is None:
            raise ValueError(
                f"Unknown class name: {cls_name!r}. Valid: {list(LABEL)}"
            )
        sw[y == cls_id] *= mult
    return sw.astype(np.float32)


# ── CV subsampling (stratified per scene) ────────────────────────────────────

def subsample_for_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    frac: float,
    seed: int,
    weights: np.ndarray = None,
) -> tuple:
    """
    Return a stratified subsample of frac*100% of the data for CV.
    Subsampling is done per-scene so every scene keeps its group identity.
    The full dataset is NOT touched — used only inside CV folds.

    If weights is provided, returns (X_sub, y_sub, groups_sub, weights_sub).
    Otherwise returns (X_sub, y_sub, groups_sub, None).
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
    w_sub = weights[keep] if weights is not None else None
    return X[keep], y[keep], groups[keep], w_sub


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(
    model_fn,
    X_scaled: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_splits: int = 5,
    subsample: float = 1.0,
    seed: int = 42,
    sample_weight: np.ndarray = None,
) -> dict:
    """
    Scene-level StratifiedGroupKFold CV.

    If subsample < 1.0, each fold trains on a stratified subsample of the
    training split (saves time; final retrain always uses the full dataset).

    Returns a dict:
      f1_macro, f1_floor, f1_cargo, f1_vehicle, f1_person, f1_pallet,
      cv_time_s, n_points_cv
    """
    t_cv_start = time.time()

    if subsample < 1.0:
        X_cv, y_cv, g_cv, sw_cv = subsample_for_cv(
            X_scaled, y, groups, subsample, seed, weights=sample_weight
        )
        print(f"  CV subsample {subsample:.0%}: {len(X_cv):,} / {len(X_scaled):,} pts",
              flush=True)
    else:
        X_cv, y_cv, g_cv, sw_cv = X_scaled, y, groups, sample_weight

    n_points_cv = len(X_cv)
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    f1_macros, reports = [], []

    print(f"\n── CV ({n_splits} folds, scene-level) — {model_name} ──", flush=True)
    for fold, (tr, val) in enumerate(sgkf.split(X_cv, y_cv, groups=g_cv)):
        t0 = time.time()
        m = model_fn()
        fit_kwargs = {}
        if sw_cv is not None:
            fit_kwargs["sample_weight"] = sw_cv[tr]
        m.fit(X_cv[tr], y_cv[tr], **fit_kwargs)
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
    cv_time_s = time.time() - t_cv_start
    print(f"  mean F1-macro = {mean_f1:.4f}", flush=True)
    print("  F1 per class (mean across folds):", flush=True)

    per_class = {}
    for cls_name in LABEL.keys():
        vals = [r[cls_name]["f1-score"] for r in reports if cls_name in r]
        f1_cls = float(np.mean(vals)) if vals else 0.0
        per_class[cls_name] = f1_cls
        print(f"    {cls_name:10s}: {f1_cls:.4f}", flush=True)

    return {
        "f1_macro":    mean_f1,
        "f1_floor":    per_class.get("floor",   0.0),
        "f1_cargo":    per_class.get("cargo",   0.0),
        "f1_vehicle":  per_class.get("vehicle", 0.0),
        "f1_person":   per_class.get("person",  0.0),
        "f1_pallet":   per_class.get("pallet",  0.0),
        "cv_time_s":   cv_time_s,
        "n_points_cv": n_points_cv,
    }


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
    parser.add_argument("--no-floor", action="store_true",
                        help="Exclude floor points (label=0) from training and CV. "
                             "Use together with a floor-removal step at predict time.")
    # ── New flags (1.2 – 1.5) ────────────────────────────────────────────────
    parser.add_argument(
        "--floor-subsample", default="none", metavar="SPEC",
        help="Subsample floor pts by proximity to non-floor. "
             "Format: 'contextual:FRAC,bulk:FRAC' (e.g. 'contextual:1.0,bulk:0.1'). "
             "'none' = keep all floor pts (default).")
    parser.add_argument(
        "--class-weight-mult", default="", metavar="SPEC",
        help="Multiply balanced sample weights for listed classes. "
             "Format: 'person:2,pallet:2'. Applied on top of class_weight=balanced.")
    parser.add_argument(
        "--max-samples-rf", type=int, default=None, metavar="N",
        help="max_samples for RandomForest in the FINAL retrain only (default None = all). "
             "Recommended: 2000000 for datasets with >10M pts to avoid OOM and pkl bloat.")
    parser.add_argument(
        "--lgbm-only-cv", action="store_true",
        help="Skip RF during CV; run only LightGBM CV folds. "
             "RF is still trained in the final retrain. ~5-10x faster CV.")
    parser.add_argument(
        "--skip-rf-retrain", action="store_true",
        help="Skip final RandomForest retrain (only fit LGBM final). "
             "Useful when you only need the LGBM .pkl for prediction "
             "and RF would OOM or be too slow.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    floor_spec    = _parse_floor_subsample(args.floor_subsample)
    cw_mult       = _parse_class_weight_mult(args.class_weight_mult)

    no_floor_tag  = " [NO-FLOOR]" if args.no_floor else ""
    fs_tag        = f" [floor-sub={args.floor_subsample}]" if floor_spec else ""
    cw_tag        = f" [cw-mult={args.class_weight_mult}]" if cw_mult else ""
    lgbm_only_tag = " [lgbm-only-cv]" if args.lgbm_only_cv else ""
    maxsamp_tag   = f" [max-samples-rf={args.max_samples_rf}]" if args.max_samples_rf else ""

    print(f"CV config: rf_cv_trees={args.cv_estimators}, "
          f"cv_subsample={args.cv_subsample:.0%}, "
          f"final_rf_trees={args.n_estimators}"
          f"{no_floor_tag}{fs_tag}{cw_tag}{lgbm_only_tag}{maxsamp_tag}",
          flush=True)

    # ── Load + scale ─────────────────────────────────────────────────────────
    X, y, groups, pts_xyz = load_dataset(data_dir, n_jobs=args.n_jobs)

    if args.no_floor:
        mask = y != 0
        n_removed = int((~mask).sum())
        print(f"  --no-floor: removing {n_removed:,} floor pts "
              f"({n_removed/len(y)*100:.1f}%) → {mask.sum():,} pts remain", flush=True)
        X, y, groups, pts_xyz = X[mask], y[mask], groups[mask], pts_xyz[mask]

    if floor_spec is not None:
        X, y, groups, pts_xyz = apply_floor_subsample(
            X, y, groups, pts_xyz, floor_spec, seed=args.seed
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ── Sample weights (class_weight_mult on top of balanced) ───────────────
    sample_weight = None
    if cw_mult:
        sample_weight = compute_class_sample_weights(y, cw_mult)
        for cls_name, mult in cw_mult.items():
            print(f"  --class-weight-mult: {cls_name} ×{mult}", flush=True)

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
            max_samples=args.max_samples_rf,
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
    if not args.lgbm_only_cv:
        cv_rf = run_cv(make_rf_cv, X_scaled, y, groups, "RandomForest",
                       subsample=args.cv_subsample, seed=args.seed,
                       sample_weight=sample_weight)
    else:
        print("\n── CV RandomForest — SKIPPED (--lgbm-only-cv) ──", flush=True)
        cv_rf = None

    cv_lgbm = run_cv(make_lgbm, X_scaled, y, groups, "LightGBM",
                     subsample=args.cv_subsample, seed=args.seed,
                     sample_weight=sample_weight)

    print(f"\n── CV Summary ──", flush=True)
    if cv_rf is not None:
        print(f"  RF    F1-macro: {cv_rf['f1_macro']:.4f}", flush=True)
    print(f"  LGBM  F1-macro: {cv_lgbm['f1_macro']:.4f}", flush=True)

    # ── Retrain on FULL dataset ───────────────────────────────────────────────
    print("\nRetraining on full dataset ...", flush=True)

    fit_kwargs_retrain = {}
    if sample_weight is not None:
        fit_kwargs_retrain["sample_weight"] = sample_weight

    rf = None
    if not args.skip_rf_retrain:
        rf = make_rf_final()
        t0 = time.time()
        rf.fit(X_scaled, y, **fit_kwargs_retrain)
        print(f"  RF ({args.n_estimators} trees) trained in {time.time()-t0:.1f}s", flush=True)
    else:
        print("  RF retrain SKIPPED (--skip-rf-retrain)", flush=True)

    lgbm = make_lgbm()
    t0 = time.time()
    lgbm.fit(X_scaled, y, **fit_kwargs_retrain)
    print(f"  LGBM trained in {time.time()-t0:.1f}s", flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    models_to_save = [("lgbm", lgbm)]
    if rf is not None:
        models_to_save.insert(0, ("rf", rf))
    for name, model in models_to_save:
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
