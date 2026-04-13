"""
bench_training.py — Training benchmark: compare 2 configs over the same dataset.

Goal: decide "floor yes or no" (and contextual subsampling) before generating
dataset v2 (~500 scenes).  Runs LGBM CV only (no RF CV, no final retrain)
so the bench completes in reasonable time.

Configs
-------
  A_baseline   : baseline — with_floor + all 17 features + class_weight=balanced (no new flags)
  C_contextual : --floor-subsample contextual:1.0,bulk:0.1
                 --class-weight-mult person:2,pallet:2

Usage (from datageneration/):
    python3 scripts/bench_training.py \\
        --data output/dataset \\
        --results results/bench_v1.csv \\
        --seed 42 \\
        --cv-subsample 0.3 \\
        --cv-estimators 100        # LGBM trees (n_estimators)
        --n-jobs -1

Smoke test (quick, ~30 s):
    python3 scripts/bench_training.py \\
        --data output/dataset \\
        --results results/bench_v1_smoke.csv \\
        --cv-subsample 0.05 \\
        --cv-estimators 20

Output
------
results/bench_v1.csv  — one row per config:
  config, seed, n_points_total, n_points_cv, cv_time_s, ram_peak_mb,
  f1_macro, f1_floor, f1_cargo, f1_vehicle, f1_person, f1_pallet

Stdout: progress + comparison table at the end.
"""

import argparse
import csv
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Resolve src/ so we can import train utilities
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from classifier.train import (
    load_dataset,
    apply_floor_subsample,
    compute_class_sample_weights,
    run_cv,
    _parse_floor_subsample,
    _parse_class_weight_mult,
)
from classifier.features import FEATURE_NAMES
from generate_dataset import LABEL
from sklearn.preprocessing import StandardScaler


# ── Config registry ───────────────────────────────────────────────────────────

CONFIGS = {
    "A_baseline": {
        "floor_subsample": "none",
        "class_weight_mult": "",
        "description": "with_floor + balanced (reference)",
    },
    "C_contextual": {
        "floor_subsample": "contextual:1.0,bulk:0.1",
        "class_weight_mult": "person:2,pallet:2",
        "description": "--floor-subsample contextual:1.0,bulk:0.1 --class-weight-mult person:2,pallet:2",
    },
}

CLASS_ORDER = ["floor", "cargo", "vehicle", "person", "pallet"]


# ── RAM peak tracking ─────────────────────────────────────────────────────────

def _peak_ram_mb() -> float:
    """Current tracemalloc peak in MB (since last reset)."""
    _, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024


# ── Per-config runner ─────────────────────────────────────────────────────────

def run_bench_config(
    config_name: str,
    config: dict,
    X_full: np.ndarray,
    y_full: np.ndarray,
    groups_full: np.ndarray,
    pts_xyz_full: np.ndarray,
    seed: int,
    cv_subsample: float,
    cv_estimators: int,
    n_jobs: int,
) -> dict:
    """
    Run CV for one config.  Returns a result dict suitable for the CSV row.
    X_full, y_full, groups_full, pts_xyz_full — pre-loaded shared arrays.
    Modifications (floor subsampling) produce NEW arrays without touching originals.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"CONFIG: {config_name}  —  {config['description']}", flush=True)
    print(f"{'='*60}", flush=True)

    tracemalloc.reset_peak()

    floor_spec = _parse_floor_subsample(config["floor_subsample"])
    cw_mult    = _parse_class_weight_mult(config["class_weight_mult"])

    # Apply floor subsampling (produces new arrays, originals unchanged)
    if floor_spec is not None:
        X, y, groups, pts_xyz = apply_floor_subsample(
            X_full, y_full, groups_full, pts_xyz_full,
            floor_spec, seed=seed,
        )
    else:
        X, y, groups = X_full, y_full, groups_full

    n_points_total = len(X)

    # Scale (fit on this config's data so floor-subsampled config scales correctly)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Sample weights
    sample_weight = None
    if cw_mult:
        sample_weight = compute_class_sample_weights(y, cw_mult)
        for cls_name, mult in cw_mult.items():
            print(f"  class-weight-mult: {cls_name} ×{mult}", flush=True)

    # LGBM factory
    def make_lgbm():
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            num_leaves=63,
            n_estimators=cv_estimators,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )

    # CV — LGBM only
    cv_result = run_cv(
        make_lgbm, X_scaled, y, groups,
        model_name=f"LightGBM [{config_name}]",
        n_splits=5,
        subsample=cv_subsample,
        seed=seed,
        sample_weight=sample_weight,
    )

    ram_mb = _peak_ram_mb()

    row = {
        "config":         config_name,
        "seed":           seed,
        "n_points_total": n_points_total,
        "n_points_cv":    cv_result["n_points_cv"],
        "cv_time_s":      round(cv_result["cv_time_s"], 1),
        "ram_peak_mb":    round(ram_mb, 1),
        "f1_macro":       round(cv_result["f1_macro"],   4),
        "f1_floor":       round(cv_result["f1_floor"],   4),
        "f1_cargo":       round(cv_result["f1_cargo"],   4),
        "f1_vehicle":     round(cv_result["f1_vehicle"], 4),
        "f1_person":      round(cv_result["f1_person"],  4),
        "f1_pallet":      round(cv_result["f1_pallet"],  4),
    }
    return row


# ── Comparison table ──────────────────────────────────────────────────────────

def _print_comparison(rows: list[dict]) -> None:
    """Print a side-by-side comparison of A_baseline vs C_contextual."""
    if len(rows) < 2:
        print("\n(Not enough rows for comparison)", flush=True)
        return

    a = next((r for r in rows if r["config"] == "A_baseline"),   None)
    c = next((r for r in rows if r["config"] == "C_contextual"), None)
    if a is None or c is None:
        return

    cols = ["f1_macro"] + [f"f1_{cl}" for cl in CLASS_ORDER] + ["cv_time_s", "ram_peak_mb"]
    labels = (
        ["F1-macro"] + [f"F1-{cl}" for cl in CLASS_ORDER] + ["time(s)", "RAM(MB)"]
    )

    header = f"{'Metric':<18} {'A_baseline':>12} {'C_contextual':>14} {'delta':>10}"
    sep    = "-" * len(header)
    print(f"\n{'='*len(header)}", flush=True)
    print("BENCH COMPARISON: C_contextual vs A_baseline", flush=True)
    print(f"{'='*len(header)}", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for col, lbl in zip(cols, labels):
        va = float(a[col])
        vc = float(c[col])
        delta = vc - va
        # Color-code F1 deltas: + is good
        if col.startswith("f1_"):
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = f"{delta:+.1f}"
        fmt_a = f"{va:.4f}" if col.startswith("f1_") else f"{va:.1f}"
        fmt_c = f"{vc:.4f}" if col.startswith("f1_") else f"{vc:.1f}"
        print(f"  {lbl:<16} {fmt_a:>12} {fmt_c:>14} {delta_str:>10}", flush=True)
    print(sep, flush=True)

    # Quick verdict
    print("\nVERDICT:", flush=True)
    d_person  = float(c["f1_person"])  - float(a["f1_person"])
    d_pallet  = float(c["f1_pallet"])  - float(a["f1_pallet"])
    d_macro   = float(c["f1_macro"])   - float(a["f1_macro"])
    d_time    = float(c["cv_time_s"])  - float(a["cv_time_s"])
    for line in [
        f"  person  {d_person:+.4f}  ({'↑ gains' if d_person > 0.005 else '↓ loses' if d_person < -0.005 else '≈ tie'})",
        f"  pallet  {d_pallet:+.4f}  ({'↑ gains' if d_pallet > 0.005 else '↓ loses' if d_pallet < -0.005 else '≈ tie'})",
        f"  macro   {d_macro:+.4f}  ({'↑ gains' if d_macro > 0.002 else '↓ loses' if d_macro < -0.002 else '≈ tie'})",
        f"  CV time {d_time:+.1f}s  ({'faster' if d_time < -10 else 'slower' if d_time > 10 else '≈ same'})",
    ]:
        print(line, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 2 training configs (LGBM CV only). "
                    "Loads dataset once, runs A_baseline and C_contextual."
    )
    parser.add_argument("--data", default="../output/dataset",
                        help="Path to dataset directory with PLY files")
    parser.add_argument("--results", default="../results/bench_v1.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-subsample", type=float, default=0.3,
                        help="Fraction of pts for CV folds (default 0.3)")
    parser.add_argument("--cv-estimators", type=int, default=100,
                        help="LGBM n_estimators in CV (default 100)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    data_dir    = Path(args.data)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()

    # ── Load dataset ONCE ────────────────────────────────────────────────────
    print(f"\nLoading dataset from {data_dir} ...", flush=True)
    X_full, y_full, groups_full, pts_xyz_full = load_dataset(
        data_dir, n_jobs=args.n_jobs
    )
    print(f"Dataset loaded: {len(X_full):,} pts, {len(FEATURE_NAMES)} features", flush=True)

    # ── Run 2 configs ────────────────────────────────────────────────────────
    rows = []
    t_bench_start = time.time()

    for config_name, config in CONFIGS.items():
        row = run_bench_config(
            config_name=config_name,
            config=config,
            X_full=X_full,
            y_full=y_full,
            groups_full=groups_full,
            pts_xyz_full=pts_xyz_full,
            seed=args.seed,
            cv_subsample=args.cv_subsample,
            cv_estimators=args.cv_estimators,
            n_jobs=args.n_jobs,
        )
        rows.append(row)
        print(f"\n  → {config_name}: F1-macro={row['f1_macro']:.4f}  "
              f"time={row['cv_time_s']:.1f}s  RAM={row['ram_peak_mb']:.0f}MB",
              flush=True)

    total_time = time.time() - t_bench_start
    print(f"\nTotal bench time: {total_time:.1f}s", flush=True)

    # ── Write CSV ────────────────────────────────────────────────────────────
    fieldnames = [
        "config", "seed", "n_points_total", "n_points_cv",
        "cv_time_s", "ram_peak_mb",
        "f1_macro", "f1_floor", "f1_cargo", "f1_vehicle", "f1_person", "f1_pallet",
    ]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to: {results_path}", flush=True)

    # ── Comparison table ─────────────────────────────────────────────────────
    _print_comparison(rows)

    tracemalloc.stop()


if __name__ == "__main__":
    main()
