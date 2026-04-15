#!/usr/bin/env python3
"""
Volume estimation evaluation — synthetic GT comparison + BBB scene report.

Usage (from datageneration/ root):
    python3 scripts/eval_volume.py [--synth-dir DIR] [--bbb-dir DIR] \
            [--cluster-classifier PATH] [--out-dir DIR]

Synthetic evaluation
--------------------
Ground-truth volume for each scene is computed from the cargo geometry stored
in metadata.json (box: w×h×d; cylinder: π·r²·h; multi-cargo: sum of all items).
The pipeline runs on each PLY and the height-field volume is compared to GT.

Metrics reported: mean, median, p90 absolute/relative error; signed bias.

BBB evaluation
--------------
No absolute GT is available.  The script prints height-field volume next to the
legacy OBB-based estimate (from cluster_features volume_obb) so the user can
validate visually.  Results are saved to <out_dir>/bbb_results.json.

Run from datageneration/ root.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cargo_geometric.params import GeometricParams                # noqa: E402
from cargo_geometric.floor import preprocess, remove_floor        # noqa: E402
from cargo_geometric.anchor import find_anchor                    # noqa: E402
from cargo_geometric.cargo import extract_cargo                   # noqa: E402
from cargo_geometric.volume import height_field_volume            # noqa: E402


# ── GT helpers ────────────────────────────────────────────────────────────────

def _cargo_gt_volume(objects: list) -> float:
    """Sum GT cargo volumes from a metadata.json objects list.

    Handles single cargo, stacked, tandem, and cargo-on-vehicle modes.
    Ignores non-cargo objects (floor, pallet, pallet_jack, person).
    """
    total = 0.0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for key, spec in obj.items():
            if not key.startswith("cargo"):
                continue
            t = spec.get("type")
            if t == "box":
                total += spec["w"] * spec["h"] * spec["d"]
            elif t == "cylinder":
                total += math.pi * spec["r"] ** 2 * spec["h"]
    return total


# ── Pipeline helper ───────────────────────────────────────────────────────────

def _run_pipeline(
    ply_path: Path,
    params: GeometricParams,
    classifier=None,
) -> dict | None:
    """Run the geometric pipeline on one PLY.

    Returns a dict with ``volume_hf`` (height-field), ``n_cargo_pts``, and
    ``floor_y``; returns None if cargo extraction fails.
    """
    try:
        pts = preprocess(ply_path, params)
    except Exception as exc:
        warnings.warn(f"{ply_path.name}: preprocess failed — {exc}")
        return None

    floor = remove_floor(pts, params)
    nonfloor = pts[~floor.floor_mask]

    anchor = find_anchor(nonfloor, floor.floor_y, params)
    if anchor is None:
        return None

    cargo_res = extract_cargo(nonfloor, anchor, params, classifier=classifier)
    if cargo_res is None or len(cargo_res.cargo_pts) == 0:
        return None

    vol = height_field_volume(cargo_res.cargo_pts, floor.floor_y)
    return {
        "volume_hf":    vol["volume_m3"],
        "footprint_m2": vol["footprint_m2"],
        "max_height":   vol["max_height"],
        "mean_height":  vol["mean_height"],
        "n_cargo_pts":  int(len(cargo_res.cargo_pts)),
        "cargo_source": cargo_res.cargo_source,
        "cargo_policy": cargo_res.cargo_policy,
        "floor_y":      float(floor.floor_y),
    }


# ── Synthetic evaluation ──────────────────────────────────────────────────────

def eval_synthetic(
    synth_dir: Path,
    params: GeometricParams,
    classifier=None,
) -> list[dict]:
    """Evaluate height-field volume on the synthetic dataset.

    Returns one row per scene: id, gt_volume, pred_volume, abs_err, rel_err.
    """
    meta_path = synth_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[eval_synth] metadata.json not found at {meta_path}", file=sys.stderr)
        return []

    with open(meta_path) as f:
        metadata = json.load(f)

    results = []
    n_total = len(metadata)
    for i, m in enumerate(metadata):
        scene_id = m["id"]
        ply_path = synth_dir / f"{scene_id:05d}.ply"
        if not ply_path.exists():
            continue

        gt_vol = _cargo_gt_volume(m["objects"])
        if gt_vol <= 0:
            # No cargo in scene (edge case) — skip
            continue

        res = _run_pipeline(ply_path, params, classifier=classifier)

        if res is None:
            print(f"  [{scene_id:4d}/{n_total}] SKIP — pipeline returned None")
            continue

        pred_vol = res["volume_hf"]
        abs_err  = abs(pred_vol - gt_vol)
        rel_err  = abs_err / gt_vol if gt_vol > 0 else float("nan")
        sign_err = (pred_vol - gt_vol) / gt_vol if gt_vol > 0 else float("nan")

        results.append({
            "id":         scene_id,
            "gt_m3":      round(gt_vol,  4),
            "pred_m3":    round(pred_vol, 4),
            "abs_err_m3": round(abs_err,  4),
            "rel_err":    round(rel_err,  4),
            "sign_err":   round(sign_err, 4),
            "source":     res["cargo_source"],
        })

        if (i + 1) % 20 == 0 or (i + 1) == n_total:
            done = len(results)
            print(f"  [{i+1:4d}/{n_total}] processed {done} scenes so far …")

    return results


def _print_synth_report(rows: list[dict]) -> None:
    if not rows:
        print("  No results to report.")
        return

    rel_errs  = np.array([r["rel_err"]  for r in rows])
    sign_errs = np.array([r["sign_err"] for r in rows])
    abs_errs  = np.array([r["abs_err_m3"] for r in rows])

    print(f"\n  Scenes evaluated : {len(rows)}")
    print(f"  Abs error  mean  : {np.mean(abs_errs):.4f} m³")
    print(f"  Abs error  median: {np.median(abs_errs):.4f} m³")
    print(f"  Abs error  p90   : {np.percentile(abs_errs, 90):.4f} m³")
    print(f"  Rel error  mean  : {np.mean(rel_errs)*100:.1f}%")
    print(f"  Rel error  median: {np.median(rel_errs)*100:.1f}%")
    print(f"  Rel error  p90   : {np.percentile(rel_errs, 90)*100:.1f}%")
    print(f"  Signed rel bias  : {np.mean(sign_errs)*100:+.1f}%  "
          f"(+ve = over-estimate)")

    # Histogram of relative error buckets
    buckets = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1e9)]
    print("\n  Relative error distribution:")
    for lo, hi in buckets:
        count = int(np.sum((rel_errs >= lo) & (rel_errs < hi)))
        pct   = 100 * count / len(rows)
        label = f"  <{hi*100:.0f}%" if hi < 1e9 else f"  ≥{lo*100:.0f}%"
        print(f"    [{lo*100:4.0f}% – {hi*100 if hi<1e9 else '∞':>4}%)  "
              f"{count:4d}  ({pct:5.1f}%)")


# ── BBB evaluation ────────────────────────────────────────────────────────────

def eval_bbb(
    bbb_dir: Path,
    params: GeometricParams,
    classifier=None,
) -> list[dict]:
    """Run pipeline on each BBB voxelised PLY and report volume."""
    ply_files = sorted(bbb_dir.glob("*.ply"))
    if not ply_files:
        print(f"  No PLY files found in {bbb_dir}", file=sys.stderr)
        return []

    results = []
    for ply_path in ply_files:
        res = _run_pipeline(ply_path, params, classifier=classifier)
        row: dict = {"scene": ply_path.name}
        if res is None:
            row.update({
                "volume_hf_m3": None,
                "footprint_m2": None,
                "max_height_m": None,
                "n_cargo_pts":  None,
                "note":         "pipeline_failed",
            })
        else:
            row.update({
                "volume_hf_m3": round(res["volume_hf"],    3),
                "footprint_m2": round(res["footprint_m2"], 3),
                "max_height_m": round(res["max_height"],   3),
                "n_cargo_pts":  res["n_cargo_pts"],
                "cargo_source": res["cargo_source"],
                "cargo_policy": res["cargo_policy"],
            })
        results.append(row)

    return results


def _print_bbb_report(rows: list[dict]) -> None:
    print(f"\n  {'Scene':<30}  {'HF vol (m³)':>12}  {'Footprint (m²)':>14}  "
          f"{'MaxH (m)':>9}  {'CargoPoints':>11}")
    print("  " + "-" * 82)
    for r in rows:
        v   = r.get("volume_hf_m3")
        fp  = r.get("footprint_m2")
        mh  = r.get("max_height_m")
        cp  = r.get("n_cargo_pts")
        v_s  = f"{v:.3f}" if v  is not None else "FAIL"
        fp_s = f"{fp:.3f}" if fp is not None else "—"
        mh_s = f"{mh:.3f}" if mh is not None else "—"
        cp_s = f"{cp:>11d}" if cp is not None else "—"
        print(f"  {r['scene']:<30}  {v_s:>12}  {fp_s:>14}  {mh_s:>9}  {cp_s}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Volume estimation evaluation")
    p.add_argument(
        "--synth-dir", type=Path,
        default=ROOT / "output" / "dataset",
        help="Synthetic dataset directory (contains metadata.json + *.ply).",
    )
    p.add_argument(
        "--bbb-dir", type=Path,
        default=ROOT / "output" / "bbb_vox035",
        help="BBB voxelised PLY directory.",
    )
    p.add_argument(
        "--cluster-classifier", dest="cluster_classifier", type=Path,
        default=None, metavar="PATH",
        help="Path to cluster classifier .pkl (optional).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "output" / "eval_volume",
        help="Output directory for JSON results.",
    )
    p.add_argument(
        "--skip-synth", action="store_true",
        help="Skip synthetic evaluation (faster iteration on BBB only).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = GeometricParams()

    # Load classifier (optional)
    classifier = None
    if args.cluster_classifier is not None:
        try:
            from cargo_geometric.cluster_classifier import ClusterClassifier
            classifier = ClusterClassifier.load(args.cluster_classifier)
            print(f"[eval_volume] Classifier loaded: {args.cluster_classifier.name}")
        except Exception as exc:
            warnings.warn(f"Could not load classifier ({exc}) — using rank-0 fallback")

    # ── Synthetic ──────────────────────────────────────────────────────────────
    if not args.skip_synth:
        print("\n══ Synthetic evaluation ══════════════════════════════════════")
        synth_rows = eval_synthetic(args.synth_dir, params, classifier=classifier)
        _print_synth_report(synth_rows)

        synth_out = args.out_dir / "synth_results.json"
        synth_out.write_text(json.dumps(synth_rows, indent=2))
        print(f"\n  Saved → {synth_out}")
    else:
        synth_rows = []
        print("\n[eval_volume] Synthetic evaluation skipped (--skip-synth).")

    # ── BBB ────────────────────────────────────────────────────────────────────
    print("\n══ BBB evaluation ════════════════════════════════════════════════")
    bbb_rows = eval_bbb(args.bbb_dir, params, classifier=classifier)
    _print_bbb_report(bbb_rows)

    bbb_out = args.out_dir / "bbb_results.json"
    bbb_out.write_text(json.dumps(bbb_rows, indent=2))
    print(f"\n  Saved → {bbb_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
