#!/usr/bin/env python3
"""
Geometric pipeline evaluation v2 — anchor ConvexHull as primary metric.

Generates:
  output/eval_real_20esc/comparison_v2.csv
  output/eval_real_20esc/summary_v2.txt

Runs from: /home/lronquilloext/Documents/Logicarc/datageneration/
  python3 scripts/eval_real_20esc_v2.py
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, stdev

REPO_ROOT    = Path(__file__).resolve().parents[1]
RESOURCES    = REPO_ROOT.parent / "Resources"
TIME_PROCESS = RESOURCES / "time_process"
CLUSTERING   = RESOURCES / "clustering"
OUTPUT_ROOT  = REPO_ROOT / "output" / "eval_real_20esc"
RUNS_DIR     = OUTPUT_ROOT / "runs_v2"
CSV_OUT      = OUTPUT_ROOT / "comparison_v2.csv"
SUMMARY_OUT  = OUTPUT_ROOT / "summary_v2.txt"
RUN_GEOM     = REPO_ROOT / "scripts" / "run_geometric.py"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ── Paula volume parser ────────────────────────────────────────────────────────

VOLUME_RE = re.compile(
    r"\[GetVolume\] Best estimation with (?:ConvexHull|MomentOfIntertia): "
    r"([\d.]+) m3"
)


def parse_paula_volume(esc_dir: Path, ply_stem: str) -> tuple[float | None, int]:
    pattern = f"{ply_stem}_C*_get_bounding_box.txt"
    files = sorted(esc_dir.glob(pattern))
    if not files:
        return None, 0
    total = 0.0
    for f in files:
        text = f.read_text(errors="replace")
        m = VOLUME_RE.search(text)
        if m:
            total += float(m.group(1))
    return total, len(files)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(ply_path: Path, run_dir: Path) -> tuple[dict | None, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUN_GEOM),
        str(ply_path), str(run_dir),
        "--preprocessed",
    ]
    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=300,
    )
    (run_dir / "stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        return None, result.stderr
    json_path = run_dir / f"{ply_path.stem}_stage123.json"
    if not json_path.exists():
        return None, f"JSON not found: {json_path}"
    try:
        return json.loads(json_path.read_text()), result.stderr
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


# ── Result extractor ───────────────────────────────────────────────────────────

def extract_results(meta: dict) -> dict:
    """Return dict with vol_anchor_ch, vol_hf, dims_anchor, n_clusters_pipe."""
    r: dict = {
        "vol_anchor_ch": 0.0,
        "vol_hf":        0.0,
        "dims_anchor":   "",
        "n_clusters_pipe": None,
    }
    if meta.get("volume") is not None:
        r["vol_hf"] = meta["volume"].get("volume_m3", 0.0)
    if meta.get("anchor_volume") is not None:
        av = meta["anchor_volume"]
        r["vol_anchor_ch"] = av.get("anchor_volume_ch_m3", 0.0)
        ext = av.get("anchor_extents_m", [])
        if len(ext) == 3:
            # Sort descending: L × W × H (longest first)
            s = sorted(ext, reverse=True)
            r["dims_anchor"] = f"{s[0]:.3f}x{s[1]:.3f}x{s[2]:.3f}"
    cargo = meta.get("cargo")
    if cargo:
        r["n_clusters_pipe"] = cargo.get("n_clusters_in_anchor")
    return r


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ply_files = sorted(TIME_PROCESS.glob("Escenario_*/Captura_*_tri_cloud.ply"))
    print(f"Found {len(ply_files)} PLY files.")

    rows = []
    crash_list = []

    for ply_path in ply_files:
        esc_name  = ply_path.parent.name
        ply_stem  = ply_path.stem
        esc_match = re.search(r"(\d+)", esc_name)
        cap_match = re.search(r"Captura_(\d+)", ply_stem)
        esc_num   = esc_match.group(1) if esc_match else "XX"
        cap_num   = cap_match.group(1) if cap_match else "YY"
        run_tag   = f"Esc{esc_num}_Cap{cap_num}"
        run_dir   = RUNS_DIR / run_tag

        print(f"[{run_tag}] {ply_path.name}", flush=True)

        # Paula reference
        vol_paula, n_clusters_paula = parse_paula_volume(
            CLUSTERING / esc_name, ply_stem
        )

        # Run pipeline
        try:
            meta, stderr = run_pipeline(ply_path, run_dir)
        except subprocess.TimeoutExpired:
            meta = None
            stderr = "TIMEOUT"
            (run_dir / "stderr.txt").write_text("TIMEOUT\n")

        if meta is None:
            rows.append({
                "escenario":        esc_name,
                "captura":          ply_stem,
                "vol_anchor_ch":    "crash",
                "vol_hf":           "crash",
                "vol_paula":        vol_paula if vol_paula is not None else "",
                "ratio_anchor_vs_paula": "crash",
                "dims_anchor":      "",
                "n_clusters_paula": n_clusters_paula,
                "n_clusters_pipe":  "",
            })
            crash_list.append(run_tag)
            print(f"  CRASH")
            continue

        r = extract_results(meta)

        if vol_paula is not None and vol_paula > 0:
            ratio = round(r["vol_anchor_ch"] / vol_paula, 4)
        else:
            ratio = None

        print(
            f"  anchor_ch={r['vol_anchor_ch']:.3f}  hf={r['vol_hf']:.3f}  "
            f"paula={vol_paula}  ratio={ratio}  dims={r['dims_anchor']}"
        )

        rows.append({
            "escenario":        esc_name,
            "captura":          ply_stem,
            "vol_anchor_ch":    r["vol_anchor_ch"],
            "vol_hf":           r["vol_hf"],
            "vol_paula":        vol_paula if vol_paula is not None else "",
            "ratio_anchor_vs_paula": ratio if ratio is not None else "",
            "dims_anchor":      r["dims_anchor"],
            "n_clusters_paula": n_clusters_paula,
            "n_clusters_pipe":  r["n_clusters_pipe"] if r["n_clusters_pipe"] is not None else "",
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "escenario", "captura",
        "vol_anchor_ch", "vol_hf", "vol_paula",
        "ratio_anchor_vs_paula",
        "dims_anchor",
        "n_clusters_paula", "n_clusters_pipe",
    ]
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {CSV_OUT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    def _numeric(key):
        return [float(r[key]) for r in rows if r[key] not in ("", "crash", None)]

    all_ratio = _numeric("ratio_anchor_vs_paula")
    scenarios = sorted({r["escenario"] for r in rows})

    def _esc_vals(esc, key):
        return [float(r[key]) for r in rows
                if r["escenario"] == esc and r[key] not in ("", "crash", None)]

    outliers = [
        r for r in rows
        if r["ratio_anchor_vs_paula"] not in ("", "crash", None)
        and (float(r["ratio_anchor_vs_paula"]) < 0.4
             or float(r["ratio_anchor_vs_paula"]) > 1.5)
    ]

    lines = []
    lines.append("=" * 78)
    lines.append("GEOMETRIC PIPELINE v2 — ANCHOR CONVEXHULL vs PAULA REFERENCE")
    lines.append("Primary metric: anchor_volume_ch (ConvexHull on full anchor, pre-ML)")
    lines.append("Paula metric: sum of min(CH, MomentOfInertia) per cluster")
    lines.append("=" * 78)
    lines.append(f"Total captures:    {len(rows)}")
    lines.append(f"Crashes:           {len(crash_list)}")
    lines.append(f"Paula ref missing: {sum(1 for r in rows if r['vol_paula'] == '')}")
    lines.append(f"Valid ratio pairs: {len(all_ratio)}")
    lines.append("")

    lines.append("── PER-SCENARIO (ratio_anchor_vs_paula) ─────────────────────────────")
    lines.append(f"{'Scenario':<15} {'N':>4} {'Mean':>7} {'Median':>7} {'Std':>7}  {'Min':>7} {'Max':>7}")
    lines.append("-" * 62)
    for esc in scenarios:
        rv = _esc_vals(esc, "ratio_anchor_vs_paula")
        n = len(rv)
        if n == 0:
            lines.append(f"{esc:<15} {0:>4}   n/a")
        elif n == 1:
            lines.append(f"{esc:<15} {n:>4} {rv[0]:>7.3f} {rv[0]:>7.3f}     n/a  {rv[0]:>7.3f} {rv[0]:>7.3f}")
        else:
            lines.append(
                f"{esc:<15} {n:>4} {mean(rv):>7.3f} {median(rv):>7.3f}"
                f" {stdev(rv):>7.3f}  {min(rv):>7.3f} {max(rv):>7.3f}"
            )
    lines.append("")

    lines.append("── GLOBAL STATS ────────────────────────────────────────────────────")
    if all_ratio:
        lines.append(
            f"  ratio_anchor_vs_paula:  "
            f"median={median(all_ratio):.4f}  mean={mean(all_ratio):.4f}"
            f"  std={stdev(all_ratio) if len(all_ratio) > 1 else 0:.4f}"
            f"  [{min(all_ratio):.3f} – {max(all_ratio):.3f}]"
        )
    lines.append("")

    lines.append("── OUTLIERS ratio < 0.4 or > 1.5 ───────────────────────────────────")
    if outliers:
        for r in outliers:
            lines.append(
                f"  {r['escenario']:15s} {r['captura']:35s} "
                f"ratio={r['ratio_anchor_vs_paula']:<6}  "
                f"anchor_ch={r['vol_anchor_ch']}  paula={r['vol_paula']}"
            )
    else:
        lines.append("  None.")
    lines.append("")

    lines.append("── CRASH LIST ──────────────────────────────────────────────────────")
    lines.append("  " + (", ".join(crash_list) if crash_list else "None."))
    lines.append("=" * 78)

    txt = "\n".join(lines)
    SUMMARY_OUT.write_text(txt)
    print(txt)
    print(f"Summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
