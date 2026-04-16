#!/usr/bin/env python3
"""
Systematic evaluation of the geometric pipeline against Paula's reference volumes
for all Escenario_01..20 scenarios.

Run from: /home/lronquilloext/Documents/Logicarc/datageneration/
  python3 scripts/eval_real_20esc.py
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, stdev

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
RESOURCES    = REPO_ROOT.parent / "Resources"
TIME_PROCESS = RESOURCES / "time_process"
CLUSTERING   = RESOURCES / "clustering"
OUTPUT_ROOT  = REPO_ROOT / "output" / "eval_real_20esc"
RUNS_DIR     = OUTPUT_ROOT / "runs"
CSV_OUT      = OUTPUT_ROOT / "comparison.csv"
SUMMARY_OUT  = OUTPUT_ROOT / "summary.txt"
RUN_GEOM     = REPO_ROOT / "scripts" / "run_geometric.py"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

VOLUME_RE = re.compile(
    r"\[GetVolume\] Best estimation with (?:ConvexHull|MomentOfInertia): "
    r"([\d.]+) m3"
)


def parse_paula_volume(esc_dir: Path, ply_stem: str) -> tuple[float | None, int]:
    """
    Returns (total_volume_m3, n_clusters) for a given captura.
    Looks for files: <ply_stem>_C*_get_bounding_box.txt
    Returns (None, 0) if no reference files found.
    """
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


def run_pipeline(ply_path: Path, run_dir: Path) -> tuple[dict | None, str]:
    """
    Runs run_geometric.py on ply_path, writing outputs to run_dir.
    Returns (json_meta_dict, stderr_str).
    json_meta_dict is None on crash (non-zero return code).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = run_dir / "stderr.txt"

    cmd = [
        sys.executable,
        str(RUN_GEOM),
        str(ply_path),
        str(run_dir),
        "--preprocessed",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Always write stderr
    stderr_log.write_text(result.stderr)

    if result.returncode != 0:
        return None, result.stderr

    # Find JSON output
    stem = ply_path.stem
    json_path = run_dir / f"{stem}_stage123.json"
    if not json_path.exists():
        return None, f"JSON not found: {json_path}"

    try:
        meta = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    return meta, result.stderr


def extract_pipeline_results(meta: dict) -> tuple[float, float, list, float, float, int | None]:
    """
    From pipeline JSON meta, extract:
      (vol_hf, vol_obb, obb_dims, vol_convhull, anchor_vol_ch, n_clusters_in_anchor)
    All volumes are 0.0 when cargo/anchor is None.
    """
    vol_hf = 0.0
    vol_obb = 0.0
    obb_dims = []
    vol_ch = 0.0
    anchor_ch = 0.0

    if meta.get("volume") is not None:
        vol_hf = meta["volume"].get("volume_m3", 0.0)

    if meta.get("obb") is not None:
        vol_obb  = meta["obb"].get("obb_volume_m3", 0.0)
        obb_dims = meta["obb"].get("obb_dims", [])
        vol_ch   = meta["obb"].get("convhull_volume_m3", 0.0)

    if meta.get("anchor_volume") is not None:
        anchor_ch = meta["anchor_volume"].get("anchor_volume_ch_m3", 0.0)

    cargo = meta.get("cargo")
    n_clusters = cargo.get("n_clusters_in_anchor") if cargo else None
    return vol_hf, vol_obb, obb_dims, vol_ch, anchor_ch, n_clusters


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Collect all PLY files
    ply_files = sorted(TIME_PROCESS.glob("Escenario_*/Captura_*_tri_cloud.ply"))
    print(f"Found {len(ply_files)} PLY files across all scenarios.")

    rows = []  # list of dicts for CSV
    crash_list = []

    for ply_path in ply_files:
        esc_name = ply_path.parent.name          # e.g. "Escenario_03"
        ply_stem = ply_path.stem                 # e.g. "Captura_01_tri_cloud"

        # Parse escenario/captura numbers for output dir naming
        esc_match  = re.search(r"(\d+)", esc_name)
        cap_match  = re.search(r"Captura_(\d+)", ply_stem)
        esc_num    = esc_match.group(1) if esc_match else "XX"
        cap_num    = cap_match.group(1) if cap_match else "YY"
        run_tag    = f"Esc{esc_num}_Cap{cap_num}"
        run_dir    = RUNS_DIR / run_tag

        print(f"\n[{run_tag}] Processing {ply_path.name} ...", flush=True)

        # ── Paula reference ──────────────────────────────────────────────────
        clustering_esc_dir = CLUSTERING / esc_name
        vol_paula, n_clusters_paula = parse_paula_volume(clustering_esc_dir, ply_stem)

        # ── Run pipeline ─────────────────────────────────────────────────────
        try:
            meta, stderr = run_pipeline(ply_path, run_dir)
        except subprocess.TimeoutExpired:
            meta = None
            stderr = "TIMEOUT"
            (run_dir / "stderr.txt").write_text("TIMEOUT\n")

        # ── Parse results ────────────────────────────────────────────────────
        if meta is None:
            vol_hf = vol_obb = vol_ch = anchor_ch = "crash"
            obb_dims_str    = ""
            n_clusters_pipe = None
            ratio_hf = ratio_obb = ratio_ch = ratio_anchor_ch = "crash"
            crash_list.append(run_tag)
            print(f"  CRASH — see {run_dir / 'stderr.txt'}")
        else:
            vol_hf, vol_obb, obb_dims, vol_ch, anchor_ch, n_clusters_pipe = extract_pipeline_results(meta)
            obb_dims_str = (
                f"{obb_dims[0]:.3f}x{obb_dims[1]:.3f}x{obb_dims[2]:.3f}"
                if len(obb_dims) == 3 else ""
            )
            if vol_paula is not None and vol_paula > 0:
                ratio_hf       = round(vol_hf    / vol_paula, 4)
                ratio_obb      = round(vol_obb    / vol_paula, 4)
                ratio_ch       = round(vol_ch     / vol_paula, 4)
                ratio_anchor_ch = round(anchor_ch / vol_paula, 4)
            else:
                ratio_hf = ratio_obb = ratio_ch = ratio_anchor_ch = None
            print(
                f"  paula={vol_paula} m3 ({n_clusters_paula} cl) | "
                f"hf={vol_hf:.3f} obb={vol_obb:.3f} ch={vol_ch:.3f} "
                f"anchor_ch={anchor_ch:.3f} m3 "
                f"({n_clusters_pipe} cl) | "
                f"ratio_hf={ratio_hf} ratio_anchor_ch={ratio_anchor_ch}"
            )

        rows.append({
            "escenario":           esc_name,
            "captura":             ply_stem,
            "n_clusters_paula":    n_clusters_paula if vol_paula is not None else "",
            "vol_paula_m3":        vol_paula if vol_paula is not None else "",
            "n_clusters_pipeline": n_clusters_pipe if n_clusters_pipe is not None else "",
            "vol_hf_m3":           vol_hf,
            "vol_obb_m3":          vol_obb,
            "dims_obb":            obb_dims_str,
            "vol_convhull_m3":     vol_ch,
            "anchor_vol_ch_m3":    anchor_ch,
            "ratio_hf":            ratio_hf        if ratio_hf        is not None else "",
            "ratio_obb":           ratio_obb       if ratio_obb       is not None else "",
            "ratio_ch":            ratio_ch        if ratio_ch        is not None else "",
            "ratio_anchor_ch":     ratio_anchor_ch if ratio_anchor_ch is not None else "",
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "escenario", "captura",
        "n_clusters_paula", "vol_paula_m3",
        "n_clusters_pipeline",
        "vol_hf_m3", "vol_obb_m3", "dims_obb", "vol_convhull_m3",
        "anchor_vol_ch_m3",
        "ratio_hf", "ratio_obb", "ratio_ch", "ratio_anchor_ch",
    ]
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {CSV_OUT}")

    # ── Generate summary ──────────────────────────────────────────────────────
    def _numeric(rows, key):
        return [float(r[key]) for r in rows if r[key] not in ("", "crash", None)]

    all_hf       = _numeric(rows, "ratio_hf")
    all_obb      = _numeric(rows, "ratio_obb")
    all_ch       = _numeric(rows, "ratio_ch")
    all_anchor_ch = _numeric(rows, "ratio_anchor_ch")

    scenarios = sorted({r["escenario"] for r in rows})

    def _esc_vals(esc, key):
        return [float(r[key]) for r in rows
                if r["escenario"] == esc and r[key] not in ("", "crash", None)]

    # Outliers on ratio_hf (primary metric)
    outliers = [
        r for r in rows
        if r["ratio_hf"] not in ("", "crash", None) and (
            float(r["ratio_hf"]) < 0.4 or float(r["ratio_hf"]) > 1.5
        )
    ]

    lines = []
    lines.append("=" * 78)
    lines.append("GEOMETRIC PIPELINE — EVALUATION AGAINST PAULA REFERENCE VOLUMES")
    lines.append("=" * 78)
    lines.append(f"Total captures evaluated: {len(rows)}")
    lines.append(f"Total crashes:            {len(crash_list)}")
    lines.append(f"Paula ref missing:        {sum(1 for r in rows if r['vol_paula_m3'] == '')}")
    lines.append(f"Valid ratio pairs:        {len(all_hf)}")
    lines.append("")

    # Per-scenario stats (height-field, primary)
    lines.append("── PER-SCENARIO STATS (ratio_hf = height-field / paula) ─────────────")
    lines.append(f"{'Scenario':<15} {'N':>4} {'Mean':>7} {'Median':>7} {'Std':>7}")
    lines.append("-" * 50)
    for esc in scenarios:
        rv = _esc_vals(esc, "ratio_hf")
        n = len(rv)
        if n == 0:
            lines.append(f"{esc:<15} {'0':>4}   n/a     n/a     n/a")
        elif n == 1:
            lines.append(f"{esc:<15} {n:>4} {rv[0]:>7.3f} {rv[0]:>7.3f}     n/a")
        else:
            lines.append(
                f"{esc:<15} {n:>4} {mean(rv):>7.3f} {median(rv):>7.3f} {stdev(rv):>7.3f}"
            )
    lines.append("")

    # Global stats — all 3 metrics
    def _gstats(label, vals):
        if not vals:
            lines.append(f"  {label}: no data")
            return
        lines.append(
            f"  {label:<20}  median={median(vals):.4f}  mean={mean(vals):.4f}"
            f"  std={stdev(vals) if len(vals)>1 else 0:.4f}"
            f"  [{min(vals):.3f} – {max(vals):.3f}]"
        )

    lines.append("── GLOBAL STATS (all 4 estimators) ─────────────────────────────────")
    _gstats("ratio_hf        (height-field)",  all_hf)
    _gstats("ratio_obb       (OBB)",           all_obb)
    _gstats("ratio_ch        (CH cargo)",      all_ch)
    _gstats("ratio_anchor_ch (CH anchor/full)", all_anchor_ch)
    lines.append("")

    # Outliers on ratio_hf
    lines.append("── OUTLIERS ratio_hf < 0.4 or > 1.5 ────────────────────────────────")
    if outliers:
        for r in outliers:
            lines.append(
                f"  {r['escenario']:15s} {r['captura']:35s} "
                f"hf={r['ratio_hf']:<6}  ch={r['ratio_ch']:<6}  "
                f"anchor_ch={r['ratio_anchor_ch']:<6}  "
                f"paula={r['vol_paula_m3']}"
            )
    else:
        lines.append("  None.")
    lines.append("")

    lines.append("── CRASH LIST ──────────────────────────────────────────────────────")
    if crash_list:
        for c in crash_list:
            lines.append(f"  {c}")
    else:
        lines.append("  None.")
    lines.append("")
    lines.append("=" * 78)

    summary_text = "\n".join(lines)
    SUMMARY_OUT.write_text(summary_text)
    print(summary_text)
    print(f"\nSummary written: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
