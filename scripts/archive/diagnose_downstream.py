#!/usr/bin/env python3
"""Diagnose YOLO → 3D projection downstream failure modes.

For each (esc, cap, cam):
 1. YOLO person bbox
 2. Project per-camera PLY to pixel coords using CAMERAS intrinsics
 3. Report:
    - bbox touches image edge? (person cut off)
    - #points projected in-frame
    - #points inside bbox
    - percentile of z for in-bbox vs overall (plausibility)
 4. Write overlay PNG: original + bbox (red) + in-bbox projected pts (green)
    + out-bbox projected pts (blue, subsample) for visual alignment check
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_person_yolo import (
    CAMERAS, best_person_detection, get_yolo,
    project_points, read_ply_binary_xyzrgb, mask_points_in_bbox,
)

BBB_ROOT = Path(
    "/home/lronquilloext/Documents/Logicarc/Resources/Capturas_BBB/"
    "2026_03_27/Escenarios_CATEC_25032026/BBB"
)
OUT = Path("/home/lronquilloext/Documents/Logicarc/datageneration/"
           "results/diagnose_overlay")

# Representative captures: mix of working + known suspect
TARGETS = [
    (4, 2, "Der"), (4, 7, "Der"), (4, 10, "Der"),
    (6, 1, "Izq"), (6, 4, "Izq"), (6, 7, "Izq"),
    (6, 1, "Der"), (6, 5, "Der"), (6, 9, "Der"),
    # Reference working case (from §17)
    (7, 1, "Izq"), (7, 1, "Der"),
]

EDGE_PX = 10  # bbox within this many px of image edge = "touches edge"


def edge_flags(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return {
        "left":   x1 <= EDGE_PX,
        "top":    y1 <= EDGE_PX,
        "right":  x2 >= w - EDGE_PX,
        "bottom": y2 >= h - EDGE_PX,
    }


def overlay(img, bbox, u, v, valid, in_bbox, out_path):
    vis = img.copy()
    # Out-of-bbox projected points (subsample) — blue
    out_mask = valid & ~in_bbox
    idx = np.where(out_mask)[0]
    if len(idx) > 5000:
        idx = np.random.choice(idx, 5000, replace=False)
    for i in idx:
        cv2.circle(vis, (int(u[i]), int(v[i])), 1, (255, 80, 0), -1)
    # In-bbox projected points — green
    idx_in = np.where(in_bbox)[0]
    for i in idx_in:
        cv2.circle(vis, (int(u[i]), int(v[i])), 1, (0, 220, 0), -1)
    # Bbox — red
    x1, y1, x2, y2 = [int(v_) for v_ in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resize for manageable file size
    h, w = vis.shape[:2]
    scale = 0.5
    vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    get_yolo()
    print(f"{'Esc':>3} {'Cap':>3} {'Cam':>4} {'conf':>5} "
          f"{'edges':>15} {'in_fr':>7} {'in_box':>7} "
          f"{'z_med_box':>9} {'z_med_all':>9}  note")
    print("-" * 100)
    for esc, cap, cam in TARGETS:
        png = BBB_ROOT / f"Escenario_{esc:02d}" / f"Captura_{cap:02d}" / f"PNG_{cam}.png"
        ply = BBB_ROOT / f"Escenario_{esc:02d}" / f"Captura_{cap:02d}" / f"PLY_{cam}.ply"
        if not png.exists() or not ply.exists():
            print(f"{esc:>3} {cap:>3} {cam:>4}  MISSING")
            continue
        img = cv2.imread(str(png))
        if img is None:
            print(f"{esc:>3} {cap:>3} {cam:>4}  PNG unreadable")
            continue
        det = best_person_detection(img, conf=0.5) or best_person_detection(img, conf=0.3)
        if det is None:
            print(f"{esc:>3} {cap:>3} {cam:>4}  NO DETECTION")
            continue
        bbox = det["bbox"]
        h, w = img.shape[:2]
        eflag = edge_flags(bbox, w, h)
        edge_str = "".join(k[0].upper() if v else "." for k, v in eflag.items())

        xyz, _ = read_ply_binary_xyzrgb(ply)
        u, v, valid = project_points(xyz, CAMERAS[cam])
        in_bbox = mask_points_in_bbox(u, v, valid, bbox)
        n_in_frame = int(valid.sum())
        n_in_box = int(in_bbox.sum())
        z_all = xyz[valid, 2]
        z_box = xyz[in_bbox, 2]
        z_med_all = float(np.median(z_all)) if len(z_all) else float("nan")
        z_med_box = float(np.median(z_box)) if len(z_box) else float("nan")

        note = []
        if any(eflag.values()):
            note.append(f"bbox@edge[{edge_str}]")
        if n_in_box < 500:
            note.append("FEW_PTS")
        if n_in_box > 0 and abs(z_med_box - z_med_all) < 0.1:
            note.append("z_overlap_scene")

        print(f"{esc:>3} {cap:>3} {cam:>4} {det['conf']:.2f} "
              f"{edge_str:>15} {n_in_frame:>7} {n_in_box:>7} "
              f"{z_med_box:>9.2f} {z_med_all:>9.2f}  {' '.join(note)}")

        overlay(img, bbox, u, v, valid, in_bbox,
                OUT / f"Esc{esc:02d}_Cap{cap:02d}_{cam}.jpg")

    print(f"\nOverlays: {OUT}")


if __name__ == "__main__":
    main()
