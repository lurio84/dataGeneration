#!/usr/bin/env python3
"""Diagnose YOLOv8n COCO person detection on failing BBB scenarios.

Runs YOLO over Esc04 Der and Esc06 Izq/Der for all capturas (01..10)
at multiple confidence thresholds and reports best person detection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

BBB_ROOT = Path(
    "/home/lronquilloext/Documents/Logicarc/Resources/Capturas_BBB/"
    "2026_03_27/Escenarios_CATEC_25032026/BBB"
)

TARGETS = [
    (4, "Der"),
    (6, "Izq"),
    (6, "Der"),
]

THRESHOLDS = [0.5, 0.3, 0.1, 0.05]


def run(model: YOLO, png: Path, conf: float):
    img = cv2.imread(str(png))
    if img is None:
        return None, None
    res = model(img, conf=conf, imgsz=640, verbose=False)[0]
    best = None
    for box in res.boxes:
        if int(box.cls[0]) != 0:
            continue
        c = float(box.conf[0])
        if best is None or c > best[0]:
            xyxy = box.xyxy[0].tolist()
            best = (c, xyxy)
    return img.shape[:2], best


def main() -> None:
    model = YOLO("yolov8n.pt")
    print(f"{'Esc':>3} {'Cap':>3} {'Cam':>5} {'imgHW':>11} "
          f"{'best_conf':>9} {'bbox (x1,y1,x2,y2)':>30} {'thr_first_hit':>13}")
    print("-" * 85)
    for esc, cam in TARGETS:
        for cap in range(1, 11):
            png = (BBB_ROOT / f"Escenario_{esc:02d}"
                   / f"Captura_{cap:02d}" / f"PNG_{cam}.png")
            if not png.exists():
                print(f"{esc:>3} {cap:>3} {cam:>5}  MISSING")
                continue
            hw, _ = run(model, png, conf=0.01)  # grab overall best w/ near-zero
            best_conf = None
            best_bbox = None
            thr_hit = None
            for thr in THRESHOLDS:
                hw_, best = run(model, png, conf=thr)
                if best is not None:
                    best_conf, best_bbox = best
                    thr_hit = thr
                    break
            if best_bbox is None:
                _, any_det = run(model, png, conf=0.01)
                if any_det is not None:
                    best_conf, best_bbox = any_det
                    thr_hit = "<0.05"
            bbox_str = (f"[{best_bbox[0]:.0f},{best_bbox[1]:.0f},"
                        f"{best_bbox[2]:.0f},{best_bbox[3]:.0f}]"
                        if best_bbox else "NONE")
            hw_str = f"{hw[0]}x{hw[1]}" if hw else "?"
            conf_str = f"{best_conf:.3f}" if best_conf is not None else "---"
            print(f"{esc:>3} {cap:>3} {cam:>5} {hw_str:>11} "
                  f"{conf_str:>9} {bbox_str:>30} {str(thr_hit):>13}")


if __name__ == "__main__":
    main()
