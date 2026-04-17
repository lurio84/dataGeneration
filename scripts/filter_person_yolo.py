#!/usr/bin/env python3
"""
filter_person_yolo.py — Remove person points from merged tri_cloud using
YOLOv8 2D detection + direct index-range mapping (Izq/Der cameras only).

Algorithm
---------
1. For each lateral camera (Izq, Der — Cenital skipped, low detection rate):
   a. Load PNG → YOLOv8n → best person detection (class 0, conf > 0.5)
   b. Load per-camera binary PLY → project all points to 2D with fixed intrinsics
   c. Mark points that project inside the detection bbox
2. Build person mask on merged cloud using verified index ranges:
     [0,          N_izq)             → Izq
     [N_izq,      N_izq + N_der)     → Der
     [N_izq+N_der, total)            → Cenital (never masked here)
3. Read merged cloud (ASCII PLY, PCL format with camera block)
4. Write filtered PLY dropping person points
   - Format: ASCII PLY, no face/camera blocks (CloudCompare-compatible)

Usage
-----
Single escenario/captura:
  python scripts/filter_person_yolo.py \\
      --capturas-dir  Resources/Capturas_BBB/2026_03_27/Escenarios_CATEC_25032026/BBB \\
      --time-process-dir  Resources/time_process \\
      --output-dir  results/filtered \\
      --escenario 4 --captura 1

Batch (all escenarios/capturas found on disk):
  python scripts/filter_person_yolo.py \\
      --capturas-dir  Resources/Capturas_BBB/... \\
      --time-process-dir  Resources/time_process \\
      --output-dir  results/filtered \\
      --batch

Run from: /home/lronquilloext/Documents/Logicarc/datageneration/
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Fixed camera intrinsics (verified 91 % colour-match) ──────────────────────
# Cameras operate at these resolutions; intrinsics are pinhole.
CAMERAS = {
    "Izq": {
        "w": 2048, "h": 1536,
        "fx": 1680.0, "fy": 1680.0, "cx": 1045.0, "cy": 758.0,
    },
    "Der": {
        "w": 2048, "h": 1536,
        "fx": 1664.0, "fy": 1664.0, "cx": 1013.0, "cy": 750.0,
    },
    # Cenital is defined for completeness but not used for person detection
    "Cenital": {
        "w": 1024, "h": 768,
        "fx": 833.0, "fy": 833.0, "cx": 517.0, "cy": 382.0,
    },
}

# Cameras used for person detection (Cenital has low detection rate)
DETECTION_CAMERAS = ["Izq", "Der"]

YOLO_MODEL    = "yolov8n.pt"
CONF_THRESH   = 0.5      # minimum confidence to accept a person detection
BBOX_PAD_PX   = 5        # pixels of padding around detection bbox


# ─────────────────────────────────────────────────────────────────────────────
# PLY I/O
# ─────────────────────────────────────────────────────────────────────────────

def _read_ply_header(f) -> tuple[int, bool, list[str]]:
    """Return (n_verts, is_binary_le, property_names) from an open binary file."""
    n_verts = 0
    is_binary = False
    props: list[str] = []
    for raw in f:
        line = raw.decode("ascii", errors="replace").strip()
        if line.startswith("format binary_little_endian"):
            is_binary = True
        elif line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("property ") and not line.startswith("property list"):
            props.append(line.split()[-1])
        elif line == "end_header":
            break
    return n_verts, is_binary, props


def read_ply_binary_xyzrgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read binary LE PLY with x,y,z,red,green,blue (15 bytes/vertex).
    Handles truncated bodies silently (returns as many complete records as present).
    Returns (xyz [N,3] float32, rgb [N,3] uint8).
    """
    with open(ply_path, "rb") as f:
        n_verts, _, _ = _read_ply_header(f)
        body = f.read()

    record = 15  # 3×float32 + 3×uint8
    n = min(n_verts, len(body) // record)
    if n < n_verts:
        print(f"    [warn] {ply_path.name}: header={n_verts} but only {n} readable vertices")

    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data = np.frombuffer(body[:n * record], dtype=dtype)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
    return xyz, rgb


def read_merged_cloud(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read ASCII tri_cloud PLY produced by PCL (has trailing camera/face blocks).
    Returns (xyz [N,3] float32, rgb [N,3] uint8 or None).
    Stops reading vertex data on any non-numeric line (camera element data).
    """
    pts: list[tuple] = []
    rgbs: list[tuple] = []
    reading = False
    has_rgb = False

    with open(ply_path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s == "end_header":
                reading = True
                continue
            if not reading:
                continue
            parts = s.split()
            if len(parts) < 3:
                break
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                break  # hit camera element data
            pts.append((x, y, z))
            if len(parts) >= 6:
                try:
                    rgbs.append((int(parts[3]), int(parts[4]), int(parts[5])))
                    has_rgb = True
                except ValueError:
                    rgbs.append((128, 128, 128))

    xyz = np.array(pts, dtype=np.float32)
    rgb = np.array(rgbs, dtype=np.uint8) if has_rgb and len(rgbs) == len(pts) else None
    return xyz, rgb


def write_ply_ascii(path: Path, xyz: np.ndarray, rgb: np.ndarray | None) -> None:
    """
    Write ASCII PLY without face/camera blocks (CloudCompare-compatible).
    rgb: (N,3) uint8 or None (writes xyz-only if None).
    """
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "ply\n",
        "format ascii 1.0\n",
        f"element vertex {n}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
    ]
    if rgb is not None:
        lines += [
            "property uchar red\n",
            "property uchar green\n",
            "property uchar blue\n",
        ]
    lines.append("end_header\n")

    with open(path, "w") as fh:
        fh.writelines(lines)
        if rgb is not None:
            for i in range(n):
                x, y, z = xyz[i]
                r, g, b = rgb[i]
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        else:
            for i in range(n):
                x, y, z = xyz[i]
                fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# YOLO helpers
# ─────────────────────────────────────────────────────────────────────────────

_yolo_model = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def best_person_detection(img_bgr: np.ndarray,
                          conf: float = CONF_THRESH,
                          ) -> dict | None:
    """
    Run YOLOv8n on img_bgr. Return the single highest-confidence person
    detection (class 0) with conf ≥ threshold, or None if none found.
    """
    model = get_yolo()
    results = model(img_bgr, conf=conf, imgsz=640, verbose=False)[0]
    best = None
    for box in results.boxes:
        if int(box.cls[0]) != 0:  # COCO class 0 = person
            continue
        c = float(box.conf[0])
        if best is None or c > best["conf"]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = {"conf": c, "bbox": [x1, y1, x2, y2]}
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Projection + masking
# ─────────────────────────────────────────────────────────────────────────────

def project_points(xyz: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points (camera frame) to pixel coords using pinhole model.
    Returns (u_int, v_int, valid_mask) — valid_mask selects points with Z>0
    that project inside the image.
    """
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    w, h = cam["w"], cam["h"]

    z = xyz[:, 2]
    valid_z = z > 0.0
    u = np.where(valid_z, fx * xyz[:, 0] / np.where(valid_z, z, 1.0) + cx, -1.0)
    v = np.where(valid_z, fy * xyz[:, 1] / np.where(valid_z, z, 1.0) + cy, -1.0)
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)
    valid = valid_z & (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    return u_int, v_int, valid


def mask_points_in_bbox(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                        bbox: list[float], pad: int = BBOX_PAD_PX) -> np.ndarray:
    """Boolean mask: True for points whose (u,v) falls inside bbox (+pad)."""
    x1, y1, x2, y2 = bbox
    return (valid
            & (u >= x1 - pad) & (u <= x2 + pad)
            & (v >= y1 - pad) & (v <= y2 + pad))


# ─────────────────────────────────────────────────────────────────────────────
# Single-capture pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_capture(
    cap_dir: Path,
    merged_path: Path,
    out_path: Path,
) -> dict:
    """
    Filter person points from one capture.

    Returns a result dict with statistics:
      n_izq, n_der, n_cenital — points per camera in merged cloud
      n_total, n_removed       — merged cloud totals
      detections               — {cam: {"conf": float, "bbox": [...]} or None}
    """
    result: dict = {
        "cap_dir": str(cap_dir),
        "merged": str(merged_path),
        "output": str(out_path),
        "n_izq": 0, "n_der": 0, "n_cenital": 0,
        "n_total": 0, "n_removed": 0,
        "detections": {},
        "error": None,
    }

    # ── 1. Count per-camera sizes to establish index ranges ──────────────────
    camera_sizes: dict[str, int] = {}
    for cam_name in ("Izq", "Der", "Cenital"):
        ply_path = cap_dir / f"PLY_{cam_name}.ply"
        if not ply_path.exists():
            print(f"    [warn] PLY_{cam_name}.ply missing — treating as 0 pts")
            camera_sizes[cam_name] = 0
            continue
        with open(ply_path, "rb") as f:
            n, _, _ = _read_ply_header(f)
            body_bytes = f.read()
        # Account for truncated bodies
        readable = len(body_bytes) // 15
        if readable < n:
            print(f"    [warn] PLY_{cam_name}.ply truncated: header={n}, readable={readable}")
        camera_sizes[cam_name] = min(n, readable)

    n_izq     = camera_sizes["Izq"]
    n_der     = camera_sizes["Der"]
    n_cenital = camera_sizes["Cenital"]
    result["n_izq"]     = n_izq
    result["n_der"]     = n_der
    result["n_cenital"] = n_cenital

    # Index ranges in merged cloud
    izq_start,     izq_end     = 0,                   n_izq
    der_start,     der_end     = n_izq,               n_izq + n_der
    cenital_start, cenital_end = n_izq + n_der,       n_izq + n_der + n_cenital
    expected_total = n_izq + n_der + n_cenital

    # ── 2. YOLO detection + projection for Izq and Der ───────────────────────
    person_mask_izq = np.zeros(n_izq,  dtype=bool)
    person_mask_der = np.zeros(n_der,  dtype=bool)

    for cam_name, cam_offset, cam_n, cam_mask in [
        ("Izq", izq_start, n_izq, person_mask_izq),
        ("Der", der_start, n_der, person_mask_der),
    ]:
        result["detections"][cam_name] = None
        if cam_n == 0:
            continue

        png_path = cap_dir / f"PNG_{cam_name}.png"
        ply_path = cap_dir / f"PLY_{cam_name}.ply"

        if not png_path.exists():
            print(f"    [{cam_name}] PNG missing — skip")
            continue
        if not ply_path.exists():
            print(f"    [{cam_name}] PLY missing — skip")
            continue

        # Load PNG
        img_bgr = cv2.imread(str(png_path))
        if img_bgr is None:
            print(f"    [{cam_name}] PNG unreadable (possibly truncated) — skip")
            continue

        # YOLO detection
        det = best_person_detection(img_bgr, conf=CONF_THRESH)
        if det is None:
            print(f"    [{cam_name}] no person detected (conf>{CONF_THRESH})")
            continue

        bb = det["bbox"]
        print(f"    [{cam_name}] person conf={det['conf']:.3f} "
              f"bbox=[{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]")
        result["detections"][cam_name] = det

        # Load per-camera PLY
        xyz_cam, _ = read_ply_binary_xyzrgb(ply_path)
        if len(xyz_cam) == 0:
            continue

        # Project to image and mark points inside bbox
        cam_info = CAMERAS[cam_name]
        u, v, valid = project_points(xyz_cam, cam_info)
        in_bbox = mask_points_in_bbox(u, v, valid, bb)

        n_marked = int(in_bbox.sum())
        print(f"    [{cam_name}] {n_marked} of {len(xyz_cam)} camera pts marked as person")

        # Write into the per-camera mask (size matches index range in merged cloud)
        actual_len = min(len(xyz_cam), len(cam_mask))
        cam_mask[:actual_len] |= in_bbox[:actual_len]

    # ── 3. Read merged cloud ─────────────────────────────────────────────────
    if not merged_path.exists():
        result["error"] = f"merged cloud not found: {merged_path}"
        return result

    merged_xyz, merged_rgb = read_merged_cloud(merged_path)
    n_total = len(merged_xyz)
    result["n_total"] = n_total

    if n_total != expected_total:
        print(f"    [warn] merged cloud has {n_total} pts, "
              f"expected {expected_total} (Izq={n_izq}+Der={n_der}+Cen={n_cenital}). "
              f"Index ranges may be off — proceeding with caution.")

    # ── 4. Build global person mask ──────────────────────────────────────────
    global_mask = np.zeros(n_total, dtype=bool)

    # Izq range
    end = min(izq_end, n_total)
    if end > izq_start:
        global_mask[izq_start:end] = person_mask_izq[:end - izq_start]

    # Der range
    end = min(der_end, n_total)
    if end > der_start:
        global_mask[der_start:end] = person_mask_der[:end - der_start]

    n_removed = int(global_mask.sum())
    result["n_removed"] = n_removed
    print(f"    Total person points to remove: {n_removed} / {n_total} "
          f"({100.0 * n_removed / max(n_total, 1):.1f} %)")

    # ── 5. Write filtered PLY ─────────────────────────────────────────────────
    keep = ~global_mask
    out_xyz = merged_xyz[keep]
    out_rgb = merged_rgb[keep] if merged_rgb is not None else None

    write_ply_ascii(out_path, out_xyz, out_rgb)
    print(f"    Saved: {out_path}  ({len(out_xyz)} pts remaining)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove person points from tri_cloud using YOLOv8 2D detection"
    )
    p.add_argument("--capturas-dir",    required=True,
                   help="Path to BBB/ directory containing Escenario_XX/Captura_YY/")
    p.add_argument("--time-process-dir", required=True,
                   help="Path to time_process/ directory")
    p.add_argument("--output-dir",      required=True,
                   help="Output root directory")
    p.add_argument("--escenario", type=int, default=None,
                   help="Escenario number (e.g. 4 → Escenario_04). Required unless --batch.")
    p.add_argument("--captura", type=int, default=None,
                   help="Captura number (e.g. 1 → Captura_01). Required unless --batch.")
    p.add_argument("--batch", action="store_true",
                   help="Process all escenarios/capturas found on disk")
    return p.parse_args()


def collect_captures(capturas_dir: Path, time_process_dir: Path
                     ) -> list[tuple[int, int]]:
    """Return sorted list of (escenario, captura) pairs that have both
    the capture directory and the matching sin_filtro cloud on disk."""
    pairs: list[tuple[int, int]] = []
    for esc_dir in sorted(capturas_dir.glob("Escenario_*")):
        try:
            esc_num = int(esc_dir.name.split("_")[-1])
        except ValueError:
            continue
        for cap_dir in sorted(esc_dir.glob("Captura_*")):
            try:
                cap_num = int(cap_dir.name.split("_")[-1])
            except ValueError:
                continue
            merged = (time_process_dir / f"Escenario_{esc_num:02d}"
                      / f"Captura_{cap_num:02d}_tri_cloud_sin_filtro.ply")
            if merged.exists():
                pairs.append((esc_num, cap_num))
    return pairs


def main() -> None:
    args = parse_args()

    capturas_dir     = Path(args.capturas_dir).resolve()
    time_process_dir = Path(args.time_process_dir).resolve()
    output_dir       = Path(args.output_dir).resolve()

    if not capturas_dir.exists():
        sys.exit(f"[error] capturas-dir not found: {capturas_dir}")
    if not time_process_dir.exists():
        sys.exit(f"[error] time-process-dir not found: {time_process_dir}")

    # Build job list
    if args.batch:
        jobs = collect_captures(capturas_dir, time_process_dir)
        if not jobs:
            sys.exit("[error] no valid escenario/captura pairs found in batch mode")
        print(f"Batch mode: {len(jobs)} captures found")
    else:
        if args.escenario is None or args.captura is None:
            sys.exit("[error] --escenario and --captura are required unless --batch is set")
        jobs = [(args.escenario, args.captura)]

    # Preload YOLO once
    print("Loading YOLOv8n model...")
    get_yolo()
    print("Model loaded.\n")

    all_results = []
    for esc_num, cap_num in jobs:
        esc_name = f"Escenario_{esc_num:02d}"
        cap_name = f"Captura_{cap_num:02d}"

        cap_dir    = capturas_dir / esc_name / cap_name
        merged     = time_process_dir / esc_name / f"{cap_name}_tri_cloud_sin_filtro.ply"
        out_path   = output_dir / esc_name / f"{cap_name}_person_filtered.ply"

        print(f"{'=' * 60}")
        print(f"{esc_name} / {cap_name}")
        print(f"{'=' * 60}")

        try:
            res = process_capture(cap_dir, merged, out_path)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            res = {
                "cap_dir": str(cap_dir), "merged": str(merged),
                "n_total": 0, "n_removed": 0,
                "detections": {}, "error": str(exc),
            }
        res["escenario"] = esc_name
        res["captura"]   = cap_name
        all_results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Scenario':<15} {'Captura':<12} {'Total':>8} {'Removed':>8} "
          f"{'%':>6}  Detections")
    print("-" * 72)
    for r in all_results:
        if r.get("error"):
            print(f"{r.get('escenario','?'):<15} {r.get('captura','?'):<12}  "
                  f"ERROR: {r['error']}")
            continue
        pct = 100.0 * r["n_removed"] / max(r["n_total"], 1)
        det_str = ", ".join(
            f"{cam}:{d['conf']:.2f}"
            for cam, d in r["detections"].items()
            if d is not None
        ) or "none"
        print(f"{r['escenario']:<15} {r['captura']:<12} "
              f"{r['n_total']:>8} {r['n_removed']:>8} {pct:>5.1f}%  {det_str}")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
