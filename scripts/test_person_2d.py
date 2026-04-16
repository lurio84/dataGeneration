#!/usr/bin/env python3
"""
test_person_2d.py — Detect persons in 2D images (YOLOv8), map to 3D via
camera projection, and exclude person points from the fused tri_cloud.

Approach:
  1. YOLOv8n detects "person" in each camera PNG (Cenital, Izq, Der).
  2. Camera projection: estimate pinhole intrinsics (fx, cx, fy, cy) from
     the PLY x/z and y/z distribution, project PLY→image, filter by bbox.
     (Raster-order PGM approach tested and found unreliable for this dataset.)
  3. Person 3D points (camera frame) → approximate world coords via:
       world_Z = camera_height - z_cam   (reliable, pitch=90° for cenital)
       world_XY via cargo-centroid alignment + azimuth search (best of
       0°/90°/180°/270°), with 0.5 m KDTree radius to absorb error.
     Fallback: height-only heuristic on the tri_cloud (guided by YOLO).
  4. Export EscXX_Cap01_person_mask.ply: person=green, floor=grey, rest=red.

Run from: /home/lronquilloext/Documents/Logicarc/datageneration/
  python3 scripts/test_person_2d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import KDTree

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
SRC          = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESOURCES    = REPO_ROOT.parent / "Resources"
BBB_BASE     = RESOURCES / "Capturas_BBB/2026_03_27/Escenarios_CATEC_25032026/BBB"
TIME_PROCESS = RESOURCES / "time_process"
OUT_DIR      = REPO_ROOT / "output" / "eval_real_20esc" / "person_2d"

# ── Camera parameters ─────────────────────────────────────────────────────────
CAMERA_INFO = {
    "Cenital": {"height_m": 3.68, "pitch_deg": 90.0},
    "Izq":     {"height_m": 3.80, "pitch_deg": 47.5},
    "Der":     {"height_m": 3.80, "pitch_deg": 47.5},
}

# ── Detection params ───────────────────────────────────────────────────────────
YOLO_MODEL    = "yolov8n.pt"
CONF_PRIMARY  = 0.3
CONF_FALLBACK = 0.1
YOLO_IMGSZ    = 640

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS_PERSON = ["Escenario_03", "Escenario_06", "Escenario_11", "Escenario_17"]
CAPTURE         = "Captura_01"

# ── Thresholds ────────────────────────────────────────────────────────────────
FLOOR_DEPTH_MARGIN  = 0.15   # z_cam > floor_z - margin → floor point (camera frame)
WORLD_Z_PERSON_MIN  = 0.30   # person world height lower bound (metres above floor)
WORLD_Z_PERSON_MAX  = 2.20   # person world height upper bound
WORLD_Z_FLOOR       = 0.05   # tri_cloud z below this → floor
KD_RADIUS_XY        = 0.25   # XY radius for bbox-centre world-coord match
PERSON_MIN_PTS      = 50     # min person cluster points (camera PLY) to trust detection

# ── Output colours ─────────────────────────────────────────────────────────────
COL_PERSON = np.array([39,  174,  96], dtype=np.uint8)
COL_CARGO  = np.array([220,  50,  50], dtype=np.uint8)
COL_FLOOR  = np.array([100, 100, 100], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# PLY I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ply_header(f):
    """Read PLY header; return (properties, body_start, n_vertices, is_binary)."""
    props = []
    n_verts = 0
    is_binary = False
    for raw in f:
        line = raw.decode("ascii", errors="replace").strip()
        if line.startswith("format binary"):
            is_binary = True
        elif line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("property "):
            parts = line.split()
            props.append(parts[-1])
        elif line == "end_header":
            break
    return props, n_verts, is_binary


def read_ply_binary_xyzrgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read binary LE PLY with x,y,z,red,green,blue fields.
    Guards against truncated bodies (e.g., PLY_Izq in some scenarios).
    Returns (pts_xyz [N,3] float32, pts_rgb [N,3] uint8).
    """
    with open(ply_path, "rb") as f:
        props, n_verts, _ = _parse_ply_header(f)
        body_bytes = f.read()

    record = 15  # 3×float32 + 3×uint8
    max_readable = len(body_bytes) // record
    n = min(n_verts, max_readable)
    if n < n_verts:
        print(f"    [warn] {ply_path.name}: header={n_verts} but only {n} readable")

    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data = np.frombuffer(body_bytes[:n * record], dtype=dtype)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
    return xyz, rgb


def read_tri_cloud(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read ASCII tri_cloud PLY (has PCL camera block, x y z red green blue).
    Returns (pts_xyz [N,3] float32, pts_rgb [N,3] uint8 or None).
    """
    pts, rgbs = [], []
    has_rgb = False
    reading_verts = False
    with open(ply_path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s == "end_header":
                reading_verts = True
                continue
            if reading_verts:
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        pts.append((x, y, z))
                        if len(parts) >= 6:
                            has_rgb = True
                            rgbs.append((int(parts[3]), int(parts[4]), int(parts[5])))
                    except ValueError:
                        break  # hit camera element data
    pts_xyz = np.array(pts, dtype=np.float32)
    pts_rgb = np.array(rgbs, dtype=np.uint8) if has_rgb and len(rgbs) == len(pts) else None
    return pts_xyz, pts_rgb


def save_ply_colored(path: Path, pts: np.ndarray, colors: np.ndarray) -> None:
    """
    Save ASCII PLY with x,y,z,red,green,blue — no PCL camera block.
    colors: (N,3) uint8 RGB.
    """
    n = len(pts)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Camera projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_intrinsics(pts_xyz: np.ndarray,
                        img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    Estimate pinhole intrinsics (fx, cx, fy, cy) from PLY distribution.
    Assumes PLY points cover most of the image area.
    Uses p2–p98 percentiles to ignore outliers.
    """
    xz = pts_xyz[:, 0] / pts_xyz[:, 2]
    yz = pts_xyz[:, 1] / pts_xyz[:, 2]
    xz_lo, xz_hi = np.percentile(xz, [2, 98])
    yz_lo, yz_hi = np.percentile(yz, [2, 98])
    fx = (img_w - 1) / (xz_hi - xz_lo)
    cx = -xz_lo * fx
    fy = (img_h - 1) / (yz_hi - yz_lo)
    cy = -yz_lo * fy
    return float(fx), float(cx), float(fy), float(cy)


def project_to_image(pts_xyz: np.ndarray,
                     fx: float, cx: float,
                     fy: float, cy: float,
                     img_w: int, img_h: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D pts to image pixels. Returns (u, v, valid_mask).
    u, v are integer pixel coordinates; valid_mask selects points inside the image.
    """
    u = fx * pts_xyz[:, 0] / pts_xyz[:, 2] + cx
    v = fy * pts_xyz[:, 1] / pts_xyz[:, 2] + cy
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)
    valid = (u_int >= 0) & (u_int < img_w) & (v_int >= 0) & (v_int < img_h)
    return u_int, v_int, valid


def validate_intrinsics_rgb(pts_xyz: np.ndarray,
                             pts_rgb: np.ndarray,
                             image_rgb: np.ndarray,
                             fx: float, cx: float,
                             fy: float, cy: float,
                             n_sample: int = 1000,
                             tol: int = 15) -> float:
    """
    Return fraction of sampled PLY points whose projected pixel has matching
    color in the image (max-channel diff ≤ tol).
    """
    H, W = image_rgb.shape[:2]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts_xyz), min(n_sample, len(pts_xyz)), replace=False)
    u, v, valid = project_to_image(pts_xyz[idx], fx, cx, fy, cy, W, H)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return 0.0
    pu, pv = u[valid_idx], v[valid_idx]
    diff = np.max(np.abs(pts_rgb[idx[valid_idx]].astype(np.int32)
                         - image_rgb[pv, pu].astype(np.int32)), axis=1)
    return float((diff <= tol).mean())


def points_in_bbox(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                   x1: float, y1: float, x2: float, y2: float,
                   pad: int = 5) -> np.ndarray:
    """Return boolean mask of points whose (u,v) falls inside the bbox (+pad)."""
    return (valid
            & (u >= x1 - pad) & (u <= x2 + pad)
            & (v >= y1 - pad) & (v <= y2 + pad))


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detection
# ─────────────────────────────────────────────────────────────────────────────

_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def detect_persons(img_bgr: np.ndarray,
                   conf: float = CONF_PRIMARY,
                   imgsz: int = YOLO_IMGSZ
                   ) -> list[dict]:
    """
    Run YOLOv8n on img_bgr (BGR). Return list of:
        {"bbox": [x1,y1,x2,y2], "conf": float}
    for detections of class 'person' (class id = 0).
    """
    model = get_yolo()
    results = model(img_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # COCO class 0 = person
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"bbox": [x1, y1, x2, y2], "conf": float(box.conf[0])})
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# 3D person extraction (camera frame → world Z)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Tri-cloud person exclusion
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_2d(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t),  np.cos(t)]], dtype=np.float64)


def find_person_in_tricloud_cenital(
        tri_xyz: np.ndarray,
        person_cam: np.ndarray,    # camera-frame person pts, z_cam ∈ [valid range]
        cargo_cam: np.ndarray,     # camera-frame above-floor pts (for centroid alignment)
        camera_height: float,
) -> tuple[np.ndarray, str]:
    """
    Map cenital-camera person points to world frame and find matching tri_cloud pts.

    Strategy:
    1. Filter person_cam to above-floor points (world_Z > WORLD_Z_PERSON_MIN).
    2. Compute person bounding box CENTROID in camera XY.
    3. For each candidate azimuth (0/90/180/270°): align cargo centroid (cam→world),
       transform person CENTROID to world XY, count tri_cloud pts in a cylinder
       of radius KD_RADIUS_XY around that XY and world_Z ∈ [z_lo, z_hi].
    4. Use best azimuth; collect all tri_cloud pts in the person's world bbox.
    5. Fallback: compact-cluster heuristic if transform fails.

    Returns (mask, method_str).
    """
    n = len(tri_xyz)
    mask = np.zeros(n, dtype=bool)
    method = "none"

    if len(person_cam) < PERSON_MIN_PTS:
        return mask, "too_few_cam_pts"

    # Step 1 – filter to above-floor person points in camera frame
    # world_Z = camera_height - z_cam; keep only z_cam < (cam_height - WORLD_Z_PERSON_MIN)
    z_floor_cam = camera_height          # z_cam at floor level
    z_thresh_cam = z_floor_cam - WORLD_Z_PERSON_MIN
    above = person_cam[:, 2] < z_thresh_cam
    person_valid = person_cam[above]
    if len(person_valid) < PERSON_MIN_PTS:
        person_valid = person_cam  # fall through with all points

    wz = camera_height - person_valid[:, 2]
    wz = np.clip(wz, 0.0, WORLD_Z_PERSON_MAX)
    z_lo = max(float(np.percentile(wz, 10)), WORLD_Z_PERSON_MIN)
    z_hi = min(float(np.percentile(wz, 90)), WORLD_Z_PERSON_MAX)
    if z_hi <= z_lo:
        z_hi = z_lo + 0.5

    # Person centroid in camera XY
    p_centroid_cam = person_valid[:, :2].mean(axis=0)  # (2,)

    # Cargo centroid in camera XY (coarse reference)
    if len(cargo_cam) > 0:
        c_centroid_cam = cargo_cam[:, :2].mean(axis=0)
    else:
        c_centroid_cam = p_centroid_cam

    # Tri-cloud non-floor centroid in world XY
    tri_nf = tri_xyz[tri_xyz[:, 2] > WORLD_Z_FLOOR]
    if len(tri_nf) == 0:
        tri_nf = tri_xyz
    c_centroid_world = tri_nf[:, :2].mean(axis=0)

    # Step 2 – azimuth search: try 0°/90°/180°/270°
    tree_xy = KDTree(tri_xyz[:, :2])
    best_az, best_score, best_offset = 0.0, -1.0, np.zeros(2)

    for az in [0, 90, 180, 270]:
        R = _rotation_2d(az)
        # Cargo centroid alignment: offset makes R*c_cam → c_world
        offset = c_centroid_world - R @ c_centroid_cam
        # Transform person centroid
        p_world_xy = R @ p_centroid_cam + offset
        idxs = tree_xy.query_ball_point(p_world_xy, r=KD_RADIUS_XY * 2)
        score = sum(1 for i in idxs
                    if z_lo <= tri_xyz[i, 2] <= z_hi)
        if score > best_score:
            best_score, best_az, best_offset = score, float(az), offset

    if best_score >= 5:
        R = _rotation_2d(best_az)
        p_world_xy = R @ p_centroid_cam + best_offset

        # Collect all tri_cloud points in person's world cylinder
        for i in tree_xy.query_ball_point(p_world_xy, r=KD_RADIUS_XY):
            if z_lo <= tri_xyz[i, 2] <= z_hi:
                mask[i] = True

        # Also expand slightly: include full person height range
        z_lo2, z_hi2 = max(0.0, z_lo - 0.1), min(WORLD_Z_PERSON_MAX, z_hi + 0.1)
        for i in tree_xy.query_ball_point(p_world_xy, r=KD_RADIUS_XY * 1.3):
            if z_lo2 <= tri_xyz[i, 2] <= z_hi2:
                mask[i] = True

        method = f"azimuth_{best_az:.0f}deg"
        if mask.sum() >= 50:
            return mask, method

    # Step 3 – fallback: compact-cluster heuristic on tall points
    tall = (tri_xyz[:, 2] >= z_lo) & (tri_xyz[:, 2] <= z_hi)
    if tall.sum() >= 20:
        tall_pts = tri_xyz[tall]
        kd2 = KDTree(tall_pts[:, :2])   # XY footprint
        visited = np.zeros(len(tall_pts), dtype=bool)
        clusters: list[list[int]] = []
        for start in range(len(tall_pts)):
            if visited[start]:
                continue
            nbrs = kd2.query_ball_point(tall_pts[start, :2], r=0.3)
            if len(nbrs) < 5:
                continue
            stack, cluster = list(nbrs), set(nbrs)
            while stack:
                cur = stack.pop()
                visited[cur] = True
                for nb in kd2.query_ball_point(tall_pts[cur, :2], r=0.3):
                    if nb not in cluster:
                        cluster.add(nb)
                        stack.append(nb)
            clusters.append(list(cluster))

        # Pick smallest-footprint cluster within person size limits
        best_cl, best_fp = None, 1e9
        for cl in clusters:
            if len(cl) < 20:
                continue
            p = tall_pts[cl]
            fp = ((p[:, 0].max() - p[:, 0].min())
                  * (p[:, 1].max() - p[:, 1].min()))
            if fp < best_fp and fp < 1.0:
                best_fp, best_cl = fp, cl
        if best_cl is not None:
            tall_idx = np.where(tall)[0]
            mask[tall_idx[best_cl]] = True
            method = "heuristic_cluster"

    return mask, method


def find_person_heuristic_only(tri_xyz: np.ndarray,
                                world_z_lo: float = 0.8,
                                world_z_hi: float = 2.0,
                                ) -> tuple[np.ndarray, str]:
    """
    Pure height+compactness heuristic on tri_cloud.
    Used when only lateral cameras detected the person (no reliable world XY).
    """
    n = len(tri_xyz)
    mask = np.zeros(n, dtype=bool)
    tall = (tri_xyz[:, 2] >= world_z_lo) & (tri_xyz[:, 2] <= world_z_hi)
    if tall.sum() < 20:
        return mask, "heuristic_no_tall_pts"
    tall_pts = tri_xyz[tall]
    kd = KDTree(tall_pts[:, :2])
    visited = np.zeros(len(tall_pts), dtype=bool)
    clusters: list[list[int]] = []
    for start in range(len(tall_pts)):
        if visited[start]:
            continue
        nbrs = kd.query_ball_point(tall_pts[start, :2], r=0.3)
        if len(nbrs) < 5:
            continue
        stack, cluster = list(nbrs), set(nbrs)
        while stack:
            cur = stack.pop()
            visited[cur] = True
            for nb in kd.query_ball_point(tall_pts[cur, :2], r=0.3):
                if nb not in cluster:
                    cluster.add(nb)
                    stack.append(nb)
        clusters.append(list(cluster))
    best_cl, best_fp = None, 1e9
    for cl in clusters:
        if len(cl) < 30:
            continue
        p = tall_pts[cl]
        fp = (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())
        if fp < best_fp:
            best_fp, best_cl = fp, cl
    # Accept if footprint < 1.0 m² (person alone ≈ 0.1 m², with nearby cargo ≤ 1.0 m²)
    if best_cl is not None and best_fp < 1.0:
        tall_idx = np.where(tall)[0]
        mask[tall_idx[best_cl]] = True
    return mask, "lateral_heuristic"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_scenario(esc_name: str, capture: str) -> dict:
    """
    Full pipeline for one scenario/capture.
    Returns a result dict with all metrics.
    """
    cap_dir   = BBB_BASE / esc_name / capture
    tri_path  = TIME_PROCESS / esc_name / f"{capture}_tri_cloud.ply"

    result = {
        "scenario":   esc_name,
        "capture":    capture,
        "detections": {},
        "validation": {},
        "n_person_pts_cam": 0,
        "n_person_pts_tri": 0,
        "n_total_tri":      0,
        "mapping_method":   "none",
        "error":            None,
    }

    if not cap_dir.exists():
        result["error"] = f"capture dir missing: {cap_dir}"
        return result
    if not tri_path.exists():
        result["error"] = f"tri_cloud missing: {tri_path}"
        return result

    # ── Step 1: YOLO detection on all 3 cameras ──────────────────────────────
    print(f"\n  {esc_name}/{capture} — YOLO detection")
    all_detections: dict[str, list] = {}
    for cam_name in CAMERA_INFO:
        png_path = cap_dir / f"PNG_{cam_name}.png"
        if not png_path.exists():
            print(f"    {cam_name}: PNG missing, skip")
            continue
        img_bgr = cv2.imread(str(png_path))
        if img_bgr is None:
            print(f"    {cam_name}: failed to read PNG")
            continue

        dets = detect_persons(img_bgr, conf=CONF_PRIMARY)
        if not dets:
            # retry with lower conf
            dets = detect_persons(img_bgr, conf=CONF_FALLBACK)
            thresh_used = CONF_FALLBACK
        else:
            thresh_used = CONF_PRIMARY

        all_detections[cam_name] = dets
        result["detections"][cam_name] = [
            {"conf": d["conf"], "bbox": d["bbox"]} for d in dets
        ]
        if dets:
            for d in dets:
                bb = d["bbox"]
                print(f"    {cam_name}: PERSON conf={d['conf']:.2f} "
                      f"bbox=[{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]"
                      f" (thresh={thresh_used})")
        else:
            print(f"    {cam_name}: no person detected")

    # Count total persons found (any camera)
    total_person_dets = sum(len(v) for v in all_detections.values())
    if total_person_dets == 0:
        print(f"    → No person detected in any camera. Skipping 3D mapping.")
        result["n_total_tri"] = len(read_tri_cloud(tri_path)[0])
        return result

    # ── Step 2: Load tri_cloud ────────────────────────────────────────────────
    tri_xyz, tri_rgb = read_tri_cloud(tri_path)
    result["n_total_tri"] = len(tri_xyz)
    print(f"  tri_cloud: {len(tri_xyz)} pts, "
          f"z=[{tri_xyz[:,2].min():.2f}, {tri_xyz[:,2].max():.2f}]")

    # ── Step 3: Camera PLY → person 3D points ────────────────────────────────
    # Cenital (pitch=90°) provides reliable world_Z = cam_height - z_cam.
    # Lateral cameras (Izq/Der) require full extrinsics for world XY, so we
    # only use them for heuristic fallback.
    person_mask = np.zeros(len(tri_xyz), dtype=bool)
    best_cam = None

    # ── Try CENITAL first (most reliable 3D transform) ────────────────────────
    cenital_dets = all_detections.get("Cenital", [])
    if cenital_dets:
        cam_name = "Cenital"
        ply_path = cap_dir / f"PLY_{cam_name}.ply"
        png_path = cap_dir / f"PNG_{cam_name}.png"
        if ply_path.exists() and png_path.exists():
            print(f"\n  Camera PLY mapping: {cam_name}")
            ply_xyz, ply_rgb = read_ply_binary_xyzrgb(ply_path)
            img_bgr = cv2.imread(str(png_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]

            fx, cx, fy, cy = estimate_intrinsics(ply_xyz, W, H)
            print(f"    Intrinsics: fx={fx:.1f} cx={cx:.1f} fy={fy:.1f} cy={cy:.1f}")

            match_rate = validate_intrinsics_rgb(ply_xyz, ply_rgb, img_rgb, fx, cx, fy, cy)
            result["validation"][cam_name] = {"match_rate_tol15": round(match_rate, 3)}
            print(f"    RGB match rate (tol=15): {match_rate:.1%}")

            cam_height = CAMERA_INFO[cam_name]["height_m"]
            u, v, valid = project_to_image(ply_xyz, fx, cx, fy, cy, W, H)
            floor_depth = np.median(ply_xyz[:, 2])

            person_pts_cam = []
            for det in cenital_dets:
                x1, y1, x2, y2 = det["bbox"]
                in_bbox = points_in_bbox(u, v, valid, x1, y1, x2, y2, pad=5)
                person_pts_cam.append(ply_xyz[in_bbox])

            if person_pts_cam:
                person_cam = np.concatenate(person_pts_cam, axis=0)
                print(f"    Person points (camera frame): {len(person_cam)}")
                wz = cam_height - person_cam[:, 2]
                print(f"    Person world_Z (cam_h - z_cam): [{wz.min():.2f}, {wz.max():.2f}] m")

                above_floor = ply_xyz[:, 2] < floor_depth - FLOOR_DEPTH_MARGIN
                cargo_cam = ply_xyz[above_floor]

                mask, method = find_person_in_tricloud_cenital(
                    tri_xyz, person_cam, cargo_cam, cam_height)
                n_found = int(mask.sum())
                print(f"    Tri-cloud person points found: {n_found} (method={method})")

                if n_found >= 50:
                    person_mask |= mask
                    best_cam = cam_name
                    result["n_person_pts_cam"] = len(person_cam)
                    result["mapping_method"] = method

    # ── Lateral-camera fallback: heuristic only (Izq/Der world XY unknown) ────
    lateral_dets = (all_detections.get("Izq", []) or all_detections.get("Der", []))
    if not person_mask.any() and lateral_dets:
        print(f"\n  Lateral cameras only — using height+compactness heuristic on tri_cloud")
        # Validate projection for Izq (just for reporting)
        for cam_name in ["Izq", "Der"]:
            dets = all_detections.get(cam_name, [])
            if not dets:
                continue
            ply_path = cap_dir / f"PLY_{cam_name}.ply"
            png_path = cap_dir / f"PNG_{cam_name}.png"
            if not ply_path.exists() or not png_path.exists():
                continue
            print(f"\n  Camera PLY validation: {cam_name}")
            ply_xyz, ply_rgb = read_ply_binary_xyzrgb(ply_path)
            img_bgr = cv2.imread(str(png_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]
            fx, cx, fy, cy = estimate_intrinsics(ply_xyz, W, H)
            match_rate = validate_intrinsics_rgb(ply_xyz, ply_rgb, img_rgb, fx, cx, fy, cy)
            result["validation"][cam_name] = {"match_rate_tol15": round(match_rate, 3)}
            print(f"    Intrinsics: fx={fx:.1f} cx={cx:.1f} fy={fy:.1f} cy={cy:.1f}")
            print(f"    RGB match rate (tol=15): {match_rate:.1%}")

            # Extract person pts for reporting (not used for world XY)
            u, v, valid = project_to_image(ply_xyz, fx, cx, fy, cy, W, H)
            pts_cam = []
            for det in dets:
                x1, y1, x2, y2 = det["bbox"]
                in_bbox = points_in_bbox(u, v, valid, x1, y1, x2, y2, pad=5)
                pts_cam.append(ply_xyz[in_bbox])
            if pts_cam:
                person_cam_lat = np.concatenate(pts_cam, axis=0)
                print(f"    Person points (camera frame): {len(person_cam_lat)}")
                result["n_person_pts_cam"] = len(person_cam_lat)
            break  # one lateral camera is enough for reporting

        mask, method = find_person_heuristic_only(
            tri_xyz, world_z_lo=0.8, world_z_hi=2.0)
        print(f"    Tri-cloud person points found: {mask.sum()} (method={method})")
        if mask.any():
            person_mask |= mask
            result["mapping_method"] = method

    result["n_person_pts_tri"] = int(person_mask.sum())
    print(f"\n  → Tri-cloud person points: {result['n_person_pts_tri']} / {result['n_total_tri']}")

    # ── Step 4: Export colored PLY ────────────────────────────────────────────
    esc_num = esc_name.split("_")[-1]
    out_path = OUT_DIR / f"Esc{esc_num}_Cap01_person_mask.ply"

    # Build per-point color array
    colors = np.tile(COL_CARGO, (len(tri_xyz), 1))
    # Floor points
    floor_mask = tri_xyz[:, 2] < WORLD_Z_FLOOR
    colors[floor_mask] = COL_FLOOR
    # Person points
    colors[person_mask] = COL_PERSON

    save_ply_colored(out_path, tri_xyz, colors)
    print(f"  → Saved: {out_path}")

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("test_person_2d.py — 2D detection + 3D person exclusion")
    print("=" * 70)

    # Preload YOLO model
    print("\nLoading YOLOv8n model...")
    get_yolo()
    print("Model loaded.")

    all_results = []
    for esc in SCENARIOS_PERSON:
        try:
            res = process_scenario(esc, CAPTURE)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {esc}: {e}")
            traceback.print_exc()
            res = {"scenario": esc, "error": str(e),
                   "n_person_pts_tri": 0, "n_total_tri": 0}
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<15} {'Person dets':>12} {'Cam pts':>10} "
          f"{'Tri pts':>10} {'Total tri':>10} {'Method':<22}")
    print("-" * 70)
    for r in all_results:
        if r.get("error"):
            print(f"{r['scenario']:<15}  ERROR: {r['error']}")
            continue
        det_str = ", ".join(
            f"{cam}:{len(dets)}" for cam, dets in r["detections"].items() if dets
        ) or "none"
        print(
            f"{r['scenario']:<15} {det_str:>12} "
            f"{r.get('n_person_pts_cam',0):>10} "
            f"{r['n_person_pts_tri']:>10} "
            f"{r['n_total_tri']:>10} "
            f"{r.get('mapping_method','none'):<22}"
        )

    # RGB validation summary
    print("\nRGB validation (match rate tol=15):")
    for r in all_results:
        if r.get("validation"):
            v_str = ", ".join(
                f"{cam}: {v['match_rate_tol15']:.0%}"
                for cam, v in r["validation"].items()
            )
            print(f"  {r['scenario']}: {v_str}")

    print(f"\nOutput PLYs in: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
