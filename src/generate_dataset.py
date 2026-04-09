"""
generate_dataset.py  —  Synthetic labeled point cloud dataset
Simulates FUSION3D stereo-camera captures of cargo pallets.

Coordinate system: X right, Y up (height), Z toward cameras   [metres]
Calibrated to real FUSION3D noise: σ flat≈6.5mm, overall≈15mm → using 10mm

Label map
  0  floor
  1  cargo (boxes)
  2  vehicle (forklift / pallet jack)
  3  person
  4  pallet base

Output per scene
  <output_dir>/<id>.ply    — binary PLY: x y z label (no PCL camera block)
  <output_dir>/metadata.json — scene parameters for every sample
"""

import os
import json
from pathlib import Path

import numpy as np
import open3d as o3d

if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
    os.environ["XDG_SESSION_TYPE"] = "x11"

# ── Configuration ──────────────────────────────────────────────────────────────
CFG = {
    "n_samples":   100,
    "seed":        42,
    "output_dir":  "../output/dataset",

    # ── Sensor noise (FUSION3D: σ flat=6.5mm, overall≈30mm measured) ──
    "noise_std":        0.025,   # m  Gaussian noise per point (calibrated to real roughness σ≈30mm)
    "dropout_ratio":    0.15,    # fraction of points removed
    "outlier_ratio":    0.03,    # fraction turned into local outliers
    "voxel_size":       0.010,   # m  voxel grid; effective NN spacing driven by noise+dropout
    "local_outlier_std": 0.055,  # m  spread of local outlier clusters

    # ── Camera positions (matching BBB system: cenital + der + izq) ──
    # Each camera sits above the scene looking toward origin.
    # pos: (X, Y, Z) in metres.  fov_deg: half-cone field of view.
    "cameras": [
        {"name": "cenital", "pos": [ 0.0, 1.9,  0.4], "fov_deg": 75.0},
        {"name": "der",     "pos": [ 2.0, 1.3,  1.8], "fov_deg": 65.0},
        {"name": "izq",     "pos": [-2.0, 1.3,  1.8], "fov_deg": 65.0},
    ],

    # ── Scene composition ──
    "p_two_boxes":  0.00,   # disabled — one box per scene
    "p_person":     0.00,   # disabled for now
    "p_forklift":   0.00,   # disabled; always use primitive traspaleta
    "p_pallet":     0.70,   # EUR pallet under cargo

    # Cargo box size range (m)
    "box_min":  0.25,
    "box_max":  1.40,

    # Floor extent (m) — half-size of the sampled floor patch
    # Real FUSION3D Z span ≈ 3m, X span ≈ 5.5m → floor_extent matches
    "floor_extent": 2.5,

    # Initial points sampled from each mesh before degradation
    # (voxel grid will reduce this to a physically realistic density)
    # Real FUSION3D: ~275k pts/scene (incl. walls+ceiling).
    # Targeting ~200-250k for floor+objects-only synthetic.
    "pts_floor":    800_000,
    "pts_pallet":    30_000,
    "pts_box":       80_000,
    "pts_forklift": 100_000,
    "pts_person":    30_000,
}

LABEL = {"floor": 0, "cargo": 1, "vehicle": 2, "person": 3, "pallet": 4}

# EUR pallet dimensions (m)
EUR_W, EUR_H, EUR_D = 1.20, 0.144, 0.80


# ── Geometry helpers ───────────────────────────────────────────────────────────

def make_pallet_mesh() -> o3d.geometry.TriangleMesh:
    """Standard EUR pallet: 1200×144×800 mm, centred on XZ at Y=0."""
    m = o3d.geometry.TriangleMesh.create_box(EUR_W, EUR_H, EUR_D)
    m.translate([-EUR_W / 2, 0.0, -EUR_D / 2])
    return m


def make_box_mesh(w: float, h: float, d: float) -> o3d.geometry.TriangleMesh:
    """Axis-aligned box centred in X and Z, bottom face at Y=0."""
    m = o3d.geometry.TriangleMesh.create_box(w, h, d)
    m.translate([-w / 2, 0.0, -d / 2])
    return m


def make_person_mesh() -> o3d.geometry.TriangleMesh:
    """Rough person: cylinder body + sphere head, standing upright at Y=0.
    Open3D create_cylinder is Z-aligned by default → rotate 90° around X to make it Y-aligned.
    """
    body = o3d.geometry.TriangleMesh.create_cylinder(radius=0.18, height=0.95, resolution=16)
    # Rotate 90° around X: Z-axis becomes Y-axis → cylinder stands upright
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1,  0]], dtype=np.float64)
    body.rotate(R, center=(0.0, 0.0, 0.0))
    # Now cylinder spans Y=-0.475..+0.475 → translate up so bottom sits on Y=0
    body.translate([0.0, 0.475, 0.0])
    head = o3d.geometry.TriangleMesh.create_sphere(radius=0.14, resolution=8)
    head.translate([0.0, 1.02, 0.0])
    return body + head


def make_pallet_jack_mesh() -> o3d.geometry.TriangleMesh:
    """
    Simplified traspaleta (pallet jack).
    Origin = body front face, floor level (Y=0, Z=0).
    Forks extend in +Z (toward cargo/pallet).
    Body extends in -Z (operator side).

    Local extents:
      Body:   X:-0.35..+0.35, Y:0..0.90, Z:-0.40..0
      Fork L: X:-0.30..-0.15, Y:0..0.08, Z:0..+1.15
      Fork R: X:+0.15..+0.30, Y:0..0.08, Z:0..+1.15
    """
    body = o3d.geometry.TriangleMesh.create_box(0.70, 0.90, 0.40)
    body.translate([-0.35, 0.0, -0.40])
    fork_l = o3d.geometry.TriangleMesh.create_box(0.15, 0.08, 1.15)
    fork_l.translate([-0.30, 0.0, 0.0])
    fork_r = o3d.geometry.TriangleMesh.create_box(0.15, 0.08, 1.15)
    fork_r.translate([ 0.15, 0.0, 0.0])
    return body + fork_l + fork_r


def load_forklift(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load forklift STL (in mm) and convert to metres.
    STL Z=0 is the cab rear, Z=1.962 are the forks.
    We translate so forks sit just behind the pallet back edge (world Z≈-0.85m)
    and the cab is further back (world Z≈-2.8m).  No overlap with cargo.
    """
    fk = o3d.io.read_triangle_mesh(stl_path)
    fk.scale(0.001, center=(0.0, 0.0, 0.0))    # mm → m
    fk.compute_vertex_normals()
    verts = np.asarray(fk.vertices)
    y_min = verts[:, 1].min()
    # Sit on floor (Y) and push back so forks (Z=1.962) end up at world Z≈-0.85
    # → Z_translate = -0.85 - 1.962 = -2.812 ≈ -2.8
    fk.translate([0.0, -y_min, -2.8])
    return fk


def sample_labeled(
    mesh: o3d.geometry.TriangleMesh,
    label: int,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample mesh surface → (pts [N,3], labels [N,])."""
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(pcd.points, dtype=np.float32)
    lbs = np.full(len(pts), label, dtype=np.uint8)
    return pts, lbs


def sample_floor(cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random uniform points on Y=0 plane within floor_extent."""
    ext = cfg["floor_extent"]
    n = cfg["pts_floor"]
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-ext, ext, n).astype(np.float32)
    pts[:, 2] = rng.uniform(-ext, ext, n).astype(np.float32)
    lbs = np.zeros(n, dtype=np.uint8)
    return pts, lbs


# ── Camera visibility filter ───────────────────────────────────────────────────

def camera_arc_filter(
    pts: np.ndarray,
    labels: np.ndarray,
    cameras: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only points inside the FOV cone of at least one camera.
    Each camera looks toward the origin from its 'pos'.
    Simple angle-based filter — no raycasting occlusion (TODO v2).
    """
    visible = np.zeros(len(pts), dtype=bool)
    for cam in cameras:
        cam_pos = np.array(cam["pos"], dtype=np.float64)
        fov_half = np.deg2rad(cam["fov_deg"] / 2.0)
        view_dir = -cam_pos / np.linalg.norm(cam_pos)   # looks at origin

        to_pts = pts.astype(np.float64) - cam_pos       # (N, 3)
        norms = np.linalg.norm(to_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        cos_a = (to_pts / norms) @ view_dir              # (N,)
        visible |= cos_a > np.cos(fov_half)

    return pts[visible], labels[visible]


# ── Distance-dependent density falloff ────────────────────────────────────────

def apply_distance_density(
    pts: np.ndarray,
    labels: np.ndarray,
    cameras: list[dict],
    rng: np.random.Generator,
    ref_dist: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate stereo-camera density falloff: density ∝ 1/d².
    At ref_dist (1.5m): no extra dropout.
    At 3m: ~75% of remaining points dropped.
    Matches the real data pattern where distant floor/walls are sparse.
    """
    min_dist = np.full(len(pts), np.inf)
    for cam in cameras:
        cam_pos = np.array(cam["pos"], dtype=np.float64)
        d = np.linalg.norm(pts.astype(np.float64) - cam_pos, axis=1)
        min_dist = np.minimum(min_dist, d)

    # p_keep = (ref_dist / d)²  clamped to [0.15, 1.0]
    p_keep = np.clip((ref_dist / np.maximum(min_dist, ref_dist)) ** 2, 0.15, 1.0)
    keep = rng.random(len(pts)) < p_keep
    return pts[keep], labels[keep]


# ── Sensor degradation (label-aware) ──────────────────────────────────────────

def degrade_labeled(
    pts: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate real-sensor imperfections while keeping label correspondence:
      1. Voxel downsampling   — limits spatial resolution
      2. Gaussian noise       — calibrated to FUSION3D σ≈10mm
      3. Point dropout        — simulates reflectance / occlusion losses
      4. Local outliers       — stray reflections, multi-path artefacts
    """
    if len(pts) == 0:
        return pts, labels

    # 1. Voxel downsampling (numpy, label-safe)
    voxel_idx = np.floor(pts / cfg["voxel_size"]).astype(np.int64)
    _, unique = np.unique(voxel_idx, axis=0, return_index=True)
    pts, labels = pts[unique], labels[unique]

    # 2. Gaussian noise
    pts = pts + rng.normal(0.0, cfg["noise_std"], pts.shape).astype(np.float32)

    # 3. Dropout
    keep = rng.random(len(pts)) > cfg["dropout_ratio"]
    pts, labels = pts[keep], labels[keep]
    if len(pts) == 0:
        return pts, labels

    # 4. Local outliers (label = 255 → "unlabeled / artefact")
    n_out = max(1, int(len(pts) * cfg["outlier_ratio"]))
    anchors = pts[rng.integers(0, len(pts), n_out)]
    offsets = rng.normal(0.0, cfg["local_outlier_std"], (n_out, 3)).astype(np.float32)
    out_pts = anchors + offsets
    out_lbs = np.full(n_out, 255, dtype=np.uint8)

    pts    = np.vstack([pts, out_pts])
    labels = np.concatenate([labels, out_lbs])
    return pts, labels


# ── Label → RGB colour map ────────────────────────────────────────────────────
#  Colours are baked into the PLY so CloudCompare shows them immediately.
#  The numeric `label` field is also kept for programmatic use.

LABEL_RGB: dict[int, tuple[int, int, int]] = {
    0:   (160, 160, 160),   # floor   — gris medio (visible en CC, distinto de carga)
    1:   (230, 126,  34),   # cargo   — naranja
    2:   ( 41, 128, 185),   # vehicle — azul
    3:   ( 39, 174,  96),   # person  — verde
    4:   (243, 156,  18),   # pallet  — amarillo
    255: (180,  60, 180),   # outlier — magenta
}
_DEFAULT_RGB = (255, 0, 255)   # magenta for unknown labels


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Map label array → uint8 RGB array (N, 3)."""
    rgb = np.full((len(labels), 3), _DEFAULT_RGB, dtype=np.uint8)
    for lv, color in LABEL_RGB.items():
        rgb[labels == lv] = color
    return rgb


# ── PLY export (binary, no PCL camera block) ───────────────────────────────────

def save_ply(path: Path, pts: np.ndarray, labels: np.ndarray) -> None:
    """
    Write binary-little-endian PLY with: x y z red green blue label.
    - RGB encodes the label → CloudCompare shows colours immediately on open.
    - `label` scalar field is kept for programmatic use.
    Hand-written header → no PCL camera block.
    """
    n = len(pts)
    rgb = labels_to_rgb(labels)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar label\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "u1"),
    ])
    data = np.empty(n, dtype=dtype)
    data["x"] = pts[:, 0].astype(np.float32)
    data["y"] = pts[:, 1].astype(np.float32)
    data["z"] = pts[:, 2].astype(np.float32)
    data["red"]   = rgb[:, 0]
    data["green"] = rgb[:, 1]
    data["blue"]  = rgb[:, 2]
    data["label"] = labels.astype(np.uint8)

    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


# ── Scene generation ───────────────────────────────────────────────────────────

def generate_scene(
    scene_id: int,
    rng: np.random.Generator,
    cfg: dict,
    forklift_mesh: o3d.geometry.TriangleMesh | None,
) -> dict:
    """
    Build one synthetic scene and return its metadata dict.
    The PLY file is written to cfg['output_dir']/<scene_id:05d>.ply

    Scene composition: floor + (optional pallet) + cargo box(es) + pallet jack.
    The pallet jack is always present, its body front face touching the cargo
    back face, rotated by a random angle ±15° around Y.
    Floor is kept outside the camera FOV filter so the ground plane is always
    fully covered (no clipping at the edges).
    """
    meta: dict = {"id": scene_id, "objects": []}

    # ── Floor (disabled for now — add back when needed) ──
    floor_pts = np.empty((0, 3), dtype=np.float32)
    floor_lbs = np.empty(0, dtype=np.uint8)
    # meta["objects"].append("floor")

    # ── Object point lists (will be FOV-filtered) ──
    obj_pts: list[np.ndarray] = []
    obj_lbs: list[np.ndarray] = []

    # ── Pallet ──
    has_pallet = rng.random() < cfg["p_pallet"]
    pallet_top_y = 0.0
    if has_pallet:
        pm = make_pallet_mesh()
        pp, pl = sample_labeled(pm, LABEL["pallet"], cfg["pts_pallet"])
        obj_pts.append(pp); obj_lbs.append(pl)
        pallet_top_y = EUR_H
        meta["objects"].append("pallet")

    # ── Cargo box 1 (always present) ──
    w1, h1, d1 = rng.uniform(cfg["box_min"], cfg["box_max"], 3).astype(float)
    # Cargo rotates in 90° steps around Y — pallet jack and floor stay axis-aligned
    rot90 = int(rng.choice([0, 1, 2, 3]))   # 0°, 90°, 180°, 270°
    if rot90 % 2 == 1:                       # 90° or 270° → swap W and D
        w1, d1 = d1, w1
    # Cargo centred on pallet/origin — no random offset
    ox, oz = 0.0, 0.0
    bm1 = make_box_mesh(w1, h1, d1)
    bm1.translate([ox, pallet_top_y, oz])
    bp1, bl1 = sample_labeled(bm1, LABEL["cargo"], cfg["pts_box"])
    obj_pts.append(bp1); obj_lbs.append(bl1)
    meta["objects"].append({"box1": {"w": round(w1,3), "h": round(h1,3), "d": round(d1,3), "rot90": rot90}})

    # ── Cargo box 2 (optional) ──
    if rng.random() < cfg["p_two_boxes"]:
        w2, h2, d2 = rng.uniform(cfg["box_min"], min(w1, cfg["box_max"]), 3).astype(float)
        placement = rng.choice(["stacked", "adjacent"])
        if placement == "stacked":
            bm2 = make_box_mesh(w2, h2, d2)
            bm2.translate([ox + rng.uniform(-0.05, 0.05),
                           pallet_top_y + h1,
                           oz + rng.uniform(-0.05, 0.05)])
        else:  # adjacent on pallet / floor
            side = rng.choice([-1, 1])
            bm2 = make_box_mesh(w2, h2, d2)
            bm2.translate([ox + side * (w1 / 2 + w2 / 2 + 0.02), pallet_top_y, oz])
        bp2, bl2 = sample_labeled(bm2, LABEL["cargo"], cfg["pts_box"])
        obj_pts.append(bp2); obj_lbs.append(bl2)
        meta["objects"].append({"box2": {"w": round(w2,3), "h": round(h2,3),
                                          "d": round(d2,3), "placement": placement}})

    # ── Pallet jack (always present) ──
    # Origin of make_pallet_jack_mesh is at body-front / fork-root (Z=0, Y=0).
    # Forks extend in +Z (toward cargo), body extends in -Z (operator side).
    # Strategy:
    #   1. Rotate jack ±15° around Y at its local origin (body front stays at Z=0).
    #   2. Translate so body front (Z=0) aligns with cargo back face.
    #      cargo back Z = oz - d1/2.
    #   3. Small random X offset so jack is not always perfectly centred on cargo.
    jack_angle = 0.0   # jack parallel to floor/axes; cargo rotates instead
    jack_x = float(rng.uniform(-0.10, 0.10))   # ligero offset X del jack
    cargo_back_z = oz - d1 / 2

    tj = make_pallet_jack_mesh()
    cos_a, sin_a = np.cos(jack_angle), np.sin(jack_angle)
    R_y = np.array([[cos_a, 0.0, sin_a],
                    [0.0,   1.0, 0.0  ],
                    [-sin_a,0.0, cos_a]], dtype=np.float64)
    tj.rotate(R_y, center=(0.0, 0.0, 0.0))   # rotate around body front (Z=0)
    # After rotation the body-front corners (at ±0.35 in X) protrude into +Z by
    # up to 0.35·|sin(angle)|.  Pull the jack back by that amount so no part of
    # the body penetrates the cargo box.
    z_clearance = 0.35 * abs(sin_a)
    tj.translate([jack_x, 0.0, cargo_back_z - z_clearance])
    vp, vl = sample_labeled(tj, LABEL["vehicle"], cfg["pts_forklift"] // 3)
    obj_pts.append(vp); obj_lbs.append(vl)
    meta["objects"].append({"pallet_jack": {"angle_deg": round(np.degrees(jack_angle), 1)}})

    # ── Apply camera FOV filter to objects only ──
    obj_all = np.vstack(obj_pts)
    lbs_all_obj = np.concatenate(obj_lbs)
    obj_all, lbs_all_obj = camera_arc_filter(obj_all, lbs_all_obj, cfg["cameras"])

    # ── Merge floor (unfiltered) + objects (filtered) ──
    pts_all = np.vstack([floor_pts, obj_all])
    lbs_all = np.concatenate([floor_lbs, lbs_all_obj])

    # ── Distance-dependent density falloff ──
    pts_all, lbs_all = apply_distance_density(pts_all, lbs_all, cfg["cameras"], rng)

    # ── Apply sensor degradation ──
    pts_all, lbs_all = degrade_labeled(pts_all, lbs_all, cfg, rng)

    meta["n_points"] = int(len(pts_all))
    meta["label_counts"] = {
        str(k): int(np.sum(lbs_all == k))
        for k in sorted(np.unique(lbs_all))
    }

    # ── Export PLY ──
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{scene_id:05d}.ply"
    save_ply(ply_path, pts_all, lbs_all)

    return meta


# ── CLI argument parsing ───────────────────────────────────────────────────────

def parse_args(cfg: dict) -> dict:
    """
    Override CFG values from command-line arguments.
    Usage examples:
        python3 generate_dataset.py --n 200 --out ../output/my_run
        python3 generate_dataset.py --seed 123 --noise 0.015 --dropout 0.3
        python3 generate_dataset.py --p-forklift 0 --p-person 0.5
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Synthetic labeled point cloud dataset generator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Dataset
    p.add_argument("--n",          type=int,   default=cfg["n_samples"],   help="Number of scenes  (default: %(default)s)")
    p.add_argument("--seed",       type=int,   default=cfg["seed"],        help="Random seed       (default: %(default)s)")
    p.add_argument("--out",        type=str,   default=cfg["output_dir"],  help="Output directory  (default: %(default)s)")
    # Sensor noise
    p.add_argument("--noise",      type=float, default=cfg["noise_std"],         help="Gaussian noise σ in metres  (default: %(default)s)")
    p.add_argument("--dropout",    type=float, default=cfg["dropout_ratio"],     help="Point dropout ratio 0-1     (default: %(default)s)")
    p.add_argument("--voxel",      type=float, default=cfg["voxel_size"],        help="Voxel grid size in metres   (default: %(default)s)")
    p.add_argument("--outliers",   type=float, default=cfg["outlier_ratio"],     help="Outlier ratio 0-1           (default: %(default)s)")
    # Scene composition probabilities
    p.add_argument("--p-two-boxes",  type=float, default=cfg["p_two_boxes"],  metavar="P", help="Prob second cargo box  (default: %(default)s)")
    p.add_argument("--p-person",     type=float, default=cfg["p_person"],     metavar="P", help="Prob person in scene   (default: %(default)s)")
    p.add_argument("--p-forklift",   type=float, default=cfg["p_forklift"],   metavar="P", help="Prob forklift in scene (default: %(default)s)")
    p.add_argument("--p-pallet",     type=float, default=cfg["p_pallet"],     metavar="P", help="Prob EUR pallet base   (default: %(default)s)")
    # Box dimensions
    p.add_argument("--box-min",    type=float, default=cfg["box_min"],  metavar="M", help="Min box side in metres (default: %(default)s)")
    p.add_argument("--box-max",    type=float, default=cfg["box_max"],  metavar="M", help="Max box side in metres (default: %(default)s)")

    args = p.parse_args()
    cfg = cfg.copy()
    cfg["n_samples"]     = args.n
    cfg["seed"]          = args.seed
    cfg["output_dir"]    = args.out
    cfg["noise_std"]     = args.noise
    cfg["dropout_ratio"] = args.dropout
    cfg["voxel_size"]    = args.voxel
    cfg["outlier_ratio"] = args.outliers
    cfg["p_two_boxes"]   = args.p_two_boxes
    cfg["p_person"]      = args.p_person
    cfg["p_forklift"]    = args.p_forklift
    cfg["p_pallet"]      = args.p_pallet
    cfg["box_min"]       = args.box_min
    cfg["box_max"]       = args.box_max
    return cfg


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args(CFG)
    rng = np.random.default_rng(cfg["seed"])

    # Load forklift STL once (reused across scenes)
    stl_path = Path(__file__).parent.parent / "data" / "forklift.stl"
    forklift_mesh: o3d.geometry.TriangleMesh | None = None
    if stl_path.exists():
        print(f"Loading forklift STL from {stl_path} …")
        forklift_mesh = load_forklift(str(stl_path))
    else:
        print(f"[warn] forklift.stl not found at {stl_path}, using primitive traspaleta instead.")

    all_meta: list[dict] = []
    n = cfg["n_samples"]
    print(f"Generating {n} scenes → {cfg['output_dir']}")
    print(f"  noise={cfg['noise_std']}m  dropout={cfg['dropout_ratio']}  voxel={cfg['voxel_size']}m")
    print(f"  p_forklift={cfg['p_forklift']}  p_person={cfg['p_person']}  p_pallet={cfg['p_pallet']}  p_two_boxes={cfg['p_two_boxes']}")

    for i in range(n):
        meta = generate_scene(i, rng, cfg, forklift_mesh)
        all_meta.append(meta)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:4d}/{n}] pts={meta['n_points']:6d}  objects={meta['objects']}")

    meta_path = Path(cfg["output_dir"]) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nDone. Metadata → {meta_path}")


if __name__ == "__main__":
    main()
