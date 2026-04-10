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
    "noise_std":        0.030,   # m  Gaussian noise per point (calibrated to real roughness σ≈30mm)
    "dropout_ratio":    0.15,    # fraction of points removed
    "outlier_ratio":    0.03,    # fraction turned into local outliers
    "voxel_size":       0.019,   # m  voxel grid; calibrated to real NN spacing ~55mm (was 10mm)
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
    "p_multi_cargo": 0.00,  # prob. of secondary cargo (stacked/tandem)
    "p_flat_cargo":  0.00,  # prob. of very flat/low cargo — hard near-floor case
    "flat_min_h":    0.03,  # m  min height in flat-cargo mode (overrides box_min_h / cyl_min_h)
    "flat_max_h":    0.15,  # m  max height in flat-cargo mode
    "p_person":     0.00,   # disabled for now
    "p_forklift":   0.00,   # disabled; always use primitive traspaleta
    "p_pallet":     1.00,   # EUR pallet always present under cargo
    "enable_floor": True,   # include floor plane points

    # Cargo box size range (m) — axis-aligned only, fits within EUR pallet footprint
    "box_min_w": 0.30,   # X width  min
    "box_max_w": 1.00,   # X width  max  (<1.20 pallet width)
    "box_min_d": 0.30,   # Z depth  min
    "box_max_d": 0.75,   # Z depth  max  (<0.80 pallet depth)
    "box_min_h": 0.25,   # Y height min
    "box_max_h": 1.40,   # Y height max

    # Floor extent (m) — half-size of the sampled floor patch
    # Real FUSION3D: X span ≈ 5.85m → half-extent 2.5m; Z span ≈ 4.4m → half-extent 2.0m
    "floor_extent_x": 2.5,   # m  half-size along X  → 5.0m total span
    "floor_extent_z": 2.0,   # m  half-size along Z  → 4.0m total span

    # ── Cylinder cargo (alternative to box) ──
    "p_cylinder":   0.00,   # probability of cylinder instead of box as primary cargo
    "cyl_min_r":    0.15,   # m  radius min  (bobina pequeña / bidón)
    "cyl_max_r":    0.40,   # m  radius max  (bobina grande / depósito)
    "cyl_min_h":    0.30,   # m  height min
    "cyl_max_h":    1.20,   # m  height max

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


def make_cylinder_mesh(r: float, h: float) -> o3d.geometry.TriangleMesh:
    """Upright cylinder centred in XZ, bottom face at Y=0.
    Open3D create_cylinder is Z-aligned by default → rotate 90° around X to make it Y-aligned.
    """
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h, resolution=32)
    R = np.array([[1, 0,  0],
                  [0, 0, -1],
                  [0, 1,  0]], dtype=np.float64)
    m.rotate(R, center=(0.0, 0.0, 0.0))
    # After rotation cylinder spans Y=-h/2..+h/2 → translate up so bottom sits on Y=0
    m.translate([0.0, h / 2, 0.0])
    return m


def _spec_w(spec: dict) -> float:
    """X-width of a primitive spec (box→w, cylinder→2r)."""
    return spec.get("w", spec.get("r", 0.0) * 2)


def _spec_d(spec: dict) -> float:
    """Z-depth of a primitive spec (box→d, cylinder→2r)."""
    return spec.get("d", spec.get("r", 0.0) * 2)


def make_primitive_mesh(spec: dict) -> o3d.geometry.TriangleMesh:
    """Create mesh from a cargo spec dict. Bottom at Y=0, centred in XZ."""
    t = spec["type"]
    if t == "box":
        return make_box_mesh(spec["w"], spec["h"], spec["d"])
    if t == "cylinder":
        return make_cylinder_mesh(spec["r"], spec["h"])
    raise ValueError(f"Unknown primitive type: {t!r}")


def sample_cargo_spec(
    cfg: dict, rng,
    max_w: float = None,
    max_d: float = None,
    max_h: float = None,
) -> dict:
    """Sample a single cargo primitive spec (box or cylinder).

    max_w / max_d: upper-bound clamps on X-width / Z-depth (stacked mode: secondary ≤ primary).
    max_h: upper-bound clamp on height (flat-cargo mode: very low / almost-floor-level).
    """
    flat_min = cfg.get("flat_min_h", 0.03)  # minimum height used only in flat-cargo mode
    if rng.random() < cfg.get("p_cylinder", 0.0):
        # Cylinder footprint is 2r × 2r — respect max_w / max_d so the upper cylinder
        # never exceeds the lower cargo footprint (stability constraint in stacked mode).
        if max_w is not None or max_d is not None:
            max_dim = min(
                max_w if max_w is not None else float("inf"),
                max_d if max_d is not None else float("inf"),
            )
            r_max = min(cfg["cyl_max_r"], max_dim / 2)
        else:
            r_max = cfg["cyl_max_r"]
        r_max = max(cfg["cyl_min_r"], r_max)
        if max_h is not None:
            h_lo, h_hi = flat_min, max(flat_min, min(cfg["cyl_max_h"], max_h))
        else:
            h_lo, h_hi = cfg["cyl_min_h"], cfg["cyl_max_h"]
        return {
            "type": "cylinder",
            "r": float(rng.uniform(cfg["cyl_min_r"], r_max)),
            "h": float(rng.uniform(h_lo, h_hi)),
        }
    w_max = min(cfg["box_max_w"], max_w) if max_w is not None else cfg["box_max_w"]
    d_max = min(cfg["box_max_d"], max_d) if max_d is not None else cfg["box_max_d"]
    if max_h is not None:
        h_lo, h_hi = flat_min, max(flat_min, min(cfg["box_max_h"], max_h))
    else:
        h_lo, h_hi = cfg["box_min_h"], cfg["box_max_h"]
    return {
        "type": "box",
        "w": float(rng.uniform(cfg["box_min_w"], max(cfg["box_min_w"], w_max))),
        "h": float(rng.uniform(h_lo, h_hi)),
        "d": float(rng.uniform(cfg["box_min_d"], max(cfg["box_min_d"], d_max))),
    }


def compose_cargo(
    specs: list,
    mode,           # "stacked" | "tandem" | None
    pallet_top_y: float,
    rng,
) -> tuple:
    """Position 1 or 2 cargo primitives and return positioned meshes.

    Modes
    -----
    stacked : secondary cargo placed on top of primary (Y direction).
    tandem  : secondary cargo placed in front of or behind primary (Z direction).
              Keeps cargo within the pallet footprint — no X-axis spread.

    Returns
    -------
    items : list of (positioned_mesh, placed_spec_dict)
        placed_spec_dict is the original spec augmented with ox/oy/oz keys.
    cargo_back_z : float
        Most negative Z extent across all items — used to anchor the jack.
    """
    if len(specs) == 1 or mode is None:
        spec = specs[0]
        mesh = make_primitive_mesh(spec)
        mesh.translate([0.0, pallet_top_y, 0.0])
        placed = {**spec, "ox": 0.0, "oy": round(pallet_top_y, 4), "oz": 0.0}
        return [(mesh, placed)], -_spec_d(spec) / 2

    spec1, spec2 = specs[0], specs[1]
    mesh1 = make_primitive_mesh(spec1)
    mesh2 = make_primitive_mesh(spec2)

    if mode == "stacked":
        h1 = spec1.get("h", 0.0)
        # Allow small offset only if spec2 is strictly smaller — keeps it on top
        max_ox = max(0.0, (_spec_w(spec1) - _spec_w(spec2)) / 2)
        max_oz = max(0.0, (_spec_d(spec1) - _spec_d(spec2)) / 2)
        ox2 = float(rng.uniform(-max_ox, max_ox))
        oz2 = float(rng.uniform(-max_oz, max_oz))
        oy2 = pallet_top_y + h1
        mesh1.translate([0.0, pallet_top_y, 0.0])
        mesh2.translate([ox2, oy2, oz2])
        # cargo_back_z uses only the BASE item: the stacked item is above, not beside,
        # so it does not affect the jack position (jack goes under the pallet, not the cargo).
        cargo_back_z = -_spec_d(spec1) / 2
        items = [
            (mesh1, {**spec1, "ox": 0.0, "oy": round(pallet_top_y, 4), "oz": 0.0}),
            (mesh2, {**spec2, "ox": round(ox2, 4), "oy": round(oy2, 4),
                     "oz": round(oz2, 4), "placement": "stacked"}),
        ]
        return items, cargo_back_z

    if mode == "tandem":
        # The two items are centred together on the pallet (Z=0):
        #   cargo1 sits in the back half, cargo2 in the front half.
        # Ensemble centre at Z=0 →
        #   oz1 = -(d2 + gap) / 2   (behind centre)
        #   oz2 = +(d1 + gap) / 2   (in front of centre)
        # Constraint for pallet fit: d1 + gap + d2 ≤ EUR_D
        gap = 0.02  # m clearance between items in Z  (matches generate_scene TANDEM_GAP)
        d1, d2 = _spec_d(spec1), _spec_d(spec2)
        oz1 = float(-(d2 + gap) / 2)
        oz2 = float(+(d1 + gap) / 2)
        mesh1.translate([0.0, pallet_top_y, oz1])
        mesh2.translate([0.0, pallet_top_y, oz2])
        # cargo_back_z: back face of cargo1 (most negative Z)
        cargo_back_z = oz1 - d1 / 2   # = -(d1 + d2 + gap) / 2
        items = [
            (mesh1, {**spec1, "ox": 0.0, "oy": round(pallet_top_y, 4), "oz": round(oz1, 4)}),
            (mesh2, {**spec2, "ox": 0.0, "oy": round(pallet_top_y, 4),
                     "oz": round(oz2, 4), "placement": "tandem"}),
        ]
        return items, cargo_back_z

    raise ValueError(f"Unknown compose mode: {mode!r}")


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
    fork_l.translate([-0.35, 0.0, 0.0])
    fork_r = o3d.geometry.TriangleMesh.create_box(0.15, 0.08, 1.15)
    fork_r.translate([ 0.20, 0.0, 0.0])
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
    """Random uniform points on Y=0 plane within floor_extent_x / floor_extent_z."""
    ext_x = cfg["floor_extent_x"]
    ext_z = cfg["floor_extent_z"]
    n = cfg["pts_floor"]
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-ext_x, ext_x, n).astype(np.float32)
    pts[:, 2] = rng.uniform(-ext_z, ext_z, n).astype(np.float32)
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
    0:   ( 60,  60,  60),   # floor   — gris oscuro, retrocede sobre fondo oscuro CC
    1:   (220,  50,  50),   # cargo   — rojo
    2:   (  0, 120, 255),   # vehicle — azul vivo
    3:   ( 39, 174,  96),   # person  — verde
    4:   (255, 210,   0),   # pallet  — amarillo
    255: ( 60,  60,  60),   # outlier — mismo gris oscuro que suelo
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
    The pallet jack is always present, axis-aligned, its body front face
    touching the pallet back face (or cargo back face if no pallet).
    Floor is kept outside the camera FOV filter so the ground plane is always
    fully covered (no clipping at the edges).
    """
    meta: dict = {"id": scene_id, "objects": []}

    # ── Floor ──
    if cfg.get("enable_floor", True):
        floor_pts, floor_lbs = sample_floor(cfg, rng)
        meta["objects"].append("floor")
    else:
        floor_pts = np.empty((0, 3), dtype=np.float32)
        floor_lbs = np.empty(0, dtype=np.uint8)

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

    # ── Cargo (primary + optional secondary) ──
    # Flat-cargo mode: very low height — simulates hard-to-separate cases near floor level.
    flat_max_h: float | None = (
        cfg.get("flat_max_h", 0.15) if rng.random() < cfg.get("p_flat_cargo", 0.0) else None
    )
    spec1 = sample_cargo_spec(cfg, rng, max_h=flat_max_h)
    specs = [spec1]
    compose_mode = None

    if rng.random() < cfg.get("p_multi_cargo", 0.0):
        compose_mode = str(rng.choice(["stacked", "tandem"]))
        if compose_mode == "stacked":
            # Secondary must not exceed primary footprint (so it doesn't topple off).
            # Primary is already within the pallet → stacked item is too.
            max_w2, max_d2, max_h2 = _spec_w(spec1), _spec_d(spec1), None
            specs.append(sample_cargo_spec(cfg, rng, max_w=max_w2, max_d=max_d2, max_h=max_h2))
        else:  # tandem
            # Both items centred together on the pallet — ensemble centre at Z=0.
            # Constraint for pallet fit: d1 + gap + d2 ≤ EUR_D
            #   → d2 ≤ EUR_D - d1 - gap
            TANDEM_GAP = 0.02  # m between items (matches compose_cargo)
            max_d2 = EUR_D - _spec_d(spec1) - TANDEM_GAP
            min_d2 = min(cfg["box_min_d"], cfg["cyl_min_r"] * 2)
            if max_d2 >= min_d2:
                max_h2 = flat_max_h   # keep flat mode consistent for secondary
                specs.append(sample_cargo_spec(cfg, rng, max_w=None, max_d=max_d2, max_h=max_h2))
            else:
                # Primary too deep to fit any secondary in tandem — fall back to stacked
                compose_mode = "stacked"
                max_w2, max_d2_st, max_h2 = _spec_w(spec1), _spec_d(spec1), None
                specs.append(sample_cargo_spec(cfg, rng, max_w=max_w2, max_d=max_d2_st, max_h=max_h2))

    cargo_items, cargo_back_z = compose_cargo(specs, compose_mode, pallet_top_y, rng)

    for i, (mesh, placed) in enumerate(cargo_items):
        pts_c, lbs_c = sample_labeled(mesh, LABEL["cargo"], cfg["pts_box"])
        obj_pts.append(pts_c)
        obj_lbs.append(lbs_c)
        meta["objects"].append({f"cargo{i + 1}": placed})

    # ── Pallet jack (always present) ──
    # Origin of make_pallet_jack_mesh: body-front / fork-root at (X=0, Y=0, Z=0).
    # Forks extend in +Z (toward cargo/pallet front), body extends in -Z.
    #
    # Jack front anchor: placed behind the cargo back face with a minimum gap
    # large enough that Gaussian noise (σ=30mm) from both primitives does not
    # cause visible interpenetration.  min_gap = 3×noise_std ≈ 0.10 m.
    # With pallet: also respect the pallet back face (Z = -EUR_D/2 = -0.40m) —
    # whichever is further back wins so forks always fit under the pallet.
    MIN_JACK_GAP = 0.10   # m  (> 3 × noise_std=0.030 m)
    if has_pallet:
        # Jack carries the pallet — always anchored slightly behind the pallet back face.
        # A 2cm clearance avoids visual noise-overlap where both surfaces share Z=-0.40m
        # while keeping the forks fully inside the pallet fork pockets.
        JACK_PALLET_CLEARANCE = 0.02   # m
        jack_front_z = -EUR_D / 2 - JACK_PALLET_CLEARANCE  # = -0.42 m, fixed
    else:
        # No pallet: jack behind cargo back face with noise gap
        jack_front_z = cargo_back_z - MIN_JACK_GAP

    tj = make_pallet_jack_mesh()
    tj.translate([0.0, 0.0, jack_front_z])
    vp, vl = sample_labeled(tj, LABEL["vehicle"], cfg["pts_forklift"] // 3)
    obj_pts.append(vp); obj_lbs.append(vl)
    meta["objects"].append({"pallet_jack": {"front_z": round(jack_front_z, 3)}})

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
    p.add_argument("--p-multi-cargo", type=float, default=cfg["p_multi_cargo"], metavar="P", help="Prob secondary cargo primitive (stacked/tandem)  (default: %(default)s)")
    p.add_argument("--p-flat-cargo",  type=float, default=cfg["p_flat_cargo"],  metavar="P", help="Prob very flat/low cargo (≤flat-max-h)           (default: %(default)s)")
    p.add_argument("--flat-min-h",    type=float, default=cfg["flat_min_h"],    metavar="M", help="Min height for flat-cargo mode (m)               (default: %(default)s)")
    p.add_argument("--flat-max-h",    type=float, default=cfg["flat_max_h"],    metavar="M", help="Max height for flat-cargo mode (m)               (default: %(default)s)")
    p.add_argument("--p-person",     type=float, default=cfg["p_person"],     metavar="P", help="Prob person in scene   (default: %(default)s)")
    p.add_argument("--p-forklift",   type=float, default=cfg["p_forklift"],   metavar="P", help="Prob forklift in scene (default: %(default)s)")
    p.add_argument("--p-pallet",     type=float, default=cfg["p_pallet"],     metavar="P", help="Prob EUR pallet base   (default: %(default)s)")
    # Cylinder cargo
    p.add_argument("--p-cylinder",   type=float, default=cfg["p_cylinder"],  metavar="P", help="Prob cylinder instead of box  (default: %(default)s)")
    p.add_argument("--cyl-min-r",    type=float, default=cfg["cyl_min_r"],   metavar="M", help="Min cylinder radius (default: %(default)s)")
    p.add_argument("--cyl-max-r",    type=float, default=cfg["cyl_max_r"],   metavar="M", help="Max cylinder radius (default: %(default)s)")
    p.add_argument("--cyl-min-h",    type=float, default=cfg["cyl_min_h"],   metavar="M", help="Min cylinder height (default: %(default)s)")
    p.add_argument("--cyl-max-h",    type=float, default=cfg["cyl_max_h"],   metavar="M", help="Max cylinder height (default: %(default)s)")
    # Box dimensions (axis-aligned)
    p.add_argument("--box-min-w",  type=float, default=cfg["box_min_w"], metavar="M", help="Min box X width  (default: %(default)s)")
    p.add_argument("--box-max-w",  type=float, default=cfg["box_max_w"], metavar="M", help="Max box X width  (default: %(default)s)")
    p.add_argument("--box-min-d",  type=float, default=cfg["box_min_d"], metavar="M", help="Min box Z depth  (default: %(default)s)")
    p.add_argument("--box-max-d",  type=float, default=cfg["box_max_d"], metavar="M", help="Max box Z depth  (default: %(default)s)")
    p.add_argument("--box-min-h",  type=float, default=cfg["box_min_h"], metavar="M", help="Min box Y height (default: %(default)s)")
    p.add_argument("--box-max-h",  type=float, default=cfg["box_max_h"], metavar="M", help="Max box Y height (default: %(default)s)")
    # Floor extent (asymmetric to match real sensor FOV)
    p.add_argument("--floor-ext-x", type=float, default=cfg["floor_extent_x"], metavar="M", help="Floor half-extent X (default: %(default)s)")
    p.add_argument("--floor-ext-z", type=float, default=cfg["floor_extent_z"], metavar="M", help="Floor half-extent Z (default: %(default)s)")

    args = p.parse_args()
    cfg = cfg.copy()
    cfg["n_samples"]      = args.n
    cfg["seed"]           = args.seed
    cfg["output_dir"]     = args.out
    cfg["noise_std"]      = args.noise
    cfg["dropout_ratio"]  = args.dropout
    cfg["voxel_size"]     = args.voxel
    cfg["outlier_ratio"]  = args.outliers
    cfg["p_multi_cargo"]  = args.p_multi_cargo
    cfg["p_flat_cargo"]   = args.p_flat_cargo
    cfg["flat_min_h"]     = args.flat_min_h
    cfg["flat_max_h"]     = args.flat_max_h
    cfg["p_person"]       = args.p_person
    cfg["p_forklift"]     = args.p_forklift
    cfg["p_pallet"]       = args.p_pallet
    cfg["p_cylinder"]     = args.p_cylinder
    cfg["cyl_min_r"]      = args.cyl_min_r
    cfg["cyl_max_r"]      = args.cyl_max_r
    cfg["cyl_min_h"]      = args.cyl_min_h
    cfg["cyl_max_h"]      = args.cyl_max_h
    cfg["box_min_w"]      = args.box_min_w
    cfg["box_max_w"]      = args.box_max_w
    cfg["box_min_d"]      = args.box_min_d
    cfg["box_max_d"]      = args.box_max_d
    cfg["box_min_h"]      = args.box_min_h
    cfg["box_max_h"]      = args.box_max_h
    cfg["floor_extent_x"] = args.floor_ext_x
    cfg["floor_extent_z"] = args.floor_ext_z
    return cfg


# ── Generation entry point (importable, no sys.argv) ──────────────────────────

def run_generation(
    cfg: dict,
    progress_cb=None,   # optional callable(i, n) for UI progress bars
) -> list[dict]:
    """
    Pure generation function: receives a fully-built cfg dict, returns metadata.
    Does not touch sys.argv — safe to import from Streamlit or other UIs.

    progress_cb: optional callable(scene_index, total) called after each scene.
    """
    rng = np.random.default_rng(cfg["seed"])

    stl_path = Path(__file__).parent.parent / "data" / "forklift.stl"
    forklift_mesh: o3d.geometry.TriangleMesh | None = None
    if stl_path.exists():
        forklift_mesh = load_forklift(str(stl_path))

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    all_meta: list[dict] = []
    n = cfg["n_samples"]
    for i in range(n):
        meta = generate_scene(i, rng, cfg, forklift_mesh)
        all_meta.append(meta)
        if progress_cb is not None:
            progress_cb(i + 1, n)

    meta_path = Path(cfg["output_dir"]) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    return all_meta


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args(CFG)

    stl_path = Path(__file__).parent.parent / "data" / "forklift.stl"
    if stl_path.exists():
        print(f"Loading forklift STL from {stl_path} …")
    else:
        print(f"[warn] forklift.stl not found at {stl_path}, using primitive traspaleta instead.")

    print(f"Generating {cfg['n_samples']} scenes → {cfg['output_dir']}")
    print(f"  noise={cfg['noise_std']}m  dropout={cfg['dropout_ratio']}  voxel={cfg['voxel_size']}m")
    print(f"  floor={cfg.get('enable_floor', True)}  p_pallet={cfg['p_pallet']}  p_person={cfg['p_person']}  p_multi_cargo={cfg['p_multi_cargo']}")

    def print_progress(i, n):
        if i % 10 == 0 or i == n:
            print(f"  [{i:4d}/{n}]")

    all_meta = run_generation(cfg, progress_cb=print_progress)
    print(f"\nDone. {len(all_meta)} scenes → {cfg['output_dir']}/metadata.json")


if __name__ == "__main__":
    main()
