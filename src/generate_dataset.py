"""
generate_dataset.py  —  Synthetic labeled point cloud dataset
Simulates FUSION3D stereo-camera captures of cargo pallets.

Coordinate system: X right, Y up (height), Z toward cameras   [metres]
Calibrated to real FUSION3D noise: σ flat≈6.5mm, overall≈15mm → using 35mm (floor roughness calibrated)

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
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)

if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
    os.environ["XDG_SESSION_TYPE"] = "x11"

# ── Configuration ──────────────────────────────────────────────────────────────
CFG = {
    "n_samples":   100,
    "seed":        42,
    "output_dir":  "../output/dataset",

    # ── Sensor noise (FUSION3D BBB empirical calibration) ──
    # Non-Gaussian mixture + quadratic depth scaling + axial ray projection.
    # Targets: frontal 3.5m σ≈10/16mm; 42° tilt 4.3m σ≈27mm; 4.6m σ≈41mm.
    "noise_gaussian_frac": 0.60,   # fraction using Gaussian core
    "noise_core_ref":      0.010,  # m  σ_core at z_ref=3m
    "noise_tail_ref":      0.025,  # m  σ_tail (t-Student) at z_ref=3m
    "noise_tail_df":       4,      # degrees of freedom for t-distribution
    "noise_z_ref":         3.0,    # m  reference depth for quadratic scaling
    "dropout_ratio":    0.15,    # fraction of points removed
    "outlier_ratio":    0.03,    # fraction turned into local outliers
    "voxel_size":       0.019,   # m  voxel grid; calibrated to real ROI NN spacing ~19.5mm
    "local_outlier_std": 0.055,  # m  spread of local outlier clusters

    # ── Camera positions (BBB real heights: cenital 3.68m, der/izq 3.80m) ──
    # All cameras placed on the arch plane at Z=0 so the view cone reaches both
    # the cargo/pallet (Z>0) and the forklift mast (Z≈-0.9m behind fork root).
    # Cenital: looks straight down (pitch=90°). fov_deg=90° covers ±2.5m floor
    #   and sees the mast top at (Y=2.78, Z≈-0.87).
    # Der/Izq: X=3.33m gives 48.8° tilt from horizontal (≈BBB 47.5°).
    #   fov_deg=70° widens cone to include mast top and far floor corners.
    "cameras": [
        {"name": "cenital", "pos": [ 0.0,  3.68, 0.0], "fov_deg": 90.0},
        {"name": "der",     "pos": [ 3.33, 3.80, 0.0], "fov_deg": 70.0},
        {"name": "izq",     "pos": [-3.33, 3.80, 0.0], "fov_deg": 70.0},
    ],
    # ref_dist for apply_distance_density: at this distance, full density is kept.
    # Calibrated to mean camera-to-cargo distance (~3.0m) at real BBB heights.
    "density_ref_dist": 3.0,

    # ── Scene composition ──
    "p_multi_cargo": 0.25,  # prob. of secondary cargo (stacked/tandem)
    "p_flat_cargo":  0.10,  # prob. of very flat/low cargo — hard near-floor case
    "flat_min_h":    0.03,  # m  min height in flat-cargo mode (overrides box_min_h / cyl_min_h)
    "flat_max_h":    0.15,  # m  max height in flat-cargo mode
    "p_person":     0.30,   # 30% de escenas tienen persona
    "p_forklift":   0.50,   # prob carretilla instead of pallet jack (0=always jack)
    "p_pallet":     0.80,   # EUR pallet present under cargo (20% sin pallet)
    "p_cargo_on_vehicle": 0.50,  # cuando no hay pallet: prob. de cargo sobre horcas
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
    "p_cylinder":   0.15,   # probability of cylinder instead of box as primary cargo
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

    # ── Forklift (carretilla elevadora) STL ──
    # Path relative to src/, or absolute.  Empty string → no STL loaded (always jack).
    "forklift_stl": "../data/carretilla.stl",
}

# ── Submodule imports (B9 refactor) ───────────────────────────────────────────
from ply_io.ply import (
    LABEL, LABEL_RGB, labels_to_rgb, save_ply,
)
from geometry.meshes import (
    EUR_W, EUR_H, EUR_D,
    JACK_FORK_H, JACK_FORK_L, JACK_BODY_D, JACK_X_HALF,
    FORKLIFT_FORK_H, FORKLIFT_FORK_L, FORKLIFT_BODY_D, FORKLIFT_X_HALF,
    make_pallet_mesh, make_box_mesh, make_cylinder_mesh,
    make_primitive_mesh, make_person_mesh, make_pallet_jack_mesh,
    load_carretilla,
)
from geometry.composition import (
    _spec_w, _spec_d, sample_cargo_spec, compose_cargo, compose_cargo_on_vehicle,
)
from sensor.noise import (
    sample_labeled, sample_floor,
    camera_arc_filter, apply_distance_density, degrade_labeled,
)

# ── Re-exports (backwards compat for test_pipeline.py and other importers) ────
__all__ = [
    "CFG",
    "LABEL", "LABEL_RGB",
    "EUR_W", "EUR_H", "EUR_D",
    "make_pallet_mesh", "make_box_mesh", "make_cylinder_mesh",
    "make_primitive_mesh", "make_person_mesh", "make_pallet_jack_mesh",
    "load_carretilla",
    "_spec_w", "_spec_d", "sample_cargo_spec", "compose_cargo", "compose_cargo_on_vehicle",
    "sample_labeled", "sample_floor",
    "camera_arc_filter", "apply_distance_density", "degrade_labeled",
    "labels_to_rgb", "save_ply",
    "generate_scene", "run_generation", "parse_args",
]



def _person_no_collision(
    px: float, pz: float, pr: float,
    box_xmin: float, box_xmax: float,
    box_zmin: float, box_zmax: float,
) -> bool:
    """True si el círculo (px, pz, radio=pr) NO solapa con el AABB [xmin..xmax]×[zmin..zmax] en XZ."""
    nx = float(np.clip(px, box_xmin, box_xmax))
    nz = float(np.clip(pz, box_zmin, box_zmax))
    return (px - nx) ** 2 + (pz - nz) ** 2 >= pr * pr



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

    # ── Vehicle type selection (first, so fork constants are known for all placements) ──
    # Per scene, either pallet jack (traspaleta) or carretilla elevadora — never both.
    is_forklift = (
        forklift_mesh is not None
        and rng.random() < cfg.get("p_forklift", 0.0)
    )
    v_fork_h = FORKLIFT_FORK_H if is_forklift else JACK_FORK_H
    v_fork_l = FORKLIFT_FORK_L if is_forklift else JACK_FORK_L
    v_body_d = FORKLIFT_BODY_D if is_forklift else JACK_BODY_D
    v_x_half = FORKLIFT_X_HALF if is_forklift else JACK_X_HALF

    # ── Object point lists (will be FOV-filtered) ──
    obj_pts: list[np.ndarray] = []
    obj_lbs: list[np.ndarray] = []

    # ── Pallet ──
    has_pallet = rng.random() < cfg["p_pallet"]
    pallet_top_y = 0.0
    if has_pallet:
        pm = make_pallet_mesh()
        if is_forklift:
            # Forklift carries the pallet on its forks: lift pallet to fork platform height.
            pm.translate([0.0, v_fork_h, 0.0])
        pp, pl = sample_labeled(pm, LABEL["pallet"], cfg["pts_pallet"])
        obj_pts.append(pp); obj_lbs.append(pl)
        pallet_top_y = (v_fork_h if is_forklift else 0.0) + EUR_H
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

    # ── Cargo composition — normal vs cargo-on-vehicle ──
    has_cargo_on_vehicle = (
        not has_pallet and rng.random() < cfg.get("p_cargo_on_vehicle", 0.50)
    )
    if has_cargo_on_vehicle:
        # BBB-5 scenario: cargo placed directly on fork tops, no pallet beneath.
        mesh, placed = compose_cargo_on_vehicle(specs[0], rng, fork_h=v_fork_h, fork_l=v_fork_l)
        cargo_items = [(mesh, placed)]
    else:
        cargo_items, cargo_back_z = compose_cargo(specs, compose_mode, pallet_top_y, rng)

    for i, (mesh, placed) in enumerate(cargo_items):
        pts_c, lbs_c = sample_labeled(mesh, LABEL["cargo"], cfg["pts_box"])
        obj_pts.append(pts_c)
        obj_lbs.append(lbs_c)
        meta["objects"].append({f"cargo{i + 1}": placed})

    # ── Vehicle (traspaleta o carretilla elevadora — mutuamente exclusivas) ──
    # Vehicle origin: fork-root at (X=0, Y=0, Z=0).
    # Forks extend in +Z (toward cargo/pallet), body in -Z (operator side).
    # vehicle_front_z: world Z of the fork root for this scene.
    MIN_VEHICLE_GAP       = 0.10   # m  (> 3 × noise_std=0.030 m)
    VEHICLE_PALLET_CLEAR  = 0.02   # m  avoids noise-overlap at pallet back face
    if has_pallet:
        vehicle_front_z = -EUR_D / 2 - VEHICLE_PALLET_CLEAR   # = -0.42 m, fixed
    elif has_cargo_on_vehicle:
        vehicle_front_z = 0.0
    else:
        # Cargo is on the floor. Jack forks (h=6 cm) can slide under floor-level cargo,
        # but forklift forks (h=25 cm) cannot — park them fully behind the cargo back face
        # so fork tips stop at cargo_back_z − MIN_VEHICLE_GAP.
        if is_forklift:
            vehicle_front_z = cargo_back_z - MIN_VEHICLE_GAP - v_fork_l
        else:
            vehicle_front_z = cargo_back_z - MIN_VEHICLE_GAP

    if is_forklift:
        vm = o3d.geometry.TriangleMesh(forklift_mesh)   # shallow copy; shared vertices/triangles
        vm.translate([0.0, 0.0, vehicle_front_z])
        n_pts_v = cfg["pts_forklift"]
        v_type  = "forklift"
    else:
        vm = make_pallet_jack_mesh()
        vm.translate([0.0, 0.0, vehicle_front_z])
        n_pts_v = cfg["pts_forklift"] // 3
        v_type  = "pallet_jack"

    vp, vl = sample_labeled(vm, LABEL["vehicle"], n_pts_v)
    obj_pts.append(vp); obj_lbs.append(vl)
    meta["objects"].append({"vehicle": {"type": v_type, "front_z": round(vehicle_front_z, 3)}})

    # ── Extra floor: extend Z to cover full vehicle body + operator margin ──
    # Person operator zone can reach vehicle_zmin − 0.70 m.
    # Use 1.5 m safety margin beyond vehicle rear to guarantee floor coverage.
    if cfg.get("enable_floor", True):
        vehicle_zmin = vehicle_front_z - v_body_d
        needed_z_min = vehicle_zmin - 1.5          # 1.5 m behind vehicle rear
        std_floor_z  = -cfg["floor_extent_z"]
        if needed_z_min < std_floor_z:
            extra_len   = std_floor_z - needed_z_min
            extra_width = cfg["floor_extent_x"] * 2
            n_extra = int(cfg["pts_floor"] * (extra_len * extra_width)
                          / (cfg["floor_extent_x"] * 2 * cfg["floor_extent_z"] * 2))
            n_extra = max(n_extra, 5_000)
            ex = rng.uniform(-extra_width / 2, extra_width / 2, n_extra).astype(np.float32)
            ez = rng.uniform(needed_z_min, std_floor_z, n_extra).astype(np.float32)
            ey = np.zeros(n_extra, dtype=np.float32)
            floor_pts = np.vstack([floor_pts, np.stack([ex, ey, ez], axis=1).astype(np.float32)])
            floor_lbs = np.concatenate([floor_lbs, np.zeros(n_extra, dtype=np.uint8)])

    # ── Persona (opcional) ──
    # Zona operario: detrás del cuerpo del vehículo.
    # Vehicle bbox en mundo: X=[−v_x_half, +v_x_half],
    #   Z=[vehicle_front_z − v_body_d, vehicle_front_z + v_fork_l].
    if rng.random() < cfg["p_person"]:
        PERSON_R = 0.30    # radio huella persona (m)
        MAX_TRY  = 20
        vehicle_xmin = -v_x_half
        vehicle_xmax = +v_x_half
        vehicle_zmin = vehicle_front_z - v_body_d   # cara trasera del cuerpo
        vehicle_zmax = vehicle_front_z + v_fork_l   # punta de las horquillas

        p_height = float(rng.uniform(1.70, 1.80))
        p_rot    = float(rng.uniform(0.0, 360.0))
        px = pz = None

        # 60% zona operario (si hay pallet), 40% perímetro cargo
        use_operator = has_pallet and (rng.random() < 0.60)
        if use_operator:
            for _ in range(MAX_TRY):
                cx = float(rng.uniform(-0.50, +0.50))
                cz = float(vehicle_zmin - rng.uniform(0.30, 0.70))
                if _person_no_collision(cx, cz, PERSON_R, vehicle_xmin, vehicle_xmax, vehicle_zmin, vehicle_zmax):
                    px, pz = cx, cz
                    break

        if px is None:   # fallback al perímetro si zona operario falló o no aplica
            half_diag = np.sqrt((EUR_W / 2) ** 2 + (EUR_D / 2) ** 2)
            for _ in range(MAX_TRY):
                theta = float(rng.uniform(0.0, 2 * np.pi))
                r     = half_diag + float(rng.uniform(0.30, 0.70))
                cx    = float(r * np.cos(theta))
                cz    = float(r * np.sin(theta))
                if _person_no_collision(cx, cz, PERSON_R, vehicle_xmin, vehicle_xmax, vehicle_zmin, vehicle_zmax):
                    px, pz = cx, cz
                    break

        if px is not None:
            _person_stl = Path(__file__).parent.parent / "data" / "person.stl"
            pm = make_person_mesh(height=p_height, y_rotation_deg=p_rot)
            pm.translate([px, 0.0, pz])
            pp, pl = sample_labeled(pm, LABEL["person"], cfg["pts_person"])
            obj_pts.append(pp)
            obj_lbs.append(pl)
            meta["objects"].append({"person": {
                "x":       round(px, 3),
                "z":       round(pz, 3),
                "height":  round(p_height, 3),
                "rot_deg": round(p_rot, 1),
                "zone":    "operator" if use_operator else "perimeter",
                "stl":     "person.stl" if _person_stl.exists() else "fallback",
            }})

    # ── Apply camera FOV filter to non-vehicle objects ──
    # Vehicle (label 2) is never filtered: full STL geometry always retained.
    # Floor, pallet, cargo, person are filtered to realistic camera coverage.
    obj_all = np.vstack(obj_pts)
    lbs_all_obj = np.concatenate(obj_lbs)
    if not cfg.get("skip_camera_filter", False):
        is_veh = lbs_all_obj == LABEL["vehicle"]
        filtered_pts, filtered_lbs = camera_arc_filter(
            obj_all[~is_veh], lbs_all_obj[~is_veh], cfg["cameras"]
        )
        obj_all    = np.vstack([filtered_pts, obj_all[is_veh]])
        lbs_all_obj = np.concatenate([filtered_lbs, lbs_all_obj[is_veh]])

    # ── Merge floor (unfiltered) + objects (filtered) ──
    pts_all = np.vstack([floor_pts, obj_all])
    lbs_all = np.concatenate([floor_lbs, lbs_all_obj])

    # ── Distance-dependent density falloff ──
    pts_all, lbs_all = apply_distance_density(pts_all, lbs_all, cfg["cameras"], rng,
                                                ref_dist=cfg.get("density_ref_dist", 3.0))

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
    p.add_argument("--noise-core-ref", type=float, default=cfg["noise_core_ref"], metavar="M", help="σ_core @ z_ref=3m for Gaussian mixture component (default: %(default)s)")
    p.add_argument("--noise-tail-ref", type=float, default=cfg["noise_tail_ref"], metavar="M", help="σ_tail @ z_ref=3m for t-Student component       (default: %(default)s)")
    p.add_argument("--dropout",    type=float, default=cfg["dropout_ratio"],     help="Point dropout ratio 0-1     (default: %(default)s)")
    p.add_argument("--voxel",      type=float, default=cfg["voxel_size"],        help="Voxel grid size in metres   (default: %(default)s)")
    p.add_argument("--outliers",   type=float, default=cfg["outlier_ratio"],     help="Outlier ratio 0-1           (default: %(default)s)")
    # Scene composition probabilities
    p.add_argument("--p-multi-cargo", type=float, default=cfg["p_multi_cargo"], metavar="P", help="Prob secondary cargo primitive (stacked/tandem)  (default: %(default)s)")
    p.add_argument("--p-flat-cargo",  type=float, default=cfg["p_flat_cargo"],  metavar="P", help="Prob very flat/low cargo (≤flat-max-h)           (default: %(default)s)")
    p.add_argument("--flat-min-h",    type=float, default=cfg["flat_min_h"],    metavar="M", help="Min height for flat-cargo mode (m)               (default: %(default)s)")
    p.add_argument("--flat-max-h",    type=float, default=cfg["flat_max_h"],    metavar="M", help="Max height for flat-cargo mode (m)               (default: %(default)s)")
    p.add_argument("--p-person",     type=float, default=cfg["p_person"],     metavar="P", help="Prob person in scene                     (default: %(default)s)")
    p.add_argument("--p-forklift",   type=float, default=cfg["p_forklift"],   metavar="P", help="Prob carretilla instead of pallet jack   (default: %(default)s)")
    p.add_argument("--p-pallet",     type=float, default=cfg["p_pallet"],     metavar="P", help="Prob EUR pallet base                      (default: %(default)s)")
    p.add_argument("--forklift-stl", type=str,   default=cfg["forklift_stl"],              help="Path to carretilla STL, relative to src/  (default: %(default)s)")
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

    # ── Validation ──
    prob_args = {
        "--p-multi-cargo": args.p_multi_cargo,
        "--p-flat-cargo":  args.p_flat_cargo,
        "--p-person":      args.p_person,
        "--p-forklift":    args.p_forklift,
        "--p-pallet":      args.p_pallet,
        "--p-cylinder":    args.p_cylinder,
        "--dropout":       args.dropout,
        "--outliers":      args.outliers,
    }
    for name, val in prob_args.items():
        if not (0.0 <= val <= 1.0):
            p.error(f"{name} must be in [0, 1], got {val}")

    minmax_pairs = [
        ("--box-min-w", args.box_min_w, "--box-max-w", args.box_max_w),
        ("--box-min-d", args.box_min_d, "--box-max-d", args.box_max_d),
        ("--box-min-h", args.box_min_h, "--box-max-h", args.box_max_h),
        ("--cyl-min-r", args.cyl_min_r, "--cyl-max-r", args.cyl_max_r),
        ("--cyl-min-h", args.cyl_min_h, "--cyl-max-h", args.cyl_max_h),
        ("--flat-min-h", args.flat_min_h, "--flat-max-h", args.flat_max_h),
    ]
    for lo_name, lo_val, hi_name, hi_val in minmax_pairs:
        if lo_val > hi_val:
            p.error(f"{lo_name} ({lo_val}) must be ≤ {hi_name} ({hi_val})")

    cfg = cfg.copy()
    cfg["n_samples"]      = args.n
    cfg["seed"]           = args.seed
    cfg["output_dir"]     = args.out
    cfg["noise_core_ref"]  = args.noise_core_ref
    cfg["noise_tail_ref"]  = args.noise_tail_ref
    cfg["dropout_ratio"]   = args.dropout
    cfg["voxel_size"]     = args.voxel
    cfg["outlier_ratio"]  = args.outliers
    cfg["p_multi_cargo"]  = args.p_multi_cargo
    cfg["p_flat_cargo"]   = args.p_flat_cargo
    cfg["flat_min_h"]     = args.flat_min_h
    cfg["flat_max_h"]     = args.flat_max_h
    cfg["p_person"]       = args.p_person
    cfg["p_forklift"]     = args.p_forklift
    cfg["p_pallet"]       = args.p_pallet
    cfg["forklift_stl"]   = args.forklift_stl
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

    forklift_mesh: o3d.geometry.TriangleMesh | None = None
    stl_cfg = cfg.get("forklift_stl", "")
    if stl_cfg:
        stl_path = (Path(__file__).parent / stl_cfg).resolve()
        if stl_path.exists():
            forklift_mesh = load_carretilla(str(stl_path))
        elif cfg.get("p_forklift", 0.0) > 0.0:
            raise FileNotFoundError(
                f"p_forklift={cfg['p_forklift']} but carretilla STL not found: {stl_path}"
            )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.ply"):
        old.unlink()
    meta_old = out_dir / "metadata.json"
    if meta_old.exists():
        meta_old.unlink()

    all_meta: list[dict] = []
    n = cfg["n_samples"]
    for i in range(n):
        meta = generate_scene(i, rng, cfg, forklift_mesh)
        all_meta.append(meta)
        if progress_cb is not None:
            progress_cb(i + 1, n)

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    return all_meta


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args(CFG)

    _stl_cfg = cfg.get("forklift_stl", "")
    if _stl_cfg:
        _stl_p = (Path(__file__).parent / _stl_cfg).resolve()
        if _stl_p.exists():
            log.info("Carretilla STL: %s  (p_forklift=%.2f)", _stl_p, cfg.get("p_forklift", 0.0))
        elif cfg.get("p_forklift", 0.0) > 0.0:
            log.error("STL no encontrado: %s — abortando (p_forklift=%.2f > 0)", _stl_p, cfg["p_forklift"])
        else:
            log.info("STL no encontrado (%s); p_forklift=0 → siempre traspaleta", _stl_p)

    log.info("Generating %d scenes → %s", cfg['n_samples'], cfg['output_dir'])
    log.info("  noise_core=%.3fm  noise_tail=%.3fm  dropout=%.2f  voxel=%.3fm",
             cfg['noise_core_ref'], cfg['noise_tail_ref'], cfg['dropout_ratio'], cfg['voxel_size'])
    log.info("  floor=%s  p_pallet=%.2f  p_person=%.2f  p_multi_cargo=%.2f",
             cfg.get('enable_floor', True), cfg['p_pallet'], cfg['p_person'], cfg['p_multi_cargo'])

    def print_progress(i, n):
        if i % 10 == 0 or i == n:
            log.info("  [%4d/%d]", i, n)

    all_meta = run_generation(cfg, progress_cb=print_progress)
    log.info("Done. %d scenes → %s/metadata.json", len(all_meta), cfg['output_dir'])


if __name__ == "__main__":
    main()
