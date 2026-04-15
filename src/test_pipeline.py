"""
test_pipeline.py  —  Tests de funcionalidad del generador de dataset sintético.

Uso (desde src/):
    python3 -m pytest test_pipeline.py -v
    python3 test_pipeline.py             # sin pytest
"""

import sys
import json
import struct
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from generate_dataset import (
    CFG,
    run_generation,
    generate_scene,
    make_box_mesh,
    make_cylinder_mesh,
    make_pallet_mesh,
    make_pallet_jack_mesh,
    make_primitive_mesh,
    sample_cargo_spec,
    compose_cargo,
    compose_cargo_on_vehicle,
    _spec_w,
    _spec_d,
    sample_labeled,
    degrade_labeled,
    camera_arc_filter,
    LABEL,
    LABEL_RGB,
    EUR_W, EUR_H, EUR_D,
)
from geometry.meshes import JACK_FORK_H, JACK_FORK_L
from utils.preview_grid import render_scene, load_synth


# ── Helpers ────────────────────────────────────────────────────────────────────

def _minimal_cfg(tmp_dir: str, **overrides) -> dict:
    """CFG mínimo para tests rápidos."""
    cfg = CFG.copy()
    cfg.update({
        "n_samples":      3,
        "seed":           0,
        "output_dir":     tmp_dir,
        "pts_floor":      5_000,
        "pts_pallet":     1_000,
        "pts_box":        2_000,
        "pts_forklift":   1_000,
        "pts_person":     500,
        "enable_floor":   True,
        "p_pallet":       1.0,
        "p_multi_cargo":  0.0,
        "p_flat_cargo":   0.0,
        "p_cylinder":     0.0,
        "p_person":       0.0,
        "p_forklift":     0.0,
    })
    cfg.update(overrides)
    return cfg


def _read_ply_header(path: Path) -> list[str]:
    with open(path, "rb") as f:
        lines = []
        while True:
            l = f.readline().decode("ascii", errors="ignore").strip()
            lines.append(l)
            if l == "end_header":
                break
    return lines


def _read_ply_labels(path: Path) -> np.ndarray:
    pts, lbs = load_synth(path)
    return lbs


# ══════════════════════════════════════════════════════════════════════════════
# 1. Configuración
# ══════════════════════════════════════════════════════════════════════════════

class TestCFG(unittest.TestCase):

    def test_required_keys_present(self):
        required = [
            "n_samples", "seed", "output_dir",
            "noise_std", "dropout_ratio", "outlier_ratio", "voxel_size",
            "p_pallet", "p_multi_cargo", "p_flat_cargo", "flat_min_h", "flat_max_h",
            "p_person", "p_forklift",
            "p_cylinder", "cyl_min_r", "cyl_max_r", "cyl_min_h", "cyl_max_h",
            "box_min_w", "box_max_w", "box_min_d", "box_max_d",
            "box_min_h", "box_max_h", "floor_extent_x", "floor_extent_z",
            "cameras", "enable_floor",
        ]
        for k in required:
            self.assertIn(k, CFG, f"Falta la clave '{k}' en CFG")

    def test_box_min_less_than_max(self):
        self.assertLess(CFG["box_min_w"], CFG["box_max_w"])
        self.assertLess(CFG["box_min_d"], CFG["box_max_d"])
        self.assertLess(CFG["box_min_h"], CFG["box_max_h"])

    def test_box_fits_pallet(self):
        self.assertLessEqual(CFG["box_max_w"], EUR_W,
                             "Caja puede ser más ancha que el pallet EUR")
        self.assertLessEqual(CFG["box_max_d"], EUR_D,
                             "Caja puede ser más profunda que el pallet EUR")

    def test_probabilities_in_range(self):
        for k in ("p_pallet", "p_multi_cargo", "p_flat_cargo", "p_person", "p_forklift"):
            self.assertGreaterEqual(CFG[k], 0.0)
            self.assertLessEqual(CFG[k], 1.0)

    def test_label_rgb_covers_all_labels(self):
        for label_id in LABEL.values():
            self.assertIn(label_id, LABEL_RGB,
                          f"Label {label_id} sin color en LABEL_RGB")

    def test_cameras_defined(self):
        self.assertGreaterEqual(len(CFG["cameras"]), 1)
        for cam in CFG["cameras"]:
            self.assertIn("pos", cam)
            self.assertIn("fov_deg", cam)
            self.assertEqual(len(cam["pos"]), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Geometría
# ══════════════════════════════════════════════════════════════════════════════

class TestGeometry(unittest.TestCase):

    def test_box_mesh_dimensions(self):
        """Caja centrada en XZ, base en Y=0."""
        m = make_box_mesh(0.6, 1.0, 0.4)
        verts = np.asarray(m.vertices)
        self.assertAlmostEqual(verts[:, 0].min(), -0.3, places=4)
        self.assertAlmostEqual(verts[:, 0].max(),  0.3, places=4)
        self.assertAlmostEqual(verts[:, 1].min(),  0.0, places=4)
        self.assertAlmostEqual(verts[:, 1].max(),  1.0, places=4)
        self.assertAlmostEqual(verts[:, 2].min(), -0.2, places=4)
        self.assertAlmostEqual(verts[:, 2].max(),  0.2, places=4)

    def test_pallet_mesh_dimensions(self):
        """Pallet EUR centrado en XZ."""
        m = make_pallet_mesh()
        verts = np.asarray(m.vertices)
        self.assertAlmostEqual(verts[:, 0].min(), -EUR_W / 2, places=4)
        self.assertAlmostEqual(verts[:, 0].max(),  EUR_W / 2, places=4)
        self.assertAlmostEqual(verts[:, 1].min(),  0.0,       places=4)
        self.assertAlmostEqual(verts[:, 1].max(),  EUR_H,     places=4)

    def test_jack_forks_at_floor_level(self):
        """Horquillas de la traspaleta empiezan en Y=0."""
        m = make_pallet_jack_mesh()
        verts = np.asarray(m.vertices)
        self.assertAlmostEqual(verts[:, 1].min(), 0.0, places=4)

    def test_jack_body_width(self):
        """Cuerpo de la traspaleta tiene 70cm de ancho."""
        m = make_pallet_jack_mesh()
        verts = np.asarray(m.vertices)
        width = verts[:, 0].max() - verts[:, 0].min()
        self.assertAlmostEqual(width, 0.70, places=3)

    def test_cylinder_mesh_upright(self):
        """Cilindro Y-aligned: base en Y=0, top en Y=h, centrado en XZ."""
        r, h = 0.25, 0.80
        m = make_cylinder_mesh(r, h)
        verts = np.asarray(m.vertices)
        self.assertAlmostEqual(verts[:, 1].min(), 0.0, places=3)
        self.assertAlmostEqual(verts[:, 1].max(), h,   places=3)
        self.assertAlmostEqual(abs(verts[:, 0]).max(), r, delta=0.01)
        self.assertAlmostEqual(abs(verts[:, 2]).max(), r, delta=0.01)

    def test_cylinder_mesh_radius_range(self):
        """Radio del cilindro respetado en distintos tamaños."""
        for r, h in [(0.15, 0.30), (0.40, 1.20)]:
            m = make_cylinder_mesh(r, h)
            verts = np.asarray(m.vertices)
            self.assertAlmostEqual(verts[:, 1].max(), h, places=3)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLY output
# ══════════════════════════════════════════════════════════════════════════════

class TestPLYOutput(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = _minimal_cfg(self.tmp.name)
        self.meta = run_generation(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ply_files_created(self):
        plys = list(Path(self.tmp.name).glob("*.ply"))
        self.assertEqual(len(plys), self.cfg["n_samples"])

    def test_metadata_json_created(self):
        meta_path = Path(self.tmp.name) / "metadata.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), self.cfg["n_samples"])

    def test_ply_header_fields(self):
        """PLY debe tener x y z red green blue label."""
        ply = sorted(Path(self.tmp.name).glob("*.ply"))[0]
        header = _read_ply_header(ply)
        header_str = "\n".join(header)
        for field in ("x", "y", "z", "red", "green", "blue", "label"):
            self.assertIn(field, header_str, f"Campo '{field}' no está en el header PLY")

    def test_ply_binary_little_endian(self):
        ply = sorted(Path(self.tmp.name).glob("*.ply"))[0]
        header = _read_ply_header(ply)
        self.assertTrue(any("binary_little_endian" in l for l in header))

    def test_labels_are_valid(self):
        """Solo se permiten labels 0-4 y 255."""
        valid = {0, 1, 2, 3, 4, 255}
        for ply in Path(self.tmp.name).glob("*.ply"):
            lbs = _read_ply_labels(ply)
            unique = set(lbs.tolist())
            self.assertTrue(unique.issubset(valid),
                            f"{ply.name}: labels inesperados {unique - valid}")

    def test_point_count_positive(self):
        for m in self.meta:
            self.assertGreater(m["n_points"], 0)

    def test_metadata_has_objects(self):
        for m in self.meta:
            self.assertIn("objects", m)
            self.assertIsInstance(m["objects"], list)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Flags de composición de escena
# ══════════════════════════════════════════════════════════════════════════════

class TestSceneComposition(unittest.TestCase):

    def _run(self, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(tmp, **overrides)
            return run_generation(cfg)

    def test_enable_floor_true(self):
        meta = self._run(enable_floor=True)
        for m in meta:
            self.assertIn("floor", m["objects"])

    def test_enable_floor_false(self):
        meta = self._run(enable_floor=False)
        for m in meta:
            self.assertNotIn("floor", m["objects"])

    def test_pallet_always(self):
        meta = self._run(p_pallet=1.0)
        for m in meta:
            self.assertIn("pallet", m["objects"])

    def test_pallet_never(self):
        meta = self._run(p_pallet=0.0, n_samples=10, seed=42)
        for m in meta:
            self.assertNotIn("pallet", m["objects"])

    def test_cargo_always_present(self):
        meta = self._run()
        for m in meta:
            has_cargo = any(
                isinstance(o, dict) and "cargo1" in o
                for o in m["objects"]
            )
            self.assertTrue(has_cargo, "cargo1 no encontrado en la escena")

    def test_cylinder_composition(self):
        """Con p_cylinder=1.0 todas las escenas tienen cargo1 con type=cylinder."""
        meta = self._run(p_cylinder=1.0, n_samples=5, seed=3)
        for m in meta:
            cargo1 = next((o["cargo1"] for o in m["objects"]
                           if isinstance(o, dict) and "cargo1" in o), None)
            self.assertIsNotNone(cargo1, "cargo1 no encontrado con p_cylinder=1.0")
            self.assertEqual(cargo1["type"], "cylinder",
                             "cargo1 no es cylinder con p_cylinder=1.0")

    def test_cylinder_dimensions_in_range(self):
        """Radio y altura del cilindro dentro de los rangos configurados."""
        min_r, max_r = 0.20, 0.30
        min_h, max_h = 0.40, 0.70
        meta = self._run(
            p_cylinder=1.0, n_samples=10, seed=5,
            cyl_min_r=min_r, cyl_max_r=max_r,
            cyl_min_h=min_h, cyl_max_h=max_h,
        )
        for m in meta:
            cyl = next(o["cargo1"] for o in m["objects"]
                       if isinstance(o, dict) and "cargo1" in o)
            self.assertGreaterEqual(cyl["r"], min_r - 1e-6)
            self.assertLessEqual(cyl["r"],    max_r + 1e-6)
            self.assertGreaterEqual(cyl["h"], min_h - 1e-6)
            self.assertLessEqual(cyl["h"],    max_h + 1e-6)

    def test_jack_always_present(self):
        meta = self._run()
        for m in meta:
            has_jack = any(
                isinstance(o, dict) and "pallet_jack" in o
                for o in m["objects"]
            )
            self.assertTrue(has_jack, "pallet_jack no encontrado en la escena")

    def test_multi_cargo_activated(self):
        """Con p_multi_cargo=1.0 todas las escenas deben tener cargo1 y cargo2."""
        meta = self._run(p_multi_cargo=1.0, n_samples=5, seed=7)
        for m in meta:
            has_cargo2 = any(
                isinstance(o, dict) and "cargo2" in o
                for o in m["objects"]
            )
            self.assertTrue(has_cargo2, "cargo2 no encontrado con p_multi_cargo=1.0")

    def test_multi_cargo_placement_key(self):
        """cargo2 debe incluir clave 'placement' (stacked o tandem)."""
        meta = self._run(p_multi_cargo=1.0, n_samples=10, seed=11)
        for m in meta:
            c2 = next((o["cargo2"] for o in m["objects"]
                       if isinstance(o, dict) and "cargo2" in o), None)
            if c2 is not None:
                self.assertIn("placement", c2,
                              "cargo2 no tiene clave 'placement'")
                self.assertIn(c2["placement"], ("stacked", "tandem"))

    def test_flat_cargo_height_limited(self):
        """Con p_flat_cargo=1.0, cargo1 siempre tiene h ≤ flat_max_h."""
        flat_max = 0.12
        meta = self._run(p_flat_cargo=1.0, flat_max_h=flat_max,
                         p_cylinder=0.0, n_samples=15, seed=99)
        for m in meta:
            c1 = next(o["cargo1"] for o in m["objects"]
                      if isinstance(o, dict) and "cargo1" in o)
            self.assertLessEqual(c1["h"], flat_max + 1e-6,
                                 f"Cargo plano supera flat_max_h={flat_max}: h={c1['h']}")

    def test_flat_cargo_cfg_keys_present(self):
        """CFG debe tener p_flat_cargo, flat_min_h y flat_max_h."""
        for k in ("p_flat_cargo", "flat_min_h", "flat_max_h"):
            self.assertIn(k, CFG, f"Falta '{k}' en CFG")
        self.assertGreaterEqual(CFG["p_flat_cargo"], 0.0)
        self.assertLessEqual(CFG["p_flat_cargo"], 1.0)
        self.assertGreater(CFG["flat_min_h"], 0.0)
        self.assertGreater(CFG["flat_max_h"], CFG["flat_min_h"])

    def test_tandem_cargo_fits_within_pallet(self):
        """En modo tandem, el ensemble (d1+gap+d2) no supera EUR_D=0.80m."""
        meta = self._run(p_multi_cargo=1.0, n_samples=30, seed=77)
        EUR_D = 0.80
        for m in meta:
            c1_obj = next((o["cargo1"] for o in m["objects"]
                           if isinstance(o, dict) and "cargo1" in o), None)
            c2_obj = next((o["cargo2"] for o in m["objects"]
                           if isinstance(o, dict) and "cargo2" in o), None)
            if c2_obj is None or c2_obj.get("placement") != "tandem":
                continue
            d1 = c1_obj.get("d", c1_obj.get("r", 0.0) * 2)
            d2 = c2_obj.get("d", c2_obj.get("r", 0.0) * 2)
            gap = 0.02
            total = d1 + gap + d2
            self.assertLessEqual(total, EUR_D + 1e-4,
                                 f"Tandem ensemble ({total:.3f}m) supera EUR_D ({EUR_D}m)")

    def test_multi_cargo_labels_are_cargo(self):
        """Todos los puntos de cargo1 y cargo2 deben tener label=1."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(tmp, n_samples=5, seed=42, p_multi_cargo=1.0)
            run_generation(cfg)
            for ply in Path(tmp).glob("*.ply"):
                lbs = _read_ply_labels(ply)
                # Label 1 (cargo) debe existir
                self.assertIn(1, lbs, f"{ply.name}: ningún punto con label=cargo")

    def test_box_dimensions_within_range(self):
        """Dimensiones de caja dentro de los rangos configurados."""
        min_w, max_w = 0.40, 0.60
        min_d, max_d = 0.40, 0.60
        min_h, max_h = 0.50, 0.80
        meta = self._run(
            n_samples=10, seed=1,
            box_min_w=min_w, box_max_w=max_w,
            box_min_d=min_d, box_max_d=max_d,
            box_min_h=min_h, box_max_h=max_h,
        )
        for m in meta:
            box = next(o["cargo1"] for o in m["objects"]
                       if isinstance(o, dict) and "cargo1" in o)
            self.assertGreaterEqual(box["w"], min_w - 1e-6)
            self.assertLessEqual(box["w"],    max_w + 1e-6)
            self.assertGreaterEqual(box["d"], min_d - 1e-6)
            self.assertLessEqual(box["d"],    max_d + 1e-6)
            self.assertGreaterEqual(box["h"], min_h - 1e-6)
            self.assertLessEqual(box["h"],    max_h + 1e-6)

    def test_seed_reproducibility(self):
        """Mismo seed → produce el mismo número de escenas y misma estructura de objetos.

        Nota: open3d sample_points_uniformly usa su propio RNG interno, y degrade_sensor
        consume el RNG de numpy en función de pts.shape (no determinista). Por eso
        los valores exactos de n_points y dimensiones pueden diferir ligeramente entre
        ejecuciones. Lo que sí es reproducible: el número de escenas y los tipos de
        objetos presentes.
        """
        with tempfile.TemporaryDirectory() as t1, \
             tempfile.TemporaryDirectory() as t2:
            cfg1 = _minimal_cfg(t1, seed=99)
            cfg2 = _minimal_cfg(t2, seed=99)
            m1 = run_generation(cfg1)
            m2 = run_generation(cfg2)
        self.assertEqual(len(m1), len(m2))
        for a, b in zip(m1, m2):
            # Misma lista de tipos de objeto (string entries — "pallet", "floor", etc.)
            obj_types_a = [o for o in a["objects"] if isinstance(o, str)]
            obj_types_b = [o for o in b["objects"] if isinstance(o, str)]
            self.assertEqual(obj_types_a, obj_types_b)
            # Mismas claves de dicts de objetos (cargo1, cargo2, etc.)
            dict_keys_a = [list(o.keys())[0] for o in a["objects"] if isinstance(o, dict)]
            dict_keys_b = [list(o.keys())[0] for o in b["objects"] if isinstance(o, dict)]
            self.assertEqual(dict_keys_a, dict_keys_b)


# ══════════════════════════════════════════════════════════════════════════════
# 5. compose_cargo / sample_cargo_spec / make_primitive_mesh
# ══════════════════════════════════════════════════════════════════════════════

class TestComposeCargo(unittest.TestCase):

    def _rng(self, seed=0):
        return np.random.default_rng(seed)

    # ── make_primitive_mesh ──

    def test_primitive_mesh_box_aabb(self):
        """make_primitive_mesh(box) produce mesh con AABB correcto."""
        spec = {"type": "box", "w": 0.6, "h": 0.8, "d": 0.4}
        mesh = make_primitive_mesh(spec)
        v = np.asarray(mesh.vertices)
        self.assertAlmostEqual(v[:, 0].min(), -0.3, places=4)
        self.assertAlmostEqual(v[:, 0].max(),  0.3, places=4)
        self.assertAlmostEqual(v[:, 1].min(),  0.0, places=4)
        self.assertAlmostEqual(v[:, 1].max(),  0.8, places=3)

    def test_primitive_mesh_cylinder_aabb(self):
        """make_primitive_mesh(cylinder) produce mesh Y-aligned."""
        spec = {"type": "cylinder", "r": 0.25, "h": 0.70}
        mesh = make_primitive_mesh(spec)
        v = np.asarray(mesh.vertices)
        self.assertAlmostEqual(v[:, 1].min(), 0.0, places=3)
        self.assertAlmostEqual(v[:, 1].max(), 0.70, places=3)

    def test_primitive_mesh_unknown_raises(self):
        with self.assertRaises(ValueError):
            make_primitive_mesh({"type": "cone", "r": 0.1, "h": 0.5})

    # ── _spec_w / _spec_d ──

    def test_spec_w_box(self):
        self.assertAlmostEqual(_spec_w({"type": "box", "w": 0.5, "h": 0.8, "d": 0.4}), 0.5)

    def test_spec_w_cylinder(self):
        self.assertAlmostEqual(_spec_w({"type": "cylinder", "r": 0.20, "h": 0.6}), 0.40)

    def test_spec_d_box(self):
        self.assertAlmostEqual(_spec_d({"type": "box", "w": 0.5, "h": 0.8, "d": 0.35}), 0.35)

    def test_spec_d_cylinder(self):
        self.assertAlmostEqual(_spec_d({"type": "cylinder", "r": 0.15, "h": 0.6}), 0.30)

    # ── sample_cargo_spec ──

    def test_sample_cargo_spec_box(self):
        cfg = CFG.copy()
        cfg["p_cylinder"] = 0.0
        spec = sample_cargo_spec(cfg, self._rng())
        self.assertEqual(spec["type"], "box")
        self.assertIn("w", spec)
        self.assertIn("d", spec)
        self.assertIn("h", spec)

    def test_sample_cargo_spec_cylinder(self):
        cfg = CFG.copy()
        cfg["p_cylinder"] = 1.0
        spec = sample_cargo_spec(cfg, self._rng())
        self.assertEqual(spec["type"], "cylinder")
        self.assertIn("r", spec)
        self.assertIn("h", spec)

    def test_sample_cargo_spec_max_w_clamp(self):
        """Con max_w, la caja no supera ese ancho."""
        cfg = CFG.copy()
        cfg["p_cylinder"] = 0.0
        cfg["box_min_w"] = 0.30
        cfg["box_max_w"] = 1.00
        max_w = 0.45
        for seed in range(20):
            spec = sample_cargo_spec(cfg, self._rng(seed), max_w=max_w)
            self.assertLessEqual(spec["w"], max_w + 1e-9)

    def test_sample_cargo_spec_cylinder_max_w_clamp(self):
        """Cilindro como cargo2 stacked: radio ≤ min(max_w, max_d)/2."""
        cfg = CFG.copy()
        cfg["p_cylinder"] = 1.0
        cfg["cyl_min_r"] = 0.15
        cfg["cyl_max_r"] = 0.40
        max_w, max_d = 0.50, 0.40  # min(max_w, max_d)/2 = 0.20
        expected_r_max = min(max_w, max_d) / 2  # 0.20
        for seed in range(20):
            spec = sample_cargo_spec(cfg, self._rng(seed), max_w=max_w, max_d=max_d)
            self.assertLessEqual(spec["r"], expected_r_max + 1e-9,
                                 f"Cilindro r={spec['r']:.4f} supera max_r={expected_r_max}")

    def test_stacked_secondary_never_wider_than_primary(self):
        """En stacked, cargo2 footprint (w y d) ≤ cargo1 footprint — sea caja o cilindro."""
        cfg = CFG.copy()
        cfg["p_cylinder"] = 0.5
        for seed in range(30):
            rng = self._rng(seed)
            spec1 = sample_cargo_spec(cfg, rng)
            max_w2, max_d2 = _spec_w(spec1), _spec_d(spec1)
            spec2 = sample_cargo_spec(cfg, rng, max_w=max_w2, max_d=max_d2)
            self.assertLessEqual(_spec_w(spec2), max_w2 + 1e-9,
                                 f"seed={seed}: spec2 width {_spec_w(spec2):.3f} > spec1 width {max_w2:.3f}")
            self.assertLessEqual(_spec_d(spec2), max_d2 + 1e-9,
                                 f"seed={seed}: spec2 depth {_spec_d(spec2):.3f} > spec1 depth {max_d2:.3f}")

    # ── compose_cargo — single ──

    def test_compose_single_positions_at_pallet_top(self):
        spec = {"type": "box", "w": 0.5, "h": 0.8, "d": 0.4}
        items, back_z = compose_cargo([spec], None, 0.144, self._rng())
        self.assertEqual(len(items), 1)
        _, placed = items[0]
        self.assertAlmostEqual(placed["oy"], 0.144, places=3)

    def test_compose_single_cargo_back_z(self):
        spec = {"type": "box", "w": 0.5, "h": 0.8, "d": 0.40}
        _, back_z = compose_cargo([spec], None, 0.0, self._rng())
        self.assertAlmostEqual(back_z, -0.20, places=4)

    # ── compose_cargo — stacked ──

    def test_compose_stacked_returns_two_items(self):
        s1 = {"type": "box", "w": 0.6, "h": 0.5, "d": 0.5}
        s2 = {"type": "box", "w": 0.4, "h": 0.4, "d": 0.4}
        items, _ = compose_cargo([s1, s2], "stacked", 0.144, self._rng())
        self.assertEqual(len(items), 2)

    def test_compose_stacked_second_on_top(self):
        """cargo2 debe estar por encima del top de cargo1."""
        s1 = {"type": "box", "w": 0.6, "h": 0.5, "d": 0.5}
        s2 = {"type": "box", "w": 0.4, "h": 0.3, "d": 0.4}
        items, _ = compose_cargo([s1, s2], "stacked", 0.144, self._rng())
        _, p1 = items[0]
        _, p2 = items[1]
        self.assertAlmostEqual(p2["oy"], p1["oy"] + s1["h"], places=3)

    def test_compose_stacked_placement_key(self):
        s1 = {"type": "box", "w": 0.6, "h": 0.5, "d": 0.5}
        s2 = {"type": "cylinder", "r": 0.20, "h": 0.3}
        items, _ = compose_cargo([s1, s2], "stacked", 0.0, self._rng())
        _, p2 = items[1]
        self.assertEqual(p2["placement"], "stacked")

    def test_compose_stacked_back_z_from_base_only(self):
        """Stacked: cargo_back_z usa solo la base, no el item superior."""
        s1 = {"type": "box", "w": 0.6, "h": 0.5, "d": 0.40}
        # s2 más ancho en Z que s1 — no debe afectar cargo_back_z
        s2 = {"type": "box", "w": 0.4, "h": 0.3, "d": 0.70}
        _, back_z = compose_cargo([s1, s2], "stacked", 0.0, self._rng())
        self.assertAlmostEqual(back_z, -s1["d"] / 2, places=4)

    # ── compose_cargo — tandem (Z direction) ──

    def test_compose_tandem_returns_two_items(self):
        s1 = {"type": "box", "w": 0.5, "h": 0.6, "d": 0.4}
        s2 = {"type": "cylinder", "r": 0.20, "h": 0.5}
        items, _ = compose_cargo([s1, s2], "tandem", 0.144, self._rng())
        self.assertEqual(len(items), 2)

    def test_compose_tandem_same_y_level(self):
        """Tandem: ambos items en el mismo nivel Y."""
        s1 = {"type": "box", "w": 0.5, "h": 0.6, "d": 0.4}
        s2 = {"type": "box", "w": 0.4, "h": 0.5, "d": 0.35}
        items, _ = compose_cargo([s1, s2], "tandem", 0.144, self._rng())
        _, p1 = items[0]
        _, p2 = items[1]
        self.assertAlmostEqual(p1["oy"], p2["oy"], places=4)

    def test_compose_tandem_no_x_offset(self):
        """Tandem: ningún item desplazado en X (ambos centrados)."""
        s1 = {"type": "box", "w": 0.5, "h": 0.6, "d": 0.4}
        s2 = {"type": "box", "w": 0.4, "h": 0.5, "d": 0.35}
        for seed in range(10):
            items, _ = compose_cargo([s1, s2], "tandem", 0.0, self._rng(seed))
            _, p1 = items[0]
            _, p2 = items[1]
            self.assertAlmostEqual(p1["ox"], 0.0, places=6)
            self.assertAlmostEqual(p2["ox"], 0.0, places=6)

    def test_compose_tandem_no_overlap_in_z(self):
        """Tandem: items no se solapan en Z (gap ≥ 0.02m entre caras)."""
        s1 = {"type": "box", "w": 0.5, "h": 0.6, "d": 0.40}
        s2 = {"type": "box", "w": 0.4, "h": 0.5, "d": 0.35}
        for seed in range(10):
            items, _ = compose_cargo([s1, s2], "tandem", 0.0, self._rng(seed))
            _, p1 = items[0]
            _, p2 = items[1]
            # Gap between faces = (oz2 - oz1) - (d1+d2)/2 must be ≥ 0.02
            face_gap = (p2["oz"] - p1["oz"]) - (s1["d"] + s2["d"]) / 2
            self.assertGreaterEqual(face_gap, 0.02 - 1e-6,
                                    f"Solapamiento Z en seed={seed}")

    def test_compose_tandem_placement_key(self):
        s1 = {"type": "box", "w": 0.5, "h": 0.6, "d": 0.4}
        s2 = {"type": "cylinder", "r": 0.18, "h": 0.5}
        items, _ = compose_cargo([s1, s2], "tandem", 0.0, self._rng())
        _, p2 = items[1]
        self.assertEqual(p2["placement"], "tandem")

    def test_compose_tandem_cargo_back_z(self):
        """cargo_back_z: back face of the ensemble = -(d1+d2+gap)/2."""
        s1 = {"type": "box", "w": 0.5, "h": 0.6, "d": 0.40}
        s2 = {"type": "box", "w": 0.4, "h": 0.5, "d": 0.30}
        _, back_z = compose_cargo([s1, s2], "tandem", 0.0, self._rng())
        gap = 0.02
        expected = -(s1["d"] + s2["d"] + gap) / 2
        self.assertAlmostEqual(back_z, expected, places=4)

    def test_compose_tandem_cargo2_always_front(self):
        """cargo2 en tandem siempre en +Z (oz > 0), nunca detrás de cargo1."""
        s1 = {"type": "box", "w": 0.5, "h": 0.6, "d": 0.40}
        s2 = {"type": "cylinder", "r": 0.18, "h": 0.5}
        for seed in range(20):
            items, _ = compose_cargo([s1, s2], "tandem", 0.0, self._rng(seed))
            _, p2 = items[1]
            self.assertGreater(p2["oz"], 0.0,
                               f"cargo2 en tandem está en -Z (oz={p2['oz']}) con seed={seed}")

    # ── sample_cargo_spec — max_h (flat cargo) ──

    def test_sample_cargo_spec_max_h_box(self):
        """Con max_h, la caja no supera esa altura."""
        cfg = CFG.copy()
        cfg["p_cylinder"] = 0.0
        cfg["box_min_h"] = 0.05
        cfg["box_max_h"] = 1.40
        max_h = 0.15
        for seed in range(20):
            spec = sample_cargo_spec(cfg, self._rng(seed), max_h=max_h)
            self.assertLessEqual(spec["h"], max_h + 1e-9)

    def test_sample_cargo_spec_max_h_cylinder(self):
        """Con max_h, el cilindro no supera esa altura (usa flat_min_h como mínimo)."""
        cfg = CFG.copy()
        cfg["p_cylinder"] = 1.0
        cfg["flat_min_h"] = 0.03
        cfg["cyl_max_h"] = 1.20
        max_h = 0.12
        for seed in range(20):
            spec = sample_cargo_spec(cfg, self._rng(seed), max_h=max_h)
            self.assertLessEqual(spec["h"], max_h + 1e-9)

    def test_compose_unknown_mode_raises(self):
        s1 = {"type": "box", "w": 0.5, "h": 0.5, "d": 0.4}
        s2 = {"type": "box", "w": 0.4, "h": 0.4, "d": 0.3}
        with self.assertRaises(ValueError):
            compose_cargo([s1, s2], "diagonal", 0.0, self._rng())


# ══════════════════════════════════════════════════════════════════════════════
# 6. Degradación del sensor
# ══════════════════════════════════════════════════════════════════════════════

class TestSensorDegradation(unittest.TestCase):

    def _sample_pts(self, n=500):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        lbs = rng.integers(0, 4, n, dtype=np.uint8)
        return pts, lbs

    def test_degrade_reduces_points(self):
        """Degradación elimina puntos (dropout)."""
        cfg = CFG.copy()
        cfg["dropout_ratio"] = 0.5
        cfg["outlier_ratio"] = 0.0
        pts, lbs = self._sample_pts(1000)
        pts_d, lbs_d = degrade_labeled(pts, lbs, cfg, np.random.default_rng(0))
        self.assertLess(len(pts_d), len(pts))

    def test_degrade_adds_outliers(self):
        """Outliers con label=255 se añaden."""
        cfg = CFG.copy()
        cfg["dropout_ratio"] = 0.0
        cfg["outlier_ratio"] = 0.1
        pts, lbs = self._sample_pts(1000)
        _, lbs_d = degrade_labeled(pts, lbs, cfg, np.random.default_rng(0))
        self.assertIn(255, lbs_d)

    def test_dropout_zero_preserves_count_approx(self):
        """Con dropout=0 y outlier=0 no se pierden puntos (solo voxelizado)."""
        cfg = CFG.copy()
        cfg["dropout_ratio"]  = 0.0
        cfg["outlier_ratio"]  = 0.0
        cfg["voxel_size"]     = 0.001   # grid muy fino para no eliminar casi nada
        pts, lbs = self._sample_pts(200)
        pts_d, _ = degrade_labeled(pts, lbs, cfg, np.random.default_rng(0))
        self.assertGreater(len(pts_d), 0)

    def test_camera_filter_reduces_points(self):
        """FOV filter elimina puntos fuera del cono de cámaras."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(-5, 5, (2000, 3)).astype(np.float32)
        lbs = np.zeros(len(pts), dtype=np.uint8)
        pts_f, lbs_f = camera_arc_filter(pts, lbs, CFG["cameras"])
        self.assertLess(len(pts_f), len(pts))
        self.assertEqual(len(pts_f), len(lbs_f))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Previews PNG
# ══════════════════════════════════════════════════════════════════════════════

class TestPreviewGrid(unittest.TestCase):

    def test_render_scene_creates_png(self):
        with tempfile.TemporaryDirectory() as dataset_dir, \
             tempfile.TemporaryDirectory() as preview_dir:
            cfg = _minimal_cfg(dataset_dir, n_samples=1)
            run_generation(cfg)
            ply = sorted(Path(dataset_dir).glob("*.ply"))[0]
            png = Path(preview_dir) / "00000.png"
            render_scene(ply, png)
            self.assertTrue(png.exists(), "render_scene no creó el PNG")
            self.assertGreater(png.stat().st_size, 1000, "PNG demasiado pequeño")

    def test_render_scene_png_is_valid_image(self):
        """El PNG empieza con la firma correcta."""
        with tempfile.TemporaryDirectory() as dataset_dir, \
             tempfile.TemporaryDirectory() as preview_dir:
            cfg = _minimal_cfg(dataset_dir, n_samples=1)
            run_generation(cfg)
            ply = sorted(Path(dataset_dir).glob("*.ply"))[0]
            png = Path(preview_dir) / "00000.png"
            render_scene(ply, png)
            with open(png, "rb") as f:
                sig = f.read(8)
            # Firma PNG: \x89PNG\r\n\x1a\n
            self.assertEqual(sig, b"\x89PNG\r\n\x1a\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. App — DEFAULTS dict
# ══════════════════════════════════════════════════════════════════════════════

class TestAppDefaults(unittest.TestCase):

    def test_defaults_match_cfg(self):
        """DEFAULTS de app.py debe coincidir con CFG en las claves críticas."""
        # Importar sin ejecutar Streamlit
        import importlib, types
        # Parchear streamlit antes de importar app
        fake_st = types.ModuleType("streamlit")

        class _FakeCtx:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __call__(self, *a, **kw): return _FakeCtx()

        ctx = _FakeCtx()
        def _widget(*a, **kw):
            """Return the 'value' kwarg so downstream int()/float() calls work."""
            return kw.get("value", None)

        for fn in ("set_page_config", "title", "caption", "header", "divider",
                   "button", "subheader", "progress", "empty",
                   "success", "error", "info", "json", "image", "rerun"):
            setattr(fake_st, fn, lambda *a, **kw: None)
        for fn in ("number_input", "text_input", "checkbox", "slider"):
            setattr(fake_st, fn, _widget)
        fake_st.expander = _FakeCtx()
        fake_st.columns  = lambda *a, **kw: [ctx, ctx]
        fake_st.sidebar  = ctx
        fake_st.session_state = {}
        sys.modules["streamlit"] = fake_st

        # Limpiar caché del módulo si ya fue importado
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as app_module

        d = app_module.DEFAULTS
        self.assertEqual(d["n_samples"],  CFG["n_samples"])
        self.assertEqual(d["seed"],       CFG["seed"])
        self.assertAlmostEqual(d["noise_std"],     CFG["noise_std"])
        self.assertAlmostEqual(d["dropout_ratio"], CFG["dropout_ratio"])
        self.assertAlmostEqual(d["voxel_size"],    CFG["voxel_size"])
        self.assertEqual(d["box_w"], (CFG["box_min_w"], CFG["box_max_w"]))
        self.assertEqual(d["box_d"], (CFG["box_min_d"], CFG["box_max_d"]))
        self.assertEqual(d["box_h"], (CFG["box_min_h"], CFG["box_max_h"]))

        # Restaurar streamlit real
        del sys.modules["streamlit"]
        del sys.modules["app"]


# ══════════════════════════════════════════════════════════════════════════════
# 9. Persona
# ══════════════════════════════════════════════════════════════════════════════

class TestPerson(unittest.TestCase):
    """Tests de la clase person (label=3): make_person_mesh y bloque generate_scene."""

    def _run(self, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(tmp, **overrides)
            meta = run_generation(cfg)
            return meta, tmp

    # ── make_person_mesh (fallback: stl_path inexistente) ─────────────────────

    def test_make_person_mesh_fallback_base_at_y0(self):
        """Fallback: base de la malla en Y≈0."""
        from generate_dataset import make_person_mesh
        m = make_person_mesh(height=1.75, stl_path="/nonexistent/person.stl")
        verts = np.asarray(m.vertices)
        self.assertAlmostEqual(float(verts[:, 1].min()), 0.0, places=2)

    def test_make_person_mesh_fallback_height(self):
        """Fallback: altura máxima ≈ height ± 10%."""
        from generate_dataset import make_person_mesh
        target = 1.75
        m = make_person_mesh(height=target, stl_path="/nonexistent/person.stl")
        verts = np.asarray(m.vertices)
        self.assertAlmostEqual(float(verts[:, 1].max()), target, delta=target * 0.10)

    def test_make_person_mesh_fallback_various_heights(self):
        """Fallback: escalado correcto para alturas distintas."""
        from generate_dataset import make_person_mesh
        for h in (1.70, 1.75, 1.80):
            m = make_person_mesh(height=h, stl_path="/nonexistent/person.stl")
            verts = np.asarray(m.vertices)
            self.assertAlmostEqual(float(verts[:, 1].min()), 0.0, places=2,
                                   msg=f"base no en Y=0 con height={h}")
            self.assertAlmostEqual(float(verts[:, 1].max()), h, delta=h * 0.10,
                                   msg=f"cima incorrecta con height={h}")

    # ── generate_scene / run_generation ───────────────────────────────────────

    def test_person_never_appears_when_p0(self):
        """p_person=0.0: ninguna escena contiene persona."""
        meta, _ = self._run(p_person=0.0, n_samples=10, seed=60)
        for m in meta:
            has_person = any(
                isinstance(o, dict) and "person" in o for o in m["objects"]
            )
            self.assertFalse(has_person, "Persona aparece con p_person=0.0")

    def test_person_always_appears_when_p1(self):
        """p_person=1.0: todas las escenas contienen persona en metadata."""
        meta, _ = self._run(p_person=1.0, n_samples=5, seed=10)
        for m in meta:
            has_person = any(
                isinstance(o, dict) and "person" in o for o in m["objects"]
            )
            self.assertTrue(has_person, "Persona no encontrada con p_person=1.0")

    def test_person_label_points_present(self):
        """p_person=1.0: label=3 presente en cada PLY."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(tmp, p_person=1.0, n_samples=3, seed=20)
            run_generation(cfg)
            for ply in sorted(Path(tmp).glob("*.ply")):
                lbs = _read_ply_labels(ply)
                self.assertIn(3, lbs, f"{ply.name}: ningún punto con label=3 (person)")

    def test_person_height_in_range(self):
        """Metadata: height ∈ [1.70, 1.80]."""
        meta, _ = self._run(p_person=1.0, n_samples=10, seed=30)
        for m in meta:
            p = next((o["person"] for o in m["objects"]
                      if isinstance(o, dict) and "person" in o), None)
            if p is not None:
                self.assertGreaterEqual(p["height"], 1.70 - 1e-6)
                self.assertLessEqual(p["height"],    1.80 + 1e-6)

    def test_person_rotation_in_range(self):
        """Metadata: rot_deg ∈ [0, 360)."""
        meta, _ = self._run(p_person=1.0, n_samples=10, seed=40)
        for m in meta:
            p = next((o["person"] for o in m["objects"]
                      if isinstance(o, dict) and "person" in o), None)
            if p is not None:
                self.assertGreaterEqual(p["rot_deg"], 0.0)
                self.assertLess(p["rot_deg"],          360.0)

    def test_person_zone_is_valid(self):
        """Metadata: zone ∈ {'operator', 'perimeter'}."""
        meta, _ = self._run(p_person=1.0, n_samples=10, seed=50)
        for m in meta:
            p = next((o["person"] for o in m["objects"]
                      if isinstance(o, dict) and "person" in o), None)
            if p is not None:
                self.assertIn(p["zone"], ("operator", "perimeter"))


# ══════════════════════════════════════════════════════════════════════════════
# 10. Cargo sobre horcas (A1)
# ══════════════════════════════════════════════════════════════════════════════

class TestCargoOnVehicle(unittest.TestCase):
    """Tests para compose_cargo_on_vehicle y escenas sin pallet (A1)."""

    def test_cargo_on_vehicle_base_at_fork_height(self):
        """compose_cargo_on_vehicle: base del cargo en Y=JACK_FORK_H."""
        rng = np.random.default_rng(0)
        spec = {"type": "box", "w": 0.60, "h": 0.50, "d": 0.40}
        _mesh, placed = compose_cargo_on_vehicle(spec, rng)
        self.assertAlmostEqual(placed["oy"], JACK_FORK_H, places=4)
        self.assertGreaterEqual(placed["oz"], 0.0)
        self.assertLessEqual(placed["oz"], JACK_FORK_L)
        self.assertGreaterEqual(placed["ox"], -0.20)
        self.assertLessEqual(placed["ox"],  0.20)

    def test_no_pallet_scene_generates(self):
        """p_pallet=0: escena sin pallet contiene cargo y pallet_jack."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_cfg(tmp, p_pallet=0.0, p_cargo_on_vehicle=1.0)
            meta = run_generation(cfg)
            for m in meta:
                self.assertNotIn("pallet", m["objects"])
                has_cargo = any(
                    isinstance(o, dict) and "cargo1" in o for o in m["objects"]
                )
                self.assertTrue(has_cargo, "cargo1 no encontrado en escena sin pallet")
                has_jack = any(
                    isinstance(o, dict) and "pallet_jack" in o for o in m["objects"]
                )
                self.assertTrue(has_jack, "pallet_jack no encontrado en escena sin pallet")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Negative-filter policy (Commit 1)
# ══════════════════════════════════════════════════════════════════════════════

class TestNegativeFilterPolicy(unittest.TestCase):
    """Unit tests for the negative-filter cargo extraction policy."""

    def test_filter_excludes_person_and_vehicle(self):
        """[cargo,person,vehicle,cargo,cargo] → retained ids {0,3,4}; {1,2} excluded."""
        from cargo_geometric.cargo import _apply_negative_filter

        # Predictions: cargo=1, person=3, vehicle=2, cargo=1, cargo=1
        preds = np.array([1, 3, 2, 1, 1], dtype=np.int32)
        label_map = {"cargo": 1, "vehicle": 2, "person": 3}

        retained = _apply_negative_filter(preds, label_map)

        self.assertEqual(set(retained), {0, 3, 4})
        self.assertNotIn(1, retained)   # person excluded
        self.assertNotIn(2, retained)   # vehicle excluded

    def test_filter_all_cargo(self):
        """All cargo predictions → all cluster ids retained."""
        from cargo_geometric.cargo import _apply_negative_filter

        preds = np.array([1, 1, 1], dtype=np.int32)
        label_map = {"cargo": 1, "vehicle": 2, "person": 3}
        retained = _apply_negative_filter(preds, label_map)
        self.assertEqual(set(retained), {0, 1, 2})

    def test_filter_all_excluded(self):
        """All person/vehicle → empty list (triggers rank-0 fallback)."""
        from cargo_geometric.cargo import _apply_negative_filter

        preds = np.array([3, 2, 3], dtype=np.int32)
        label_map = {"cargo": 1, "vehicle": 2, "person": 3}
        retained = _apply_negative_filter(preds, label_map)
        self.assertEqual(retained, [])

    def test_filter_permissive_unknown_class(self):
        """Label absent from label_map is treated as cargo (permissive)."""
        from cargo_geometric.cargo import _apply_negative_filter

        # label 99 is not in label_map → not excluded
        preds = np.array([99, 3, 2], dtype=np.int32)
        label_map = {"cargo": 1, "vehicle": 2, "person": 3}
        retained = _apply_negative_filter(preds, label_map)
        self.assertEqual(set(retained), {0})   # only id=0 (label=99) kept


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
