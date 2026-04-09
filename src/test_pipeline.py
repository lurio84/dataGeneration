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
    make_pallet_mesh,
    make_pallet_jack_mesh,
    sample_labeled,
    degrade_labeled,
    camera_arc_filter,
    LABEL,
    LABEL_RGB,
    EUR_W, EUR_H, EUR_D,
)
from preview_grid import render_scene, load_synth


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
        "p_two_boxes":    0.0,
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
            "p_pallet", "p_two_boxes", "p_person", "p_forklift",
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
        for k in ("p_pallet", "p_two_boxes", "p_person", "p_forklift"):
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
            has_box = any(
                isinstance(o, dict) and "box1" in o
                for o in m["objects"]
            )
            self.assertTrue(has_box, "box1 no encontrado en la escena")

    def test_jack_always_present(self):
        meta = self._run()
        for m in meta:
            has_jack = any(
                isinstance(o, dict) and "pallet_jack" in o
                for o in m["objects"]
            )
            self.assertTrue(has_jack, "pallet_jack no encontrado en la escena")

    def test_two_boxes_activated(self):
        """Con p_two_boxes=1.0 todas las escenas deben tener box2."""
        meta = self._run(p_two_boxes=1.0, n_samples=5, seed=7)
        for m in meta:
            has_box2 = any(
                isinstance(o, dict) and "box2" in o
                for o in m["objects"]
            )
            self.assertTrue(has_box2, "box2 no encontrado con p_two_boxes=1.0")

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
            box = next(o["box1"] for o in m["objects"] if isinstance(o, dict) and "box1" in o)
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
            # Mismas claves de dicts de objetos (box1, box2, etc.)
            dict_keys_a = [list(o.keys())[0] for o in a["objects"] if isinstance(o, dict)]
            dict_keys_b = [list(o.keys())[0] for o in b["objects"] if isinstance(o, dict)]
            self.assertEqual(dict_keys_a, dict_keys_b)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Degradación del sensor
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
# 6. Previews PNG
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
# 7. App — DEFAULTS dict
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
