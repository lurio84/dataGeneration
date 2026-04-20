"""
gen_presentation.py — Generate clean forklift PLY for team presentation.

No noise, no dropout, no outliers.  High density, p_forklift=1.0.
Run from src/:  python3 gen_presentation.py
Output: ../output/presentation/
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import CFG, run_generation

cfg = copy.deepcopy(CFG)

# ── Presentation overrides ──────────────────────────────────────────────
cfg["n_samples"]      = 6
cfg["seed"]           = 7
cfg["output_dir"]     = "../output/presentation"

# No sensor degradation
cfg["noise_core_ref"] = 1e-6
cfg["noise_tail_ref"] = 1e-6
cfg["dropout_ratio"]  = 0.0
cfg["outlier_ratio"]  = 0.0
cfg["voxel_size"]     = 0.006   # 6mm → very fine, high-density

# Scene: forklift always, pallet always, single cargo, ~half with person
cfg["p_forklift"]     = 1.0
cfg["p_pallet"]       = 1.0
cfg["p_person"]       = 0.5    # some scenes include person (verifies head fix)
cfg["p_multi_cargo"]  = 0.0
cfg["p_flat_cargo"]   = 0.0
cfg["skip_camera_filter"] = True   # presentación: mostrar geometría completa del STL

# High sampling before voxelisation
cfg["pts_floor"]      = 2_000_000
cfg["pts_pallet"]     =   100_000
cfg["pts_box"]        =   300_000
cfg["pts_forklift"]   =   500_000

# ── Run ──────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

print("Generating 3 clean forklift scenes → output/presentation/ …")
meta = run_generation(cfg)

total_pts = sum(m["n_points"] for m in meta)
print(f"Done. {len(meta)} scenes, {total_pts:,} total points.")
for m in meta:
    veh = next((o['vehicle']['type'] for o in m['objects'] if isinstance(o, dict) and 'vehicle' in o), '?')
    print(f"  scene {m['id']:05d}: {m['n_points']:,} pts  vehicle={veh}")
