# datageneration

Synthetic labeled point cloud dataset generator for stereo-camera cargo inspection.

Simulates the **FUSION3D** sensor system (CATEC/BBB): three cameras (cenital + der + izq) capturing pallets, cargo boxes, forklifts, and people in a warehouse floor environment.

## Structure

```
datageneration/
├── src/
│   ├── generate_dataset.py   ← main generator
│   ├── preview_grid.py       ← visual overview (PNG grid)
│   ├── view_png.py           ← 3-view PNG for a single scene
│   ├── analyze.py            ← statistical comparison vs real data
│   ├── basic.py              ← original prototype (visualization only)
│   └── multiple_elements.py  ← original prototype (visualization only)
├── data/
│   └── forklift.stl          ← forklift mesh
├── output/
│   ├── dataset/              ← generated PLY files + metadata.json
│   ├── previews/             ← PNG overviews
│   └── analysis/             ← comparison figures vs real data
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset

Run from `src/`:

```bash
cd src/

# Default: 500 scenes → output/dataset/
python3 generate_dataset.py

# Custom number and output folder
python3 generate_dataset.py --n 200 --out ../output/my_run

# Different seed (reproducibility)
python3 generate_dataset.py --seed 99 --n 100

# Adjust sensor noise (default calibrated to FUSION3D σ≈10mm)
python3 generate_dataset.py --noise 0.015 --dropout 0.30 --voxel 0.020

# Scene composition
python3 generate_dataset.py --p-forklift 0.8 --p-person 0.4 --p-two-boxes 0.5

# Only boxes on pallet, no vehicles or people
python3 generate_dataset.py --p-forklift 0 --p-person 0 --p-pallet 1.0

# Small boxes only
python3 generate_dataset.py --box-min 0.2 --box-max 0.6
```

### All options

| Argument | Default | Description |
|---|---|---|
| `--n` | 100 | Number of scenes |
| `--seed` | 42 | Random seed |
| `--out` | `../output/dataset` | Output directory |
| `--noise` | 0.010 | Gaussian noise σ (metres) |
| `--dropout` | 0.20 | Fraction of points randomly removed |
| `--voxel` | 0.015 | Voxel grid size (metres), limits point density |
| `--outliers` | 0.03 | Fraction turned into local outlier clusters |
| `--p-forklift` | 0.50 | Probability of forklift in scene |
| `--p-person` | 0.25 | Probability of person in scene |
| `--p-pallet` | 0.70 | Probability of EUR pallet base |
| `--p-two-boxes` | 0.40 | Probability of second cargo box |
| `--box-min` | 0.25 | Minimum cargo box side (metres) |
| `--box-max` | 1.40 | Maximum cargo box side (metres) |

## Output format

Each scene produces one binary PLY file with four properties per point:

```
property float x
property float y
property float z
property uchar label
```

| Label | Class |
|---|---|
| 0 | floor |
| 1 | cargo (boxes) |
| 2 | vehicle (forklift / pallet jack) |
| 3 | person |
| 4 | pallet base (EUR 1.2 × 0.8 m) |
| 255 | outlier / artefact |

`metadata.json` contains per-scene object list, point count, and label counts.

## Visualize

### PNG grid (quick overview of the full dataset)

```bash
cd src/

# 50 scenes, top-down view only
python3 preview_grid.py --n 50

# 30 scenes with 3 views each (top + front + side)
python3 preview_grid.py --n 30 --views 3
```

Output: `output/previews/dataset_grid_n<N>_v<V>.png`

### 3-view PNG for a single scene

```bash
cd src/
python3 view_png.py 42            # scene 00042, opens window
python3 view_png.py 42 --save     # save PNG without opening window
```

### CloudCompare (interactive 3D)

1. Open CloudCompare
2. `File → Open → output/dataset/00042.ply`
3. In the DB tree, select the cloud
4. Properties panel → `SF display params` → select scalar field `label`
5. Apply a colour ramp to distinguish classes

## Statistical analysis vs real data

Compares the generated dataset against FUSION3D merged captures
(requires `../Resources/Capturas_BBB/` to be present):

```bash
cd src/
python3 analyze.py
```

Saves 6 figures to `output/analysis/`:
- `fig1_scene_stats.png` — point count and bounding box distributions
- `fig2_height_dist.png` — height profiles (floor-relative)
- `fig3_density_xz.png` — top-down point density heatmaps
- `fig4_roughness.png`  — local surface roughness σ (proxy for sensor noise)
- `fig5_nn_spacing.png` — nearest-neighbour point spacing
- `fig6_label_dist.png` — synthetic label distribution

## Coordinate system

| Axis | Direction |
|---|---|
| X | right |
| Y | up (height, floor at Y = 0) |
| Z | depth (toward cameras) |

Units: **metres**

## Scene geometry

```
Cameras (cenital + der + izq)
        ↓        ↘        ↙
        ┌──────────────────┐  ← floor Y=0
        │   [pallet]       │
        │   [cargo box]    │
        │     ← [forklift at Z≈−2.8 m]
        └──────────────────┘
```

Sensor noise calibrated to real FUSION3D measurements:
- Surface roughness σ flat ≈ 6.5 mm, overall ≈ 15 mm
- Generator default: σ = 10 mm, voxel = 15 mm
