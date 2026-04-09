# datageneration

Synthetic labeled point cloud dataset generator for stereo-camera cargo inspection.

Simulates the **FUSION3D** sensor system (CATEC/BBB): three cameras (cenital + der + izq) capturing pallets, cargo boxes, pallet jacks, and people in a warehouse environment. Noise parameters calibrated against real FUSION3D captures.

## Structure

```
datageneration/
├── src/
│   ├── generate_dataset.py   ← main generator (CLI + importable)
│   ├── app.py                ← Streamlit UI
│   ├── preview_grid.py       ← generate one PNG per scene (3 views)
│   ├── analyze.py            ← statistical comparison vs real data
│   └── test_pipeline.py      ← pytest test suite (37 tests)
├── data/
│   └── forklift.stl          ← forklift mesh (optional)
├── output/
│   ├── dataset/              ← generated PLY files + metadata.json
│   ├── previews/             ← PNG previews
│   └── analysis/             ← comparison figures vs real data
├── DECISIONS.md              ← design decisions log
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset

All commands run from `src/`:

```bash
cd src/

# Default: 100 scenes → output/dataset/
python3 generate_dataset.py

# Custom number, seed and output folder
python3 generate_dataset.py --n 500 --seed 99 --out ../output/my_run

# With cylinders as cargo (bobinas, bidones) — 50% probability
python3 generate_dataset.py --n 100 --p-cylinder 0.5

# Only cylinders
python3 generate_dataset.py --n 100 --p-cylinder 1.0

# Adjust sensor noise (defaults calibrated to real FUSION3D)
python3 generate_dataset.py --noise 0.030 --dropout 0.15 --voxel 0.019
```

### All options

| Argument | Default | Description |
|---|---|---|
| `--n` | 100 | Number of scenes |
| `--seed` | 42 | Random seed |
| `--out` | `../output/dataset` | Output directory |
| `--noise` | 0.030 | Gaussian noise σ (metres) — calibrated to FUSION3D |
| `--dropout` | 0.15 | Fraction of points randomly removed |
| `--voxel` | 0.019 | Voxel grid size (metres) — calibrated to real NN spacing |
| `--outliers` | 0.03 | Fraction turned into local outlier clusters |
| `--p-pallet` | 1.0 | Probability of EUR pallet base |
| `--p-cylinder` | 0.0 | Probability of cylinder instead of box as primary cargo |
| `--p-two-boxes` | 0.0 | Probability of second cargo box (experimental) |
| `--p-person` | 0.0 | Probability of person in scene |
| `--p-forklift` | 0.0 | Probability of forklift STL (requires `data/forklift.stl`) |
| `--box-min-w` | 0.30 | Min cargo box X width (m) |
| `--box-max-w` | 1.00 | Max cargo box X width (m) |
| `--box-min-d` | 0.30 | Min cargo box Z depth (m) |
| `--box-max-d` | 0.75 | Max cargo box Z depth (m) |
| `--box-min-h` | 0.25 | Min cargo box Y height (m) |
| `--box-max-h` | 1.40 | Max cargo box Y height (m) |
| `--cyl-min-r` | 0.15 | Min cylinder radius (m) |
| `--cyl-max-r` | 0.40 | Max cylinder radius (m) |
| `--cyl-min-h` | 0.30 | Min cylinder height (m) |
| `--cyl-max-h` | 1.20 | Max cylinder height (m) |
| `--floor-ext-x` | 2.5 | Floor half-extent along X (m) |
| `--floor-ext-z` | 2.0 | Floor half-extent along Z (m) |

## Streamlit UI

```bash
cd src/
streamlit run app.py
```

Controls all parameters from a web interface, generates previews inline, and shows metadata.

## Output format

Each scene produces one binary PLY file (`output/dataset/<id:05d>.ply`):

```
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar label
```

RGB is baked from the label — open directly in CloudCompare and colours appear automatically.

| Label | Class | Colour |
|---|---|---|
| 0 | floor | dark grey |
| 1 | cargo (box or cylinder) | red |
| 2 | vehicle (pallet jack / forklift) | blue |
| 3 | person | green |
| 4 | pallet base (EUR 1.2 × 0.8 m) | yellow |
| 255 | outlier / artefact | dark grey |

`metadata.json` contains per-scene object list, point count, and label counts.

## Previews

```bash
cd src/

# Generate one PNG per scene (3 views: top / front / side)
python3 preview_grid.py

# Custom dataset and output folder
python3 preview_grid.py --dataset ../output/my_run --out ../output/my_previews
```

Output: `output/previews/<id:05d>.png`

## Statistical analysis vs real data

Compares synthetic output against real FUSION3D captures.
Requires `../../Resources/Capturas_BBB/2026_03_23/` to be present.

```bash
cd src/
python3 analyze.py
```

Saves 7 figures to `output/analysis/`:

| Figure | Content |
|---|---|
| fig1 | Scene stats (point count, bounding box) |
| fig2 | Height distributions |
| fig3 | Top-down point density heatmaps |
| fig4 | Surface roughness σ |
| fig5 | Nearest-neighbour point spacing |
| fig6 | Label distribution |
| fig7 | Direct overlay: synthetic vs real |

## Tests

```bash
cd src/
python3 -m pytest test_pipeline.py -v
```

37 tests covering geometry, PLY output, scene composition, sensor degradation, previews, and UI defaults.

## Coordinate system

| Axis | Direction |
|---|---|
| X | right |
| Y | up (height, floor at Y = 0) |
| Z | depth (toward cameras) |

Units: **metres**

> Note: real FUSION3D clouds use Z as the height axis. `analyze.py` remaps automatically.

## Sensor noise (calibrated to real FUSION3D)

| Parameter | Value | Calibration result |
|---|---|---|
| Gaussian noise σ | 30 mm | Floor roughness: synth 24.8 mm vs real 29.8 mm ✅ |
| Voxel grid | 19 mm | NN spacing: synth 47.5 mm vs real 51.3 mm ✅ |
| Dropout | 15% | — |
| Outlier ratio | 3% | — |
| Floor extent X | ±2.5 m | Footprint: synth 5.29 m vs real 5.00 m ✅ |
| Floor extent Z | ±2.0 m | Footprint: synth 4.23 m vs real 3.97 m ✅ |
