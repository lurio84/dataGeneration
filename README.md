# datageneration

Synthetic labeled point cloud dataset generator for stereo-camera cargo inspection.

Simulates the **FUSION3D** sensor system (CATEC/BBB): three cameras (cenital + der + izq) capturing pallets, cargo boxes, cylinders, pallet jacks, and people in a warehouse environment. Noise parameters calibrated against real FUSION3D captures.

## Requirements

- Python **3.10+** (uses `X | Y` union type syntax)
- open3d 0.19.0, numpy 2.2.6, streamlit ≥ 1.35

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
datageneration/
├── src/
│   ├── generate_dataset.py        ← main generator (CLI + importable)
│   ├── geometry/
│   │   ├── meshes.py              ← mesh builders: pallet, cargo, jack, carretilla, person
│   │   └── composition.py        ← cargo placement logic
│   ├── sensor/noise.py            ← sampling, sensor degradation, camera filter
│   ├── ply_io/ply.py              ← PLY export (label→RGB, save_ply)
│   ├── classifier/                ← 5-class ML classifier (train/predict/evaluate)
│   ├── app.py                     ← Streamlit UI
│   ├── utils/preview_grid.py      ← PNG previews (3 views)
│   ├── analyze.py                 ← comparison vs real FUSION3D captures
│   └── test_pipeline.py           ← pytest test suite (101 tests)
├── data/
│   ├── carretilla.stl             ← forklift STL used by --p-forklift
│   └── person.stl                 ← person scan (optional; fallback: cylinder+sphere)
├── yolov8n.pt                     ← YOLOv8 weights for filter_person_yolo.py (not tracked in git)
├── docs/
│   └── ADDING_ASSETS.md           ← step-by-step guide for adding new STL assets
├── output/
│   ├── dataset/                   ← generated PLY files + metadata.json
│   ├── previews/                  ← PNG previews
│   └── analysis/                  ← comparison figures vs real data
├── scripts/
│   ├── *.py                       ← pipeline and analysis scripts (see ## Scripts below)
│   └── archive/                   ← retired debug scripts (not part of active pipeline)
├── DECISIONS.md                   ← design decisions log
└── requirements.txt
```

## Vehicles available

Each scene contains exactly one vehicle (mutually exclusive):

| Vehicle | Type key in metadata | How selected |
|---------|---------------------|--------------|
| Traspaleta (pallet jack) | `pallet_jack` | default; when `--p-forklift` not hit |
| Carretilla elevadora (forklift) | `forklift` | when `rng < --p-forklift` and `--forklift-stl` exists |

To add more vehicle types, see [docs/ADDING_ASSETS.md](docs/ADDING_ASSETS.md).

## Generate dataset

All commands run from `src/`:

```bash
cd src/

# Default: 100 scenes → output/dataset/
python3 generate_dataset.py

# Custom number, seed and output folder
python3 generate_dataset.py --n 500 --seed 99 --out ../output/my_run

# Mixed cargo: boxes + cylinders (bobinas, bidones)
python3 generate_dataset.py --n 100 --p-cylinder 0.5

# Multi-cargo: secondary item stacked or in tandem — 70% of scenes
python3 generate_dataset.py --n 100 --p-multi-cargo 0.7

# Flat/low cargo (hard near-floor cases) — 30% of scenes
python3 generate_dataset.py --n 100 --p-flat-cargo 0.3

# Adjust sensor noise (defaults calibrated to real FUSION3D)
python3 generate_dataset.py --noise-core-ref 0.010 --noise-tail-ref 0.025 --dropout 0.15 --voxel 0.019

# Carretilla elevadora in every scene (requires data/carretilla.stl)
python3 generate_dataset.py --n 100 --p-forklift 1.0

# Mix: 50% pallet jack, 50% carretilla
python3 generate_dataset.py --n 200 --p-forklift 0.5
```

### All options

| Argument | Default | Description |
|---|---|---|
| `--n` | 100 | Number of scenes |
| `--seed` | 42 | Random seed |
| `--out` | `../output/dataset` | Output directory |
| `--noise-core-ref` | 0.010 | σ_core @ z_ref=3m for Gaussian mixture component (m) |
| `--noise-tail-ref` | 0.025 | σ_tail @ z_ref=3m for t-Student component (m) |
| `--dropout` | 0.15 | Fraction of points randomly removed |
| `--voxel` | 0.019 | Voxel grid size (metres) |
| `--outliers` | 0.03 | Fraction turned into local outlier clusters |
| `--p-pallet` | 0.80 | Probability of EUR pallet base (1.2 × 0.8 m) |
| `--p-cylinder` | 0.15 | Probability of cylinder instead of box as primary cargo |
| `--p-multi-cargo` | 0.25 | Probability of a second cargo item (stacked or tandem) |
| `--p-flat-cargo` | 0.10 | Probability of very low/flat cargo (near-floor hard case) |
| `--flat-min-h` | 0.03 | Min height in flat-cargo mode (m) |
| `--flat-max-h` | 0.15 | Max height in flat-cargo mode (m) |
| `--p-person` | 0.30 | Probability of person in scene |
| `--p-forklift` | 0.00 | Probability of carretilla elevadora instead of pallet jack |
| `--forklift-stl` | `../data/carretilla.stl` | Path to forklift STL (relative to `src/`) |
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

## Multi-cargo modes

When `--p-multi-cargo > 0`, a second cargo item is added using one of two modes chosen randomly:

- **stacked** — cargo2 placed on top of cargo1. cargo2 footprint always ≤ cargo1 footprint (stability). Any primitive type combination allowed (box+box, box+cylinder, cylinder+cylinder).
- **tandem** — cargo1 and cargo2 placed side by side in Z, centred together over the pallet. Combined depth `d1 + 0.02 + d2 ≤ 0.80 m` (EUR pallet depth). If the primary is too deep for any secondary to fit, falls back to stacked.

## Streamlit UI

```bash
cd src/
streamlit run app.py
```

Controls all parameters from a web interface, shows a live progress bar, and renders previews inline. Includes a "Restaurar valores por defecto" button to reset all sliders.

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

## Tests

```bash
cd src/
python3 -m pytest test_pipeline.py -v
```

101 tests covering geometry, PLY output, scene composition (single/stacked/tandem), flat-cargo mode, sensor degradation, previews, UI defaults, carretilla elevadora STL, and 5-class ML classifier (features, training, inference).

```bash
cd src/
# Pipeline tests only
python3 -m pytest test_pipeline.py -v
# All tests with coverage
python3 -m pytest test_pipeline.py test_classifier.py -v --cov=. --cov-report=html:../htmlcov
```

## Statistical analysis vs real data

Compares synthetic output against real FUSION3D captures.
Requires `../../Resources/Capturas_BBB/2026_03_23/` to be present.

```bash
cd src/
python3 analyze.py
```

Saves 7 figures to `output/analysis/`.

## 5-class per-point ML Classifier (§11)

Trains on the synthetic dataset and labels any PLY with 5 semantic classes:
`floor` (0) · `cargo` (1) · `vehicle` (2) · `person` (3) · `pallet` (4) · `outlier` (255)

### Train

```bash
cd src/
python3 classifier/train.py \
    [--data ../output/dataset] [--out ../models] \
    [--cv-estimators 100] [--cv-subsample 0.3] [--n-jobs -1]
```

Runs scene-level 5-fold CV for RandomForest and LightGBM, then retrains both on the full dataset.
Saves `models/classifier_rf.pkl` and `models/classifier_lgbm.pkl`.

| Model | CV F1-macro |
|---|---|
| RandomForest (300 trees) | 0.9157 |
| LightGBM | pending |

Per-class F1 (RF): floor 0.9843 · cargo 0.9541 · vehicle 0.8688 · person 0.9886 · pallet 0.7830

### Predict

```bash
cd src/
python3 classifier/predict.py input.ply output_labeled.ply [--model lgbm]
```

Accepts ASCII or binary PLY. Outputs binary-LE PLY with `x y z red green blue label` fields.

### Evaluate (confusion matrix + feature importance)

```bash
cd src/
python3 classifier/evaluate.py \
    --dataset-dir ../output/dataset \
    --output-dir ../output/classifier_eval
```

Saves normalised confusion matrices, F1 per class, feature importance plot, and RF vs LightGBM comparison.

## Scripts

All scripts run from the repo root (`datageneration/`), not from `src/`.

| Script | Purpose |
|--------|---------|
| `run_geometric.py` | **Main pipeline** — runs the geometric cargo extraction pipeline (stages 0–3) on a real PLY |
| `filter_person_yolo.py` | **Main pipeline** — filters person clusters from real captures using YOLOv8 2D detection |
| `eval_real_20esc.py` | Evaluates the geometric pipeline against Paula's reference volumes for all 20 scenarios |
| `eval_volume.py` | Volume estimation analysis |
| `plot_eval_20esc.py` | Generates plots from `eval_real_20esc.py` output |
| `train_person_raw.py` | Trains the raw person point-cloud classifier |
| `train_cluster_classifier.py` | Trains the cluster-level ML classifier |
| `export_visual_plys.py` | Exports coloured PLYs for visual inspection |
| `render_bbb_verify.py` | Renders BBB captures for verification |
| `render_bbb_diff.py` | Diff rendering between two BBB runs |
| `validate_synthetic.py` | Validates synthetic dataset statistics |
| `analyze_predictions.py` | Analyses per-scene prediction outputs |
| `binary_metrics.py` | Computes binary classification metrics |
| `bench_training.py` | Benchmarks training speed |
| `test_person_2d.py` | Exploratory tests for 2D person detection |
| `test_person_heuristic.py` | Exploratory tests for heuristic person detection |
| `scripts/archive/` | Retired debug/diagnostic scripts (not part of active pipeline) |

## Coordinate system

| Axis | Direction |
|---|---|
| X | right |
| Y | up (height, floor at Y = 0) |
| Z | depth (toward cameras) |

Units: **metres**

> Note: real FUSION3D clouds use Z as the height axis. `analyze.py` remaps automatically.

## Sensor noise (calibrated to real FUSION3D)

Non-Gaussian mixture model with quadratic depth scaling and axial ray projection (implemented in `sensor/noise.py`).

| Parameter | Value | Notes |
|---|---|---|
| `noise_core_ref` | 10 mm | σ_core (Gaussian) @ z_ref=3 m |
| `noise_tail_ref` | 25 mm | σ_tail (t-Student df=4) @ z_ref=3 m |
| `noise_gaussian_frac` | 0.60 | Fraction of points using Gaussian core |
| Depth scaling | quadratic | σ(Z) = σ_ref · (Z/3)² |
| Voxel grid | 19 mm | Calibrated to real NN spacing |
| Dropout | 15% | — |
| Outlier ratio | 3% | — |
| Floor extent X | ±2.5 m | Footprint: synth 5.29 m vs real 5.00 m |
| Floor extent Z | ±2.0 m | Footprint: synth 4.23 m vs real 3.97 m |

Empirical targets (FUSION3D BBB, measured 2026-04-20): cenital 3.5 m → σ_axial≈16 mm; der 4.3 m tilt 42° → σ_axial≈27 mm; izq 4.6 m tilt 42° → σ_axial≈41 mm.
Re-run `analyze.py` after regenerating dataset to verify NN density ratio vs real captures.
