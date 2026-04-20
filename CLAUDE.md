# CLAUDE.md — datageneration

Instrucciones permanentes para Claude Code en este repo.

## Repo

- **Rama de trabajo:** `developLucas` — push libre
- **Main/master:** nunca push sin permiso explícito del usuario
- **Remote:** SSH (no HTTPS). Ver `reference_bitbucket_ssh.md` en memoria si hay problemas de acceso.

## Comandos clave

```bash
# Tests (ejecutar siempre desde src/)
cd src && python3 -m pytest test_pipeline.py test_classifier.py -v

# Tests con cobertura (más lento)
make test

# Generar dataset sintético (traspaleta)
cd src && python3 generate_dataset.py --n 100 --seed 42

# Generar con carretilla elevadora al 100%
cd src && python3 generate_dataset.py --n 100 --p-forklift 1.0

# Generar mixto 50/50 traspaleta/carretilla
cd src && python3 generate_dataset.py --n 100 --p-forklift 0.5

# Inspeccionar STL nuevo (snippet de diagnóstico, no commitear)
cd src && python3 -c "
import open3d as o3d, numpy as np
m = o3d.io.read_triangle_mesh('../data/carretilla.stl')
v = np.asarray(m.vertices); e = v.max(0)-v.min(0)
print(f'extent {e}  min {v.min(0)}  max {v.max(0)}')"

# Entrenar clasificador (lanzar en background — tarda ~15 min)
# Flags recomendados: --lgbm-only-cv ahorra 5-10x tiempo de CV; --skip-rf-retrain para solo guardar LGBM
cd src && python3 -u classifier/train.py --out ../models --cv-estimators 100 --cv-subsample 0.3 --lgbm-only-cv --n-jobs -1

# Predecir sobre PLY real
cd src && python3 classifier/predict.py <input.ply> <output.ply>

# Evaluar con plots
cd src && python3 classifier/evaluate.py --dataset-dir ../output/dataset --output-dir ../output/classifier_eval
```

## Estructura src/

```
src/
├── generate_dataset.py      # CFG, generate_scene, run_generation, parse_args
│                            #   → genera escena: floor→pallet→cargo→vehicle→persona
│                            #   → vehicle selección: rng < p_forklift → carretilla, si no → jack
├── ply_io/ply.py            # LABEL, LABEL_RGB, save_ply, labels_to_rgb
├── geometry/
│   ├── meshes.py            # make_pallet/box/cylinder/person_mesh, make_pallet_jack_mesh
│   │                        # load_carretilla() (STL data/carretilla.stl)
│   │                        # Constantes: JACK_*/FORKLIFT_* FORK_H/L/BODY_D/X_HALF
│   └── composition.py       # sample_cargo_spec, compose_cargo (normal/stacked/tandem)
│                            # compose_cargo_on_vehicle(spec, rng, *, fork_h, fork_l)
├── sensor/noise.py          # sample_labeled, sample_floor, camera_arc_filter, degrade_labeled
│                            # compute_axial_noise: mezcla Gaussiana+t-Student, σ cuadrático en Z
├── utils/preview_grid.py    # render_scene, load_synth (importado por app.py y test_pipeline.py)
├── utils/view_png.py        # standalone, visualización local
├── classifier/
│   ├── features.py          # 19 features, K_NEIGHBORS=20, LOCAL_RADIUS_M=0.15, extract_features()
│   ├── train.py             # RF + LightGBM, CV StratifiedGroupKFold, guarda en models/
│   │                        # Flags útiles: --lgbm-only-cv, --skip-rf-retrain, --max-samples-rf
│   ├── predict.py           # inferencia PLY → PLY etiquetado (--align auto|fusion3d|none)
│   └── evaluate.py          # confusion matrix + feature importance → output/classifier_eval/
├── cargo_geometric/         # Pipeline extracción de cargo en datos BBB reales
│   ├── params.py            # GeometricParams dataclass — fuente única de verdad
│   ├── floor.py             # RANSAC floor removal + synthetic_floor() fallback
│   ├── anchor.py            # DBSCAN 2D para detectar bulto de cargo
│   ├── cargo.py             # Extracción con filtro negativo ML
│   ├── cluster_features.py  # 23 features a nivel de cluster
│   ├── cluster_classifier.py# Carga e inferencia del clasificador de clusters
│   ├── cluster_gt.py        # Ground truth de clusters (debug)
│   ├── pallet.py            # Detección de palet EUR
│   └── volume.py            # Estimación de volumen 2.5D height-field
├── test_pipeline.py         # ~118 tests de generación de datos (incl. floor, labels, tandem)
├── test_classifier.py       # 8 tests del clasificador
├── test_cluster_features.py # Tests de cluster_features.py
├── test_cluster_gt.py       # Tests de cluster_gt.py
├── test_cluster_classifier.py
├── test_cargo_classifier_integration.py
├── app.py                   # Streamlit UI (streamlit run app.py desde src/)
└── analyze.py               # comparación sintético vs real FUSION3D
```

## Vehículos

Un vehículo por escena, mutuamente exclusivos:
- **Traspaleta** (`pallet_jack`): procedural, siempre disponible.
- **Carretilla** (`forklift`): `load_carretilla()` desde `data/carretilla.stl`. Activar con `--p-forklift` > 0.

`compose_cargo_on_vehicle` usa las mismas constantes `JACK_*` / `FORKLIFT_*` en ambos casos.
Para añadir assets: `docs/ADDING_ASSETS.md`.

## Estado y tareas pendientes

Ver `DECISIONS.md` — sección `## Pendiente` al final. Es la única fuente de verdad sobre qué falta.

## Restricciones importantes

- PLY: escribir siempre con `save_ply()` de `ply_io/ply.py` — PCL añade un bloque camera que rompe CloudCompare
- Floor removal con RANSAC: validar altura del plano tras ajuste (puede fittear sobre cargo)
- `models/*.pkl` están gitignoreados — no asumir que existen sin verificar `ls models/`
- Dataset ≥500 escenas: **coordinar con Paula antes de generar**

## Labels

| ID | Clase | Color |
|----|-------|-------|
| 0 | floor | gris (60,60,60) |
| 1 | cargo | rojo (220,50,50) |
| 2 | vehicle | azul (0,120,255) |
| 3 | person | verde (39,174,96) |
| 4 | pallet | amarillo (255,210,0) |
| 255 | outlier | gris (60,60,60) |
