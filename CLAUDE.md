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

# Generar dataset sintético
cd src && python3 generate_dataset.py --n 100 --seed 42

# Entrenar clasificador (lanzar en background — tarda ~15 min)
cd src && python3 -u classifier/train.py --out ../models --cv-estimators 100 --cv-subsample 0.3 --n-jobs -1

# Predecir sobre PLY real
cd src && python3 classifier/predict.py <input.ply> <output.ply>

# Evaluar con plots
cd src && python3 classifier/evaluate.py --dataset-dir ../output/dataset --output-dir ../output/classifier_eval
```

## Estructura src/

```
src/
├── generate_dataset.py      # CFG, generate_scene, run_generation, parse_args
├── ply_io/ply.py            # LABEL, LABEL_RGB, save_ply, labels_to_rgb
├── geometry/meshes.py       # EUR_W/H/D, make_*_mesh, load_forklift
├── geometry/composition.py  # _spec_w, _spec_d, sample_cargo_spec, compose_cargo
├── sensor/noise.py          # sample_labeled, sample_floor, camera_arc_filter, degrade_labeled
├── utils/preview_grid.py    # render_scene, load_synth (importado por app.py y test_pipeline.py)
├── utils/view_png.py        # standalone, visualización local
├── classifier/
│   ├── features.py          # K_NEIGHBORS=20, LOCAL_RADIUS_M=0.15, extract_features()
│   ├── train.py             # RF + LightGBM, CV StratifiedGroupKFold, guarda en models/
│   ├── predict.py           # inferencia PLY → PLY etiquetado
│   └── evaluate.py          # confusion matrix + feature importance → output/classifier_eval/
├── test_pipeline.py         # 79 tests de generación de datos  \
├── test_classifier.py       # 8 tests del clasificador         / → 87 total
├── app.py                   # Streamlit UI (streamlit run app.py desde src/)
└── analyze.py               # comparación sintético vs real FUSION3D
```

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
