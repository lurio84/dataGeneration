# Dataset sintético — Decisiones de diseño

Documento vivo. Se actualiza a medida que se toman decisiones.

---

## 1. ¿Cómo va a ser el formato de etiqueta?

**Estado: ✅ Decidido**

- **Granularidad:** por punto — campo `label` (uchar) en cada vértice del PLY.
- **Clases:**
  | ID | Clase | Color RGB (PLY/CloudCompare) |
  |----|-------|------------------------------|
  | 0 | floor | Gris oscuro (60, 60, 60) |
  | 1 | cargo | Rojo (220, 50, 50) |
  | 2 | vehicle | Azul vivo (0, 120, 255) |
  | 3 | person | Verde (39, 174, 96) |
  | 4 | pallet | Amarillo (255, 210, 0) |
  | 255 | outlier/artefacto | Gris oscuro (60, 60, 60) |
- **Archivo:** etiqueta embebida en el mismo PLY, sin archivo separado. Formato binary little-endian: `x y z red green blue label`.
- **Color:** RGB calculado automáticamente desde `label` al generar → CloudCompare muestra colores sin configuración. Solo para visualización; el campo semántico son los enteros.
- **Salida por escena:** `{id:05d}.ply` + `{id:05d}.png` preview 3 vistas + `metadata.json` global.

---

## 2. Generar dataset etiquetado inicial (caso simple)

**Estado: ✅ Diseño cerrado — pendiente ejecutar dataset completo**

- **Script:** `src/generate_dataset.py`, listo y funcional.
- **Configuración validada (2026-04-09):**
  - Pallet EUR siempre presente (p_pallet=1.0)
  - 1 caja de cargo ortogonal, sin rotación, dimensiones aleatorias independientes: w∈[0.30,1.00]m, d∈[0.30,0.75]m, h∈[0.25,1.40]m
  - Traspaleta primitiva (cuerpo 0.70×0.90×0.40m + 2 horquillas 0.15×0.08×1.15m) siempre presente
  - Jack anclado al borde trasero del pallet (front_z=−0.40m), sin penetración
  - Caja centrada en pallet (ox=oz=0), jack centrado (jack_x=0)
  - Suelo activo: floor_extent_x=2.5m, floor_extent_z=2.0m (asimétrico, calibrado al FOV real)
- **N escenas:** 100 por defecto, reproducible con seed=42.
- **Dataset v1 ejecutado (2026-04-10).** Ver §8 para resultados completos.

---

## 3. ¿Qué otros primitivos de Open3D tienen sentido?

**Estado: 🟡 Parcialmente decidido — cilindro implementado**

- **Ya implementados:** caja (`create_box`), pallet (caja), traspaleta (3 cajas), persona (cilindro + esfera), **cilindro de cargo** (`make_cylinder_mesh`).
- **Decisión basada en análisis de datos reales (2026-04-09):**
  - Análisis de 6 escenas FUSION3D reales (marzo 2026) confirmó que todos los datos reales son cajas rectangulares (Asp XZ 1.0–1.3, r_std/r_mean 0.23–0.56).
  - El cilindro añade diversidad sintética de formas no vistas — bobinas industriales, bidones — necesaria para que el clasificador generalice.
  - El cono (señales de almacén) se descarta: anecdótico, sin caso de uso en las capturas.
  - L-shape: aplazado — los datos reales no lo requieren aún.
- **Cilindro implementado (2026-04-09):**
  - Función `make_cylinder_mesh(r, h)` — Y-aligned, base en Y=0, centrado en XZ.
  - Parámetro `p_cylinder` (0.0–1.0): probabilidad de cilindro en lugar de caja como carga primaria.
  - Rangos: radio ∈ [0.15, 0.40] m, altura ∈ [0.30, 1.20] m.
  - Controlable desde CLI (`--p-cylinder`, `--cyl-min-r`, etc.) y desde la app Streamlit.
- **Pendiente:**
  - L-shape (caja irregular): aplazado a iteración posterior.

---

## 4. ¿Queremos simular suelo?

**Estado: ✅ Decidido — activo**

- **Implementación:** `sample_floor()` activa, controlada por `enable_floor=True` en CFG.
- **Extents asimétricos:** floor_extent_x=2.5m, floor_extent_z=2.0m → span 5.0×4.0m, calibrado al FOV real (X≈5.85m, Z≈4.4m).
- **Color:** gris oscuro (60,60,60) para que retroceda visualmente en CloudCompare sobre fondo oscuro.

---

## 5. Función para combinar primitivos en casos complejos

**Estado: ✅ Implementado (2026-04-10)**

- **Funciones:** `make_primitive_mesh(spec)`, `sample_cargo_spec(cfg, rng, max_w, max_d, max_h)`, `compose_cargo(specs, mode, pallet_top_y, rng)`
- **Parámetro:** `p_multi_cargo` (0–1) reemplaza `p_two_boxes`; también `p_flat_cargo`, `flat_min_h`, `flat_max_h`
- **Modos implementados:**
  - `stacked`: cargo2 encima del cargo1 en Y. Cargo2 nunca más ancho/profundo que cargo1 (estabilidad). `cargo_back_z` solo de cargo1.
  - `tandem`: cargo1 y cargo2 centrados juntos sobre el pallet en Z. `oz1 = -(d2+gap)/2`, `oz2 = +(d1+gap)/2`. Restricción: `d1 + 0.02 + d2 ≤ EUR_D=0.80m`. Si no cabe, fallback a stacked automático.
- **Tipos mezclados:** cualquier combinación caja+caja, caja+cilindro, cilindro+cilindro.
- **Flat cargo:** `p_flat_cargo` activa modo de carga muy baja (h ≤ flat_max_h), simula casos difíciles cerca del suelo.
- **Tests:** 79 pytest (todos passing). `test_tandem_cargo_fits_within_pallet` verifica la restricción de pallet.

---

## 6. Búsqueda de recursos mesh 3D (.stl / .obj)

**Estado: 🟡 Parcialmente resuelto (2026-04-10)**

- **Persona ✅:** `data/person.stl` — "Tim full figure" (Thingiverse thing:719701, CC BY-SA, by thowe). Scan fotogramétrico real de ~50 fotos, procesado con Netfabb+Meshmixer. Decimado a 40k triángulos, Z-up→Y-up, base plana eliminada. Listo para uso.
- **Forklift ⏸:** `forklift.stl` referenciado en el código pero ausente en `data/`; el sistema usa el primitivo de traspaleta automáticamente. No es prioritario mientras el primitivo sea suficiente para el clasificador.
- **Pendiente (baja prioridad):** meshes de carretilla elevadora o cargas irregulares si el clasificador necesita más variabilidad.

---

## 7. Calibración de ruido vs sensor real

**Estado: ✅ Calibrado (2026-04-09)**

- **Método:** análisis cuantitativo sintético vs real con `analyze.py` (6 escenas FUSION3D reales, ROI crop X:±2.5m Y:-0.15..2.5m Z:±2.0m para comparación justa).
- **Parámetros finales:**
  - noise_std=0.035m (Gaussiano por punto; recalibrado 2026-04-10, era 0.030m)
  - voxel_size=0.019m (downsampling; calibrado a NN spacing real)
  - dropout_ratio=0.15, outlier_ratio=0.03, local_outlier_std=0.055m
  - Falloff de densidad ∝ 1/d² desde cámaras
- **Resultados finales (2026-04-10, noise_std=0.035m, 20 escenas vs 6 reales):**
  | Métrica | Sintético | Real FUSION3D | Estado |
  |---|---|---|---|
  | NN spacing | 49.9mm | 51.3mm | ✅ (2.7%) |
  | Floor roughness σ | 26.7mm | 29.7mm | ✅ (gap 3.1mm) |
  | Footprint X | 5.35m | 5.00m | ✅ |
  | Footprint Z | 4.41m | 3.97m | ✅ |
  | Puntos/escena | 63k | 235k ROI | ℹ️ ver nota |
- **Nota puntos/escena:** diferencia (3.7x) explicada por techo (~2.4m) y paredes en datos reales que el sintético no modela. La densidad local (NN spacing) sí está calibrada — es la métrica relevante para el clasificador.
- **Nota techo:** el real tiene un pico de puntos a Y≈2.4m (techo del almacén) que el sintético no reproduce. Tener en cuenta al diseñar features de ML v6.
- **Nota roughness cargo:** la banda Y=0.2–1.6m no es válida para calibración — mide variación geométrica entre caras, no ruido del sensor.
- **Pendiente con Paula:** validar patrones de oclusión y reflexiones especulares no cubiertos por el modelo Gaussiano.

---

## 8. Generar dataset final

**Estado: ✅ Dataset v1 generado (2026-04-10) — listo para revisión con Paula**

- **Dataset v1 (validación):** 100 escenas, seed=42, parámetros calibrados.
- **Comando ejecutado:**
  ```
  cd src && python3 generate_dataset.py \
    --n 100 --seed 42 \
    --p-cylinder 0.15 \
    --p-multi-cargo 0.25 \
    --p-flat-cargo 0.10
  ```
- **Probabilidades de escena activas:**
  | Parámetro | Valor | Justificación |
  |---|---|---|
  | `p_person` | 0.30 | Calibrado (§10) |
  | `p_multi_cargo` | 0.25 | ~25% escenas con carga doble (stacked/tandem) |
  | `p_flat_cargo` | 0.10 | ~10% caso difícil near-floor (h≤15cm) |
  | `p_cylinder` | 0.15 | ~15% cilindros (bobinas/bidones); diversidad sintética |
- **Resultados analyze.py vs 6 escenas reales:**
  | Métrica | Sintético | Real FUSION3D | Estado |
  |---|---|---|---|
  | NN spacing | 49.9mm | 51.3mm | ✅ (2.7%) |
  | Floor roughness σ | 26.8mm | 30.0mm | ✅ (gap +3.2mm) |
  | Footprint X | 5.38m | 5.00m | ✅ |
  | Footprint Z | 4.46m | 3.97m | ✅ |
  | Puntos/escena | 64k | 235k ROI | ℹ️ ver §7 |
- **Distribución de labels:**
  floor 72.9% · vehicle 9.4% · cargo 8.6% · pallet 5.0% · outlier 2.9% · person 1.3%
- **Metadatos:** `results/metadata_v1.json` (PLYs excluidos de git por .gitignore).
- **Previews:** 100 PNG en `output/previews/` (excluidos de git).
- **Pendiente (dataset definitivo ≥500 escenas):**
  - Validación oclusión/reflexiones con Paula (§7)
  - Decidir si añadir techo/paredes para igualar point count real
  - Comando: `cd src && python3 generate_dataset.py --n 500 --seed 42 --p-cylinder 0.15 --p-multi-cargo 0.25 --p-flat-cargo 0.10`

---

## 9. Interfaz Streamlit (app.py)

**Estado: ✅ Implementado (2026-04-09)**

- **Fichero:** `src/app.py`, ejecutar con `streamlit run app.py` desde `src/`
- **Parámetros controlables desde la UI:**
  - Dataset: n_samples, seed, output_dir
  - Composición de escena: enable_floor, floor_extent_x/z, p_pallet, p_multi_cargo, p_flat_cargo, p_person, p_cylinder
  - Dimensiones de cilindro: cyl_r (min/max), cyl_h (min/max)
  - Dimensiones de caja: rangos min/max de w, d, h
  - Ruido del sensor: noise_std, dropout_ratio, outlier_ratio, voxel_size, local_outlier_std
- **Funcionalidades:** botón "Restaurar valores por defecto", barra de progreso por escena, vista previa PNG con hover azul + click abre imagen completa en nueva pestaña, visor de metadata.json, indicador de presencia de data/person.stl
- **Tests:** `src/test_pipeline.py` — 79 tests totales (todos passing); ejecutar con `python3 -m pytest test_pipeline.py -v`

---

## 10. Primitivo persona (p_person)

**Estado: ✅ Decidido e implementado (2026-04-10)**

- **Geometría:** malla STL realista (`data/person.stl`). Si el archivo no existe, fallback automático al cilindro+esfera escalado proporcionalmente a la altura objetivo.
  - STL criteria: watertight, pose neutral de pie (A-pose o recta, no T-pose), < 50k triángulos, licencia CC0 o CC-BY.
  - Fuentes candidatas: GrabCAD (`"human figure standing neutral pose STL"`), Sketchfab (filtro CC0, `"human standing neutral pose low-poly"`).
  - Pre-procesado manual (MeshLab/Blender): escalar a 1.75m Y-up, base en Y=0, centrado en XZ, exportar STL binario en metros.
- **`p_person`:** 0.30 (30% de escenas). Era 0.00.
- **Altura:** 1.70–1.80m por escena (`rng.uniform(1.70, 1.80)`).
- **Rotación Y:** aleatoria 0–360° por escena.
- **Zonas de placement (con jack siempre presente):**
  - Zona operario (60% de apariciones, requiere pallet presente): detrás del cuerpo del jack, Z ≈ `jack_back_z − [0.30..0.70]m`, X ∈ (−0.50, +0.50). El operario empuja la traspaleta desde ahí.
  - Zona perímetro cargo (40%, o fallback si hay colisión): ángulo aleatorio 0-360° alrededor del pallet/cargo, radio = `half_diag + [0.30..0.70]m`.
- **Collision check:** círculo de huella persona (r=0.30m) vs AABB del jack en XZ. Hasta 20 reintentos por zona.
- **Metadata:** campo `person` en `objects` con `x`, `z`, `height`, `rot_deg`, `zone` ("operator"/"perimeter"), `stl` ("person.stl"/"fallback").
- **Label:** 3, color verde (39, 174, 96). Sin cambios al label map ni formato PLY.
- **Tests:** clase `TestPerson` (8 tests) añadida en `test_pipeline.py` (sección 9).

---

## 11. Clasificador ML por punto — 5 clases (§11)

**Estado: ✅ Implementado (2026-04-10)**

- **Problema:** etiquetar escenas reales sin ground truth manual.
- **Solución:** entrenar sobre sintético calibrado (§7 + §8), predecir sobre PLYs reales.
- **Archivos:**
  - `src/classifier/features.py` — extracción de 15 features por punto
  - `src/classifier/train.py` — entrenamiento RF + LightGBM, CV escena-level
  - `src/classifier/predict.py` — inferencia sobre PLY sin etiquetar → PLY etiquetado
  - `src/classifier/evaluate.py` — métricas, confusion matrix, feature importance
  - `src/test_classifier.py` — 8 tests pytest (todos passing)
  - `models/` — directorio versionado; `.pkl` gitignoreados (demasiado grandes)

### Features (15 per-point)

| # | Nombre | Descripción |
|---|--------|-------------|
| 0 | `y` | Altura absoluta |
| 1 | `y_norm` | Altura relativa en la escena |
| 2 | `z` | Profundidad (distancia a cámaras) |
| 3 | `dist_xz` | Distancia radial horizontal al origen |
| 4 | `local_density` | Puntos dentro de radio 0.15 m |
| 5 | `nbr_y_mean` | Media de Y en los k=20 vecinos |
| 6 | `nbr_y_std` | Std de Y en los k vecinos (rugosidad local) |
| 7 | `height_range_local` | max(Y_nbrs) − min(Y_nbrs) |
| 8 | `normal_y` | Componente Y del normal estimado por PCA local |
| 9 | `curvature` | λ₃ / (λ₁+λ₂+λ₃) |
| 10 | `planarity` | (λ₂−λ₃) / λ₁ |
| 11 | `linearity` | (λ₁−λ₂) / λ₁ |
| 12 | `sphericity` | λ₃ / λ₁ |
| 13 | `verticality` | \|normal_y\| |
| 14 | `dist_centroid_xz` | Distancia XZ al centroide de la escena |

Implementación: `scipy.spatial.cKDTree` + `np.linalg.eigh` para PCA local (k=20 vecinos).
Paralelización: `joblib.Parallel(prefer="threads")` sobre escenas en `load_dataset`.

### Modelos comparados

| Modelo | Hiperparámetros clave |
|--------|----------------------|
| RandomForest | n_estimators=300, max_depth=20, class_weight='balanced', n_jobs=-1 |
| LightGBM | num_leaves=63, class_weight='balanced', n_jobs=-1 |

CV: `StratifiedGroupKFold(n_splits=5)`, group=scene_id (sin data leakage entre puntos de la misma escena).

### Resultados CV (dataset v1 — 100 escenas sintéticas, seed=42)

CV config: `--cv-estimators 100 --cv-subsample 0.3` (30% estratificado por escena, 1.88M/6.25M pts).
Feature extraction: 6.25M pts en 15.4s con 16 threads.

| Clase | RF F1 | LGBM F1 |
|-------|-------|---------|
| floor | 0.9843 | 0.9755 |
| cargo | 0.9541 | 0.9495 |
| vehicle | 0.8688 | 0.8399 |
| person | 0.9886 | 0.9417 |
| pallet | 0.7830 | 0.7489 |
| **macro** | **0.9157** | **0.8911** |

RF folds: 0.9123 / 0.9194 / 0.9193 / 0.9134 / 0.9142 (σ=0.003, muy estable).
LGBM folds: 0.8999 / 0.9053 / 0.8687 / 0.8863 / 0.8951 (σ=0.013, más varianza que RF).
Nota: `pallet` es la clase más difícil en ambos modelos — superficie plana baja, geométricamente similar al suelo. RF supera LGBM en todas las clases; se usará RF como modelo principal. `class_weight='balanced'` compensa la baja frecuencia de `person` (1.4%) con éxito en ambos.
Retrain full dataset (6.25M pts): RF 300 trees → 1864s, LGBM → 72s. Tamaños: classifier_rf.pkl=1.9G, classifier_lgbm.pkl=3.4M.

### Evaluación real V2 (pendiente post-Paula)

- Dataset v1 entrenado sobre sintético calibrado (§8).
- **Validación cualitativa V1 — resultado (2026-04-13):** `predict.py` sobre `logicarc_cargo_segmentation/colored_clouds/T01_centro_bulto_grande_colored.ply` → FALLO: 69% clasificado como person, 0% como cargo. **Causa identificada:** las colored_clouds son crops pre-segmentados (1,595 pts, Y∈[-0.27, 1.15m]), no escenas completas. El clasificador espera escenas enteras con suelo en Y=0 y contexto 5×4m — los features posicionales (`dist_xz`, `y`, `z`, `dist_centroid_xz`) son incoherentes sobre un crop.
- **Fix implementado (2026-04-13):** `predict.py` ahora incluye:
  - `--align auto|fusion3d|none` (default: auto): detecta eje de suelo por histograma, hace swap Y↔Z si FUSION3D (Z-up), translada suelo a Y=0.
  - ROI crop X:±2.5m, Y:-0.15..2.5m, Z:±2.0m — puntos fuera → label=255 (outlier), no pasan por clasificador.
  - Input correcto: `Resources/Capturas_BBB/*/FUSION3D/fusion3d_merged_*.ply` (binary-LE, ~24–271k pts). No usar tri_cloud ni colored_clouds (distintos sistemas de coordenadas / crops pre-segmentados).
- **Resultado validación visual (2026-04-13, Escenario_02/Captura_01, FUSION3D):**
  - Carga principal (rojo) identificada correctamente ✅
  - Traspaleta real diferente al modelo sintético (3 cajas primitivas) → confusión con person esperada
  - Dos cajas adicionales entre carga principal y persona → clasificadas como person (verde), no como cargo
  - Person sobrepredicado (~18% ROI vs 1.4% training): traspaleta real + cajas secundarias + persona juntos
- **evaluate.py — OOM con RF:** re-entrenar 5 folds × 300 árboles RF sobre 6.25M pts excede la RAM. Workaround: `--model lgbm` para confusion matrix; feature importance RF generado desde pkl guardado (sin re-entrenamiento).
- V2: re-etiquetar ≥5 escenas reales **completas** con 5 clases (Paula + usuario) → test set formal con métricas reales por clase.

---

## Pendiente

- [x] Rellenar columna LGBM en tabla §11 con F1 por clase (2026-04-13)
- [x] `evaluate.py` → confusion matrix LGBM + feature importance RF en `output/classifier_eval/` (RF CV omitido: OOM, ver §11 evaluación real V2)
- [x] `predict.py` → validación sobre FUSION3D escena completa — carga principal detectada, fixes de alineamiento Y↔Z y ROI crop implementados (ver §11 evaluación real V2)
- [ ] Commit rama developLucas (Track B + B9 + CLAUDE.md + §11 resultados + hallazgos evaluate/predict)
- [ ] Dataset ≥500 escenas — **coordinar con Paula antes de generar**
