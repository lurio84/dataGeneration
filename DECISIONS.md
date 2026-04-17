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

**Estado: ✅ Implementado (2026-04-10) — ampliado con cargo-on-vehicle (2026-04-13)**

- **Funciones:** `make_primitive_mesh(spec)`, `sample_cargo_spec(cfg, rng, max_w, max_d, max_h)`, `compose_cargo(specs, mode, pallet_top_y, rng)`, `compose_cargo_on_vehicle(spec, rng)`
- **Parámetro:** `p_multi_cargo` (0–1) reemplaza `p_two_boxes`; también `p_flat_cargo`, `flat_min_h`, `flat_max_h`
- **Modos implementados:**
  - `stacked`: cargo2 encima del cargo1 en Y. Cargo2 nunca más ancho/profundo que cargo1 (estabilidad). `cargo_back_z` solo de cargo1.
  - `tandem`: cargo1 y cargo2 centrados juntos sobre el pallet en Z. `oz1 = -(d2+gap)/2`, `oz2 = +(d1+gap)/2`. Restricción: `d1 + 0.02 + d2 ≤ EUR_D=0.80m`. Si no cabe, fallback a stacked automático.
  - `cargo_on_vehicle`: cargo (spec único) sobre las horcas de la traspaleta. Base en Y=JACK_FORK_H=0.06m. XZ aleatorio dentro del footprint: ox∈[-0.20,+0.20], oz∈[d/2, JACK_FORK_L-d/2]. Jack queda en origen (front_z=0). Activado via `p_cargo_on_vehicle` cuando `has_pallet=False`.
- **Tipos mezclados:** cualquier combinación caja+caja, caja+cilindro, cilindro+cilindro.
- **Flat cargo:** `p_flat_cargo` activa modo de carga muy baja (h ≤ flat_max_h), simula casos difíciles cerca del suelo.
- **Tests:** 89 pytest (todos passing). `TestCargoOnVehicle` verifica base en Y=JACK_FORK_H y escena sin pallet completa.

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

- **Dataset v1 (validación):** 100 escenas, seed=42, `output/dataset/` (regenerado limpio 2026-04-13 — ver §14).
- **CFG activo para dataset v2:**
  | Parámetro | v1 | v2 | Justificación cambio |
  |---|---|---|---|
  | `p_pallet` | 1.00 | **0.80** | 20% escenas sin pallet — patrón BBB-5 real |
  | `p_cargo_on_vehicle` | — | **0.50** | Cuando sin pallet: 50% cargo sobre horcas |
  | `p_person` | 0.30 | 0.30 | Sin cambio |
  | `p_multi_cargo` | 0.25 | 0.25 | Sin cambio |
  | `p_flat_cargo` | 0.10 | 0.10 | Sin cambio |
  | `p_cylinder` | 0.15 | 0.15 | Sin cambio |
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
- **Metadatos:** `output/dataset/metadata.json` (auto-generado por generate_dataset.py, PLYs excluidos de git por .gitignore).
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
- **Tests:** clase `TestPerson` (8 tests) añadida en `test_pipeline.py` (sección 9). Total: 89 tests.

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

### Features (16 per-point) — post-rediseño 2026-04-13

**Eliminadas** (causaban predicción posicional en anillo — validado visualmente en 6 escenas BBB):
- `dist_xz` — distancia radial XZ al origen → vehicle/person clasificados por posición, no forma
- `dist_centroid_xz` — distancia XZ al centroide de escena → mismo efecto

| # | Nombre | Descripción |
|---|--------|-------------|
| 0 | `y` | Altura absoluta |
| 1 | `y_norm` | Altura relativa en la escena |
| 2 | `z` | Profundidad (distancia a cámaras) |
| 3 | `local_density` | Puntos dentro de radio 0.15 m |
| 4 | `nbr_y_mean` | Media de Y en los k=20 vecinos |
| 5 | `nbr_y_std` | Std de Y en los k vecinos (rugosidad local) |
| 6 | `height_range_local` | max(Y_nbrs) − min(Y_nbrs) |
| 7 | `normal_y` | Componente Y del normal estimado por PCA local |
| 8 | `curvature` | λ₃ / (λ₁+λ₂+λ₃) |
| 9 | `planarity` | (λ₂−λ₃) / λ₁ |
| 10 | `linearity` | (λ₁−λ₂) / λ₁ |
| 11 | `sphericity` | λ₃ / λ₁ |
| 12 | `verticality` | \|normal_y\| |
| 13 | `normal_y_std` | Std de normal_y en k-NN → regularidad de superficie |
| 14 | `lam_ratio_12` | λ₁/λ₂ → elongación (horcas/persona vs caras de caja) |
| 15 | `planarity_large` | Planarity a k=50 → forma macro-escala |

Implementación: `scipy.spatial.cKDTree` + `np.linalg.eigh` para PCA local (k=20 y k=50).
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
- [x] `evaluate.py` → confusion matrix LGBM + feature importance RF en `output/classifier_eval/`
- [x] `predict.py` → validación sobre FUSION3D escena completa — fixes align + ROI crop implementados
- [x] **Commit ce54270** en `developLucas` — features 16, meshes reales, A1 cargo-on-vehicle, 89 tests
- [x] Features rediseñadas — eliminados dist_xz + dist_centroid_xz, añadidos 3 nuevas (2026-04-13)
- [x] Meshes reales — pallet 5 tablones, traspaleta cuerpo bajo + horcas planas + timón en U (2026-04-13)
- [x] A1 — cargo-on-vehicle: compose_cargo_on_vehicle, p_pallet=0.80, p_cargo_on_vehicle=0.50 (2026-04-13)
- [x] **Decidir estrategia floor para v2** — con_suelo (ver §13, cerrado 2026-04-13)
- [x] Dataset v1 regenerado limpio (100 escenas, CFG actual, 2026-04-13) — ver §14
- [x] Feature 17 `height_above_local_floor` añadida (2026-04-13)
- [x] Flags training: `--floor-subsample`, `--class-weight-mult`, `--max-samples-rf`, `--lgbm-only-cv`
- [x] `scripts/bench_training.py` — comparador 2 configs LGBM-only CV
- [ ] **Fase 1.7 — Retrain sobre v1 limpio + validación visual BBB** (siguiente paso, ver plan `floofy-rolling-pumpkin.md`)
- [ ] Re-correr bench sobre v1 limpio para confirmar F1 vs históricos 0.89
- [ ] Dataset v2 ≥500 escenas — Paula OK, pendiente tras Fase 1.7
- [ ] Retrain RF+LGBM sobre dataset v2 con 17 features + flags optimización
- [ ] Validación visual post-retrain v2 sobre 6 escenas BBB

---

## 12. Experimento no-floor — entrenamiento sin clase suelo

**Estado: 🔴 Decisión pendiente (2026-04-13)**

**Motivación:** floor representa ~77% de los puntos en el dataset (label=0). Es la clase más fácil de separar (Y bajo, planarity=1, normal_y≈1). Hipótesis: excluirlo del training podría mejorar F1 en clases difíciles (vehicle, pallet) y reducir tiempo de entrenamiento ~4×.

**Implementación:**
- `train.py --no-floor`: filtra label=0 antes de CV y retrain final
- `predict.py --floor-threshold Y`: puntos con Y < threshold → label=0 directamente sin clasificador
- Modelos guardados en `models/no_floor/` (RF + LGBM)
- Predicciones BBB en `output/bbb_nofloor/`

**Resultados CV RF (139 escenas, 16 features nuevas):**

| Clase | with_floor (ref v1, features antiguas) | no_floor (experimento) | Δ |
|-------|---------------------------------------|------------------------|---|
| floor | 0.9843 | — (excluida) | — |
| cargo | 0.9541 | 0.9258 | −0.028 |
| vehicle | 0.8688 | 0.8599 | −0.009 |
| person | 0.9886 | 0.8956 | **−0.093** |
| pallet | 0.7830 | **0.8625** | **+0.080** |
| macro | 0.9157 | 0.8859 | — |

Nota: la comparación no es perfectamente justa — with_floor usa features antiguas (15), no_floor usa features nuevas (16).

**Validación visual sobre 6 escenas BBB reales:**
- ✅ Person sobrepredicado (anillo verde) desaparece completamente
- ✅ Pallet mejora visiblemente (esc4, esc6)
- ✅ Threshold Y<0.05 funciona — suelo limpio sin falsos positivos en floor
- ❌ Person real no detectada en esc4 (trade-off inaceptable para producción)
- ⚠️ Vehicle sigue confuso — problema de geometría del mesh, no de floor/nofloor
- ⚠️ Comparación parcialmente injusta: with_floor nunca se reentrenó con 16 features

**Opciones abiertas para análisis:**

| Opción | Descripción | Ventaja | Riesgo |
|--------|-------------|---------|--------|
| A — with_floor (baseline) | Entrenar con las 5 clases incluyendo floor | Person conservado | Tiempo training en v2, anillo puede reaparecer |
| B — no_floor puro | Excluir floor completamente | +pallet, 4× más rápido CV | Person −9%, pierde personas reales |
| C — floor subsampled | Incluir floor pero solo 10-20% de sus puntos (same absolute count as no_floor) | Mantiene contexto de floor sin dominar | Más complejo, sin validar |
| D — two-stage | Clasificador binario floor/no-floor → clasificador 4-clases para no-floor | Cada modelo más simple | Doble inferencia, más complejidad de pipeline |
| E — with_floor + cv-subsample ajustado | Entrenar con floor, reducir cv-subsample a 0.06 para compensar el 5× de datos en v2 | Sin cambios de arquitectura | CV con menos datos puede ser menos fiable |

**Decisión cerrada (2026-04-13):** **Opción A — with_floor**. Ver §13 para el bench honesto que sentenció la decisión.

---

## 13. Bench floor sí/no + decisión final (2026-04-13)

**Estado: ✅ Decidido — entrenar CON suelo.**

### Infra nueva añadida (commit `a7e82a6`)

- **Feature 17** `height_above_local_floor` (`features.py`): mediana de Y por celda XZ 0.5 m. Motivación: desacoplar información de altura del volumen de puntos floor en training.
- **4 flags nuevos en `train.py`**, todos con defaults que preservan comportamiento anterior:
  - `--floor-subsample contextual:F,bulk:F` — KDTree non-floor → mask contextual/bulk.
  - `--class-weight-mult person:2,pallet:2` — sample_weight encima de `balanced`.
  - `--max-samples-rf N` — cap en bootstrap de RF para evitar OOM en retrain v2.
  - `--lgbm-only-cv` — salta CV de RF (5-10× speedup).
- **`scripts/bench_training.py`** — compara 2 configs sobre mismo dataset con load+features cacheados, LGBM-only CV, escribe CSV.

### Bench 1 — 139 escenas contaminadas (dirty)

Ejecutado sobre mix de 100 v1 oficiales + 39 legacy de iteraciones previas con CFG desconocida. Resultados en `results/bench_v1_dirty139.csv`.

| Métrica | A_baseline (with_floor) | C_contextual (floor:0.1 bulk) | Δ |
|---|---|---|---|
| F1-macro | 0.8405 | 0.8151 | **−0.025** |
| F1-cargo | 0.9272 | 0.9072 | −0.020 |
| F1-vehicle | 0.6738 | 0.6184 | **−0.055** |
| F1-person | 0.8134 | 0.8222 | +0.009 |
| F1-pallet | 0.8198 | 0.7825 | **−0.037** |
| CV time (s) | 216.1 | 128.6 | −87.5 (1.7× más rápido) |
| RAM real (`/usr/bin/time -v`) | — | 4.3 GB peak | — |

### Lecciones del bench

1. **C no renta.** Intercambia 87s de CV por −0.025 de F1-macro, con pallet y vehicle como grandes perdedores. Person apenas mejora.
2. **La intuición "70% de puntos = 70% del tiempo" es falsa.** En árboles (LGBM/RF) el tiempo lo gasta el boundary difícil, no el volumen homogéneo. Floor es trivial de separar (un split en Y<0.1 y listo); no itera sobre esos puntos. El ahorro real al quitar suelo es modesto y se paga con F1.
3. **El OOM en retrain v2 NO lo resuelve floor subsampling** — lo resuelve `--max-samples-rf 2000000`, que es ortogonal al floor.
4. **tracemalloc subestima RAM** (reportó 2.3 GB mientras el proceso real usaba 4.3 GB según `time -v`). No usar para decidir OOM.
5. **Las F1 absolutas estaban bajas** (person 0.81 vs histórico 0.94) — contaminación por 39 escenas legacy, no fallo de features. Ver §14.

### Decisión final

- **Estrategia floor v2:** **Opción A — with_floor, sin subsampling.** El default `--floor-subsample none` queda como definitivo.
- **Speedup v2:** por `--lgbm-only-cv` + `--max-samples-rf 2000000` + `--cv-subsample 0.10`. Proyección: retrain v2 (500 escenas, ~30M pts) en ~40-50 min, sin OOM.
- **Class weights:** ~~`--class-weight-mult person:2,pallet:2` activo~~ → **Revertido en §15**: medido contraproducente, baja F1 en TODAS las clases, no usar.
- **Feature 17:** se queda. El test unitario confirma que el algoritmo es correcto; puede aportar en v2 aunque en v1 contaminado no desplazó el F1.

### Nueva memoria registrada

`feedback_tree_training_cost.md` (type: feedback) — "Para clasificadores de árboles (RF/LGBM), el tiempo de training NO es proporcional a la fracción de puntos por clase; es proporcional a la dificultad del boundary. Reducir clases homogéneas como floor (77%) ahorra menos tiempo del que parece y suele costar F1 en clases vecinas."

---

## 14. Limpieza dataset v1 (2026-04-13)

**Estado: ✅ Ejecutado.**

**Problema detectado durante el bench:** `output/dataset/` acumulaba **139 PLYs** — mix de 100 v1 oficiales (2026-04-10) + 39 legacy de iteraciones previas con CFG desconocida (meshes antiguos, posible distinta estadística de ruido). El clasificador entrenaba sobre dos distribuciones a la vez, explicando la caída de 0.05 en F1-macro vs histórico (0.84 actual vs 0.89 §11).

**Acción:**
1. Regeneración limpia de 100 escenas con CFG actual (`n_samples=100, seed=42, noise=0.035, voxel=0.019, p_pallet=0.80, p_cargo_on_vehicle=0.50, p_person=0.30`).
2. `output/dataset/` (139 legacy) borrado.
3. `output/dataset_v1_clean/` → renombrado a `output/dataset/` (path canónico).
4. `output/dataset/metadata.json` (100 entradas, auto-generado).
5. Residuales borrados: `output/a1_preview/`, `output/analysis/`, `output/fase0/`, `output/train_*.log`, `results/metadata_v1.json` (stale), `__pycache__`, `.pytest_cache`.
6. Bench contaminado preservado: `results/bench_v1_dirty139.{csv,log}` (histórico).

**Estado final del repo:**
- `output/dataset/` — 100 PLYs limpios (96 MB) + `metadata.json`
- `output/classifier_eval/` — plots históricos (~124 KB)
- `results/` — solo `bench_v1_dirty139.{csv,log}`
- Sin modelos entrenados aún (todos los `.pkl` borrados en Fase 0 del plan)

**Siguiente paso:** re-correr bench sobre v1 limpio para medir F1 honesto (debería estar cerca del 0.89 histórico de §11 LGBM). Si matchea → Fase 1.7 (retrain + validación visual BBB). Si no matchea → investigar features (eliminación de `dist_xz` sin reemplazo equivalente puede haber dolido más de lo pensado).

---

## 15. Diagnóstico LGBM v1 sobre BBB real — 3 hallazgos (2026-04-14)

**Estado:** Diagnóstico cerrado. Quedan 2 problemas estructurales pendientes (ver final).

Sesión de validación visual sobre las 6 escenas BBB reales reveló múltiples problemas en cascada en el modelo entrenado sobre v1 limpio. Los descubrimientos se hicieron por iteración: modelo → predicción real → render PNG y CloudCompare → diagnóstico → fix → siguiente iteración.

### Hallazgo 1 — Feature `z` era positional leakage

**Síntoma:** En las primeras predicciones sobre BBB se observaron **bandas paralelas a ejes** en TOP view (rojo/azul/verde organizado en franjas perpendiculares a Z), idéntico patrón al "anillo verde" histórico de §11 pero rotado a ejes rectos.

**Causa:** [features.py](src/classifier/features.py) incluía `z` (depth raw) como feature 2 con comentario "proxy for sensor noise level". En sintético los objetos están en posiciones fijas en Z (cargo en origen, person en perímetro), el modelo memorizaba rangos Z → clase. En real producía bandas. La justificación de "noise proxy" era falsa: el ruido dependiente de distancia ya lo capturan `local_density`, `nbr_y_std`, `normal_y_std`, `planarity`.

**Fix:** Feature `z` eliminada de `FEATURE_NAMES` y de `extract_features()`. De 17 → 16 features. CV F1-macro bajó 0.085 (0.85 → 0.77) — exactamente el tamaño del leak. Tests 92/92 verde.

**Lección general:** cualquier feature posicional cruda (x, y, z, dist_*) en clasificadores per-point sobre datasets sintéticos centrados es un riesgo de leakage. La regla general: features deben ser invariantes a translación de la nube, salvo la altura `y` que es legítima por su correlación física con la clase.

### Hallazgo 2 — Distribution shift de densidad sintético→real

**Síntoma:** Tras quitar `z`, las bandas desaparecen pero las predicciones siguen mal. Distribución sobre BBB real (vs target sintético):

| Clase | Target | Sin voxelizar | Con vox 0.035 |
|---|---|---|---|
| floor | 77% | 2% | **18-21%** |
| cargo | 9% | 38% | **7-15%** |
| vehicle | 5% | 42% | 34-50% (mesh §12) |
| person | 2% | 3% | **5-17%** |
| pallet | 7% | 14% | **2-6%** |

**Causa medida:** densidad real **3.2× mayor** que sintética (mediana `local_density` 537 vs 153, p90 783 vs 229). Real BBB scene: ~270k pts. Synth scene: ~60k pts. Ratio total 4.8×. El feature `local_density` (radio 0.15 m) está completamente fuera del rango aprendido en sintético, y otros features de vecindario indirectamente afectados.

**Causa raíz del shift:** sintético usa voxel 0.019 m + `camera_arc_filter` agresivo que dropea muchos puntos. Real FUSION3D fusiona 3 cámaras sin filtro equivalente y produce densidad nativa mucho mayor.

**Fix validado en sesión:** voxelizar el PLY real a **0.035 m** antes de predict alinea la densidad media con el sintético (med 178 vs 163, p90 250 vs 229). Al predict con BBB voxelizado:
- Floor sube de 2% a 18-21% ✅
- Cargo baja de 38% a 7-15% ✅
- Pallet vuelve a rango razonable ✅
- Person y vehicle no se arreglan completamente (otros problemas)

**Pendiente de implementar permanentemente:** añadir flag `--voxel-size FLOAT` a [predict.py](src/classifier/predict.py) que voxelice el input PLY antes de extraer features. Default sugerido `0.035` (medido empíricamente). En esta sesión se hizo el preprocessing manual con un script externo usando `open3d.geometry.PointCloud.voxel_down_sample`.

**Lección general:** features absolutas de vecindario (densidad, conteos, distancias) son sensibles a densidad de muestreo. Si los pipelines de captura sintético y real difieren en densidad → inevitablemente hay shift. Soluciones: (a) calibrar densidad al emitir/preprocesar, (b) usar features normalizadas relativas a la densidad de la escena, (c) usar features puramente geométricas invariantes (PCA descriptors).

### Hallazgo 3 — `--class-weight-mult person:2,pallet:2` era contraproducente

**Síntoma:** Person sobre-predicho en BBB real con `class_weight='balanced' × person:2`. Hipótesis inicial: el multiplicador encima de `balanced` (que ya pesa person ~55× por su frecuencia 1.8%) lleva el peso efectivo a ~110×, generando over-prediction.

**Verificación CV:** retrain LGBM-only con y sin el multiplicador sobre v1 limpio (sin `z`):

| Clase | con `person:2,pallet:2` | **sin multiplicador** | Δ |
|---|---|---|---|
| floor | 0.9590 | **0.9719** | +0.013 |
| cargo | 0.8327 | **0.8889** | **+0.056** |
| vehicle | 0.5622 | **0.6164** | +0.054 |
| person | 0.7223 | **0.8213** | **+0.099** |
| pallet | 0.7693 | **0.8034** | +0.034 |
| **macro** | **0.7691** | **0.8204** | **+0.051** |

**Confirmado:** quitar el multiplicador mejora F1 en TODAS las clases, incluido person (+0.099, contra-intuitivo). El plan original que añadía el multiplicador estaba equivocado.

**Causa:** sample weights × balanced sobre minoritarias muy pequeñas (person 1.8%) produce gradientes excesivos que sobreajustan el modelo a patrones específicos de esa clase en el set sintético. En CV (con StratifiedGroupKFold scene-level) los patrones no se generalizan al validation fold.

**Fix:** el flag `--class-weight-mult` se mantiene en train.py como infra opcional pero **no se usa por defecto**. Comando de training canónico de aquí en adelante:
```
python3 -u classifier/train.py --data ../output/dataset --out ../models \
    --max-samples-rf 2000000 --lgbm-only-cv --skip-rf-retrain \
    --cv-subsample 0.3 --cv-estimators 100 --n-estimators 300 --n-jobs -1
```

**Lección general:** class_weight='balanced' del propio LGBM/RF ya hace lo que hace falta. Multiplicar encima sobre minoritarias muy desbalanceadas es contraproducente — sobreajusta por gradiente excesivo, especialmente si la minoritaria sintética no representa bien las minoritarias reales.

### Estado de modelo y problemas restantes

**Modelo actual** (`models/classifier_lgbm.pkl` post-cierre):
- Features: 16 (sin `z`)
- Trained sin multiplicadores
- CV F1-macro: 0.8204 (honesto)
- Per-class CV: floor 0.97 / cargo 0.89 / vehicle 0.62 / person 0.82 / pallet 0.80
- LGBM only (no RF, saltado por OOM en máquina sin swap; el flag `--skip-rf-retrain` añadido cubre este caso)

**Problemas estructurales NO resueltos en esta sesión** (para discusión en próxima):

1. **Confusión cargo↔person en estructuras verticales reales.** El bulto BBB (caja vertical) se está dividiendo entre rojo (cargo) y verde (person) en las predicciones sobre las 6 BBB voxelizadas. Ambas clases son "geometría vertical alta" y los features actuales no las separan suficiente cuando hay ruido real. Opciones a explorar:
   - Features adicionales tipo `bbox_aspect_ratio`, `cluster_size_estimated`, `num_blobs_at_height`, etc.
   - Mejor aumentación en sintético (mayor variedad de personas/cargos)
   - Approach distinto al per-point (segmentación por instancia)

2. **Vehicle sobre-predicho** (~40-50% de la nube real). Es el problema de mesh §12 conocido — la traspaleta sintética del commit ce54270 no casa con la geometría real. La cura no es del clasificador, es del mesh o del dataset v2 con mesh mejorado.

### Nuevas memorias registradas

- `feedback_density_shift.md` — voxelizar PLYs reales a 0.035 m antes de predict para alinear densidad con sintético
- Actualización de `feedback_tree_training_cost.md` o nueva sobre el `class_weight_mult` contraproducente

### Infra committeada en esta sesión

- `src/classifier/features.py` — feature `z` eliminada (16 features totales)
- `src/classifier/train.py` — flag `--skip-rf-retrain` añadido
- `output/bench_v1_vox035_nocw/` — 6 PLYs voxelizados predichos (mejor resultado de la sesión, evidencia visual)
- `output/bench_v1_vox035_nocw_png/` — 6 PNGs preview de ese resultado
- `output/bbb_vox035/` — 6 BBB reales voxelizados a 0.035 m (input para próxima iteración)
- `results/bench_v1_clean.csv` — bench A vs C sobre v1 limpio (ejecutado en sesión previa)

---

## §16 — Segmentación de persona en datos reales (2026-04-16)

**Problema:** cuando persona y carga están a <15cm, DBSCAN (eps=0.15) las fusiona en un solo cluster. El cluster-level classifier ve un cluster grande con dimensiones de carga → proba=1.00 cargo. La persona se incluye en el cálculo de volumen.

### Approaches evaluados y descartados

**1. Per-point classifier (LGBM, 19 features)**

Ejecutado sobre Esc03, Esc06, Esc17 con voxel 0.035m. Resultados:
- Esc03: 49% de la escena clasificada como person (proba media 0.92) — la carga entera se confunde con persona
- Esc06: 18% person, pero distribuido en franjas horizontales, no localizado en la persona real
- Esc17: 10.5% person, mismo patrón difuso

**Diagnóstico:** features geométricos locales (k-NN PCA, normales, curvatura a k=20/50) no distinguen superficie vertical de caja vs torso humano. La confusión es estructural, no de threshold. Las probabilidades forman un gradiente suave, no una frontera nítida.

PLYs de diagnóstico: `output/eval_real_20esc/perpoint/`

**2. Heurística de columna vertical**

Algoritmo: proyectar en planta XZ → celdas 0.15m → detectar columnas altas (Y>1.0m, range>0.8m) → componentes conexos con footprint < 0.5m² en el borde.

Ejecutado sobre los 20 escenarios (v2 con MIN_HEIGHT=1.0m):
- Esc06: detección correcta (1,050 pts, footprint 0.135m²)
- Resto: o no detecta (persona no cumple criterios) o false positives (carga alta confundida con persona)
- La heurística funciona en ~1/20 escenarios — demasiado frágil

PLYs: `output/eval_real_20esc/heuristic/`

**3. Substracción temporal (multi-captura)**

10 capturas por escenario disponibles. Comparación Cap01 vs Cap02 (Esc06, voxel 0.05m):
- 48% de puntos tienen distancia >10cm entre capturas
- Mediana de distancia: 9.4cm — todo el ruido de registro supera el movimiento de la persona
- Los bboxes de pts "moved" y "static" se superponen completamente

**Causa:** el registro entre capturas consecutivas del mismo escenario tiene ~9cm de error (ruido sensor + calibración imperfecta entre las 3 cámaras). No es posible separar movimiento de persona del ruido.

**4. Nube cenital solamente**

Las cámaras BBB tienen nube individual por cámara (Cenital, Izq, Der) en `Capturas_BBB/2026_03_27/`.
Hipótesis: la cenital (pitch=90°, mirando hacia abajo) no vería la persona.
Verificación visual: **la cenital SÍ ve a la persona** en todos los escenarios probados (Esc03, Esc06, Esc11, Esc17) — el pitch=47.5°/90° no es suficientemente vertical para excluirla.

### Approach 5 — YOLO 2D + proyección per-camera (FUNCIONA)

**Estado: ✅ Implementado y validado (2026-04-17)**

**Setup:** YOLO v8 nano detecta persona en PNG rectificadas de cámaras laterales (Izq/Der). Los PLY per-camera están en frame de cámara con modelo pinhole verificado. Proyección 3D→2D identifica qué puntos caen dentro del bbox. El merged sin_filtro es concatenación exacta Izq→Der→Cenital, así que se eliminan por índice.

**Hallazgos clave:**
- **No se necesitan extrínsecas de Paula.** Los intrínsecos se estimaron por color-matching PLY↔PNG con optimización Nelder-Mead (91% accuracy):
  - Izq: fx=fy=1680, cx=1045, cy=758 (2048×1536)
  - Der: fx=fy=1664, cx=1013, cy=750 (2048×1536)
  - Cenital: fx=fy=833, cx=517, cy=382 (1024×768)
- Intrínsecos 100% estables entre los 19 escenarios
- YOLO detecta persona en los 19 escenarios (conf 0.51–0.88 en laterales)
- Bbox persona NO solapa con carga en ningún escenario (verificado visualmente)
- Merged sin_filtro = concat exacta Izq+Der+Cenital (verificado por conteo y RGB boundary)

**Resultado (E07):** 214,749 pts eliminados (6.2%), persona correctamente identificada, carga intacta.

**Script:** `scripts/filter_person_yolo.py` (~300 líneas, self-contained)

**Limitación:** si persona se interpone delante de carga (oclusión), el bbox capturaría puntos de carga. En los 19 escenarios actuales no ocurre.

### Análisis: clasificador per-punto en nubes raw (2026-04-17)

**Hallazgo:** las nubes per-camera raw tienen densidad 8.5× mayor que las filtradas (NN distance 3.1mm vs 26.5mm). A esta densidad las features locales (normales, curvatura, PCA) capturan pliegues de ropa y curvatura corporal que antes eran invisibles.

**Resultados con pseudo-labels YOLO (leave-one-escenario-out, 6 escenarios):**

| Feature set | F1 cargo | F1 person | Notas |
|---|---|---|---|
| 8 features geométricas | 0.785 | 0.804 | Solo PCA eigenvalue ratios |
| 10 feat (geo + normal_y_std + height) | 0.849 | 0.858 | Sweet spot |
| 13 feat (+ RGB) | 0.871 | 0.884 | RGB aporta +2pp |

**Comparación con nubes filtradas:** F1 pasó de 0.655 → 0.85 (10 feat). La densidad es el factor clave.

**Caso difícil:** E01 (cargo retractilado negro, superficie lisa similar a ropa) baja F1 a ~0.70. YOLO sigue funcionando ahí.

**Decisión:** entrenar clasificador per-punto sobre datos reales pseudo-etiquetados por YOLO (sin sintéticos, sin domain gap). Complementa a YOLO como fallback cuando no hay PNG.

### Datos de capturas per-camera

- `Capturas_BBB/2026_03_27/` contiene nubes por cámara individual (PLY_Cenital/Izq/Der) + PNG rectificadas + PGM disparidad + config_base.ini para los 19 escenarios (01-19)
- `Capturas_BBB/2026_03_23/` contiene estructura similar con nombres largos (BBB25503007_izq, etc.) + FUSION3D merged
- `tri_cloud_sin_filtro.ply` tiene ~3.4M pts con RGB (vs ~12k el filtrado) — la fusión descarta 99.6% de los puntos
- config_base.ini: 3 cámaras, Cenital (serial 25461175, h=3.68m, pitch=90°), Izq (25503007, h=3.80m, pitch=47.5°), Der (25503005, h=3.80m, pitch=47.5°)

### Scripts

- `scripts/filter_person_yolo.py` — filtrado persona YOLO 2D per-camera (FUNCIONA, producción)
- `scripts/test_person_heuristic.py` — heurística columna vertical (descartado)
- `scripts/test_person_2d.py` — prototipo YOLO (reemplazado por filter_person_yolo.py)
- `scripts/validate_synthetic.py` — validación per-point en sintético
