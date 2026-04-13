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

- **Dataset v1 (validación):** 100 escenas iniciales, seed=42. `output/dataset/` acumula actualmente 139 PLYs de iteraciones previas.
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
- [ ] **Decidir estrategia no-floor para v2** — ver §12 para análisis completo
- [ ] Dataset v2 ≥500 escenas — Paula OK obtenido, pendiente tras decisión §12
- [ ] Retrain RF+LGBM sobre dataset v2 con 16 features
- [ ] Validación visual post-retrain sobre 6 escenas BBB (comparar con output/fase0/)

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

**Decisión pendiente:** analizar opciones con más profundidad antes de generar dataset v2.
