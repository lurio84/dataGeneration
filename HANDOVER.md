# HANDOVER — Logicarc (datageneration + cargo_segmentation)

Documento de cierre de iteración (2026-04-23). Complementa a [`DECISIONS.md`](DECISIONS.md) y al [`README.md`](README.md) del repo C++ (`logicarc_cargo_segmentation`): no reescribe lo que ya está ahí, sino que resume **estado vivo**, **estrategias abiertas con recomendación**, y **callejones sin salida que NO hay que volver a recorrer**.

Lectura mínima previa: `DECISIONS.md` §15 (diagnóstico LGBM real), §16 (persona), §17 (pipeline end-to-end), §18 (carretilla).

---

## 0. Contexto de producto (para nuevos incorporados)

- **Producto: INESARC** — arco logístico (ceiling-mounted, no arco físico) que mide volumen + peso + etiqueta de bultos al pasar por debajo. Cliente: **IOVI** (iNES OPTICS, SL). Desarrollo: **CATEC**. CATEC cubre **solo medida volumétrica**.
- **Dos configuraciones de captura** soportadas:
  - **BBB config** — 3 cámaras estéreo ceiling-mounted (Cenital pitch=90° + Izq/Der pitch=47.5° a ~3.8 m). Bultos retractilados (negro, transparente, etc.).
  - **Helios config** — 4 cámaras ToF floor-mounted.
- **Restricción metrológica del cliente (meeting abril 2026):** la base del OBB debe clamparse al mínimo **1.2 m × 0.8 m** (pallet EUR). Si la base medida es menor, es error de medición, no bulto menor. **Acción A48 pendiente — no implementada.**
- **Precisión sensor FUSION3D** (`reference_sensor_precision.md`): σ en banda estrecha 9–11 mm; banda realista 15–20 mm; distribución **no Gaussiana, colas pesadas**. Voxel/celda mínimo defendible **40 mm**. σ depende de profundidad y ángulo (izq 4.6 m tilt 42° → σ_axial ≈ 41 mm; cenital 3.5 m frontal → σ_perp ≈ 16 mm).
- **Acciones CATEC pendientes** (del último meeting):

  | Acción | Descripción | Estado |
  |---|---|---|
  | A48 | OBB base clamp 1.2 × 0.8 m + mejora cálculo volumen | ❌ No implementada |
  | A52 | Compartir código en BitBucket | ✅ Parcial (developLucas pusheado) |
  | A54 | Mejora algoritmo segmentación | 🟡 En progreso (§16/17) |
  | A57 | Medir tiempos de procesado por algoritmo | ❌ Pendiente |
  | A58 | Correr pipeline sobre 20 nubes BBB nuevas (IOVI 23-mar) | 🟡 Parcial (~50% éxito, §17) |
  | A60 | Compartir última parte del algoritmo en BitBucket | ❌ Pendiente |

- **Equipo** (a la hora de cerrar): Alejandro M. Prado (PM), Manuel Correa (SW senior), Jose y Paula (CATEC — Paula hace medida + calibración), David y Paloma (IOVI).
- **Bloqueador transversal:** **calibración extrínseca cámara→mundo no está en los repos accesibles** — el driver propietario `BBBDriverConsole` (máquina de Paula) las computa en runtime. Faltan matrices 4×4, intrínsecos estéreo (fx/fy/cx/cy, baseline), yaw. Recuperados empíricamente vía color-matching (91% accuracy, §16) pero debería pedirse formalmente a Paula.

---

## 1. Estado actual (abril 2026)

### 1.1 `logicarc_cargo_segmentation` (C++)

Baseline maduro. Pipeline clásico (RANSAC floor → Euclidean clustering → Valley/Watershed split → edge-ratio → scoring) con dos configuraciones: `params.yaml` (nubes BBB pre-filtradas T01–T10) y `params_fusion3d.yaml` (nubes crudas fusionadas). **12/12 tests de regresión pasan**, OBB por ConvexHull + rotating calipers.

RF per-point v5 (`models/point_classifier_v5.pkl`, 17 features) complementa el pipeline geométrico: **F1 = 0.921**, 17/20 escenarios validados.

Punto de entrada: `build/select_cargo <cloud.ply>`. Librería integrable vía `ClusterSelector::selectCargo()`.

### 1.2 `datageneration` (Python, rama `developLucas`)

Generador sintético FUSION3D + clasificador ML 5-clases (floor/cargo/vehicle/person/pallet) + pipeline de extracción de carga sobre datos reales BBB.

- **Dataset v1 limpio**: 100 escenas, ruido calibrado contra FUSION3D real (NN spacing 49.9 mm vs 51.3 mm real, §7).
- **Clasificador activo**: `models/classifier_lgbm.pkl`, 19 features, CV F1-macro ≈ 0.82 honesto sobre v1 limpio.
- **Filtrado persona**: YOLOv8 2D per-cámara (§16) — funciona en 29/30 capturas con conf 0.83–0.91, no requiere extrínsecas de Paula (intrínsecos descubiertos por color-matching, 91% accuracy).
- **Pipeline end-to-end extracción carga** (§17): **~50% éxito visual en 15 escenarios BBB**. Cerrado como baseline operativo.
- **Carretilla elevadora** (§18): STL del cliente integrado como vehículo alternativo a la traspaleta, mutex por escena vía `--p-forklift`.
- 92/92 tests passing. Último commit: `fc64f98`.

---

## 2. Problemas abiertos vivos

### 2.A — Anchor mal colocado (§17 final)

Escenarios afectados: **Esc09, Esc10, Esc13**.

`find_anchor` escoge un bulto vertical que no es la carga. Fallo **upstream**: si el anchor está mal, nada de lo que venga después importa. No se resuelve con scoring.

### 2.B — Cargo + traspaleta fusionados en el mismo cluster (§17 final)

Escenarios afectados: **Esc07, Esc08**.

El slice dentro del anchor no separa cargo de traspaleta antes de clusterizar. DBSCAN `eps=0.08` los une; `pallet.py` devuelve 0 candidatos y por tanto no se puede restringir cargo por la detección del pallet.

### 2.C — Confusión cargo↔person en estructuras verticales reales (§15 problema A)

En BBB voxelizado (0.035 m) el bulto real se parte entre rojo (cargo) y verde (person). Ambas son "geometría vertical alta" y los features per-point (PCA k=20/50 + altura) no las discriminan con ruido real. La clase person se sobrepredice a ~18% ROI (vs 1.4% training).

### 2.D — Vehicle sobre-predicho (~40–50%) por mesh §12 (§15 problema B)

La traspaleta sintética (cuerpo bajo + horcas planas + timón en U) no casa con la geometría real del cliente. El clasificador usa la clase vehicle como catch-all cuando no sabe. Problema de dataset/mesh, no de features.

### 2.E — Integridad de datos

- `Esc04/Cap01/PNG_Der.png` corrupto (libpng).
- `Esc06/Cap01/PLY_Izq.ply` truncado al 18%.

Re-captura a pedir a Paula.

### 2.F — ROI localization es el verdadero cuello de botella del height-field (`project_height_field_pipeline.md` + `feedback_roi_localization_fragile.md`)

Validado 2026-04-20 sobre 24 capturas T01–T10 con GT explícito.

El pipeline height-field 2.5D funciona bien sobre bultos grandes (≥2 m³): **median |err| ≈ 11–13%**, 10/12 cámaras en ±15% — comercialmente útil. Sobre bultos pequeños (~0.13 m³): **median |err| = 300%, irrecuperable** con los recursos actuales del repo.

**Causa:** density-peak + ROI ±0.7 m para anclar el height-field **no generaliza** bajo clutter. Escenas con distractores a altura de cargo (persona, mango de traspaleta, estanterías) tienen picos XY más densos que la propia carga → el ROI se ancla en el clutter, no ve la carga. Para cargos <1 m³ adyacentes a persona+jack, error ~300–1000%.

Ninguno de los filtros 2D ensayados arregla esto (ver §4). El problema es "dónde está la carga", no "qué puntos contar".

### 2.G — Calibración extrínseca pendiente de Paula

Bloqueador transversal. El approach YOLO 2D de §16 **no la necesita** para filtrar persona (intrínsecos recuperados por color-matching). Pero sí es necesaria para:

- Hard-coded ROI per camera para destrabar 2.F (height-field ROI).
- Backprojection precisa PLY↔PNG sin error 10–20% píxel.
- Fusión per-cámara con prior de posición.

Pedir formalmente: matrices 4×4 cámara→mundo, fx/fy/cx/cy estéreo, baseline, yaw.

### 2.H — Medida del volumen es un segundo frente abierto (separado de la segmentación)

Aun con segmentación perfecta de la carga, el **cómo se calcula el volumen** tiene dos opciones en tensión:

- **OBB ConvexHull + rotating calipers** (activo en `logicarc_cargo_segmentation`) — 10–25% más ajustado que MOI, 12/12 tests T01–T10. **Sobreestima 5–20× cuando el cluster viene fragmentado** en varios DBSCAN (el bounding box abraza los gaps). Inflamiento sistemático en cargos no-caja (+27% cilindros).
- **Height-field 2.5D per-camera** (`feedback_height_field_volume.md`, `project_height_field_pipeline.md`) — agnóstico a forma superior (funciona para sacos, cilindros, pilas escalonadas, cargas fragmentadas). Factor **12–20× de mejora sobre OBB-of-union** en BBB voxelizado (6 escenas: 24.2 → 4.1 m³, 31.6 → 4.3 m³, etc.). Limitación: bloqueado en bultos pequeños por 2.F (ROI), no por el agregador de alturas.
- **A48 (clamp min 1.2×0.8 m)** es ortogonal a ambos — es un post-procesado contractual.

---

## 3. Estrategias — análisis con trade-offs

### Para 2.A + 2.B (bugs de clustering downstream del YOLO)

| Opción | Qué resuelve | Coste | Riesgo |
|--------|--------------|-------|--------|
| **Spike 30' sobre `pallet.py`** ⭐ | Si el pallet se detecta robustamente, el anchor se ancla en una pieza universal (generaliza a otros almacenes). | 30 min exploración + 2–4 h si viable | Hoy devuelve 0 candidatos; puede no ser rescatable |
| Mejorar scoring del anchor (heurística por dimensiones esperadas) | 2.A | 1 día | Frágil: cada almacén tiene cargas distintas |
| DBSCAN multi-escala en el slice | 2.B | 1–2 días | Costo computacional, no ataca 2.A |
| Aceptar baseline 50% y flaggear outliers | — | 0 | Tope de producto |

**Recomendación:** spike de `pallet.py` primero (30 min decidirán si es vía muerta o no). Si da señal → invertir 2–4 h en endurecerlo. Si no → caer al plan de aceptar baseline + outlier flagging, y mover esfuerzo a 2.C/D.

### Para 2.C (cargo↔person vertical)

| Opción | Cómo ataca | Coste | Riesgo |
|--------|------------|-------|--------|
| Feature `verticality_at_large_scale` (PCA k≈200) | Captura "anchura" del objeto sin salir de per-point | Medio | Puede no ser suficiente — la diferencia real es aspect ratio del cluster |
| **Per-point sobre nubes raw per-cámara** ⭐ | Ya validado en §16: F1 cargo 0.871 / person 0.884 con 13 features sobre densidad 8.5× mayor (pliegues de ropa visibles) | Medio | Requiere tener per-camera disponibles en producción |
| Cambio de paradigma: instance segmentation (PointNet++) | Ataca el problema de raíz | Alto | Fuera de scope para iteración corta |
| Augmentación sintética de personas (posturas, alturas, ropa) | Más diversidad | Alto (regen v2) | Requiere 2.D resuelto antes |
| Threshold post-hoc cargo > person | Hack rápido | Bajo | Sesga, no resuelve |

**Recomendación:** entrenar per-point sobre nubes raw con pseudo-labels YOLO (§16, ya validado F1≈0.88). Es lo que más F1 aporta por hora invertida.

### Para 2.D (vehicle over-prediction)

| Opción | Ventaja | Riesgo |
|--------|---------|--------|
| **Regenerar mesh traspaleta con geometría más genérica/estándar** ⭐ | Ataca la causa raíz, reutilizable | 1 día de modelado + regen parcial |
| Mezcla de múltiples geometrías de vehicle en v2 | Robustez por diversidad | Requiere modelar varias |
| Eliminar clase vehicle (unir a outlier) | Simplifica | Cambia semántica del producto |
| Filtrar post-hoc por bounding box de zona vehicle | Parche operativo | Frágil entre almacenes |

**Recomendación:** regenerar la mesh de traspaleta con medidas estándar EUR (longitud horcas 1.15 m, ancho 0.52 m, altura cuerpo 1.20 m si hay timón). Es lo mínimo para desbloquear v2.

### Para 2.H (volumen: OBB vs height-field)

| Opción | Ventaja | Riesgo |
|--------|---------|--------|
| **Migrar a height-field 2.5D como método primario** ⭐ | Agnóstico a forma y fragmentación, 12–20× más ajustado en BBB voxelizado, matches la métrica comercial real del cliente (volumen de transporte = footprint × altura) | Limitado en bultos pequeños hasta que 2.F se resuelva (ROI prior externo) |
| Mantener OBB ConvexHull y añadir A48 (clamp 1.2×0.8 m) | Cero disrupción, cumple petición contractual del cliente | Sigue sobreestimando en fragmentación y +27% en cilindros |
| Ensemble: min(OBB, height-field) por escena | Aprovecha lo mejor de cada uno | Complica auditoría; elegir ad-hoc dificulta justificar ante cliente |

**Recomendación:** height-field 2.5D como método primario; A48 se aplica como post-processing encima (clamp de base mínima). Documentar cambio con Paula antes de desplegar — afecta la "ConvexHull" que ella reporta hoy (que, según `feedback_paula_convexhull_aabb.md`, es realmente AABB).

### Para 2.F (ROI localization — bloqueador crítico de bultos pequeños)

Lecciones de 2026-04-20 — **ningún filtro 2D/3D interno arregla esto** (ver §4 callejones sin salida 13–16). Soluciones que sí generalizan:

| Opción | Esfuerzo | Valor |
|--------|----------|-------|
| **Extrínsecas de Paula → ROI hard-coded por cámara** ⭐ | Bajo (pedir + integrar) | Máximo: cierra el problema de raíz |
| Template matching 1.2×0.8 m a h=0.15 m (shape prior, no plane prior) | Medio | Generaliza a almacenes con pallet EUR estándar |
| Custom pallet detector entrenado sobre sintético con ruido físico nuevo (§7) | Alto | Generaliza a pallets no-EUR si se amplía sintético |
| ROI manual por usuario (MVP) | Mínimo | Válido como producto hasta resolver lo anterior |

**Recomendación:** pedir extrínsecas a Paula. Todo lo demás es plan B. Sin extrínsecas, bultos pequeños quedan fuera del spec comercial actual del height-field.

### Decisión transversal: ¿generar dataset v2 (≥500 escenas) ahora?

**NO, todavía no.** Motivos:

1. v2 heredaría 2.C y 2.D si se lanza sin tocar features/mesh. 3–4 h de CPU tiradas.
2. El ROI mayor hoy viene del per-point sobre nubes raw per-cámara (§16), que **no necesita v2** para mejorar el F1.
3. Paula ya dio OK para v2, pero antes hay que cerrar mesh de traspaleta v2 (2.D) y validar una iteración v1.5 (+100 escenas con mesh nuevo) sobre BBB real.

**Orden recomendado:**

1. **Pedir extrínsecas a Paula** (email, coste 0). Destraba 2.F, 2.G, y reduce error de backprojection en YOLO 2D.
2. Spike `pallet.py` (30 min). Destraba 2.A + 2.B si da señal.
3. Entrenar per-point sobre raw per-cámara con pseudo-labels YOLO (§16). Medir F1 sobre holdout. Si F1 convence → modelo de producción; 2.C neutralizado.
4. Re-medir NN density sintético↔real tras el nuevo modelo de ruido físico (`project_noise_calibration_pending.md`). El ratio histórico 3.2× era artefacto del σ=35 mm viejo; el nuevo (σ_core=10 mm axial) puede haber cerrado o incluso invertido el gap. `analyze.py`.
5. Rehacer mesh traspaleta (2.D). Regenerar v1.5 (100 escenas, mismo seed). Retrain LGBM y comparar con v1 sobre BBB.
6. Si v1.5 mejora → lanzar v2 (500 escenas) con mesh nuevo + ruido nuevo.
7. Migrar volume estimation a height-field 2.5D con A48 como post (§3 opción 2.H).
8. Validación final con Paula sobre 6–10 escenarios.
9. Cumplir acciones A52/A57/A60 pendientes (compartir código, medir tiempos, entrega final).

---

## 4. Callejones sin salida — NO repetir

Lista priorizada de cosas que se probaron, no funcionaron, y tienen razón documentada para no volver a intentarse. **Leer antes de reabrir cualquier discusión sobre estos puntos.**

1. **Floor subsampling en training** (§13). Intuición "70% de puntos = 70% de tiempo" es falsa en árboles: el tiempo lo marca el boundary difícil, no el volumen homogéneo. Ahorra ~90 s de CV pero cuesta −0.025 en F1-macro, con pallet y vehicle como grandes perdedores. Default `--floor-subsample none`.

2. **`--class-weight-mult person:2,pallet:2`** (§15 hallazgo 3). Multiplicar encima de `class_weight='balanced'` sobre minoritarias muy desbalanceadas (person 1.8%) baja F1 en **TODAS** las clases. Person −0.099 contra-intuitivo. Quitarlo sube macro +0.051. Flag existe en `train.py` pero default vacío; no activar.

3. **Feature `z` (depth raw)** (§15 hallazgo 1). Era leakage posicional — producía bandas paralelas a ejes en predicciones reales. "Proxy de ruido" falso: el ruido dependiente de distancia ya lo capturan `local_density`, `nbr_y_std`, `normal_y_std`, `planarity`. Regla general: **ninguna feature posicional cruda** (`x`, `y`, `z`, `dist_*`) en clasificadores per-point sobre datasets sintéticos centrados. Salvo `y` absoluta por su correlación física con clase.

4. **Sustracción temporal multi-captura para separar persona** (§16 approach 3). Ruido de registro entre capturas consecutivas ~9 cm — supera el movimiento humano. Los bboxes de puntos "movidos" y "estáticos" se solapan completamente. No es viable con el sensor actual.

5. **Heurística de columna vertical para persona** (§16 approach 2). Funciona en 1/20 escenarios. Demasiado frágil: o no detecta (persona no cumple Y>1.0 m + footprint <0.5 m²) o falsos positivos con carga alta.

6. **Fine-tune YOLOv8n sobre BBB** (§17 corrección diagnóstica). El diagnóstico inicial "YOLO falla en ángulos BBB" era **erróneo**. Re-ejecutando COCO-pre-trained sobre las 30 capturas problemáticas detecta persona con conf 0.83–0.91 en 29/30. Fine-tune no resuelve nada; los fallos son downstream (clustering), no de detección.

7. **`predict.py` sobre `colored_clouds/*.ply` (crops pre-segmentados)** (§11 validación V1). 69% clasificado como person. Los crops son 1,5k pts con Y∈[-0.27, 1.15 m], sin suelo ni contexto — el clasificador fue entrenado sobre escenas enteras 5×4 m. **Input correcto: `Resources/Capturas_BBB/*/FUSION3D/fusion3d_merged_*.ply`**.

8. **`run_geometric.py --preprocessed` (usar tri_cloud pre-procesadas de Paula)** (§17). Inutilizable para detección de pallet: la banda del pallet queda con 5 pts en todo el anchor. Usar **`tri_cloud_sin_filtro`** + pipeline completo con RANSAC.

9. **Cluster classifier ML (negative_filter) sobre-incluye carga** (§17). Sin clase persona explícita retiene 19/22 clusters. Volver a rank-0 legacy es más robusto para este objetivo.

10. **Experimento no-floor (entrenar sin clase suelo)** (§12). Person −0.093 — pierde personas reales en esc4. Trade-off inaceptable para producción aunque mejore pallet +0.080. Decisión cerrada: **with_floor** (§13).

11. **Cono / L-shape como primitivos de cargo** (§3). Cono anecdótico sin caso de uso en capturas. L-shape no lo requieren los datos reales aún.

12. **`tracemalloc` para decidir OOM de RF** (§13 lección 4). Subestima RAM (2.3 GB reportados vs 4.3 GB reales con `time -v`). No fiable. Usar `--max-samples-rf` + `--skip-rf-retrain` en máquinas sin swap.

13. **YOLO-seg COCO `person` filter + backproyección para arreglar ROI del height-field** (`feedback_roi_localization_fragile.md`, 2026-04-20). Mejora +1.4% en bultos grandes y **−12% en pequeños** — peor. La persona NO es el distractor dominante del ROI: la **traspaleta amarilla** lo es, y COCO no la tiene como clase. No volver a probar.

14. **HSV amarillo máscara para traspaleta + backproyección** (id.). Elimina puntos del jack pero **no mueve el centro del ROI**, que es lo que importa. Los puntos eliminados estaban en sitios que el height-field ya ignoraba. No mejora.

15. **Multi-slice centroid (tall → mid → low) para re-anclar el ROI** (id.). Arregla T05/T07 diagonal-der pero rompe otros grandes y empeora todos los pequeños. No generaliza.

16. **RANSAC pallet plane / occupancy mask a altura del pallet para anclar ROI** (id.). El ruido σ=15–20 mm con colas pesadas contamina la banda 0–25 cm lo suficiente para que los "inliers del plano pallet" cubran 10–30 m² de falso suelo. El noise ceiling inunda cualquier detección 3D que intente aislar el pallet del suelo sin prior externo. Los filtros 2D cortan clutter pero no te dicen DÓNDE está el pallet.

17. **ConvexHull de Paula (`Resources/volume_estimation/get_bounding_box`) como ground truth de volumen** (`feedback_paula_convexhull_aabb.md`). El ejecutable C++ etiqueta "ConvexHull" pero el valor coincide con **AABB**: Esc06/Cap01 Paula=5.08 m³, AABB=5.11 m³, Open3D ConvexHull real sobre los mismos puntos=2.50 m³. La implementación PCL sobre nubes cáscara-hueca parece envolver el bbox, no seguir la superficie. Usar solo como referencia de segmentación (qué puntos pertenecen a qué cluster), nunca como GT de volumen.

---

## 5. Referencias clave

| Documento | Qué contiene |
|-----------|--------------|
| [`DECISIONS.md`](DECISIONS.md) | Log técnico exhaustivo de decisiones §1–§18. **Fuente de verdad.** |
| [`CLAUDE.md`](CLAUDE.md) | Comandos canónicos, estructura `src/`, labels, restricciones. |
| [`README.md`](README.md) | CLI, flags, pipeline, formato de salida. |
| [`docs/ADDING_ASSETS.md`](docs/ADDING_ASSETS.md) | Cómo añadir nuevos STLs (aplicable para mesh traspaleta v2). |
| [`../logicarc_cargo_segmentation/README.md`](../logicarc_cargo_segmentation/README.md) | Pipeline C++ complementario. |

Plan histórico desactualizado (solo contexto, NO vigente): `~/.claude/plans/floofy-rolling-pumpkin.md` — precede a §16/17/18; útil únicamente para entender el árbol de opciones de problemas 2.C y 2.D.

## 6. Comandos de entrada rápida

```bash
# Tests
cd src && python3 -m pytest -v   # 92/92

# Generar dataset v1 (reproducible)
cd src && python3 generate_dataset.py --n 100 --seed 42

# Training canónico (LGBM, sin OOM)
cd src && python3 -u classifier/train.py --data ../output/dataset --out ../models \
    --max-samples-rf 2000000 --lgbm-only-cv --skip-rf-retrain \
    --cv-subsample 0.3 --cv-estimators 100 --n-estimators 300 --n-jobs -1 --seed 42

# Predict sobre BBB real (voxelizar antes a 0.035 m manual — ver plan)
cd src && python3 classifier/predict.py input_vox035.ply output.ply --model lgbm --align auto

# Pipeline completo extracción de carga real
python3 scripts/filter_person_yolo.py <escenario>     # filtra persona por 3 cámaras
python3 scripts/run_geometric.py <merged.ply>         # extracción geométrica end-to-end

# C++ pipeline
cd ../logicarc_cargo_segmentation/build && ./select_cargo <cloud.ply> output_cargo.ply
```

---

*Documento de handover. Última actualización: 2026-04-23.*
