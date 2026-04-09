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
- **Acción pendiente:** ejecutar el dataset definitivo (≥100 escenas).

---

## 3. ¿Qué otros primitivos de Open3D tienen sentido?

**Estado: ❓ Pendiente decidir**

- **Ya implementados:** caja (`create_box`), pallet (caja), traspaleta (3 cajas), persona (cilindro + esfera).
- **Candidatos a añadir:**
  - Cilindro → bobinas, bidones, depósitos
  - Cono → señales de advertencia de almacén
  - Caja irregular (L-shape con dos boxes solapados) → cargas no estándar
  - Esfera → sin uso claro en logística, descartada en principio
- **Pendiente:** decidir cuáles se implementan antes del dataset final.

---

## 4. ¿Queremos simular suelo?

**Estado: ✅ Decidido — activo**

- **Implementación:** `sample_floor()` activa, controlada por `enable_floor=True` en CFG.
- **Extents asimétricos:** floor_extent_x=2.5m, floor_extent_z=2.0m → span 5.0×4.0m, calibrado al FOV real (X≈5.85m, Z≈4.4m).
- **Color:** gris oscuro (60,60,60) para que retroceda visualmente en CloudCompare sobre fondo oscuro.

---

## 5. Función para combinar primitivos en casos complejos

**Estado: ❓ Pendiente implementar**

- **Existe parcialmente:** lógica de `box2` con modos `stacked` / `adjacent` (deshabilitada con `p_two_boxes=0.00`).
- **Objetivo:** función genérica que reciba una lista de primitivos con posición relativa y los combine en una escena — por ejemplo solapar dos cajas, apilar tres, carga irregular.
- **Casos de uso:** carga compleja, múltiples bultos sobre un palé, cargas asimétricas.

---

## 6. Búsqueda de recursos mesh 3D (.stl / .obj)

**Estado: ❓ Pendiente buscar**

- **Situación actual:** `forklift.stl` referenciado en el código pero no presente en `data/`; el sistema cae al primitivo de traspaleta automáticamente.
- **Recursos a buscar:**
  - Vehículos: carretilla elevadora, transpaleta, apilador
  - Cargas: cajas reales, objetos voluminosos
  - Persona: mesh realista para reemplazar el primitivo cilindro+esfera
- **Fuentes candidatas:** GrabCAD, Thingiverse, BlenderKit, repositorios GitHub de datasets 3D industriales.

---

## 7. Calibración de ruido vs sensor real

**Estado: ✅ Calibrado (2026-04-09)**

- **Método:** análisis cuantitativo sintético vs real con `analyze.py` (6 escenas FUSION3D reales, ROI crop X:±2.5m Y:-0.15..2.5m Z:±2.0m para comparación justa).
- **Parámetros finales:**
  - noise_std=0.030m (Gaussiano por punto)
  - voxel_size=0.019m (downsampling; calibrado a NN spacing real)
  - dropout_ratio=0.15, outlier_ratio=0.03, local_outlier_std=0.055m
  - Falloff de densidad ∝ 1/d² desde cámaras
- **Resultados validados:**
  | Métrica | Sintético | Real FUSION3D | Estado |
  |---|---|---|---|
  | NN spacing | 47.5mm | 51.3mm | ✅ |
  | Floor roughness σ | 24.8mm | 29.8mm (gap 5mm) | ✅ |
  | Footprint X | 5.29m | 5.00m | ✅ |
  | Footprint Z | 4.23m | 3.97m | ✅ |
- **Nota:** la roughness en banda de cargo (Y=0.2–1.6m) no es una métrica válida — mide variación geométrica de caras laterales de la caja, no ruido del sensor.
- **Pendiente con Paula:** validar patrones de oclusión y reflexiones especulares no cubiertos por el modelo Gaussiano.

---

## 8. Generar dataset final

**Estado: 🟡 Listo para ejecutar — pendiente decidir N escenas y primitivos extra**

- Pipeline completo y calibrado. Pendiente antes de lanzar:
  - Decidir N escenas (mínimo 100, recomendado 500+)
  - Decidir si añadir primitivos adicionales (sección 3) antes del dataset final
  - Validación oclusión/reflexiones con Paula (sección 7)
- Comando: `cd src && python3 generate_dataset.py --n 500 --seed 42`
