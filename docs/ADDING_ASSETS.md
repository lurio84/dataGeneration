# Guía: Añadir un nuevo asset 3D al generador

Esta guía describe el workflow completo para incorporar un STL nuevo (vehículo, elemento de escena u objeto de referencia) al generador sintético. Está escrita para que un compañero sin contexto previo pueda completar la tarea.

---

## Convenciones del frame de mundo

El generador usa el siguiente sistema de coordenadas. **Tu STL debe respetar estas convenciones** (o transformarlo en el loader).

| Eje | Dirección | Notas |
|-----|-----------|-------|
| X | Derecha | Ancho de la escena. Pallet centrado en X=0. |
| Y | Arriba (altura) | Suelo en Y=0. Objetos apoyados con Y_min=0. |
| Z | Profundidad hacia cámaras | Z=0 en origen del pallet. +Z apunta a las cámaras. |

**Unidades: metros.**  
Origen del vehículo: fork-root en (0, 0, 0). Horquillas en +Z, cuerpo en −Z.

> ⚠ Los PLY reales del sensor FUSION3D usan Z como altura. `analyze.py` hace el remap automáticamente. Los STL sintéticos se generan con Y-up.

---

## Paso 1 — Obtener el STL

- Pide el archivo al proveedor o exporta desde CAD/Blender.
- Si supera ~5 MB, reduce la geometría (decimation en Blender/MeshLab) para no inflar el repo.
- Anota las unidades declaradas por el origen del fichero (mm, cm, m).

---

## Paso 2 — Inspeccionar bbox y unidades

Ejecuta el siguiente snippet desde `src/` (no commitear):

```python
import open3d as o3d, numpy as np
stl = "/ruta/al/asset.stl"
mesh = o3d.io.read_triangle_mesh(stl)
v = np.asarray(mesh.vertices)
ext = v.max(axis=0) - v.min(axis=0)
print(f"BBox min : X={v[:,0].min():.3f}  Y={v[:,1].min():.3f}  Z={v[:,2].min():.3f}")
print(f"BBox max : X={v[:,0].max():.3f}  Y={v[:,1].max():.3f}  Z={v[:,2].max():.3f}")
print(f"Extent   : X={ext[0]:.3f}  Y={ext[1]:.3f}  Z={ext[2]:.3f}")
print(f"Centroid : X={v[:,0].mean():.3f}  Y={v[:,1].mean():.3f}  Z={v[:,2].mean():.3f}")
# Heurística unidades
mn = ext.min()
print("Unidades probables:", "metros" if mn < 5 else "mm" if mn < 5000 else "revisar")
```

Con los valores decide:
- `scale`: `0.001` si mm, `0.01` si cm, `1.0` si metros.
- ¿El eje de altura es Y o Z? Necesitas rotación si es Z-up.
- ¿Qué extremo del eje Z tiene las horquillas/frente?

---

## Paso 3 — Copiar el STL al repo

```
datageneration/data/<nombre>.stl
```

El `.gitattributes` ya marca `*.stl binary`. Commit en `developLucas`.

---

## Paso 4 — Crear el loader en `geometry/meshes.py`

Añade tras los loaders existentes (`load_forklift`, `load_carretilla`) un nuevo loader siguiendo el patrón:

```python
def load_mi_asset(stl_path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.scale(SCALE, center=(0.0, 0.0, 0.0))   # unidades → metros
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    # Centrar X (si el STL no está centrado en X=0)
    x_center = (v[:,0].max() + v[:,0].min()) / 2.0
    # Y: base en suelo (Y_min → 0)
    y_min = v[:,1].min()
    # Z: fork/front root en world Z=0
    #   Si frente del asset está en Z_max: z_translate = -(Z_max - FORK_L)
    #   Si frente del asset está en Z_min: z_translate = -Z_min  (sin flip)
    z_translate = ...
    mesh.translate([-x_center, -y_min, z_translate])
    return mesh
```

Exporta las constantes geométricas derivadas del bbox (mismo patrón que `FORKLIFT_*`):

```python
MI_ASSET_FORK_H:  float = ...   # altura de la plataforma / horquillas (m)
MI_ASSET_FORK_L:  float = ...   # longitud horquillas en +Z
MI_ASSET_BODY_D:  float = ...   # profundidad del cuerpo en −Z
MI_ASSET_X_HALF:  float = ...   # semi-ancho (X_extent / 2)
```

---

## Paso 5 — Integrar en `generate_scene()` / `_place_vehicle()`

Opciones comunes:

**A) Nuevo vehículo mutuamente exclusivo con los existentes**  
Amplía el branching en `generate_dataset.py` donde se selecciona el tipo de vehículo (bloque `is_forklift`). Añade una probabilidad nueva en `CFG` y `parse_args`.

**B) Elemento adicional de escena (no vehículo)**  
Añade un bloque nuevo en `generate_scene()` con su probabilidad, posicionamiento (usando `translate`) y `sample_labeled(mesh, LABEL["..."], n_pts)`. Crea un nuevo label en `ply_io/ply.py` si necesitas clase nueva.

En ambos casos, pasa las constantes geométricas (`FORK_H`, `FORK_L`, `BODY_D`, `X_HALF`) a `compose_cargo_on_vehicle` si el asset puede llevar carga encima.

---

## Paso 6 — Añadir el flag CLI y la clave en `CFG`

En `generate_dataset.py`:

```python
CFG = {
    ...
    "p_mi_asset": 0.0,
    "mi_asset_stl": "../data/mi_asset.stl",
}
```

En `parse_args()`:

```python
p.add_argument("--p-mi-asset",   type=float, default=cfg["p_mi_asset"])
p.add_argument("--mi-asset-stl", type=str,   default=cfg["mi_asset_stl"])
```

En `run_generation()`: carga el mesh antes del bucle y pásalo a `generate_scene()`.

---

## Paso 7 — Tests

En `test_pipeline.py`, crea una clase `TestMiAsset(unittest.TestCase)` con:

- `test_load_bbox`: Y_min≈0, dimensiones en rangos físicos plausibles.
- `test_scene_p1_has_label`: con `p_mi_asset=1.0` → label presente, meta correcto.
- `test_default_p0_no_asset`: con `p_mi_asset=0.0` → asset no aparece.
- `test_mutex_with_other_vehicles` (si aplica): un solo vehículo por escena.

Ejecutar desde `src/`:

```bash
python3 -m pytest test_pipeline.py -v -k TestMiAsset
```

---

## Paso 8 — Verificación visual

1. Genera 10–20 escenas con el nuevo asset al 100%:

```bash
cd src && python3 generate_dataset.py --n 20 --seed 42 --p-mi-asset 1.0 --out ../output/test_asset
```

2. Abre 2–3 PLY en **CloudCompare**. Los labels se mapean a colores automáticamente. Comprueba:
   - Vehículo/asset en posición correcta (no flotando, no interpenetrando cargo).
   - Horquillas alineadas bajo el pallet amarillo.
   - Persona no solapando el nuevo asset.

3. Lanza Streamlit para vista interactiva:

```bash
cd src && streamlit run app.py
```

4. Genera una muestra mixta y revisa `metadata.json` para confirmar distribución correcta.

---

## Referencia rápida de archivos a modificar

| Archivo | Qué añadir |
|---------|-----------|
| `data/<nombre>.stl` | El STL |
| `src/geometry/meshes.py` | Constantes `MI_ASSET_*` + función `load_mi_asset()` |
| `src/generate_dataset.py` | CFG, import, branching, parse_args, run_generation, main |
| `src/test_pipeline.py` | Clase `TestMiAsset` con ≥3 tests |
| `README.md` | Actualizar tabla CLI y sección Structure |
| `DECISIONS.md` | Nueva sección con motivación y decisiones de diseño |
