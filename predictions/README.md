# predictions/

Resultados de inferencia del clasificador 5 clases sobre escenas FUSION3D reales.
Los `.ply` están gitignoreados (regenerables) — este README sí se versiona.

## Archivos

| Archivo | Escenario | Fuente |
|---------|-----------|--------|
| `20260323_esc1.ply` | Escenario 1, 2026-03-23 | `Capturas_BBB/2026_03_23/1/FUSION3D/fusion3d_merged_20260323_140209.ply` |
| `20260323_esc2.ply` | Escenario 2, 2026-03-23 | `Capturas_BBB/2026_03_23/2/FUSION3D/fusion3d_merged_20260323_141634.ply` |
| `20260323_esc3.ply` | Escenario 3, 2026-03-23 | `Capturas_BBB/2026_03_23/3/FUSION3D/fusion3d_merged_20260323_142554.ply` |
| `20260323_esc4.ply` | Escenario 4, 2026-03-23 | `Capturas_BBB/2026_03_23/4/FUSION3D/fusion3d_merged_20260323_143258.ply` |
| `20260323_esc5.ply` | Escenario 5, 2026-03-23 | `Capturas_BBB/2026_03_23/5/FUSION3D/fusion3d_merged_20260323_143800.ply` |
| `20260323_esc6.ply` | Escenario 6, 2026-03-23 | `Capturas_BBB/2026_03_23/6/FUSION3D/fusion3d_merged_20260323_144511.ply` |

## Colores (CloudCompare)

| Label | Clase | Color |
|-------|-------|-------|
| 0 | floor | gris (60,60,60) |
| 1 | cargo | rojo (220,50,50) |
| 2 | vehicle | azul (0,120,255) |
| 3 | person | verde (39,174,96) |
| 4 | pallet | amarillo (255,210,0) |
| 255 | outlier | gris (60,60,60) |

## Regenerar

```bash
cd src
python3 classifier/predict.py \
  "../../Resources/Capturas_BBB/2026_03_23/1/FUSION3D/fusion3d_merged_20260323_140209.ply" \
  ../predictions/20260323_esc1.ply
```

Pipeline: auto-detect floor axis → swap Y↔Z si Z-up → translate floor→Y=0 → ROI crop X:±2.5m Y:-0.15..2.5m Z:±2.0m → RF classify.
