# scripts/archive/

Scripts de diagnóstico retirados del pipeline activo. Útiles puntualmente para depurar problemas específicos, pero no forman parte del flujo normal de trabajo.

| Script | Propósito original |
|--------|-------------------|
| `diagnose_cluster_classifier.py` | Diagnóstico del clasificador de clusters (confusión, casos difíciles) |
| `diagnose_downstream.py` | Diagnóstico de fallos en etapas downstream del pipeline geométrico |
| `diagnose_vehicle_overfilter.py` | Diagnóstico de sobrefiltraje de vehículos |
| `diagnose_yolo_failures.py` | Análisis de fallos de detección YOLO 2D |
| `dump_clusters_debug.py` | Volcado de clusters intermedios para inspección manual en CloudCompare |
