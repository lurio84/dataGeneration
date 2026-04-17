#!/usr/bin/env bash
# Run geometric pipeline on Captura_01 of every escenario that has a filtered PLY.
set -e
cd "$(dirname "$0")/.."
mkdir -p predictions/pipeline_demo/_batch_results
for d in predictions/pipeline_demo/Escenario_*; do
  esc=$(basename "$d")
  in_ply="$d/Captura_01_person_filtered.ply"
  if [ ! -f "$in_ply" ]; then
    echo "SKIP $esc (no filtered)"; continue
  fi
  out_dir="$d/geometric_full"
  mkdir -p "$out_dir"
  python3 -u scripts/run_geometric.py "$in_ply" "$out_dir" 2>&1 | tail -1
done
