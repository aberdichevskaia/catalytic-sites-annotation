#!/usr/bin/env bash
set -euo pipefail

# ---- paths ----
BASE_DIR="/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9"
DATASET_CSV="${BASE_DIR}/dataset.csv"

# Folder with *.txt label files (names do NOT matter)
LABELS_DIR="${BASE_DIR}"   # or "${BASE_DIR}/labels" if you move them

# Optional JSON for redundancy stats
PROTEIN_JSON="/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"

# Output
OUT_DIR="/home/iscb/wolfson/annab4/catalytic-sites-annotation/dataset_analysis/new_plots_font_path"

# Settings
WEIGHT_COL="W_Structure"
STRICT_LABELS="0"   # 1 -> crash if missing labels; 0 -> warn + drop for label-dependent reports

python3 /home/iscb/wolfson/annab4/catalytic-sites-annotation/dataset_analysis/clean_script/dataset_report.py all \
  --dataset_csv "${DATASET_CSV}" \
  --labels_dir "${LABELS_DIR}" \
  --out_dir "${OUT_DIR}" \
  --weight_col "${WEIGHT_COL}" \
  --strict_duplicates \
  --protein_json "${PROTEIN_JSON}"
