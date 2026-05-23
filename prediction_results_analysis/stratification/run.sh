#!/bin/bash
# run.sh — Run all stratifications for a single model.
# Usage: bash run.sh <model_name>
# Example: bash run.sh ablate09_esm2_3B_graphV2

MODEL=${1:?"Usage: $0 <model_name>"}

SPLITS_DIR=/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9
RETRAIN_DIR=/home/iscb/wolfson/annab4/scannet_retrains
PIPELINE_DIR=/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/pipelines
STRUCT_DIR=/home/iscb/wolfson/annab4/Data/PDB_files
PROTEIN_JSON=/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json
OUT_DIR=/home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/outputs/stratification/$MODEL

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

PKLS=(
    "$RETRAIN_DIR/${MODEL}_fold1_v1/test_results.pkl"
    "$RETRAIN_DIR/${MODEL}_fold2_v1/test_results.pkl"
    "$RETRAIN_DIR/${MODEL}_fold3_v1/test_results.pkl"
    "$RETRAIN_DIR/${MODEL}_fold4_v1/test_results.pkl"
    "$RETRAIN_DIR/${MODEL}_fold5_v1/test_results.pkl"
)
SPLITS=(
    "$SPLITS_DIR/split1.txt"
    "$SPLITS_DIR/split2.txt"
    "$SPLITS_DIR/split3.txt"
    "$SPLITS_DIR/split4.txt"
    "$SPLITS_DIR/split5.txt"
)

# ── RSA / EC / chemotype / model_source stratification ────────────────────────
python "$SCRIPT_DIR/stratify.py" \
    --split_txts "${SPLITS[@]}" \
    --results_pkl "${PKLS[@]}" \
    --structure_dirs "$STRUCT_DIR" \
    --pipeline_folder "$PIPELINE_DIR" \
    --dataset_csv "$SPLITS_DIR/dataset.csv" \
    --out_dir "$OUT_DIR" \
    --report_every 800

# ── AF vs PDB (Stratification #1) ─────────────────────────────────────────────
python "$SCRIPT_DIR/stratify_af_vs_pdb.py" \
    --split_txts "${SPLITS[@]}" \
    --results_pkl "${PKLS[@]}" \
    --dataset_csv "$SPLITS_DIR/dataset.csv" \
    --protein_json "$PROTEIN_JSON" \
    --out_dir "$OUT_DIR"

# ── pLDDT bins (Stratification #2) ────────────────────────────────────────────
python "$SCRIPT_DIR/stratify_plddt.py" \
    --split_txts "${SPLITS[@]}" \
    --results_pkl "${PKLS[@]}" \
    --structure_dirs "$STRUCT_DIR" \
    --out_dir "$OUT_DIR" \
    --report_every 500

echo "Done → $OUT_DIR"
