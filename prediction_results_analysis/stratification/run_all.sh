#!/bin/bash
# run_all.sh — Run stratify.py for all models, then compare.py across them.

SPLITS_DIR=/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9
RETRAIN_DIR=/home/iscb/wolfson/annab4/scannet_retrains
PIPELINE_DIR=/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/pipelines
STRUCT_DIR=/home/iscb/wolfson/annab4/Data/PDB_files
OUT_BASE=/home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/outputs/stratification
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

MODELS=(
    "MLP_winner"
    "ablate09_esm2_3B_graphV2"
    "ablate11_pwm_retrain"
    "ablate11_sequence_retrain"
)

# ── stage 1: build residue tables ──────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "stratify: $MODEL"
    echo "=========================================="

    python "$SCRIPT_DIR/stratify.py" \
        --split_txts \
            "$SPLITS_DIR/split1.txt" \
            "$SPLITS_DIR/split2.txt" \
            "$SPLITS_DIR/split3.txt" \
            "$SPLITS_DIR/split4.txt" \
            "$SPLITS_DIR/split5.txt" \
        --results_pkl \
            "$RETRAIN_DIR/${MODEL}_fold1_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold2_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold3_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold4_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold5_v1/test_results.pkl" \
        --structure_dirs "$STRUCT_DIR" \
        --pipeline_folder "$PIPELINE_DIR" \
        --dataset_csv "$SPLITS_DIR/dataset.csv" \
        --out_dir "$OUT_BASE/$MODEL" \
        --report_every 800

    echo "Done: $MODEL → $OUT_BASE/$MODEL"
    echo ""
done

# ── stage 2: comparison across all models ──────────────────────────────────────
echo "=========================================="
echo "compare: all models"
echo "=========================================="

MODEL_DIRS=()
for MODEL in "${MODELS[@]}"; do
    MODEL_DIRS+=("$OUT_BASE/$MODEL")
done

python "$SCRIPT_DIR/compare.py" \
    --model_dirs  "${MODEL_DIRS[@]}" \
    --model_names "${MODELS[@]}" \
    --out_dir "$OUT_BASE/comparison"

echo ""
echo "All done → $OUT_BASE"
