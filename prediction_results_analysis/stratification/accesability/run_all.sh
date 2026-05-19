#!/bin/bash
# Accessibility stratification for ablate01 × {esm2_150M, esm2_150M_retrain, pwm, sequence}.
# All use weight_based_v9/split1-5. fold1-5 v1, predictions non-overlapping across folds.

SCRIPT=/home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/stratification/accesability/stratify.py
SPLITS_DIR=/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9
RETRAIN_DIR=/home/iscb/wolfson/annab4/scannet_retrains
PIPELINE_DIR=/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/pipelines
STRUCT_DIR=/home/iscb/wolfson/annab4/Data/PDB_files
OUT_BASE=/home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/outputs

MODELS=(
    "MLP_winner"
    "ablate09_esm2_3B_graphV2"
    "ablate11_pwm_retrain"
    "ablate11_sequence_retrain"
)

for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running: $MODEL"
    echo "=========================================="

    python "$SCRIPT" \
        --split_txts \
            "$SPLITS_DIR/split1.txt" \
            "$SPLITS_DIR/split2.txt" \
            "$SPLITS_DIR/split3.txt" \
            "$SPLITS_DIR/split4.txt" \
            "$SPLITS_DIR/split5.txt" \
        --structure_dirs "$STRUCT_DIR" \
        --results_pkl \
            "$RETRAIN_DIR/${MODEL}_fold1_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold2_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold3_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold4_v1/test_results.pkl" \
            "$RETRAIN_DIR/${MODEL}_fold5_v1/test_results.pkl" \
        --out_dir "$OUT_BASE/residue_accessibility_${MODEL}" \
        --pipeline_folder "$PIPELINE_DIR" \
        --report_every 800

    echo "Done: $MODEL -> $OUT_BASE/residue_accessibility_${MODEL}"
    echo ""
done

echo "All models done."
