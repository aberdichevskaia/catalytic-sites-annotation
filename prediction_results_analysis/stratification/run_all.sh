#!/bin/bash
# run_all.sh — Run all stratifications for all models, then compare.py across them.

SPLITS_DIR=/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9
RETRAIN_DIR=/home/iscb/wolfson/annab4/scannet_retrains
PIPELINE_DIR=/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/pipelines
STRUCT_DIR=/home/iscb/wolfson/annab4/Data/PDB_files
PROTEIN_JSON=/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json
OUT_BASE=/home/iscb/wolfson/annab4/scannet_retrains/stratification
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

SPLITS=(
    "$SPLITS_DIR/split1.txt"
    "$SPLITS_DIR/split2.txt"
    "$SPLITS_DIR/split3.txt"
    "$SPLITS_DIR/split4.txt"
    "$SPLITS_DIR/split5.txt"
)

# ── helper: run all three stratification scripts for one model ─────────────────
# Caller sets PKLS array and MODEL before calling.
run_stratifications() {
    local OUT="$OUT_BASE/$MODEL"
    echo "=========================================="
    echo "stratify: $MODEL"
    echo "=========================================="

    python "$SCRIPT_DIR/stratify.py" \
        --split_txts "${SPLITS[@]}" \
        --results_pkl "${PKLS[@]}" \
        --structure_dirs "$STRUCT_DIR" \
        --pipeline_folder "$PIPELINE_DIR" \
        --dataset_csv "$SPLITS_DIR/dataset.csv" \
        --out_dir "$OUT" \
        --report_every 800

    python "$SCRIPT_DIR/stratify_af_vs_pdb.py" \
        --split_txts "${SPLITS[@]}" \
        --results_pkl "${PKLS[@]}" \
        --dataset_csv "$SPLITS_DIR/dataset.csv" \
        --protein_json "$PROTEIN_JSON" \
        --out_dir "$OUT"

    python "$SCRIPT_DIR/stratify_plddt.py" \
        --split_txts "${SPLITS[@]}" \
        --results_pkl "${PKLS[@]}" \
        --structure_dirs "$STRUCT_DIR" \
        --out_dir "$OUT" \
        --report_every 500

    echo "Done: $MODEL → $OUT"
    echo ""
}

# ── ablation models: best-by-val versions from esm2_best_by_val.csv ───────────

# MODEL="ablate01_esm2_final_3B_graphV2"
# PKLS=(
#     "$RETRAIN_DIR/ablate01_esm2_final_3B_graphV2_fold1_v7/test_results.pkl"
#     "$RETRAIN_DIR/ablate01_esm2_final_3B_graphV2_fold2_v1/test_results.pkl"
#     "$RETRAIN_DIR/ablate01_esm2_final_3B_graphV2_fold3_v11/test_results.pkl"
#     "$RETRAIN_DIR/ablate01_esm2_final_3B_graphV2_fold4_v1/test_results.pkl"
#     "$RETRAIN_DIR/ablate01_esm2_final_3B_graphV2_fold5_v10/test_results.pkl"
# )
# run_stratifications

# MODEL="ablate09_esm2_final_3B_graphV2"
# PKLS=(
#     "$RETRAIN_DIR/ablate09_esm2_final_3B_graphV2_fold1_v10/test_results.pkl"
#     "$RETRAIN_DIR/ablate09_esm2_final_3B_graphV2_fold2_v9/test_results.pkl"
#     "$RETRAIN_DIR/ablate09_esm2_final_3B_graphV2_fold3_v2/test_results.pkl"
#     "$RETRAIN_DIR/ablate09_esm2_final_3B_graphV2_fold4_v9/test_results.pkl"
#     "$RETRAIN_DIR/ablate09_esm2_final_3B_graphV2_fold5_v8/test_results.pkl"
# )
# run_stratifications

# MODEL="ablate10_esm2_final_3B_graphV2"
# PKLS=(
#     "$RETRAIN_DIR/ablate10_esm2_final_3B_graphV2_fold1_v10/test_results.pkl"
#     "$RETRAIN_DIR/ablate10_esm2_final_3B_graphV2_fold2_v9/test_results.pkl"
#     "$RETRAIN_DIR/ablate10_esm2_final_3B_graphV2_fold3_v8/test_results.pkl"
#     "$RETRAIN_DIR/ablate10_esm2_final_3B_graphV2_fold4_v4/test_results.pkl"
#     "$RETRAIN_DIR/ablate10_esm2_final_3B_graphV2_fold5_v4/test_results.pkl"
# )
# run_stratifications

# MODEL="ablate12_esm2_final_3B_graphV2"
# PKLS=(
#     "$RETRAIN_DIR/ablate12_esm2_final_3B_graphV2_fold1_v8/test_results.pkl"
#     "$RETRAIN_DIR/ablate12_esm2_final_3B_graphV2_fold2_v5/test_results.pkl"
#     "$RETRAIN_DIR/ablate12_esm2_final_3B_graphV2_fold3_v9/test_results.pkl"
#     "$RETRAIN_DIR/ablate12_esm2_final_3B_graphV2_fold4_v3/test_results.pkl"
#     "$RETRAIN_DIR/ablate12_esm2_final_3B_graphV2_fold5_v10/test_results.pkl"
# )
# run_stratifications

MODEL="ablate11_pwm_final_3B_graphV2"
PKLS=(
    "$RETRAIN_DIR/ablate11_pwm_final_3B_graphV2_fold1_v3/test_results.pkl"
    "$RETRAIN_DIR/ablate11_pwm_final_3B_graphV2_fold2_v10/test_results.pkl"
    "$RETRAIN_DIR/ablate11_pwm_final_3B_graphV2_fold3_v7/test_results.pkl"
    "$RETRAIN_DIR/ablate11_pwm_final_3B_graphV2_fold4_v1/test_results.pkl"
    "$RETRAIN_DIR/ablate11_pwm_final_3B_graphV2_fold5_v1/test_results.pkl"
)
run_stratifications

MODEL="ablate11_sequence_final_3B_graphV2"
PKLS=(
    "$RETRAIN_DIR/ablate11_sequence_final_3B_graphV2_fold1_v10/test_results.pkl"
    "$RETRAIN_DIR/ablate11_sequence_final_3B_graphV2_fold2_v10/test_results.pkl"
    "$RETRAIN_DIR/ablate11_sequence_final_3B_graphV2_fold3_v8/test_results.pkl"
    "$RETRAIN_DIR/ablate11_sequence_final_3B_graphV2_fold4_v2/test_results.pkl"
    "$RETRAIN_DIR/ablate11_sequence_final_3B_graphV2_fold5_v8/test_results.pkl"
)
run_stratifications

# ── stage 2: comparison across ablation models ─────────────────────────────────
# echo "=========================================="
# echo "compare: ablation models"
# echo "=========================================="

# python "$SCRIPT_DIR/compare.py" \
#     --model_dirs \
#         "$OUT_BASE/ablate01_esm2_final_3B_graphV2" \
#         "$OUT_BASE/ablate09_esm2_final_3B_graphV2" \
#         "$OUT_BASE/ablate10_esm2_final_3B_graphV2" \
#         "$OUT_BASE/ablate12_esm2_final_3B_graphV2" \
#         "$OUT_BASE/ablate11_pwm_final_3B_graphV2" \
#         "$OUT_BASE/ablate11_sequence_final_3B_graphV2" \
#     --model_names ablate01 ablate09 ablate10 ablate12 ablate11_pwm ablate11_seq \
#     --out_dir "$OUT_BASE/comparison"

# echo ""
# echo "All done → $OUT_BASE"
