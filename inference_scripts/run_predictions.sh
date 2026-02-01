#!/bin/bash

export KERAS_BACKEND=jax


proteins=("D3Z6P0_A" "O76463_A" "O14815_A" "O60911_A" "P02879_A" "7DMQ_A" "A0A075FHI6_A" "P70705_A" "Q64430_A" "Q04656_A" "Q13613_A"  "A0A3M9ZW55_A")
python_script="/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/predict_bindingsites.py"

error_log="prediction_errors.log"
> "$error_log"
#folds=(1 2 3 4 5)
for protein in "${proteins[@]}"

do
    echo "Launch for $protein"
    
    #mode="cv_catalytic${fold}"
    #if python "$python_script" "$protein" --assembly --mode cv_catalytic; then
    if python "$python_script" "$protein" --mode cv_catalytic --use_ESM2 --esm2_dir /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/esm2_150M_layer30 --predictions_folder /home/iscb/wolfson/annab4/outputs/worst_best_predictions ; then
        echo "Ready for $protein"
    else 
        echo "Error for $protein" | tee -a "$error_log"
    fi
done

echo "All predictions are ready."
