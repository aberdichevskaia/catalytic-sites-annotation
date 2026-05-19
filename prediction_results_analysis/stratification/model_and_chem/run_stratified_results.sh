models=(
    "MLP_winner"
    "ablate09_esm2_3B_graphV2"
    "ablate11_pwm_retrain"
    "ablate11_sequence_retrain"
)
OUT_BASE=/home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/outputs/stratification_results

for model in "${models[@]}"; do
    echo "Processing model: $model"
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/stratification/model_and_chem/stratified_results.py \
        --pkl /home/iscb/wolfson/annab4/scannet_retrains/merged/${model}_v1/test.pkl \
        --dataset-csv /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv \
        --split-txts /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split1.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split2.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split3.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split4.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split5.txt \
        --out-dir ${OUT_BASE}/${model} \
        --k 5 \
        --n-boot 1000 \
        --seed 0 \
        --progress-every 50 \
        --no-ec --no-chemotype --model-source \
        --out-metrics-model-source ms.csv \
        --out-plot-model-source ms.png

    echo "Finished processing model: $model"
done
