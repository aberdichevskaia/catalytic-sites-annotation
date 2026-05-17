base_names=(
    "MLP_winner"
    "ablate09_esm2_3B_graphV2"
    "ablate11_pwm_retrain"
    "ablate11_sequence_retrain"
)

for base_name in "${base_names[@]}"; do
  python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py merge \
    --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
    --run_tpl ${base_name}_fold{fold}_v1 \
    --folds 1 2 3 4 5 \
    --subsets validation test train \
    --save_dir /home/iscb/wolfson/annab4/scannet_retrains/merged/${base_name}_v1 \
    --dedupe_train mean \
    --max_k 10 \
    --save_plots
done 

# versions=(1 2 3 4 5 6)

# base_name="ablate11_pwm_retrain"

# for ver in "${versions[@]}"
# do
#   python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py merge \
#     --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
#     --run_tpl ${base_name}_fold{fold}_v${ver} \
#     --folds 1 2 3 4 5 \
#     --subsets validation test \
#     --save_dir /home/iscb/wolfson/annab4/scannet_retrains/merged/${base_name}_v${ver} \
#     --dedupe_train mean \
#     --max_k 10 \
#     --save_plots 
# done
