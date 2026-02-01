base_name="ablate12_esm2_retrain"

versions=(1 2 3 4 5 6)
for ver in "${versions[@]}"
do
  python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py merge \
    --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
    --run_tpl ${base_name}_fold{fold}_v${ver} \
    --folds 1 2 3 4 5 \
    --subsets test validation \
    --save_dir /home/iscb/wolfson/annab4/scannet_retrains/merged/${base_name}_v${ver} \
    --dedupe_train mean \
    --max_k 10 \
    --save_plots 
done
