python /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/find_threshold.py \
  --input_dir /home/iscb/wolfson/annab4/scannet_retrains/merged/ablate11_esm2_retrain_v3 \
  --output_dir /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/plots_ablate11_esm2 \
  --n_thresholds 500 \
  --b_list 3,8,15,30 \
  --alpha_conservative 0.60 \
  --avg_limit_discovery 15 \
  --p95_limit_discovery 30 \
  --b_show 8
