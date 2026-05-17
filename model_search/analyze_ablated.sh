#!/bin/bash
# Analyze ablated (config 1, 3B only) runs from scannet_retrains.
# Metrics computed directly from result dirs (no SLURM logs needed).
python /home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/analyze_logs.py \
  --results_base_dir /home/iscb/wolfson/annab4/ablate_grid \
  --version_col seed \
  --expected_n_folds 5 \
  --subsets validation test \
  --out_csv ablated_log_summary.csv \
  --version_csv ablated_version_merged_summary.csv \
  --grouped_csv ablated_grouped_summary.csv
