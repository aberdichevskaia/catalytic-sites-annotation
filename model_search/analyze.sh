#!/bin/bash
# Analyze baseline (esm2_grid_search.py) runs from scannet_grid.
# Metrics computed directly from result dirs; logs used for status/error tracking.
python /home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/analyze_logs.py \
  --logs_dir /home/iscb/wolfson/annab4/scannet_grid/logs \
  --results_base_dir /home/iscb/wolfson/annab4/scannet_grid \
  --runs_csv /home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/runs.csv \
  --version_col seed \
  --expected_n_folds 5 \
  --subsets validation test \
  --out_csv log_summary.csv \
  --version_csv log_version_merged_summary.csv \
  --grouped_csv log_grouped_summary.csv
