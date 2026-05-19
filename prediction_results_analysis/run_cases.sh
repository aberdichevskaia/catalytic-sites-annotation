python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py cases \
  --results-path /home/iscb/wolfson/annab4/scannet_retrains/merged/ablate08_esm2_retrain_v5/test.pkl \
  --dataset-csv /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv \
  --mode topk \
  --k 5 \
  --fp-th 0.6 \
  --fp-weight 1.0 --fn-weight 2.0 \
  --topn 30 \
  --out-csv /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/outputs/cases_test_full.csv
