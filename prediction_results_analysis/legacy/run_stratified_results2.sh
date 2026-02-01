python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/legacy/stratified_results2.py \
  --pkl /home/iscb/wolfson/annab4/scannet_retrains/merged/ablate12_esm2_retrain_v2/test.pkl \
  --dataset_csv /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv \
  --split_txts /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split1.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split2.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split3.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split4.txt /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split5.txt \
  --out_dir /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/plots \
  --k 5 \
  --n_boot 1000 \
  --seed 0 \
  --progress_every 50
