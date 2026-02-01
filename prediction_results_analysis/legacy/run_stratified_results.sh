python /home/iscb/wolfson/annab4/catalytic-sites-annotation/play_with_plot_data/stratified_results.py \
  --pkl /home/iscb/wolfson/annab4/catalytic-sites-annotation/cross_validation/merged/test.pkl \
  --dataset_csv /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv \
  --split_txts \
    /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split1.txt \
    /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split2.txt \
    /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split3.txt \
    /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split4.txt \
    /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split5.txt \
  --out_dir /home/iscb/wolfson/annab4/catalytic-sites-annotation/play_with_plot_data/plots/stratified_test \
  --k 5
