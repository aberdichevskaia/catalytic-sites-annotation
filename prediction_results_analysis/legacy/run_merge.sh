#cfg=$1
base_name="ablate12_ESM2_focal_alpha_1_gamma_0"  #"ablate${cfg}_ESM2"

#echo "Merge results for config $cfg"

versions=(1 2 3 4 5 6)
for ver in "${versions[@]}"
do
  echo "Launch for version $ver"

  python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/legacy/merge_results.py \
    --base_dir /home/iscb/wolfson/annab4/scannet_experiments_outputs \
    --run_tpl ${base_name}_fold{fold}_v${ver} \
    --save_dir /home/iscb/wolfson/annab4/scannet_experiments_outputs/merged_results/${base_name}_v${ver} \
    --subsets validation test \
    --folds 1 2 3 4 5 \
    --dedupe_train mean 

  echo "Done for version $ver"
done

# python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/legacy/merge_results.py \
#   --base_dir /home/iscb/wolfson/annab4/scannet_experiments_outputs \
#   --run_tpl ablate00_ESM2_fold{fold}_v${ver} \
#   --save_dir /home/iscb/wolfson/annab4/scannet_experiments_outputs/merged_results/ablate00_ESM2_v${ver} \
#   --subsets validation test train \
#   --folds 1 2 3 4 5 \
#   --dedupe_train mean \
#   --save_plots