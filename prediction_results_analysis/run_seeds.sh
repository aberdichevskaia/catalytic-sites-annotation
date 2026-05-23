aa_attributes=("pwm" ) #  "sequence"  "pwm" 

configs=("10")  # "00" "01" "02" "03" "04" "05" "06" "07"

graphs=("graphV2")

for aa_attribute in "${aa_attributes[@]}"
do
  for cfg in "${configs[@]}"
  do
    base_name="ablate${cfg}_${aa_attribute}_final_3B_graphV2"  

    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py seeds \
      --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
      --save_dir /home/iscb/wolfson/annab4/scannet_retrains/seeds/${base_name} \
      --run_tpl "${base_name}_fold{fold}_v{version}" \
      --versions 1 2 3 4 5 6 7 8 9 10 11 12 \
      --subsets validation_results test_results \
      --max_k 10 \
      #--baseline_base_dir /path/to/baseline_runs \
      #--baseline_run_tpl "baseline_fixed/results_adaptive_cutoff_cv{fold}"
  done

done

 

# base_name="ablate05_esm2_3B_graphV2"      #"ablate01_esm2_3B_graphV1"

# #MLP_fold1_v6__ESM2_3B_MLP__layer24__arch=MLP__hd=512-128__drop=0.3__norm=layernorm__act=relu__lr=0.0001__wd=0.0__bs=1__fold=1__seed=6

# python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py seeds \
#   --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
#   --save_dir /home/iscb/wolfson/annab4/scannet_retrains/seeds/${base_name} \
#   --run_tpl "${base_name}_fold{fold}_v{version}" \
#   --versions 1 2 3 4 5 6 7 8 9 10 11 12 \
#   --subsets validation_results test_results \
#   --max_k 10 


# aa_attributes=("sequence"  "pwm" ) 

# for aa_attribute in "${aa_attributes[@]}"
# do
#   base_name="ablate11_${aa_attribute}_retrain" 

#   python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py seeds \
#     --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
#     --save_dir /home/iscb/wolfson/annab4/scannet_retrains/seeds/${base_name} \
#     --run_tpl "${base_name}_fold{fold}_v{version}" \
#     --versions 1 2 3 4 5 6 \
#     --subsets validation test \
#     --max_k 10 
#   done