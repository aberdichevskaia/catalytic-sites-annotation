aa_attributes=("esm2") #  "sequence"  "pwm" 

configs=( "08" "09" "10" "11" "12")  # "00" "01" "02" "03" "04" "05" "06" "07"

for aa_attribute in "${aa_attributes[@]}"
do
  for cfg in "${configs[@]}"
  do
    base_name="ablate${cfg}_${aa_attribute}_retrain" 

    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py seeds \
      --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
      --save_dir /home/iscb/wolfson/annab4/scannet_retrains/seeds/${base_name}_v2 \
      --run_tpl "${base_name}_fold{fold}_v{version}" \
      --versions 7 8 9 10 11 12 \
      --subsets validation test \
      --max_k 10 \
      #--baseline_base_dir /path/to/baseline_runs \
      #--baseline_run_tpl "baseline_fixed/results_adaptive_cutoff_cv{fold}"
  done


  # base_name="default_${aa_attribute}" 

  # python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py seeds \
  #   --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
  #   --save_dir /home/iscb/wolfson/annab4/scannet_retrains/seeds/${base_name} \
  #   --run_tpl "${base_name}_fold{fold}_v{version}" \
  #   --versions 1 2 3 4 5 6 \
  #   --subsets validation test \
  #   --max_k 10 
done



# base_name="default_${aa_attribute}" 

# python /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/evaluate_results.py seeds \
#   --base_dir /home/iscb/wolfson/annab4/scannet_retrains \
#   --save_dir /home/iscb/wolfson/annab4/scannet_retrains/seeds/${base_name} \
#   --run_tpl "${base_name}_fold{fold}_v{version}" \
#   --versions 1 2 3 4 5 6 \
#   --subsets validation test \
#   --max_k 10 