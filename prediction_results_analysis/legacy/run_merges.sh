source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate datascience


configs=("11" "12")
for cfg in "${configs[@]}"
do
  echo "Launch for config $cfg"
  output_file="ablate${cfg}_ESM2.txt"

  /home/iscb/wolfson/annab4/catalytic-sites-annotation/prediction_results_analysis/legacy/run_merge.sh ${cfg} > ${output_file}

  echo "Done for $cfg"
done
