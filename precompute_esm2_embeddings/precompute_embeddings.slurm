#!/bin/bash

#SBATCH --job-name=precompute_esm2_PPBS_16
#SBATCH --output=output_precompute_esm2_PPBS_16.txt  
#SBATCH --error=error_precompute_esm2_PPBS_16.txt
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=8
#SBATCH --mem=500000                      
#SBATCH --time=1440  
#SBATCH --mail-type=ALL,TIME_LIMIT_80 
#SBATCH --mail-user=annab4@mail.tau.ac.il                

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate scannet_esm2_py311

layers=(30) # 16 18 24 30
out_dir="/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/esm2_150M_layer${layer}"   #/home/iscb/wolfson/annab4/DB/PPBS/esm2_150M_layer${layer}
splits_dir="/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9"   #"/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/datasets/PPBS"

for layer in "${layers[@]}"
do
  python /home/iscb/wolfson/annab4/catalytic-sites-annotation/esm2_aa_attributes/precompute_esm2.py \
    --splits_dir "${splits_dir}" \
    --layer ${layer} \
    --out_dir "${out_dir}" \
    --device cuda \
    --skip_existing \
    --max_tokens 8000 \
    --max_batch_size 16
done
