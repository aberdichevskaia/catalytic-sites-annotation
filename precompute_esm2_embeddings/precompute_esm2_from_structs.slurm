#!/bin/bash

#SBATCH --job-name=precompute_esm2_human_isoforms
#SBATCH --output=output_precompute_esm2_human_isoforms.txt  
#SBATCH --error=error_precompute_esm2_human_isoforms.txt
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

esm2_model="3B"   # 8M | 35M | 150M | 650M | 3B | 15B
layers=(36)         # default layer for 150M; for 3B use 36, for 15B use 48
struct_dir="/home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms"  #"/home/iscb/wolfson/jeromet/AFDB/Human_v2"

for layer in "${layers[@]}"
do
  out_dir="/home/iscb/wolfson/annab4/Data/Human_Isoforms_ESM2_${esm2_model}_layer${layer}"

  python /home/iscb/wolfson/annab4/catalytic-sites-annotation/precompute_esm2_embeddings/precompute_esm2_from_structs.py \
    --struct_dir "${struct_dir}" \
    --out_dir    "${out_dir}" \
    --esm2_model "${esm2_model}" \
    --layer      ${layer} \
    --device     cuda \
    --origin_mode auto \
    --skip_existing
done
