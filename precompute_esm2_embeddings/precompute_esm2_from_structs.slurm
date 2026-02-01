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

layers=(30) #16 18 24 30
struct_dir="/home/iscb/wolfson/jeromet/AFDB/Human_v2"       # "/home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms" 

for layer in "${layers[@]}"
do
  out_dir="/home/iscb/wolfson/annab4/Data/Human_ESM2_150M_layer${layer}"  # "/home/iscb/wolfson/annab4/Data/Human_Isoforms_ESM2_150M_layer${layer}"

  python /home/iscb/wolfson/annab4/catalytic-sites-annotation/precompute_esm2_embeddings/precompute_esm2_from_structs.py \
    --struct_dir "${struct_dir}" \
    --out_dir "${out_dir}" \
    --layer ${layer} \
    --device cuda \
    --origin_mode auto \
    --skip_existing
done
