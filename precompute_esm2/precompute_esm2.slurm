#!/bin/bash

#SBATCH --job-name=precompute_esm2_catalytic_150M
#SBATCH --partition=killable
##SBATCH --constraint="h100"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500000
#SBATCH --time=1440
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --mail-user=annab4@mail.tau.ac.il

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate scannet_esm2_py311

esm2_model="150M"   # 8M | 35M | 150M | 650M | 3B | 15B
layers=(30)         # default layer for 150M; for 3B use 36, for 15B use 48
splits_dir="/path/to/cross_validation_dir"

for layer in "${layers[@]}"
do
  out_dir="/path/to/ESM2_${esm2_model}_layer${layer}"

  python /home/iscb/wolfson/annab4/catalytic-sites-annotation/precompute_esm2_embeddings/precompute_esm2.py \
    --splits_dir    "${splits_dir}" \
    --out_dir       "${out_dir}" \
    --esm2_model    "${esm2_model}" \
    --layer         ${layer} \
    --device        cuda \
    --skip_existing \
    --max_tokens    8000 \
    --max_batch_size 16
done
