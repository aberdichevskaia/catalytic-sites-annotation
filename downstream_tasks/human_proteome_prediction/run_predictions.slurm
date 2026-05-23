#!/bin/bash
#SBATCH --job-name=predict_human_proteome_full
#SBATCH --partition=gpu-h100-killable
#SBATCH --constraint="l40s|h100"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=8
#SBATCH --mem=500000                      
#SBATCH --time=1440  
#SBATCH --mail-type=ALL,TIME_LIMIT_80 
#SBATCH --mail-user=annab4@mail.tau.ac.il                

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
export KERAS_BACKEND=jax
conda activate scannet_keras3

export SCANNET_ROOT=/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub
export PYTHONPATH=/home/iscb/wolfson/annab4/catalytic-sites-annotation:${PYTHONPATH}

struct_dir="/home/iscb/wolfson/jeromet/AFDB/Human_v2" # "/home/iscb/wolfson/annab4/DB/Human/Human" 

# ESM2 version
PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/human_proteome_prediction/predict.py \
  --structures_dir "${struct_dir}" \
  --msa_dir       /home/iscb/wolfson/annab4/Data/MSA_human \
  --meta_json /home/iscb/wolfson/annab4/DB/all_proteins/human_meta.json \
  --use_esm2 --esm2_dir /home/iscb/wolfson/annab4/Data/Human_ESM2_3B_layer36 \
  --esm2_version esm2_t36_3B_UR50D \
  --ncores ${SLURM_CPUS_PER_TASK} \
  --ablation ablate01 \
  --out_csv /home/iscb/wolfson/annab4/outputs/human_catalytic_screen_withESM2_ablate01.csv


# MSA version
# PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/human_proteome_prediction/predict.py \
#   --structures_dir /home/iscb/wolfson/jeromet/AFDB/Human_v2 \
#   --msa_dir       /home/iscb/wolfson/annab4/Data/MSA_human \
#   --meta_json /home/iscb/wolfson/annab4/DB/all_proteins/human_meta.json \
#   --use_msa \
#   --ncores ${SLURM_CPUS_PER_TASK} \
#   --out_csv /home/iscb/wolfson/annab4/outputs/human_catalytic_screen_2.csv

