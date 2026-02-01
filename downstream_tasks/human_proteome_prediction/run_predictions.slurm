#!/bin/bash
#SBATCH --job-name=predict_human_proteome_v2
#SBATCH --output=output_predict_human_proteome_v2.txt  
#SBATCH --error=error_predict_human_proteome_v2.txt
#SBATCH --partition=killable
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

struct_dir="/home/iscb/wolfson/jeromet/AFDB/Human_v2" # "/home/iscb/wolfson/annab4/DB/Human/Human" 

# ESM2 version
PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/human_proteome_prediction/predict.py \
  --structures_dir "${struct_dir}" \
  --msa_dir       /home/iscb/wolfson/annab4/Data/MSA_human \
  --meta_json /home/iscb/wolfson/annab4/DB/all_proteins/human_meta.json \
  --use_esm2 --esm2_dir /home/iscb/wolfson/annab4/Data/Human_ESM2_150M_layer30 \
  --ncores ${SLURM_CPUS_PER_TASK} \
  --out_csv /home/iscb/wolfson/annab4/outputs/human_catalytic_screen_withESM2_v2.csv


# MSA version
# PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/human_proteome_prediction/predict.py \
#   --structures_dir /home/iscb/wolfson/jeromet/AFDB/Human_v2 \
#   --msa_dir       /home/iscb/wolfson/annab4/Data/MSA_human \
#   --meta_json /home/iscb/wolfson/annab4/DB/all_proteins/human_meta.json \
#   --use_msa \
#   --ncores ${SLURM_CPUS_PER_TASK} \
#   --out_csv /home/iscb/wolfson/annab4/outputs/human_catalytic_screen_2.csv

