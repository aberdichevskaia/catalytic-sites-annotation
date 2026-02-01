#!/bin/bash
#SBATCH --job-name=predict_isoforms
#SBATCH --output=output_predict_isoforms.txt  
#SBATCH --error=error_predict_isoforms.txt
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

PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/isoforms_analysis/predict.py \
  --structures_dir /home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms \
  --meta_json      /home/iscb/wolfson/annab4/slurm_outputs/isoform_meta.json \
  --use_esm2 --esm2_dir /home/iscb/wolfson/annab4/Data/Human_Isoforms_ESM2_150M_layer30 \
  --ncores ${SLURM_CPUS_PER_TASK} \
  --out_csv /home/iscb/wolfson/annab4/outputs/human_isoforms_screen_withESM2.csv

  #  --msa_dir        /home/iscb/wolfson/annab4/Data/MSA_Human_Isoforms \
  #  --use_msa \
