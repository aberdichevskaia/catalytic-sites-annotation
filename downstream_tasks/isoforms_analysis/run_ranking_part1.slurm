#!/bin/bash
#SBATCH --job-name=rank_isoforms_predictions_part1
#SBATCH --output=output_rank_isoforms_predictions_part1.txt  
#SBATCH --error=error_rank_isoforms_predictions_part1.txt
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

PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/isoforms/rank_isoforms_part1.py \
  --structures_dir /home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms \
  --msa_dir        /home/iscb/wolfson/annab4/Data/MSA_Human_Isoforms \
  --out_npz        /home/iscb/wolfson/annab4/outputs/isoforms_preds_hybrid.npz \
  --manifest_csv   /home/iscb/wolfson/annab4/outputs/isoforms_preds_hybrid_manifest.csv \
  --ncores ${SLURM_CPUS_PER_TASK} \
  --min_prob 0.1

