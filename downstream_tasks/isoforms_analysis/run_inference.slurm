#!/bin/bash
#SBATCH --job-name=isoforms_inference
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
export PYTHONPATH=/home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/isoforms_analysis:${PYTHONPATH}

PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/isoforms_analysis/run_inference.py \
  --structures_dir /home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms \
  --esm2_dir       /home/iscb/wolfson/annab4/Data/Human_Isoforms_ESM2_3B_layer36 \
  --esm2_version esm2_t36_3B_UR50D \
  --out_npz        /home/iscb/wolfson/annab4/outputs/isoforms_preds_esm2_ablate12.npz \
  --manifest_csv   /home/iscb/wolfson/annab4/outputs/isoforms_preds_esm2_manifest_ablate12.csv \
  --ablation ablate12 \
  --ncores ${SLURM_CPUS_PER_TASK} \
  --min_prob 0.1
