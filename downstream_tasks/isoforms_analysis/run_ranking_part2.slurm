#!/bin/bash

#SBATCH --job-name=rank_isoforms_part2
#SBATCH --output=output_rank_isoforms_part2.txt  
#SBATCH --error=error_rank_isoforms_part2.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio  


PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/isoforms/rank_isoforms_part2.py \
  --dump_npz /home/iscb/wolfson/annab4/outputs/isoforms_preds_hybrid.npz \
  --out_csv  /home/iscb/wolfson/annab4/outputs/isoforms_spread_scores.csv \
  --nproc 64 \
  --min_isoforms 2
