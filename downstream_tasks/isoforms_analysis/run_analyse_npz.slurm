#!/bin/bash
#SBATCH --job-name=analyse_isoforms
#SBATCH --output=output_analyse_isoforms.txt
#SBATCH --error=error_analyse_isoforms.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000
#SBATCH --time=1440
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio

PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/isoforms_analysis/analyse_npz.py \
  --dump_npz     /home/iscb/wolfson/annab4/outputs/isoforms_preds_esm2.npz \
  --out_csv      /home/iscb/wolfson/annab4/outputs/isoforms_analysis.csv \
  --tau          0.35 \
  --nproc        ${SLURM_CPUS_PER_TASK} \
  --min_isoforms 2
