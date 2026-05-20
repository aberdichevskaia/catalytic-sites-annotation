#!/bin/bash
#SBATCH --job-name=annotate_isoforms
#SBATCH --output=output_annotate_isoforms.txt
#SBATCH --error=error_annotate_isoforms.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=60
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il

# CPU-only postprocessing: NPZ -> annotated CSV.
# Run rank_isoforms_part1.py (GPU) first to produce the NPZ.

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio

PYTHONUNBUFFERED=1 python -u /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/isoforms_analysis/predict.py \
  --dump_npz   /home/iscb/wolfson/annab4/outputs/isoforms_preds_esm2.npz \
  --meta_json  /home/iscb/wolfson/annab4/DB/all_proteins/isoform_meta.json \
  --thresholds 0.35 0.65 0.85 \
  --out_csv    /home/iscb/wolfson/annab4/outputs/human_isoforms_screen_withESM2.csv
