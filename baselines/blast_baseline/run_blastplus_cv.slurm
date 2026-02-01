#!/bin/bash

#SBATCH --job-name=blast_baseline
#SBATCH --output=output_blast_baseline.txt  
#SBATCH --error=error_blast_baseline.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=32
#SBATCH --mem=500000                      
#SBATCH --time=4320   
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio   

SPLITS="/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split*.txt"
DATASET="/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv"
OUT="/home/iscb/wolfson/annab4/DB/all_proteins/blast_baseline"

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/blast_baseline/run_blastplus_cv.py \
  --splits_glob "$SPLITS" \
  --dataset_csv "$DATASET" \
  --out_dir "$OUT" \
  --engine blastp \
  --threads 32 \

